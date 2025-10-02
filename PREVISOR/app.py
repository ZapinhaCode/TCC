import pandas as pd
import os
import re
import numpy as np

def juntar_datasets(
    aneel_dir='../ANEEL/Data/Filtrados',
    inmet_dir='../INMET/Data/Filtrados',
    saida_dir='Data',
):
    """
    Junta os datasets filtrados de ANEEL e INMET por Data e Hora (UTC),
    salvando um novo CSV com as informações combinadas.
    """
    if not os.path.exists(saida_dir):
        os.makedirs(saida_dir)

    aneel_files = [f for f in os.listdir(aneel_dir) if f.endswith('.csv')]

    inmet_files = []
    for root, dirs, files in os.walk(inmet_dir):
        for file in files:
            if file.endswith('.csv'):
                ano = os.path.basename(root)
                inmet_files.append((ano, os.path.join(root, file), file))

    cidades_consolidadas = dict()
    for aneel_file in aneel_files:
        ano = ''.join(filter(str.isdigit, aneel_file))
        aneel_path = os.path.join(aneel_dir, aneel_file)
        for inmet_ano, inmet_path, inmet_file in inmet_files:
            if inmet_ano != ano:
                continue
            cidade = inmet_file.replace('_filtrado.csv', '').lower()
            df_aneel = pd.read_csv(aneel_path, sep=';', dtype=str)
            df_inmet = pd.read_csv(inmet_path, sep=';', dtype=str)
            df_aneel.columns = [c.strip().lower() for c in df_aneel.columns]
            df_inmet.columns = [c.strip().lower() for c in df_inmet.columns]
            if 'datiniciointerrupcao' in df_aneel.columns:
                dt = df_aneel['datiniciointerrupcao'].astype(str)
                df_aneel['data'] = dt.str[:10]
                hora_raw = dt.str[11:]
                df_aneel['hora (utc)'] = hora_raw.str.replace(r'[^0-9]', '', regex=True).str.zfill(4)
            if 'municipio' in df_aneel.columns:
                mask = df_aneel['municipio'].str.lower().str.contains(cidade, na=False)
                df_aneel_cidade = df_aneel[mask]
            else:
                df_aneel_cidade = df_aneel
            if 'data' in df_aneel_cidade.columns and 'data' in df_inmet.columns and 'hora (utc)' in df_aneel_cidade.columns and 'hora (utc)' in df_inmet.columns:
                df_merged = pd.merge(
                    df_aneel_cidade,
                    df_inmet,
                    on=['data', 'hora (utc)'],
                    suffixes=('_aneel', '_inmet'),
                    how='right'
                )
                if cidade not in cidades_consolidadas:
                    cidades_consolidadas[cidade] = []
                cidades_consolidadas[cidade].append(df_merged)

    for cidade, dfs in cidades_consolidadas.items():
        df_merged = pd.concat(dfs, ignore_index=True)

        # Padroniza a coluna data para dd/mm/yyyy
        def padroniza_data(val):
            if pd.isna(val):
                return ''
            val = str(val).strip()
            if re.match(r'^\d{4}[-/]\d{2}[-/]\d{2}$', val):
                partes = re.split('[-/]', val)
                return f"{partes[2]}/{partes[1]}/{partes[0]}"
            if re.match(r'^\d{2}-\d{2}-\d{4}$', val):
                return val.replace('-', '/')
            if re.match(r'^\d{2}/\d{2}/\d{4}$', val):
                return val
            return val

        df_merged['data'] = df_merged['data'].apply(padroniza_data)

        # --- RISCO DE CHUVA ---
        df_merged['chuva_mm'] = pd.to_numeric(df_merged['chuva (mm)'].str.replace(',', '.'), errors='coerce').fillna(0)
        def rain_risk(row):
            chuva = row['chuva_mm']
            if pd.isna(chuva):
                return 'unknown'
            if chuva < 5:
                return 'baixo'
            if chuva < 15:
                return 'moderado'
            if chuva < 30:
                return 'alto'
            return 'muito_alto'
        df_merged['risco de chuva'] = df_merged.apply(rain_risk, axis=1)

        # --- RISCO DE VENTO ---
        def wind_risk(row):
            try:
                raj = float(str(row.get('raj. vento (m/s)', '')).replace(',', '.'))
            except:
                return 'unknown'
            if pd.isna(raj):
                return 'unknown'
            if raj < 10:
                return 'baixo'
            if raj < 15:
                return 'moderado'
            if raj < 21:
                return 'alto'
            if raj < 25:
                return 'muito_alto'
            return 'critico'
        df_merged['risco de vento'] = df_merged.apply(wind_risk, axis=1)

        # --- RISCO DE VEGETAÇÃO (apenas vento e chuva acumulada 24h) ---
        df_merged['data_hora'] = pd.to_datetime(df_merged['data'] + ' ' + df_merged['hora (utc)'].str.zfill(4), format='%d/%m/%Y %H%M', errors='coerce')
        df_merged = df_merged.sort_values('data_hora')
        df_merged['chuva_24h'] = df_merged['chuva_mm'].rolling(window=24, min_periods=1).sum()

        def vegetacao_risk(row):
            chuva_24h = row.get('chuva_24h', None)
            # Pega o maior valor de vento entre os três campos
            vento_campos = []
            for campo in ['vel. vento (m/s)', 'dir. vento (m/s)', 'raj. vento (m/s)']:
                try:
                    valor = float(str(row.get(campo, '0')).replace(',', '.'))
                    vento_campos.append(valor)
                except:
                    continue
            raj_max = max(vento_campos) if vento_campos else np.nan
            try:
                chuva_24h = float(str(chuva_24h).replace(',', '.'))
            except:
                chuva_24h = np.nan

            # Risco crítico
            if not pd.isna(raj_max) and not pd.isna(chuva_24h):
                if raj_max >= 21 and chuva_24h >= 80:
                    return 'critico'
                if raj_max >= 15 and chuva_24h >= 50:
                    return 'alto'
            return 'baixo'

        df_merged['risco de vegetacao'] = df_merged.apply(vegetacao_risk, axis=1)

        out_cidade = cidade.replace(' ', '_').replace('/', '_')
        out_path = os.path.join(saida_dir, f'cidade_{out_cidade}.csv')
        df_merged[['data', 'hora (utc)', 'risco de chuva', 'risco de vento', 'risco de vegetacao']].rename(columns={'hora (utc)': 'hora'}).to_csv(out_path, index=False, sep=';')
        print(f'✅ Arquivo consolidado salvo para {cidade}: {out_path}')

if __name__ == "__main__":
    juntar_datasets()
