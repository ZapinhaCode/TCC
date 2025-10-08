import pandas as pd
import os
import re
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def juntar_datasets(
    aneel_dir='../ANEEL/Data/Filtrados',
    inmet_dir='../INMET/Data/Filtrados',
    saida_dir='Data/XGBoost',
):
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
            # Use merge INNER para garantir apenas datas/horas presentes nos dois arquivos
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
        if 'chuva (mm)' in df_merged.columns:
            df_merged['chuva_mm'] = pd.to_numeric(df_merged['chuva (mm)'].str.replace(',', '.'), errors='coerce').fillna(0)
        else:
            df_merged['chuva_mm'] = 0

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
        if 'raj. vento (m/s)' in df_merged.columns:
            df_merged['raj. vento (m/s)'] = pd.to_numeric(df_merged['raj. vento (m/s)'].str.replace(',', '.'), errors='coerce').fillna(0)
        else:
            df_merged['raj. vento (m/s)'] = 0

        def wind_risk(row):
            try:
                raj = float(row.get('raj. vento (m/s)', 0))
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

        out_cidade = cidade.replace(' ', '_').replace('/', '_')
        out_path = os.path.join(saida_dir, f'cidade_{out_cidade}.csv')
        df_merged[['data', 'hora (utc)', 'chuva_mm', 'raj. vento (m/s)', 'risco de chuva', 'risco de vento']].rename(columns={'hora (utc)': 'hora'}).to_csv(out_path, index=False, sep=';')
        print(f'✅ Arquivo consolidado salvo para {cidade}: {out_path}')

def treinar_xgboost(saida_dir='Data/XGBoost'):
    relatorio_dir = os.path.join(saida_dir, 'RelatorioClassificacao')
    if not os.path.exists(relatorio_dir):
        os.makedirs(relatorio_dir)

    for file in os.listdir(saida_dir):
        if not file.endswith('.csv'):
            continue
        cidade_path = os.path.join(saida_dir, file)
        print(f'\nTreinando modelo para: {file}')
        df = pd.read_csv(cidade_path, sep=';')

        # Garante que as colunas numéricas estejam no formato correto
        df['chuva_mm'] = pd.to_numeric(df.get('chuva_mm', np.nan), errors='coerce')
        df['raj. vento (m/s)'] = pd.to_numeric(df.get('raj. vento (m/s)', np.nan), errors='coerce')

        # Remove linhas sem valores necessários
        df = df.dropna(subset=['chuva_mm', 'raj. vento (m/s)', 'risco de chuva'])

        # Cria coluna alvo binária: 1 para alto/muito_alto, 0 para baixo/moderado
        df['risco_chuva_bin'] = df['risco de chuva'].map({'baixo': 0, 'moderado': 0, 'alto': 1, 'muito_alto': 1})

        print(f"Linhas válidas para treino/teste: {len(df)}")
        print(f"Distribuição do alvo: {df['risco_chuva_bin'].value_counts().to_dict()}")

        X = df[['chuva_mm', 'raj. vento (m/s)']].values
        y = df['risco_chuva_bin'].values

        if len(X) < 2 or len(np.unique(y)) < 2:
            print("Dados insuficientes para treino/teste. Pulando este arquivo.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        matriz_confusao = confusion_matrix(y_test, y_pred)
        relatorio = classification_report(y_test, y_pred)

        print("Matriz de confusão:")
        print(matriz_confusao)
        print("\nRelatório de classificação:")
        print(relatorio)

        # Salva o relatório em arquivo
        relatorio_path = os.path.join(relatorio_dir, f'relatorio_{file.replace(".csv", "")}.txt')
        with open(relatorio_path, 'w') as f:
            f.write(f"Arquivo: {file}\n")
            f.write("Matriz de confusão:\n")
            f.write(np.array2string(matriz_confusao))
            f.write("\n\nRelatório de classificação:\n")
            f.write(relatorio)
        print(f"Relatório salvo em: {relatorio_path}")

if __name__ == "__main__":
    juntar_datasets()
    treinar_xgboost()
