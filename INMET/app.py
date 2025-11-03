import pandas as pd
import os

def filtrar_dados_csv(input_path, output_path):
    colunas_principais = [
        "Data", "Hora (UTC)", "Temp. Ins. (C)", "Temp. Max. (C)", "Temp. Min. (C)",
        "Umi. Ins. (%)", "Umi. Max. (%)", "Umi. Min. (%)",
        "Pto Orvalho Ins. (C)", "Pto Orvalho Max. (C)", "Pto Orvalho Min. (C)",
        "Pressao Ins. (hPa)", "Pressao Max. (hPa)", "Pressao Min. (hPa)",
        "Vel. Vento (m/s)", "Dir. Vento (m/s)", "Raj. Vento (m/s)",
        "Radiacao (KJ/m²)", "Chuva (mm)"
    ]
    try:
        df = pd.read_csv(input_path, sep=';', dtype=str, on_bad_lines="skip")
    except Exception as e:
        print(f'⚠️ Erro ao ler {input_path} ({e}), tentando pular linhas problemáticas...')
        return

    df = df[[col for col in colunas_principais if col in df.columns]]

    # Filtra apenas se "Raj. Vento (m/s)" ou "Chuva (mm)" estão vazias ou NaN
    col_vento = "Raj. Vento (m/s)"
    col_chuva = "Chuva (mm)"
    cond_vento = df[col_vento].isna() | (df[col_vento].astype(str).str.strip() == '') if col_vento in df.columns else pd.Series([True]*len(df))
    cond_chuva = df[col_chuva].isna() | (df[col_chuva].astype(str).str.strip() == '') if col_chuva in df.columns else pd.Series([True]*len(df))
    df_filtrado = df[~(cond_vento | cond_chuva)]

    df_filtrado.to_csv(output_path, index=False, sep=';')


def filtrar_todos_csvs(data_dir='Data', filtrados_dir='Data/Filtrados'):
    if not os.path.exists(filtrados_dir):
        os.makedirs(filtrados_dir)
    for root, dirs, files in os.walk(data_dir):
        if 'Filtrados' in root:
            continue
        for file in files:
            if file.endswith('.csv'):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, data_dir)
                out_dir = os.path.join(filtrados_dir, rel_path) if rel_path != '.' else filtrados_dir
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                nome_base = os.path.splitext(file)[0]
                output_path = os.path.join(out_dir, f'{nome_base}_filtrado.csv')
                try:
                    filtrar_dados_csv(input_path, output_path)
                    print(f'✅ Arquivo filtrado salvo: {output_path}')
                except Exception as e:
                    print(f'❌ Erro ao filtrar {input_path}: {e}')

if __name__ == "__main__":
    filtrar_todos_csvs(data_dir='Data', filtrados_dir='Data/Filtrados')