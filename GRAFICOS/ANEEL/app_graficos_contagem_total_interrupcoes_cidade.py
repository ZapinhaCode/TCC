import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_FILTRADOS = "../ANEEL/Data/Filtrados"

arquivos_csv = [
    'interrupcoes_rge_sul_filtrado_2020.csv',
    'interrupcoes_rge_sul_filtrado_2021.csv',
    'interrupcoes_rge_sul_filtrado_2022.csv',
    'interrupcoes_rge_sul_filtrado_2023.csv',
    'interrupcoes_rge_sul_filtrado_2024.csv',
]

# Lista de causas CLIMÁTICAS que devem ser mantidas no gráfico, após a simplificação
CAUSAS_CLIMATICAS = [
    'Temporal',
    'Descarga Atmosferica',
    'Vento',
    'Galho/Arvore/Vegetacao sobre rede',
    'Inundacao/Alagamento',
    'Arvore ou Vegetacao' 
]

def carregar_e_agregar_dados(arquivos):
    """Carrega todos os arquivos CSV e os concatena."""
    dfs = []
    for arquivo in arquivos:
        caminho = os.path.join(DATA_FILTRADOS, arquivo)
        try:
            # Lendo com o separador correto e tratando possíveis problemas de encoding/quote
            df = pd.read_csv(caminho, sep=';', dtype=str, encoding='latin1', low_memory=False)
            df['Ano'] = int(arquivo.split('_')[-1].split('.')[0])
            dfs.append(df)
        except FileNotFoundError:
            print(f"⚠️ Aviso: Arquivo não encontrado: {caminho}")
    return pd.concat(dfs, ignore_index=True)

def limpar_e_simplificar_causas(df):
    """Simplifica a coluna DscFatoGeradorInterrupcao e filtra as causas climáticas."""
    
    causa_col = 'DscFatoGeradorInterrupcao'

    # 1. Remove aspas duplas, que são comuns em CSVs ANEEL com diferentes delimitadores
    df[causa_col] = df[causa_col].str.replace('"', '').str.strip()
    
    # 2. Simplifica a string para obter a causa final (último elemento após '/')
    # É necessário usar .str.split('/').str[-1] e .str.split(';').str[-1] devido a inconsistências
    df['Causa_Simples'] = df[causa_col].apply(
        lambda x: x.split('/')[-1].split(';')[-1].strip()
    )
    
    # 3. Filtra o DataFrame para manter apenas as causas CLIMÁTICAS relevantes
    df_climatico = df[df['Causa_Simples'].isin(CAUSAS_CLIMATICAS)].copy()
    
    # 4. Ajusta a coluna de Cidade, limpando os nomes completos
    df_climatico['DscConjuntoUnidadeConsumidora_Simples'] = df_climatico['DscConjuntoUnidadeConsumidora'].apply(
        lambda x: 'PASSO FUNDO' if 'PASSO FUNDO' in x.upper() else ('SANTA MARIA' if 'SANTA MARIA' in x.upper() else x)
    )

    # 5. Converte a coluna de data para manipulação de sazonalidade
    df_climatico['DatInicioInterrupcao'] = pd.to_datetime(df_climatico['DatInicioInterrupcao'], errors='coerce')
    
    return df_climatico

df_total = carregar_e_agregar_dados(arquivos_csv)
df_final = limpar_e_simplificar_causas(df_total)

if df_final.empty:
    print("Nenhum dado climático válido encontrado após a limpeza e filtragem.")
    exit()

# Garante que a pasta Images existe
images_dir = 'Images'
os.makedirs(images_dir, exist_ok=True)

print(f"✅ Total de registros climáticos para análise: {len(df_final)}")
sns.set_style("whitegrid")

contagem_causas = df_final['Causa_Simples'].value_counts().sort_values(ascending=True)
plt.figure(figsize=(10, 7))
sns.barplot(x=contagem_causas.values, y=contagem_causas.index, palette="mako")
plt.xlabel('Número de Interrupções', fontsize=12)
plt.ylabel('Causa Simplificada', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'contagem_por_causas.png'))
plt.show()
