import os
import pandas as pd
import matplotlib.pyplot as plt

# Caminho para os arquivos filtrados da ANEEL
pasta_aneel = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'TCC', 'ANEEL', 'Data', 'Filtrados'))
arquivos = [
    'interrupcoes_rge_sul_filtrado_2020.csv',
    'interrupcoes_rge_sul_filtrado_2021.csv',
    'interrupcoes_rge_sul_filtrado_2022.csv',
    'interrupcoes_rge_sul_filtrado_2023.csv',
    'interrupcoes_rge_sul_filtrado_2024.csv'
]

# Lista para armazenar todos os DataFrames
dfs = []

for arq in arquivos:
    caminho = os.path.join(pasta_aneel, arq)
    if not os.path.exists(caminho):
        print(f'Arquivo não encontrado: {caminho}')
        continue
    df = pd.read_csv(caminho, sep=';', dtype=str)
    # Usa a coluna correta de data
    if 'DatGeracaoConjuntoDados' in df.columns:
        df['data'] = pd.to_datetime(df['DatGeracaoConjuntoDados'], errors='coerce')
    elif 'DatInicioInterrupcao' in df.columns:
        df['data'] = pd.to_datetime(df['DatInicioInterrupcao'], errors='coerce')
    else:
        print(f'Coluna de data não encontrada em {caminho}')
        print(f'Colunas disponíveis: {list(df.columns)}')
        continue
    dfs.append(df)

if not dfs:
    print("Nenhum dado carregado. Verifique o nome da coluna de data nos arquivos CSV.")
    exit()

# Junta todos os DataFrames
df_total = pd.concat(dfs, ignore_index=True)

# Remove datas inválidas
df_total = df_total.dropna(subset=['data'])

# Agrupa por mês
df_total['ano_mes'] = df_total['data'].dt.to_period('M')
serie_mensal = df_total.groupby('ano_mes').size()

# Gráfico de série temporal mensal
plt.figure(figsize=(14,6))
plt.plot(serie_mensal.index.astype(str), serie_mensal.values, marker='o', color='royalblue', linewidth=2)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.title('Série Temporal de Interrupções Elétricas por Mês (2020–2024)', fontsize=16, pad=20)
plt.xlabel('Ano-Mês', fontsize=12)
plt.ylabel('Número de Interrupções', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.show()

# Série temporal trimestral (opcional)
serie_trimestral = df_total.groupby(df_total['data'].dt.to_period('Q')).size()

plt.figure(figsize=(10,5))
plt.bar(serie_trimestral.index.astype(str), serie_trimestral.values, color='darkorange', width=0.6)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.title('Série Temporal de Interrupções Elétricas por Trimestre (2020–2024)', fontsize=16, pad=20)
plt.xlabel('Ano-Trimestre', fontsize=12)
plt.ylabel('Número de Interrupções', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.show()