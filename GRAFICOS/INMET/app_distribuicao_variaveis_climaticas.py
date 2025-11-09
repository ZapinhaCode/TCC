import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

inmet_filtrados_dir = os.path.join(os.path.dirname(__file__), '../../INMET/Data/Filtrados')
anos = ['2020', '2021', '2022', '2023', '2024']
cidades = ['LagoaVermelha', 'PassoFundo', 'SantaMaria']

# Lista para armazenar todos os dados
df_list = []

for ano in anos:
    for cidade in cidades:
        arq = os.path.join(inmet_filtrados_dir, ano, f"{cidade}_filtrado.csv")
        if os.path.exists(arq):
            df = pd.read_csv(arq, sep=';', dtype=str)
            df['Ano'] = ano
            df['Cidade'] = cidade
            df_list.append(df)

if not df_list:
    print("Nenhum arquivo filtrado encontrado.")
    exit()

# Junta todos os anos e cidades
df_total = pd.concat(df_list, ignore_index=True)

variaveis = [
    'Temp. Ins. (C)', 'Vel. Vento (m/s)', 'Raj. Vento (m/s)', 'Pressao Ins. (hPa)', 'Chuva (mm)'
]
variaveis = [v for v in variaveis if v in df_total.columns]

if not variaveis:
    print("Nenhuma das variáveis desejadas encontrada nos arquivos filtrados.")
    print("Colunas disponíveis:", df_total.columns.tolist())
    exit()

# Converte para numérico
for var in variaveis:
    df_total[var] = pd.to_numeric(df_total[var].astype(str).str.replace(',', '.'), errors='coerce')

images_dir = os.path.join(os.path.dirname(__file__), '../Images/INMET')
os.makedirs(images_dir, exist_ok=True)

# Calcula a média de cada variável para o gráfico de barras laterais
medias = df_total[variaveis].mean().sort_values()

plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.barplot(x=medias.values, y=medias.index, palette="viridis")
plt.xlabel('Valor Médio')
plt.ylabel('Variável Climática')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'distribuicao_variaveis_climaticas.png'))
plt.show()