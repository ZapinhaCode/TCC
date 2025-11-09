import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

inmet_filtrados_dir = os.path.join(os.path.dirname(__file__), '../../INMET/Data/Filtrados')
anos = ['2020', '2021', '2022', '2023', '2024']
cidades = ['LagoaVermelha', 'PassoFundo', 'SantaMaria']
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

# Variáveis climáticas de interesse
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

# Cria a coluna de interrupção pela regra (chuva > 10mm ou raj. vento > 10m/s)
df_total['Situação'] = ((df_total['Chuva (mm)'] > 10) | (df_total['Raj. Vento (m/s)'] > 10)).map({False: 'Normalidade', True: 'Interrupção'})

# Garante que a pasta Images/INMET existe
images_dir = os.path.join(os.path.dirname(__file__), '../Images/INMET')
os.makedirs(images_dir, exist_ok=True)

# Prepara o DataFrame para o boxplot agrupado
df_melt = df_total.melt(
    id_vars=['Situação'],
    value_vars=variaveis,
    var_name='Variável',
    value_name='Valor'
)

plt.figure(figsize=(14, 8))
sns.set(style="whitegrid")
sns.boxplot(x='Variável', y='Valor', hue='Situação', data=df_melt, showfliers=False)
plt.xlabel('Variável Climática')
plt.ylabel('Valor')
plt.legend(title='Situação')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'comparativo_climatico_interrupcao_vs_normalidade.png'))
plt.show()