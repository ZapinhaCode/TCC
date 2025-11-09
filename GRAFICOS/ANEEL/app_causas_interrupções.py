import os
import pandas as pd
import matplotlib.pyplot as plt

# Diretório dos arquivos ANEEL filtrados
aneel_filtrados_dir = '../ANEEL/Data/Filtrados'
anos = ['2020', '2021', '2022', '2023', '2024']

# Lista para armazenar todas as interrupções
df_list = []

for ano in anos:
    arq = os.path.join(aneel_filtrados_dir, f'interrupcoes_rge_sul_filtrado_{ano}.csv')
    if os.path.exists(arq):
        df = pd.read_csv(arq, sep=';', dtype=str)
        df['Ano'] = ano
        df_list.append(df)

if not df_list:
    print("Nenhum arquivo encontrado.")
    exit()

# Junta todos os anos
df_total = pd.concat(df_list, ignore_index=True)

# Usa a coluna correta para cidade
cidade_col = 'DscConjuntoUnidadeConsumidora'

# Contagem total de interrupções por cidade
contagem = df_total[cidade_col].value_counts().sort_values(ascending=False)

# Gráfico
plt.figure(figsize=(10, 6))
contagem.plot(kind='bar')
plt.title('Contagem total de interrupções por cidade')
plt.xlabel('Cidade')
plt.ylabel('Número de interrupções')
plt.tight_layout()
plt.savefig('contagem_interrupcoes_por_cidade.png')
plt.show()