import os
import pandas as pd
import matplotlib.pyplot as plt

# Diretório dos arquivos ANEEL filtrados
aneel_filtrados_dir = '../../ANEEL/Data/Filtrados'
anos = ['2020', '2021', '2022', '2023', '2024']

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

df_total = pd.concat(df_list, ignore_index=True)
cidade_col = 'DscConjuntoUnidadeConsumidora'
contagem = df_total[cidade_col].value_counts().sort_values(ascending=False)

images_dir = os.path.join('../Images', 'ANEEL')
os.makedirs(images_dir, exist_ok=True)

# Gráfico de barras horizontais
plt.figure(figsize=(10, 8))
contagem.plot(kind='barh')
plt.xlabel('Número de interrupções')
plt.ylabel('Cidade')

# Adiciona os valores abaixo das barras
for i, (cidade, valor) in enumerate(zip(contagem.index, contagem.values)):
    plt.text(valor, i, str(valor), va='bottom', ha='left', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'contagem_interrupcoes_por_cidade.png'))
plt.show()