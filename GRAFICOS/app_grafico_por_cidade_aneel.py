import os
import pandas as pd
import matplotlib.pyplot as plt

# Caminho absoluto para os arquivos filtrados da ANEEL
pasta_aneel = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'TCC', 'ANEEL', 'Data', 'Filtrados'))
arquivos = [
    'interrupcoes_rge_sul_filtrado_2020.csv',
    'interrupcoes_rge_sul_filtrado_2021.csv',
    'interrupcoes_rge_sul_filtrado_2022.csv',
    'interrupcoes_rge_sul_filtrado_2023.csv',
    'interrupcoes_rge_sul_filtrado_2024.csv'
]

# Municípios de interesse
municipios = ['Porto Alegre', 'Passo Fundo', 'Lagoa Vermelha', 'Santa Maria']
contagem = {m: 0 for m in municipios}

# Conta registros por município em todos os anos
for arq in arquivos:
    caminho = os.path.join(pasta_aneel, arq)
    if not os.path.exists(caminho):
        print(f'Arquivo não encontrado: {caminho}')
        continue
    df = pd.read_csv(caminho, sep=';', dtype=str)
    if 'DscConjuntoUnidadeConsumidora' in df.columns:
        for m in municipios:
            contagem[m] += df['DscConjuntoUnidadeConsumidora'].str.lower().str.contains(m.lower()).sum()
    else:
        print(f'Coluna "DscConjuntoUnidadeConsumidora" não encontrada em {caminho}')

# Gráfico
plt.figure(figsize=(8,5))
plt.bar(contagem.keys(), contagem.values(), color='royalblue')
plt.ylabel('Quantidade de Registros')
plt.title('Quantidade de Interrupções por Município (ANEEL) 2020/2024')
plt.tight_layout()
plt.show()