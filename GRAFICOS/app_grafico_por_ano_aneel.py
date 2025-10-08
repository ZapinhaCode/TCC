import os
import pandas as pd
import matplotlib.pyplot as plt

# Caminho para os arquivos filtrados da ANEEL
pasta_aneel = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'TCC', 'ANEEL', 'Data', 'Filtrados'))
arquivos = [
    ('2020', 'interrupcoes_rge_sul_filtrado_2020.csv'),
    ('2021', 'interrupcoes_rge_sul_filtrado_2021.csv'),
    ('2022', 'interrupcoes_rge_sul_filtrado_2022.csv'),
    ('2023', 'interrupcoes_rge_sul_filtrado_2023.csv'),
    ('2024', 'interrupcoes_rge_sul_filtrado_2024.csv')
]

anos = []
contagem = []

for ano, arq in arquivos:
    caminho = os.path.join(pasta_aneel, arq)
    if not os.path.exists(caminho):
        print(f'Arquivo não encontrado: {caminho}')
        continue
    df = pd.read_csv(caminho, sep=';', dtype=str)
    anos.append(ano)
    contagem.append(len(df))

# Gráfico
plt.figure(figsize=(8,5))
plt.bar(anos, contagem, color='seagreen')
plt.ylabel('Quantidade de Interrupções')
plt.xlabel('Ano')
plt.title('Volume de Interrupções Elétricas por Ano (2020–2024)')
plt.tight_layout()
plt.show()