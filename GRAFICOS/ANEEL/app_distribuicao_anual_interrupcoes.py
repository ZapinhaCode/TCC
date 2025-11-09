import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_FILTRADOS = "../../ANEEL/Data/Filtrados"

arquivos_csv = [
    'interrupcoes_rge_sul_filtrado_2020.csv',
    'interrupcoes_rge_sul_filtrado_2021.csv',
    'interrupcoes_rge_sul_filtrado_2022.csv',
    'interrupcoes_rge_sul_filtrado_2023.csv',
    'interrupcoes_rge_sul_filtrado_2024.csv',
]

# Função para carregar e agregar dados (reutilizada de scripts anteriores)
def carregar_e_agregar_dados(arquivos):
    """Carrega todos os arquivos CSV e os concatena."""
    dfs = []
    for arquivo in arquivos:
        caminho = os.path.join(DATA_FILTRADOS, arquivo)
        try:
            # Lendo com o separador correto e forçando a leitura como string para evitar erros
            df = pd.read_csv(caminho, sep=';', dtype=str, encoding='latin1', low_memory=False)
            
            # Extrai o ano do nome do arquivo
            df['Ano'] = int(arquivo.split('_')[-1].split('.')[0])
            dfs.append(df)
        except FileNotFoundError:
            print(f"⚠️ Aviso: Arquivo não encontrado: {caminho}")
    return pd.concat(dfs, ignore_index=True)

# --- 2. Processamento de Dados ---

df_total = carregar_e_agregar_dados(arquivos_csv)

if df_total.empty:
    print("Nenhum dado encontrado para processamento.")
    exit()

# Contagem de interrupções por ano
contagem_por_ano = df_total['Ano'].value_counts().sort_index()

# Garante que a pasta Images existe
images_dir = '../Images/ANEEL'
os.makedirs(images_dir, exist_ok=True)
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

sns.lineplot(
    x=contagem_por_ano.index, 
    y=contagem_por_ano.values, 
    marker='o',
    color='#FF6F00',
    linewidth=2.5,
    markersize=8
)

# Adiciona os valores exatos acima de cada ponto de dados
for ano, valor in contagem_por_ano.items():
    plt.text(ano, valor + 15, str(valor), ha='center', va='bottom', fontsize=11, weight='bold')

plt.xlabel('Ano', fontsize=12)
plt.ylabel('Número de Interrupções', fontsize=12)
plt.xticks(contagem_por_ano.index)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'distribuicao_anual.png'))
plt.show()
print("✅ Gráfico de Distribuição Anual gerado e salvo em 'Images/distribuicao_anual.png'.")