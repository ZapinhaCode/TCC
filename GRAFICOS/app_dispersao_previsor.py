import os
import pandas as pd
import matplotlib.pyplot as plt

# Caminhos das bases já consolidadas (exemplo usando Random Forest, pode trocar para XGBoost)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PREVISOR', 'Data', 'Random Forest'))
arquivos = [
    'cidade_portoalegre.csv',
    'cidade_passofundo.csv',
    'cidade_lagoavermelha.csv',
    'cidade_santamaria.csv'
]
cidades_legenda = {
    'cidade_portoalegre.csv': 'Porto Alegre',
    'cidade_passofundo.csv': 'Passo Fundo',
    'cidade_lagoavermelha.csv': 'Lagoa Vermelha',
    'cidade_santamaria.csv': 'Santa Maria'
}

for arq in arquivos:
    caminho = os.path.join(base_dir, arq)
    if not os.path.exists(caminho):
        print(f'Arquivo não encontrado: {caminho}')
        continue
    df = pd.read_csv(caminho, sep=';')
    # Conta número de interrupções por data/hora (1 se houve interrupção, 0 se não)
    df['interrupcao'] = df['risco de chuva'].apply(lambda x: 1 if x in ['alto', 'muito_alto'] else 0)
    # Agrupa por chuva e vento, somando interrupções
    chuva_group = df.groupby('chuva_mm')['interrupcao'].sum().reset_index()
    vento_group = df.groupby('raj. vento (m/s)')['interrupcao'].sum().reset_index()

    # Gráfico de dispersão: Chuva vs Interrupções
    plt.figure(figsize=(7,4))
    plt.scatter(chuva_group['chuva_mm'], chuva_group['interrupcao'], color='royalblue')
    plt.xlabel('Chuva (mm)')
    plt.ylabel('Número de Interrupções')
    plt.title(f'Dispersão: Chuva x Interrupções - {cidades_legenda[arq]}')
    plt.tight_layout()
    plt.show()

    # Gráfico de dispersão: Vento vs Interrupções
    plt.figure(figsize=(7,4))
    plt.scatter(vento_group['raj. vento (m/s)'], vento_group['interrupcao'], color='darkorange')
    plt.xlabel('Vento (m/s)')
    plt.ylabel('Número de Interrupções')
    plt.title(f'Dispersão: Vento x Interrupções - {cidades_legenda[arq]}')
    plt.tight_layout()
    plt.show()