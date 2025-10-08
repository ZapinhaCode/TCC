import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

def extrai_matriz_confusao(relatorio_path):
    with open(relatorio_path, 'r') as f:
        texto = f.read()
    # Procura pela matriz de confusão no texto
    match = re.search(r'Matriz de confusão:\s*\n(\[\[.*?\]\])', texto, re.DOTALL)
    if match:
        matriz_str = match.group(1)
        # Extrai todos os números da matriz
        numeros = [int(n) for n in re.findall(r'\d+', matriz_str)]
        # Calcula o tamanho da matriz (assume quadrada)
        lado = int(len(numeros) ** 0.5)
        matriz = np.array(numeros).reshape((lado, lado))
        return matriz
    else:
        print(f'Matriz de confusão não encontrada em {relatorio_path}')
        return None

# Caminhos dos relatórios (exemplo para Random Forest)
dir_rf = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PREVISOR', 'Data', 'Random Forest', 'RelatorioClassificacao'))
arquivos = [arq for arq in os.listdir(dir_rf) if arq.endswith('.txt')]

for arquivo in arquivos:
    caminho = os.path.join(dir_rf, arquivo)
    matriz = extrai_matriz_confusao(caminho)
    if matriz is not None:
        plt.figure(figsize=(4,4))
        sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Prev. 0', 'Prev. 1'], yticklabels=['Real 0', 'Real 1'])
        plt.title(f'Matriz de Confusão - {arquivo.replace("relatorio_", "").replace(".txt", "").capitalize()}', pad=20)
        plt.xlabel('Previsto')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.show()