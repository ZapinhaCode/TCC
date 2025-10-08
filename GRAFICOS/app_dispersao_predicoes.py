import os
import re
import matplotlib.pyplot as plt

def extrai_predicoes_reais(relatorio_path):
    with open(relatorio_path, 'r') as f:
        texto = f.read()
    # Procura pela matriz de confusão no texto
    match = re.search(r'Matriz de confusão:\s*\n(\[\[.*?\]\])', texto, re.DOTALL)
    if match:
        matriz_str = match.group(1)
        # Extrai todos os números da matriz
        numeros = [int(n) for n in re.findall(r'\d+', matriz_str)]
        lado = int(len(numeros) ** 0.5)
        if lado * lado != len(numeros):
            print(f'Matriz de confusão com formato inesperado em {relatorio_path}')
            return [], []
        matriz = [numeros[i*lado:(i+1)*lado] for i in range(lado)]
        # Reconstrói listas de reais e previstos a partir da matriz
        y_true = []
        y_pred = []
        for real in range(lado):
            for pred in range(lado):
                y_true += [real] * matriz[real][pred]
                y_pred += [pred] * matriz[real][pred]
        return y_true, y_pred
    else:
        print(f'Matriz de confusão não encontrada em {relatorio_path}')
        return [], []

# Escolha o modelo: 'Random Forest' ou 'XGBoost'
modelo = 'Random Forest'
dir_relatorios = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..', 'PREVISOR', 'Data', modelo, 'RelatorioClassificacao'
    )
)
arquivos = [arq for arq in os.listdir(dir_relatorios) if arq.endswith('.txt')]

for arquivo in arquivos:
    caminho = os.path.join(dir_relatorios, arquivo)
    y_true, y_pred = extrai_predicoes_reais(caminho)
    if y_true and y_pred:
        plt.figure(figsize=(5,5))
        plt.scatter(y_true, y_pred, alpha=0.2)
        plt.xlabel('Valor Real')
        plt.ylabel('Valor Previsto')
        plt.title(f'Dispersão de Predições - {arquivo.replace("relatorio_", "").replace(".txt", "").capitalize()}', pad=20)
        plt.xticks([0,1])
        plt.yticks([0,1])
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()