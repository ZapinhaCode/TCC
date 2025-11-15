import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_report(file_path):
    """
    Lê um arquivo de relatório e extrai a cidade, o modelo e a matriz de confusão.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extrair o nome do modelo (XGBoost ou Random Forest)
        if 'xgboost' in os.path.basename(file_path).lower():
            model_name = 'XGBOOST'
        elif 'random_forest' in os.path.basename(file_path).lower():
            model_name = 'RANDOM FOREST'
        else:
            model_name = 'Desconhecido'

        # Extrair o nome da cidade
        city_match = re.search(r'relatorio_(.*)_(xgboost|random_forest)\.txt', os.path.basename(file_path), re.IGNORECASE)
        city_name = city_match.group(1).replace('_', ' ').title() if city_match else 'Desconhecida'

        # Extrair os valores da matriz de confusão
        matrix_match = re.search(r'\[\[\s*(\d+)\s+(\d+)\]\s*\[\s*(\d+)\s+(\d+)\]\]', content)
        if not matrix_match:
            print(f"Aviso: Não foi possível encontrar a matriz no arquivo: {file_path}")
            return None

        tn = int(matrix_match.group(1))
        fp = int(matrix_match.group(2))
        fn = int(matrix_match.group(3))
        tp = int(matrix_match.group(4))
        matrix = np.array([[tn, fp], [fn, tp]])

        return {
            'city': city_name,
            'model': model_name,
            'matrix': matrix
        }

    except Exception as e:
        print(f"Erro ao processar o arquivo {file_path}: {e}")
        return None

def plot_confusion_matrices(results, save_path):
    """
    Plota uma grade de matrizes de confusão (3 cidades x 2 modelos) e salva como PNG.
    """
    cities = sorted(list(set(r['city'] for r in results)))
    models = ['XGBOOST', 'RANDOM FOREST']

    fig, axes = plt.subplots(nrows=len(cities), ncols=len(models), figsize=(12, 18))
    fig.suptitle('Comparação de Matrizes de Confusão (Teste)', fontsize=20, y=1.02)

    for row, city in enumerate(cities):
        for col, model in enumerate(models):
            data = next((r for r in results if r['city'] == city and r['model'] == model), None)
            if data is None:
                axes[row, col].axis('off')
                axes[row, col].set_title(f'{city} - {model}\n(Não encontrado)')
                continue

            matrix = data['matrix']
            group_names = ['Verdadeiro Negativo (TN)', 'Falso Positivo (FP)', 
                           'Falso Negativo (FN)', 'Verdadeiro Positivo (TP)']
            group_counts = [f"{value}" for value in matrix.flatten()]
            labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
            labels = np.asarray(labels).reshape(2, 2)

            sns.heatmap(matrix, ax=axes[row, col], annot=labels, fmt='', 
                        cmap='Blues', cbar=False, annot_kws={"size": 12})

            axes[row, col].set_title(f'{city} - {model}', fontsize=14, weight='bold')
            axes[row, col].set_xlabel('Previsão', fontsize=12)
            axes[row, col].set_ylabel('Verdadeiro', fontsize=12)
            axes[row, col].set_xticklabels(['Negativo (0)', 'Positivo (1)'])
            axes[row, col].set_yticklabels(['Negativo (0)', 'Positivo (1)'])

    plt.tight_layout(pad=3.0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Gráfico salvo em: {save_path}")

# --- Execução Principal ---
if __name__ == "__main__":
    # Caminhos dos diretórios dos relatórios
    report_dirs = [
        '../../ANALISE/Data/XGBoost',
        '../../ANALISE/Data/Random Forest'
    ]

    file_paths = []
    for d in report_dirs:
        file_paths.extend(glob.glob(os.path.join(d, '*.txt')))

    if not file_paths:
        print("Nenhum arquivo .txt encontrado nos diretórios de relatório.")
        print("Por favor, coloque os arquivos de relatório nas pastas XGBoost e Random Forest.")
    else:
        results = []
        for path in file_paths:
            data = parse_report(path)
            if data:
                results.append(data)

        if results:
            save_path = os.path.join('../Images/ANALISE', 'matriz_confusao_comparativo.png')
            plot_confusion_matrices(results, save_path)