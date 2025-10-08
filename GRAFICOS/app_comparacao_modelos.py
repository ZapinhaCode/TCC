import os
import re
import matplotlib.pyplot as plt
import numpy as np

def extrai_metricas(relatorio_path):
    with open(relatorio_path, 'r') as f:
        texto = f.read()
    # Extrai acurácia (linha que contém 'accuracy' e um número)
    match_acc = re.search(r'accuracy[\s:]*([0-9.]+)', texto, re.IGNORECASE)
    acuracia = float(match_acc.group(1)) if match_acc else None

    # Extrai macro avg (linha que começa com 'macro avg')
    match_macro = re.search(r'macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', texto)
    if match_macro:
        precisao = float(match_macro.group(1))
        recall = float(match_macro.group(2))
        f1 = float(match_macro.group(3))
    else:
        precisao = recall = f1 = None
    return acuracia, precisao, recall, f1

def coleta_metricas(diretorio):
    metricas = {}
    if not os.path.exists(diretorio):
        print(f'Diretório não encontrado: {diretorio}')
        return metricas
    for arquivo in os.listdir(diretorio):
        if arquivo.endswith('.txt'):
            cidade = arquivo.replace('relatorio_', '').replace('.txt', '')
            acuracia, precisao, recall, f1 = extrai_metricas(os.path.join(diretorio, arquivo))
            metricas[cidade] = {'Acurácia': acuracia, 'Precisão': precisao, 'Recall': recall, 'F1-score': f1}
    return metricas

# Diretórios dos relatórios (relativos ao local deste script)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PREVISOR', 'Data'))
dir_rf = os.path.join(base_dir, 'Random Forest', 'RelatorioClassificacao')
dir_xgb = os.path.join(base_dir, 'XGBoost', 'RelatorioClassificacao')

metricas_rf = coleta_metricas(dir_rf)
metricas_xgb = coleta_metricas(dir_xgb)

# Cidades presentes nas duas pastas
cidades = sorted(set(metricas_rf.keys()) & set(metricas_xgb.keys()))
metrica_labels = ['Acurácia', 'Precisão', 'Recall', 'F1-score']

for cidade in cidades:
    rf_vals = [metricas_rf[cidade][m] if metricas_rf[cidade][m] is not None else 0 for m in metrica_labels]
    xgb_vals = [metricas_xgb[cidade][m] if metricas_xgb[cidade][m] is not None else 0 for m in metrica_labels]

    x = np.arange(len(metrica_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, rf_vals, width, label='Random Forest', color='royalblue')
    rects2 = ax.bar(x + width/2, xgb_vals, width, label='XGBoost', color='darkorange')

    ax.set_ylabel('Valor')
    ax.set_ylim(0, 1)
    ax.set_title(f'Comparação de Métricas - {cidade.replace("cidade_", "").replace("_", " ").title()}', pad=30)
    ax.set_xticks(x)
    ax.set_xticklabels(metrica_labels, fontsize=11)
    ax.legend()

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()