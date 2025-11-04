import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, f1_score
)

# --- 1. CONFIGURAÇÕES GLOBAIS ---

# Defina os diretórios base conforme a estrutura do seu projeto
ANEEL_DIR = '../ANEEL/Data/Filtrados'
INMET_DIR = '../INMET/Data/Filtrados'
SAIDA_DIR = 'Data/Random Forest'

# Anos para processar (AJUSTADO: Alinhado com o log de console, que parou em 2023)
ANOS = list(range(2020, 2024))  # 2020, 2021, 2022, 2023

# Mapeia o nome do filtro (ANEEL) para o nome do arquivo (INMET)
# (CORRIGIDO: Adicionado _filtrado.csv aos nomes dos arquivos INMET)
CIDADES_CONFIG = {
    'Lagoa Vermelha': 'LagoaVermelha_filtrado.csv',
    'Passo Fundo': 'PassoFundo_filtrado.csv',
    'Porto Alegre': 'PortoAlegre_filtrado.csv',
    'Santa Maria': 'SantaMaria_filtrado.csv'
}

# Features a serem usadas do INMET
# (As colunas de limpeza são baseadas no exemplo fornecido)
COLS_METEO_PARA_CONVERTER = [
    'Temp. Ins. (C)', 'Vel. Vento (m/s)', 'Raj. Vento (m/s)',
    'Pressao Ins. (hPa)', 'Chuva (mm)'
]
FEATURES = [
    'Temp. Ins. (C)', 'Vel. Vento (m/s)', 'Raj. Vento (m/s)',
    'Pressao Ins. (hPa)', 'Chuva (mm)'
]
TARGET = 'interrupcao_real'


# --- 2. FUNÇÕES AUXILIARES ---

def capture_output(func, *args, **kwargs):
    """Captura a saída de uma função (como print) e retorna como string."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        func(*args, **kwargs)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    return output


# --- 3. FUNÇÕES DE CARREGAMENTO DE DADOS ---

def load_aneel_data(aneel_dir, anos):
    """Carrega e concatena todos os arquivos de interrupção da ANEEL."""
    print(f"[ANEEL] Carregando dados de {anos[0]} a {anos[-1]}...")
    all_interruptions = []
    for ano in anos:
        filename = f'interrupcoes_rge_sul_filtrado_{ano}.csv'
        filepath = os.path.join(aneel_dir, filename)
        try:
            df_int = pd.read_csv(filepath, sep=';', decimal=',', low_memory=False)
            df_int['AnoFonte'] = ano  # Adiciona coluna para referência
            all_interruptions.append(df_int)
            print(f"  - Carregado: {filename} ({len(df_int)} registros)")
        except FileNotFoundError:
            print(f"  - AVISO: Arquivo não encontrado: {filepath}. Pulando.")
        except Exception as e:
            print(f"  - ERRO ao ler {filepath}: {e}. Pulando.")

    if not all_interruptions:
        print("[ANEEL] ERRO CRÍTICO: Nenhum arquivo da ANEEL foi carregado.")
        return None

    df_full = pd.concat(all_interruptions, ignore_index=True)
    print(f"[ANEEL] Total de {len(df_full)} registros de interrupção carregados.\n")
    return df_full


def load_inmet_data_for_city(inmet_dir, anos, cidade_arquivo_nome):
    """Carrega e concatena todos os dados do INMET para UMA cidade."""
    print(f"[INMET] Carregando dados para '{cidade_arquivo_nome}' de {anos[0]} a {anos[-1]}...")
    all_meteo = []
    for ano in anos:
        # Estrutura: ../INMET/Data/Filtrados/2020/LagoaVermelha_filtrado.csv
        filepath = os.path.join(inmet_dir, str(ano), cidade_arquivo_nome)
        try:
            df_meteo = pd.read_csv(filepath, sep=';', quotechar='"', decimal=',')
            df_meteo['AnoFonte'] = ano
            all_meteo.append(df_meteo)
            print(f"  - Carregado: {filepath} ({len(df_meteo)} registros)")
        except FileNotFoundError:
            print(f"  - AVISO: Arquivo não encontrado: {filepath}. Pulando.")
        except Exception as e:
            print(f"  - ERRO ao ler {filepath}: {e}. Pulando.")

    if not all_meteo:
        print(f"[INMET] ERRO CRÍTICO: Nenhum arquivo do INMET foi carregado para {cidade_arquivo_nome}.")
        return None

    df_full = pd.concat(all_meteo, ignore_index=True)
    print(f"[INMET] Total de {len(df_full)} registros meteorológicos carregados para {cidade_arquivo_nome}.\n")
    return df_full


# --- 4. FUNÇÃO DE PRÉ-PROCESSAMENTO E MERGE ---

def preprocess_and_merge_data(df_clima_raw, df_aneel_raw, cidade_nome_filtro):
    """
    Limpa, processa e une os dados meteorológicos e de interrupção
    para uma cidade específica.
    """
    print(f"[Processamento] Iniciando pipeline para: {cidade_nome_filtro}")

    # --- 4.1. Processamento INMET (Clima) ---
    df_clima = df_clima_raw.copy()
    # Renomeia colunas para consistência (exemplo do script anterior)
    df_clima = df_clima.rename(columns={'Data': 'Data', 'Hora (UTC)': 'Hora'})

    # Garante que as colunas de features existem antes de converter
    cols_presentes = [col for col in COLS_METEO_PARA_CONVERTER if col in df_clima.columns]
    
    for col in cols_presentes:
        # Limpeza robusta (remove aspas, troca vírgula, converte para float)
        df_clima[col] = (
            df_clima[col]
            .astype(str)
            .str.replace('"', '', regex=False)
            .str.replace(',', '.', regex=False)
            .replace('', np.nan)
        )
        df_clima[col] = pd.to_numeric(df_clima[col], errors='coerce')

    # Remove linhas onde as features essenciais são nulas
    df_clima.dropna(subset=cols_presentes, inplace=True)

    # Criar a coluna de datetime para o join
    try:
        df_clima['Datetime'] = pd.to_datetime(
            df_clima['Data'] + ' ' + df_clima['Hora'].astype(str).str.zfill(4),
            format='%d/%m/%Y %H%M', errors='coerce'
        )
    except Exception as e:
        print(f"[Processamento] ERRO ao converter Data/Hora do INMET: {e}")
        # Tenta formato alternativo (YYYY-MM-DD) se o primeiro falhar
        try:
             df_clima['Datetime'] = pd.to_datetime(
                df_clima['Data'] + ' ' + df_clima['Hora'].astype(str).str.zfill(4),
                format='%Y-%m-%d %H%M', errors='coerce'
            )
        except Exception as e2:
            print(f"[Processamento] ERRO FATAL ao converter Data/Hora do INMET (formato desconhecido): {e2}")
            return None

    df_clima = df_clima.dropna(subset=['Datetime'])
    df_clima = df_clima.drop_duplicates(subset=['Datetime']) # Garante unicidade
    df_clima = df_clima.set_index('Datetime')

    # Agrupar dados por hora (resample) para garantir 1 registro/hora
    # Usa a média se houver múltiplos registros na mesma hora (raro)
    df_clima_hourly = df_clima[cols_presentes].resample('H').mean()
    # Remove horas que não tinham dados (resultam em NaN após resample)
    df_clima_hourly = df_clima_hourly.dropna(subset=FEATURES)

    print(f"[Processamento] Dados INMET limpos. {len(df_clima_hourly)} registros/hora válidos.")


    # --- 4.2. Processamento ANEEL (Interrupções) ---
    df_aneel = df_aneel_raw.copy()
    
    # Filtro 1: Apenas a cidade de interesse
    # (Ajuste: usar regex \b para garantir a palavra exata, ex: 'Porto Alegre' e não 'Alegrete')
    cidade_regex = r'\b' + pd.Series(cidade_nome_filtro).str.replace(r'[^\w\s]', '', regex=True)[0] + r'\b'
    df_aneel_cidade = df_aneel[
        df_aneel['DscConjuntoUnidadeConsumidora'].str.contains(cidade_regex, case=False, na=False, regex=True)
    ].copy()
    
    if df_aneel_cidade.empty:
        print(f"[Processamento] AVISO: Nenhuma interrupção encontrada para '{cidade_nome_filtro}'.")
        # Retorna os dados do INMET com alvo = 0
        df_clima_hourly[TARGET] = 0
        return df_clima_hourly[FEATURES + [TARGET]]

    # Filtro 2: Apenas interrupções reais, não programadas e ambientais
    df_interrupcoes_reais = df_aneel_cidade[
        (df_aneel_cidade['IdeMotivoInterrupcao'] == 0) &
        (df_aneel_cidade['DscTipoInterrupcao'] == 'Não Programada') &
        (df_aneel_cidade['DscFatoGeradorInterrupcao'].str.contains('Meio ambiente', na=False, case=False))
    ].copy()

    if df_interrupcoes_reais.empty:
        print(f"[Processamento] AVISO: Nenhuma interrupção (Não Programada, Meio Ambiente) encontrada para '{cidade_nome_filtro}'.")
        df_clima_hourly[TARGET] = 0
        return df_clima_hourly[FEATURES + [TARGET]]

    # Focar na data de início da interrupção (resolução horária)
    df_interrupcoes_reais['DatetimeInicio'] = pd.to_datetime(df_interrupcoes_reais['DatInicioInterrupcao'], errors='coerce')
    df_interrupcoes_reais = df_interrupcoes_reais.dropna(subset=['DatetimeInicio'])
    
    # Arredonda para o "chão" da hora (ex: 14:59 -> 14:00)
    df_interrupcoes_reais['HoraInterrupcao'] = df_interrupcoes_reais['DatetimeInicio'].dt.floor('H')

    # Marcar horas com interrupções reais (Target)
    # Pega apenas os índices únicos de hora
    horas_com_interrupcao = df_interrupcoes_reais['HoraInterrupcao'].unique()

    print(f"[Processamento] {len(horas_com_interrupcao)} horas únicas com interrupções reais (Não Prog, Ambiental) encontradas.")

    # --- 4.3. Merge e Criação da Variável Alvo ---
    df_final = df_clima_hourly.copy()

    # Variável Alvo: 1 se o índice (hora) está na lista de horas com interrupção, 0 caso contrário
    df_final[TARGET] = df_final.index.isin(horas_com_interrupcao).astype(int)

    print(f"Dados prontos. Total de amostras: {len(df_final)}")
    target_count = df_final[TARGET].sum()
    if len(df_final) > 0:
        print(f"Total de eventos de interrupção real: {target_count} ({target_count / len(df_final) * 100:.2f}%)")
    
    # Garante que só temos as colunas necessárias
    return df_final[FEATURES + [TARGET]]


# --- 5. FUNÇÃO DE TREINAMENTO E AVALIAÇÃO ---

def train_and_evaluate_model(df_final, cidade_nome, saida_dir):
    """Treina o modelo Random Forest com Grid Search e salva o relatório."""

    if df_final is None or df_final.empty:
        print("[Treinamento] Não foi possível processar os dados. Encerrando treinamento.")
        return

    print(f"\n[Treinamento] Iniciando para: {cidade_nome}")
    
    X = df_final[FEATURES]
    y = df_final[TARGET]

    # --- Verificação de Classe Única ---
    # MODIFICADO: Verifica se há pelo menos 2 classes (0 e 1)
    if len(np.unique(y)) < 2:
        print(f"  AVISO: Apenas UMA classe (Todos {y.iloc[0]}) encontrada para {cidade_nome}.")
        print("  Treinamento abortado. Gerando relatório de classe única.")
        
        # Texto de métricas (do exemplo "bom")
        metricas_texto = "A avaliação do desempenho dos modelos foi realizada por meio das métricas: \\textit{acurácia}, \\textit{precisão}, \\textit{revocação (recall)}, \\textit{F1-score} e \\textit{matriz de confusão}."

        # Gerar um relatório "dummy" de erro
        report_content = f"""
========================================================
  RELATÓRIO DE TREINAMENTO RANDOM FOREST (CLASSE ÚNICA)
========================================================
{metricas_texto}
--------------------------------------------------------
Modelo treinado para a cidade: {cidade_nome}
Features utilizadas: {FEATURES}
Target Column (Alvo): {TARGET}
        
ERRO: Treinamento abortado.
O dataset SÓ contém a classe 0 (Sem Interrupção).
Total de amostras: {len(df_final)}
Total de eventos de interrupção real: {df_final[TARGET].sum()}
========================================================
"""
        # Salvar o relatório de erro
        safe_city_name = cidade_nome.replace(" ", "_").replace("/", "")
        report_filename = f'relatorio_{safe_city_name}_random_forest_ERRO_CLASSE_UNICA.txt'
        report_path = os.path.join(saida_dir, report_filename)
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"\n[SUCESSO] Relatório de erro (classe única) salvo em: {report_path}")
        except Exception as e:
            print(f"\n[ERRO] Falha ao salvar relatório de erro em {report_path}: {e}")
        return # Aborta a função de treinamento para esta cidade

    # --- Se houver 2 classes, o treinamento normal continua ---
    print("  2. Definindo Features e Alvo...")
    
    # Divisão dos dados (80% treino, 20% teste) com estratificação
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        # Fallback se a estratificação falhar (poucas amostras)
        print("  AVISO: Stratify falhou (poucas amostras de uma classe). Tentando sem stratify.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Verifica se o split resultou em apenas uma classe no treino
    if len(np.unique(y_train)) < 2:
        print(f"  ERRO: O conjunto de treino para {cidade_nome} só contém uma classe após o split. Abortando GridSearch.")
        # (Isso pode acontecer se houver < 5 eventos positivos, e o cv=3 falhar)
        return

    print(f"  Tamanho do conjunto de treino: {len(X_train)} (Positivos: {y_train.sum()})")
    print(f"  Tamanho do conjunto de teste: {len(X_test)} (Positivos: {y_test.sum()})")

    # --- Configuração do Grid Search ---
    print("  3. Iniciando Grid Search para Random Forest (pode levar alguns minutos)...")
    
    # Grade de parâmetros (mantida do exemplo "bom")
    param_grid = {
        'n_estimators': [100, 200],         # Reduzido para velocidade
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced']       # Essencial para dados desbalanceados
    }
    
    # Usando F1 como métrica de otimização
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                               scoring='f1', cv=3, verbose=1, n_jobs=-1) # cv=3 para velocidade

    try:
        grid_search.fit(X_train, y_train)
    except ValueError as e:
        print(f"  ERRO CRÍTICO durante o fit do GridSearchCV: {e}")
        print("  Isso geralmente acontece se o 'cv' (cross-validation) não puder criar folds com ambas as classes.")
        return

    best_rf = grid_search.best_estimator_
    print(f"  Melhores Hiperparâmetros encontrados: {grid_search.best_params_}")
    print(f"  Melhor F1 na Validação Cruzada (CV): {grid_search.best_score_:.4f}")

    # --- Avaliação no Conjunto de Teste ---
    print("  4. Avaliando o modelo no Conjunto de Teste...")
    
    y_pred = best_rf.predict(X_test)
    
    # Verifica se o predict_proba está disponível (se o modelo não colapsou para uma classe)
    try:
        y_proba = best_rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except (AttributeError, ValueError):
        print("  AVISO: Não foi possível calcular o AUC (provavelmente o modelo só prevê uma classe).")
        auc = 0.0 # Define AUC como 0 se falhar

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred) # F1 para a classe positiva (1)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Relatório de Classificação (capturado como string)
    class_report_str = capture_output(
        classification_report, y_test, y_pred,
        target_names=['0 (Sem Interrupção)', '1 (Com Interrupção)'],
        zero_division=0  # CORRIGIDO: de zero_division_behavior para zero_division
    )

    # Importância das Features
    feature_importance = pd.Series(best_rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    
    # Texto de métricas (do exemplo "bom")
    metricas_texto = "A avaliação do desempenho dos modelos foi realizada por meio das métricas: \\textit{acurácia}, \\textit{precisão}, \\textit{revocação (recall)}, \\textit{F1-score} e \\textit{matriz de confusão}."

    # --- Geração do Relatório ---
    report_content = f"""
========================================================
  RELATÓRIO DE TREINAMENTO RANDOM FOREST (EVENTOS REAIS)
========================================================
{metricas_texto}
--------------------------------------------------------
Modelo treinado para a cidade: {cidade_nome}
Features utilizadas: {FEATURES}
Target Column (Alvo): {TARGET}
Proporção Positiva (Interrupções Reais no dataset total): {df_final[TARGET].mean() * 100:.2f}%
(Baseado em {df_final[TARGET].sum()} eventos positivos em {len(df_final)} amostras)
--------------------------------------------------------
Melhores Hiperparâmetros (GridSearch CV={grid_search.cv} Folds):
{grid_search.best_params_}

Melhor F1 na Validação Cruzada (CV): {grid_search.best_score_:.4f}
--------------------------------------------------------
Métricas no Conjunto de Teste (20%):

Acurácia: {accuracy:.4f}
AUC: {auc:.4f}
F1-Score (Classe 1): {f1:.4f}
--------------------------------------------------------
Matriz de Confusão (Teste):
  [Verdadeiro Negativo (TN)   Falso Positivo (FP)]
  [Falso Negativo (FN)      Verdadeiro Positivo (TP)]
{conf_matrix}

Relatório de Classificação (Teste):
{class_report_str}
--------------------------------------------------------
Importância das Features (Baseada em Gini):
{feature_importance.to_string()}
========================================================
"""
    # Exibe resumo no terminal
    print("\n  5. Relatório de Treinamento Gerado (Salvo em arquivo):")
    print(f"  Acurácia no Teste: {accuracy:.4f}")
    print(f"  F1-Score (Classe 1) no Teste: {f1:.4f}")
    
    # Salvar o relatório
    # Garante que o nome do arquivo é seguro
    safe_city_name = cidade_nome.replace(" ", "_").replace("/", "")
    report_filename = f'relatorio_{safe_city_name}_random_forest.txt'
    report_path = os.path.join(saida_dir, report_filename)
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"\n[SUCESSO] Relatório salvo em: {report_path}")
    except Exception as e:
        print(f"\n[ERRO] Falha ao salvar relatório em {report_path}: {e}")


# --- 6. EXECUÇÃO PRINCIPAL ---

def main():
    """Orquestra o pipeline de carregamento e treinamento para todas as cidades."""
    
    # Garante que o diretório de saída exista
    try:
        os.makedirs(SAIDA_DIR, exist_ok=True)
        print(f"Diretório de saída verificado: {SAIDA_DIR}")
    except Exception as e:
        print(f"ERRO CRÍTICO: Não foi possível criar o diretório de saída {SAIDA_DIR}. Erro: {e}")
        return

    # 1. Carregar todos os dados da ANEEL (uma única vez)
    df_aneel_full_raw = load_aneel_data(ANEEL_DIR, ANOS)
    if df_aneel_full_raw is None:
        print("Execução abortada pois dados da ANEEL não foram carregados.")
        return

    # 2. Iterar por cada cidade, carregar seus dados INMET e treinar
    for cidade_nome_filtro, cidade_arquivo_nome in CIDADES_CONFIG.items():
        
        print(f"\n{'='*70}\nProcessando pipeline para: {cidade_nome_filtro}\n{'='*70}")
        
        # 2a. Carregar dados INMET para esta cidade
        df_clima_raw = load_inmet_data_for_city(INMET_DIR, ANOS, cidade_arquivo_nome)
        if df_clima_raw is None:
            print(f"Pulando {cidade_nome_filtro} por falta de dados INMET.")
            continue
            
        # 2b. Processar e unir os dados
        df_processed = preprocess_and_merge_data(df_clima_raw, df_aneel_full_raw, cidade_nome_filtro)
        if df_processed is None or df_processed.empty:
            print(f"Pulando {cidade_nome_filtro} por falha no pré-processamento.")
            continue

        # 2c. Treinar e avaliar o modelo
        train_and_evaluate_model(df_processed, cidade_nome_filtro, SAIDA_DIR)

    print(f"\n{'='*70}\nPipeline concluído para todas as cidades.\n{'='*70}")


if __name__ == "__main__":
    main()