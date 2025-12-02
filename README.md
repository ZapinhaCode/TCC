# Análise de Interrupções de Energia e Variáveis Climáticas no RS (2020-2024)

## Visão Geral

Este projeto realiza uma análise detalhada das interrupções de energia elétrica no Rio Grande do Sul entre 2020 e 2024, correlacionando eventos de falha com variáveis meteorológicas. Utiliza dados públicos da ANEEL (interrupções) e do INMET (clima), além de técnicas de ciência de dados, aprendizado de máquina e visualização para gerar insights sobre padrões, causas e previsibilidade das interrupções.

---

## Estrutura do Projeto
```
    TCC/
    │
    ├── ANALISE/
    │ ├── app_xgboost.py
    │ ├── app_random_forest.py
    │ └── Data/
    │ ├── XGBoost/
    │ │ └── relatorio_<cidade>xgboost.txt
    │ └── Random Forest/
    │ └── relatorio<cidade>random_forest.txt
    │
    ├── ANEEL/
    │ ├── app.py
    │ └── Data/
    │ ├── interrupcoes-energia-eletrica-2020.csv
    │ ├── ...
    │ └── Filtrados/
    │ └── interrupcoes_rge_sul_filtrado<ano>.csv
    │
    ├── INMET/
    │ ├── app.py
    │ └── Data/
    │ ├── <ano>/
    │ │ └── <Cidade>.csv
    │ └── Filtrados/
    │ └── <ano>/<Cidade>_filtrado.csv
    │
    ├── GRAFICOS/
    │ ├── ANALISE/
    │ │ └── app_matriz_confusao_analise.py
    │ ├── ANEEL/
    │ │ ├── app_causas_interrupções.py
    │ │ ├── app_distribuicao_anual_interrupcoes.py
    │ │ └── app_graficos_contagem_total_interrupcoes_cidade.py
    │ └── Images/
    │ ├── ANALISE/
    │ └── ANEEL/
    │
    └── README.md
```

---

## Fluxo de Trabalho

### 1. **Coleta e Filtragem dos Dados**

- **ANEEL:**  
  - Arquivos originais em `ANEEL/Data/`.
  - Filtragem e limpeza em `ANEEL/app.py`, gerando arquivos em `ANEEL/Data/Filtrados/`.
  - Principais campos: `DscConjuntoUnidadeConsumidora` (cidade), `DscFatoGeradorInterrupcao` (causa), datas e horários das interrupções.

- **INMET:**  
  - Arquivos originais em `INMET/Data/<ano>/<Cidade>.csv`.
  - Filtragem e limpeza em `INMET/app.py`, gerando arquivos em `INMET/Data/Filtrados/<ano>/<Cidade>_filtrado.csv`.
  - Principais variáveis: `Temp. Ins. (C)`, `Vel. Vento (m/s)`, `Raj. Vento (m/s)`, `Pressao Ins. (hPa)`, `Chuva (mm)`.

---

### 2. **Análise Exploratória e Visualização**

- **Gráficos ANEEL:**
  - Contagem total de interrupções por cidade (`app_graficos_contagem_total_interrupcoes_cidade.py`)
  - Distribuição anual e causas das interrupções (`app_distribuicao_anual_interrupcoes.py`, `app_causas_interrupções.py`)
  - Gráficos salvos em `GRAFICOS/Images/ANEEL/`

- **Gráficos INMET:**
  - Distribuição das variáveis climáticas (boxplots, barras, pizza)
  - Comparação entre situações de interrupção e normalidade (boxplot agrupado)
  - Gráficos salvos em `GRAFICOS/Images/INMET/`

- **Gráficos de Análise de Modelos:**
  - Matrizes de confusão comparativas entre modelos e cidades (`app_matriz_confusao_analise.py`)
  - Salvos em `GRAFICOS/Images/ANALISE/`

---

### 3. **Modelagem Preditiva**

- **Modelos Utilizados:**
  - XGBoost (`app_xgboost.py`)
  - Random Forest (`app_random_forest.py`)

- **Pipeline:**
  1. Carregamento dos dados filtrados do INMET.
  2. Criação da variável alvo `possivel_interrupcao` (chuva > 10mm OU rajada de vento > 10m/s).
  3. Balanceamento das classes com SMOTE.
  4. Otimização de hiperparâmetros com GridSearchCV.
  5. Ajuste de limiar de decisão para maximizar F1-score.
  6. Avaliação com métricas: acurácia, F1, recall, precision, AUC, matriz de confusão.
  7. Relatórios salvos em `ANALISE/Data/XGBoost/` e `ANALISE/Data/Random Forest/`.

- **Validação Cruzada:**
  - Os modelos são avaliados por cidade e por ano, com validação cruzada estratificada.

---

### 4. **Validação com Dados Reais da ANEEL**

- Os scripts fazem a correspondência entre datas/horários dos eventos do INMET e as interrupções reais da ANEEL.
- A coluna `interrupcao_real` é criada para indicar se houve interrupção real naquela data.
- Permite comparar a performance dos modelos com a realidade.

---

## Como Executar

1. **Filtrar os dados brutos:**
   - Execute `ANEEL/app.py` e `INMET/app.py` para gerar os arquivos filtrados.

2. **Treinar e avaliar os modelos:**
   - Execute `ANALISE/app_xgboost.py` e `ANALISE/app_random_forest.py`.

3. **Gerar gráficos e relatórios:**
   - Execute os scripts em `GRAFICOS/ANEEL/`, `GRAFICOS/INMET/` e `GRAFICOS/ANALISE/` conforme desejado.

4. **Verifique os resultados:**
   - Relatórios de modelos: `ANALISE/Data/XGBoost/` e `ANALISE/Data/Random Forest/`
   - Gráficos: `GRAFICOS/Images/ANEEL/`, `GRAFICOS/Images/INMET/`, `GRAFICOS/Images/ANALISE/`

---

## Principais Resultados

- **Desbalanceamento de Classes:**  
  A maioria dos registros representa situações de normalidade, com poucos eventos extremos (interrupções).

- **Importância das Variáveis:**  
  Chuva e vento são os principais fatores meteorológicos associados a interrupções.

- **Desempenho dos Modelos:**  
  O ajuste de hiperparâmetros e limiar de decisão melhora a capacidade dos modelos em identificar corretamente eventos de interrupção.

- **Validação Real:**  
  A correspondência entre previsão e interrupções reais permite avaliar a utilidade prática dos modelos.

---

## Observações Importantes

- **Atenção:**  
  A variável alvo `possivel_interrupcao` é baseada em uma regra meteorológica (chuva > 10mm OU rajada de vento > 10m/s) e não representa diretamente as interrupções reais reportadas pela ANEEL.
- **Limitações:**  
  O modelo não prevê interrupções reais, mas sim condições meteorológicas severas que podem estar associadas a elas.
- **Reprodutibilidade:**  
  Todos os caminhos e nomes de arquivos devem ser mantidos conforme a estrutura apresentada para garantir o funcionamento dos scripts.

---

## Dependências

- Python 3.8+

Instale todas as dependências com:

```bash
pip install -r requirements.txt
```