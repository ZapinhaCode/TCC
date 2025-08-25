import pandas as pd
import re
import os

# Caminho da pasta onde está o arquivo
DATA_DIR = "Data"
DATA_FILTRADO = "Data/Filtrados"

# Nome do arquivo de entrada
# FILENAME = "interrupcoes-energia-eletrica-2024.csv"
# FILENAME = "interrupcoes-energia-eletrica-2023.csv"
# FILENAME = "interrupcoes-energia-eletrica-2022.csv"
# FILENAME = "interrupcoes-energia-eletrica-2021.csv"
FILENAME = "interrupcoes-energia-eletrica-2020.csv"

# Monta o caminho completo do arquivo
input_file = os.path.join(DATA_DIR, FILENAME)

# Lê o dataset
df = pd.read_csv(
    input_file,
    sep=';',
    dtype=str,
    encoding='latin1',
    low_memory=False
)

# FIltra apenas interrupções relacionadas a RGE do RS e também cidades específicas
conjuntos = [
    'Passo Fundo 1',
    'PORTO ALEGRE 1',
    'PORTO ALEGRE 2',
    'PORTO ALEGRE 3',
    'PORTO ALEGRE 4 - CENTRO',
    'PORTO ALEGRE 4 - CENTRO 2',
    'PORTO ALEGRE 5',
    'PORTO ALEGRE 6',
    'PORTO ALEGRE 7',
    'PORTO ALEGRE 8',
    'PORTO ALEGRE 9',
    'PORTO ALEGRE 10',
    'PORTO ALEGRE 11',
    'PORTO ALEGRE 12',
    'PORTO ALEGRE 13',
    'PORTO ALEGRE 14',
    'PORTO ALEGRE 15',
    'PORTO ALEGRE 16',
    'PORTO ALEGRE 17',
    'PORTO ALEGRE 18',
    'PORTO ALEGRE 19',
    'PORTO ALEGRE 20',
    'Santa Maria',
    'SANTA MARIA',
    'SANTA MARIA 1',
    'SANTA MARIA 2',
    'SANTA MARIA 4',
    'SANTA MARIA 5',
    'Lagoa Vermelha'
]

df_rge_sul = df[
    df['SigAgente'].str.contains('RGE SUL', na=False) &
    df['DscConjuntoUnidadeConsumidora'].isin(conjuntos)
]

# Lista de valores a excluir (sem relação com eventos climáticos)
valores_excluir = [
    "Interna;Nao Programada;Terceiros;Ligacao clandestina",
    "Interna;Nao Programada;Meio Ambiente;Animais",
    "Interna;Nao Programada;Terceiros;Empresas de servicos publicos ou suas contratadas",
    "Interna;Programada;Manutencao;Preventiva",
    "Interna;Nao Programada;Falha Operacional;Servico mal executado",
    "Interna;Nao Programada;Nao classificada",
    "Interna;Nao Programada;Proprias do Sistema;Nao identificada",
    "Interna;Nao Programada;Terceiros;Defeito interno nao afetando outras unidades consumidoras",
    "Interna;Programada;Alteracao;Para melhoria",
    "Interna;Programada;Manutencao;Corretiva",
    "Interna;Nao Programada;Terceiros;Vandalismo",
    "Interna;Programada;Alteracao;Para ampliacao"
]

# Remove as linhas com essas descrições
regex_excluir = "|".join(map(re.escape, valores_excluir))
df_rge_sul = df_rge_sul[~df_rge_sul['DscFatoGeradorInterrupcao'].str.contains(regex_excluir, na=False)]

# Salva o arquivo filtrado na pasta Data
output_file = os.path.join(DATA_FILTRADO, "interrupcoes_rge_sul_filtrado.csv")
df_rge_sul.to_csv(output_file, index=False, sep=';')

print(f'✅ Arquivo salvo: {output_file}')