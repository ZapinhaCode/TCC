import pandas as pd
import re
import os

DATA_DIR = "Data"
DATA_FILTRADO = "Data/Filtrados"

conjuntos = [
    'Passo Fundo 1',
    'Santa Maria',
    'SANTA MARIA',
    'SANTA MARIA 1',
    'SANTA MARIA 2',
    'SANTA MARIA 4',
    'SANTA MARIA 5',
    'Lagoa Vermelha'
]

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

def processar_csv_aneel(input_file, output_file):
    df = pd.read_csv(input_file, sep=';', dtype=str, encoding='latin1', low_memory=False)
    df_rge_sul = df[
        df['SigAgente'].str.contains('RGE SUL', na=False) &
        df['DscConjuntoUnidadeConsumidora'].isin(conjuntos)
    ]
    
    regex_excluir = "|".join(map(re.escape, valores_excluir))
    df_rge_sul = df_rge_sul[~df_rge_sul['DscFatoGeradorInterrupcao'].str.contains(regex_excluir, na=False)]
    
    df_rge_sul.loc[
        df_rge_sul['DscConjuntoUnidadeConsumidora'].str.startswith('PORTO ALEGRE', na=False), 
        'DscConjuntoUnidadeConsumidora'
    ] = 'Porto Alegre'

    df_rge_sul.loc[
        df_rge_sul['DscConjuntoUnidadeConsumidora'].str.upper().str.startswith('SANTA MARIA', na=False), 
        'DscConjuntoUnidadeConsumidora'
    ] = 'Santa Maria'

    df_rge_sul.loc[
        df_rge_sul['DscConjuntoUnidadeConsumidora'].str.upper().str.startswith('PASSO FUNDO', na=False), 
        'DscConjuntoUnidadeConsumidora'
    ] = 'Passo Fundo'

    df_rge_sul.to_csv(output_file, index=False, sep=';')
    print(f'✅ Arquivo salvo: {output_file}')

def processar_todos_csvs():
    if not os.path.exists(DATA_FILTRADO):
        os.makedirs(DATA_FILTRADO)
    for file in os.listdir(DATA_DIR):
        if file.endswith('.csv') and file.startswith('interrupcoes-energia-eletrica-'):
            ano = file.split('-')[-1].replace('.csv', '')
            input_path = os.path.join(DATA_DIR, file)
            output_path = os.path.join(DATA_FILTRADO, f'interrupcoes_rge_sul_filtrado_{ano}.csv')
            try:
                processar_csv_aneel(input_path, output_path)
            except Exception as e:
                print(f'❌ Erro ao processar {input_path}: {e}')

if __name__ == "__main__":
    processar_todos_csvs()