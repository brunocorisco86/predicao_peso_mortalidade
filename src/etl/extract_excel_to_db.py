import pandas as pd
import sqlite3
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import settings

# Define file paths using settings
excel_file_path = settings.FEATURES_EXCEL_PATH
db_file_path = settings.DATABASE_PATH

# Mapping from descriptive headers to codes based on the provided markdown table
header_to_code_mapping = {
    'Aviário Convencional': 'a01',
    'Aviário Climatizado': 'a02',
    'Aviário Dark House': 'a03',
    'Possui Inlets': 'a04',
    'Não Possui Inlets': 'a05',
    'Exaustores Chore Time': 'a06',
    'Exaustores GSI': 'a07',
    'Aquecimento principal por Fornalha': 'a08',
    'Aquecimento principal por Campânulas': 'a09',
    'climatização adequada': 'a10',
    'climatização inadequada': 'a11',
    'Possui 2 silos': 'a12',
    'Painel Chore Time': 'b01',
    'possui clorador ativo e conforme': 'b02',
    'Painel Edge': 'b03',
    'Usa acido para neutralizar pH H2O': 'b04',
    'Mortalidade Total Alta': 'c01',
    'Mortalidade Total Baixa': 'c02',
    'Mortalidade Inicial Alta': 'c03',
    'Mortalidade Inicial Baixa': 'c04',
    'Idade Matriz Baixa': 'c05',
    'Idade Matriz Alta': 'c06',
    'IEP seco alto (ranking)': 'c07',
    'IEP seco baixo (ranking)': 'c08',
    'CAAJ Alto (ranking)': 'c09',
    'CAAJ Baixo (ranking)': 'c10',
    'fator multiplicador peso 35dias abaixo': 'c11',
    'fator multiplicador peso 35dias acima': 'c12',
    'peso pintainho baixo': 'c13',
    'peso pintainho alto': 'c14',
    'peso pintainho em gramas': 'c15',
    'linhagem pintainho': 'c16',
    'fornecedor pintainho': 'c17',
    'Microrregiões com desafio de sanidade (aerossaculite)': 'd01',
    'Microrregiões sem desafio de sanidade (aerossaculite)': 'd02',
    'produtores com alto desafio de aerossaculite': 'd03',
    'produtores com baixo desafio de aerossaculite': 'd04',
    'produtores com desafio de aerossaculite dentro da média': 'd05',
    'vazio curto (14 a 18)': 'f01',
    'vazio médio (19 a 24)': 'f02',
    'vazio longo (mais do que 25)': 'f03',
    'numero de camas (1 a 4)': 'f04',
    'numero de camas (5 a 9)': 'f05',
    'numero de camas (mais do que 10)': 'f06',
    'lotes medicados até os 14 dias': 'f07',
    'tempo de jejum em horas': 'f15',
    'produtores com alto rendimento financeiro (10% top)': 'h01',
    'produtores com baixo rendimento financeiro (10% bottom)': 'h02',
    'Nota Promob Acima de 90%': 'i01',
    'CLASSIFICAÇÃO AVIARIO A': 'i02',
    'CLASSIFICAÇÃO AVIARIO B': 'i03',
    'CLASSIFICAÇÃO AVIARIO C': 'i04',
    'CLASSIFICAÇÃO AVIARIO D': 'i05',
    'CLASSIFICAÇÃO AVIARIO E': 'i06',
    'Aquecimento nota zero': 'i07',
    'Aviário Global Gap': 'i08',
    'Participa do projeto PICS': 'i09',
    'Assertividade - peso abaixo': 'i10',
    'Assertividade - peso acima': 'i11',
    'Densidade Nominal': 'i12',
    'FORNECEDEDOR C.VALE': 'o01',
    'FORNECEDOR PLUMA': 'o02',
    'FORNECEDOR CARMINATTI': 'o03',
    'FORNECEDOR GLOBOAVES': 'o04',
    'FORNECEDOR GRANJA REAL': 'o05',
    'teste prezinha com sorgo': 's01',
    'Passou por desafio de temperatura até os 21dias': 'v01',
    'Alojamento em Microrregião Conforme': 'w01',
    'Região Assis Chateubriand': 'w02',
    'Região Palotina': 'w03',
    'Região Terra Roxa': 'w04',
    'Região Nova Santa Rosa': 'w05',
    'Região Maripá': 'w06',
    'Região Toledo': 'w07',
    'Região Francisco Alves': 'w08',
    'Região Iporã': 'w09',
    'Região Tupãssi': 'w10',
    'Região Terceiros': 'w11',
    'distancia abatedouro longe/perto': 'x01',
    'distancia abatedouro em km': 'x02',
    'Lote de inverno': 'y01',
    'Lote de verão': 'y02',
    'lote abatido no primeiro turno': 'y03',
    'lote abatido no segundo turno': 'y04',
    'mês alojamento': 'y05',
    'mês abate': 'y06',
}

def extract_and_load_sheet(sheet_name, excel_path, db_path, header_map):
    """
    Extracts data from a specified Excel sheet, renames columns based on a map,
    removes duplicate rows to guarantee referential integrity, and loads it into an SQLite table.
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        # Drop duplicate rows upfront to ensure clean dimension tables
        df.drop_duplicates(inplace=True)

        # Rename columns that exist in our mapping
        df.rename(columns=header_map, inplace=True)

        # 1. Substitua espaços no header por underscore e letras minusculas no header
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        # Remove 'bin_' prefix and '_bin' suffix from headers
        df.columns = df.columns.str.replace('bin_', '', regex=False).str.replace('_bin', '', regex=False)

        # 2. data_alojamento é formato de data
        if 'data_alojamento' in df.columns:
            df['data_alojamento'] = pd.to_datetime(df['data_alojamento'], errors='coerce')
            df['data_alojamento'] = df['data_alojamento'].dt.strftime('%Y-%m-%d') # Format as YYYY-MM-DD string

        # 3. c17 dados em MAIUSCULO
        if 'c17' in df.columns:
            df['c17'] = df['c17'].astype(str).str.upper()

        # 4. em lote_composto padronizar e remover sufixo de núcleo (-N123)
        if 'lote_composto' in df.columns:
            def clean_lote(val):
                if pd.isna(val) or val is None:
                    return None
                val_str = str(val).strip()
                parts = val_str.split('-')
                if len(parts) >= 2:
                    res = f"{parts[0]}-{parts[1]}"
                else:
                    res = val_str
                return res.replace('-0', '-')
            df['lote_composto'] = df['lote_composto'].apply(clean_lote)

            # Adicionar coluna fazenda (RN-06: número antes do hífen)
            def extract_fazenda(lote_str):
                if pd.isna(lote_str) or lote_str is None:
                    return None
                parts = str(lote_str).split('-')
                if parts[0].isdigit():
                    return int(parts[0])
                return None
            df['fazenda'] = df['lote_composto'].apply(extract_fazenda)

        # 5. retire os prefixos 'bin_' e sufixo '_bin' quando houverem (from values)
        for col in df.select_dtypes(include=['object', 'string']).columns:
            df[col] = df[col].astype(str).str.replace('bin_', '', regex=False).str.replace('_bin', '', regex=False)

        # Re-drop duplicates after data cleaning just in case string formatting generated extra matches
        df.drop_duplicates(inplace=True)

        # Connect to SQLite database
        conn = sqlite3.connect(db_path)

        # Explicitly drop table if it exists
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {sheet_name.lower()}")
        conn.commit()

        # Save DataFrame to SQLite table with lowercase sheet name
        df.to_sql(sheet_name.lower(), conn, if_exists='replace', index=False)

        conn.close()
        print(f"Successfully extracted '{sheet_name}' ({len(df)} unique rows) and loaded into '{db_path}' table '{sheet_name.lower()}'.")

    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_path}")
    except KeyError:
        print(f"Error: Sheet '{sheet_name}' not found in {excel_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Ensure the database directory exists
    os.makedirs(os.path.dirname(db_file_path), exist_ok=True)

    # Process 'VARIABLES' sheet
    extract_and_load_sheet('VARIABLES', excel_file_path, db_file_path, header_to_code_mapping)

    # Process 'CONSTANTES' sheet
    extract_and_load_sheet('CONSTANTES', excel_file_path, db_file_path, header_to_code_mapping)