import pandas as pd
import sqlite3
import os
import sys
import re
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.logger import logger
from config.settings import settings

# --- Configuration --- #
EXTRACAO_MTECH_DIR = settings.EXTRACAO_MTECH_DIR
DB_FILE = settings.DATABASE_PATH
DATABASE_DIR = os.path.dirname(DB_FILE)
TABLE_NAME = 'extracao_mtech_data'
EXCEL_SHEET_NAME = 'ag-grid'

# --- Helper Functions --- #
def rationalize_header(header_name: str) -> str:
    """
    Racionaliza o nome do cabeçalho: minúsculas, substitui espaços por underscores,
    e remove caracteres especiais.
    """
    header_name = header_name.lower()
    header_name = re.sub(r'\s+', '_', header_name) # Replace spaces with underscores
    header_name = re.sub(r'[^a-z0-9_]+', '', header_name) # Remove special characters
    # Add specific rationalizations if known common headers exist
    # Example: if 'Lote Composto' becomes 'lote_composto', no further change needed here
    return header_name

def process_lote_composto(lote_composto_value) -> str:
    """
    Processa a coluna 'Lote Composto':
    - Traz o dado anterior ao segundo hifen "-"
    - Substitui "-0" por "-"
    """
    if pd.isna(lote_composto_value) or lote_composto_value is None:
        return None

    val_str = str(lote_composto_value).strip()
    parts = val_str.split('-')
    if len(parts) >= 2:
        result = '-'.join(parts[:2])
    else:
        result = val_str

    result = result.replace('-0', '-')
    return result

def extract_fazenda(lote_composto_value):
    """
    Extrai o código da fazenda a partir do lote_composto.
    """
    if pd.isna(lote_composto_value) or lote_composto_value is None:
        return None
    val_str = str(lote_composto_value).strip()
    if '-' in val_str:
        first_part = val_str.split('-')[0]
        if first_part.isdigit():
            return int(first_part)
    elif val_str.isdigit():
        return int(val_str)
    return None

# --- Main ETL Logic --- #
def run_etl():
    logger.info("Starting ETL process for extracao_mtech data...")
    all_dataframes = []

    # Ensure database directory exists
    os.makedirs(DATABASE_DIR, exist_ok=True)

    for filename in sorted(os.listdir(EXTRACAO_MTECH_DIR)):
        if filename.endswith(('.xlsx', '.xls')) and not filename.startswith('~'):
            file_path = os.path.join(EXTRACAO_MTECH_DIR, filename)
            logger.info(f"Processing file: {filename}")
            try:
                # Read the specified sheet
                df = pd.read_excel(file_path, sheet_name=EXCEL_SHEET_NAME)

                # Rationalize headers
                df.columns = [rationalize_header(col) for col in df.columns]

                # Process 'lote_composto' column if it exists
                if 'lote_composto' in df.columns:
                    df['lote_composto'] = df['lote_composto'].apply(process_lote_composto)
                else:
                    logger.warning(f"'lote_composto' column not found in {filename}. Skipping processing for this column.")

                # Create 'fazenda' column from 'lote_composto'
                if 'lote_composto' in df.columns:
                    df['fazenda'] = df['lote_composto'].apply(extract_fazenda)
                else:
                    logger.warning(f"'lote_composto' column not found in {filename}. Cannot create 'fazenda'.")

                # Convert specified columns to date format
                date_columns = ['data_alojamento', 'data_hora_transao', 'data_evento', 'data_criao']
                for col in date_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
                    else:
                        logger.warning(f"''{col}'' column not found in {filename}. Skipping date conversion for this column.")

                # Drop rows where 'data_alojamento' is null, as it's a NOT NULL column
                if 'data_alojamento' in df.columns:
                    initial_rows = len(df)
                    df.dropna(subset=['data_alojamento'], inplace=True)
                    if len(df) < initial_rows:
                        logger.warning(f"Dropped {initial_rows - len(df)} rows from {filename} due to null 'data_alojamento'.")
                else:
                    logger.warning(f"'data_alojamento' column not found in {filename}. Cannot enforce NOT NULL constraint.")

                # Convert specified columns to numeric types
                int_columns = ['mortalidade', 'descartados', 'estoque_aves', 'idade', 'cab_alojadas']
                float_columns = ['peso', '_mortalidade', '_descartados', '_mortalidade__descartados']

                for col in int_columns:
                    if col in df.columns:
                        # Convert to numeric (float), non-convertible become NaN
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        # Keep as float to handle potential non-numeric values as NaN
                        # If you need integer type, you must ensure data is clean or define how to handle NaNs
                    else:
                        logger.warning(f"''{col}'' column not found in {filename}. Skipping integer conversion for this column.")

                for col in float_columns:
                    if col in df.columns:
                        # Convert to numeric (float)
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    else:
                        logger.warning(f"''{col}'' column not found in {filename}. Skipping float conversion for this column.")

                all_dataframes.append(df)

            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue

    if not all_dataframes:
        logger.warning("No Excel files found or processed in extracao_mtech directory.")
        return

    # Concatenate all dataframes
    master_df = pd.concat(all_dataframes, ignore_index=True)
    initial_rows = len(master_df)
    logger.info(f"Initial total rows combined: {initial_rows}")

    # --- Data Quality Rules --- #
    # 1. Remove absolute duplicates
    master_df.drop_duplicates(inplace=True)
    rows_after_dedup = len(master_df)
    logger.info(f"Rows after dropping duplicates: {rows_after_dedup} (Dropped {initial_rows - rows_after_dedup})")

    # 2. Filter valid ages and map to idade_ref
    def get_idade_ref(idade):
        if pd.isna(idade):
            return pd.NA
        try:
            idade_int = int(float(idade))
        except (ValueError, TypeError):
            return pd.NA
        valid_ages = [4, 7, 14, 21, 28, 35, 42]
        for ref in valid_ages:
            if ref - 1 <= idade_int <= ref + 1:
                return ref
        return pd.NA

    if 'idade' in master_df.columns:
        master_df['idade_ref'] = master_df['idade'].apply(get_idade_ref)
        master_df.dropna(subset=['idade_ref'], inplace=True)
        master_df['idade_ref'] = master_df['idade_ref'].astype(int)
        rows_after_age_filter = len(master_df)
        logger.info(f"Rows after age filter: {rows_after_age_filter} (Dropped {rows_after_dedup - rows_after_age_filter})")
        logger.info(f"Age distribution:\n{master_df['idade_ref'].value_counts().sort_index()}")

    # 3. Cast fazenda to INTEGER
    if 'fazenda' in master_df.columns:
        master_df['fazenda'] = pd.to_numeric(master_df['fazenda'], errors='coerce').astype('Int64')

    final_rows = len(master_df)
    logger.info(f"Final rows ready for DB: {final_rows}")

    # Save to SQLite database
    try:
        conn = sqlite3.connect(DB_FILE)
        master_df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
        conn.close()
        logger.info(f"Data successfully loaded into {DB_FILE} in table '{TABLE_NAME}'.")
    except Exception as e:
        logger.error(f"Error saving data to SQLite database: {e}")

if __name__ == "__main__":
    # Ensure logging is set up when running directly
    from src.utils.logger import setup_logging
    setup_logging()
    run_etl()