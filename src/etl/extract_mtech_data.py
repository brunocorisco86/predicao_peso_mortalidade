# src/etl/extract_mtech_data.py

import pandas as pd
import sqlite3
import os
import re
from src.utils.logger import logger
from config.settings import settings

# --- Configuration --- #
EXTRACAO_MTECH_DIR = 'data/raw/extracao_mtech/'
DATABASE_DIR = 'database/'
DB_FILE = os.path.join(DATABASE_DIR, 'prediction_data.db')
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

def process_lote_composto(lote_composto_value: str) -> str:
    """
    Processa a coluna 'Lote Composto':
    - Traz o dado anterior ao segundo hifen "-"
    - Substitui "-0" por "-"
    """
    if not isinstance(lote_composto_value, str):
        return lote_composto_value # Return as is if not a string

    parts = lote_composto_value.split('-')
    if len(parts) >= 2:
        # Join parts up to the second hyphen (index 1)
        result = '-'.join(parts[:2])
    else:
        result = lote_composto_value

    result = result.replace('-0', '-')
    return result

# --- Main ETL Logic --- #
def run_etl():
    logger.info("Starting ETL process for extracao_mtech data...")
    all_dataframes = []

    # Ensure database directory exists
    os.makedirs(DATABASE_DIR, exist_ok=True)

    for filename in os.listdir(EXTRACAO_MTECH_DIR):
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
                int_columns = ['mortalidade', '_mortalidade', 'descartados', '_descartados', '_mortalidade__descartados']
                float_columns = ['peso']

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
    logger.info(f"Total rows processed: {len(master_df)}")

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
