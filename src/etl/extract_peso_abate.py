import pandas as pd
import sqlite3
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.logger import logger, setup_logging
from config.settings import settings

PESO_ABATE_EXCEL = settings.PESO_ABATE_EXCEL_PATH
DB_FILE = settings.DATABASE_PATH
TABLE_NAME = 'peso_abate'

def extract_and_load_peso_abate():
    logger.info(f"Starting extraction of slaughter weight dataset from {PESO_ABATE_EXCEL}...")
    
    if not os.path.exists(PESO_ABATE_EXCEL):
        logger.error(f"Excel file not found at {PESO_ABATE_EXCEL}")
        return

    df = pd.read_excel(PESO_ABATE_EXCEL)
    logger.info(f"Loaded Excel file with {len(df)} rows and columns: {list(df.columns)}")

    # Clean and standardize column names
    # Expected columns: ['LoteComposto', 'Idade', 'D.Produção', 'PesoMedio', 'GMD']
    df.rename(columns={
        'LoteComposto': 'lote_composto',
        'Idade': 'idade_abate',
        'D.Produção': 'data_producao',
        'PesoMedio': 'peso_medio_abate_kg',
        'GMD': 'gmd_abate'
    }, inplace=True)

    # Standardize lote_composto: replace '-0' with '-' and strip whitespace
    df['lote_composto'] = df['lote_composto'].astype(str).str.strip().str.replace('-0', '-', regex=False)

    # Convert data_producao to YYYY-MM-DD
    if 'data_producao' in df.columns:
        df['data_producao'] = pd.to_datetime(df['data_producao'], errors='coerce').dt.strftime('%Y-%m-%d')

    # Convert numeric fields
    df['idade_abate'] = pd.to_numeric(df['idade_abate'], errors='coerce')
    df['peso_medio_abate_kg'] = pd.to_numeric(df['peso_medio_abate_kg'], errors='coerce')
    df['gmd_abate'] = pd.to_numeric(df['gmd_abate'], errors='coerce')

    # Calculate weight in grams
    df['peso_abate_g'] = df['peso_medio_abate_kg'] * 1000.0

    # Extract fazenda ID from lote_composto
    def extract_fazenda(lote_str):
        if pd.isna(lote_str):
            return None
        parts = str(lote_str).split('-')
        if parts[0].isdigit():
            return int(parts[0])
        return None

    df['fazenda'] = df['lote_composto'].apply(extract_fazenda)

    # Deduplicate by lote_composto keeping the latest/valid record
    initial_count = len(df)
    df.drop_duplicates(subset=['lote_composto'], keep='first', inplace=True)
    logger.info(f"Deduplicated slaughter records: {initial_count} -> {len(df)} unique lotes.")

    # Save to SQLite database
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
    
    # Create index on lote_composto
    cursor = conn.cursor()
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_peso_abate_lote ON {TABLE_NAME} (lote_composto);")
    conn.commit()
    conn.close()

    logger.info(f"Successfully loaded {len(df)} slaughter weight records into '{TABLE_NAME}' table in {DB_FILE}.")

if __name__ == "__main__":
    setup_logging()
    extract_and_load_peso_abate()
