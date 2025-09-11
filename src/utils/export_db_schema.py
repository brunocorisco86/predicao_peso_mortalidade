# src/utils/export_db_schema.py

import sqlite3
import os
from src.utils.logger import logger
from config.settings import settings

DATABASE_DIR = 'database/'
DB_FILE = os.path.join(DATABASE_DIR, 'prediction_data.db')
SCHEMA_OUTPUT_DIR = 'docs/'
SCHEMA_FILE = os.path.join(SCHEMA_OUTPUT_DIR, 'db_schema.sql')

def export_db_schema():
    logger.info(f"Exporting database schema from {DB_FILE}...")

    if not os.path.exists(DB_FILE):
        logger.error(f"Database file not found: {DB_FILE}. Please run the ETL script first.")
        return

    os.makedirs(SCHEMA_OUTPUT_DIR, exist_ok=True)

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Query sqlite_master to get CREATE TABLE statements
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        with open(SCHEMA_FILE, 'w') as f:
            f.write("-- SQLite Database Schema Export\n")
            f.write(f"-- Database: {DB_FILE}\n")
            f.write(f"-- Exported on: {pd.Timestamp.now()}\n\n") # Using pd.Timestamp for consistency

            if not tables:
                logger.warning("No tables found in the database.")
                f.write("-- No tables found.\n")
            else:
                for table_name, create_sql in tables:
                    f.write(f"-- Table: {table_name}\n")
                    f.write(f"{create_sql};\n\n")

        conn.close()
        logger.info(f"Database schema successfully exported to {SCHEMA_FILE}.")

    except Exception as e:
        logger.error(f"Error exporting database schema: {e}")

if __name__ == "__main__":
    # Ensure logging is set up when running directly
    from src.utils.logger import setup_logging
    setup_logging()
    # Need pandas for pd.Timestamp.now() - ensure it's installed
    import pandas as pd
    export_db_schema()
