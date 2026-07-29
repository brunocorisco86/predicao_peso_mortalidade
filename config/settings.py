# config/settings.py

import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

class Settings:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATABASE_PATH = os.path.join(BASE_DIR, "database", "prediction_data.db")
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH}")
    EXTRACAO_MTECH_DIR = os.path.join(BASE_DIR, "data", "raw", "extracao_mtech")
    FEATURES_EXCEL_PATH = os.path.join(BASE_DIR, "data", "raw", "features", "BANCO_VARIAVEIS.xlsx")
    PESO_ABATE_EXCEL_PATH = os.path.join(BASE_DIR, "data", "raw", "peso_abate", "export_peso_abate_2023_2026.xlsx")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    UNIFIED_CSV_PATH = os.path.join(PROCESSED_DIR, "unified_data.csv")
    API_KEY = os.getenv("API_KEY", "your_default_api_key")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
