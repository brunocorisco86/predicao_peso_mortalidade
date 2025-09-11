# config/settings.py

import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/database.db")
    API_KEY = os.getenv("API_KEY", "your_default_api_key")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
