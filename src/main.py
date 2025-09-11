# src/main.py

from src.utils.logger import logger
from config.settings import settings

def run_analysis():
    logger.info("Starting data analysis...")
    logger.debug(f"Database URL: {settings.DATABASE_URL}")
    logger.info("Data analysis completed successfully.")

if __name__ == "__main__":
    run_analysis()
