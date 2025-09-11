# src/utils/logger.py

import logging
from config.settings import settings

def setup_logging():
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            # logging.FileHandler("app.log") # Opcional: para salvar logs em arquivo
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()
