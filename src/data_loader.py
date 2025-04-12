import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(file_path: str, sheet_name: str = "E Comm") -> pd.DataFrame:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        logger.info(f"Dataset loaded successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
