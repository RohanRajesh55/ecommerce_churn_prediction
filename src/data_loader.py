import pandas as pd
import os
import logging

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(file_path: str, sheet_name: str = "E Comm", verbose: bool = True) -> pd.DataFrame:
    """
    Load an Excel dataset from the given file path and sheet name.

    Parameters:
        file_path (str): The path to the Excel file.
        sheet_name (str): The name of the sheet to load (default "E Comm").
        verbose (bool): If True, logs detailed information (default True).

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
        Exception: For any generic exception during loading.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")

        if not file_path.endswith(('.xlsx', '.xls')):
            raise ValueError(f"Invalid file format: {file_path}. Expected an Excel file (.xlsx or .xls).")

        abs_path = os.path.abspath(file_path)
        df = pd.read_excel(abs_path, sheet_name=sheet_name)

        if verbose:
            logger.info(f"Dataset loaded successfully from {abs_path}")
            logger.info(f"Dataset shape: {df.shape}")

        return df

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise