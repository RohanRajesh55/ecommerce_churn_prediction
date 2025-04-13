import joblib
import pandas as pd
import logging

# Set up module-level logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def save_model(model_dict: dict, file_path: str) -> None:
    """
    Save a model dictionary to a file using joblib.
    
    The model dictionary typically contains items like the preprocessor and the trained model.
    
    Parameters:
        model_dict (dict): Dictionary containing the model components.
        file_path (str): Destination file path (e.g., "models/best_model.pkl").
    
    Raises:
        Exception: If saving the model fails.
    """
    try:
        joblib.dump(model_dict, file_path)
        logger.info(f"Model saved successfully at {file_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise Exception(f"Error saving model: {e}")

def load_model(file_path: str):
    """
    Load the model dictionary from a file using joblib.
    
    Parameters:
        file_path (str): The path of the file containing the saved model.
        
    Returns:
        dict: The loaded model dictionary.
    
    Raises:
        Exception: If loading fails.
    """
    try:
        model_dict = joblib.load(file_path)
        logger.info(f"Model loaded successfully from {file_path}")
        return model_dict
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise Exception(f"Error loading model: {e}")

def predict(model_dict: dict, input_data: dict):
   
    try:
        preprocessor = model_dict['preprocessor']
        model = model_dict['model']
        input_df = pd.DataFrame([input_data])
        processed_data = preprocessor.transform(input_df)
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)[:, 1]
        logger.info("Prediction completed successfully.")
        return predictions, probabilities
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise Exception(f"Error during prediction: {e}")