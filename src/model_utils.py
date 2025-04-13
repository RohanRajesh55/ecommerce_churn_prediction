import joblib
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def save_model(model_dict, file_path: str):
    joblib.dump(model_dict, file_path)
    logger.info(f"Model saved to {file_path}")

def load_model(file_path: str):
    logger.info(f"Loading model from {file_path}")
    return joblib.load(file_path)

def predict(model_dict, input_data: dict):
    try:
        preprocessor = model_dict['preprocessor']
        model = model_dict['model']
        input_df = pd.DataFrame([input_data])
        
        logger.info("Transforming input data")
        processed_data = preprocessor.transform(input_df)
        
        logger.info("Making predictions")
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)[:, 1]
        
        return predictions.tolist(), probabilities.tolist()
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise
