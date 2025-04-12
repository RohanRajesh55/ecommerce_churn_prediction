import joblib
import pandas as pd

def save_model(model_dict, file_path: str):
    joblib.dump(model_dict, file_path)

def load_model(file_path: str):
    return joblib.load(file_path)

def predict(model_dict, input_data: dict):
    preprocessor = model_dict['preprocessor']
    model = model_dict['model']
    input_df = pd.DataFrame([input_data])
    processed_data = preprocessor.transform(input_df)
    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)[:, 1]
    return predictions, probabilities
