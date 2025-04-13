import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import yaml
import joblib

# Import custom modules from your src package
from src.data_loader import load_dataset
from src.preprocessor import Preprocessor
from src.model import ChurnClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """
    Load configuration settings from a YAML file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    # Load configuration from the config file.
    config = load_config("config.yml")
    
    # Load the dataset using the specified file path and sheet name.
    df = load_dataset(
        file_path=config["paths"]["raw_data"],
        sheet_name=config["data"]["sheet_name"],
        verbose=True
    )
    
    target = config["data"]["target"]
    # Exclude the target column from the list of features.
    features = [col for col in df.columns if col != target]
    X = df[features]
    y = df[target]
    
    # Split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=42,
        stratify=y
    )
    logger.info("Data split into training and testing sets.")
    
    # Build the preprocessing pipeline.
    preprocessor = Preprocessor()
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Remove 'CustomerID' if present, as it is an identifier.
    if "CustomerID" in numeric_features:
        numeric_features.remove("CustomerID")
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    preprocessor.build_pipeline(numeric_features, categorical_features)
    
    # Transform the training and test data.
    X_train_processed = preprocessor.preprocess_data(X_train)
    X_test_processed = preprocessor.preprocess_data(X_test)
    logger.info("Data preprocessing complete.")
    
    # Initialize and train the churn classifier.
    classifier = ChurnClassifier(config)
    model = classifier.train(X_train_processed, y_train, X_test_processed, y_test)
    
    # Save the combined pipeline (preprocessor and model)
    preprocessor.save_pipeline_and_model(model, config["paths"]["model_output_full"])
    
    # Evaluate the trained model.
    metrics = classifier.evaluate(X_test_processed, y_test)
    logger.info("Evaluation metrics:")
    for metric, score in metrics.items():
        if metric not in ['confusion_matrix', 'roc_curve', 'pr_curve']:
            logger.info(f"{metric}: {score:.4f}")

if __name__ == '__main__':
    main()