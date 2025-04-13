import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import joblib

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Preprocessor:
    """
    A preprocessing pipeline for numeric and categorical features using scikit-learn.
    Handles missing values, scaling, encoding, and feature extraction.
    """
    def __init__(self):
        self.pipeline = None
        self.numeric_features = []
        self.categorical_features = []

    def build_pipeline(self, numeric_features: list, categorical_features: list):
        """
        Build the preprocessing pipeline using ColumnTransformer.
        This handles imputation, scaling for numeric features, and one-hot encoding for categorical features.
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        # Numeric transformer: Impute missing values with median and scale the data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical transformer: Impute missing values with the most frequent value and apply one-hot encoding
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Dense output for easier processing
        ])

        # Combining both transformers in a ColumnTransformer
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # Drop any other columns not specified
        )
        logger.info("Preprocessing pipeline built successfully")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the preprocessing pipeline to the given DataFrame and return the transformed data.
        The dataframe is expected to include columns that match the specified numeric and categorical features.
        """
        # Drop the 'CustomerID' column or any other column that should not be part of the model features
        df_clean = df.drop(columns=["CustomerID"], errors='ignore')  # Use errors='ignore' to avoid crashes if column is absent
        
        # Apply the transformations (fit + transform)
        processed_data = self.pipeline.fit_transform(df_clean)
        
        # Get the feature names after one-hot encoding and scaling
        feature_names = self.get_feature_names()
        
        # Convert the transformed data back into a DataFrame with appropriate column names
        return pd.DataFrame(processed_data, columns=feature_names)

    def get_feature_names(self) -> list:
        """
        Retrieve output feature names after preprocessing (numeric + one-hot encoded features).
        """
        # Get feature names for the categorical features after one-hot encoding
        cat_features = self.pipeline.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_features)
        
        # Combine the numeric and categorical feature names
        return self.numeric_features + list(cat_features)

    def save_pipeline_and_model(self, model, file_path: str):
        """
        Save the fitted preprocessing pipeline and model as a joblib file.
        This will allow for re-use of the model and pipeline in deployment.
        """
        joblib.dump({'preprocessor': self.pipeline, 'model': model}, file_path)
        logger.info(f"Saved pipeline and model to {file_path}")

    def load_pipeline_and_model(self, file_path: str):
        """
        Load the preprocessing pipeline and model from a saved joblib file.
        This is useful for deploying the model or making predictions.
        """
        data = joblib.load(file_path)
        self.pipeline = data['preprocessor']
        model = data['model']
        logger.info(f"Loaded pipeline and model from {file_path}")
        return model
