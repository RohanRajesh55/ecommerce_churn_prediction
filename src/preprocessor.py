import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import joblib

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
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        logger.info("Preprocessing pipeline built successfully")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the preprocessing pipeline to the given DataFrame and return the transformed data.
        """
        df_clean = df.drop(columns=["CustomerID"])
        processed_data = self.pipeline.fit_transform(df_clean)
        feature_names = self.get_feature_names()
        return pd.DataFrame(processed_data, columns=feature_names)

    def get_feature_names(self) -> list:
        """
        Retrieve output feature names after preprocessing (numeric + one-hot encoded features).
        """
        # Get categorical feature names from OneHotEncoder
        cat_features = self.pipeline.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
            self.categorical_features
        )
        return self.numeric_features + list(cat_features)

    def save_pipeline_and_model(self, model, file_path: str):
        """
        Save the fitted pipeline and model as a joblib file.
        """
        joblib.dump({'preprocessor': self.pipeline, 'model': model}, file_path)
        logger.info(f"Saved pipeline and model to {file_path}")
