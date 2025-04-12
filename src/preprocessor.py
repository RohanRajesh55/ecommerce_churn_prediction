import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self):
        self.pipeline = None

    def build_pipeline(self, numeric_features: list, categorical_features: list) -> None:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Updated parameter
        ])
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        logger.info("Preprocessing pipeline built successfully.")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        preprocessed_data = self.pipeline.fit_transform(df)
        feature_names = self.get_feature_names(df)
        preprocessed_df = pd.DataFrame(preprocessed_data, columns=feature_names)
        logger.info("Data preprocessing completed successfully.")
        return preprocessed_df

    def get_feature_names(self, df: pd.DataFrame) -> list:
        passthrough_cols = [
            col for col in df.columns if col not in self.pipeline.transformers[0][2] + self.pipeline.transformers[1][2]
        ]
        num_features = self.pipeline.named_transformers_['num'].named_steps['scaler'].get_feature_names_out()
        cat_features = self.pipeline.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
        return list(num_features) + list(cat_features) + passthrough_cols
