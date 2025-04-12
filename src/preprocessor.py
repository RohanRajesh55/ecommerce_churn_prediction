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
    def __init__(self):
        self.pipeline = None

    def build_pipeline(self, numeric_features: list, categorical_features: list):
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
            remainder='drop'  # Ignore columns not in numeric/categorical features
        )
        logger.info("Preprocessing pipeline built successfully")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.drop(columns=["CustomerID"])  # Remove identifier column
        processed_data = self.pipeline.fit_transform(df_clean)
        feature_names = self.get_feature_names(df_clean)
        return pd.DataFrame(processed_data, columns=feature_names)

    def get_feature_names(self, df: pd.DataFrame) -> list:
        num_features = self.pipeline.named_transformers_['num'].get_feature_names_out()
        cat_features = self.pipeline.named_transformers_['cat'].get_feature_names_out()
        return list(num_features) + list(cat_features)

    def save_pipeline_and_model(self, model, file_path: str):
        joblib.dump({'preprocessor': self.pipeline, 'model': model}, file_path)
        logger.info(f"Saved pipeline and model to {file_path}")
