import os
import pandas as pd
from src.data_loader import load_dataset
from src.preprocessor import Preprocessor

# Add project root to Python path
project_root = r"C:\ecommerce_churn_prediction"
os.sys.path.append(project_root)

# Ensure processed folder exists
processed_folder_path = r"C:\ecommerce_churn_prediction\data\processed"
if not os.path.exists(processed_folder_path):
    os.makedirs(processed_folder_path)

# Load dataset
file_path = r"C:\ecommerce_churn_prediction\data\raw\E Commerce Dataset.xlsx"
df = load_dataset(file_path, sheet_name="E Comm")

# Define features
numeric_features = [
    "Tenure", "WarehouseToHome", "HourSpendOnApp", "NumberOfDeviceRegistered",
    "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount", "DaySinceLastOrder", "CashbackAmount"
]
categorical_features = [
    "PreferredLoginDevice", "CityTier", "PreferredPaymentMode", "Gender",
    "PreferedOrderCat", "MaritalStatus", "Complain"
]

# Initialize and build pipeline
preprocessor = Preprocessor()
preprocessor.build_pipeline(numeric_features, categorical_features)

# Preprocess data
preprocessed_df = preprocessor.preprocess_data(df)

# Save preprocessed data
output_file = os.path.join(processed_folder_path, "preprocessed_data.csv")
preprocessed_df.to_csv(output_file, index=False)
print(f"Preprocessed data saved to {output_file}")
