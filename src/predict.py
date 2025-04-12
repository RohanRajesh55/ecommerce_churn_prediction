import sys
import os
import pandas as pd
from src.model_utils import load_model, predict

# Fix module import issues
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load model
model_path = os.path.join(project_root, "models", "pipeline_and_model.pkl")
model_dict = load_model(model_path)

# Input data with ALL REQUIRED FEATURES (adjust values as needed)
input_data = {
    "Tenure": 24,
    "WarehouseToHome": 5,
    "HourSpendOnApp": 3,
    "NumberOfDeviceRegistered": 2,
    "OrderAmountHikeFromlastYear": 10,
    "CouponUsed": 5,
    "OrderCount": 12,
    "DaySinceLastOrder": 30,
    "CashbackAmount": 200,
    "PreferredLoginDevice": "Mobile",
    "CityTier": "Tier 1",
    "PreferredPaymentMode": "Credit Card",
    "Gender": "Male",
    "PreferedOrderCat": "Laptop & Accessory",
    "MaritalStatus": "Single",
    "Complain": 0,
    "SatisfactionScore": 4,  # Added
    "NumberOfAddress": 2      # Added
}

# Make prediction
predictions, probabilities = predict(model_dict, input_data)
print("Predictions:", predictions)
print("Probabilities:", probabilities)
