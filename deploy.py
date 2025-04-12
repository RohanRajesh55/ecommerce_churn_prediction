import streamlit as st
from src.model_utils import load_model
import pandas as pd
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "pipeline_and_model.pkl")
model_dict = load_model(model_path)

# Create web interface
st.title("Customer Churn Predictor")

# Input form
with st.form("churn_form"):
    st.header("Customer Details")
    tenure = st.slider("Tenure (months)", 0, 120, 24)
    warehouse_to_home = st.slider("Warehouse to Home Distance", 0, 50, 5)
    # Add all other features similarly...
    
    submitted = st.form_submit_button("Predict")
    if submitted:
        input_data = {
            "Tenure": tenure,
            "WarehouseToHome": warehouse_to_home,
            # Add all other features...
        }
        prediction, proba = predict(model_dict, input_data)
        st.success(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
        st.metric("Churn Probability", f"{proba[0]*100:.2f}%")
