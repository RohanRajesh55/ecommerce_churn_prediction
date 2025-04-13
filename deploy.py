import streamlit as st
import pandas as pd
import joblib
import os
from typing import Optional, Any

def load_saved_model(file_path: str) -> Optional[Any]:
    """
    Load the saved model pipeline from the specified file path using joblib.
    """
    if not os.path.exists(file_path):
        st.error(f"Model file not found at {file_path}")
        return None
    try:
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Use the file that contains the full pipeline (preprocessor + model)
model_path = r"C:\ecommerce_churn_prediction\models\full_pipeline_model.pkl"

# Load the full pipeline.
model_pipeline = load_saved_model(model_path)

if model_pipeline is None:
    st.error("Failed to load the full pipeline model. Please check the logs.")
    st.stop()
else:
    # Expecting the saved model to be a dictionary with keys "model" and "preprocessor"
    if isinstance(model_pipeline, dict):
        model = model_pipeline.get("model")
        preprocessor = model_pipeline.get("preprocessor")
    else:
        model = model_pipeline
        preprocessor = None
        st.error("Preprocessor not found in the model pipeline. Please ensure your training stage saved both preprocessor and model.")
        st.stop()

st.title("E-commerce Customer Churn Prediction")
st.header("Input Customer Details")

# Build the input form that exactly matches the features used during training.
with st.form("churn_form"):
    st.subheader("Numeric Features")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=24)
    city_tier = st.number_input("City Tier", min_value=1, max_value=3, value=1)
    warehouse_to_home = st.number_input("Warehouse To Home Distance", min_value=0, max_value=50, value=5)
    hour_spent_on_app = st.number_input("Hours Spent On App", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    num_devices = st.number_input("Number Of Devices Registered", min_value=0, max_value=10, value=1)
    satisfaction_score = st.number_input("Satisfaction Score", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
    num_addresses = st.number_input("Number Of Addresses", min_value=0, max_value=10, value=1)
    complain = st.number_input("Number of Complaints", min_value=0, max_value=10, value=0)
    order_amount_hike = st.number_input("Order Amount Hike From Last Year", min_value=0.0, max_value=1000.0, value=100.0, step=10.0)
    coupon_used = st.number_input("Coupon Used (times)", min_value=0, max_value=20, value=0)
    order_count = st.number_input("Order Count", min_value=0, max_value=100, value=10)
    day_since_last_order = st.number_input("Days Since Last Order", min_value=0, max_value=365, value=30)
    cashback_amount = st.number_input("Cashback Amount", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)
    
    st.subheader("Categorical Features")
    preferred_login_device = st.selectbox("Preferred Login Device", options=["Computer", "Mobile Phone", "Phone"])
    preferred_payment_mode = st.selectbox("Preferred Payment Mode", options=["CC", "COD", "Credit Card", "Debit Card", "E wallet", "UPI"])
    gender = st.selectbox("Gender", options=["Male", "Female"])
    prefered_order_cat = st.selectbox("Preferred Order Category", options=["Fashion", "Grocery", "Laptop & Accessory", "Mobile", "Mobile Phone", "Others"])
    marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
    
    submit = st.form_submit_button("Predict")

if submit:
    # Assemble the inputs into a DataFrame with column names matching the training schema.
    input_data = pd.DataFrame([{
        "Tenure": tenure,
        "CityTier": city_tier,
        "WarehouseToHome": warehouse_to_home,
        "HourSpendOnApp": hour_spent_on_app,
        "NumberOfDeviceRegistered": num_devices,
        "SatisfactionScore": satisfaction_score,
        "NumberOfAddress": num_addresses,
        "Complain": complain,
        "OrderAmountHikeFromlastYear": order_amount_hike,
        "CouponUsed": coupon_used,
        "OrderCount": order_count,
        "DaySinceLastOrder": day_since_last_order,
        "CashbackAmount": cashback_amount,
        "PreferredLoginDevice": preferred_login_device,
        "PreferredPaymentMode": preferred_payment_mode,
        "Gender": gender,
        "PreferedOrderCat": prefered_order_cat,
        "MaritalStatus": marital_status
    }])
    
    st.subheader("Input Data")
    st.write(input_data)
    
    try:
        # Apply the preprocessor transform.
        processed_data = preprocessor.transform(input_data)
        
        # Make predictions using the trained model.
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[:, 1]
        
        result_text = "Churn" if prediction[0] == 1 else "No Churn"
        st.success(f"Prediction: {result_text}")
        st.metric("Churn Probability", f"{probability[0] * 100:.2f}%")
    except Exception as e:
        st.error(f"Error during prediction: {e}")