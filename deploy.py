import streamlit as st
import pandas as pd
import joblib
import os

def load_saved_model(file_path):
    return joblib.load(file_path)

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
model = load_saved_model(model_path)

st.title("E-commerce Customer Churn Prediction")
st.header("Input Customer Details")

with st.form("churn_form"):
    tenure = st.slider("Tenure (months)", 0, 120, 24)
    warehouse_to_home = st.slider("Warehouse to Home Distance", 0, 50, 5)
    satisfaction_score = st.slider("Satisfaction Score", 0, 10, 5)
    product_cost = st.slider("Product Cost", 0, 500, 250)
    discount_offered = st.slider("Discount Offered", 0, 100, 10)
    delivery_time = st.slider("Delivery Time (days)", 1, 30, 7)
    submit = st.form_submit_button("Predict")

if submit:
    input_data = pd.DataFrame([[tenure, warehouse_to_home, satisfaction_score, product_cost, discount_offered, delivery_time]],
                              columns=["Tenure", "WarehouseToHome", "SatisfactionScore", "ProductCost", "DiscountOffered", "DeliveryTime"])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]
    st.success(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
    st.metric("Churn Probability", f"{probability[0] * 100:.2f}%")