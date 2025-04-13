# deploy.py
import streamlit as st
import pandas as pd
import joblib
import os

# Function to load the saved model
def load_saved_model(file_path):
    if not os.path.exists(file_path):
        st.error(f"Model file not found at {file_path}")
        return None
    return joblib.load(file_path)

# Path to the model
model_path = r"C:\ecommerce_churn_prediction\models\best_model.pkl"

# Load the model
model = load_saved_model(model_path)

# Check if the model is loaded successfully
if model is None:
    st.error("Failed to load the model. Please check the server logs.")
else:
    # Streamlit App Title
    st.title("E-commerce Customer Churn Prediction")
    st.header("Input Customer Details")

    # Create a form to take input from the user
    with st.form("churn_form"):
        # Add all necessary features here
        tenure = st.slider("Tenure (months)", 0, 120, 24)
        warehouse_to_home = st.slider("Warehouse to Home Distance", 0, 50, 5)
        satisfaction_score = st.slider("Satisfaction Score", 0, 10, 5)
        product_cost = st.slider("Product Cost", 0, 500, 250)
        discount_offered = st.slider("Discount Offered", 0, 100, 10)
        delivery_time = st.slider("Delivery Time (days)", 1, 30, 7)

        # Additional features based on the model's requirements
        order_amount_hike = st.slider("Order Amount Hike From Last Year", 0, 1000, 100)
        hour_spent_on_app = st.slider("Hours Spent on App", 0, 100, 5)
        number_of_device_registered = st.slider("Number of Devices Registered", 0, 10, 1)
        # Add more features based on model's requirement

        submit = st.form_submit_button("Predict")

    # When the submit button is pressed
    if submit:
        # Prepare input data
        input_data = pd.DataFrame([[tenure, warehouse_to_home, satisfaction_score, product_cost, discount_offered, delivery_time,
                                    order_amount_hike, hour_spent_on_app, number_of_device_registered]],
                                  columns=["Tenure", "WarehouseToHome", "SatisfactionScore", "ProductCost", "DiscountOffered", 
                                           "DeliveryTime", "OrderAmountHikeFromlastYear", "HourSpendOnApp", "NumberOfDeviceRegistered"])
        
        # Make predictions
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1]
        
        # Display results
        st.success(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
        st.metric("Churn Probability", f"{probability[0] * 100:.2f}%")
