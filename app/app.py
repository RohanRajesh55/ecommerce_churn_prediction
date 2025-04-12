import streamlit as st
from src.model_utils import load_model, predict

# Load the saved model
model_path = r"C:\ecommerce_churn_prediction\models\best_model.pkl"
model = load_model(model_path)

# Streamlit app UI
st.title("Customer Churn Prediction")
st.header("Enter Customer Details")

# Collect user input
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=120, value=24)
warehouse_to_home = st.number_input("Distance from Warehouse to Home", min_value=0, max_value=50, value=5)
hour_spend_on_app = st.number_input("Hours Spent on App", min_value=0, max_value=24, value=3)
number_of_device_registered = st.number_input("Number of Devices Registered", min_value=0, max_value=10, value=2)
order_amount_hike_from_last_year = st.number_input("Order Amount Hike from Last Year", min_value=-100, max_value=100, value=10)
coupon_used = st.number_input("Coupons Used", min_value=0, max_value=50, value=5)
order_count = st.number_input("Order Count", min_value=0, max_value=100, value=12)
day_since_last_order = st.number_input("Days Since Last Order", min_value=0, max_value=365, value=30)
cashback_amount = st.number_input("Cashback Amount", min_value=0, max_value=1000, value=200)

# One-hot encoded categorical features (example for PreferredLoginDevice and CityTier)
preferred_login_device_desktop = st.radio("Preferred Login Device", ["Desktop", "Mobile"]) == "Desktop"
city_tier_1 = st.radio("City Tier", ["Tier 1", "Tier 2"]) == "Tier 1"

# Prepare input data for prediction
input_data = {
    "Tenure": tenure,
    "WarehouseToHome": warehouse_to_home,
    "HourSpendOnApp": hour_spend_on_app,
    "NumberOfDeviceRegistered": number_of_device_registered,
    "OrderAmountHikeFromlastYear": order_amount_hike_from_last_year,
    "CouponUsed": coupon_used,
    "OrderCount": order_count,
    "DaySinceLastOrder": day_since_last_order,
    "CashbackAmount": cashback_amount,
    "PreferredLoginDevice_Desktop": int(preferred_login_device_desktop),
    "PreferredLoginDevice_Mobile": int(not preferred_login_device_desktop),
    "CityTier_1": int(city_tier_1),
    "CityTier_2": int(not city_tier_1),
}

# Make predictions
if st.button("Predict"):
    predictions, probabilities = predict(model, input_data)
    
    # Display results
    st.subheader("Prediction Results")
    st.write(f"Churn Prediction: {'Yes' if predictions[0] == 1 else 'No'}")
    st.write(f"Churn Probability: {probabilities[0]:.2f}")
