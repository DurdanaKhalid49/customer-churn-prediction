import streamlit as st
import joblib
import pandas as pd

# Load model and label encoder
model = joblib.load("model/churn_pipeline.joblib")
label_encoder = joblib.load("model/label_encoder.joblib")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.markdown("""
    <style>
        h1 {
            color: #333333;
            text-align: center;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Customer Churn Prediction")

st.markdown("Fill in the customer information to get churn prediction.")

# ---------------------------
# USER INPUT FIELDS
# ---------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

# ---------------------------
# PREDICTION LOGIC
# ---------------------------
if st.button("Predict Churn"):
    try:
        # Convert gender to binary
        gender_binary = 1 if gender == "Male" else 0

        # Create input dataframe
        input_data = {
            "Gender": gender_binary,
            "Tenure Months": tenure,
            "Monthly Charges": monthly_charges,
            "Total Charges": total_charges,
            "Contract": contract,
            "Internet Service": internet_service,
            "Payment Method": payment_method
        }

        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction_encoded = model.predict(input_df)[0]
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]

        if prediction == "Yes":
            st.error("⚠️ This customer is likely to churn.")
        else:
            st.success("✅ This customer is likely to stay.")
    except Exception as e:
        st.error(f"Prediction Error: {e}")


st.markdown("---")
st.markdown("Made by Durdana Khalid")
