import streamlit as st
import pandas as pd
import joblib

# Load Model and Feature Names
@st.cache_resource  # Cache to improve performance
def load_model():
    try:
        model = joblib.load("churn_model.pkl")  # Load trained model
        feature_names = joblib.load("feature_names.pkl")  # Load feature names
        return model, feature_names
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

# Preprocessing User Input
def preprocess_input(input_data, feature_names):
    """Convert user input into the format expected by the model."""
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)
    
    # Ensure all columns match the model's feature set
    final_input = pd.DataFrame(0, index=[0], columns=feature_names)
    final_input.update(df)
    
    return final_input

# Predict Churn
def predict_churn(input_data):
    """Make a churn prediction based on user input."""
    model, feature_names = load_model()
    if model is None:
        return None, None
    
    features = preprocess_input(input_data, feature_names)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[:, 1][0]
    
    return prediction, probability

# Streamlit UI
st.title("Customer Churn Prediction App")
st.write("Fill in the details below to predict whether a customer will churn.")

# User Inputs
input_data = {
    "Tenure Months": st.number_input("Tenure (Months)", min_value=0, value=12),
    "Monthly Charges": st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0),
    "Total Charges": st.number_input("Total Charges ($)", min_value=0.0, value=600.0),
    "Contract": st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"]),
    "Internet Service": st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
    "Payment Method": st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
}

if st.button("Predict Churn"):
    prediction, probability = predict_churn(input_data)
    if prediction is not None:
        st.write("## Prediction Result")
        st.write("**Churn:** Yes" if prediction == 1 else "**Churn:** No")
        st.write(f"**Churn Probability:** {probability:.2%}")
    else:
        st.error("Failed to generate prediction. Check the model files.")
