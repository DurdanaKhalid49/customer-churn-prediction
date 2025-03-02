from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load model and preprocessing tools
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect user inputs
        tenure = float(request.form["tenure"])
        monthly_charges = float(request.form["monthly_charges"])
        total_charges = float(request.form["total_charges"])
        payment_method = request.form["payment_method"]
        internet_service = request.form["internet_service"]
        contract_type = request.form["contract_type"]

        # Convert inputs to DataFrame
        input_data = pd.DataFrame([{
            "Payment Method": payment_method,
            "Internet Service": internet_service,
            "Contract": contract_type,
            "Tenure Months": tenure,
            "Monthly Charges": monthly_charges,
            "Total Charges": total_charges
        }])

        print("🔍 DEBUG: Input DataFrame before encoding")
        print(input_data)

        # One-hot encode categorical features
        input_encoded = pd.get_dummies(input_data, columns=["Payment Method", "Internet Service", "Contract"])

        print("🔍 DEBUG: Encoded Input DataFrame")
        print(input_encoded.columns)

        # Align with model features
        final_input = pd.DataFrame(0, index=[0], columns=feature_names)
        final_input.update(input_encoded)
        final_input.rename(columns={"tenure": "Tenure Months",
                                    "monthly_charges": "Monthly Charges",
                                    "total_charges": "Total Charges"}, inplace=True)

        # Scale numerical features
        final_input[["Tenure Months", "Monthly Charges", "Total Charges"]] = scaler.transform(
            final_input[["Tenure Months", "Monthly Charges", "Total Charges"]])

        # Make prediction
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][1] * 100

        # Convert prediction to text
        result_text = "Customer is likely to churn." if prediction == 1 else "Customer is not likely to churn."

        return render_template("result.html", prediction=result_text, probability=round(probability, 2))

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
