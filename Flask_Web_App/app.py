from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
app = Flask(__name__)

MODEL_PATH = os.path.join("model", "churn_pipeline.joblib")
ENCODER_PATH = os.path.join("model", "label_encoder.joblib")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            gender = request.form.get("gender")
            tenure = float(request.form.get("tenure"))
            monthly_charges = float(request.form.get("monthly_charges"))
            total_charges = float(request.form.get("total_charges"))
            contract = request.form.get("contract")
            internet_service = request.form.get("internet_service")
            payment_method = request.form.get("payment_method")

            gender_binary = 1 if gender == "Male" else 0

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
            pred_encoded = model.predict(input_df)[0]
            prediction = label_encoder.inverse_transform([pred_encoded])[0]
        except Exception as e:
            prediction = f"Error occurred: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # use Railway port or default to 8080
    app.run(host="0.0.0.0", port=port)  
