from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and label encoder
model = joblib.load("model/churn_pipeline.joblib")
label_encoder = joblib.load("model/label_encoder.joblib")

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
    app.run(debug=True)
