from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

# Load the trained model and feature names
model_data = joblib.load("customer_churn_model.pkl")
model = model_data["model"]
feature_names = joblib.load("feature_names.pkl")  # Load saved feature names

# Check if feature_names is empty
if not feature_names:
    raise ValueError("Feature names are not loaded correctly.")

# Initialize Flask app
app = Flask(__name__)

def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    if feature_names:
        df = df.reindex(columns=feature_names, fill_value=0)
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()  # Collect user input as a dictionary
        processed_input = preprocess_input(input_data)
        # Ensure processed input matches feature names
        processed_input = processed_input.reindex(columns=feature_names, fill_value=0)

        # Check the shape before prediction
        print(f"Processed Input Shape: {processed_input.shape}")  # Debugging

        prediction = model.predict(processed_input)[0]  # Make prediction
        probability = model.predict_proba(processed_input)[0][1]

        return jsonify({"prediction": int(prediction), "probability": float(probability)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
