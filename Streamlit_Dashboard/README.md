# Customer Churn Prediction - Streamlit Dashboard

This repository contains a **Streamlit dashboard** for predicting customer churn based on seven key features. The model is trained on telecom customer data and predicts whether a customer is likely to leave the company.

---

## 🎯 Objective

Build an interactive, elegant dashboard that:
- Accepts customer inputs
- Predicts churn probability
- Displays a result with clean visuals

---

## 🧰 Technologies Used

- Python
- Streamlit
- Pandas
- scikit-learn
- Joblib

---

## 📂 Folder Structure
```
customer-churn-dashboard/
├── dashboard_app.py # Main Streamlit app
├── model/
  ├── churn_pipeline.joblib # Final ML pipeline (model + preprocessing)
  ├── label_encoder.joblib # Label encoder for decoding predictions
└── requirements.txt # Required Python packages
├── README.md
```

## 📊 Input Features

The app expects the following 7 features:

1. `Gender` (Male/Female)
2. `Tenure Months`
3. `Monthly Charges`
4. `Total Charges`
5. `Contract`
6. `Internet Service`
7. `Payment Method`

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/customer-churn-dashboard.git
cd customer-churn-dashboard
```
### 2. Set up virtual environment
```
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```
### 3. Run the Streamlit App
```
streamlit run dashboard_app.py
```
Visit the local URL shown in the terminal.

📦 Deployment
You can deploy this app on:

Streamlit Cloud

Render

Hugging Face Spaces

🧠 Model Info
Trained on a cleaned and preprocessed telecom churn dataset.

Uses a full scikit-learn pipeline with:

ColumnTransformer (OneHotEncoder, Scaling)

RandomForestClassifier as the final model

Output decoded using a LabelEncoder

## Author
### Durdana Khalid
Connect with me on LinkedIn

📄 License
This project is licensed under the MIT License.
