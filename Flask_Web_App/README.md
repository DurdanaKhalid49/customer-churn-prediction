# 🧠 Customer Churn Prediction - Flask Web App

This repository contains a **Flask web application** that predicts customer churn using a trained machine learning model. Users can input customer details through a stylish and elegant HTML form, and the app returns a prediction on whether the customer is likely to churn or not.

---

## 🚀 Demo

![Flask App Demo](demo_screenshot.png) <!-- (optional: add screenshot in repo) -->

---

## 🧰 Technologies Used

- Python
- Flask
- scikit-learn
- Pandas
- Joblib
- HTML5 & CSS3

---

## 📂 Folder Structure
```
Flask_Web_App/
├── static/
│ └── style.css # CSS styling for the frontend
├── templates/
│ └── index.html # Stylish HTML form layout
├── model/
│ └──  churn_pipeline.joblib # Final ML pipeline (model + preprocessing)
│ └── label_encoder.joblib # Label encoder for decoding predictions
└── app.py # Main Flask application
├── requirements.txt
├── README.md


```

## 📊 Input Features

The model expects the following **7 features**:

1. `Gender` (Male/Female)
2. `Tenure Months`
3. `Monthly Charges`
4. `Total Charges`
5. `Contract` (Month-to-month / One year / Two year)
6. `Internet Service` (DSL / Fiber optic / No)
7. `Payment Method` (Electronic check / Mailed check / Bank transfer / Credit card)

---

## ⚙️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/customer-churn-flask-app.git
cd customer-churn-flask-app
```
### 2. Set up virtual environment
```
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```
### 3. Run the Flask App
```
python app.py
Visit http://127.0.0.1:5000/ in your browser.
```
📦 Deployment
You can deploy this app on:

Render

Railway

Heroku (requires additional setup)

🧠 Model Training
The model is trained using:

Preprocessing pipeline (including OneHotEncoding, Scaling)

Final model: XGBoostClassifier

LabelEncoder to convert model outputs (0/1) back to No/Yes

## Author
### Durdana Khalid
Connect with me on LinkedIn

📄 License
This project is open-source and free to use under the MIT License.
