Customer Churn Prediction 🚀

📌 Project Overview

Customer churn is a major challenge for businesses, especially in industries like telecom, banking, and SaaS. This project aims to predict whether a customer will churn using machine learning models like Random Forest and XGBoost. By identifying high-risk customers, businesses can take proactive measures to retain them.

🔹 Dataset Information

The dataset includes customer details such as tenure, contract type, monthly charges, and payment methods.

The target variable is Churn (Yes/No).

🔹 Approach

1️⃣ Data Preprocessing

Handled missing values.

Encoded categorical features using Label Encoding & One-Hot Encoding.

Scaled numerical features.

2️⃣ Exploratory Data Analysis (EDA)

Visualized churn distribution.

Analyzed correlations between features and churn.

Identified key factors affecting customer retention.

3️⃣ Feature Engineering

Created new meaningful features to improve model accuracy.

Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

4️⃣ Model Training & Evaluation

Used Random Forest, XGBoost, and Logistic Regression models.

Evaluated performance using Accuracy, Precision, Recall, and F1-Score.

Tuned hyperparameters for optimal performance.

5️⃣ Results & Insights

Achieved an accuracy of 87% using the XGBoost model.

Found that contract type, monthly charges, and tenure were the most important factors influencing churn.

🔹 Technologies Used

Python, pandas, NumPy, Matplotlib, Seaborn

Scikit-learn, XGBoost, SMOTE

Jupyter Notebook

🔹 How to Use

1️⃣ Clone this repository:

git clone https://github.com/yourusername/Customer_Churn_Prediction.git  

2️⃣ Install dependencies:

pip install -r requirements.txt  

3️⃣ Open and run notebook.ipynb in Jupyter Notebook.

🔹 Key Learnings

✔ The impact of contract type on churn.✔ How SMOTE helps handle imbalanced datasets.✔ The power of XGBoost for classification problems.

📊 Next Steps: Deploying this model as a web app using Flask or Streamlit!

💡 Contributions & Feedback: If you have suggestions, feel free to open an issue or fork the repository! 🎯
