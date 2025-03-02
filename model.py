import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Load dataset
df = pd.read_excel("D:/Self_made Projects/Customer Churn Prediction/data/Telco_customer_churn.xlsx")
df.drop(columns=['CustomerID',"Count","Zip Code","Latitude","Longitude","Country",'City',"State","Lat Long","Churn Value",'Churn Reason',"Churn Score"], inplace=True)
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
df['Senior Citizen'] = df['Senior Citizen'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Partner'] = df['Partner'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Dependents'] = df['Dependents'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Phone Service'] = df['Phone Service'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Paperless Billing'] = df['Paperless Billing'].apply(lambda x: 1 if x == 'Yes' else 0)
df = pd.get_dummies(df, columns=["Multiple Lines","Online Security","Online Backup","Device Protection","Tech Support","Streaming TV","Streaming Movies" ], prefix='Category', drop_first=True)


# Define features and target
X = df.drop(columns=["Churn Label"])  # Features
y = df["Churn Label"]  # Target variable

# Convert categorical target to numeric
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Converts 'Yes/No' to 1/0

# Save label encoder for later use
joblib.dump(label_encoder, "label_encoder.pkl")

# Identify categorical and numerical features
categorical_features = ["Payment Method", "Internet Service", "Contract"]
numerical_features = ["Tenure Months", "Monthly Charges", "Total Charges"]


# Encode categorical features
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
# Convert non-numeric values to NaN
X_encoded[numerical_features] = X_encoded[numerical_features].apply(pd.to_numeric, errors='coerce')

# Fill NaN values with median (or another strategy)
X_encoded[numerical_features] = X_encoded[numerical_features].fillna(X_encoded[numerical_features].median())

# Apply scaling
X_encoded[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])


# Save feature names (important for Flask)
joblib.dump(X_encoded.columns.tolist(), "feature_names.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")  # Save scaler for Flask app

print("✅ Model trained and saved successfully!")
