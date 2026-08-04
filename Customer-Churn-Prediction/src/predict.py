import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/customer_churn_model.pkl")

# Sample customer data
sample = pd.DataFrame({
    "Customer_ID": [1001],
    "Age": [35],
    "Gender": [1],                # Male=1, Female=0
    "Tenure": [5],
    "Monthly_Charges": [2500],
    "Total_Charges": [125000],
    "Contract": [0],              # Encoded value
    "Internet_Service": [1],      # Encoded value
    "Payment_Method": [2]         # Encoded value
})

# Predict
prediction = model.predict(sample)

print("=" * 40)
print("Customer Churn Prediction")
print("=" * 40)

if prediction[0] == 1:
    print("Prediction : Customer Will Churn")
else:
    print("Prediction : Customer Will Not Churn")