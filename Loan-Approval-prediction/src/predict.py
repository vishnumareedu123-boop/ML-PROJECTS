import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/loan_model.pkl")

# Sample input (change values as needed)
sample = pd.DataFrame({
    "Customer_ID": [1001],
    "Gender": [1],
    "Married": [1],
    "Dependents": [0],
    "Education": [1],
    "Self_Employed": [0],
    "Applicant_Income": [5000],
    "Coapplicant_Income": [2000],
    "Loan_Amount": [150],
    "Loan_Amount_Term": [360],
    "Credit_History": [1],
    "Property_Area": [2]
})

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Not Approved")