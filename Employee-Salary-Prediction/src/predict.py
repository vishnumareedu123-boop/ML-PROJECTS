import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/employee_salary_model.pkl")

# Sample employee data
sample = pd.DataFrame({
    "Employee_ID": [1001],
    "Age": [30],
    "Gender": [1],          # Male=1, Female=0
    "Education": [0],       # Bachelor=0, Master=1, PhD=2 (depends on LabelEncoder)
    "Experience": [5],
    "Job_Role": [2],        # Encoded value
    "City": [1]             # Encoded value
})

# Predict salary
prediction = model.predict(sample)

print("=" * 40)
print("Predicted Employee Salary")
print("=" * 40)
print(f"Predicted Salary: ₹{prediction[0]:,.2f}")