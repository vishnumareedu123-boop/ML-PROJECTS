import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/house_price_model.pkl")

# Sample input (same order as training features)
sample = pd.DataFrame({
    "crim": [0.00632],
    "zn": [18.0],
    "indus": [2.31],
    "chas": [0],
    "nox": [0.538],
    "rm": [6.575],
    "age": [65.2],
    "dis": [4.0900],
    "rad": [1],
    "tax": [296],
    "ptratio": [15.3],
    "b": [396.90],
    "lstat": [4.98]
})

# Predict
prediction = model.predict(sample)

print("Predicted House Price:", prediction[0])