import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocess import preprocess_data

# Load and preprocess data
df = preprocess_data("data/loan.csv")

# Drop Loan_ID if present
if "Loan_ID" in df.columns:
    df = df.drop("Loan_ID", axis=1)

# Features and Target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save Model
joblib.dump(model, "models/loan_model.pkl")

print("Model Saved Successfully!")