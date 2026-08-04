import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import preprocess_data

# Load dataset
df = preprocess_data("data/house_price.csv")

# Features and Target
X = df.drop("medv", axis=1)
y = df["medv"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Mean Absolute Error :", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error  :", mean_squared_error(y_test, y_pred))
print("R2 Score            :", r2_score(y_test, y_pred))

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save Model
joblib.dump(model, "models/house_price_model.pkl")

print("Model Saved Successfully!")