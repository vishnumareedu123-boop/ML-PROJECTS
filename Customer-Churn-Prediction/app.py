from flask import Flask, request, render_template_string
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("models/customer_churn_model.pkl")

html = """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Churn Prediction</title>
</head>
<body>

<h2>Customer Churn Prediction</h2>

<form method="POST">

Customer ID:<br>
<input type="number" name="Customer_ID" required><br><br>

Age:<br>
<input type="number" name="Age" required><br><br>

Gender (Male=1, Female=0):<br>
<input type="number" name="Gender" required><br><br>

Tenure (Years):<br>
<input type="number" name="Tenure" required><br><br>

Monthly Charges:<br>
<input type="number" name="Monthly_Charges" required><br><br>

Total Charges:<br>
<input type="number" name="Total_Charges" required><br><br>

Contract (Month-to-Month=0, One Year=1, Two Year=2):<br>
<input type="number" name="Contract" required><br><br>

Internet Service (DSL=0, Fiber=1, None=2):<br>
<input type="number" name="Internet_Service" required><br><br>

Payment Method (Cash=0, Credit Card=1, Debit Card=2, UPI=3):<br>
<input type="number" name="Payment_Method" required><br><br>

<input type="submit" value="Predict">

</form>

{% if prediction %}
<h2>{{ prediction }}</h2>
{% endif %}

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        data = pd.DataFrame([{
            "Customer_ID": int(request.form["Customer_ID"]),
            "Age": int(request.form["Age"]),
            "Gender": int(request.form["Gender"]),
            "Tenure": int(request.form["Tenure"]),
            "Monthly_Charges": float(request.form["Monthly_Charges"]),
            "Total_Charges": float(request.form["Total_Charges"]),
            "Contract": int(request.form["Contract"]),
            "Internet_Service": int(request.form["Internet_Service"]),
            "Payment_Method": int(request.form["Payment_Method"])
        }])

        result = model.predict(data)[0]

        if result == 1:
            prediction = "Customer Will Churn"
        else:
            prediction = "Customer Will Not Churn"

    return render_template_string(html, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)