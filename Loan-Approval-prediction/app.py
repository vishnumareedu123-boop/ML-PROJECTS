from flask import Flask, request, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("models/loan_model.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Loan Approval Prediction</title>
</head>
<body>

<h2>Loan Approval Prediction</h2>

<form method="POST">

Customer ID:<br>
<input type="number" name="Customer_ID" required><br><br>

Gender (Male=1, Female=0):<br>
<input type="number" name="Gender" required><br><br>

Married (Yes=1, No=0):<br>
<input type="number" name="Married" required><br><br>

Dependents:<br>
<input type="number" name="Dependents" required><br><br>

Education (Graduate=1, Not Graduate=0):<br>
<input type="number" name="Education" required><br><br>

Self Employed (Yes=1, No=0):<br>
<input type="number" name="Self_Employed" required><br><br>

Applicant Income:<br>
<input type="number" name="Applicant_Income" required><br><br>

Coapplicant Income:<br>
<input type="number" step="0.01" name="Coapplicant_Income" required><br><br>

Loan Amount:<br>
<input type="number" name="Loan_Amount" required><br><br>

Loan Amount Term:<br>
<input type="number" name="Loan_Amount_Term" required><br><br>

Credit History (1 or 0):<br>
<input type="number" name="Credit_History" required><br><br>

Property Area (Urban=2, Semiurban=1, Rural=0):<br>
<input type="number" name="Property_Area" required><br><br>

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
            "Gender": int(request.form["Gender"]),
            "Married": int(request.form["Married"]),
            "Dependents": int(request.form["Dependents"]),
            "Education": int(request.form["Education"]),
            "Self_Employed": int(request.form["Self_Employed"]),
            "Applicant_Income": int(request.form["Applicant_Income"]),
            "Coapplicant_Income": float(request.form["Coapplicant_Income"]),
            "Loan_Amount": int(request.form["Loan_Amount"]),
            "Loan_Amount_Term": int(request.form["Loan_Amount_Term"]),
            "Credit_History": int(request.form["Credit_History"]),
            "Property_Area": int(request.form["Property_Area"])
        }])

        result = model.predict(data)[0]

        if result == 1:
            prediction = "✅ Loan Approved"
        else:
            prediction = "❌ Loan Not Approved"

    return render_template_string(HTML, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)