from flask import Flask, request, render_template_string
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("models/employee_salary_model.pkl")

html = """
<!DOCTYPE html>
<html>
<head>
    <title>Employee Salary Prediction</title>
</head>
<body>

<h2>Employee Salary Prediction</h2>

<form method="POST">

Employee ID:<br>
<input type="number" name="Employee_ID" required><br><br>

Age:<br>
<input type="number" name="Age" required><br><br>

Gender (Male=1, Female=0):<br>
<input type="number" name="Gender" required><br><br>

Education (Bachelor=0, Master=1, PhD=2):<br>
<input type="number" name="Education" required><br><br>

Experience (Years):<br>
<input type="number" name="Experience" required><br><br>

Job Role (Encoded Value):<br>
<input type="number" name="Job_Role" required><br><br>

City (Encoded Value):<br>
<input type="number" name="City" required><br><br>

<input type="submit" value="Predict Salary">

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
            "Employee_ID": int(request.form["Employee_ID"]),
            "Age": int(request.form["Age"]),
            "Gender": int(request.form["Gender"]),
            "Education": int(request.form["Education"]),
            "Experience": int(request.form["Experience"]),
            "Job_Role": int(request.form["Job_Role"]),
            "City": int(request.form["City"])
        }])

        result = model.predict(data)[0]

        prediction = f"Predicted Employee Salary: ₹{result:,.2f}"

    return render_template_string(html, prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)