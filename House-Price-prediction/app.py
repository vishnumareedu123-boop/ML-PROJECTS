from flask import Flask, request, render_template_string
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("models/house_price_model.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>House Price Prediction</title>
</head>
<body>

<h2>House Price Prediction</h2>

<form method="POST">

CRIM:<br>
<input type="text" name="crim" required><br><br>

ZN:<br>
<input type="text" name="zn" required><br><br>

INDUS:<br>
<input type="text" name="indus" required><br><br>

CHAS (0 or 1):<br>
<input type="text" name="chas" required><br><br>

NOX:<br>
<input type="text" name="nox" required><br><br>

RM:<br>
<input type="text" name="rm" required><br><br>

AGE:<br>
<input type="text" name="age" required><br><br>

DIS:<br>
<input type="text" name="dis" required><br><br>

RAD:<br>
<input type="text" name="rad" required><br><br>

TAX:<br>
<input type="text" name="tax" required><br><br>

PTRATIO:<br>
<input type="text" name="ptratio" required><br><br>

B:<br>
<input type="text" name="b" required><br><br>

LSTAT:<br>
<input type="text" name="lstat" required><br><br>

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
            "crim": float(request.form["crim"]),
            "zn": float(request.form["zn"]),
            "indus": float(request.form["indus"]),
            "chas": int(request.form["chas"]),
            "nox": float(request.form["nox"]),
            "rm": float(request.form["rm"]),
            "age": float(request.form["age"]),
            "dis": float(request.form["dis"]),
            "rad": float(request.form["rad"]),
            "tax": float(request.form["tax"]),
            "ptratio": float(request.form["ptratio"]),
            "b": float(request.form["b"]),
            "lstat": float(request.form["lstat"])
        }])

        result = model.predict(data)[0]

        prediction = f"🏠 Predicted House Price: {result:.2f}"

    return render_template_string(HTML, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)