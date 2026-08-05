Customer Churn Prediction
Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave (churn) or stay with a company based on customer information.

This project is built using Python, Pandas, Scikit-learn, and Flask.

Features
Data Preprocessing
Exploratory Data Analysis (EDA)
Label Encoding
Logistic Regression Model
Model Evaluation
Model Saving using Joblib
Prediction Script
Flask Web Application
Project Structure
Customer-Churn-Prediction/
│
├── data/
│   └── customer_churn.csv
│
├── models/
│   └── customer_churn_model.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── app.py
├── Customer_Churn_Prediction.ipynb
├── requirements.txt
├── README.md
└── .gitignore
Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Flask
Joblib
Installation

Clone the repository:

git clone <repository-url>

Go to the project folder:

cd Customer-Churn-Prediction

Install the required packages:

pip install -r requirements.txt
Train the Model
py src/train.py
Predict
py src/predict.py
Run the Flask Application
py app.py

Open your browser and visit:

http://127.0.0.1:5000
Machine Learning Algorithm
Logistic Regression
Project Workflow
Load Dataset
Data Cleaning
Label Encoding
Feature Selection
Train-Test Split
Model Training
Model Evaluation
Save Model
Predict Customer Churn
Deploy with Flask
Output

The application predicts whether a customer is likely to:

Customer Will Churn
Customer Will Not Churn
Author

Vishnu

Machine Learning & AI Project