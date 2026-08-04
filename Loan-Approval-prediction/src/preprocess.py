import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(filepath):

    # Load dataset
    df = pd.read_csv(filepath)

    # Fill missing values
    df = df.ffill()

    # Encode categorical columns
    le = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col])

    return df

if __name__ == "__main__":
    df = preprocess_data("data/loan.csv")
    print(df.head())