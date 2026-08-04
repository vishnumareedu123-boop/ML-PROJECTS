import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)

    # Fill missing values
    df = df.ffill()

    # Encode categorical columns
    encoder = LabelEncoder()

    df["Gender"] = encoder.fit_transform(df["Gender"])
    df["Education"] = encoder.fit_transform(df["Education"])
    df["Job_Role"] = encoder.fit_transform(df["Job_Role"])
    df["City"] = encoder.fit_transform(df["City"])

    return df


if __name__ == "__main__":
    df = preprocess_data("data/employee_salary.csv")
    print(df.head())