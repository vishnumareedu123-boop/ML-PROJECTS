import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(filepath):

    # Load Dataset
    df = pd.read_csv(filepath)

    # Fill Missing Values
    df = df.ffill()

    # Encode Categorical Columns
    encoder = LabelEncoder()

    df["Gender"] = encoder.fit_transform(df["Gender"])
    df["Contract"] = encoder.fit_transform(df["Contract"])
    df["Internet_Service"] = encoder.fit_transform(df["Internet_Service"])
    df["Payment_Method"] = encoder.fit_transform(df["Payment_Method"])
    df["Churn"] = encoder.fit_transform(df["Churn"])

    return df


if __name__ == "__main__":

    df = preprocess_data("data/customer_churn.csv")

    print(df.head())