import pandas as pd

def preprocess_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)

    # Fill missing values
    df = df.ffill()

    return df


if __name__ == "__main__":
    df = preprocess_data("data/house_price.csv")
    print(df.head())