import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

DB_PATH = "data/churn.db"

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()
    return df

def preprocess(df):
    df = df.drop(columns=["customerID"])

    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]

    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"])

    return df

def split_data(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs("data", exist_ok=True)
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    print(f"Train size: {len(X_train)}")
    print(f"Test size:  {len(X_test)}")
    print(f"Churn rate in train: {y_train.mean():.2%}")
    print(f"Churn rate in test:  {y_test.mean():.2%}")
    print(f"\nFeatures: {list(X.columns)}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data()
    print(f"Raw data shape: {df.shape}")
    df = preprocess(df)
    print(f"Processed data shape: {df.shape}")
    split_data(df)