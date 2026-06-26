import pandas as pd
import sqlite3
import os

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
DB_PATH = "data/churn.db"

def load_to_sqlite():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("customers", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Loaded {len(df)} rows into SQLite")

def run_eda():
    conn = sqlite3.connect(DB_PATH)

    print("\n--- Churn Rate Overall ---")
    result = pd.read_sql("SELECT AVG(Churn) as churn_rate FROM customers", conn)
    print(result)

    print("\n--- Churn Rate by Contract Type ---")
    result = pd.read_sql("""
        SELECT Contract, AVG(Churn) as churn_rate, COUNT(*) as total
        FROM customers
        GROUP BY Contract
        ORDER BY churn_rate DESC
    """, conn)
    print(result)

    print("\n--- Churn Rate by Internet Service ---")
    result = pd.read_sql("""
        SELECT InternetService, AVG(Churn) as churn_rate, COUNT(*) as total
        FROM customers
        GROUP BY InternetService
        ORDER BY churn_rate DESC
    """, conn)
    print(result)

    print("\n--- Average Monthly Charges by Churn ---")
    result = pd.read_sql("""
        SELECT Churn, AVG(MonthlyCharges) as avg_monthly, AVG(tenure) as avg_tenure
        FROM customers
        GROUP BY Churn
    """, conn)
    print(result)

    print("\n--- Top 5 High Risk Segments ---")
    result = pd.read_sql("""
        SELECT Contract, InternetService, PaymentMethod,
               AVG(Churn) as churn_rate, COUNT(*) as total
        FROM customers
        GROUP BY Contract, InternetService, PaymentMethod
        HAVING total > 50
        ORDER BY churn_rate DESC
        LIMIT 5
    """, conn)
    print(result)

    conn.close()

if __name__ == "__main__":
    load_to_sqlite()
    run_eda()