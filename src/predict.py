import pandas as pd
import pickle

def load_model():
    with open("model/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, feature_names

def predict_customer(customer_data: dict):
    model, feature_names = load_model()
    df = pd.DataFrame([customer_data])
    df = df.reindex(columns=feature_names, fill_value=0)
    prob = model.predict_proba(df)[:, 1][0]
    prediction = int(prob > 0.5)
    return {
        "churn_probability": round(float(prob), 4),
        "will_churn": bool(prediction),
        "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
    }