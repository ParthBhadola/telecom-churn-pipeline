import pandas as pd
import shap
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LogisticRegression

def load_model_and_data():
    with open("model/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    X_test = pd.read_csv("data/X_test.csv")
    return model, X_test, feature_names

def global_explanation(model, X_test):
    os.makedirs("plots", exist_ok=True)

    explainer = shap.LinearExplainer(model, X_test)
    shap_values = explainer.shap_values(X_test)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=X_test.columns)
    feature_importance = feature_importance.sort_values(ascending=True).tail(15)

    plt.figure(figsize=(10, 8))
    feature_importance.plot(kind="barh", color="#4C72B0")
    plt.title("Global Feature Importance (SHAP)", fontsize=14)
    plt.xlabel("Mean |SHAP Value|")
    plt.tight_layout()
    plt.savefig("plots/shap_global.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Global SHAP plot saved to plots/shap_global.png")

    return explainer, shap_values

def customer_explanation(model, X_test, shap_values, customer_index=0):
    sv = shap_values[customer_index]
    feature_impacts = list(zip(X_test.columns, sv))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\nTop 5 reasons for customer {customer_index} prediction:")
    for feat, val in feature_impacts[:5]:
        direction = "increases" if val > 0 else "decreases"
        print(f"  {feat}: {direction} churn probability by {abs(val):.4f}")

    top_features = feature_impacts[:10]
    features = [f[0] for f in top_features]
    values = [f[1] for f in top_features]
    colors = ["#d73027" if v > 0 else "#4575b4" for v in values]

    plt.figure(figsize=(10, 6))
    plt.barh(features, values, color=colors)
    plt.axvline(x=0, color="black", linewidth=0.8)
    plt.title(f"Customer {customer_index} — Churn Prediction Explanation", fontsize=13)
    plt.xlabel("SHAP Value (red = increases churn, blue = decreases churn)")
    plt.tight_layout()
    plt.savefig(f"plots/shap_customer_{customer_index}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Customer {customer_index} plot saved to plots/shap_customer_{customer_index}.png")

if __name__ == "__main__":
    model, X_test, feature_names = load_model_and_data()
    print("Running global explanation...")
    explainer, shap_values = global_explanation(model, X_test)
    print("\nRunning individual customer explanations...")
    customer_explanation(model, X_test, shap_values, customer_index=0)
    customer_explanation(model, X_test, shap_values, customer_index=5)