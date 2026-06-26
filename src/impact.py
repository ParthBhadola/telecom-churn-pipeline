import pandas as pd
import pickle

def load_model():
    with open("model/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, feature_names

def calculate_impact(avg_monthly_revenue, retention_cost_per_customer, top_n=100):
    model, feature_names = load_model()
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    probs = model.predict_proba(X_test)[:, 1]

    results = X_test.copy()
    results["churn_probability"] = probs
    results["actual_churn"] = y_test.values

    at_risk = results[results["churn_probability"] > 0.5].copy()
    at_risk = at_risk.sort_values("churn_probability", ascending=False)
    top_customers = at_risk.head(top_n)

    revenue_saved = top_n * avg_monthly_revenue * 12
    total_retention_cost = top_n * retention_cost_per_customer
    net_benefit = revenue_saved - total_retention_cost
    roi = (net_benefit / total_retention_cost) * 100

    print(f"\n===== Financial Impact Calculator =====")
    print(f"Top {top_n} at-risk customers identified")
    print(f"Average churn probability: {top_customers['churn_probability'].mean():.2%}")
    print(f"\nRevenue saved (12 months):  Rs.{revenue_saved:,.0f}")
    print(f"Total retention cost:        Rs.{total_retention_cost:,.0f}")
    print(f"Net benefit:                 Rs.{net_benefit:,.0f}")
    print(f"ROI:                         {roi:.1f}%")

    return {
        "top_n": top_n,
        "revenue_saved": revenue_saved,
        "retention_cost": total_retention_cost,
        "net_benefit": net_benefit,
        "roi": roi
    }

if __name__ == "__main__":
    calculate_impact(
        avg_monthly_revenue=74,
        retention_cost_per_customer=500,
        top_n=100
    )