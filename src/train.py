import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=3, random_state=42, eval_metric="logloss")
}

mlflow.set_experiment("telecom-churn")

best_model = None
best_auc = 0
best_name = ""

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)

        mlflow.log_param("model", name)
        mlflow.log_metric("auc_roc", auc)

        if name == "XGBoost":
            mlflow.xgboost.log_model(model, name)
        else:
            mlflow.sklearn.log_model(model, name)

        print(f"\n===== {name} =====")
        print(f"AUC-ROC: {auc:.4f}")
        print(classification_report(y_test, preds))

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_name = name

os.makedirs("model", exist_ok=True)
with open("model/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model/feature_names.pkl", "wb") as f:
    pickle.dump(list(X_train.columns), f)

print(f"\nBest model: {best_name} with AUC-ROC: {best_auc:.4f}")
print("Saved to model/best_model.pkl")