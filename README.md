# ChurnIQ — Telecom Churn Prediction Pipeline

Predict which customers will churn, understand why and quantify the financial impact of retaining them.

---

## What it does

- Predicts customer churn probability using a trained ML model
- Explains every prediction with SHAP — showing exactly which features drove the decision
- Quantifies financial impact — revenue saved, retention cost and ROI for any intervention campaign
- Tracks all model experiments with MLflow

---

## System Architecture
Raw CSV Data

↓

SQL-based EDA (SQLite)

↓

Preprocessing — encoding, scaling, train-test split

↓

3 Models trained and logged to MLflow

(Logistic Regression, Random Forest, XGBoost)

↓

Best model selected by AUC-ROC

↓

SHAP explainability — global and per-customer

↓

Financial impact calculator

↓

FastAPI endpoint + Streamlit dashboard

---

## Results

| Model | AUC-ROC |
|---|---|
| Logistic Regression | 0.8357 |
| Random Forest | 0.8200 |
| XGBoost | 0.8140 |

Best model: Logistic Regression — AUC-ROC 0.8357

**Financial impact on top 100 at-risk customers:**
- Revenue saved: $88,800
- Retention cost: $50,000
- Net benefit: $38,800
- ROI: 77.6%

---

## Key Findings from EDA

- Overall churn rate: 26.6%
- Month-to-month contracts churn at 42.7% vs 2.8% for two-year contracts
- Fiber optic customers churn at 41.9% despite paying more
- Highest risk segment: month-to-month + fiber optic + electronic check = 60% churn rate
- Churned customers pay $74 avg monthly vs $61 for retained customers

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data storage | SQLite |
| EDA | SQL + Pandas |
| Preprocessing | Scikit-learn |
| Models | Logistic Regression, Random Forest, XGBoost |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| Backend API | FastAPI |
| Frontend | Streamlit |

---

## Project Structure
telecom-churn-pipeline/

src/

ingest.py       — load CSV to SQLite, SQL EDA

preprocess.py   — feature engineering and train-test split

train.py        — train 3 models, log to MLflow

explain.py      — SHAP global and customer explanations

impact.py       — financial impact calculator

predict.py      — load model and return prediction

main.py             — FastAPI backend

app.py              — Streamlit dashboard

plots/              — SHAP visualisation outputs

data/               — dataset (gitignored)

model/              — saved model (gitignored)

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/ParthBhadola/telecom-churn-pipeline.git
cd telecom-churn-pipeline
```

**2. Install dependencies**
```bash
pip install pandas scikit-learn xgboost shap mlflow fastapi uvicorn streamlit python-dotenv
```

**3. Download dataset**

Download Telco Customer Churn dataset from Kaggle and place it in `data/` folder.

**4. Run the pipeline**
```bash
python src/ingest.py
python src/preprocess.py
python src/train.py
python src/explain.py
python src/impact.py
```

**5. Run the app**
```bash
streamlit run app.py
```

**6. Run the API**
```bash
uvicorn main:app --reload
```

**7. View MLflow experiments**
```bash
python -m mlflow ui
```

---

## Interview Talking Points

**Why Logistic Regression beat XGBoost?**
On clean, structured tabular data with proper class imbalance handling,
simpler models can outperform. Logistic Regression also trains faster
and is more interpretable — important for a business-facing tool.

**Why SHAP over feature importance?**
Feature importance shows global averages. SHAP shows per-prediction
contribution — telling the business exactly why this specific customer
will churn and what intervention to use.

**What is the business value?**
The model identifies top 100 at-risk customers with 88% average churn
probability. Retaining them generates a 77.6% ROI on the retention spend.