import time
import logging
import json
import os
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel, validator
from src.predict import predict_customer
from src.impact import calculate_impact

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

file_handler = logging.FileHandler("logs/predictions.log")
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

app = FastAPI(
    title="ChurnIQ API",
    description="Telecom churn prediction with SHAP explainability and financial impact analysis",
    version="1.0.0"
)

class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract_Month_to_month: float = 0
    Contract_One_year: float = 0
    Contract_Two_year: float = 0
    InternetService_Fiber_optic: float = 0
    InternetService_DSL: float = 0
    PaymentMethod_Electronic_check: float = 0

    @validator("tenure")
    def tenure_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Tenure must be positive")
        return v

    @validator("MonthlyCharges")
    def monthly_charges_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Monthly charges must be positive")
        return v

    @validator("TotalCharges")
    def total_charges_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Total charges must be positive")
        return v

@app.get("/")
def home():
    return {"message": "ChurnIQ API is running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Logistic Regression",
        "auc_roc": 0.8357,
        "features": 26,
        "training_samples": 5625
    }

@app.post("/predict")
def predict(data: CustomerData):
    start = time.time()
    logger.info(f"Prediction request received: {data.model_dump()}")

    result = predict_customer(data.model_dump())

    response_time = round((time.time() - start) * 1000, 2)
    result["response_time_ms"] = response_time

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": data.model_dump(),
        "prediction": result
    }
    logger.info(json.dumps(log_entry))

    return result

@app.get("/impact")
def impact(avg_revenue: float = 74, retention_cost: float = 500, top_n: int = 100):
    if avg_revenue <= 0:
        return {"error": "Average revenue must be positive"}
    if retention_cost <= 0:
        return {"error": "Retention cost must be positive"}
    if top_n <= 0:
        return {"error": "Top N must be positive"}

    start = time.time()
    result = calculate_impact(avg_revenue, retention_cost, top_n)
    result["response_time_ms"] = round((time.time() - start) * 1000, 2)
    return result