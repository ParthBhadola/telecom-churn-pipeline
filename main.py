from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_customer
from src.impact import calculate_impact

app = FastAPI()

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

@app.get("/")
def home():
    return {"message": "Telecom Churn API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    customer = data.model_dump()
    result = predict_customer(customer)
    return result

@app.get("/impact")
def impact(avg_revenue: float = 74, retention_cost: float = 500, top_n: int = 100):
    result = calculate_impact(avg_revenue, retention_cost, top_n)
    return result