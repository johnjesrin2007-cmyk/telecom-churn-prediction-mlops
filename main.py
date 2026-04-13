from fastapi import FastAPI
import os
import joblib
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="ML Model API")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")


model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from: {MODEL_PATH}")
else:
    print(f"❌ Model not found at: {MODEL_PATH}")



class InputData(BaseModel):
    customer_id:str
    gender:str
    SeniorCitizen:float
    Partner:str
    Dependents:str
    tenure:float
    PhoneService:str
    MultipleLines:str
    InternetService:str
    OnlineSecurity:str
    OnlineBackup:str
    DeviceProtection:str
    TechSupport:str
    StreamingTV:str
    StreamingMovies:str
    Contract:str
    PaperlessBilling:str
    PaymentMethod:str
    MonthlyCharges:float
    TotalCharges:float

@app.get("/")
def home():
    return {"message": "ML Model API is running successfully"}



@app.post("/predict")
def predict(data: InputData):

    if model is None:
        return {"error": "Model not loaded. Train model first."}

    try:
        input_df = pd.DataFrame([data.dict()])

        prediction = model.predict(input_df)

        return {
            "input": data.dict(),
            "predicted_price": float(prediction[0])
        }

    except Exception as e:
        return {"error": str(e)}