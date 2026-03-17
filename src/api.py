# ============================================================
# IMPORT LIBRARIES
# ============================================================
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# ============================================================
# DEFINE PROJECT PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/house_price_model.pkl")

# ============================================================
# LOAD MODEL
# ============================================================
model = joblib.load(MODEL_PATH)
print("Model loaded successfully")

# ============================================================
# INITIALIZE FASTAPI APP
# ============================================================
app = FastAPI(title="House Price Prediction API")

# ============================================================
# DEFINE INPUT SCHEMA
# ============================================================
class HouseFeatures(BaseModel):
    area_sqft: float
    bedrooms: int
    bathrooms: int
    parking: int
    floor: int
    property_type: str
    location: str
    city_tier: str
    age_years: int

# Optional: support batch predictions
class BatchInput(BaseModel):
    data: List[HouseFeatures]

# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================
@app.get("/")
def root():
    return {"message": "House Price Prediction API is running"}

# ============================================================
# SINGLE PREDICTION ENDPOINT
# ============================================================
@app.post("/predict")
def predict(input: HouseFeatures):
    # Convert input to DataFrame
    df = pd.DataFrame([input.dict()])
    # Predict
    prediction = model.predict(df)
    return {"predicted_price": float(prediction[0])}

# ============================================================
# BATCH PREDICTION ENDPOINT
# ============================================================
@app.post("/predict_batch")
def predict_batch(batch: BatchInput):
    # Convert list of HouseFeatures to DataFrame
    df = pd.DataFrame([item.dict() for item in batch.data])
    predictions = model.predict(df)
    return {"predicted_prices": predictions.tolist()}