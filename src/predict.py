# ============================================================
# IMPORT LIBRARIES
# ============================================================
import os
import joblib
import pandas as pd


# ============================================================
# DEFINE PROJECT PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "house_price_model.pkl")


# ============================================================
# LOAD MODEL
# ============================================================
model = joblib.load(MODEL_PATH)

print("Model loaded successfully")


# ============================================================
# CREATE SAMPLE INPUT
# ============================================================
sample_input = pd.DataFrame({
    "area_sqft": [1500],
    "bedrooms": [3],
    "bathrooms": [2],
    "parking": [1],
    "floor": [3],
    "property_type": ["Independent House"],
    "location": ["Hyderabad"],
    "city_tier": ["tier1"],
    "age_years": [4]
})


# ============================================================
# MAKE PREDICTION
# ============================================================
prediction = model.predict(sample_input)

print("\nPredicted Price:", prediction[0])