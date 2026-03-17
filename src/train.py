# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================
import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# DEFINE PROJECT PATHS
# ============================================================
# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "house_data.csv")

# Model save path
MODEL_PATH = os.path.join(BASE_DIR, "models", "house_price_model.pkl")


# ============================================================
# MLFLOW TRACKING CONFIGURATION
# ============================================================
# Define MLflow tracking location
MLFLOW_TRACKING_URI = "http://localhost:7000"

# Set tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Define experiment name
mlflow.set_experiment("house-price-prediction")


# ============================================================
# LOAD DATASET
# ============================================================
df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully")
print(df.head())


# ============================================================
# DATA PREPROCESSING
# ============================================================
# Remove unnecessary column
df = df.drop(columns=["id"])

# Split dataset into features and target
X = df.drop("price_inr", axis=1)
y = df["price_inr"]

# Define column types
categorical_cols = ["property_type", "location", "city_tier"]
numerical_cols = ["area_sqft", "bedrooms", "bathrooms", "parking", "floor", "age_years"]


# ============================================================
# FEATURE ENCODING
# ============================================================
# Encode categorical columns and keep numeric columns unchanged
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)


# ============================================================
# MODEL DEFINITION
# ============================================================
# Define machine learning model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# ============================================================
# CREATE PIPELINE
# ============================================================
# Combine preprocessing and model into a single pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])


# ============================================================
# TRAIN TEST SPLIT
# ============================================================
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# START MLFLOW RUN
# ============================================================
with mlflow.start_run():

    # --------------------------------------------------------
    # LOG PARAMETERS
    # --------------------------------------------------------
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)


    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------
    pipeline.fit(X_train, y_train)

    print("\nModel training completed.")


    # --------------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------------
    predictions = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation Results")
    print("MSE:", mse)
    print("R2 Score:", r2)


    # --------------------------------------------------------
    # LOG METRICS TO MLFLOW
    # --------------------------------------------------------
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)


    # --------------------------------------------------------
    # SAVE MODEL LOCALLY
    # --------------------------------------------------------
    joblib.dump(pipeline, MODEL_PATH)

    print("\nModel saved at:", MODEL_PATH)


    # --------------------------------------------------------
    # LOG MODEL ARTIFACT TO MLFLOW
    # --------------------------------------------------------
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model"
    )


# ============================================================
# SAMPLE PREDICTION TEST
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

prediction = pipeline.predict(sample_input)

print("\nSample Prediction:", prediction[0])