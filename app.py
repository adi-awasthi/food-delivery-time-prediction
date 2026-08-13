from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.data.data_cleaning import add_distance_feature, add_time_of_day_features
from src.features.build_features import add_time_period_feature
from src.features.preprocessing import encode_ordinal_columns
from src.models.train_utils import load_params, setup_mlflow

PREPROCESSOR_PATH = "models/preprocessor.joblib"

model_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    params = load_params()
    setup_mlflow()

    model_name = params["registry"]["model_name"]
    client = mlflow.tracking.MlflowClient()
    version_info = client.get_model_version_by_alias(model_name, "production")

    model_state["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
    model_state["model"] = mlflow.pyfunc.load_model(f"models:/{model_name}@production")
    model_state["model_name"] = model_name
    model_state["model_version"] = version_info.version
    model_state["ordinal_columns"] = params["preprocessing"]["ordinal_columns"]

    yield
    model_state.clear()


app = FastAPI(title="Food Delivery Time Prediction API", lifespan=lifespan)

# Optional interview-demo page -- purely static, doesn't touch /predict's
# logic. Mounted under /static (idiomatic FastAPI static serving), served
# directly at the root path so opening the bare URL shows it immediately.
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def demo_page():
    return FileResponse("static/demo.html")


class DeliveryPredictionRequest(BaseModel):
    Delivery_person_Age: float = Field(..., ge=15, le=50)
    Delivery_person_Ratings: float = Field(..., ge=1, le=5)
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Vehicle_condition: int = Field(..., ge=0, le=3)
    order_datetime: datetime
    Road_traffic_density: Literal["Low", "Medium", "High", "Jam", "Unknown"]
    multiple_deliveries: Literal["0", "1", "2", "3", "Unknown"]
    Weatherconditions: Literal[
        "Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog", "Windy", "Unknown"
    ]
    Type_of_order: Literal["Snack", "Drinks", "Buffet", "Meal"]
    Type_of_vehicle: Literal["motorcycle", "scooter", "electric_scooter", "bicycle"]
    Festival: Literal["Yes", "No", "Unknown"]
    City: Literal["Urban", "Metropolitan", "Semi-Urban", "Unknown"]


class DeliveryPredictionResponse(BaseModel):
    predicted_delivery_time_minutes: float
    model_name: str
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=DeliveryPredictionResponse)
def predict(request: DeliveryPredictionRequest):
    df = pd.DataFrame([request.model_dump()])
    df["order_datetime"] = pd.to_datetime(df["order_datetime"])

    # Same feature-derivation functions used at training time -- a single
    # source of truth for distance/time-of-day/time_period logic, so serving
    # can never silently drift from how the model was actually trained.
    df = add_distance_feature(df)
    df = add_time_of_day_features(df)
    df = add_time_period_feature(df)
    df = encode_ordinal_columns(df, model_state["ordinal_columns"])

    preprocessor = model_state["preprocessor"]
    X_processed = preprocessor.transform(df)
    # Wrap back into a DataFrame with the exact training-time feature names
    # -- the registered model's signature was inferred against a DataFrame
    # with these names (see preprocessing.py / evaluate.py), not a bare
    # ndarray.
    X_processed_df = pd.DataFrame(
        X_processed, columns=preprocessor.get_feature_names_out()
    )

    prediction = model_state["model"].predict(X_processed_df)

    return DeliveryPredictionResponse(
        predicted_delivery_time_minutes=float(prediction[0]),
        model_name=model_state["model_name"],
        model_version=model_state["model_version"],
    )
