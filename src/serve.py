import os
import numpy as np
import pandas as pd
import boto3
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

S3_BUCKET = os.environ["S3_BUCKET"]
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

FEATURE_NAMES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tao them dac trung tu du lieu Wine Quality (giong nhu trong train.py)."""
    df = df.copy()
    df["log_residual_sugar"] = np.log1p(df["residual sugar"])
    df["log_chlorides"] = np.log1p(df["chlorides"])
    df["log_free_sulfur_dioxide"] = np.log1p(df["free sulfur dioxide"])
    df["log_total_sulfur_dioxide"] = np.log1p(df["total sulfur dioxide"])
    df["free_total_so2_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-6)
    df["alcohol_density"] = df["alcohol"] / df["density"]
    df["acid_ph"] = (df["fixed acidity"] + df["volatile acidity"]) / df["pH"]
    return df


def download_model():
    """Tai file model.pkl tu S3 ve may khi server khoi dong."""
    s3 = boto3.client("s3")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
    print(f"Model downloaded from s3://{S3_BUCKET}/{S3_MODEL_KEY}")


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """Endpoint kiem tra suc khoe server."""
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan.

    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}
    """
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features")

    df = pd.DataFrame([req.features], columns=FEATURE_NAMES)
    df = engineer_features(df)
    pred = int(model.predict(df)[0])
    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {"prediction": pred, "label": labels[pred]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
