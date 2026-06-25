import os
import numpy as np
import boto3
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

S3_BUCKET = os.environ["S3_BUCKET"]
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def engineer_features(x: np.ndarray) -> np.ndarray:
    """
    Tao them dac trung tu du lieu Wine Quality (giong nhu trong train.py).
    x: ndarray shape (n_samples, 12) theo thu tu:
       [fixed acidity, volatile acidity, citric acid, residual sugar,
        chlorides, free sulfur dioxide, total sulfur dioxide, density,
        pH, sulphates, alcohol, wine_type]
    """
    fixed = x[:, 0]
    volatile = x[:, 1]
    residual_sugar = x[:, 3]
    chlorides = x[:, 4]
    free_so2 = x[:, 5]
    total_so2 = x[:, 6]
    density = x[:, 7]
    ph = x[:, 8]
    alcohol = x[:, 10]

    engineered = np.stack([
        np.log1p(residual_sugar),
        np.log1p(chlorides),
        np.log1p(free_so2),
        np.log1p(total_so2),
        free_so2 / (total_so2 + 1e-6),
        alcohol / density,
        (fixed + volatile) / ph,
    ], axis=1)

    return np.concatenate([x, engineered], axis=1)


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

    x = np.array([req.features], dtype=float)
    x = engineer_features(x)
    pred = int(model.predict(x)[0])
    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {"prediction": pred, "label": labels[pred]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
