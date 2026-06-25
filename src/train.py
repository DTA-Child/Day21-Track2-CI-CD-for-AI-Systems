import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


def _ensure_experiment():
    """
    Dam bao MLflow dung artifact location cuc bo khi tracking URI la sqlite.
    Neu khong, mlflow.sklearn.log_model se gap loi 'mlflow-artifacts' scheme.
    """
    tracking_uri = mlflow.get_tracking_uri()
    if tracking_uri and tracking_uri.startswith("sqlite"):
        experiment_name = "wine_quality"
        artifact_location = "file://" + os.path.abspath("mlartifacts")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
        mlflow.set_experiment(experiment_name)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tao them dac trung tu du lieu Wine Quality de cai thien do chinh xac.
    Ho tro ca ten cot co dau cach (tu file CSV goc) va dau gach duoi (tu test).
    """
    df = df.copy()

    # Chuẩn hóa tên cột: dùng tên có dấu cách để tính toán
    col_map = {c: c.replace("_", " ") for c in df.columns}
    df_renamed = df.rename(columns=col_map)

    df["log_residual_sugar"] = np.log1p(df_renamed["residual sugar"])
    df["log_chlorides"] = np.log1p(df_renamed["chlorides"])
    df["log_free_sulfur_dioxide"] = np.log1p(df_renamed["free sulfur dioxide"])
    df["log_total_sulfur_dioxide"] = np.log1p(df_renamed["total sulfur dioxide"])
    df["free_total_so2_ratio"] = df_renamed["free sulfur dioxide"] / (df_renamed["total sulfur dioxide"] + 1e-6)
    df["alcohol_density"] = df_renamed["alcohol"] / df_renamed["density"]
    df["acid_ph"] = (df_renamed["fixed acidity"] + df_renamed["volatile acidity"]) / df_renamed["pH"]
    return df


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    df_train = engineer_features(df_train)
    df_eval = engineer_features(df_eval)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    _ensure_experiment()

    with mlflow.start_run():
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
