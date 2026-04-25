import json
import os

import mlflow
import numpy as np
import skops.io as sio
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris_classification")

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

model_type = params["train"]["model_type"]

if model_type == "logistic_regression":
    model = LogisticRegression(
        max_iter=params["train"]["logistic_regression"]["max_iter"],
        C=params["train"]["logistic_regression"]["C"],
        random_state=params["train"]["random_state"],
    )

elif model_type == "random_forest":
    model = RandomForestClassifier(
        n_estimators=params["train"]["random_forest"]["n_estimators"],
        max_depth=params["train"]["random_forest"]["max_depth"],
        random_state=params["train"]["random_state"],
    )

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

metrics = {
    "model_type": model_type,
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average="macro"),
    "recall": recall_score(y_test, y_pred, average="macro"),
}

os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

sio.dump(model, "models/model.pkl")

with open("metrics/train_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)


with mlflow.start_run() as run:
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("random_state", params["train"]["random_state"])
    
    if model_type == "logistic_regression":
        mlflow.log_param("max_iter", params["train"]["logistic_regression"]["max_iter"])
        mlflow.log_param("C", params["train"]["logistic_regression"]["C"])
    else:
        mlflow.log_param("n_estimators", params["train"]["random_forest"]["n_estimators"])
        mlflow.log_param("max_depth", params["train"]["random_forest"]["max_depth"])
    
    mlflow.log_metric("accuracy", metrics["accuracy"])
    mlflow.log_metric("precision", metrics["precision"])
    mlflow.log_metric("recall", metrics["recall"])
    
    mlflow.log_artifact("models/model.pkl")
    mlflow.log_artifact("metrics/train_metrics.json")
    
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name=f"iris_{model_type}",
    )
