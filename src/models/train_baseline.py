import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
TARGET_COLUMN = "Time_taken(min)"


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_train_test():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, y_train, X_test, y_test


def run_baseline():
    load_dotenv()
    params = load_params()

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    X_train, y_train, X_test, y_test = load_train_test()

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    cv_mae_scores = -cross_val_score(
        LinearRegression(), X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
    )

    with mlflow.start_run(run_name="baseline"):
        mlflow.set_tag("model_type", "LinearRegression")
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("cv_mae_mean", cv_mae_scores.mean())
        mlflow.log_metric("cv_mae_std", cv_mae_scores.std())
        for i, score in enumerate(cv_mae_scores):
            mlflow.log_metric(f"cv_mae_fold_{i}", score)

        signature = mlflow.models.infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(
            model, name="model", signature=signature, input_example=X_train.head(5)
        )

    return {
        "test_mae": test_mae,
        "test_r2": test_r2,
        "cv_mae_scores": cv_mae_scores,
        "cv_mae_mean": cv_mae_scores.mean(),
        "cv_mae_std": cv_mae_scores.std(),
    }


if __name__ == "__main__":
    results = run_baseline()
    print("test_mae:", results["test_mae"])
    print("test_r2:", results["test_r2"])
    print("cv_mae_scores:", results["cv_mae_scores"])
    print("cv_mae_mean:", results["cv_mae_mean"])
    print("cv_mae_std:", results["cv_mae_std"])
