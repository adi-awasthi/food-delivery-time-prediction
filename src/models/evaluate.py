import json

import joblib
import mlflow
import mlflow.lightgbm
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

from src.models.train_utils import load_params, load_train_test, setup_mlflow
from src.models.train import MODEL_ARTIFACT_PATH

METRICS_PATH = "metrics.json"


def evaluate_model(
    params_path: str = "params.yaml",
    model_path: str = MODEL_ARTIFACT_PATH,
    metrics_output_path: str = METRICS_PATH,
):
    params = load_params(params_path)
    lgbm_params = params["train"]["lightgbm"]
    cv_folds = params["evaluation"]["cv_folds"]

    setup_mlflow()
    X_train, y_train, X_test, y_test = load_train_test()

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    # cross_val_score refits fresh copies internally, so this uses the same
    # hyperparameters as the trained model but is not just reusing its
    # single train/test fit -- consistent with how CV MAE was measured
    # throughout Phase 6.
    cv_mae_scores = -cross_val_score(
        LGBMRegressor(**lgbm_params, verbosity=-1),
        X_train,
        y_train,
        cv=cv_folds,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    metrics = {
        "test_mae": test_mae,
        "test_r2": test_r2,
        "cv_mae_mean": cv_mae_scores.mean(),
        "cv_mae_std": cv_mae_scores.std(),
    }

    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with mlflow.start_run(run_name="pipeline_evaluation"):
        mlflow.set_tag("model_type", "LightGBM_tuned")
        mlflow.set_tag("stage", "dvc_pipeline_evaluation")
        mlflow.log_params(lgbm_params)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        signature = mlflow.models.infer_signature(X_train, y_pred)
        mlflow.lightgbm.log_model(
            model, name="model", signature=signature, input_example=X_train.head(5)
        )

    return metrics


if __name__ == "__main__":
    metrics = evaluate_model()
    print(json.dumps(metrics, indent=2))
