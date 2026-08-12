import mlflow
import mlflow.lightgbm
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from src.models.train_utils import load_train_test, setup_mlflow

LGBM_PARAM_DISTRIBUTIONS = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "num_leaves": [15, 31, 63, 127, 255],
    "n_estimators": [100, 300, 500, 800, 1200],
    "reg_alpha": [0, 0.1, 0.5, 1, 5],
    "reg_lambda": [0, 0.1, 0.5, 1, 5],
}


def train_default_lightgbm():
    setup_mlflow()
    X_train, y_train, X_test, y_test = load_train_test()

    model = LGBMRegressor(random_state=42, verbosity=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    cv_mae_scores = -cross_val_score(
        LGBMRegressor(random_state=42, verbosity=-1),
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="lgbm_default"):
        mlflow.set_tag("model_type", "LightGBM_default")
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("cv_mae_mean", cv_mae_scores.mean())
        mlflow.log_metric("cv_mae_std", cv_mae_scores.std())
        for i, score in enumerate(cv_mae_scores):
            mlflow.log_metric(f"cv_mae_fold_{i}", score)

        signature = mlflow.models.infer_signature(X_train, y_pred)
        mlflow.lightgbm.log_model(
            model, name="model", signature=signature, input_example=X_train.head(5)
        )

    return {
        "test_mae": test_mae,
        "test_r2": test_r2,
        "cv_mae_scores": cv_mae_scores,
        "cv_mae_mean": cv_mae_scores.mean(),
        "cv_mae_std": cv_mae_scores.std(),
    }


def tune_lightgbm(n_iter: int = 30, cv: int = 5, random_state: int = 42):
    setup_mlflow()
    X_train, y_train, X_test, y_test = load_train_test()

    base_model = LGBMRegressor(random_state=random_state, verbosity=-1, n_jobs=-1)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=LGBM_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    cv_results = search.cv_results_

    for i in range(len(cv_results["params"])):
        with mlflow.start_run(run_name=f"lgbm_trial_{i}"):
            mlflow.set_tag("model_type", "LightGBM_tuning_trial")
            mlflow.log_params(cv_results["params"][i])
            mlflow.log_metric("cv_mae_mean", -cv_results["mean_test_score"][i])
            mlflow.log_metric("cv_mae_std", cv_results["std_test_score"][i])
            for fold in range(cv):
                mlflow.log_metric(
                    f"cv_mae_fold_{fold}", -cv_results[f"split{fold}_test_score"][i]
                )

    best_index = search.best_index_
    best_params = search.best_params_
    best_cv_mae_mean = -cv_results["mean_test_score"][best_index]
    best_cv_mae_std = cv_results["std_test_score"][best_index]
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    with mlflow.start_run(run_name="lgbm_tuned_best"):
        mlflow.set_tag("model_type", "LightGBM_tuned")
        mlflow.log_params(best_params)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("cv_mae_mean", best_cv_mae_mean)
        mlflow.log_metric("cv_mae_std", best_cv_mae_std)
        for fold in range(cv):
            mlflow.log_metric(
                f"cv_mae_fold_{fold}", -cv_results[f"split{fold}_test_score"][best_index]
            )
        signature = mlflow.models.infer_signature(X_train, y_pred)
        mlflow.lightgbm.log_model(
            best_model, name="model", signature=signature, input_example=X_train.head(5)
        )

    return {
        "best_params": best_params,
        "cv_mae_mean": best_cv_mae_mean,
        "cv_mae_std": best_cv_mae_std,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "best_model": best_model,
    }


if __name__ == "__main__":
    results = train_default_lightgbm()
    print("test_mae:", results["test_mae"])
    print("test_r2:", results["test_r2"])
    print("cv_mae_scores:", results["cv_mae_scores"])
    print("cv_mae_mean:", results["cv_mae_mean"])
    print("cv_mae_std:", results["cv_mae_std"])
