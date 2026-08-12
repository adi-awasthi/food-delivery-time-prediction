import mlflow
import mlflow.sklearn
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

from src.models.train_utils import load_train_test, setup_mlflow

# Best configs from Step 2 (RF) and Step 4 (LightGBM) tuning -- held fixed
# here, this step combines the already-tuned models rather than re-tuning.
BEST_RF_PARAMS = {
    "n_estimators": 100,
    "min_samples_split": 20,
    "min_samples_leaf": 2,
    "max_features": 0.8,
    "max_depth": None,
}

BEST_LGBM_PARAMS = {
    "learning_rate": 0.01,
    "num_leaves": 127,
    "n_estimators": 500,
    "reg_alpha": 0.1,
    "reg_lambda": 0,
}


def build_stacking_model() -> StackingRegressor:
    base_estimators = [
        ("rf", RandomForestRegressor(**BEST_RF_PARAMS, random_state=42, n_jobs=-1)),
        ("lgbm", LGBMRegressor(**BEST_LGBM_PARAMS, random_state=42, verbosity=-1)),
    ]
    return StackingRegressor(
        estimators=base_estimators,
        final_estimator=LinearRegression(),
        cv=5,
        n_jobs=-1,
    )


def train_stacking_model(cv: int = 5):
    setup_mlflow()
    X_train, y_train, X_test, y_test = load_train_test()

    cv_mae_scores = -cross_val_score(
        build_stacking_model(),
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=1,  # each stacking fit is already internally parallelized
    )

    final_model = build_stacking_model()
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    with mlflow.start_run(run_name="stacking_rf_lgbm"):
        mlflow.set_tag("model_type", "Stacking_RF_LightGBM")
        mlflow.log_params({f"rf__{k}": v for k, v in BEST_RF_PARAMS.items()})
        mlflow.log_params({f"lgbm__{k}": v for k, v in BEST_LGBM_PARAMS.items()})
        mlflow.log_param("final_estimator", "LinearRegression")
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("cv_mae_mean", cv_mae_scores.mean())
        mlflow.log_metric("cv_mae_std", cv_mae_scores.std())
        for i, score in enumerate(cv_mae_scores):
            mlflow.log_metric(f"cv_mae_fold_{i}", score)

        signature = mlflow.models.infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(
            final_model,
            name="model",
            signature=signature,
            input_example=X_train.head(5),
            # The stacking model nests a lightgbm.Booster, which mlflow's
            # default "skops" serialization refuses to trust by default
            # (security safeguard against arbitrary deserialization).
            # cloudpickle is the standard fallback for models containing
            # non-sklearn-native components.
            serialization_format="cloudpickle",
        )

    return {
        "test_mae": test_mae,
        "test_r2": test_r2,
        "cv_mae_scores": cv_mae_scores,
        "cv_mae_mean": cv_mae_scores.mean(),
        "cv_mae_std": cv_mae_scores.std(),
    }


if __name__ == "__main__":
    results = train_stacking_model()
    print("test_mae:", results["test_mae"])
    print("test_r2:", results["test_r2"])
    print("cv_mae_scores:", results["cv_mae_scores"])
    print("cv_mae_mean:", results["cv_mae_mean"])
    print("cv_mae_std:", results["cv_mae_std"])
