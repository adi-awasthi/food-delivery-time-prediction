import mlflow
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PowerTransformer

from src.models.train_utils import load_train_test, setup_mlflow

# Best config found by the Step 4 LightGBM RandomizedSearchCV -- held fixed
# here since this step is isolating the effect of the target transformation,
# not re-tuning.
BEST_LGBM_PARAMS = {
    "learning_rate": 0.01,
    "num_leaves": 127,
    "n_estimators": 500,
    "reg_alpha": 0.1,
    "reg_lambda": 0,
}


def run_target_transform_ab_test(cv: int = 5):
    setup_mlflow()
    X_train, y_train, X_test, y_test = load_train_test()

    results = {}

    for variant, model in [
        (
            "raw_target",
            LGBMRegressor(**BEST_LGBM_PARAMS, random_state=42, verbosity=-1),
        ),
        (
            "power_transformed_target",
            TransformedTargetRegressor(
                regressor=LGBMRegressor(
                    **BEST_LGBM_PARAMS, random_state=42, verbosity=-1
                ),
                transformer=PowerTransformer(),
            ),
        ),
    ]:
        cv_mae_scores = -cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )

        with mlflow.start_run(run_name=f"lgbm_tuned_{variant}"):
            mlflow.set_tag("model_type", "LightGBM_tuned")
            mlflow.set_tag("target_transform", variant)
            mlflow.log_params(BEST_LGBM_PARAMS)
            mlflow.log_metric("cv_mae_mean", cv_mae_scores.mean())
            mlflow.log_metric("cv_mae_std", cv_mae_scores.std())
            for i, score in enumerate(cv_mae_scores):
                mlflow.log_metric(f"cv_mae_fold_{i}", score)

        results[variant] = {
            "cv_mae_mean": cv_mae_scores.mean(),
            "cv_mae_std": cv_mae_scores.std(),
            "cv_mae_scores": cv_mae_scores,
        }

    return results


if __name__ == "__main__":
    results = run_target_transform_ab_test()
    for variant, r in results.items():
        print(variant, "cv_mae_mean:", r["cv_mae_mean"], "cv_mae_std:", r["cv_mae_std"])
