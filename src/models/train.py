import joblib
from lightgbm import LGBMRegressor

from src.models.train_utils import load_params, load_train_test

MODEL_ARTIFACT_PATH = "models/model.joblib"


def train_final_model(
    params_path: str = "params.yaml", model_output_path: str = MODEL_ARTIFACT_PATH
):
    """
    Trains the final selected model (tuned LightGBM, see MLflow run
    lgbm_tuned_best, tagged selected_for_deployment=true) using the
    hyperparameters recorded in params.yaml. This is the pipeline's
    production training step -- unlike Phase 6's experiment scripts, it
    only ever trains this one config, no search.
    """
    params = load_params(params_path)
    lgbm_params = params["train"]["lightgbm"]

    X_train, y_train, X_test, y_test = load_train_test()

    model = LGBMRegressor(**lgbm_params, verbosity=-1)
    model.fit(X_train, y_train)

    joblib.dump(model, model_output_path)
    return model


if __name__ == "__main__":
    train_final_model()
