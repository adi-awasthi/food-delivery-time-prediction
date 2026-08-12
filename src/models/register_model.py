import json

import mlflow

from src.models.train_utils import load_params, setup_mlflow

METRICS_PATH = "metrics.json"
PARAMS_PATH = "params.yaml"


def _get_latest_evaluation_run(client: mlflow.tracking.MlflowClient, params: dict):
    experiment = client.get_experiment_by_name(params["mlflow"]["experiment_name"])
    runs = client.search_runs(
        experiment.experiment_id,
        filter_string="tags.stage = 'dvc_pipeline_evaluation'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            "No dvc_pipeline_evaluation run found -- run the evaluation stage first."
        )
    # search_runs doesn't populate run.outputs (needed to find the logged
    # model's id below) -- only client.get_run does, so re-fetch the full run.
    return client.get_run(runs[0].info.run_id)


def register_new_version(params_path: str = PARAMS_PATH) -> str:
    """
    Registers the model produced by the most recent pipeline evaluation run,
    creating the registered model if it doesn't exist yet, and always
    aliases the new version "staging" -- this only marks "latest trained
    candidate", it never promotes to production. Safe to run on every
    `dvc repro`: it never silently changes what's currently serving.
    """
    params = load_params(params_path)
    model_name = params["registry"]["model_name"]

    setup_mlflow()
    client = mlflow.tracking.MlflowClient()

    run = _get_latest_evaluation_run(client, params)
    # MLflow 3.x logs models as a separate "Logged Model" entity (not the
    # older run-artifact-path style), so it must be referenced via
    # models:/<model_id>, not runs:/<run_id>/<artifact_path>.
    logged_model_id = run.outputs.model_outputs[0].model_id
    model_uri = f"models:/{logged_model_id}"

    result = mlflow.register_model(model_uri, model_name)
    client.set_registered_model_alias(model_name, "staging", result.version)

    return result.version


def promote_to_production(
    metrics_path: str = METRICS_PATH, params_path: str = PARAMS_PATH
) -> dict:
    """
    Explicit, separate action -- NOT part of `dvc repro`. Checks the
    "staging" version's test MAE against evaluation.promotion_mae_threshold
    in params.yaml, and only if it passes, repoints "production" to that
    version. Called explicitly here for the initial promotion, and later by
    CI (Phase 11) only after pytest passes.
    """
    params = load_params(params_path)
    model_name = params["registry"]["model_name"]
    threshold = params["evaluation"]["promotion_mae_threshold"]

    setup_mlflow()
    client = mlflow.tracking.MlflowClient()

    with open(metrics_path) as f:
        metrics = json.load(f)
    test_mae = metrics["test_mae"]

    staging_version = client.get_model_version_by_alias(model_name, "staging")

    promoted = test_mae <= threshold
    if promoted:
        client.set_registered_model_alias(
            model_name, "production", staging_version.version
        )

    return {
        "promoted": promoted,
        "version": staging_version.version,
        "test_mae": test_mae,
        "threshold": threshold,
    }


if __name__ == "__main__":
    import sys

    if "--promote" in sys.argv:
        # Called explicitly by CI (Phase 11) only after pytest has already
        # passed. promote_to_production() re-checks the threshold itself
        # too, so this fails loudly (non-zero exit) rather than silently
        # doing nothing if metrics.json is somehow stale or inconsistent
        # with what the tests just validated.
        result = promote_to_production()
        print(result)
        if not result["promoted"]:
            print(
                f"NOT promoted: test_mae {result['test_mae']} exceeds "
                f"threshold {result['threshold']}"
            )
            sys.exit(1)
    else:
        version = register_new_version()
        print(f"Registered version {version}, aliased 'staging'.")
