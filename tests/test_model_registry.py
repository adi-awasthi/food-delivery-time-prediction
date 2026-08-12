import mlflow
import numpy as np


def test_registered_model_exists(mlflow_client, registered_model_name):
    """Catches: the model name being wrong somewhere, or registration
    silently never having happened."""
    registered_model = mlflow_client.get_registered_model(registered_model_name)
    assert registered_model.name == registered_model_name


def test_staging_alias_resolves(mlflow_client, registered_model_name):
    """Catches: the "staging" alias being unset, deleted, or left dangling
    after a bad register_model.py run."""
    version = mlflow_client.get_model_version_by_alias(registered_model_name, "staging")
    assert version is not None
    assert version.version is not None


def test_staging_version_is_ready(mlflow_client, registered_model_name):
    """Catches: a registration that started but never finished (e.g. an
    interrupted upload) -- MLflow allows querying a version mid-registration,
    so READY specifically confirms it's actually usable, not just present."""
    version = mlflow_client.get_model_version_by_alias(registered_model_name, "staging")
    assert version.status == "READY"


def test_staging_model_loads(registered_model_name):
    """Catches: environment/serialization mismatches -- e.g. the LightGBM
    version used to log the model differs from what's installed here, or the
    artifact got corrupted/partially uploaded."""
    model_uri = f"models:/{registered_model_name}@staging"
    model = mlflow.pyfunc.load_model(model_uri)
    assert model is not None


def test_staging_model_predicts_finite_values(registered_model_name, test_data):
    """Catches: signature drift -- if a future retrain changes the feature
    columns without updating the registry consistently, predict would either
    error or silently emit garbage. A load-only check would miss the
    "silently emits garbage" case; this doesn't."""
    X_test, _ = test_data
    model_uri = f"models:/{registered_model_name}@staging"
    model = mlflow.pyfunc.load_model(model_uri)

    sample = X_test.head(20)
    predictions = model.predict(sample)

    assert len(predictions) == len(sample)
    assert np.all(np.isfinite(predictions))
