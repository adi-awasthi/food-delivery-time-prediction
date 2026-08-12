import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import pytest

from src.models.train_utils import load_params, load_train_test, setup_mlflow


@pytest.fixture(scope="session", autouse=True)
def _configure_mlflow():
    setup_mlflow()


@pytest.fixture(scope="session")
def params():
    return load_params()


@pytest.fixture(scope="session")
def registered_model_name(params):
    return params["registry"]["model_name"]


@pytest.fixture(scope="session")
def promotion_threshold(params):
    return params["evaluation"]["promotion_mae_threshold"]


@pytest.fixture(scope="session")
def mlflow_client(_configure_mlflow):
    return mlflow.tracking.MlflowClient()


@pytest.fixture(scope="session")
def test_data():
    _, _, X_test, y_test = load_train_test()
    return X_test, y_test
