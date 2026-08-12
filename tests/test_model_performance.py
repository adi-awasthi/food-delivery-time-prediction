import mlflow
from sklearn.metrics import mean_absolute_error


def test_staging_model_meets_mae_threshold(
    registered_model_name, promotion_threshold, test_data
):
    """The actual promotion gate: the staging-aliased model's test MAE must
    not exceed evaluation.promotion_mae_threshold from params.yaml -- read
    from there, not duplicated as a literal here, so this test and
    register_model.py's promote_to_production() can never drift apart."""
    X_test, y_test = test_data
    model_uri = f"models:/{registered_model_name}@staging"
    model = mlflow.pyfunc.load_model(model_uri)

    predictions = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, predictions)

    assert test_mae <= promotion_threshold, (
        f"test_mae {test_mae:.4f} exceeds promotion threshold {promotion_threshold}"
    )
