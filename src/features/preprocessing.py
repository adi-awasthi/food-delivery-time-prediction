import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.data_preparation import (
    load_cleaned_data,
    split_train_test,
    impute_person_columns,
)
from src.features.build_features import add_time_period_feature

TARGET_COLUMN = "Time_taken(min)"
DROP_COLUMNS = ["ID", "Delivery_person_ID", "order_datetime"]

# Ordered categories for genuinely ordinal columns. "Unknown" (the
# missing-value placeholder from Phase 2) is intentionally excluded from
# this order and mapped to a fixed -1 instead -- see encode_ordinal_columns.
ORDINAL_COLUMNS = {
    "Road_traffic_density": ["Low", "Medium", "High", "Jam"],
    "multiple_deliveries": ["0", "1", "2", "3"],
}

ONEHOT_COLUMNS = [
    "Weatherconditions",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
    "time_period",
]

# Restaurant_latitude/longitude and Delivery_location_latitude/longitude are
# deliberately excluded here. distance_km is a deterministic function of
# those 4 columns, so including both adds multicollinearity risk for the
# linear baseline and redundant splits for tree models with no new
# information. They're kept in data/interim/ for reference, just not fed
# into the ColumnTransformer's output features.
NUMERIC_COLUMNS = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "distance_km",
    "order_hour",
    "order_day_of_week",
    "Vehicle_condition",
]


def encode_ordinal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Manual dict-based mapping instead of sklearn's OrdinalEncoder, so
    "Unknown" can be pinned to a fixed code (-1) outside the real 0..n-1
    order -- sklearn would otherwise assign it the next sequential integer,
    placing it as if "beyond Jam" / "beyond 3 deliveries", which isn't what
    it means. This mapping is fixed domain knowledge, not fit from data, so
    it's safe to apply before or after the split with no leakage risk.
    """
    df = df.copy()
    for col, order in ORDINAL_COLUMNS.items():
        mapping = {category: rank for rank, category in enumerate(order)}
        df[f"{col}_ordinal"] = df[col].map(mapping).fillna(-1).astype(int)
    return df.drop(columns=list(ORDINAL_COLUMNS.keys()))


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y


def build_preprocessor() -> ColumnTransformer:
    scaled_columns = NUMERIC_COLUMNS + [f"{c}_ordinal" for c in ORDINAL_COLUMNS]
    return ColumnTransformer(
        transformers=[
            # StandardScaler here is for the Linear Regression baseline's
            # benefit (and any other coefficient/gradient-based model in the
            # stack) -- it needs comparably-scaled inputs to fit sensibly.
            # It's a no-op for Random Forest / LightGBM: tree splits only
            # compare a feature against thresholds of itself, so they're
            # scale-invariant and unaffected by this transform.
            ("scale", StandardScaler(), scaled_columns),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ONEHOT_COLUMNS,
            ),
        ]
    )


PREPROCESSOR_ARTIFACT_PATH = "models/preprocessor.joblib"
PROCESSED_TRAIN_PATH = "data/processed/train.csv"
PROCESSED_TEST_PATH = "data/processed/test.csv"


def run_preprocessing_pipeline(
    test_size: float = 0.2,
    random_state: int = 42,
    preprocessor_output_path: str = PREPROCESSOR_ARTIFACT_PATH,
    train_output_path: str = PROCESSED_TRAIN_PATH,
    test_output_path: str = PROCESSED_TEST_PATH,
):
    df = load_cleaned_data()
    train_df, test_df = split_train_test(df, test_size, random_state)

    # fit-on-train-only imputation, deferred from Phase 2
    train_df, test_df = impute_person_columns(train_df, test_df)

    # deterministic feature engineering / encoding, safe on both splits
    train_df = add_time_period_feature(train_df)
    test_df = add_time_period_feature(test_df)
    train_df = encode_ordinal_columns(train_df)
    test_df = encode_ordinal_columns(test_df)

    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out())

    joblib.dump(preprocessor, preprocessor_output_path)

    train_out = pd.DataFrame(X_train_processed, columns=feature_names)
    train_out[TARGET_COLUMN] = y_train.to_numpy()
    train_out.to_csv(train_output_path, index=False)

    test_out = pd.DataFrame(X_test_processed, columns=feature_names)
    test_out[TARGET_COLUMN] = y_test.to_numpy()
    test_out.to_csv(test_output_path, index=False)

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "train_df": train_df,
        "test_df": test_df,
    }


if __name__ == "__main__":
    run_preprocessing_pipeline()
