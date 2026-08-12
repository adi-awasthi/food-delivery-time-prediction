import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.data_cleaning import (
    fit_age_imputation,
    apply_age_imputation,
    fit_ratings_imputation,
    apply_ratings_imputation,
)

INTERIM_DATA_PATH = "data/interim/swiggy_cleaned.csv"


def load_cleaned_data(path: str = INTERIM_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["order_datetime"])


def split_train_test(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, random_state=random_state)


def impute_person_columns(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fits the Age/Ratings imputation statistics on train only, then applies
    them to both train and test. This is the step Phase 2 deliberately
    deferred fit_age_imputation/fit_ratings_imputation for -- computing these
    medians on the full dataset (pre-split) would leak test-set values into
    the statistic used to fill missing train (and test) rows.
    """
    person_medians, global_age_median = fit_age_imputation(train_df)
    train_df = apply_age_imputation(train_df, person_medians, global_age_median)
    test_df = apply_age_imputation(test_df, person_medians, global_age_median)

    rating_median = fit_ratings_imputation(train_df)
    train_df = apply_ratings_imputation(train_df, rating_median)
    test_df = apply_ratings_imputation(test_df, rating_median)

    return train_df, test_df
