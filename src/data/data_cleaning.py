import numpy as np
import pandas as pd
import yaml

RAW_DATA_PATH = "data/raw/swiggy.csv"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Weatherconditions stores every value with a redundant "conditions " prefix,
    # so its missing sentinel reads as "conditions NaN" rather than bare "NaN".
    # Strip the prefix here so the generic NaN check below also catches this
    # column instead of silently missing it.
    df["Weatherconditions"] = df["Weatherconditions"].str.replace(
        "conditions ", "", regex=False
    )

    # Strip leading/trailing whitespace from every string column. The source
    # data pads values inconsistently (e.g. "Jam " for Road_traffic_density),
    # which would otherwise stop both the NaN check below and later categorical
    # encoding from recognizing padded and unpadded values as the same category.
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    df[string_cols] = df[string_cols].apply(lambda col: col.str.strip())

    # The source data encodes missing values as the literal string "NaN " (with
    # a trailing space), not a real null -- pandas' default read_csv NA
    # detection only matches an exact "NaN", so that trailing space lets these
    # slip through as ordinary strings. Verified via a full unique-value audit
    # of every column that "NaN" is never a legitimate category value anywhere
    # in this dataset, so it's safe to treat every exact "NaN" string, in any
    # column, as a real missing value.
    df[string_cols] = df[string_cols].replace("NaN", np.nan)

    return df


def cast_person_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"])
    df["Delivery_person_Ratings"] = pd.to_numeric(df["Delivery_person_Ratings"])
    return df


def fit_age_imputation(df: pd.DataFrame) -> tuple[pd.Series, float]:
    """
    Computes per-Delivery_person_ID median age, plus a global median as a
    fallback for delivery persons with no known age at all.

    Call this on TRAIN data only once the split exists (Phase 4) -- fitting on
    the full dataset would leak test-set ages into the statistic used to fill
    train (and test) rows.
    """
    person_medians = df.groupby("Delivery_person_ID")["Delivery_person_Age"].median()
    global_median = df["Delivery_person_Age"].median()
    return person_medians, global_median


def apply_age_imputation(
    df: pd.DataFrame, person_medians: pd.Series, global_median: float
) -> pd.DataFrame:
    df = df.copy()
    missing = df["Delivery_person_Age"].isna()

    # first pass: fill from this delivery person's own known median age
    from_person = df.loc[missing, "Delivery_person_ID"].map(person_medians)
    df.loc[missing, "Delivery_person_Age"] = from_person

    # second pass: delivery persons with no known age anywhere -> global median
    still_missing = df["Delivery_person_Age"].isna()
    df.loc[still_missing, "Delivery_person_Age"] = global_median

    return df


def invalidate_rating_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratings are a 1-5 scale; a small number of rows record a 6, which is
    outside the valid range and can't be corrected to a specific true value.
    Treat these as missing so they flow through the same median imputation
    as genuinely missing ratings, rather than guessing a correction.
    """
    df = df.copy()
    df.loc[df["Delivery_person_Ratings"] == 6, "Delivery_person_Ratings"] = np.nan
    return df


def fit_ratings_imputation(df: pd.DataFrame) -> float:
    """
    Computes the global median rating. Call this on TRAIN data only once the
    split exists (Phase 4), same leakage reasoning as fit_age_imputation.
    """
    return df["Delivery_person_Ratings"].median()


def apply_ratings_imputation(df: pd.DataFrame, median_rating: float) -> pd.DataFrame:
    df = df.copy()
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].fillna(median_rating)
    return df


def fix_restaurant_coordinate_sign_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    A subset of rows have restaurant latitude AND longitude both sign-flipped
    (e.g. lat around -30.9 instead of 30.9). Taking abs() of both recovers
    coordinates that fall fully within India's bounding box for every affected
    row -- verified directly on this dataset, not assumed.
    """
    df = df.copy()
    sign_flipped = df["Restaurant_latitude"] < 0
    df.loc[sign_flipped, "Restaurant_latitude"] = df.loc[
        sign_flipped, "Restaurant_latitude"
    ].abs()
    df.loc[sign_flipped, "Restaurant_longitude"] = df.loc[
        sign_flipped, "Restaurant_longitude"
    ].abs()
    return df


def drop_null_island_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    A subset of rows have all four coordinates (restaurant + delivery,
    lat + lon) clustered near (0, 0) -- a placeholder for "no real location",
    not an actual point near India. There's no recoverable signal for the
    haversine distance feature on these rows, so they're dropped rather than
    imputed with a fabricated distance.
    """
    null_island = (
        (df["Restaurant_latitude"].abs() < 1)
        & (df["Restaurant_longitude"].abs() < 1)
        & (df["Delivery_location_latitude"].abs() < 1)
        & (df["Delivery_location_longitude"].abs() < 1)
    )
    return df.loc[~null_island].copy()


def normalize_city_spelling(df: pd.DataFrame) -> pd.DataFrame:
    """
    "Metropolitian" is a misspelling of "Metropolitan" present in the source
    data itself. This is a spelling correction on a category label, not an
    inference or imputation -- the typo doesn't introduce any ambiguity about
    what the category means, so correcting it just avoids carrying a typo
    into feature encoding later (which would otherwise treat it as a
    distinct category from "Metropolitan").
    """
    df = df.copy()
    df["City"] = df["City"].replace("Metropolitian", "Metropolitan")
    return df


CATEGORICAL_UNKNOWN_FILL_COLUMNS = [
    "Weatherconditions",
    "Road_traffic_density",
    "multiple_deliveries",
    "Festival",
    "City",
]


def fill_remaining_categorical_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in the remaining categorical columns with an
    explicit "Unknown" category, rather than deferring to preprocessing.
    This is a deterministic rule (not a learned statistic), so it carries no
    leakage risk and doesn't need to wait for the train/test split the way
    Age/Ratings imputation does. Preserving missingness as its own category
    (rather than guessing a value) also keeps whatever signal the
    missingness pattern itself might carry -- e.g. if a field tends to go
    unrecorded under specific conditions, a tree-based model can pick up on
    that "Unknown" category directly.
    """
    df = df.copy()
    for col in CATEGORICAL_UNKNOWN_FILL_COLUMNS:
        df[col] = df[col].fillna("Unknown")
    return df


def build_order_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines Order_Date with Time_Orderd into order_datetime. NaT wherever
    Time_Orderd is missing -- impute_missing_order_datetime fills those in a
    separate step.
    """
    df = df.copy()
    order_date = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y")
    df["order_datetime"] = order_date + pd.to_timedelta(df["Time_Orderd"])
    return df


def impute_missing_order_datetime(
    df: pd.DataFrame, gap_minutes: int
) -> pd.DataFrame:
    """
    Fills missing order_datetime from Time_Order_picked minus a fixed gap
    (gap_minutes, read from params.yaml -- data_cleaning.order_to_pickup_imputation_gap_minutes).
    This is not a generic "typical order-to-pickup time" assumption --
    across all 43,862 rows where both Time_Orderd and Time_Order_picked are
    known, the gap between them takes on exactly one of three values (5, 10,
    or 15 minutes), split almost evenly three ways, with no other value ever
    observed. Subtracting the median/most-central gap (10 min) is therefore
    correct to within +/-5 minutes on every row, not a rough average with a
    long tail.

    Order_Date always matches Time_Orderd's calendar day, not
    Time_Order_picked's -- confirmed on real rows where pickup happens just
    after midnight (e.g. Order_Date=05-03-2022, Time_Orderd=23:50:00,
    Time_Order_picked=00:05:00; the pickup is really on 06-03-2022, one day
    after Order_Date). So the imputed time is computed in time-of-day space
    (mod 24h) and then re-attached to Order_Date directly. Naively pinning
    Time_Order_picked to Order_Date's day first and then subtracting the gap
    would roll the date back an extra day whenever pickup wrapped past
    midnight, which is exactly the bug this avoids.
    """
    df = df.copy()
    missing = df["order_datetime"].isna()

    order_date_midnight = pd.to_datetime(
        df.loc[missing, "Order_Date"], format="%d-%m-%Y"
    )
    picked_time_of_day = pd.to_timedelta(df.loc[missing, "Time_Order_picked"])
    gap = pd.to_timedelta(gap_minutes, unit="m")
    order_time_of_day = (picked_time_of_day - gap) % pd.to_timedelta(1, unit="D")

    df.loc[missing, "order_datetime"] = order_date_midnight + order_time_of_day
    return df


RAW_DATETIME_SOURCE_COLUMNS = ["Order_Date", "Time_Orderd", "Time_Order_picked"]


def drop_raw_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order_Date, Time_Orderd, and Time_Order_picked are fully superseded by
    order_datetime, order_hour, and order_day_of_week with no information
    loss. Dropping them avoids carrying redundant raw columns whose original
    missingness (e.g. Time_Orderd) no longer reflects anything, since the
    derived columns are already complete.
    """
    return df.drop(columns=RAW_DATETIME_SOURCE_COLUMNS)


def clean_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time_taken(min) arrives as a string like "(min) 24". Strip the "(min) "
    prefix and cast to int to get the actual regression target.
    """
    df = df.copy()
    df["Time_taken(min)"] = (
        df["Time_taken(min)"].str.replace("(min) ", "", regex=False).astype(int)
    )
    return df


EARTH_RADIUS_KM = 6371.0


def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Haversine great-circle distance (km) between the restaurant and delivery
    point. Purely a deterministic function of the 4 coordinate columns -- no
    fitting involved, so safe to compute before the train/test split.
    """
    df = df.copy()
    lat1 = np.radians(df["Restaurant_latitude"])
    lon1 = np.radians(df["Restaurant_longitude"])
    lat2 = np.radians(df["Delivery_location_latitude"])
    lon2 = np.radians(df["Delivery_location_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    df["distance_km"] = EARTH_RADIUS_KM * c
    return df


def add_time_of_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hour of day and day of week the order was placed, read off order_datetime.
    Deterministic transforms of an already-complete column -- no fitting
    involved, so safe to compute before the train/test split.
    """
    df = df.copy()
    df["order_hour"] = df["order_datetime"].dt.hour
    df["order_day_of_week"] = df["order_datetime"].dt.dayofweek
    return df


INTERIM_DATA_PATH = "data/interim/swiggy_cleaned.csv"
PARAMS_PATH = "params.yaml"


def run_cleaning_pipeline(
    input_path: str = RAW_DATA_PATH,
    output_path: str = INTERIM_DATA_PATH,
    params_path: str = PARAMS_PATH,
) -> pd.DataFrame:
    """
    Pre-split-safe cleaning pipeline: everything here is either a
    deterministic transform or a decision that doesn't require fitting a
    statistic on train-only data. Age/Ratings imputation is deliberately
    excluded -- those are fit on train and applied in Phase 4, so this output
    still has missing values in Delivery_person_Age and
    Delivery_person_Ratings by design.
    """
    with open(params_path) as f:
        params = yaml.safe_load(f)
    gap_minutes = params["data_cleaning"]["order_to_pickup_imputation_gap_minutes"]

    df = load_raw_data(input_path)
    df = normalize_missing_values(df)
    df = cast_person_numeric_columns(df)
    df = invalidate_rating_outliers(df)
    df = fix_restaurant_coordinate_sign_errors(df)
    df = drop_null_island_rows(df)
    df = normalize_city_spelling(df)
    df = fill_remaining_categorical_missing(df)
    df = build_order_datetime(df)
    df = impute_missing_order_datetime(df, gap_minutes)
    df = add_distance_feature(df)
    df = add_time_of_day_features(df)
    df = drop_raw_datetime_columns(df)
    df = clean_target_column(df)

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    run_cleaning_pipeline()
