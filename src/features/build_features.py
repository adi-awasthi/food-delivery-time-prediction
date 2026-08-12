import pandas as pd


def add_time_period_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buckets order_hour into a coarse time-of-day category using conventional
    day-part boundaries (not fitted from data, so safe pre- or post-split).
    Kept alongside order_hour rather than replacing it -- EDA found a
    non-monotonic pattern (a dip at 15-16 between two higher periods) that a
    4-bucket split can't fully preserve on its own.
    """
    df = df.copy()
    bins = [-1, 5, 11, 16, 23]
    labels = ["Night", "Morning", "Afternoon", "Evening"]
    df["time_period"] = pd.cut(df["order_hour"], bins=bins, labels=labels)
    return df
