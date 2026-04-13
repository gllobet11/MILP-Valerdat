"""
Feature engineering for the LightGBM demand forecast.
Builds a flat feature DataFrame from raw daily demand history.
"""

import pandas as pd
import numpy as np


def build_features(demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame [date, product_id, demand], return a feature DataFrame
    ready for LightGBM training/inference.

    Rows with NaN (due to lags at series start) are dropped.
    """
    demand_df = demand_df.copy()
    demand_df["date"] = pd.to_datetime(demand_df["date"])
    demand_df = demand_df.sort_values(["product_id", "date"]).reset_index(drop=True)

    # ── Calendar features ────────────────────────────────────────────────────
    demand_df["day_of_week"] = demand_df["date"].dt.dayofweek          # 0=Mon
    demand_df["week_of_year"] = demand_df["date"].dt.isocalendar().week.astype(int)
    demand_df["month"] = demand_df["date"].dt.month
    demand_df["quarter"] = demand_df["date"].dt.quarter
    demand_df["is_weekend"] = (demand_df["day_of_week"] >= 5).astype(int)
    demand_df["day_of_year"] = demand_df["date"].dt.dayofyear

    # ── Lag & rolling features (computed per product) ─────────────────────--
    grp = demand_df.groupby("product_id")["demand"]

    for lag in [7, 14, 28]:
        demand_df[f"lag_{lag}"] = grp.shift(lag)

    for window in [7, 28]:
        demand_df[f"rolling_mean_{window}"] = (
            grp.shift(1).transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        demand_df[f"rolling_std_{window}"] = (
            grp.shift(1).transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        )

    # ── Product identity (LightGBM treats as categorical) ────────────────────
    demand_df["product_code"] = demand_df["product_id"].str.extract(r"(\d+)").astype(int)

    # Drop rows where lags are unavailable (first 28 days per product)
    feature_cols = [
        "product_code", "day_of_week", "week_of_year", "month",
        "quarter", "is_weekend", "day_of_year",
        "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_mean_28",
        "rolling_std_7", "rolling_std_28",
    ]
    demand_df = demand_df.dropna(subset=feature_cols).reset_index(drop=True)

    return demand_df[["date", "product_id"] + feature_cols + ["demand"]]
