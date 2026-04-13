"""
Demand forecast module.
Trains three global LightGBM quantile models (q10, q50, q90) on historical
daily demand, then aggregates predictions to weekly periods for the optimizer.

Safety stock formula:
    SS_i = z_alpha * sigma_i * sqrt(LT_i)
where sigma_i = (q90 - q10) / (2 * 1.645), z_alpha = 1.645 (95% SL).
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error


QUANTILES = [0.10, 0.50, 0.90]
Z_ALPHA = 1.645  # 95% service level SL = Cstockout/ (Cstockout+ Cholding)

FEATURE_COLS = [
    "product_code", "day_of_week", "week_of_year", "month",
    "quarter", "is_weekend", "day_of_year",
    "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28",
    "rolling_std_7", "rolling_std_28",
]

LGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
}


def train_forecast(
    feature_df: pd.DataFrame,
    products_df: pd.DataFrame,
    val_months: int = 3,
) -> tuple[dict, pd.DataFrame]:
    """
    Train quantile LightGBM models.

    Returns
    -------
    models : dict {quantile -> fitted LGBMRegressor}
    val_metrics : DataFrame with per-product MAPE on validation set
    """
    feature_df = feature_df.copy()
    feature_df["date"] = pd.to_datetime(feature_df["date"])

    cutoff = feature_df["date"].max() - pd.DateOffset(months=val_months)
    train_mask = feature_df["date"] <= cutoff

    X_train = feature_df.loc[train_mask, FEATURE_COLS]
    y_train = feature_df.loc[train_mask, "demand"]
    X_val   = feature_df.loc[~train_mask, FEATURE_COLS]
    y_val   = feature_df.loc[~train_mask, "demand"]

    models = {}
    for q in QUANTILES:
        print(f"  [forecast] Training q={q:.2f} model …")
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **LGB_PARAMS)
        m.fit(X_train, y_train)
        models[q] = m

    # ── Validation metrics ────────────────────────────────────────────────
    val_df = feature_df.loc[~train_mask].copy()
    val_df["pred_q50"] = models[0.50].predict(X_val)
    val_df["pred_q50"] = val_df["pred_q50"].clip(lower=0)

    # MAPE per product (avoid division by zero)
    val_metrics = (
        val_df[val_df["demand"] > 0]
        .groupby("product_id")
        .apply(
            lambda g: mean_absolute_percentage_error(g["demand"], g["pred_q50"]),
            include_groups=False,
        )
        .reset_index(name="mape")
    )
    avg_mape = val_metrics["mape"].mean()
    print(f"  [forecast] Validation avg MAPE (q50): {avg_mape:.1%}")

    return models, val_metrics


def generate_forecast(
    models: dict,
    feature_df: pd.DataFrame,
    products_df: pd.DataFrame,
    horizon_start: str,
    n_weeks: int = 4,
) -> pd.DataFrame:
    """
    Generate weekly forecasts for the optimization horizon.

    Strategy: use the last known lag/rolling values (from the most recent
    available rows) to predict one day at a time for each product, then
    aggregate to weekly buckets.

    Returns
    -------
    forecast_df : [product_id, period_t, demand_hat, sigma, safety_stock]
    """
    feature_df = feature_df.copy()
    feature_df["date"] = pd.to_datetime(feature_df["date"])
    horizon_start_dt = pd.Timestamp(horizon_start)

    # Build per-product prediction rows for each day of the horizon
    horizon_dates = pd.date_range(horizon_start_dt, periods=n_weeks * 7, freq="D")
    week_map = {d: (i // 7) + 1 for i, d in enumerate(horizon_dates)}

    records = []
    for pid, grp in feature_df.groupby("product_id"):
        grp = grp.sort_values("date")
        last_known = grp.iloc[-1]  # most recent feature row

        for date in horizon_dates:
            # Calendar features
            row = {
                "product_code": last_known["product_code"],
                "day_of_week": date.dayofweek,
                "week_of_year": date.isocalendar()[1],
                "month": date.month,
                "quarter": date.quarter,
                "is_weekend": int(date.dayofweek >= 5),
                "day_of_year": date.timetuple().tm_yday,
                # reuse last-known lag/rolling (reasonable for short horizon)
                "lag_7": last_known["lag_7"],
                "lag_14": last_known["lag_14"],
                "lag_28": last_known["lag_28"],
                "rolling_mean_7": last_known["rolling_mean_7"],
                "rolling_mean_28": last_known["rolling_mean_28"],
                "rolling_std_7": last_known["rolling_std_7"],
                "rolling_std_28": last_known["rolling_std_28"],
            }
            records.append({"date": date, "product_id": pid, **row})

    pred_df = pd.DataFrame(records)
    X_pred = pred_df[FEATURE_COLS]

    pred_df["q10"] = np.clip(models[0.10].predict(X_pred), 0, None)
    pred_df["q50"] = np.clip(models[0.50].predict(X_pred), 0, None)
    pred_df["q90"] = np.clip(models[0.90].predict(X_pred), 0, None)
    pred_df["period_t"] = pred_df["date"].map(week_map)

    # ── Aggregate to weekly totals ────────────────────────────────────────
    weekly = (
        pred_df.groupby(["product_id", "period_t"])
        .agg(demand_hat=("q50", "sum"), q10=("q10", "sum"), q90=("q90", "sum"))
        .reset_index()
    )
    weekly["demand_hat"] = weekly["demand_hat"].clip(lower=0.1)

    # sigma from quantile spread
    weekly["sigma"] = (weekly["q90"] - weekly["q10"]) / (2 * Z_ALPHA)
    weekly["sigma"] = weekly["sigma"].clip(lower=0.01)

    # merge lead time
    lt = products_df.set_index("product_id")["lead_time_weeks"]
    weekly["lead_time"] = weekly["product_id"].map(lt)

    # dynamic safety stock
    weekly["safety_stock"] = (
        Z_ALPHA * weekly["sigma"] * np.sqrt(weekly["lead_time"])
    ).clip(lower=0)

    return weekly[["product_id", "period_t", "demand_hat", "sigma", "safety_stock"]]
