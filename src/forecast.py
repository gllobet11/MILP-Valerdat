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


def compute_empirical_sigma(
    models: dict,
    feature_df: pd.DataFrame,
    val_months: int = 3,
) -> pd.Series:
    """
    Sigma empírica semanal del error de forecast en validación.
    Autocontenida: genera las predicciones internamente para no depender
    de que el caller haya añadido pred_q50 previamente.
    Más robusta que la sigma cuantílica para SKUs de alta varianza.
    """
    feature_df = feature_df.copy()
    feature_df["date"] = pd.to_datetime(feature_df["date"])

    cutoff = feature_df["date"].max() - pd.DateOffset(months=val_months)
    val_df = feature_df[feature_df["date"] > cutoff].copy()

    val_df["pred_q50"] = models[0.50].predict(val_df[FEATURE_COLS]).clip(0)

    val_df["week"] = val_df["date"].dt.to_period("W")
    weekly_err = (
        val_df.groupby(["product_id", "week"])
        .agg(actual=("demand", "sum"), pred=("pred_q50", "sum"))
        .reset_index()
    )
    weekly_err["error"] = weekly_err["actual"] - weekly_err["pred"]
    return weekly_err.groupby("product_id")["error"].std().rename("sigma_empirical")


def generate_forecast(
    models: dict,
    feature_df: pd.DataFrame,
    products_df: pd.DataFrame,
    horizon_start: str,
    n_weeks: int = 4,
    empirical_sigma: pd.Series = None,
) -> pd.DataFrame:
    """
    Generate weekly forecasts for the optimization horizon.

    Strategy: recursive day-ahead forecasting per product. Each day's
    prediction is appended to the working history so that lag and rolling
    features for subsequent days reflect previously predicted values rather
    than stale last-known values.

    Fixes applied vs. original:
      1. Recursive inference: lags/rolling recomputed each step from the
         growing history (predicted values feed future lags).
      2. Sigma aggregated correctly: computed per day, then combined via
         quadrature sum (sqrt of sum of squares) — quantiles are not additive.
      3. demand_hat clipped to 0, not 0.1 (MOQ in MILP handles minimum orders).

    Returns
    -------
    forecast_df : DataFrame[product_id, period_t, demand_hat, sigma, safety_stock]
    """
    feature_df = feature_df.copy()
    feature_df["date"] = pd.to_datetime(feature_df["date"])
    horizon_start_dt = pd.Timestamp(horizon_start)

    horizon_dates = pd.date_range(horizon_start_dt, periods=n_weeks * 7, freq="D")
    week_map = {d: (i // 7) + 1 for i, d in enumerate(horizon_dates)}

    # Lead time lookup
    lt_map = products_df.set_index("product_id")["lead_time_weeks"]

    all_records = []

    for pid, grp in feature_df.groupby("product_id"):
        grp = grp.sort_values("date").reset_index(drop=True)

        # Working history: real demand column only — we'll recompute features
        # from it at each step. Keep enough tail to cover the longest lag (28d)
        # plus the longest rolling window (28d) → 56 rows is safe.
        history = grp[["date", "product_id", "demand", "product_code"]].tail(60).copy()

        product_code = int(grp["product_code"].iloc[-1])

        for date in horizon_dates:
            # ── Calendar features (deterministic) ────────────────────────
            row = {
                "product_code": product_code,
                "day_of_week":  date.dayofweek,
                "week_of_year": date.isocalendar()[1],
                "month":        date.month,
                "quarter":      date.quarter,
                "is_weekend":   int(date.dayofweek >= 5),
                "day_of_year":  date.timetuple().tm_yday,
            }

            # ── Lag features from rolling history ────────────────────────
            demand_series = history["demand"].values  # ordered oldest → newest

            def _lag(n: int) -> float:
                """demand n days ago; NaN if not enough history."""
                return float(demand_series[-n]) if len(demand_series) >= n else np.nan

            def _rolling_mean(window: int) -> float:
                tail = demand_series[-window:] if len(demand_series) >= window else demand_series
                return float(np.mean(tail))

            def _rolling_std(window: int) -> float:
                tail = demand_series[-window:] if len(demand_series) >= window else demand_series
                return float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0

            row["lag_7"]          = _lag(7)
            row["lag_14"]         = _lag(14)
            row["lag_28"]         = _lag(28)
            row["rolling_mean_7"] = _rolling_mean(7)
            row["rolling_mean_28"]= _rolling_mean(28)
            row["rolling_std_7"]  = _rolling_std(7)
            row["rolling_std_28"] = _rolling_std(28)

            # ── Predict all three quantiles ───────────────────────────────
            X = pd.DataFrame([row])[FEATURE_COLS]
            q10 = float(np.clip(models[0.10].predict(X)[0], 0, None))
            q50 = float(np.clip(models[0.50].predict(X)[0], 0, None))
            q90 = float(np.clip(models[0.90].predict(X)[0], 0, None))

            # Sigma at daily level (correct: before aggregation)
            sigma_daily = (q90 - q10) / (2 * Z_ALPHA)

            all_records.append({
                "date":        date,
                "product_id":  pid,
                "period_t":    week_map[date],
                "q50":         q50,
                "q10":         q10,
                "q90":         q90,
                "sigma_daily": sigma_daily,
            })

            # ── Append q50 prediction to history for next step ───────────
            new_row = pd.DataFrame([{
                "date":         date,
                "product_id":   pid,
                "demand":       q50,   # predicted value feeds future lags
                "product_code": product_code,
            }])
            history = pd.concat([history, new_row], ignore_index=True)

    pred_df = pd.DataFrame(all_records)

    # ── Aggregate to weekly buckets ───────────────────────────────────────
    weekly = (
        pred_df.groupby(["product_id", "period_t"])
        .agg(
            demand_hat  =("q50",         "sum"),
            q10         =("q10",         "sum"),   # for reference only
            q90         =("q90",         "sum"),   # for reference only
            # FIX 2: quadrature sum — sigma of a sum of independent r.v.s
            # is sqrt(sum of variances), not sum of sigmas
            sigma       =("sigma_daily", lambda x: float(np.sqrt((x ** 2).sum()))),
        )
        .reset_index()
    )

    # FIX 3: clip to 0, not 0.1 — MOQ constraints in the MILP handle minimums
    weekly["demand_hat"] = weekly["demand_hat"].clip(lower=0)
    weekly["sigma"]      = weekly["sigma"].clip(lower=0.01)

    # ── Hybrid sigma: max(quantile sigma, empirical sigma) ────────────────
    if empirical_sigma is not None:
        weekly["sigma_empirical"] = weekly["product_id"].map(empirical_sigma).fillna(0)
        weekly["sigma"] = weekly[["sigma", "sigma_empirical"]].max(axis=1)
        weekly["sigma"] = weekly["sigma"].clip(lower=0.01)

    # ── Dynamic safety stock ──────────────────────────────────────────────
    weekly["lead_time"]    = weekly["product_id"].map(lt_map)
    weekly["safety_stock"] = (
        Z_ALPHA * weekly["sigma"] * np.sqrt(weekly["lead_time"])
    ).clip(lower=0)

    return weekly[["product_id", "period_t", "demand_hat", "sigma", "safety_stock"]]