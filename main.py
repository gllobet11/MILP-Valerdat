"""
main.py — Supply Chain Optimization Pipeline
=============================================

Runs the full end-to-end pipeline:
  1. Generate synthetic data
  2. Build ML features
  3. Train LightGBM quantile forecast
  4. Generate weekly demand forecast for optimization horizon
  5. Solve MILP purchase optimization
  6. Report KPIs and save outputs

Usage
-----
    python main.py

Outputs
-------
  data/products.csv
  data/suppliers.csv
  data/supplier_products.csv
  data/historical_demand.csv
  data/forecast_output.csv
  data/purchase_plan.csv
  data/stock_plan.csv
"""

import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from src.data_generator import generate_all
from src.feature_engineering import build_features
from src.forecast import train_forecast, generate_forecast, compute_empirical_sigma
from src.optimizer import build_and_solve
from src.reporting import report_kpis

DATA_DIR = "data"
HORIZON_START = "2026-01-01"   # first day of the optimization month
N_WEEKS = 4                    # optimization horizon


def main():
    print("\n" + "=" * 65)
    print("  SUPPLY CHAIN OPTIMIZATION PIPELINE — Valerdat")
    print("=" * 65)

    # ── Step 1: Synthetic data ────────────────────────────────────────────
    print("\n[1/5] Generating synthetic data …")
    data = generate_all(DATA_DIR)
    products_df       = data["products"]
    suppliers_df      = data["suppliers"]
    supplier_products = data["supplier_products"]
    demand_df         = data["demand"]
    W                 = data["W"]

    # ── Step 2: Feature engineering ──────────────────────────────────────
    print("\n[2/5] Building ML features …")
    feature_df = build_features(demand_df)
    print(f"  Feature matrix: {feature_df.shape[0]:,} rows × {feature_df.shape[1]} cols")

    # ── Step 3: Train forecast ────────────────────────────────────────────
    print("\n[3/5] Training LightGBM quantile models …")
    models, val_metrics = train_forecast(feature_df, products_df, val_months=3)

    # ── Step 3.5: Empirical sigma para sigma híbrida ──────────────────────
    print("  [forecast] Computing empirical sigma from validation errors …")
    emp_sigma = compute_empirical_sigma(models, feature_df, val_months=3)

    # ── Step 4: Generate forecast for optimizer horizon ───────────────────
    print("\n[4/5] Forecasting demand for optimization horizon …")
    forecast_df = generate_forecast(
        models, feature_df, products_df,
        horizon_start=HORIZON_START,
        n_weeks=N_WEEKS,
        empirical_sigma=emp_sigma,
    )
    forecast_df.to_csv(f"{DATA_DIR}/forecast_output.csv", index=False)
    print(f"  Forecast saved → {DATA_DIR}/forecast_output.csv")
    print(f"  Sample (first 8 rows):\n{forecast_df.head(8).to_string(index=False)}")

    # ── Step 5: MILP optimization ─────────────────────────────────────────
    print("\n[5/5] Solving MILP …")
    purchase_plan, stock_plan, status = build_and_solve(
        forecast_df=forecast_df,
        products_df=products_df,
        suppliers_df=suppliers_df,
        supplier_products_df=supplier_products,
        W=W,
        time_limit_sec=300,
    )

    if status not in ("Optimal", "Not Solved"):
        print(f"  WARNING: Solver status is '{status}'. Results may be suboptimal.")

    purchase_plan.to_csv(f"{DATA_DIR}/purchase_plan.csv", index=False)
    stock_plan.to_csv(f"{DATA_DIR}/stock_plan.csv", index=False)
    print(f"  Purchase plan saved → {DATA_DIR}/purchase_plan.csv "
          f"({len(purchase_plan)} order lines)")

    # ── Reporting ──────────────────────────────────────────────────────────
    report_kpis(
        purchase_plan=purchase_plan,
        stock_plan=stock_plan,
        forecast_df=forecast_df,
        products_df=products_df,
        suppliers_df=suppliers_df,
        supplier_products_df=supplier_products,
        val_metrics=val_metrics,
    )

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
