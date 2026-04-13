# test_forecast.py — ejecutar desde ~/proyectos/Valerdat/
# python test_forecast.py
#
# Testa SOLO el módulo generate_forecast con los datos ya generados
# (asume que data/ existe de una run anterior de main.py).
# Compara el nuevo forecast recursivo contra el original estático
# y visualiza diferencias en demand_hat y safety_stock.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.feature_engineering import build_features
from src.forecast import train_forecast, generate_forecast, compute_empirical_sigma, FEATURE_COLS

DATA_DIR  = Path("data")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

HORIZON_START = "2026-01-01"
N_WEEKS = 4

# ── 1. Cargar datos ya existentes (no regenerar) ──────────────────────────────
print("[1/4] Loading existing data …")
products_df = pd.read_csv(DATA_DIR / "products.csv")
demand_df   = pd.read_csv(DATA_DIR / "historical_demand.csv", parse_dates=["date"])
print(f"  {len(products_df)} products · {len(demand_df):,} demand rows")

# ── 2. Features + train (necesario: los modelos no se persisten a disco) ──────
print("[2/4] Building features and training models …")
feature_df = build_features(demand_df)
models, val_metrics = train_forecast(feature_df, products_df, val_months=3)

# ── 3. Generar forecast con el nuevo módulo recursivo ─────────────────────────
print("[3/4] Running NEW recursive generate_forecast …")
empirical_sigma = compute_empirical_sigma(models, feature_df, val_months=3)

forecast_new = generate_forecast(
    models, feature_df, products_df,
    horizon_start=HORIZON_START,
    n_weeks=N_WEEKS,
    empirical_sigma=empirical_sigma,
)

# Cargar forecast anterior (guardado por main.py) para comparar
forecast_old = pd.read_csv(DATA_DIR / "forecast_output.csv")

print("\n── New forecast sample (first 12 rows) ──")
print(forecast_new.head(12).to_string(index=False))

# ── 4. Comparación y validación ───────────────────────────────────────────────
print("\n[4/4] Comparing old vs new …")

merged = forecast_new.merge(
    forecast_old[["product_id", "period_t", "demand_hat", "sigma", "safety_stock"]],
    on=["product_id", "period_t"],
    suffixes=("_new", "_old"),
)

# Métricas de diferencia
for col in ["demand_hat", "sigma", "safety_stock"]:
    diff = (merged[f"{col}_new"] - merged[f"{col}_old"]).abs()
    pct  = (diff / merged[f"{col}_old"].clip(lower=1e-6) * 100)
    print(f"  {col:15s}  mean_abs_diff={diff.mean():.2f}  mean_pct_diff={pct.mean():.1f}%  max_pct_diff={pct.max():.1f}%")

# Sanity checks
assert (forecast_new["demand_hat"] >= 0).all(),      "demand_hat has negatives"
assert (forecast_new["sigma"] >= 0.01).all(),         "sigma below floor"
assert (forecast_new["safety_stock"] >= 0).all(),     "safety_stock has negatives"
assert forecast_new["period_t"].isin([1,2,3,4]).all(),"unexpected period_t values"
assert forecast_new["product_id"].nunique() == len(products_df), "missing products"
print("\n  ✅ All sanity checks passed")

# ── SS empirical validation: ¿el SS cubre el p95 del error semanal? ─────────
print("\n── Empirical SS validation (coverage vs p95 weekly error) ──")
_cutoff = feature_df["date"].max() - pd.DateOffset(months=3)
_val = feature_df[feature_df["date"] > _cutoff].copy()
_val["pred_q50"] = models[0.50].predict(_val[FEATURE_COLS]).clip(0)
_val["week"] = _val["date"].dt.to_period("W")
weekly_err = _val.groupby(["product_id", "week"]).agg(
    actual=("demand", "sum"),
    pred=("pred_q50", "sum")
).reset_index()
weekly_err["error"] = weekly_err["actual"] - weekly_err["pred"]

p95_error = weekly_err.groupby("product_id")["error"].quantile(0.95)
coverage  = forecast_new.groupby("product_id")["safety_stock"].mean()

ratio = (coverage / p95_error.clip(lower=1)).rename("ss_coverage_ratio")
print(ratio.describe().round(2))
print("  (ratio > 1.0 → SS covers the p95 forecast error → target met)")

# ── SS coverage check: ¿el SS cubre al menos 1 semana de demanda promedio? ───
print("\n── Safety stock coverage (ss_weeks = safety_stock / demand_hat) ──")
print(forecast_new.assign(
    ss_weeks=lambda df: df["safety_stock"] / df["demand_hat"].clip(lower=1)
).groupby("period_t")["ss_weeks"].describe().round(2))

# ── 5. Plots de comparación ───────────────────────────────────────────────────
cat_map  = products_df.set_index("product_id")["category"]
name_map = products_df.set_index("product_id")["name"]
merged["category"] = merged["product_id"].map(cat_map)

PALETTE = {"electronics": "#4C72B0", "consumables": "#55A868", "industrial": "#C44E52"}

# Plot A: demand_hat new vs old — scatter por producto (semana 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("demand_hat: Recursive (new) vs Static (old) — Week 1", fontsize=13, fontweight="bold")

for ax, cat in zip(axes, ["electronics", "consumables", "industrial"]):
    sub = merged[(merged["category"] == cat) & (merged["period_t"] == 1)]
    ax.scatter(sub["demand_hat_old"], sub["demand_hat_new"],
               color=PALETTE[cat], alpha=0.7, s=50, edgecolors="white", linewidths=0.5)
    lim = max(sub["demand_hat_old"].max(), sub["demand_hat_new"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Static forecast (old)", fontsize=10)
    ax.set_ylabel("Recursive forecast (new)", fontsize=10)
    ax.set_title(cat.capitalize(), color=PALETTE[cat], fontweight="bold")
    ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(PLOTS_DIR / "test_forecast_scatter.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# Plot B: safety_stock new vs old — barras por categoría y semana
fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
fig.suptitle("Safety Stock: Recursive (new) vs Static (old) — by week", fontsize=13, fontweight="bold")

for ax, t in zip(axes, [1, 2, 3, 4]):
    sub = merged[merged["period_t"] == t].copy()
    sub["category"] = sub["product_id"].map(cat_map)
    grp = sub.groupby("category")[["safety_stock_new", "safety_stock_old"]].mean()
    x = np.arange(len(grp))
    w = 0.35
    ax.bar(x - w/2, grp["safety_stock_old"], w, label="Old", color="#888888", alpha=0.7)
    ax.bar(x + w/2, grp["safety_stock_new"], w, label="New", color=C if (C := "#00C2A8") else "#00C2A8", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(grp.index, fontsize=9)
    ax.set_title(f"Week {t}", fontsize=11)
    ax.set_ylabel("Avg SS (units)" if t == 1 else "")
    ax.grid(axis="y", alpha=0.3)
    if t == 1:
        ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(PLOTS_DIR / "test_forecast_safety_stock.png", dpi=130, bbox_inches="tight")
plt.close(fig)

print(f"\n  Plots saved → {PLOTS_DIR}/test_forecast_scatter.png")
print(f"  Plots saved → {PLOTS_DIR}/test_forecast_safety_stock.png")
print("\n✅ test_forecast.py complete — review plots/ for visual diff")