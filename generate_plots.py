"""
generate_plots.py — Visualization suite for the supply chain optimization pipeline.

Saves all plots to plots/ directory.

Usage
-----
    python generate_plots.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import lightgbm as lgb

from src.feature_engineering import build_features
from src.forecast import train_forecast, FEATURE_COLS

# ── Setup ─────────────────────────────────────────────────────────────────────
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

PALETTE = {
    "electronics": "#4C72B0",
    "consumables":  "#55A868",
    "industrial":   "#C44E52",
}
CAT_ORDER = ["electronics", "consumables", "industrial"]

# ── Load data ─────────────────────────────────────────────────────────────────
products  = pd.read_csv("data/products.csv")
suppliers = pd.read_csv("data/suppliers.csv")
sp        = pd.read_csv("data/supplier_products.csv")
demand    = pd.read_csv("data/historical_demand.csv", parse_dates=["date"])
forecast  = pd.read_csv("data/forecast_output.csv")
purchase  = pd.read_csv("data/purchase_plan.csv")
stock     = pd.read_csv("data/stock_plan.csv")

W        = 4_777.0
vol_map  = products.set_index("product_id")["unit_volume"]
cat_map  = products.set_index("product_id")["category"]
name_map = products.set_index("product_id")["name"]

feature_df = build_features(demand)
models, val_metrics = train_forecast(feature_df, products, val_months=3)

val_metrics = val_metrics.merge(products[["product_id","category"]], on="product_id")
avg_wk      = demand.groupby("product_id")["demand"].mean() * 7


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_feature_distributions():
    """
    Grid of histograms for engineered features.

    Lag and rolling features are NORMALISED by each product's own rolling_mean_28
    so that the distribution shows *relative* variation (ratio vs. baseline) rather
    than raw units. Without normalisation, mixing products with 15 units/day
    (laptops) and 300 units/day (paper) creates a heavily right-skewed, multi-modal
    blob that is hard to interpret. After normalisation, values cluster around 1.0
    and the shape reflects the true demand variability signal.

    Calendar features are shown as-is (they are already dimensionless).
    """
    fd = feature_df.copy()
    fd["category"] = fd["product_id"].map(cat_map)

    # Normalise lag / rolling features by each row's rolling_mean_28 (proxy baseline)
    norm_cols = ["lag_7", "lag_14", "lag_28",
                 "rolling_mean_7", "rolling_mean_28",
                 "rolling_std_7", "rolling_std_28"]
    cal_cols  = ["day_of_week", "week_of_year", "month", "is_weekend", "day_of_year"]

    eps = 1e-6
    for col in norm_cols:
        fd[col + "_norm"] = fd[col] / (fd["rolling_mean_28"] + eps)

    feat_plot = [(c + "_norm", lbl, True) for c, lbl in [
        ("lag_7",          "Lag 7d / baseline"),
        ("lag_14",         "Lag 14d / baseline"),
        ("lag_28",         "Lag 28d / baseline"),
        ("rolling_mean_7", "Roll. Mean 7d / baseline"),
        ("rolling_mean_28","Roll. Mean 28d / baseline"),
        ("rolling_std_7",  "Roll. Std 7d / baseline"),
        ("rolling_std_28", "Roll. Std 28d / baseline"),
    ]] + [(c, lbl, False) for c, lbl in [
        ("day_of_week",  "Day of Week (0=Mon)"),
        ("week_of_year", "Week of Year"),
        ("month",        "Month"),
        ("is_weekend",   "Is Weekend"),
        ("day_of_year",  "Day of Year"),
    ]]

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(
        "Feature Distributions by Category\n"
        "(lag/rolling shown as ratio to each product's 28-day rolling mean — "
        "removes product-scale effect)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax, (col, lbl, is_norm) in zip(axes.flat, feat_plot):
        for cat in CAT_ORDER:
            sub = fd[fd["category"] == cat][col].dropna()
            if is_norm:
                # clip at [0, 4] — outlier spikes (promos) are shown but not dominant
                sub = sub.clip(0, 4)
                bins = 40
            else:
                bins = min(40, sub.nunique())
            ax.hist(sub, bins=bins, alpha=0.55, label=cat,
                    color=PALETTE[cat], edgecolor="none", density=True)

        if is_norm:
            ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_xlabel("Ratio (1.0 = baseline)", fontsize=8)
        ax.set_title(lbl, fontsize=10)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=8)

    # Hide the unused 12th panel
    axes.flat[-1].set_visible(False)

    handles = [plt.Rectangle((0,0),1,1, color=PALETTE[c], alpha=0.7) for c in CAT_ORDER]
    fig.legend(handles, CAT_ORDER, loc="lower center", ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    path = PLOTS_DIR / "01_feature_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DEMAND DISTRIBUTION BY CATEGORY
# ═══════════════════════════════════════════════════════════════════════════════

def plot_demand_distributions():
    """
    Box plots of weekly demand per product, grouped by category.
    """
    weekly = demand.copy()
    weekly["week"] = weekly["date"].dt.to_period("W")
    wk = weekly.groupby(["product_id","week"])["demand"].sum().reset_index()
    wk["category"] = wk["product_id"].map(cat_map)
    wk["name"]     = wk["product_id"].map(name_map)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.suptitle("Weekly Demand Distribution by Category", fontsize=14, fontweight="bold")

    for ax, cat in zip(axes, CAT_ORDER):
        sub = wk[wk["category"] == cat]
        pids = sub.groupby("product_id")["demand"].median().sort_values().index.tolist()
        data = [sub[sub["product_id"] == p]["demand"].values for p in pids]
        names = [name_map[p] for p in pids]

        bp = ax.boxplot(data, vert=False, patch_artist=True,
                        flierprops=dict(marker=".", markersize=2, alpha=0.3),
                        medianprops=dict(color="white", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(PALETTE[cat])
            patch.set_alpha(0.75)

        ax.set_yticks(range(1, len(names) + 1))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Weekly demand (units)", fontsize=10)
        ax.set_title(cat.capitalize(), fontsize=12, color=PALETTE[cat], fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.tight_layout()
    path = PLOTS_DIR / "02_demand_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DEMAND SEASONALITY
# ═══════════════════════════════════════════════════════════════════════════════

def plot_seasonality():
    """
    Avg demand by day-of-week and by month, split by category.
    """
    d = demand.copy()
    d["category"]    = d["product_id"].map(cat_map)
    d["day_of_week"] = d["date"].dt.dayofweek
    d["month"]       = d["date"].dt.month

    dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Demand Seasonality Patterns", fontsize=14, fontweight="bold")

    # Day-of-week
    ax = axes[0]
    for cat in CAT_ORDER:
        sub = d[d["category"] == cat].groupby("day_of_week")["demand"].mean()
        ax.plot(sub.index, sub.values, marker="o", label=cat,
                color=PALETTE[cat], linewidth=2, markersize=6)
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_labels)
    ax.set_title("Average demand by day of week", fontsize=12)
    ax.set_ylabel("Avg daily demand (units)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Month
    ax = axes[1]
    for cat in CAT_ORDER:
        sub = d[d["category"] == cat].groupby("month")["demand"].mean()
        ax.plot(sub.index, sub.values, marker="o", label=cat,
                color=PALETTE[cat], linewidth=2, markersize=6)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, rotation=45)
    ax.set_title("Average demand by month", fontsize=12)
    ax.set_ylabel("Avg daily demand (units)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / "03_demand_seasonality.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LGBM FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance():
    """
    Gain-based feature importance for the q50 model.
    """
    model = models[0.50]
    imp = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=FEATURE_COLS,
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#4C72B0" if "lag" in f or "rolling" in f
              else "#55A868" if f == "product_code"
              else "#C44E52"
              for f in imp.index]
    ax.barh(imp.index, imp.values, color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("Feature importance (gain)", fontsize=11)
    ax.set_title("LightGBM Feature Importance — q50 model", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Legend
    legend_items = [
        plt.Rectangle((0,0),1,1, color="#4C72B0", label="Lag / Rolling features"),
        plt.Rectangle((0,0),1,1, color="#55A868", label="Product identity"),
        plt.Rectangle((0,0),1,1, color="#C44E52", label="Calendar features"),
    ]
    ax.legend(handles=legend_items, fontsize=9, loc="lower right")

    plt.tight_layout()
    path = PLOTS_DIR / "04_feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FORECAST ACCURACY (MAPE DISTRIBUTION)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_mape_distribution():
    """
    MAPE distribution as violin + strip plot, one panel per category.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    fig.suptitle("Forecast Accuracy — MAPE Distribution by Category\n(validation set, last 3 months)",
                 fontsize=13, fontweight="bold")

    for ax, cat in zip(axes, CAT_ORDER):
        sub = val_metrics[val_metrics["category"] == cat]["mape"] * 100
        parts = ax.violinplot(sub, positions=[0], showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(PALETTE[cat])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(2)

        # Jitter strip
        jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(sub))
        ax.scatter(jitter, sub.values, s=30, color=PALETTE[cat],
                   alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)

        ax.axhline(30, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(0.38, 31, "30% threshold", fontsize=8, color="grey", va="bottom")

        ax.set_title(f"{cat.capitalize()}\nmedian {sub.median():.1f}%", fontsize=11,
                     color=PALETTE[cat], fontweight="bold")
        ax.set_ylabel("MAPE (%)", fontsize=10)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(-0.4, 0.8)

    plt.tight_layout()
    path = PLOTS_DIR / "05_mape_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FORECAST vs ACTUAL — SAMPLE PRODUCTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_forecast_samples():
    """
    Actual vs predicted (q10/q50/q90) for 6 representative products
    over the validation period.
    """
    cutoff = feature_df["date"].max() - pd.DateOffset(months=3)
    val_fd = feature_df[feature_df["date"] > cutoff].copy()

    X_val = val_fd[FEATURE_COLS]
    val_fd["q10"] = np.clip(models[0.10].predict(X_val), 0, None)
    val_fd["q50"] = np.clip(models[0.50].predict(X_val), 0, None)
    val_fd["q90"] = np.clip(models[0.90].predict(X_val), 0, None)

    # Pick 2 products per category with median MAPE
    sample_pids = []
    for cat in CAT_ORDER:
        cat_vm = val_metrics[val_metrics["category"] == cat].sort_values("mape")
        mid = len(cat_vm) // 2
        sample_pids += cat_vm.iloc[mid - 1: mid + 1]["product_id"].tolist()

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Forecast vs Actual — Sample Products (Validation Period)",
                 fontsize=14, fontweight="bold")

    for ax, pid in zip(axes.flat, sample_pids):
        sub  = val_fd[val_fd["product_id"] == pid].sort_values("date")
        cat  = cat_map[pid]
        mape = val_metrics[val_metrics["product_id"] == pid]["mape"].values[0]

        ax.fill_between(sub["date"], sub["q10"], sub["q90"],
                        alpha=0.25, color=PALETTE[cat], label="80% interval (q10–q90)")
        ax.plot(sub["date"], sub["demand"], color="black", linewidth=0.8,
                alpha=0.8, label="Actual")
        ax.plot(sub["date"], sub["q50"], color=PALETTE[cat], linewidth=1.5,
                label="Forecast q50")
        ax.set_title(f"{name_map[pid]}  [{cat}]  MAPE={mape:.1%}", fontsize=10)
        ax.set_ylabel("Daily demand (units)", fontsize=9)
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        ax.grid(alpha=0.3)

    axes.flat[0].legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    path = PLOTS_DIR / "06_forecast_samples.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PURCHASE PLAN — ORDERS BY SUPPLIER × WEEK
# ═══════════════════════════════════════════════════════════════════════════════

def plot_purchase_plan():
    """
    Stacked bar: total spend per week, stacked by supplier.
    + Secondary axis: number of products ordered per week.

    Supplier profiles:
      S0 – Alpha  domestic,   fast (1-2w LT),  EUR 150 fixed cost
      S1 – Beta   regional,   fast (1-2w LT),  EUR  80 fixed cost (cheapest)
      S2 – Gamma  international, slow (2-4w LT), EUR 200 fixed cost
      S3 – Delta  specialist, medium (1-3w LT), EUR 120 fixed cost

    Why week 2 dominates:
      Initial stock covers ~2 weeks of demand, so week 1 barely needs restocking.
      By week 2 the stock has depleted one full week; the optimizer then bulk-orders
      for weeks 2-4 in one pass to minimise fixed logistics costs. The warehouse
      capacity constraint (W) caps this at 99% utilisation and forces the residual
      to spill into weeks 3 and 4.
    """
    sup_colors = {"S0": "#4C72B0", "S1": "#55A868", "S2": "#C44E52", "S3": "#DD8452"}
    sup_labels = {
        "S0": "Alpha  (domestic,  LT 1-2w, fix €150)",
        "S1": "Beta   (regional,  LT 1-2w, fix €80)",
        "S2": "Gamma  (intl,      LT 2-4w, fix €200)",
        "S3": "Delta  (specialist,LT 1-3w, fix €120)",
    }

    pivot = (
        purchase.groupby(["period_t","supplier_id"])["order_cost"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=["S0","S1","S2","S3"], fill_value=0)
    )
    n_prods = purchase.groupby("period_t")["product_id"].nunique()

    fig, ax1 = plt.subplots(figsize=(11, 6))
    fig.suptitle(
        "Monthly Purchase Plan — Spend by Supplier & Week\n"
        "Week 2 spike: bulk replenishment after week-1 depletion, capped by warehouse capacity (W)",
        fontsize=12, fontweight="bold",
    )

    bottom = np.zeros(len(pivot))
    for sup in ["S0","S1","S2","S3"]:
        vals = pivot[sup].values / 1_000
        ax1.bar(pivot.index, vals, bottom=bottom,
                color=sup_colors[sup], label=sup_labels[sup],
                edgecolor="white", linewidth=0.5)
        bottom += vals

    ax1.set_xlabel("Week", fontsize=11)
    ax1.set_ylabel("Spend (EUR thousands)", fontsize=11)
    ax1.set_xticks([1,2,3,4])
    ax1.set_xticklabels(["Week 1\n(early LT orders\nonly)", "Week 2\n★ warehouse\nbinds 99%",
                          "Week 3\n(overflow\nfrom W)", "Week 4\n(end-of-\nhorizon)"])
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}k"))
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(pivot.index, n_prods.reindex(pivot.index, fill_value=0).values,
             marker="D", color="black", linewidth=2, markersize=8, label="# products ordered")
    ax2.set_ylabel("Products ordered", fontsize=11)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.set_ylim(0, 60)

    plt.tight_layout()
    path = PLOTS_DIR / "07_purchase_plan.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. WAREHOUSE UTILISATION OVER HORIZON
# ═══════════════════════════════════════════════════════════════════════════════

def plot_warehouse():
    """
    Warehouse ending-stock volume for the 4 optimised weeks (t=1..4).

    t=0 (initial stock) is intentionally excluded from the bars because it is a
    fixed input parameter, not a decision variable — the warehouse constraint
    only applies to *ending stock* after each optimised period. It is shown
    separately as a horizontal reference arrow to avoid the misleading ">100%"
    impression.
    """
    stock["vol"]      = stock["product_id"].map(vol_map) * stock["ending_stock"]
    stock["category"] = stock["product_id"].map(cat_map)

    pivot = (
        stock.groupby(["period_t","category"])["vol"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=CAT_ORDER, fill_value=0)
    )

    # Initial stock volume (parameter, not plotted as bar)
    init_vol = sum(
        (products[products["category"] == cat]["product_id"]
         .map(products.set_index("product_id")["initial_stock"])
         * products[products["category"] == cat]["product_id"].map(vol_map)
        ).sum()
        for cat in CAT_ORDER
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "Warehouse Ending-Stock Volume — Optimisation Horizon\n"
        "(constraint applies to weeks 1-4 only; initial stock is a fixed input)",
        fontsize=12, fontweight="bold",
    )

    bottom = np.zeros(len(pivot))
    for cat in CAT_ORDER:
        ax.bar(pivot.index, pivot[cat].values, bottom=bottom,
               color=PALETTE[cat], label=cat.capitalize(),
               edgecolor="white", linewidth=0.5, width=0.55)
        bottom += pivot[cat].values

    # Capacity line
    ax.axhline(W, color="#C44E52", linestyle="--", linewidth=2.5,
               label=f"Capacity W = {W:,.0f}")
    ax.fill_between([0.5, 4.5], W, bottom.max() * 1.12,
                    color="#C44E52", alpha=0.05)

    # Initial stock as an annotated arrow on the left
    ax.axhline(init_vol, color="grey", linestyle=":", linewidth=1.2, alpha=0.6)
    ax.text(
        0.5, init_vol + 80,
        f"Pre-period stock: {init_vol:,.0f} m³ ({init_vol/W:.0%} of W)  ← not constrained by W",
        ha="center", va="bottom", fontsize=8.5, color="grey",
        transform=ax.get_yaxis_transform(),
    )

    ax.set_xticks([1,2,3,4])
    ax.set_xticklabels(["Week 1\n63%", "Week 2\n★ 99%\n(binding)", "Week 3\n83%", "Week 4\n25%"])
    ax.set_ylabel("Volume (m³ equivalent)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_ylim(0, max(bottom.max(), init_vol) * 1.15)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Utilisation labels on bars
    totals = pivot.sum(axis=1)
    for t, v in totals.items():
        ax.text(t, v + 60, f"{v:,.0f}", ha="center", va="bottom",
                fontsize=8, color="black")

    plt.tight_layout()
    path = PLOTS_DIR / "08_warehouse_utilisation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. COST BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cost_breakdown():
    """
    Two horizontal bar charts side by side:
      Left  — cost components (purchase, holding, fixed logistics)
      Right — purchase spend by category
    """
    fix_total = purchase.groupby(["supplier_id","period_t"]).apply(
        lambda g: suppliers.set_index("supplier_id").loc[g.name[0],"fixed_cost"],
        include_groups=False,
    ).sum()
    hld = stock.merge(products[["product_id","holding_cost"]], on="product_id")
    total_hold = (hld["ending_stock"] * hld["holding_cost"]).sum()
    total_pur  = purchase["order_cost"].sum()
    grand      = total_pur + total_hold + fix_total

    comp_labels = ["Purchase &\nvar. logistics", "Holding cost", "Fixed logistics"]
    comp_values = [total_pur, total_hold, fix_total]
    comp_colors = ["#4C72B0", "#55A868", "#DD8452"]

    p2 = purchase.merge(products[["product_id","category"]], on="product_id")
    cat_spend = p2.groupby("category")["order_cost"].sum().reindex(CAT_ORDER)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Cost Breakdown — Optimization Horizon (4 weeks)   Grand total: EUR {grand/1e6:.2f}M",
                 fontsize=13, fontweight="bold")

    # ── Left: cost components (horizontal bars) ───────────────────────────
    bars1 = ax1.barh(comp_labels, [v/1_000 for v in comp_values],
                     color=comp_colors, edgecolor="white", height=0.5)
    ax1.set_xlabel("EUR thousands", fontsize=10)
    ax1.set_title("Cost components", fontsize=11)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}k"))
    ax1.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars1, comp_values):
        pct = val / grand * 100
        ax1.text(bar.get_width() + grand * 0.0003 / 1_000, bar.get_y() + bar.get_height() / 2,
                 f"EUR {val/1e3:,.1f}k  ({pct:.1f}%)",
                 va="center", fontsize=9)
    ax1.set_xlim(right=max(comp_values) / 1_000 * 1.35)

    # ── Right: spend by category (horizontal bars) ────────────────────────
    bars2 = ax2.barh(CAT_ORDER, cat_spend.values / 1_000,
                     color=[PALETTE[c] for c in CAT_ORDER],
                     edgecolor="white", height=0.5)
    ax2.set_xlabel("EUR thousands", fontsize=10)
    ax2.set_title("Purchase spend by category", fontsize=11)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}k"))
    ax2.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars2, cat_spend.values):
        ax2.text(bar.get_width() + total_pur * 0.003 / 1_000,
                 bar.get_y() + bar.get_height() / 2,
                 f"EUR {val/1e3:,.0f}k  ({val/total_pur:.0%})",
                 va="center", fontsize=9)
    ax2.set_xlim(right=cat_spend.max() / 1_000 * 1.35)

    plt.tight_layout()
    path = PLOTS_DIR / "09_cost_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SAFETY STOCK vs INITIAL STOCK
# ═══════════════════════════════════════════════════════════════════════════════

def plot_safety_stock():
    """
    Replaces the scatter (which looked suspicious because initial stock coverage
    was uniformly distributed 1.5-2.5w by design, giving a meaningless x-axis).

    Instead: side-by-side horizontal bar charts comparing
      - Initial stock coverage (weeks) — what we have
      - Safety stock requirement (weeks) — what we need to hold
    sorted by safety stock, colour-coded by category.

    This makes it easy to see which products have the tightest or loosest
    buffer (initial - SS) and where re-ordering is most urgent.
    """
    avg_ss   = forecast.groupby("product_id")["safety_stock"].mean()
    init_cov = products.set_index("product_id")["initial_stock"] / avg_wk
    ss_cov   = avg_ss / avg_wk

    df = pd.DataFrame({
        "initial_weeks": init_cov,
        "ss_weeks":      ss_cov,
        "buffer":        init_cov - ss_cov,
        "category":      cat_map,
        "name":          name_map,
    }).sort_values("ss_weeks", ascending=True).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 14), sharey=True)
    fig.suptitle(
        "Initial Stock vs Safety Stock Coverage per Product\n"
        "(sorted by safety stock requirement; buffer = initial − SS)",
        fontsize=13, fontweight="bold",
    )

    y      = np.arange(len(df))
    colors = [PALETTE[c] for c in df["category"]]

    # Left: initial stock
    ax1.barh(y, df["initial_weeks"], color=colors, alpha=0.75,
             edgecolor="white", height=0.7)
    ax1.set_xlabel("Weeks of demand", fontsize=10)
    ax1.set_title("Initial stock coverage", fontsize=11)
    ax1.axvline(df["initial_weeks"].mean(), color="black", linestyle="--",
                linewidth=1, alpha=0.5, label=f"Mean {df['initial_weeks'].mean():.1f}w")
    ax1.legend(fontsize=9)
    ax1.grid(axis="x", alpha=0.3)
    ax1.invert_xaxis()   # grows left so products align in the middle

    # Right: safety stock
    ax2.barh(y, df["ss_weeks"], color=colors, alpha=0.75,
             edgecolor="white", height=0.7)
    ax2.set_xlabel("Weeks of demand", fontsize=10)
    ax2.set_title("Safety stock requirement  SS = z·σ·√LT", fontsize=11)
    ax2.axvline(df["ss_weeks"].mean(), color="black", linestyle="--",
                linewidth=1, alpha=0.5, label=f"Mean {df['ss_weeks'].mean():.2f}w")
    ax2.legend(fontsize=9)
    ax2.grid(axis="x", alpha=0.3)

    # Product names in the middle
    ax1.set_yticks(y)
    ax1.set_yticklabels([])
    ax2.set_yticks(y)
    ax2.set_yticklabels(df["name"], fontsize=7.5)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    # Category legend
    handles = [plt.Rectangle((0,0),1,1, color=PALETTE[c], alpha=0.75) for c in CAT_ORDER]
    fig.legend(handles, [c.capitalize() for c in CAT_ORDER],
               loc="lower center", ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    path = PLOTS_DIR / "10_safety_stock_vs_initial.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nGenerating plots …")
    plot_feature_distributions()
    plot_demand_distributions()
    plot_seasonality()
    plot_feature_importance()
    plot_mape_distribution()
    plot_forecast_samples()
    plot_purchase_plan()
    plot_warehouse()
    plot_cost_breakdown()
    plot_safety_stock()
    print(f"\nAll plots saved to {PLOTS_DIR.resolve()}/")
