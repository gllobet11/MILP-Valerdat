"""
KPI reporting for the supply chain optimization pipeline.
Prints structured summaries and returns DataFrames for further analysis.
"""

import pandas as pd
import numpy as np


def report_kpis(
    purchase_plan: pd.DataFrame,
    stock_plan: pd.DataFrame,
    forecast_df: pd.DataFrame,
    products_df: pd.DataFrame,
    suppliers_df: pd.DataFrame,
    supplier_products_df: pd.DataFrame,
    val_metrics: pd.DataFrame,
) -> dict:
    """
    Compute and print KPIs. Returns a dict of summary DataFrames.
    """
    sep = "=" * 65

    # ── 1. Forecast accuracy ──────────────────────────────────────────────
    print(f"\n{sep}")
    print("  FORECAST ACCURACY (validation set)")
    print(sep)
    avg_mape = val_metrics["mape"].mean()
    med_mape = val_metrics["mape"].median()
    print(f"  Avg MAPE : {avg_mape:.1%}")
    print(f"  Median MAPE : {med_mape:.1%}")
    print(f"  Products with MAPE > 30% : {(val_metrics['mape'] > 0.30).sum()} / {len(val_metrics)}")

    # ── 2. Cost breakdown ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  COST BREAKDOWN")
    print(sep)

    sp = supplier_products_df.set_index(["product_id", "supplier_id"])
    sup = suppliers_df.set_index("supplier_id")
    prod = products_df.set_index("product_id")

    if not purchase_plan.empty:
        total_purchase = purchase_plan["order_cost"].sum()

        # fixed logistics: count distinct (product, supplier, period) orders and multiply by fix_cost
        fix_cost_total = purchase_plan.groupby(["supplier_id", "period_t"]).apply(
            lambda g: sup.loc[g.name[0], "fixed_cost"] if len(g) > 0 else 0,
            include_groups=False,
        ).sum()

        # stockout penalties
        pen = (
            stock_plan.merge(
                products_df[["product_id", "stockout_penalty"]], on="product_id"
            )
        )
        total_stockout_cost = (pen["stockout"] * pen["stockout_penalty"]).sum()

        # holding costs
        hld = (
            stock_plan.merge(
                products_df[["product_id", "holding_cost"]], on="product_id"
            )
        )
        total_holding = (hld["ending_stock"] * hld["holding_cost"]).sum()

        grand_total = total_purchase + fix_cost_total + total_holding + total_stockout_cost
    else:
        total_purchase = fix_cost_total = total_holding = total_stockout_cost = grand_total = 0.0

    rows = [
        ("Purchase + variable logistics", total_purchase),
        ("Fixed logistics",               fix_cost_total),
        ("Holding cost",                  total_holding),
        ("Stockout penalties",            total_stockout_cost),
        ("GRAND TOTAL",                   grand_total),
    ]
    for label, val in rows:
        marker = "  ─" if label != "GRAND TOTAL" else "  ═"
        print(f"{marker} {label:<34} EUR {val:>12,.2f}")

    # ── 3. Service level ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  SERVICE LEVEL (stock >= safety stock)")
    print(sep)

    sl_df = (
        stock_plan.merge(forecast_df[["product_id", "period_t", "safety_stock"]], on=["product_id", "period_t"])
    )
    # Use a small absolute tolerance to absorb LP floating-point imprecision
    sl_df["meets_ss"] = sl_df["ending_stock"] >= sl_df["safety_stock"] - 0.5
    sl_by_product = sl_df.groupby("product_id")["meets_ss"].mean().reset_index(name="service_level")
    overall_sl = sl_df["meets_ss"].mean()

    print(f"  Overall service level : {overall_sl:.1%}")
    below_90 = sl_by_product[sl_by_product["service_level"] < 0.90]
    if below_90.empty:
        print("  All products achieve >= 90% service level.")
    else:
        print(f"  Products below 90% SL ({len(below_90)}):")
        for _, row in below_90.iterrows():
            print(f"    {row['product_id']}  {row['service_level']:.0%}")

    # ── 4. Stockout rate ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  STOCKOUT RATE")
    print(sep)

    total_stockout = stock_plan["stockout"].sum()
    total_demand   = forecast_df["demand_hat"].sum()
    stockout_rate  = total_stockout / total_demand if total_demand > 0 else 0
    print(f"  Total stockout units : {total_stockout:,.1f}")
    print(f"  Total forecast demand: {total_demand:,.1f}")
    print(f"  Stockout rate        : {stockout_rate:.2%}")

    # ── 5. Purchase plan summary ──────────────────────────────────────────
    print(f"\n{sep}")
    print("  PURCHASE PLAN SUMMARY (by supplier × week)")
    print(sep)

    if not purchase_plan.empty:
        summary = (
            purchase_plan.groupby(["supplier_id", "period_t"])
            .agg(
                n_products=("product_id", "nunique"),
                total_units=("units", "sum"),
                total_cost=("order_cost", "sum"),
            )
            .reset_index()
        )
        print(summary.to_string(index=False))
    else:
        print("  No orders placed.")

    # ── 6. Supplier utilization ───────────────────────────────────────────
    print(f"\n{sep}")
    print("  SUPPLIER UTILIZATION")
    print(sep)

    if not purchase_plan.empty:
        su = (
            purchase_plan.groupby("supplier_id")["period_t"]
            .nunique()
            .reset_index(name="active_weeks")
        )
        su["total_spend"] = (
            purchase_plan.groupby("supplier_id")["order_cost"].sum().values
        )
        print(su.to_string(index=False))
    else:
        print("  No supplier activity.")

    print(f"\n{sep}\n")

    return {
        "val_metrics": val_metrics,
        "cost_breakdown": pd.DataFrame(rows, columns=["component", "eur"]),
        "service_level": sl_by_product,
        "purchase_summary": summary if not purchase_plan.empty else pd.DataFrame(),
    }
