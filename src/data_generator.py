"""
Synthetic data generator for supply chain optimization.
Produces master data (products, suppliers, supplier-product relations)
and 18 months of daily demand history with realistic patterns.

Calibration principles (v2):
  - Unit values fixed per product name (not random within category)
  - Holding cost: 20-35% p.a. → 0.04-0.07% per week of unit value
  - Stockout penalty: 1.2-3.5× unit value (category-dependent)
  - Initial stock: 3-6 weeks of average weekly demand (post demand-gen)
  - Lead time: per supplier-product pair, not per product
  - Warehouse W: 1.5× one-week demand volume (creates binding pressure)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

# ── Product catalogue ────────────────────────────────────────────────────────
# Columns: name, category, unit_value (EUR), daily_demand_base (units),
#          unit_volume (m³ equivalent), moq_base (units, reference supplier)
PRODUCT_CATALOG = [
    # ── Electronics ──────────────────────────────────────────────────────────
    # (high value, medium-high demand — B2B distributor scale)
    ("Laptop 15\"",        "electronics", 1_200, 15,   0.80,   5),
    ("Laptop 13\"",        "electronics",   900, 20,   0.70,   5),
    ("Monitor 24\"",       "electronics",   250, 30,   0.60,  10),
    ("Monitor 27\"",       "electronics",   350, 25,   0.70,   8),
    ("Keyboard Wireless",  "electronics",    45, 50,   0.15,  20),
    ("Mouse Wireless",     "electronics",    35, 60,   0.08,  20),
    ("USB-C Hub",          "electronics",    55, 45,   0.05,  30),
    ("Webcam HD",          "electronics",    80, 35,   0.10,  15),
    ("Headset BT",         "electronics",   120, 30,   0.15,  15),
    ("SSD 1TB",            "electronics",    90, 40,   0.03,  30),
    ("SSD 512GB",          "electronics",    60, 50,   0.03,  30),
    ("RAM 16GB",           "electronics",    55, 35,   0.02,  30),
    ("RAM 8GB",            "electronics",    30, 45,   0.02,  40),
    ("GPU Entry",          "electronics",   350, 20,   0.40,   5),
    ("Tablet 10\"",        "electronics",   280, 25,   0.30,   8),
    ("Phone Charger 65W",  "electronics",    25, 70,   0.05,  40),
    ("Extension Cable",    "electronics",    15, 80,   0.10,  60),
    # ── Consumables ──────────────────────────────────────────────────────────
    # (low value, high demand, varied volume)
    ("Paper A4 (500sh)",   "consumables",    6, 300,  0.25, 200),
    ("Paper A3 (250sh)",   "consumables",    9, 120,  0.30, 100),
    ("Pen Box (12u)",      "consumables",    5, 100,  0.05,  80),
    ("Marker Set",         "consumables",    8,  60,  0.05,  50),
    ("Staples Box",        "consumables",    3,  80,  0.03, 100),
    ("Binder A4",          "consumables",    4, 100,  0.08,  80),
    ("Sticky Notes",       "consumables",    3, 120,  0.03, 150),
    ("Label Roll",         "consumables",   12,  80,  0.05,  60),
    ("Toner Black",        "consumables",   65,  50,  0.10,  30),
    ("Toner Color",        "consumables",   85,  25,  0.12,  15),
    ("Ink Cartridge Blk",  "consumables",   18,  60,  0.05,  40),
    ("Ink Cartridge Col",  "consumables",   24,  40,  0.05,  30),
    ("Cleaning Wipes",     "consumables",    8, 150,  0.08, 100),
    ("Hand Sanitizer 1L",  "consumables",    7, 180,  0.10, 120),
    ("Coffee Pods (50u)",  "consumables",   15, 150,  0.15,  80),
    ("Tea Bags (100u)",    "consumables",    6, 100,  0.08,  80),
    # ── Industrial ───────────────────────────────────────────────────────────
    # (medium value, medium-high demand; safety items carry higher penalty)
    ("Safety Helmet",           "industrial",  35,  60,  0.30,  30),
    ("Safety Gloves L",         "industrial",  12, 120,  0.08,  60),
    ("Safety Gloves M",         "industrial",  12, 100,  0.08,  60),
    ("Hi-Vis Vest",             "industrial",  18,  80,  0.10,  40),
    ("Safety Boots 42",         "industrial",  85,  30,  0.30,  15),
    ("Safety Boots 43",         "industrial",  85,  35,  0.30,  15),
    ("Safety Boots 44",         "industrial",  85,  25,  0.30,  10),
    ("First Aid Kit",           "industrial",  45,  45,  0.25,  20),
    ("Fire Extinguisher 2kg",   "industrial",  55,  25,  0.40,  10),
    ("Fire Extinguisher 6kg",   "industrial",  95,  15,  0.60,   8),
    ("Cable Ties (100u)",       "industrial",   8, 150,  0.04, 100),
    ("Zip Ties Heavy",          "industrial",  12, 100,  0.06,  60),
    ("WD-40 400ml",             "industrial",   9, 120,  0.05,  80),
    ("Lubricant Spray",         "industrial",  11,  90,  0.05,  60),
    ("Drill Bit Set",           "industrial",  45,  60,  0.20,  30),
    ("Screwdriver Set",         "industrial",  55,  50,  0.15,  25),
    ("Utility Knife",           "industrial",  15,  70,  0.05,  50),
]

assert len(PRODUCT_CATALOG) == 50, "Need exactly 50 products"

# Holding cost rate (fraction of unit value per week), by category
#   Target: 20-35% per annum / 52 weeks
HOLDING_RATE = {
    "electronics": (0.0040, 0.0067),  # 21-35% p.a. (obsolescence premium)
    "consumables": (0.0030, 0.0050),  # 16-26% p.a.
    "industrial":  (0.0030, 0.0050),  # 16-26% p.a.
}

# Stockout penalty multiplier (× unit value), by category
STOCKOUT_MULT = {
    "electronics": (1.5, 2.5),   # high margin, customer-facing
    "consumables": (1.2, 1.8),   # easy to substitute, lower margin
    "industrial":  (2.0, 3.5),   # safety compliance, production stoppage
}

# Supplier profiles: (name, fixed_cost EUR, min_order EUR, base_lead_time weeks)
SUPPLIERS = [
    ("SupplierAlpha",  150,   500, 1),   # domestic, fast
    ("SupplierBeta",    80,   300, 1),   # regional discount supplier
    ("SupplierGamma",  200, 1_000, 3),   # international, wide catalogue
    ("SupplierDelta",  120,   600, 2),   # specialist, mid-range lead time
]


def generate_products() -> pd.DataFrame:
    """Return product master table with realistic, name-consistent parameters."""
    records = []
    for i, (name, cat, val, _, vol, _) in enumerate(PRODUCT_CATALOG):
        hc_lo, hc_hi = HOLDING_RATE[cat]
        pen_lo, pen_hi = STOCKOUT_MULT[cat]
        records.append({
            "product_id":       f"P{i:02d}",
            "name":             name,
            "category":         cat,
            "unit_value":       float(val),
            # holding cost EUR per unit per week
            "holding_cost":     round(val * RNG.uniform(hc_lo, hc_hi), 4),
            # stockout penalty EUR per unit
            "stockout_penalty": round(val * RNG.uniform(pen_lo, pen_hi), 2),
            # storage volume (m³ equivalent)
            "unit_volume":      float(vol),
            # initial_stock set in generate_all() after demand is known
            "initial_stock":    0,
            # lead_time moved to supplier_products; kept here as max across suppliers
            "lead_time_weeks":  1,
        })
    return pd.DataFrame(records)


def generate_suppliers() -> pd.DataFrame:
    """Return supplier master table."""
    return pd.DataFrame([
        {
            "supplier_id":     f"S{j}",
            "name":            name,
            "fixed_cost":      float(fc),
            "min_order_euros": float(mo),
            "base_lead_time":  int(lt),
        }
        for j, (name, fc, mo, lt) in enumerate(SUPPLIERS)
    ])


def generate_supplier_products(
    products_df: pd.DataFrame,
    suppliers_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign each product to 2–3 suppliers with:
      - price premium vs. cheapest source (5–25%)
      - variable logistics cost (1–4% of buy price)
      - MOQ derived from catalogue moq_base with per-supplier variation
      - lead_time per supplier-product pair (supplier base ± 1 week)
    """
    n_suppliers = len(suppliers_df)
    records = []

    # Build moq_base and demand_base lookups from catalogue
    moq_base_map  = {f"P{i:02d}": moq  for i, (*_, moq)  in enumerate(PRODUCT_CATALOG)}

    for _, prod in products_df.iterrows():
        pid  = prod["product_id"]
        val  = prod["unit_value"]
        moq0 = moq_base_map[pid]

        n_sources = int(RNG.integers(2, n_suppliers + 1))   # 2 or 3 suppliers
        assigned  = RNG.choice(n_suppliers, size=n_sources, replace=False)

        for rank, sid_idx in enumerate(assigned):
            sup = suppliers_df.iloc[sid_idx]
            # cheapest source gets base price; alternatives carry a premium
            premium  = 1.0 if rank == 0 else float(RNG.uniform(1.05, 1.25))
            buy_cost = round(val * premium, 2)
            var_cost = round(buy_cost * float(RNG.uniform(0.01, 0.04)), 4)

            # MOQ: base ± 50% variation, minimum 1
            moq = max(1, int(round(moq0 * float(RNG.uniform(0.5, 1.5)))))

            # Lead time: supplier base ± 1 week (min 1)
            lt = max(1, int(sup["base_lead_time"] + RNG.integers(-1, 2)))

            records.append({
                "product_id":        pid,
                "supplier_id":       sup["supplier_id"],
                "buy_cost":          buy_cost,
                "var_logistics_cost": var_cost,
                "MOQ":               moq,
                "lead_time_weeks":   lt,
            })

    df = pd.DataFrame(records)

    # Update products_df.lead_time_weeks to the max lead time across suppliers
    max_lt = df.groupby("product_id")["lead_time_weeks"].max()
    products_df["lead_time_weeks"] = products_df["product_id"].map(max_lt).fillna(1).astype(int)

    return df


def generate_demand(
    products_df: pd.DataFrame,
    start_date: str = "2024-07-01",
    n_days: int = 547,  # ~18 months
) -> pd.DataFrame:
    """
    Generate daily demand per product using catalogue base demand levels with:
      - weekly seasonality (weekday / weekend multiplier by category)
      - annual sine-wave seasonality
      - random promotional spikes (~5% of days)
      - Poisson noise
    """
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    # Pull base demand from catalogue
    demand_base_map = {f"P{i:02d}": base for i, (*_, base, _, _) in enumerate(PRODUCT_CATALOG)}

    # Weekend demand multiplier by category
    weekend_mult = {
        "electronics": 0.3,   # B2B electronics: very low weekend orders
        "consumables": 0.2,   # office supplies: almost no weekend demand
        "industrial":  0.1,   # industrial: near-zero on weekends
    }

    records = []
    for _, prod in products_df.iterrows():
        pid = prod["product_id"]
        cat = prod["category"]
        base = demand_base_map[pid]

        # Slight individual trend over 18 months
        trend_slope = float(RNG.uniform(-0.0005, 0.002))

        for d_idx, date in enumerate(dates):
            mu = base * (1 + trend_slope * d_idx)

            # Weekly seasonality
            if date.dayofweek >= 5:
                mu *= weekend_mult[cat]

            # Annual seasonality: peak in Q4 (office budget spend / winter PPE)
            mu *= 1 + 0.25 * np.sin(2 * np.pi * (date.dayofyear - 60) / 365)

            # Promotional / bulk order spike (~5% of workdays)
            if date.dayofweek < 5 and RNG.random() < 0.05:
                mu *= float(RNG.uniform(1.5, 2.5))

            demand = int(RNG.poisson(max(mu, 0.1)))
            records.append({"date": date, "product_id": pid, "demand": demand})

    return pd.DataFrame(records)


def generate_all(data_dir: str = "data") -> dict:
    """
    Generate and persist all synthetic datasets.
    Returns a dict with all DataFrames and the warehouse capacity scalar W.
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    products_df  = generate_products()
    suppliers_df = generate_suppliers()
    demand_df    = generate_demand(products_df)

    # ── Initial stock: 1.5–2.5 weeks of average weekly demand ───────────────
    #    Deliberately kept moderate so the warehouse capacity constraint can bind
    avg_weekly = (
        demand_df.groupby("product_id")["demand"]
        .mean() * 7              # daily mean → weekly mean
    )
    products_df["initial_stock"] = (
        products_df["product_id"]
        .map(avg_weekly)
        .apply(lambda w: max(1, int(round(w * float(RNG.uniform(1.5, 2.5))))))
    )

    # ── Supplier-product links (uses updated products_df for lead_time) ───────
    sp_df = generate_supplier_products(products_df, suppliers_df)

    # ── Warehouse capacity: 1.5× one-week demand volume ──────────────────────
    #    Safety stock alone requires ~0.4× weekly volume (lower bound).
    #    1.5× gives enough room to hold SS + pipeline stock while remaining
    #    tight enough to force the optimizer to spread bulk orders across weeks.
    vol = products_df.set_index("product_id")["unit_volume"]
    weekly_vol = float((avg_weekly * vol).sum())
    W = weekly_vol * 1.5

    products_df.to_csv(f"{data_dir}/products.csv",          index=False)
    suppliers_df.to_csv(f"{data_dir}/suppliers.csv",         index=False)
    sp_df.to_csv(f"{data_dir}/supplier_products.csv",        index=False)
    demand_df.to_csv(f"{data_dir}/historical_demand.csv",    index=False)

    # Summary
    print(f"[data_generator] {len(products_df)} products | "
          f"{len(suppliers_df)} suppliers | "
          f"{len(sp_df)} supplier-product links | "
          f"{len(demand_df):,} demand rows")
    print(f"[data_generator] Warehouse capacity W = {W:,.0f} volume-units")

    avg_init_weeks = (
        products_df["product_id"].map(
            products_df.set_index("product_id")["initial_stock"] / avg_weekly
        ).mean()
    )
    print(f"[data_generator] Avg initial stock coverage = {avg_init_weeks:.1f} weeks")

    return {
        "products":          products_df,
        "suppliers":         suppliers_df,
        "supplier_products": sp_df,
        "demand":            demand_df,
        "W":                 W,
    }


if __name__ == "__main__":
    generate_all()
