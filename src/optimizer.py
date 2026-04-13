"""
MILP purchase optimizer using PuLP + HiGHS.

Decision variables
------------------
p[i,j,t]  continuous >= 0  : units purchased of product i from supplier j in week t
y[i,j,t]  binary            : 1 if order (i,j,t) is placed
z[j,t]    binary            : 1 if supplier j is active in week t
u[i,t]    continuous >= 0  : stockout units of product i in week t
S[i,t]    continuous >= 0  : ending stock of product i after week t

Objective
---------
min  Σ (buy_cost + var_cost) * p
   + Σ fix_cost * y
   + Σ holding_cost * S
   + Σ stockout_penalty * u

Constraints
-----------
1. Inventory balance  : S[i,t] = S[i,t-1] + Σ_j p[i,j,t] - D[i,t] + u[i,t]
2. Safety stock floor : S[i,t] >= SS[i,t]
3. MOQ big-M (lower)  : p[i,j,t] >= MOQ[i,j] * y[i,j,t]
4. MOQ big-M (upper)  : p[i,j,t] <= M_big * y[i,j,t]
5. Min order supplier : Σ_i buy_cost[i,j]*p[i,j,t] >= MinOrder[j] * z[j,t]
6. Warehouse capacity : Σ_i volume[i] * S[i,t] <= W
"""

import pandas as pd
import pulp


def build_and_solve(
    forecast_df: pd.DataFrame,
    products_df: pd.DataFrame,
    suppliers_df: pd.DataFrame,
    supplier_products_df: pd.DataFrame,
    W: float,
    time_limit_sec: int = 300,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Build and solve the MILP.

    Returns
    -------
    purchase_plan : DataFrame [product_id, supplier_id, period_t, units, order_cost]
    stock_plan    : DataFrame [product_id, period_t, ending_stock, stockout]
    status        : solver status string
    """
    # ── Index sets ────────────────────────────────────────────────────────
    products = products_df["product_id"].tolist()
    suppliers = suppliers_df["supplier_id"].tolist()
    periods = sorted(forecast_df["period_t"].unique().tolist())

    # ── Parameter lookups ─────────────────────────────────────────────────
    prod = products_df.set_index("product_id")
    sup  = suppliers_df.set_index("supplier_id")
    sp   = supplier_products_df.set_index(["product_id", "supplier_id"])

    # valid (product, supplier) pairs
    valid_ij = set(zip(supplier_products_df["product_id"], supplier_products_df["supplier_id"]))

    # demand and safety stock
    fc = forecast_df.set_index(["product_id", "period_t"])

    def D(i, t):
        return fc.loc[(i, t), "demand_hat"]

    def SS(i, t):
        return fc.loc[(i, t), "safety_stock"]

    # Big-M: large enough to not cut off feasible solutions
    M_big = max(forecast_df["demand_hat"]) * len(periods) * 3

    # ── Model ─────────────────────────────────────────────────────────────
    model = pulp.LpProblem("SupplyChainOptimizer", pulp.LpMinimize)

    # ── Decision variables ────────────────────────────────────────────────
    p = {
        (i, j, t): pulp.LpVariable(f"p_{i}_{j}_{t}", lowBound=0, cat="Continuous")
        for i in products for j in suppliers for t in periods
        if (i, j) in valid_ij
    }
    y = {
        (i, j, t): pulp.LpVariable(f"y_{i}_{j}_{t}", cat="Binary")
        for i in products for j in suppliers for t in periods
        if (i, j) in valid_ij
    }
    z = {
        (j, t): pulp.LpVariable(f"z_{j}_{t}", cat="Binary")
        for j in suppliers for t in periods
    }
    u = {
        (i, t): pulp.LpVariable(f"u_{i}_{t}", lowBound=0, cat="Continuous")
        for i in products for t in periods
    }
    S = {
        (i, t): pulp.LpVariable(f"S_{i}_{t}", lowBound=0, cat="Continuous")
        for i in products for t in periods
    }

    # ── Objective ─────────────────────────────────────────────────────────
    purchase_cost = pulp.lpSum(
        (sp.loc[(i, j), "buy_cost"] + sp.loc[(i, j), "var_logistics_cost"]) * p[i, j, t]
        for (i, j, t) in p
    )
    fixed_logistics = pulp.lpSum(
        sup.loc[j, "fixed_cost"] * y[i, j, t]
        for (i, j, t) in y
    )
    holding = pulp.lpSum(
        prod.loc[i, "holding_cost"] * S[i, t]
        for i in products for t in periods
    )
    stockout_penalty = pulp.lpSum(
        prod.loc[i, "stockout_penalty"] * u[i, t]
        for i in products for t in periods
    )

    model += purchase_cost + fixed_logistics + holding + stockout_penalty

    # ── Constraints ───────────────────────────────────────────────────────
    for i in products:
        S0 = prod.loc[i, "initial_stock"]
        for idx, t in enumerate(periods):
            t_prev = periods[idx - 1] if idx > 0 else None
            stock_prev = S[i, t_prev] if t_prev is not None else S0

            purchases_t = pulp.lpSum(
                p[i, j, t] for j in suppliers if (i, j, t) in p
            )

            # 1. Inventory balance
            model += (
                S[i, t] == stock_prev + purchases_t - D(i, t) + u[i, t],
                f"inv_balance_{i}_{t}",
            )

            # 2. Safety stock floor
            model += (
                S[i, t] >= SS(i, t),
                f"safety_stock_{i}_{t}",
            )

            # 6. Warehouse capacity (aggregated across all products per period)

        for j in suppliers:
            if (i, j) not in sp.index:
                continue
            moq = sp.loc[(i, j), "MOQ"]
            for t in periods:
                if (i, j, t) not in p:
                    continue
                # 3. MOQ lower bound
                model += (
                    p[i, j, t] >= moq * y[i, j, t],
                    f"moq_lower_{i}_{j}_{t}",
                )
                # 4. MOQ upper bound (big-M)
                model += (
                    p[i, j, t] <= M_big * y[i, j, t],
                    f"moq_upper_{i}_{j}_{t}",
                )

    # 5. Min order per supplier
    for j in suppliers:
        min_order = sup.loc[j, "min_order_euros"]
        for t in periods:
            order_value = pulp.lpSum(
                sp.loc[(i, j), "buy_cost"] * p[i, j, t]
                for i in products if (i, j, t) in p
            )
            model += (
                order_value >= min_order * z[j, t],
                f"min_order_{j}_{t}",
            )

    # 6. Warehouse capacity per period
    for t in periods:
        model += (
            pulp.lpSum(prod.loc[i, "unit_volume"] * S[i, t] for i in products) <= W,
            f"warehouse_{t}",
        )

    # ── Solve ─────────────────────────────────────────────────────────────
    print(f"  [optimizer] Solving MILP: {len(products)} products × "
          f"{len(suppliers)} suppliers × {len(periods)} periods …")
    print(f"  [optimizer] Variables: {len(p)+len(y)+len(z)+len(u)+len(S):,}  "
          f"  (binary: {len(y)+len(z):,})")

    # Prefer HiGHS; fall back to CBC (bundled with PuLP)
    if pulp.HiGHS_CMD().available():
        solver = pulp.HiGHS_CMD(msg=True, timeLimit=time_limit_sec)
    else:
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=time_limit_sec)
    status_code = model.solve(solver)
    status = pulp.LpStatus[status_code]
    print(f"  [optimizer] Status: {status}  |  Obj = {pulp.value(model.objective):,.2f}")

    # ── Extract solution ───────────────────────────────────────────────────
    purchase_records = []
    for (i, j, t), var in p.items():
        val = pulp.value(var)
        if val is not None and val > 1e-4:
            buy_c = sp.loc[(i, j), "buy_cost"]
            var_c = sp.loc[(i, j), "var_logistics_cost"]
            purchase_records.append({
                "product_id": i,
                "supplier_id": j,
                "period_t": t,
                "units": round(val, 2),
                "buy_cost_per_unit": buy_c,
                "var_cost_per_unit": var_c,
                "order_cost": round((buy_c + var_c) * val, 2),
            })
    purchase_plan = pd.DataFrame(purchase_records)
    if purchase_plan.empty:
        purchase_plan = pd.DataFrame(
            columns=["product_id","supplier_id","period_t","units","buy_cost_per_unit","var_cost_per_unit","order_cost"]
        )

    stock_records = []
    for i in products:
        for t in periods:
            s_val = pulp.value(S[i, t])
            u_val = pulp.value(u[i, t])
            stock_records.append({
                "product_id": i,
                "period_t": t,
                "ending_stock": s_val or 0.0,
                "stockout": round(u_val or 0, 4),
            })
    stock_plan = pd.DataFrame(stock_records)

    return purchase_plan, stock_plan, status
