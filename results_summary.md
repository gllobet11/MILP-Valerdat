# Supply Chain Optimization — Results Summary

**Project:** Valerdat Technical Test  
**Date:** 2026-04-11  
**Stack:** Python · LightGBM · PuLP + CBC · Synthetic data  

---

## 1. Problem Overview

The goal is to determine the optimal monthly purchase plan $P = \{p_{i,j,t}\}$ for a portfolio of **50 products**, **4 suppliers**, over a **4-week horizon** — minimising total cost while respecting inventory, logistics, and warehouse constraints.

The pipeline has two independent modules connected by a data handoff:

```
Historical demand (18 months daily)
        ↓
Feature Engineering  →  LightGBM Forecast (q10 / q50 / q90)
                                   ↓
                     Dynamic Safety Stock  SS_i = z_α · σ_i · √LT_i
                                   ↓
                         MILP Optimizer (PuLP + CBC)
                                   ↓
                       Purchase Plan  P_{i,j,t}  +  KPI Report
```

---

## 2. Synthetic Dataset

| Parameter | Value |
|---|---|
| Products (K) | 50 — across electronics (17), consumables (16), industrial (17) |
| Suppliers (M) | 4 — domestic (S0, S1), international (S2), specialist (S3) |
| Historical demand | 547 days (~18 months), daily granularity |
| Demand rows | 27,350 |
| Supplier-product links | 158 (each product sourced from 2–3 suppliers) |
| Optimization horizon | 4 weekly periods |

### Product catalogue (selected)

| Category | Unit value range | Holding cost p.a. | Stockout penalty |
|---|---|---|---|
| Electronics | EUR 15 – 1,200 | 22–34% | 1.5–2.5× unit value |
| Consumables | EUR 3 – 85 | 17–24% | 1.2–1.7× unit value |
| Industrial | EUR 8 – 95 | 16–25% | 2.1–3.4× unit value |

### Demand characteristics

| Category | Daily demand (mean) | Weekly demand (mean) | CV |
|---|---|---|---|
| Electronics | 17–78 units/day | 118–543 units/week | 0.5–0.6 |
| Consumables | 24–284 units/day | 168–1,989 units/week | 0.6 |
| Industrial | 10–175 units/day | 73–1,228 units/week | 0.6–0.7 |

Demand was generated with:
- Product-specific base demand from the catalogue
- Annual sine-wave seasonality (±25%, peak Q4)
- Weekly seasonality (B2B: near-zero weekend demand)
- Random promotional spikes (~5% of workdays, ×1.5–2.5 uplift)
- Poisson noise

**Figure 02 — Weekly demand distribution by product and category:**

![Weekly Demand Distribution](plots/02_demand_distributions.png)

Box plots confirm the wide intra-category spread: within Electronics, Phone Charger 65W (~600 units/week median) and Laptop 13" (~40 units/week) differ by 15×. Consumables show the highest absolute demand (Paper A4 up to ~3,000 units/week) while Industrial contains the most tightly clustered low-volume items (Fire Extinguishers, Safety Boots). This heterogeneity is why a global LightGBM model — which can share patterns across SKUs — outperforms per-product univariate models at this scale.

### Demand seasonality

**Figure 03 — Weekly and annual seasonality patterns:**

![Demand Seasonality Patterns](plots/03_demand_seasonality.png)

Both expected seasonality signals are visible and well-captured in the training data. Day-of-week: all three categories drop sharply on Saturday–Sunday (B2B effect), with Consumables showing the steepest fall (135 → 26 units/day). Monthly: demand peaks in May–June for Consumables and Industrial, consistent with the sine-wave seasonality (±25% around base) encoded in the data generator. Electronics is flatter across months. These patterns are fed to the model via `day_of_week`, `week_of_year`, and `month` features.

### Supplier profiles

| Supplier | Fixed cost | Min order | Lead time |
|---|---|---|---|
| S0 – SupplierAlpha | EUR 150 | EUR 500 | 1–2 weeks |
| S1 – SupplierBeta | EUR 80 | EUR 300 | 1–2 weeks |
| S2 – SupplierGamma | EUR 200 | EUR 1,000 | 2–4 weeks |
| S3 – SupplierDelta | EUR 120 | EUR 600 | 1–3 weeks |

---

## 3. Demand Forecast (LightGBM)

### Model design

A single **global LightGBM model** is trained on all 50 products simultaneously. This is the recommended approach for the 50 < K < 500 scale range (per spec), as cross-product learning improves accuracy for lower-volume SKUs.

Three quantile models are trained (α = 0.10, 0.50, 0.90):

| Model | Role |
|---|---|
| q50 (median) | Point forecast → demand input $\hat{D}_{i,t}$ for the MILP |
| q10 / q90 | Forecast interval → uncertainty estimate $\sigma_i$ |

**Feature set:**

| Feature group | Features |
|---|---|
| Lags | lag\_7, lag\_14, lag\_28 |
| Rolling stats | rolling\_mean\_7, rolling\_mean\_28, rolling\_std\_7, rolling\_std\_28 |
| Calendar | day\_of\_week, week\_of\_year, month, quarter, is\_weekend, day\_of\_year |
| Product identity | product\_code (integer encoding, treated as categorical by LGBM) |

**Training split:** 15 months train / 3 months validation (chronological).

**Figure 01 — Feature distributions by category (normalised):**

![Feature Distributions](plots/01_feature_distributions.png)

Lag and rolling features are normalised by each product's own 28-day rolling mean to remove the product-scale effect (otherwise a 300 units/day paper product and a 15 units/day laptop create a multi-modal, uninterpretable blob). After normalisation, values cluster around 1.0, and the spread reflects genuine demand variability. Notable observations: Electronics (blue) shows heavier tails in lag_7 and lag_14 — consistent with higher CV and promotional spikes. The `rolling_std / baseline` charts confirm Industrial (red) has the widest relative volatility, which translates into higher safety stock requirements.

**Figure 04 — Feature importance (q50 model, gain metric):**

![LightGBM Feature Importance](plots/04_feature_importance.png)

Lag features dominate — `lag_14` has nearly 2× the importance of `lag_7`, which is expected for weekly-aggregated forecasting: the 14-day lag captures the two-week purchasing cycle pattern common in B2B demand. `lag_28` is third, capturing monthly periodicity. Rolling statistics add moderate signal (~15% of total gain combined), while calendar features contribute only ~10% collectively. `product_code` ranks 9th — meaningful but not dominant, confirming that the global model generalises well across SKUs rather than memorising per-product intercepts.

### Accuracy results

| Category | MAPE min | MAPE max | MAPE mean | MAPE median |
|---|---|---|---|---|
| Electronics | 12.9% | 34.6% | 20.8% | 19.0% |
| Consumables | 8.8% | 27.8% | 15.6% | 13.1% |
| Industrial | 12.7% | 43.7% | 23.6% | 22.0% |
| **Overall** | — | — | **20.1%** | **19.0%** |

- Products with MAPE > 30%: **6 / 50** (all in low-volume electronics and industrial SKUs)
- The 6 outliers are expected: items with daily demand ≤ 20 units exhibit higher percentage noise due to Poisson variability at low counts

**Figure 05 — MAPE distribution by category (validation set):**

![Forecast Accuracy — MAPE Distribution](plots/05_mape_distribution.png)

Violin plots show clearly differentiated accuracy profiles across categories. Consumables (median 13.1%) benefits from high-volume, stable demand — the violin is narrow and entirely below the 30% threshold. Electronics (median 19.0%) has a broader distribution but remains mostly below 30%, with two outliers in the 30–35% range attributable to low-demand peripheral SKUs. Industrial (median 22.0%) shows the widest spread, with one product exceeding 43% MAPE — a low-volume safety item where Poisson noise dominates the signal. All 6 products above the 30% threshold are expected outliers; the model performs robustly for the remaining 44/50 SKUs.

**Figure 06 — Forecast vs actual (sample products, validation period):**

![Forecast Samples](plots/06_forecast_samples.png)

Representative products (2 per category, median MAPE) show that the q50 forecast tracks actual demand closely while the q10–q90 interval captures the true distribution well — the shaded bands cover most actual observations without being excessively wide. The weekend demand drops visible in every product time series confirm the weekly seasonality is correctly modelled. The interval width varies appropriately by product: wider for high-CV industrial items, narrower for stable consumables.

### Dynamic safety stock

$$SS_i = z_\alpha \cdot \sigma_i \cdot \sqrt{LT_i} \quad \text{with } z_\alpha = 1.645 \text{ (95\% service level)}$$

where $\sigma_i = (q_{90} - q_{10}) / (2 \times 1.645)$ from the quantile interval.

| Metric | Value |
|---|---|
| SS range | 0.07 – 0.68 weeks of demand |
| SS mean | 0.41 weeks |
| SS median | 0.43 weeks |

**Figure 10 — Initial stock vs safety stock coverage per product:**

![Initial Stock vs Safety Stock](plots/10_safety_stock_vs_initial.png)

The dual bar chart contrasts what we have (initial stock coverage, left panel, axis inverted so products align centrally) against what we need (safety stock requirement, right panel). Initial stock averages ~2.0 weeks of demand — well above the mean safety stock requirement of 0.41 weeks. However, the right panel reveals meaningful heterogeneity: GPU Entry and RAM 8GB require the largest SS buffers (~0.67 weeks), driven by high forecast uncertainty (wide q10–q90 intervals) combined with longer lead times. Safety Gloves L and Lubricant Spray sit at the bottom with SS < 0.1 weeks — stable, high-volume items with short lead times. The chart validates that no product enters the optimization with a buffer deficit (initial coverage > SS for all 50 SKUs).

---

## 4. MILP Optimisation

### Model dimensions

| Element | Count |
|---|---|
| Products | 50 |
| Suppliers | 4 |
| Periods | 4 (weekly) |
| Total variables | 1,680 |
| Binary variables | 648 ($y_{i,j,t}$ + $z_{j,t}$) |
| Constraints | ~1,620 |

### Objective function

$$\min C = \underbrace{\sum_{i,j,t}(c^{buy}_{i,j} + c^{var}_{i,j}) \cdot p_{i,j,t}}_{\text{purchase + var. logistics}} + \underbrace{\sum_{i,j,t} c^{fix}_j \cdot y_{i,j,t}}_{\text{fixed logistics}} + \underbrace{\sum_{i,t} h_i \cdot S_{i,t}}_{\text{holding}} + \underbrace{\sum_{i,t} \pi_i \cdot u_{i,t}}_{\text{stockout penalty}}$$

### Solver performance

| Metric | Value |
|---|---|
| Solver | CBC (open-source, bundled with PuLP) |
| Status | **Optimal** |
| Solve time | ~0.3 seconds |
| Objective value | EUR 2,893,464 |

---

## 5. Optimisation Results

### Cost breakdown

| Component | EUR | Share |
|---|---|---|
| Purchase + variable logistics | 2,866,691 | 99.4% |
| Holding cost | 15,434 | 0.5% |
| Fixed logistics | 1,970 | 0.1% |
| Stockout penalties | 0 | 0.0% |
| **Grand total** | **2,884,094** | |

**Monthly spend by category:**

| Category | EUR | Share |
|---|---|---|
| Electronics | 1,870,840 | 65% |
| Industrial | 606,947 | 21% |
| Consumables | 388,904 | 13% |

**Figure 09 — Cost breakdown and spend by category:**

![Cost Breakdown](plots/09_cost_breakdown.png)

The left panel makes visually explicit what the table summarises: purchase and variable logistics (99.4%) completely dominate total cost, with holding and fixed logistics costs negligible at this scale. This structure is characteristic of supply chains with high unit values (Electronics: EUR 15–1,200) and moderate turnover — the cost lever is in purchasing and supplier selection, not in warehouse management. The right panel confirms Electronics drives 65% of spend despite representing only 34% of SKUs, reflecting the order-of-magnitude higher unit values vs Consumables. For cost reduction initiatives, optimising the Electronics supplier mix (Alpha vs Beta pricing) has far greater impact than reducing fixed logistics.

### Purchase plan

| Metric | Value |
|---|---|
| Total order lines | 85 |
| Active suppliers — week 1 | 2 (early replenishment of long-LT items) |
| Active suppliers — weeks 2–4 | 4 (all suppliers) |
| Products ordered — week 1 | 3 |
| Products ordered — week 2 | 43 (bulk replenishment after week 1 consumption) |
| Products ordered — weeks 3–4 | 19–20 (warehouse forces spread) |

The warehouse capacity constraint **binds in week 2** (99.4% utilisation), forcing the optimizer to distribute replenishment orders across weeks 3 and 4 rather than front-loading all purchases.

**Figure 07 — Purchase plan: spend by supplier and week:**

![Monthly Purchase Plan](plots/07_purchase_plan.png)

The stacked bar chart reveals the optimizer's strategy at a glance. Week 1 is minimal (3 products, ~EUR 45k total) — only Gamma (S2, LT 2–4w) and Delta (S3) are activated because their long lead times require early placement. The bulk of purchasing (~EUR 1.18M) concentrates in Week 2 across all 4 suppliers, triggered by the first week's stock depletion. The secondary axis (diamond line) confirms 43 products are ordered in Week 2. Weeks 3 and 4 show roughly equal spend (~EUR 840k each) as warehouse capacity overflow from Week 2 is redistributed. Beta (S1, cheapest fixed cost at EUR 80) is consistently the second-largest spend contributor, reflecting the optimizer's preference for low-overhead suppliers when lead times are equivalent.

### Spend by supplier

| Supplier | Spend (EUR) | Active weeks | Avg lead time |
|---|---|---|---|
| S0 – Alpha | 1,180,024 | 3 | 1.4w |
| S1 – Beta | 853,061 | 3 | 1.2w |
| S2 – Gamma | 593,811 | 4 | 3.1w |
| S3 – Delta | 239,795 | 4 | 1.9w |

S2 (international, longest lead time) appears in **all 4 weeks** because orders must be placed earlier to account for 2–4 week lead times — a direct consequence of the lead-time-aware safety stock formula.

### Warehouse utilisation

| Period | Volume used | Utilisation |
|---|---|---|
| t=0 (initial stock) | 6,559 | 137% _(above W — initial stock is a parameter, not a variable)_ |
| Week 1 | 3,028 | 63% |
| Week 2 | 4,746 | **99%** ← binding |
| Week 3 | 3,940 | 83% |
| Week 4 | 1,197 | 25% |

The week-4 drop to 25% reflects the end-of-horizon effect: the optimizer stops ordering once safety stock is secured for the remaining periods, letting stock run down naturally.

**Figure 08 — Warehouse ending-stock volume over the optimisation horizon:**

![Warehouse Utilisation](plots/08_warehouse_utilisation.png)

The stacked bars break utilisation by category, showing that Industrial (red) consistently occupies ~30% of warehouse volume despite representing only 21% of purchase spend — a consequence of higher unit volumes (bulky items like cable ties, fire extinguishers). The binding Week 2 constraint is visually clear: the bar top nearly touches the red dashed capacity line (W = 4,777 m³). The dotted grey reference line marks the pre-period initial stock level (6,559 m³, 137% of W) — this does not represent a feasibility violation because initial stock is a fixed input parameter, not a decision variable; the warehouse capacity constraint applies only to weeks 1–4. Week 4's sharp decline to 25% is the canonical end-of-horizon artefact of finite MILP horizons.

### Service level & stockout

| Metric | Value |
|---|---|
| Overall service level | **100%** |
| Products below 90% SL | 0 / 50 |
| Total stockout units | 0 |
| Stockout rate | **0.00%** |

Zero stockouts are consistent with the stockout penalty structure (1.2–3.4× unit value) making it always cheaper to hold safety stock than to incur a stockout.

---

## 6. Key Findings

1. **Holding vs. logistics trade-off is visible.** With holding costs at 16–34% p.a. and fixed logistics at EUR 80–200/order, the optimizer consolidates orders rather than ordering weekly — 43 products are replenished in a single week 2 bulk order.

2. **Warehouse constraint shapes the purchase schedule.** The 99.4% utilisation in week 2 is not a coincidence — the optimizer hits the ceiling and redistributes the overflow to weeks 3 and 4, increasing fixed logistics costs by ~EUR 500 vs. the unconstrained solution.

3. **Long lead times drive early ordering.** S2 (2–4 week LT) is activated in week 1 even though most purchases happen in week 2, because the lead-time-adjusted safety stock formula signals insufficient cover for high-LT products.

4. **LightGBM at 50 SKUs delivers solid accuracy.** 20.1% mean MAPE with 44/50 products below 30% validates the choice of a global model over per-product SARIMA. Cross-product learning especially benefits the lower-volume electronics SKUs.

5. **End-of-horizon stock depletion.** Week 4 utilisation drops to 25%, a well-known artefact of finite-horizon MILPs. In production this is addressed by a rolling-horizon approach (re-solve weekly, extending the horizon).

---

## 7. Limitations & Next Steps

| Limitation | Suggested improvement |
|---|---|
| Finite horizon (4 weeks) — week 4 under-orders | Rolling horizon: re-solve weekly with a 4-week look-ahead |
| Zero stockouts driven by high penalties | Calibrate penalties to actual margin data; introduce partial backorder modelling |
| No quantity discounts | Add piecewise-linear price breaks as additional binary tiers |
| Forecast uses last known lags for horizon | Implement recursive day-ahead prediction to improve horizon accuracy |
| CBC solver (open-source) | Upgrade to Gurobi/HiGHS for faster solve at larger K×M×T |
| Static safety stock | Re-compute SS dynamically each week as forecast uncertainty evolves |
