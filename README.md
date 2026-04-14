# Prueba Técnica Valerdat — Supply Chain Optimization

**Fecha:** 2026-04-10
**Tags:** #valerdat #optimization #forecasting #supply-chain #MILP

---

## 1. Descripción del Problema

Optimización de compras para un portafolio de **K productos** gestionados con **M proveedores** a lo largo de un horizonte temporal (1 mes).

**Objetivo:** Determinar el conjunto óptimo de compras $P = \{p_{i,j,t}\}$ que minimice el coste total respetando todas las restricciones del sistema.

---

## 2. Parámetros del Sistema

| Parámetro       | Descripción                                                 |
| --------------- | ----------------------------------------------------------- |
| $D_{i,t}$       | Demanda esperada del producto $i$ en período $t$            |
| $S_{i,0}$       | Stock inicial del producto $i$                              |
| $SS_i$          | Safety stock del producto $i$                               |
| $c^{buy}_{i,j}$ | Coste de compra del producto $i$ al proveedor $j$           |
| $c^{fix}_{j}$   | Coste fijo logístico por pedido al proveedor $j$            |
| $c^{var}_{i,j}$ | Coste variable logístico por unidad                         |
| $h_i$           | Coste de holding (amortización) del producto $i$            |
| $\pi_i$         | Penalización por stockout del producto $i$                  |
| $MOQ_{i,j}$     | Cantidad mínima de pedido del producto $i$ al proveedor $j$ |
| $MinOrder_j$    | Pedido mínimo en euros al proveedor $j$                     |
| $W$             | Capacidad total del almacén                                 |

---

## 3. Variables de Decisión

- $p_{i,j,t} \geq 0$ — cantidad comprada del producto $i$ al proveedor $j$ en período $t$ (continua)
- $y_{i,j,t} \in \{0,1\}$ — binaria: ¿se activa el pedido del producto $i$ al proveedor $j$ en $t$?
- $z_{j,t} \in \{0,1\}$ — binaria: ¿se activa el proveedor $j$ en el período $t$?
- $u_{i,t} \geq 0$ — unidades de stockout del producto $i$ en período $t$
- $S_{i,t} \geq 0$ — stock al final del período $t$ para el producto $i$

---

## 4. Función Objetivo (MILP)

$$\min C = \sum_{i,j,t} c^{buy}_{i,j} \cdot p_{i,j,t} + \sum_{i,j,t} \left( c^{fix}_{j} \cdot y_{i,j,t} + c^{var}_{i,j} \cdot p_{i,j,t} \right) + \sum_{i,t} h_i \cdot S_{i,t} + \sum_{i,t} \pi_i \cdot u_{i,t}$$

---

## 5. Restricciones

### Balance de inventario
$$S_{i,t} = S_{i,t-1} + \sum_j p_{i,j,t} - D_{i,t} + u_{i,t} \geq 0$$

### Safety stock
$$S_{i,t} \geq SS_i \quad \forall i, t$$

### MOQ — Big-M linking
$$p_{i,j,t} \geq MOQ_{i,j} \cdot y_{i,j,t}$$
$$p_{i,j,t} \leq M \cdot y_{i,j,t}$$

### Pedido mínimo por proveedor (en euros)
$$\sum_i c^{buy}_{i,j} \cdot p_{i,j,t} \geq MinOrder_j \cdot z_{j,t}$$

### Capacidad de almacén
$$\sum_i v_i \cdot S_{i,t} \leq W \quad \forall t$$

---

## 6. Tipo de Problema y Solvers

El problema es un **MILP (Mixed Integer Linear Program)** por la presencia de variables binarias ($y_{i,j,t}$, $z_{j,t}$).

| Solver                  | Cuándo usar                     |
| ----------------------- | ------------------------------- |
| **HiGHS** (open source) | Prototipo y escalas medianas    |
| **Gurobi / CPLEX**      | Producción, K×M×T grande        |
| **OR-Tools (CP-SAT)**   | Alternativa open source robusta |

Stack en Python: `pulp` + HiGHS o `gurobipy` directamente.

---

## 7. Integración con el Forecast de Demanda

### Separación de responsabilidades

| Módulo | Herramienta | Pregunta que responde | Output |
|---|---|---|---|
| **Forecast** | LightGBM / Prophet | ¿Cuánto venderé? | $\hat{D}_{i,t}$, $\sigma_{i,t}$ |
| **Optimizer** | Gurobi / HiGHS | ¿Cuánto compro y a quién? | $p_{i,j,t}$ |

**El link entre módulos es un handoff de datos** — el optimizer consume el DataFrame de forecast como parámetro fijo.

### Safety stock derivado del forecast
$$SS_i = z_\alpha \cdot \sigma_i \cdot \sqrt{LT_i}$$

Donde $\sigma_i$ proviene del intervalo de confianza del modelo de forecast (quantile regression).

### Pipeline end-to-end

```
Raw data → Feature Engineering → LightGBM Forecast → D̂_{i,t} + σ_{i,t}
                                                              ↓
                                                    Safety Stock dinámico
                                                              ↓
                                                    MILP Optimizer (Gurobi)
                                                              ↓
                                                    Plan de compras P_{i,j,t}
                                                              ↓
                                                    Ventas reales observadas
                                                              ↓
                                                    Reentrenamiento + recalibración
```

### Elección de modelo de forecast según escala

| Escala | Modelo | Justificación |
|---|---|---|
| K < 50 productos | Prophet / SARIMA por producto | Interpretable, auditable |
| 50 < K < 500 | LightGBM global | Un modelo, features de producto |
| K > 500 | LightGBM + embeddings de producto | Cross-learning entre SKUs |

---

## 8. Stack Técnico Completo

```python
1. data_ingestion       → pandas / SQL
2. feature_engineering  → lags, rolling stats, calendar features, precio, promociones
3. forecast_model       → LightGBM con quantile regression (q10, q50, q90)
4. uncertainty_output   → D_hat + sigma por producto y período
5. optimizer_input      → DataFrame [product_id, period_t, demand, sigma, SS]
6. milp_solver          → PuLP + HiGHS (prototipo) / Gurobi (producción)
7. output               → plan de compras P_{i,j,t}
8. monitoring           → MAPE, bias, stockout rate, coste realizado
```

---
