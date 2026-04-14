# Informe Técnico — Supply Chain Optimization
**Prueba Técnica · Valerdat**

| | |
|---|---|
| **Autor** | Gerard Llobet Pérez |
| **Fecha** | Abril 2026 |
| **Stack** | Python · LightGBM · PuLP + CBC · Synthetic data |
| **Repositorio** | github.com/gerardlp/valerdat-supply-chain |

---

## 1. Descripción del Problema

El objetivo de la prueba es determinar el conjunto óptimo de compras $P = \{p_{i,j,t}\}$ para un portafolio de $K$ productos gestionados con $M$ proveedores a lo largo de un horizonte temporal de 1 mes, de forma que se minimice el coste total respetando todas las restricciones del sistema: inventario, MOQ, pedido mínimo por proveedor y capacidad de almacén.

### 1.1 Decisión de diseño: pipeline en dos módulos

El problema tiene dos naturalezas distintas que no conviene mezclar en un único modelo. Una parte es estadística —predecir la demanda futura con cuantificación de incertidumbre— y otra es de optimización combinatoria —decidir qué comprar, a quién y cuándo minimizando coste. Se diseñó un pipeline de dos módulos independientes con un handoff limpio de datos:

```
Historical demand (18 months daily)
        ↓
Feature Engineering  →  LightGBM Forecast (q10 / q50 / q90)
                                   ↓
                     Dynamic Safety Stock  SSᵢ = z_α · σᵢ · √LTᵢ
                                   ↓
                         MILP Optimizer (PuLP + CBC)
                                   ↓
                       Purchase Plan  P_{i,j,t}  +  KPI Report
```

Esta separación implica que cambiar el modelo de forecast no requiere modificar el optimizador, y viceversa.

### 1.2 Parámetros del sistema

| Parámetro | Descripción |
|---|---|
| $D_{i,t}$ | Demanda esperada del producto $i$ en período $t$ |
| $S_{i,0}$ | Stock inicial del producto $i$ |
| $SS_i$ | Safety stock del producto $i$ |
| $c^{buy}_{i,j}$ | Coste de compra del producto $i$ al proveedor $j$ |
| $c^{fix}_{j}$ | Coste fijo logístico por pedido al proveedor $j$ |
| $c^{var}_{i,j}$ | Coste variable logístico por unidad |
| $h_i$ | Coste de holding (amortización) del producto $i$ |
| $\pi_i$ | Penalización por stockout del producto $i$ |
| $MOQ_{i,j}$ | Cantidad mínima de pedido del producto $i$ al proveedor $j$ |
| $MinOrder_j$ | Pedido mínimo en euros al proveedor $j$ |
| $W$ | Capacidad total del almacén (m³) |

### 1.3 Variables de decisión

- $p_{i,j,t} \geq 0$ — cantidad comprada del producto $i$ al proveedor $j$ en período $t$ (continua)
- $y_{i,j,t} \in \{0,1\}$ — binaria: ¿se activa el pedido del producto $i$ al proveedor $j$ en $t$?
- $z_{j,t} \in \{0,1\}$ — binaria: ¿se activa el proveedor $j$ en el período $t$?
- $u_{i,t} \geq 0$ — unidades de stockout del producto $i$ en período $t$
- $S_{i,t} \geq 0$ — stock al final del período $t$ para el producto $i$

### 1.4 Función objetivo (MILP)

$$\min\; C = \underbrace{\sum_{i,j,t}\!\left(c^{buy}_{i,j} + c^{var}_{i,j}\right) p_{i,j,t}}_{\text{compra + logística variable}} + \underbrace{\sum_{i,j,t} c^{fix}_{j}\, y_{i,j,t}}_{\text{logística fija}} + \underbrace{\sum_{i,t} h_i\, S_{i,t}}_{\text{holding}} + \underbrace{\sum_{i,t} \pi_i\, u_{i,t}}_{\text{stockout}}$$

### 1.5 Restricciones

**Balance de inventario:**
$$S_{i,t} = S_{i,t-1} + \sum_j p_{i,j,t} - D_{i,t} + u_{i,t} \geq 0 \qquad \forall\, i,\, t$$

**Safety stock mínimo:**
$$S_{i,t} \geq SS_i \qquad \forall\, i,\, t$$

**MOQ — Big-M linking:**
$$p_{i,j,t} \geq MOQ_{i,j} \cdot y_{i,j,t}$$
$$p_{i,j,t} \leq M \cdot y_{i,j,t}$$

**Pedido mínimo por proveedor:**
$$\sum_i c^{buy}_{i,j}\, p_{i,j,t} \geq MinOrder_j \cdot z_{j,t} \qquad \forall\, j,\, t$$

**Capacidad de almacén:**
$$\sum_i v_i\, S_{i,t} \leq W \qquad \forall\, t$$

---

## 2. Dataset Sintético

Al no disponer de datos reales de cliente, se generó un dataset sintético representativo de un entorno B2B.

**KPIs del dataset:** 50 productos · 4 proveedores · 547 días de histórico (~18 meses) · 27,350 registros de demanda diaria

### 2.1 Catálogo de productos

| Categoría | Rango unit value | Holding cost p.a. | Penalización stockout |
|---|---|---|---|
| Electronics | EUR 15 – 1,200 | 22–34% | 1.5–2.5× unit value |
| Consumables | EUR 3 – 85 | 17–24% | 1.2–1.7× unit value |
| Industrial | EUR 8 – 95 | 16–25% | 2.1–3.4× unit value |

### 2.2 Características de la demanda

La demanda fue generada con estacionalidad anual de onda sinusoidal (±25%, pico Q4), estacionalidad semanal B2B (demanda casi nula en fin de semana), spikes promocionales aleatorios (~5% de días laborables, ×1.5–2.5) y ruido de Poisson. El coeficiente de variación (CV) oscila entre 0.5 y 0.7.

| Categoría | Demanda diaria (media) | Demanda semanal (media) | CV |
|---|---|---|---|
| Electronics | 17–78 u/día | 118–543 u/semana | 0.5–0.6 |
| Consumables | 24–284 u/día | 168–1,989 u/semana | 0.6 |
| Industrial | 10–175 u/día | 73–1,228 u/semana | 0.6–0.7 |

### 2.3 Perfiles de proveedores

| Proveedor | Coste fijo | Pedido mínimo | Lead time |
|---|---|---|---|
| S0 – SupplierAlpha (doméstico) | EUR 150 | EUR 500 | 1–2 semanas |
| S1 – SupplierBeta (regional) | EUR 80 | EUR 300 | 1–2 semanas |
| S2 – SupplierGamma (internacional) | EUR 200 | EUR 1,000 | 2–4 semanas |
| S3 – SupplierDelta (especialista) | EUR 120 | EUR 600 | 1–3 semanas |

---

## 3. Módulo de Forecast (LightGBM)

### 3.1 Diseño del modelo

Se entrenó un único modelo LightGBM global sobre los 50 productos simultáneamente con **quantile regression**, produciendo tres modelos independientes:

| Modelo | Rol |
|---|---|
| $q_{50}$ (mediana) | Forecast puntual → entrada $\hat{D}_{i,t}$ al MILP |
| $q_{10}$ / $q_{90}$ | Intervalo de forecast → estimación de incertidumbre $\sigma_i$ |

La elección de un modelo global frente a 50 modelos per-SKU está motivada por el **cross-learning**: los SKUs de bajo volumen, donde el ruido Poisson domina la señal, se benefician de los patrones aprendidos de los SKUs de alto volumen de la misma categoría. Para la escala $50 < K < 500$ es el enfoque recomendado.

### 3.2 Feature engineering

| Grupo de features | Features |
|---|---|
| Lags | `lag_7`, `lag_14`, `lag_28` |
| Rolling stats | `rolling_mean_7`, `rolling_mean_28`, `rolling_std_7`, `rolling_std_28` |
| Calendario | `day_of_week`, `week_of_year`, `month`, `quarter`, `is_weekend`, `day_of_year` |
| Identidad producto | `product_code` (int encoding, tratado como categórico por LGBM) |

Split de entrenamiento: **15 meses train / 3 meses validación** (cronológico). Todas las features de lag y rolling se computan con `shift(1)` previo al rolling para evitar data leakage en entrenamiento.

### 3.3 Inferencia recursiva

Durante la inferencia sobre el horizonte de 4 semanas se aplica **forecasting recursivo**: tras predecir el día $t$, el $q_{50}$ predicho se añade al histórico de trabajo para que $\text{lag}_7$ del día $t+7$ use ese valor y no el último día conocido del histórico estático. Esto evita la propagación de lags obsoletos a lo largo del horizonte, especialmente relevante a partir de la semana 2.

### 3.4 Precisión — resultados de validación

**MAPE global: 20.1% (media) · 19.0% (mediana) · 44/50 productos < 30%**

| Categoría | MAPE min | MAPE max | MAPE media | MAPE mediana |
|---|---|---|---|---|
| Electronics | 12.9% | 34.6% | 20.8% | 19.0% |
| Consumables | 8.8% | 27.8% | 15.6% | 13.1% |
| Industrial | 12.7% | 43.7% | 23.6% | 22.0% |
| **Global** | — | — | **20.1%** | **19.0%** |

Los 6 productos por encima del 30% son outliers esperados: ítems con demanda diaria $\leq 20$ unidades donde el ruido Poisson domina la señal. Los 44 restantes rinden de forma robusta.

---

## 4. Safety Stock Dinámico

El safety stock conecta el módulo de forecast con el optimizador: traduce la incertidumbre del forecast en un buffer de inventario que el MILP está obligado a mantener en cada período.

### 4.1 Fórmula

$$SS_i = z_\alpha \cdot \sigma_i \cdot \sqrt{LT_i} \qquad \text{con } z_\alpha = 1.645 \text{ (95\% service level)}$$

El nivel de servicio óptimo por SKU se derivaría del problema del vendedor de periódico (*newsvendor*):

$$SL^* = \frac{C_{stockout}}{C_{stockout} + C_{holding}}$$

El 95% global es un punto de partida conservador calibrable con márgenes reales.

### 4.2 Sigma híbrida

La $\sigma_i$ no se toma directamente del intervalo cuantílico del modelo. Se usa una **sigma híbrida** que toma el máximo de dos estimaciones:

$$\sigma_i = \max\!\left(\,\sigma^{quantile}_i,\;\sigma^{empirical}_i\right)$$

| Componente | Definición | Cuándo domina |
|---|---|---|
| $\sigma^{quantile}_i$ | $\displaystyle\sqrt{\sum_{d=1}^{7} \left(\frac{q_{90,d} - q_{10,d}}{2 \times 1.645}\right)^{\!2}}$ — suma en cuadratura de sigmas diarias | Modelo bien calibrado |
| $\sigma^{empirical}_i$ | $\text{std}\!\left(\hat{e}_{i,w}\right)$ sobre el set de validación — error real observado | Modelo optimista (intervalos estrechos) |
| $\sigma^{hybrid}_i = \max(\sigma^q_i, \sigma^e_i)$ | Toma el mayor de los dos | Siempre activo |

La suma en cuadratura sobre 7 días es matemáticamente correcta para agregar desviaciones estándar de variables independientes:

$$\sigma\!\left(\sum_{d} X_d\right) = \sqrt{\sum_{d} \sigma_d^2}$$

Usar suma directa sobreestimaría el SS por un factor $\sqrt{7} \approx 2.65$.

### 4.3 Validación empírica

Se validó que el SS cubre efectivamente el $p_{95}$ del error semanal de forecast:

$$\text{ratio}_i = \frac{SS_i}{p_{95}\!\left(|\hat{e}_{i,w}|\right)}$$

| Estadístico | Valor (sin híbrida) | Valor (con híbrida) |
|---|---|---|
| Ratio mediano | 0.87× | **1.46×** |
| Ratio mínimo | 0.22× | **0.84×** |
| Productos con ratio $\geq 1.0$ | 50% | **~75%** |

Sin la sigma híbrida, el 50% de los productos quedaban por debajo del objetivo de cobertura.

---

## 5. Módulo MILP — Optimizador

### 5.1 Dimensiones del modelo

**1,680 variables totales · 648 binarias ($y_{i,j,t}$ + $z_{j,t}$) · ~1,620 restricciones · Resuelto en 0.3s**

| Elemento | Valor |
|---|---|
| Productos ($K$) | 50 |
| Proveedores ($M$) | 4 |
| Períodos ($T$) | 4 (semanas) |
| Solver | CBC (open-source, bundled con PuLP) |
| Status | **Optimal** |
| Valor objetivo | EUR 2,116,247 |

### 5.2 Justificación del solver

CBC es open-source y bundled con PuLP, sin dependencias comerciales para el prototipo. Para este tamaño (1,680 variables, 648 binarias) resuelve en 0.3 segundos. En producción con $K \times M \times T$ más grande, el paso natural es **Gurobi** o **HiGHS**: mejor branching heurístico y paralelización. El diseño del modelo MILP es idéntico — cambiar el solver es una línea de código en PuLP.

---

## 6. Resultados de la Optimización

### 6.1 Desglose de coste

| Componente | EUR | Peso |
|---|---|---|
| Compra + logística variable | 2,100,960 | 99.3% |
| Coste de holding | 13,602 | 0.6% |
| Logística fija | 1,650 | 0.1% |
| Penalización stockout | 35 | 0.0% |
| **TOTAL** | **2,116,247** | **100%** |

El coste de compra y logística variable domina al 99.3%. La palanca de ahorro real está en la negociación del precio de compra con los proveedores de Electronics (65% del gasto), no en reducir costes logísticos o de holding.

### 6.2 Plan de compras

Con la sigma híbrida elevando los niveles de safety stock, el stock inicial es suficiente para cubrir la demanda de la semana 1 sin activar ningún pedido. Todo el reaprovisionamiento se concentra en las semanas 2–4:

| Semana | Productos pedidos | Proveedores activos | Descripción |
|---|---|---|---|
| Semana 1 | 0 | 0 | Stock inicial cubre la demanda sin compras |
| Semana 2 | 31 | 4 | Bulk order — almacén al 99% de capacidad |
| Semana 3 | 22 | 4 | Overflow del almacén de semana 2 |
| Semana 4 | 18 | 4 | Cierre de horizonte (efecto end-of-horizon) |

### 6.3 Gasto por proveedor

| Proveedor | Gasto (EUR) | Semanas activas |
|---|---|---|
| S0 – Alpha | 918,413 | 3 (semanas 2–4) |
| S1 – Beta | 616,475 | 3 (semanas 2–4) |
| S2 – Gamma | 414,035 | 3 (semanas 2–4) |
| S3 – Delta | 152,037 | 3 (semanas 2–4) |

### 6.4 Utilización del almacén

| Período | Volumen usado (m³) | Utilización | Observación |
|---|---|---|---|
| $t=0$ (stock inicial) | 6,559 | 137% | Parámetro fijo — no sujeto a restricción $W$ |
| Semana 1 | 3,028 | 63% | Consumo sin compras |
| Semana 2 | 4,746 | **99%** | ⚠ Restricción binding |
| Semana 3 | 3,940 | 83% | Overflow redistributed |
| Semana 4 | 1,197 | 25% | Efecto end-of-horizon |

La restricción de almacén activa en la semana 2 (99.4%) no es coincidencia: el optimizador quería consolidar más pedidos en esa semana para minimizar los costes fijos de logística, pero el techo $W$ lo impide y redistribuye el exceso a las semanas 3 y 4.

El stock inicial superior a $W$ (137%) es un parámetro fijo de entrada, no una variable de decisión, por lo que no representa una infeasibility del modelo.

### 6.5 Nivel de servicio

**Service level global: 100% · 0/50 productos por debajo del 90% SL · 5.2 unidades de stockout (0.01% — artefacto de redondeo CBC)**

---

## 7. Conclusiones y Hallazgos Clave

- **Trade-off holding vs. logística visible.** Con holding costs al 16–34% p.a. y logística fija EUR 80–200/pedido, el optimizador consolida en lugar de ordenar semanalmente. 31 productos se reaprovisionan en un único bulk order en semana 2.

- **La restricción de almacén conforma el plan de compras.** El 99.4% de utilización en semana 2 es el resultado directo del optimizador maximizando la consolidación hasta el techo $W$, redistribuyendo el exceso a semanas 3 y 4.

- **LightGBM global a 50 SKUs: 20.1% MAPE con 44/50 productos bajo el 30%.** El cross-product learning beneficia especialmente a los SKUs de bajo volumen de Electronics e Industrial.

- **La sigma híbrida cierra el gap de cobertura.** Sin el fix, el 50% de los productos tenían un ratio $SS / p_{95}\text{-error} < 0.87\times$. Con $\max(\sigma^{quantile}, \sigma^{empirical})$, el ratio mediano sube a $1.46\times$ y el mínimo a $0.84\times$.

- **Inferencia recursiva elimina el sesgo de lags estáticos.** El forecast estático reutilizaba el último lag conocido (diciembre, pico de demanda) para todo el horizonte. El método recursivo recalcula lags correctamente para enero, reduciendo la demanda estimada en ~15–20%.

- **Efecto end-of-horizon en semana 4** (25% utilización). Artefacto clásico de MILPs de horizonte finito. En producción se resuelve con rolling horizon: re-resolver semanalmente con look-ahead de 4 semanas.

---

## 8. Limitaciones y Próximos Pasos

| Limitación | Mejora propuesta |
|---|---|
| Horizonte finito de 4 semanas — semana 4 infra-ordena | Rolling horizon: re-resolver semanalmente con look-ahead de 4 semanas |
| 5.2 unidades de stockout por redondeo de CBC | Ajustar tolerancia de integralidad o post-procesar la solución entera |
| Sin descuentos por volumen | Añadir price breaks con tiers binarios adicionales en el MILP |
| CBC lento (~200s) a la escala actual | Upgrade a Gurobi o HiGHS para $K \times M \times T$ mayor |
| $\sigma^{empirical}$ calculada sobre ventana fija de 3 meses | Usar ventana deslizante o expanding window; ponderar errores recientes más |
| Datos sintéticos — sin validación en datos reales | Calibrar penalizaciones de stockout y holding con márgenes reales de cliente |

---

## 9. Stack Técnico

| Módulo | Tecnología | Función |
|---|---|---|
| Data ingestion | pandas / CSV | Carga y limpieza de datos históricos |
| Feature engineering | pandas (lags, rolling, calendar) | Construcción del feature matrix |
| Forecast model | LightGBM (quantile regression) | $q_{10}$ / $q_{50}$ / $q_{90}$ sobre 50 SKUs global |
| Safety stock | NumPy (cuadratura + sigma híbrida) | SS dinámico por producto y período |
| MILP solver | PuLP + CBC | Optimización del plan de compras |
| Reporting | pandas + tabulate | KPI report y métricas de validación |
| Visualización | matplotlib | 10 plots de análisis y resultados |
| Producción (propuesto) | Gurobi / HiGHS | Solver para escala $K \times M \times T$ mayor |
