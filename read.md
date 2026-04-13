❌ Bug: compute_empirical_sigma necesita pred_q50 pero no lo recibe
pythondef compute_empirical_sigma(val_df: pd.DataFrame) -> pd.Series:
    weekly_err = (
        val_df.groupby(["product_id", "week"])
        .agg(actual=("demand", "sum"), pred=("pred_q50", "sum"))  # ← necesita pred_q50
    )
La función espera que val_df ya tenga una columna pred_q50, pero eso no está garantizado — depende de que el caller la haya añadido antes. Si alguien llama compute_empirical_sigma con el feature_df crudo sin la columna, falla con KeyError.
La función debería ser autocontenida: recibir los modelos y generar ella misma las predicciones, o al menos validar que la columna existe. La forma más limpia:
pythondef compute_empirical_sigma(
    models: dict,
    feature_df: pd.DataFrame,
    val_months: int = 3,
) -> pd.Series:
    """
    Computes empirical weekly forecast error std per product on the
    validation set. Self-contained: generates predictions internally.
    """
    feature_df = feature_df.copy()
    feature_df["date"] = pd.to_datetime(feature_df["date"])

    cutoff = feature_df["date"].max() - pd.DateOffset(months=val_months)
    val_df = feature_df[feature_df["date"] > cutoff].copy()

    # Genera predicciones internamente — no depende del caller
    val_df["pred_q50"] = models[0.50].predict(val_df[FEATURE_COLS]).clip(0)

    val_df["week"] = val_df["date"].dt.to_period("W")
    weekly_err = (
        val_df.groupby(["product_id", "week"])
        .agg(actual=("demand", "sum"), pred=("pred_q50", "sum"))
        .reset_index()
    )
    weekly_err["error"] = weekly_err["actual"] - weekly_err["pred"]
    return weekly_err.groupby("product_id")["error"].std().rename("sigma_empirical")
Y en main.py el call queda limpio y sin duplicar código:
python# Antes (duplica la lógica de validación):
cutoff = feature_df["date"].max() - pd.DateOffset(months=3)
val_df = feature_df[feature_df["date"] > cutoff].copy()
val_df["pred_q50"] = models[0.50].predict(val_df[FEATURE_COLS]).clip(0)
empirical_sigma = compute_empirical_sigma(val_df)

# Después (autocontenido):
empirical_sigma = compute_empirical_sigma(models, feature_df, val_months=3)

⚠️ Detalle: la sigma empírica es semanal, la cuantílica es diaria agregada
python# sigma cuantílica: sqrt(Σσ_daily²) sobre 7 días → unidades: demanda/semana
# sigma empírica:   std(error_semanal)             → unidades: demanda/semana
weekly["sigma_empirical"] = weekly["product_id"].map(empirical_sigma).fillna(0)
weekly["sigma"] = weekly[["sigma", "sigma_empirical"]].max(axis=1)
Esto está correcto — ambas sigmas están expresadas en las mismas unidades (desviación estándar semanal) antes del max(). No hay problema de escala. Solo lo señalo para que lo tengas claro si te preguntan.