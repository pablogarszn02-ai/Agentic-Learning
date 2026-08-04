"""Herramienta: predicción de demanda vía el modelo XGBoost entrenado (complementa la
detección de anomalías basada en reglas, no la reemplaza)."""
import joblib
import pandas as pd

from src.data_loader import weekly_all
from src.ml.features import build_features, FEATURES

MODEL_PATH = "models/demand_forecast_xgb.pkl"

# Se cargan/calculan UNA sola vez al importar este módulo (igual que data_loader.py)
_bundle = joblib.load(MODEL_PATH)
_model = _bundle["model"]
_features_df = build_features(weekly_all)


def get_ml_forecast(store_id: str, dept_id: str, year_month: str) -> dict:
    """Predice, semana por semana, las ventas de un mes usando el modelo entrenado,
    y las suma para comparar contra lo que realmente ocurrió."""
    store_id, dept_id, year_month = int(store_id), int(dept_id), year_month[:7]
    period = pd.Period(year_month, freq="M")

    rows = _features_df[
        (_features_df["Store"] == store_id)
        & (_features_df["Dept"] == dept_id)
        & (_features_df["Date"].dt.to_period("M") == period)
    ]
    if rows.empty:
        return {"error": f"no hay datos suficientes (features/lags) para store={store_id}, dept={dept_id}, mes={year_month}"}

    X = rows[FEATURES]
    preds = _model.predict(X)
    predicted_total = float(preds.sum())
    actual_total = float(rows["Weekly_Sales"].sum())
    deviation_pct = (actual_total - predicted_total) / predicted_total * 100 if predicted_total != 0 else None

    return {
        "actual_monthly_sales": round(actual_total, 2),
        "ml_predicted_monthly_sales": round(predicted_total, 2),
        "deviation_vs_ml_prediction_pct": round(deviation_pct, 2) if deviation_pct is not None else None,
        "weeks_used": len(rows),
        "note": (
            "La predicción del modelo usa como insumo ventas reales de semanas "
            "anteriores (momentum reciente + mismo periodo año pasado), no es una "
            "predicción 'a ciegas' — complementa, no reemplaza, la comparación por reglas."
        ),
    }