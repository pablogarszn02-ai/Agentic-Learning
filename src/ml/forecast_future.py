"""
Forecasting recursivo hacia el futuro (horizonte configurable, default 1 trimestre = 13 semanas).

Cómo funciona (importante entender esto, no es "magia"):
- Cada semana futura usa como sales_lag_1 la PREDICCIÓN de la semana anterior
  (no existe un dato real todavía) -> el error se puede ir acumulando semana a semana.
- Variables externas del futuro (clima, combustible, CPI, desempleo, festivo) NO EXISTEN
  aún -> se usa como proxy el valor real de la MISMA semana del año anterior (asumiendo
  que el clima/economía no cambia bruscamente de un año a otro).
- La incertidumbre reportada es una HEURÍSTICA simple (crece con cada paso), no un
  intervalo estadístico riguroso -- suficiente para comunicar "confía menos en las
  semanas lejanas", pero no reemplaza un modelo probabilístico formal.
"""
import numpy as np
import pandas as pd
import joblib

from src.data_loader import weekly_all
from src.ml.features import FEATURES, build_features

MODEL_PATH = "models/demand_forecast_xgb_production.pkl"
DEFAULT_HORIZON_WEEKS = 13  # 1 trimestre

_bundle = joblib.load(MODEL_PATH)
_model = _bundle["model"]
_features_df = build_features(weekly_all)


def _get_last_known_row(store_id: int, dept_id: int):
    rows = _features_df[(_features_df["Store"] == store_id) & (_features_df["Dept"] == dept_id)].sort_values("Date")
    return rows.iloc[-1] if not rows.empty else None


def _lookup_same_week_last_year(store_id: int, dept_id: int, target_date, column: str):
    """Proxy para variables externas futuras: usa el valor real de la misma semana,
    un año antes (existe porque ya ocurrió)."""
    target_last_year = target_date - pd.Timedelta(weeks=52)
    row = weekly_all[
        (weekly_all["Store"] == store_id)
        & (weekly_all["Dept"] == dept_id)
        & (weekly_all["Date"] >= target_last_year - pd.Timedelta(days=3))
        & (weekly_all["Date"] <= target_last_year + pd.Timedelta(days=3))
    ]
    return row.iloc[0][column] if not row.empty else None


def forecast_future(store_id: int, dept_id: int, horizon_weeks: int = DEFAULT_HORIZON_WEEKS) -> dict:
    last_row = _get_last_known_row(store_id, dept_id)
    if last_row is None:
        return {"error": f"no hay historial para store={store_id}, dept={dept_id}"}

    history = _features_df[(_features_df["Store"] == store_id) & (_features_df["Dept"] == dept_id)].sort_values("Date")
    recent_sales = list(history["Weekly_Sales"].tail(4))
    last_date = last_row["Date"]

    predictions = []
    for step in range(1, horizon_weeks + 1):
        future_date = last_date + pd.Timedelta(weeks=step)

        sales_lag_1 = predictions[-1]["predicted_sales"] if predictions else float(last_row["Weekly_Sales"])
        rolling_mean_4 = float(np.mean(recent_sales[-4:]))

        sales_lag_52 = _lookup_same_week_last_year(store_id, dept_id, future_date, "Weekly_Sales")
        if sales_lag_52 is None:
            sales_lag_52 = rolling_mean_4  # respaldo si no hay dato de hace un año

        temp = _lookup_same_week_last_year(store_id, dept_id, future_date, "Temperature")
        fuel = _lookup_same_week_last_year(store_id, dept_id, future_date, "Fuel_Price")
        cpi = _lookup_same_week_last_year(store_id, dept_id, future_date, "CPI")
        unemp = _lookup_same_week_last_year(store_id, dept_id, future_date, "Unemployment")
        is_holiday = _lookup_same_week_last_year(store_id, dept_id, future_date, "IsHoliday")

        feature_row = pd.DataFrame([{
            "Store": store_id, "Dept": dept_id,
            "Type_encoded": last_row["Type_encoded"], "Size": last_row["Size"],
            "month": future_date.month, "week_of_year": int(future_date.isocalendar()[1]), "year": future_date.year,
            "IsHoliday": int(is_holiday) if is_holiday is not None else 0,
            "Temperature": temp if temp is not None else last_row["Temperature"],
            "Fuel_Price": fuel if fuel is not None else last_row["Fuel_Price"],
            "CPI": cpi if cpi is not None else last_row["CPI"],
            "Unemployment": unemp if unemp is not None else last_row["Unemployment"],
            "MarkDown1": 0, "MarkDown2": 0, "MarkDown3": 0, "MarkDown4": 0, "MarkDown5": 0,
            "sales_lag_1": sales_lag_1, "sales_lag_52": sales_lag_52, "rolling_mean_4": rolling_mean_4,
        }])[FEATURES]

        pred = max(float(_model.predict(feature_row)[0]), 0)  # las ventas no pueden ser negativas

        # Incertidumbre creciente (heurística, no estadística rigurosa):
        # semana 1 ~ ±8%, cada semana adicional suma ±3 puntos porcentuales
        uncertainty_pct = 8 + (step - 1) * 3

        predictions.append({
            "week": step,
            "date": str(future_date.date()),
            "predicted_sales": round(pred, 2),
            "uncertainty_pct": uncertainty_pct,
            "range_low": round(pred * (1 - uncertainty_pct / 100), 2),
            "range_high": round(pred * (1 + uncertainty_pct / 100), 2),
        })
        recent_sales.append(pred)

    monthly_totals = {}
    for p in predictions:
        month = p["date"][:7]
        monthly_totals[month] = round(monthly_totals.get(month, 0) + p["predicted_sales"], 2)

    return {
        "store_id": store_id,
        "dept_id": dept_id,
        "horizon_weeks": horizon_weeks,
        "weekly_forecast": predictions,
        "monthly_totals": monthly_totals,
        "methodology_note": (
            "Forecasting recursivo: cada semana usa la predicción anterior como insumo "
            "(no hay datos reales del futuro). Variables externas usan como proxy la "
            "misma semana del año pasado. La incertidumbre crece con cada semana -- "
            "confiar más en las primeras semanas que en las últimas del horizonte."
        ),
    }