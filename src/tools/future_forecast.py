"""Herramienta del Agente Forecaster: predicción de ventas hacia el futuro."""
from src.ml.forecast_future import forecast_future, DEFAULT_HORIZON_WEEKS


def get_future_forecast(store_id: str, dept_id: str, horizon_weeks: str = None) -> dict:
    store_id, dept_id = int(store_id), int(dept_id)
    horizon = int(horizon_weeks) if horizon_weeks else DEFAULT_HORIZON_WEEKS
    return forecast_future(store_id, dept_id, horizon_weeks=horizon)