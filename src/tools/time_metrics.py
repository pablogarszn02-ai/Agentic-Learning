"""Herramientas: métricas temporales de negocio (Rolling Year, YTD)."""
import pandas as pd
from src.data_loader import get_group


def calculate_rolling_year(store_id: str, dept_id: str, as_of_month: str) -> dict:
    """RY: suma de los últimos 12 meses hasta as_of_month, vs. los 12 meses previos a esos."""
    store_id, dept_id, as_of_month = int(store_id), int(dept_id), as_of_month[:7]
    rows = get_group(store_id, dept_id)
    as_of_period = pd.Period(as_of_month, freq="M")

    current_window = rows[(rows["year_month"] > as_of_period - 12) & (rows["year_month"] <= as_of_period)]
    previous_window = rows[(rows["year_month"] > as_of_period - 24) & (rows["year_month"] <= as_of_period - 12)]

    if len(current_window) < 10 or len(previous_window) < 10:
        return {"error": "no hay suficientes meses de historia para calcular 2 Rolling Years completos"}

    current_ry = current_window["monthly_sales"].sum()
    previous_ry = previous_window["monthly_sales"].sum()
    deviation = (current_ry - previous_ry) / previous_ry * 100

    return {
        "as_of_month": as_of_month,
        "rolling_year_current_12m": round(float(current_ry), 2),
        "rolling_year_previous_12m": round(float(previous_ry), 2),
        "rolling_year_deviation_pct": round(float(deviation), 2),
    }


def calculate_ytd(store_id: str, dept_id: str, as_of_month: str) -> dict:
    """YTD: acumulado del año hasta as_of_month, vs. el mismo periodo del año anterior."""
    store_id, dept_id, as_of_month = int(store_id), int(dept_id), as_of_month[:7]
    as_of_period = pd.Period(as_of_month, freq="M")
    year, month = as_of_period.year, as_of_period.month

    rows = get_group(store_id, dept_id)
    ytd_current = rows[(rows["year"] == year) & (rows["calendar_month"] <= month)]["monthly_sales"].sum()
    ytd_previous = rows[(rows["year"] == year - 1) & (rows["calendar_month"] <= month)]["monthly_sales"].sum()

    if ytd_previous == 0:
        return {"error": "no hay YTD del año anterior disponible para comparar"}

    deviation = (ytd_current - ytd_previous) / ytd_previous * 100
    return {
        "as_of_month": as_of_month,
        "ytd_current_year": round(float(ytd_current), 2),
        "ytd_previous_year_same_period": round(float(ytd_previous), 2),
        "ytd_deviation_pct": round(float(deviation), 2),
    }
