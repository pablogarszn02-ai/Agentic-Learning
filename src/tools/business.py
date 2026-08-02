"""Herramientas: contexto de negocio (materialidad, tendencia de largo plazo)."""
import numpy as np
import pandas as pd
from src.data_loader import monthly_all, store_monthly_totals, get_group


def get_department_materiality(store_id: str, dept_id: str, year_month: str) -> dict:
    """% que representa un departamento del total de ventas de la tienda ese mes."""
    store_id, dept_id, year_month = int(store_id), int(dept_id), year_month[:7]
    period = pd.Period(year_month, freq="M")

    dept_row = monthly_all[
        (monthly_all["Store"] == store_id)
        & (monthly_all["Dept"] == dept_id)
        & (monthly_all["year_month"] == period)
    ]
    if dept_row.empty:
        return {"error": "no se encontró ese depto/mes"}
    dept_sales = float(dept_row.iloc[0]["monthly_sales"])

    store_total = store_monthly_totals.get((store_id, period))
    if store_total is None:
        return {"error": "no se encontró el total de la tienda para ese mes"}

    contribution_pct = dept_sales / store_total * 100
    return {
        "department_sales": round(dept_sales, 2),
        "store_total_sales_same_month": round(float(store_total), 2),
        "department_contribution_pct": round(contribution_pct, 2),
        "interpretation": (
            "Alta materialidad (>10% del total de la tienda)" if contribution_pct > 10
            else "Materialidad media (1-10%)" if contribution_pct > 1
            else "Baja materialidad (<1% del total de la tienda)"
        ),
    }


def get_sales_trend(store_id: str, dept_id: str) -> dict:
    """Tendencia de largo plazo (regresión lineal) de una tienda+departamento."""
    store_id, dept_id = int(store_id), int(dept_id)
    rows = get_group(store_id, dept_id)
    if len(rows) < 6:
        return {"error": "no hay suficiente historial para calcular una tendencia confiable (mínimo 6 meses)"}

    x = rows["month_index"].values.astype(float)
    y = rows["monthly_sales"].values.astype(float)
    x_norm = x - x.min()

    slope, _intercept = np.polyfit(x_norm, y, 1)
    avg_sales = y.mean()
    monthly_growth_pct = (slope / avg_sales) * 100 if avg_sales != 0 else 0

    direction = "creciendo" if monthly_growth_pct > 0.5 else "decreciendo" if monthly_growth_pct < -0.5 else "estable"

    return {
        "trend_direction": direction,
        "avg_monthly_change_pct": round(float(monthly_growth_pct), 2),
        "avg_monthly_sales_overall": round(float(avg_sales), 2),
        "months_analyzed": len(rows),
    }
