"""Herramienta: detección de anomalías mensuales (mes vs. mismo mes en años anteriores)."""
from src.data_loader import monthly_yoy


def find_biggest_monthly_anomaly() -> dict:
    """Utilidad interna (no es una tool de Claude): encuentra la peor anomalía de todo el dataset."""
    worst = monthly_yoy.loc[monthly_yoy["deviation_pct"].idxmin()]
    return {
        "store_id": int(worst["Store"]),
        "dept_id": int(worst["Dept"]),
        "year_month": str(worst["year_month"]),
    }


def get_monthly_anomaly_data(store_id: str, dept_id: str, year_month: str) -> dict:
    store_id, dept_id, year_month = int(store_id), int(dept_id), year_month[:7]
    row = monthly_yoy[
        (monthly_yoy["Store"] == store_id)
        & (monthly_yoy["Dept"] == dept_id)
        & (monthly_yoy["year_month"].astype(str) == year_month)
    ]
    if row.empty:
        return {"error": f"no se encontró store={store_id}, dept={dept_id}, mes={year_month}"}
    row = row.iloc[0]
    return {
        "actual_monthly_sales": round(float(row["monthly_sales"]), 2),
        "expected_monthly_sales_same_month_last_years_avg": round(float(row["expected_monthly_sales"]), 2),
        "deviation_pct": round(float(row["deviation_pct"]), 2),
        "years_of_same_month_history_used": int(row["years_of_same_month_history"]),
        "avg_temperature": round(float(row["avg_temperature"]), 1),
        "avg_fuel_price": round(float(row["avg_fuel_price"]), 3),
        "any_week_was_holiday": bool(row["any_holiday"]),
        "store_type": row["store_type"],
    }
