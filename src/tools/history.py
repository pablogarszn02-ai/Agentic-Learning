"""Herramienta: serie histórica mensual completa de una tienda+departamento."""
from src.data_loader import get_group


def get_full_sales_history(store_id: str, dept_id: str) -> dict:
    store_id, dept_id = int(store_id), int(dept_id)
    rows = get_group(store_id, dept_id)
    if rows.empty:
        return {"error": f"no se encontró historial para store={store_id}, dept={dept_id}"}
    series = [
        {"year_month": str(r["year_month"]), "monthly_sales": round(float(r["monthly_sales"]), 2)}
        for _, r in rows.iterrows()
    ]
    return {"store_id": store_id, "dept_id": dept_id, "monthly_series": series}
