"""
Carga los 3 CSV de Walmart, los une, y prepara las tablas mensuales
que usan todas las herramientas del agente. Se ejecuta UNA sola vez
cuando este módulo se importa (Python cachea el resultado).
"""
import pandas as pd

DATA_PATH = "Walmart_data"


def _load_raw() -> pd.DataFrame:
    train = pd.read_csv(f"{DATA_PATH}/train.csv")
    features = pd.read_csv(f"{DATA_PATH}/features.csv")
    stores = pd.read_csv(f"{DATA_PATH}/stores.csv")

    merged = train.merge(features, on=["Store", "Date"], how="left", suffixes=("", "_feat"))
    merged = merged.merge(stores, on="Store", how="left")
    merged = merged.drop(columns=["IsHoliday_feat"])
    return merged


def _build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["year_month"] = df["Date"].dt.to_period("M")

    monthly = df.groupby(["Store", "Dept", "year_month"]).agg(
        monthly_sales=("Weekly_Sales", "sum"),
        weeks_count=("Weekly_Sales", "count"),
        avg_temperature=("Temperature", "mean"),
        avg_fuel_price=("Fuel_Price", "mean"),
        avg_cpi=("CPI", "mean"),
        avg_unemployment=("Unemployment", "mean"),
        any_holiday=("IsHoliday", "max"),
        store_type=("Type", "first"),
        store_size=("Size", "first"),
    ).reset_index()

    # Filtro de calidad: meses completos, sin ventas netas negativas (errores/devoluciones)
    monthly = monthly[(monthly["weeks_count"] >= 4) & (monthly["monthly_sales"] > 0)].copy()
    monthly["calendar_month"] = monthly["year_month"].dt.month
    monthly["year"] = monthly["year_month"].dt.year
    monthly["month_index"] = monthly["year_month"].apply(lambda p: p.year * 12 + p.month)
    return monthly


def _build_yoy(monthly: pd.DataFrame) -> pd.DataFrame:
    """Calcula el promedio esperado comparando el MISMO mes calendario en años anteriores
    (respeta estacionalidad, en vez de comparar contra el promedio anual completo)."""
    monthly_yoy = monthly.copy()
    group = monthly_yoy.groupby(["Store", "Dept", "calendar_month"])["monthly_sales"]
    group_sum = group.transform("sum")
    group_count = group.transform("count")

    monthly_yoy["years_of_same_month_history"] = group_count - 1
    monthly_yoy["expected_monthly_sales"] = (
        (group_sum - monthly_yoy["monthly_sales"]) / monthly_yoy["years_of_same_month_history"]
    )
    monthly_yoy = monthly_yoy[monthly_yoy["years_of_same_month_history"] >= 2].copy()
    monthly_yoy["deviation_pct"] = (
        (monthly_yoy["monthly_sales"] - monthly_yoy["expected_monthly_sales"])
        / monthly_yoy["expected_monthly_sales"] * 100
    )
    return monthly_yoy


# --- Se ejecuta una sola vez, al importar este módulo ---
_raw = _load_raw()
monthly_all = _build_monthly(_raw)
monthly_yoy = _build_yoy(monthly_all)
store_monthly_totals = monthly_all.groupby(["Store", "year_month"])["monthly_sales"].sum().rename("store_total_sales")


def get_group(store_id: int, dept_id: int) -> pd.DataFrame:
    """Serie histórica ordenada de una tienda+departamento específicos."""
    return monthly_all[(monthly_all["Store"] == store_id) & (monthly_all["Dept"] == dept_id)].sort_values("year_month")
