import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from anthropic import Anthropic

client = Anthropic()

DATA_PATH = "Walmart_data/merged_dataset.csv"
df = pd.read_csv(DATA_PATH)

# --- 1. Fechas y agregación mensual base (se reutiliza para TODAS las herramientas) ---
df["Date"] = pd.to_datetime(df["Date"])
df["year_month"] = df["Date"].dt.to_period("M")

monthly_all = df.groupby(["Store", "Dept", "year_month"]).agg(
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

# Filtro de calidad base: meses completos, sin ventas netas negativas
monthly_all = monthly_all[(monthly_all["weeks_count"] >= 4) & (monthly_all["monthly_sales"] > 0)].copy()
monthly_all["calendar_month"] = monthly_all["year_month"].dt.month
monthly_all["year"] = monthly_all["year_month"].dt.year
monthly_all["month_index"] = monthly_all["year_month"].apply(lambda p: p.year * 12 + p.month)  # para ordenar/regresión

# Ventas totales de la TIENDA por mes (para materialidad) — suma de todos los deptos
store_monthly_totals = monthly_all.groupby(["Store", "year_month"])["monthly_sales"].sum().rename("store_total_sales")

# --- 2. Base específica para detección de anomalías (mes vs mismo mes años anteriores) ---
monthly_yoy = monthly_all.copy()
group = monthly_yoy.groupby(["Store", "Dept", "calendar_month"])["monthly_sales"]
group_sum = group.transform("sum")
group_count = group.transform("count")
monthly_yoy["years_of_same_month_history"] = group_count - 1
monthly_yoy["expected_monthly_sales"] = (group_sum - monthly_yoy["monthly_sales"]) / monthly_yoy["years_of_same_month_history"]
monthly_yoy = monthly_yoy[monthly_yoy["years_of_same_month_history"] >= 2].copy()
monthly_yoy["deviation_pct"] = (
    (monthly_yoy["monthly_sales"] - monthly_yoy["expected_monthly_sales"]) / monthly_yoy["expected_monthly_sales"] * 100
)


def _get_group(store_id, dept_id):
    return monthly_all[(monthly_all["Store"] == store_id) & (monthly_all["Dept"] == dept_id)].sort_values("year_month")


def find_biggest_monthly_anomaly() -> dict:
    worst = monthly_yoy.loc[monthly_yoy["deviation_pct"].idxmin()]
    return {"store_id": int(worst["Store"]), "dept_id": int(worst["Dept"]), "year_month": str(worst["year_month"])}


# --- HERRAMIENTA 1: anomalía mensual (ya existía) ---
def get_monthly_anomaly_data(store_id: str, dept_id: str, year_month: str) -> dict:
    store_id, dept_id, year_month = int(store_id), int(dept_id), year_month[:7]
    row = monthly_yoy[(monthly_yoy["Store"] == store_id) & (monthly_yoy["Dept"] == dept_id) & (monthly_yoy["year_month"].astype(str) == year_month)]
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


# --- HERRAMIENTA 2: serie histórica completa (ya existía) ---
def get_full_sales_history(store_id: str, dept_id: str) -> dict:
    store_id, dept_id = int(store_id), int(dept_id)
    rows = _get_group(store_id, dept_id)
    if rows.empty:
        return {"error": f"no se encontró historial para store={store_id}, dept={dept_id}"}
    series = [{"year_month": str(r["year_month"]), "monthly_sales": round(float(r["monthly_sales"]), 2)} for _, r in rows.iterrows()]
    return {"store_id": store_id, "dept_id": dept_id, "monthly_series": series}


# --- HERRAMIENTA 3 (NUEVA): Rolling Year — ventana móvil de 12 meses ---
def calculate_rolling_year(store_id: str, dept_id: str, as_of_month: str) -> dict:
    store_id, dept_id, as_of_month = int(store_id), int(dept_id), as_of_month[:7]
    rows = _get_group(store_id, dept_id)
    as_of_period = pd.Period(as_of_month, freq="M")

    current_window = rows[(rows["year_month"] > as_of_period - 12) & (rows["year_month"] <= as_of_period)]
    previous_window = rows[(rows["year_month"] > as_of_period - 24) & (rows["year_month"] <= as_of_period - 12)]

    if len(current_window) < 10 or len(previous_window) < 10:
        return {"error": "no hay suficientes meses de historia para calcular 2 Rolling Years completos (se requieren ~10+ meses en cada ventana)"}

    current_ry = current_window["monthly_sales"].sum()
    previous_ry = previous_window["monthly_sales"].sum()
    deviation = (current_ry - previous_ry) / previous_ry * 100

    return {
        "as_of_month": as_of_month,
        "rolling_year_current_12m": round(float(current_ry), 2),
        "rolling_year_previous_12m": round(float(previous_ry), 2),
        "rolling_year_deviation_pct": round(float(deviation), 2),
        "months_in_current_window": len(current_window),
        "months_in_previous_window": len(previous_window),
    }


# --- HERRAMIENTA 4 (NUEVA): YTD — acumulado del año hasta el mes indicado ---
def calculate_ytd(store_id: str, dept_id: str, as_of_month: str) -> dict:
    store_id, dept_id, as_of_month = int(store_id), int(dept_id), as_of_month[:7]
    as_of_period = pd.Period(as_of_month, freq="M")
    year, month = as_of_period.year, as_of_period.month

    rows = _get_group(store_id, dept_id)
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


# --- HERRAMIENTA 5 (NUEVA): Materialidad — % que representa el depto del total de la tienda ---
def get_department_materiality(store_id: str, dept_id: str, year_month: str) -> dict:
    store_id, dept_id, year_month = int(store_id), int(dept_id), year_month[:7]
    period = pd.Period(year_month, freq="M")

    dept_row = monthly_all[(monthly_all["Store"] == store_id) & (monthly_all["Dept"] == dept_id) & (monthly_all["year_month"] == period)]
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


# --- HERRAMIENTA 6 (NUEVA): Tendencia general (regresión lineal simple) ---
def get_sales_trend(store_id: str, dept_id: str) -> dict:
    store_id, dept_id = int(store_id), int(dept_id)
    rows = _get_group(store_id, dept_id)
    if len(rows) < 6:
        return {"error": "no hay suficiente historial para calcular una tendencia confiable (mínimo 6 meses)"}

    x = rows["month_index"].values.astype(float)
    y = rows["monthly_sales"].values.astype(float)
    x_norm = x - x.min()  # para que el slope sea interpretable en "por mes"

    slope, intercept = np.polyfit(x_norm, y, 1)
    avg_sales = y.mean()
    monthly_growth_pct = (slope / avg_sales) * 100 if avg_sales != 0 else 0

    direction = "creciendo" if monthly_growth_pct > 0.5 else "decreciendo" if monthly_growth_pct < -0.5 else "estable"

    return {
        "trend_direction": direction,
        "avg_monthly_change_pct": round(float(monthly_growth_pct), 2),
        "avg_monthly_sales_overall": round(float(avg_sales), 2),
        "months_analyzed": len(rows),
    }


tools = [
    {"name": "get_monthly_anomaly_data", "description": "Datos de un mes específico: ventas reales vs. esperado (mismo mes, años anteriores), desviación %, contexto.",
     "input_schema": {"type": "object", "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}, "year_month": {"type": "string"}}, "required": ["store_id", "dept_id", "year_month"]}},
    {"name": "get_full_sales_history", "description": "Serie histórica COMPLETA de ventas mensuales de una tienda+depto, para ver el patrón general.",
     "input_schema": {"type": "object", "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}}, "required": ["store_id", "dept_id"]}},
    {"name": "calculate_rolling_year", "description": "Rolling Year (RY): suma de los últimos 12 meses hasta el mes indicado, comparado contra los 12 meses previos a esos. Útil para suavizar estacionalidad y ver tendencia real de negocio.",
     "input_schema": {"type": "object", "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}, "as_of_month": {"type": "string", "description": "YYYY-MM"}}, "required": ["store_id", "dept_id", "as_of_month"]}},
    {"name": "calculate_ytd", "description": "YTD (Year to Date): acumulado del año hasta el mes indicado, comparado contra el mismo periodo del año anterior.",
     "input_schema": {"type": "object", "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}, "as_of_month": {"type": "string", "description": "YYYY-MM"}}, "required": ["store_id", "dept_id", "as_of_month"]}},
    {"name": "get_department_materiality", "description": "Qué % de las ventas TOTALES de la tienda representa este departamento en un mes dado. Sirve para saber si una anomalía es relevante para el negocio o insignificante.",
     "input_schema": {"type": "object", "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}, "year_month": {"type": "string"}}, "required": ["store_id", "dept_id", "year_month"]}},
    {"name": "get_sales_trend", "description": "Tendencia general de largo plazo (creciendo/decreciendo/estable) de una tienda+depto, usando regresión lineal sobre todo su historial.",
     "input_schema": {"type": "object", "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}}, "required": ["store_id", "dept_id"]}},
]

DISPATCH = {
    "get_monthly_anomaly_data": get_monthly_anomaly_data,
    "get_full_sales_history": get_full_sales_history,
    "calculate_rolling_year": calculate_rolling_year,
    "calculate_ytd": calculate_ytd,
    "get_department_materiality": get_department_materiality,
    "get_sales_trend": get_sales_trend,
}


def run_agent(user_question: str):
    messages = [{"role": "user", "content": user_question}]
    total_input_tokens = 0
    total_output_tokens = 0
    while True:
        response = client.messages.create(model="claude-sonnet-4-6", max_tokens=2000, tools=tools, messages=messages)
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            print("\n🤖 RESPUESTA FINAL:\n", "".join(b.text for b in response.content if b.type == "text"))
            break
        cost_estimate = (total_input_tokens / 1_000_000 * 3) + (total_output_tokens / 1_000_000 * 15)
        print(f"\n💰 Tokens usados: {total_input_tokens} entrada / {total_output_tokens} salida — ~${cost_estimate:.4f} USD")

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\n🔧 Claude pidió usar: {block.name}({block.input})")
                func = DISPATCH.get(block.name)
                result = func(**block.input) if func else {"error": "herramienta desconocida"}
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})

        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    anomaly = find_biggest_monthly_anomaly()
    print(f"📍 Anomalía detectada: {anomaly}\n")

    run_agent(
        f"Investiga la tienda {anomaly['store_id']}, departamento {anomaly['dept_id']}, "
        f"mes {anomaly['year_month']}. Tienes varias herramientas disponibles (anomalía mensual, "
        f"historial completo, Rolling Year, YTD, materialidad del departamento, y tendencia general). "
        f"Úsalas de forma cruzada para VALIDAR si esto es una anomalía real e importante para el "
        f"negocio, o si es ruido/estacionalidad normal de bajo impacto. Da una conclusión final clara "
        f"indicando qué tan seria es esta situación y por qué."
    )
