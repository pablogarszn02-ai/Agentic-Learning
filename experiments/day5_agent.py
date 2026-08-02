import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from anthropic import Anthropic

client = Anthropic()

DATA_PATH = "Walmart_data/merged_dataset.csv"
df = pd.read_csv(DATA_PATH)

# --- 1. Fechas y agregación mensual ---
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

# --- 2. Filtro de calidad: meses completos, sin ventas netas negativas (errores/devoluciones) ---
monthly = monthly[(monthly["weeks_count"] >= 4) & (monthly["monthly_sales"] > 0)].copy()

# --- 3. Extraemos el mes calendario (1-12), para comparar "Mayo contra Mayo", no contra el año completo ---
monthly["calendar_month"] = monthly["year_month"].dt.month

# --- 4. Promedio esperado = promedio del MISMO mes calendario en OTROS años (leave-one-out) ---
group = monthly.groupby(["Store", "Dept", "calendar_month"])["monthly_sales"]
group_sum = group.transform("sum")
group_count = group.transform("count")

monthly["years_of_same_month_history"] = group_count - 1  # excluyendo el propio mes
monthly["expected_monthly_sales"] = (group_sum - monthly["monthly_sales"]) / monthly["years_of_same_month_history"]

# --- 5. Exigimos al menos 2 años de historial del MISMO mes calendario para comparar con confianza ---
monthly = monthly[monthly["years_of_same_month_history"] >= 2].copy()

monthly["deviation_pct"] = (
    (monthly["monthly_sales"] - monthly["expected_monthly_sales"])
    / monthly["expected_monthly_sales"] * 100
)


def find_biggest_monthly_anomaly() -> dict:
    worst = monthly.loc[monthly["deviation_pct"].idxmin()]
    return {
        "store_id": int(worst["Store"]),
        "dept_id": int(worst["Dept"]),
        "year_month": str(worst["year_month"]),
    }


def get_monthly_anomaly_data(store_id: str, dept_id: str, year_month: str) -> dict:
    store_id = int(store_id)
    dept_id = int(dept_id)
    year_month = year_month[:7]
    row = monthly[
        (monthly["Store"] == store_id)
        & (monthly["Dept"] == dept_id)
        & (monthly["year_month"].astype(str) == year_month)
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
        "avg_cpi": round(float(row["avg_cpi"]), 2),
        "avg_unemployment": round(float(row["avg_unemployment"]), 2),
        "any_week_was_holiday": bool(row["any_holiday"]),
        "store_type": row["store_type"],
        "store_size": int(row["store_size"]),
    }


def get_full_sales_history(store_id: str, dept_id: str) -> dict:
    """Segunda herramienta: devuelve TODA la serie histórica mensual de una tienda+depto de una sola vez."""
    store_id = int(store_id)
    dept_id = int(dept_id)
    rows = monthly[(monthly["Store"] == store_id) & (monthly["Dept"] == dept_id)].sort_values("year_month")
    if rows.empty:
        return {"error": f"no se encontró historial para store={store_id}, dept={dept_id}"}
    series = [
        {"year_month": str(r["year_month"]), "monthly_sales": round(float(r["monthly_sales"]), 2)}
        for _, r in rows.iterrows()
    ]
    return {"store_id": store_id, "dept_id": dept_id, "monthly_series": series}


tools = [
    {
        "name": "get_monthly_anomaly_data",
        "description": (
            "Devuelve datos de un mes específico de una tienda+departamento: ventas reales, "
            "el promedio esperado (calculado comparando el MISMO mes calendario en años "
            "anteriores, para respetar estacionalidad), desviación %, y contexto "
            "(clima, combustible, CPI, desempleo, festivo, tipo/tamaño de tienda). "
            "Úsala cuando ya sepas exactamente qué mes investigar."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string"},
                "dept_id": {"type": "string"},
                "year_month": {"type": "string", "description": "Formato YYYY-MM"},
            },
            "required": ["store_id", "dept_id", "year_month"]
        }
    },
    {
        "name": "get_full_sales_history",
        "description": (
            "Devuelve la serie histórica COMPLETA de ventas mensuales de una tienda+departamento "
            "en una sola llamada. Úsala cuando necesites ver la tendencia general o el patrón "
            "estacional completo, en vez de pedir mes por mes (más eficiente para analizar tendencias)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string"},
                "dept_id": {"type": "string"},
            },
            "required": ["store_id", "dept_id"]
        }
    }
]


def run_agent(user_question: str):
    messages = [{"role": "user", "content": user_question}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            print("\n🤖 RESPUESTA FINAL:\n", final_text)
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\n🔧 Claude pidió usar: {block.name}({block.input})")
                if block.name == "get_monthly_anomaly_data":
                    result = get_monthly_anomaly_data(**block.input)
                elif block.name == "get_full_sales_history":
                    result = get_full_sales_history(**block.input)
                else:
                    result = {"error": "herramienta desconocida"}
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    anomaly = find_biggest_monthly_anomaly()
    print(f"📍 Anomalía más grande (ya comparando mes-contra-mismo-mes-año-anterior): {anomaly}\n")

    run_agent(
        f"Investiga por qué las ventas de la tienda {anomaly['store_id']}, "
        f"departamento {anomaly['dept_id']}, en el mes {anomaly['year_month']}, "
        f"tuvieron una desviación tan grande respecto a lo esperado para ese mismo mes "
        f"en años anteriores. Usa las herramientas que consideres necesarias — puedes "
        f"revisar la serie histórica completa si te ayuda a entender el patrón. "
        f"Dame una conclusión clara."
    )
