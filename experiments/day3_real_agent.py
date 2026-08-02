import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from anthropic import Anthropic

client = Anthropic()

DATA_PATH = "Walmart_data/merged_dataset.csv"
df = pd.read_csv(DATA_PATH)

# --- 1. Convertimos Date a datetime real y extraemos año-mes ---
df["Date"] = pd.to_datetime(df["Date"])
df["year_month"] = df["Date"].dt.to_period("M")

# --- 2. Agregamos a nivel mensual ---
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

# --- 3. Limpieza robusta: descartamos meses que son <1% de la mediana histórica ---
# (probablemente errores de registro, no anomalías de negocio reales)
group_median = monthly.groupby(["Store", "Dept"])["monthly_sales"].transform("median")
print(monthly[(monthly["Store"] == 31) & (monthly["Dept"] == 18)][["year_month", "monthly_sales"]])
print("Mediana calculada:", group_median[(monthly["Store"] == 31) & (monthly["Dept"] == 18)].iloc[0] if not monthly[(monthly["Store"] == 31) & (monthly["Dept"] == 18)].empty else "N/A")
monthly = monthly[
    (monthly["weeks_count"] >= 4)
    & (monthly["monthly_sales"] >= 0.01 * group_median)
].copy()

# --- 4. Exigimos mínimo 4 meses de historial por tienda+depto ---
monthly["historial_meses"] = monthly.groupby(["Store", "Dept"])["monthly_sales"].transform("count")
monthly = monthly[monthly["historial_meses"] >= 4].copy()

# --- 5. Calculamos promedio histórico y desviación ---
monthly["expected_monthly_sales"] = monthly.groupby(["Store", "Dept"])["monthly_sales"].transform("mean")
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
    # Normalizamos: si Claude manda "2010-05-01" o "2010-05", nos quedamos solo con "2010-05"
    year_month = year_month[:7]
    row = monthly[
        (monthly["Store"] == store_id)
        & (monthly["Dept"] == dept_id)
        & (monthly["year_month"].astype(str) == year_month)
    ]
    if row.empty:
        return {"error": f"no se encontró la combinación store={store_id}, dept={dept_id}, mes={year_month}"}
    row = row.iloc[0]
    return {
        "actual_monthly_sales": round(float(row["monthly_sales"]), 2),
        "expected_monthly_sales_historical_avg": round(float(row["expected_monthly_sales"]), 2),
        "deviation_pct": round(float(row["deviation_pct"]), 2),
        "months_of_history_available": int(row["historial_meses"]),
        "avg_temperature": round(float(row["avg_temperature"]), 1),
        "avg_fuel_price": round(float(row["avg_fuel_price"]), 3),
        "avg_cpi": round(float(row["avg_cpi"]), 2),
        "avg_unemployment": round(float(row["avg_unemployment"]), 2),
        "any_week_was_holiday": bool(row["any_holiday"]),
        "store_type": row["store_type"],
        "store_size": int(row["store_size"]),
    }


tools = [
    {
        "name": "get_monthly_anomaly_data",
        "description": (
            "Devuelve datos reales de ventas MENSUALES (agregadas) para una tienda, "
            "departamento y mes específicos: ventas del mes, promedio histórico mensual "
            "esperado, desviación %, y contexto (clima, combustible, CPI, desempleo, "
            "festivo, tipo/tamaño de tienda)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string", "description": "Número de tienda, ej. '11'"},
                "dept_id": {"type": "string", "description": "Número de departamento, ej. '18'"},
                "year_month": {"type": "string", "description": "Mes exacto, formato YYYY-MM, ej. '2010-05'"},
            },
            "required": ["store_id", "dept_id", "year_month"]
        }
    }
]


def run_agent(user_question: str):
    messages = [{"role": "user", "content": user_question}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
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
    print(f"📍 Anomalía mensual más grande encontrada: {anomaly}\n")

    run_agent(
        f"Investiga por qué las ventas de la tienda {anomaly['store_id']}, "
        f"departamento {anomaly['dept_id']}, en el mes {anomaly['year_month']}, "
        f"tuvieron una desviación tan grande respecto al promedio histórico mensual. "
        f"Dame una explicación clara basada en los datos."
    )