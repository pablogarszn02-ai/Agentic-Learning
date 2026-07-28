import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from anthropic import Anthropic

client = Anthropic()

DATA_PATH = "Walmart_data/merged_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Solo consideramos ventas positivas para calcular promedios confiables
df_positive = df[df["Weekly_Sales"] > 0].copy()

# Contamos cuántas semanas de historial real tiene cada tienda+depto
historial = df_positive.groupby(["Store", "Dept"])["Weekly_Sales"].transform("count")
promedio = df_positive.groupby(["Store", "Dept"])["Weekly_Sales"].transform("mean")

df_positive["expected_sales"] = promedio
df_positive["historial_semanas"] = historial

# Exigimos mínimo 20 semanas de historial para considerar el promedio confiable
df = df_positive[df_positive["historial_semanas"] >= 20].copy()

df["deviation_pct"] = (df["Weekly_Sales"] - df["expected_sales"]) / df["expected_sales"] * 100


def find_biggest_anomaly() -> dict:
    """Encuentra la fila con la caída porcentual más grande de todo el dataset."""
    worst = df.loc[df["deviation_pct"].idxmin()]
    return {
        "store_id": int(worst["Store"]),
        "dept_id": int(worst["Dept"]),
        "date": worst["Date"],
    }


def get_sales_anomaly_data(store_id: str, dept_id: str, date: str) -> dict:
    """Herramienta real: consulta el dataset unido para una tienda+depto+fecha específicos."""
    store_id = int(store_id)
    dept_id = int(dept_id)
    row = df[(df["Store"] == store_id) & (df["Dept"] == dept_id) & (df["Date"] == date)]
    if row.empty:
        return {"error": "no se encontró esa combinación de tienda/depto/fecha"}
    row = row.iloc[0]
    return {
        "actual_sales": round(float(row["Weekly_Sales"]), 2),
        "expected_sales_historical_avg": round(float(row["expected_sales"]), 2),
        "deviation_pct": round(float(row["deviation_pct"]), 2),
        "temperature": row["Temperature"],
        "fuel_price": row["Fuel_Price"],
        "cpi": row["CPI"],
        "unemployment": row["Unemployment"],
        "is_holiday": bool(row["IsHoliday"]),
        "store_type": row["Type"],
        "store_size": int(row["Size"]),
    }


tools = [
    {
        "name": "get_sales_anomaly_data",
        "description": (
            "Devuelve datos reales de ventas para una tienda, departamento y fecha "
            "específicos: ventas reales, promedio histórico esperado, desviación %, "
            "y contexto (clima, combustible, CPI, desempleo, festivo, tipo/tamaño de tienda)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string", "description": "Número de tienda, ej. '13'"},
                "dept_id": {"type": "string", "description": "Número de departamento, ej. '38'"},
                "date": {"type": "string", "description": "Fecha exacta, formato YYYY-MM-DD"},
            },
            "required": ["store_id", "dept_id", "date"]
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
                if block.name == "get_sales_anomaly_data":
                    result = get_sales_anomaly_data(**block.input)
                else:
                    result = {"error": "herramienta desconocida"}
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    anomaly = find_biggest_anomaly()
    print(f"📍 Anomalía real más grande encontrada: {anomaly}\n")

    run_agent(
        f"Investiga por qué las ventas de la tienda {anomaly['store_id']}, "
        f"departamento {anomaly['dept_id']}, en la fecha {anomaly['date']}, "
        f"tuvieron una desviación tan grande respecto al promedio histórico. "
        f"Dame una explicación clara basada en los datos."
    )