"""Define las herramientas expuestas a Claude y el loop agéntico principal."""
import json
from dotenv import load_dotenv
load_dotenv()
from anthropic import Anthropic

from src.tools.anomaly import get_monthly_anomaly_data
from src.tools.history import get_full_sales_history
from src.tools.time_metrics import calculate_rolling_year, calculate_ytd
from src.tools.business import get_department_materiality, get_sales_trend

client = Anthropic()

tools = [
    {
        "name": "get_monthly_anomaly_data",
        "description": "Datos de un mes específico: ventas reales vs. esperado (mismo mes, años anteriores), desviación %, contexto.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string"},
                "dept_id": {"type": "string"},
                "year_month": {"type": "string", "description": "YYYY-MM"},
            },
            "required": ["store_id", "dept_id", "year_month"],
        },
    },
    {
        "name": "get_full_sales_history",
        "description": "Serie histórica COMPLETA de ventas mensuales de una tienda+depto, para ver el patrón general.",
        "input_schema": {
            "type": "object",
            "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}},
            "required": ["store_id", "dept_id"],
        },
    },
    {
        "name": "calculate_rolling_year",
        "description": "Rolling Year (RY): suma de los últimos 12 meses hasta el mes indicado, comparado contra los 12 meses previos. Suaviza estacionalidad.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string"},
                "dept_id": {"type": "string"},
                "as_of_month": {"type": "string", "description": "YYYY-MM"},
            },
            "required": ["store_id", "dept_id", "as_of_month"],
        },
    },
    {
        "name": "calculate_ytd",
        "description": "YTD (Year to Date): acumulado del año hasta el mes indicado, vs. el mismo periodo del año anterior.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string"},
                "dept_id": {"type": "string"},
                "as_of_month": {"type": "string", "description": "YYYY-MM"},
            },
            "required": ["store_id", "dept_id", "as_of_month"],
        },
    },
    {
        "name": "get_department_materiality",
        "description": "Qué % de las ventas TOTALES de la tienda representa este departamento en un mes dado. Sirve para saber si una anomalía es relevante para el negocio.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string"},
                "dept_id": {"type": "string"},
                "year_month": {"type": "string", "description": "YYYY-MM"},
            },
            "required": ["store_id", "dept_id", "year_month"],
        },
    },
    {
        "name": "get_sales_trend",
        "description": "Tendencia general de largo plazo (creciendo/decreciendo/estable), usando regresión lineal sobre todo el historial.",
        "input_schema": {
            "type": "object",
            "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}},
            "required": ["store_id", "dept_id"],
        },
    },
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
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=2000, tools=tools, messages=messages
        )
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            print("\n🤖 RESPUESTA FINAL:\n", final_text)
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\n🔧 Claude pidió usar: {block.name}({block.input})")
                func = DISPATCH.get(block.name)
                result = func(**block.input) if func else {"error": "herramienta desconocida"}
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})

        messages.append({"role": "user", "content": tool_results})

    cost_estimate = (total_input_tokens / 1_000_000 * 3) + (total_output_tokens / 1_000_000 * 15)
    print(f"\n💰 Tokens usados: {total_input_tokens} entrada / {total_output_tokens} salida — ~${cost_estimate:.4f} USD")
