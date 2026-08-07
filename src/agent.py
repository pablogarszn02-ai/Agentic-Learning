"""Agente Analista: investiga anomalías de ventas PASADAS usando herramientas de diagnóstico."""
from src.agent_core import run_agent_loop

from src.tools.anomaly import get_monthly_anomaly_data
from src.tools.history import get_full_sales_history
from src.tools.time_metrics import calculate_rolling_year, calculate_ytd
from src.tools.business import get_department_materiality, get_sales_trend
from src.tools.forecast import get_ml_forecast

SYSTEM_PROMPT = (
    "Eres un analista de datos senior de retail. Investigas anomalías de ventas usando "
    "las herramientas disponibles y entregas conclusiones claras y accionables. "
    "Sé directo: prioriza el insight sobre el relleno. No repitas en texto los números "
    "que ya mostraste en una tabla. Usa máximo 1-2 tablas cortas en total, no una por "
    "herramienta. Evita frases de relleno ('aquí tienes un análisis completo y detallado'). "
    "La conclusión final no debe superar las 200 palabras."
)

tools = [
    {"name": "get_monthly_anomaly_data", "description": "Datos de un mes específico: ventas reales vs. esperado (mismo mes, años anteriores), desviación %, contexto.",
     "input_schema": {"type": "object", "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}, "year_month": {"type": "string"}}, "required": ["store_id", "dept_id", "year_month"]}},
    {"name": "get_full_sales_history", "description": "Serie histórica COMPLETA de ventas mensuales de una tienda+depto.",
     "input_schema": {"type": "object", "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}}, "required": ["store_id", "dept_id"]}},
    {"name": "calculate_rolling_year", "description": "Rolling Year (RY): últimos 12 meses vs. los 12 previos.",
     "input_schema": {"type": "object", "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}, "as_of_month": {"type": "string"}}, "required": ["store_id", "dept_id", "as_of_month"]}},
    {"name": "calculate_ytd", "description": "YTD: acumulado del año vs. mismo periodo del año anterior.",
     "input_schema": {"type": "object", "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}, "as_of_month": {"type": "string"}}, "required": ["store_id", "dept_id", "as_of_month"]}},
    {"name": "get_department_materiality", "description": "% del total de la tienda que representa el departamento.",
     "input_schema": {"type": "object", "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}, "year_month": {"type": "string"}}, "required": ["store_id", "dept_id", "year_month"]}},
    {"name": "get_sales_trend", "description": "Tendencia de largo plazo (creciendo/decreciendo/estable).",
     "input_schema": {"type": "object", "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}}, "required": ["store_id", "dept_id"]}},
    {"name": "get_ml_forecast", "description": "Predicción del modelo ML para un MES YA OCURRIDO (segunda señal de validación, no forecasting futuro).",
     "input_schema": {"type": "object", "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}, "year_month": {"type": "string"}}, "required": ["store_id", "dept_id", "year_month"]}},
]

DISPATCH = {
    "get_monthly_anomaly_data": get_monthly_anomaly_data,
    "get_full_sales_history": get_full_sales_history,
    "calculate_rolling_year": calculate_rolling_year,
    "calculate_ytd": calculate_ytd,
    "get_department_materiality": get_department_materiality,
    "get_sales_trend": get_sales_trend,
    "get_ml_forecast": get_ml_forecast,
}


def run_agent(user_question: str):
    run_agent_loop(tools, DISPATCH, SYSTEM_PROMPT, user_question, max_tokens=1000)