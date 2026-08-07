"""Agente de Recomendación: convierte diagnóstico + proyección en ACCIONES concretas
y cuantificadas de negocio (no describe la situación, la resuelve en decisiones).
"""
from src.agent_core import run_agent_loop

from src.tools.anomaly import get_monthly_anomaly_data
from src.tools.business import get_department_materiality, get_sales_trend
from src.tools.future_forecast import get_future_forecast
from src.tools.history import get_full_sales_history

SYSTEM_PROMPT = (
    "Eres un consultor senior de decisiones de negocio en retail. NO es tu trabajo "
    "solo diagnosticar (qué pasó) ni solo proyectar (qué pasará) -- tu trabajo es "
    "convertir esa información en RECOMENDACIONES CONCRETAS Y CUANTIFICADAS: qué hacer, "
    "cuánto (números/porcentajes), y cuándo. Prohibido dar recomendaciones vagas tipo "
    "'monitorear de cerca' o 'evaluar la situación' sin una acción específica asociada. "
    "Si la situación no amerita acción (baja materialidad, dentro de rango normal), dilo "
    "explícitamente y justifica por qué NO actuar también es una recomendación válida. "
    "Sé directo, prioriza la decisión sobre la narrativa. Máximo 250 palabras en la "
    "recomendación final, sin contar tablas."
)

tools = [
    {"name": "get_monthly_anomaly_data", "description": "Diagnóstico de un mes: desviación vs. esperado histórico, contexto.",
     "input_schema": {"type": "object", "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}, "year_month": {"type": "string"}}, "required": ["store_id", "dept_id", "year_month"]}},
    {"name": "get_department_materiality", "description": "% del total de la tienda que representa el departamento -- clave para decidir si vale la pena actuar.",
     "input_schema": {"type": "object", "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}, "year_month": {"type": "string"}}, "required": ["store_id", "dept_id", "year_month"]}},
    {"name": "get_sales_trend", "description": "Tendencia de largo plazo (creciendo/decreciendo/estable).",
     "input_schema": {"type": "object", "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}}, "required": ["store_id", "dept_id"]}},
    {"name": "get_future_forecast", "description": "Proyección de ventas futuras (forecasting recursivo) -- base para dimensionar recomendaciones de inventario.",
     "input_schema": {"type": "object", "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}, "horizon_weeks": {"type": "string", "description": "Default 13. Opcional."}}, "required": ["store_id", "dept_id"]}},
    {"name": "get_full_sales_history", "description": "Serie histórica completa -- contexto de patrón/estacionalidad.",
     "input_schema": {"type": "object", "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}}, "required": ["store_id", "dept_id"]}},
]

DISPATCH = {
    "get_monthly_anomaly_data": get_monthly_anomaly_data,
    "get_department_materiality": get_department_materiality,
    "get_sales_trend": get_sales_trend,
    "get_future_forecast": get_future_forecast,
    "get_full_sales_history": get_full_sales_history,
}


def run_recommender_agent(user_question: str):
    run_agent_loop(tools, DISPATCH, SYSTEM_PROMPT, user_question, max_tokens=1800)