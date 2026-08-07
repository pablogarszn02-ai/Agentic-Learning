"""Agente Forecaster: proyecta ventas FUTURAS para soporte de decisiones de inventario.
No investiga causas del pasado (eso es trabajo del Agente Analista) -- su foco es
exclusivamente hacia adelante, comunicando incertidumbre de forma honesta.
"""
from src.agent_core import run_agent_loop
from src.tools.future_forecast import get_future_forecast
from src.tools.history import get_full_sales_history

SYSTEM_PROMPT = (
    "Eres un especialista en planeación de demanda e inventario de retail. Tu trabajo "
    "es proyectar ventas futuras y traducir esa proyección en implicaciones claras para "
    "decisiones de inventario/reabastecimiento. NO investigas causas del pasado -- ese es "
    "otro agente. Sé directo, menciona explícitamente qué tan confiable es cada tramo del "
    "horizonte (la incertidumbre crece con las semanas más lejanas), y da una recomendación "
    "concreta de planeación cuando tengas suficiente información."
)

tools = [
    {
        "name": "get_future_forecast",
        "description": "Predice ventas futuras (forecasting recursivo) para las próximas semanas de una tienda+depto.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string"},
                "dept_id": {"type": "string"},
                "horizon_weeks": {"type": "string", "description": "Semanas a predecir, default 13 (1 trimestre). Opcional."},
            },
            "required": ["store_id", "dept_id"],
        },
    },
    {
        "name": "get_full_sales_history",
        "description": "Serie histórica completa de ventas mensuales -- útil como contexto antes de proyectar (ej. para reconocer estacionalidad).",
        "input_schema": {
            "type": "object",
            "properties": {"store_id": {"type": "string"}, "dept_id": {"type": "string"}},
            "required": ["store_id", "dept_id"],
        },
    },
]

DISPATCH = {
    "get_future_forecast": get_future_forecast,
    "get_full_sales_history": get_full_sales_history,
}


def run_forecaster_agent(user_question: str):
    run_agent_loop(tools, DISPATCH, SYSTEM_PROMPT, user_question, max_tokens=2500)