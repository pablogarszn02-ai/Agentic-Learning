"""Punto de entrada del Agente Forecaster (independiente del Analista, por ahora --
en el futuro, LangGraph decidirá cuál invocar según la pregunta del usuario).

Uso:
    python forecast_agent.py                  -> pide tienda/depto por consola
    python forecast_agent.py 10 5              -> tienda 10, depto 5, directo
    python forecast_agent.py 10 5 8            -> tienda 10, depto 5, horizonte de 8 semanas
"""
import sys
from src.agent_forecaster import run_forecaster_agent

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        store_id, dept_id = sys.argv[1], sys.argv[2]
        horizon = sys.argv[3] if len(sys.argv) >= 4 else "13"
    else:
        store_id = input("¿Tienda? (store_id): ").strip()
        dept_id = input("¿Departamento? (dept_id): ").strip()
        horizon = input("¿Horizonte en semanas? (Enter para default = 13): ").strip() or "13"

    run_forecaster_agent(
        f"Necesito planear inventario para la tienda {store_id}, departamento {dept_id}, "
        f"para las próximas {horizon} semanas. Dame la proyección y qué tan confiable es "
        f"cada parte del periodo."
    )