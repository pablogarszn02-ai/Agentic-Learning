"""Punto de entrada del Agente de Recomendación (independiente, por ahora --
LangGraph después coordinará Analista + Forecaster + Recomendador según el caso).

Uso:
    python recommend_agent.py              -> pide tienda/depto/mes por consola
    python recommend_agent.py 11 18 2010-05 -> directo
"""
import sys
from src.agent_recommender import run_recommender_agent

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        store_id, dept_id, year_month = sys.argv[1], sys.argv[2], sys.argv[3]
    else:
        store_id = input("¿Tienda? (store_id): ").strip()
        dept_id = input("¿Departamento? (dept_id): ").strip()
        year_month = input("¿Mes a evaluar? (YYYY-MM): ").strip()

    run_recommender_agent(
        f"Evalúa la situación de la tienda {store_id}, departamento {dept_id}, "
        f"mes {year_month}. Investiga lo necesario (diagnóstico del mes, materialidad, "
        f"tendencia, y proyección futura si aplica) y dame una recomendación de negocio "
        f"concreta: ¿hay que actuar? ¿qué acción específica, con qué magnitud, y con qué "
        f"urgencia?"
    )