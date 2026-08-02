"""Punto de entrada: encuentra la anomalía más grande y le pide al agente que la investigue."""
from src.tools.anomaly import find_biggest_monthly_anomaly
from src.agent import run_agent

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
