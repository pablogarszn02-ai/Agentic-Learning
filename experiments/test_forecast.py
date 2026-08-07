"""Prueba rápida y aislada del forecasting recursivo — sin tocar la API de Claude."""
import json
from src.ml.forecast_future import forecast_future

# Prueba con una combinación tienda+depto con buen historial (ajusta si quieres probar otra)
result = forecast_future(store_id=1, dept_id=1, horizon_weeks=13)

print(json.dumps(result, indent=2, ensure_ascii=False))