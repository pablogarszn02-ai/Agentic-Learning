# Agentic Retail Intelligence

Agente de IA que investiga anomalías de ventas en datos reales de retail (dataset de Walmart Sales Forecasting, Kaggle), usando razonamiento con múltiples herramientas ("tool use") sobre la API de Claude — sin frameworks de por medio, para demostrar el mecanismo agéntico de raíz.

## ¿Qué hace?

1. Detecta automáticamente la anomalía de ventas mensual más significativa en el dataset (comparando cada mes contra el mismo mes calendario en años anteriores, para respetar estacionalidad).
2. Un agente de IA investiga esa anomalía usando 6 herramientas especializadas:
   - **Anomalía mensual** — desviación vs. el mismo mes en años previos
   - **Historial completo** — serie de tiempo completa para detectar patrones
   - **Rolling Year (RY)** — ventana móvil de 12 meses
   - **YTD (Year to Date)** — acumulado del año vs. año anterior
   - **Materialidad** — qué % representa el departamento del total de la tienda (evita alarmas por anomalías irrelevantes)
   - **Tendencia** — dirección de largo plazo (regresión lineal)
3. El agente decide por sí mismo qué herramientas usar y en qué orden, cruza los resultados, y entrega una conclusión de negocio sobre si la anomalía es real, seria, y accionable.

## Arquitectura

```
src/
├── data_loader.py       # Carga y prepara los datos (una sola vez)
├── tools/
│   ├── anomaly.py        # Detección de anomalías
│   ├── history.py        # Serie histórica
│   ├── time_metrics.py   # Rolling Year, YTD
│   └── business.py       # Materialidad, tendencia
└── agent.py               # Definición de tools + loop agéntico (Claude API)
main.py                    # Punto de entrada
```

## Cómo correrlo

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Crea un archivo `.env` con:
```
ANTHROPIC_API_KEY=tu_key_aqui
```

Descarga el dataset de [Walmart Sales Forecasting (Kaggle)](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data) y coloca `train.csv`, `features.csv`, `stores.csv` en una carpeta `Walmart_data/`.

Corre:
```bash
python main.py
```

## Próximos pasos del proyecto

- Componente de Machine Learning (forecasting) como agente especializado
- Orquestación multi-agente con LangGraph
- Empaquetado como servidor MCP para acceso directo desde Claude Desktop
