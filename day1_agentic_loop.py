import json
from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

client = Anthropic()

tools = [
    {
        "name": "get_sales_anomaly_data",
        "description": (
            "Devuelve datos detallados de una anomalía de ventas detectada "
            "en una tienda: ventas reales vs. esperadas, y contexto "
            "(clima, promociones activas, quiebres de stock)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {
                    "type": "string",
                    "description": "ID de la tienda a investigar, ej. 'store_042'"
                }
            },
            "required": ["store_id"]
        }
    }
]


def get_sales_anomaly_data(store_id: str) -> dict:
    fake_data = {
        "store_042": {
            "expected_sales": 15200,
            "actual_sales": 9100,
            "deviation_pct": -40.1,
            "weather": "tormenta severa, alerta vial 3 días",
            "active_promotions": [],
            "stock_out_events": ["categoría: lácteos, 2 días sin stock"]
        }
    }
    return fake_data.get(store_id, {"error": "tienda no encontrada"})


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
    run_agent(
        "¿Por qué cayeron tanto las ventas en la tienda store_042 "
        "la semana pasada? Investígalo y dame una explicación clara."
    )