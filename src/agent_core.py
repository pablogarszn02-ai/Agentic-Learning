"""Loop agéntico genérico (tool-use) — compartido por todos los agentes del proyecto."""
import json
from dotenv import load_dotenv
load_dotenv()
from anthropic import Anthropic

client = Anthropic()


def run_agent_loop(tools: list, dispatch: dict, system_prompt: str, user_question: str, max_tokens: int = 1000):
    messages = [{"role": "user", "content": user_question}]
    total_input_tokens = 0
    total_output_tokens = 0

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            print("\n RESPUESTA FINAL:\n", final_text)
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\n🔧 Pidió usar: {block.name}({block.input})")
                func = dispatch.get(block.name)
                result = func(**block.input) if func else {"error": "herramienta desconocida"}
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})

        messages.append({"role": "user", "content": tool_results})

    cost = (total_input_tokens / 1_000_000 * 3) + (total_output_tokens / 1_000_000 * 15)
    print(f"\n Tokens: {total_input_tokens} entrada / {total_output_tokens} salida — ~${cost:.4f} USD")