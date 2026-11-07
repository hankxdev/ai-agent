import json
from datetime import datetime

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from config import API_KEY, BASE_URL, MODEL
from tools import tools, TOOL_REGISTRY




def execute_tool_call(call) -> str:
    name = call.function.name
    raw_args = call.function.arguments
    print(f"AI is trying to call {name},  call args: {raw_args}")

    if name not in TOOL_REGISTRY:
        result = {"error": f"unknow tool call {name}"}
    else:
        func, args_model = TOOL_REGISTRY[name]
        try:
            if args_model is not None:
                args = args_model.model_validate_json(raw_args)
                result = func(**args.model_dump())
            else:
                result = func()
        except ValidationError as e:
            return json.dumps({"error": f"validation error：{e}"}, ensure_ascii=False)
        except ValueError as e:
            return json.dumps({"error": f"value error:{e}"}, ensure_ascii=False)

    print("tool call result:", result)
    return json.dumps(result, ensure_ascii=False)

MAX_STEPS = 3

def run_agent(messages:list) -> None:
    for step in range(0, MAX_STEPS+1):
        request = client.chat.completions.create(  # type:ignore
            model=MODEL,
            messages=messages,
            tools=tools,
            stream=False
        )

        message = request.choices[0].message
        messages.append(message.model_dump(exclude_unset=True))

        if not message.tool_calls:
            print("model answer without tool calls", message.content)
            return
        for call in message.tool_calls:
            print(f"the {step} time tool call, {call.function.name}({call.function.arguments})")
            result = execute_tool_call(call)
            print(f"the {step} time tool call returns: {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result
            })

    print("AI tool call reaches max steps, stopped")


def main() -> None:
    messages = [
        {
            "role": "system",
            "content": "you are a helpful tool, call tools when needed, then answer with tool result"
        }
    ]

    while True:
        user_input  = input("You: ").strip()
        if user_input in ["exit", "quit"]:
            break

        if not user_input:
            continue
        messages.append({
            "role": "user",
            "content": user_input
        })

        run_agent(messages)


if __name__ == "__main__":
    main()