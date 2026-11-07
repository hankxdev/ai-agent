import json

from pydantic import ValidationError

from tools import TOOL_REGISTRY, ai_model_call

MAX_STEPS = 3


def execute_tool_call(name: str, raw_args: str) -> str:
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
        except (ValueError, TypeError) as e:
            return json.dumps({"error": f"value error:{e}"}, ensure_ascii=False)

    print("tool call result:", result)
    return json.dumps(result, ensure_ascii=False)


def run_agent(messages: list) -> None:
    for step in range(MAX_STEPS + 1):
        stream = ai_model_call(True, messages)
        text_parts: list[str] = []
        tool_call_buffer: dict[int, dict] = {}
        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                text_parts.append(delta.content)
                print(delta.content, end="", flush=True)

            for call in delta.tool_calls or []:
                slot = tool_call_buffer.setdefault(
                    call.index,
                    {
                        "id": None,
                        "name": "",
                        "arguments": "",
                    }
                )
                print(f"分组的call：{call}")

                if call.id:
                    slot["id"] = call.id

                if call.function:
                    if call.function.name:
                        slot["name"] = call.function.name
                    if call.function.arguments:
                        slot["arguments"] += call.function.arguments
        print()
        content = "".join(text_parts) or None

        if not tool_call_buffer:
            messages.append({
                "role": "assistant",
                "content": content
            })
            return
        tool_call_payload = [
            {
                "id": slot["id"],
                "type": "function",
                "function": {
                    "name": slot["name"],
                    "arguments": slot["arguments"],
                },
            }
            for slot in tool_call_buffer.values()
        ]

        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_call_payload
        })

        for slot in tool_call_buffer.values():
            print(f"  [第 {step} 步] 调用 {slot['name']}({slot['arguments']})")

            result = execute_tool_call(slot["name"], slot["arguments"])

            print(f"  [第 {step} 步] 返回 {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": slot["id"],
                "content": result,
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
        user_input = input("You: ").strip()
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
