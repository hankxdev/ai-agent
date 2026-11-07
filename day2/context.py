# noinspection DuplicatedCode
import json

from pydantic import ValidationError

from tools import TOOL_REGISTRY, ai_model_call

MAX_STEPS = 3
MAX_HISTORY_TURNS = 3


def trim_history(messages: list, keep_last_turns: int = MAX_HISTORY_TURNS) -> list:
    if len(messages) <= MAX_HISTORY_TURNS - 1:
        return messages

    turn_starts = [
        index for index, message in enumerate(messages)
        if message.get("role") == "user"
    ]

    if len(turn_starts) <= keep_last_turns:
        return messages

    cut = turn_starts[-keep_last_turns]

    return messages[:1] + messages[cut:]


def summarize_history(messages: list, keep_last_turns: int = MAX_HISTORY_TURNS) -> list:
    if len(messages) <= 2:
        return messages

    turn_starts = [
        index for index, message in enumerate(messages)
        if message.get("role") == "user"
    ]

    if len(turn_starts) <= keep_last_turns:
        return messages
    cut = turn_starts[-keep_last_turns]

    old = messages[1:cut]
    recent = messages[cut:]

    transcript = "\n".join(
        f"{message.get("role")}: {message.get('content') or '[tool call]'}"
        for message in old
    )

    response = ai_model_call(False,
                             [
                                 {
                                     "role": "system",
                                     "content": ("把对话压缩成不超过 200 字的中文摘要。"
                                                 "只保留已确认的事实、结论和用户明确表达的偏好，丢弃寒暄和中间过程。"
                                                 "只输出纯JSON。不要输出markdown")
                                 }, {
                                 "role": "user",
                                 "content": (
                                         "JSON output, format as :"
                                         + json.dumps({"summary": "summary"}, ensure_ascii=False)
                                         + "\n\n dialog content: \n" + transcript
                                 )
                             }
                             ],
                             {"type": "json_object"}
                             )
    content = response.choices[0].message.content;
    print(content)
    try:
        dialog = json.loads(content)
        summary = dialog.get("summary", "")
    except json.JSONDecodeError:
        print("summary parse failed, raw content:", content)
        raise

    summary_message = {
        "role": "user",
        "content": f"[History Summary]: {summary}"
    }

    return messages[:1] + [summary_message] + recent


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

        messages[:] = summarize_history(messages)

        stream = ai_model_call(True, messages, None, {"include_usage": True})
        text_parts: list[str] = []
        tool_call_buffer: dict[int, dict] = {}
        usage = None

        for chunk in stream:
            if not chunk.choices:
                continue

            if getattr(chunk, "usage", None):
                usage = chunk.usage

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
        if usage:
            print(
                f"[the {step} token input] {usage.prompt_tokens}"
                f"/ output： {usage.completion_tokens}"
                f"/ sum: {usage.total_tokens}"
            )
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
        if user_input.lower() in {"exit", "quit"}:
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
