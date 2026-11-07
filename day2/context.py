# noinspection DuplicatedCode
import json

from pydantic import ValidationError

from tools import TOOL_REGISTRY, ai_model_call

MAX_STEPS = 3
MAX_HISTORY_TURNS = 3

# 历史策略二选一："trim" 裁剪（零成本、丢信息）| "summarize" 摘要（多一次调用、保留信息）
HISTORY_STRATEGY = "summarize"

SUMMARY_PREFIX = "[History Summary]:"


def _is_summary(message: dict) -> bool:
    """判断一条消息是否是历史摘要占位消息"""
    content = message.get("content")
    return isinstance(content, str) and content.startswith(SUMMARY_PREFIX)


def _turn_starts(messages: list) -> list[int]:
    """轮次起点：role 为 user 且不是摘要占位消息

    摘要消息的 role 也是 user，如果算进轮次，会导致保留轮次偏移，
    且上一轮的摘要会被当成一轮对话再次压缩，信息反复损失。
    """
    return [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user" and not _is_summary(message)
    ]


def trim_history(messages: list, keep_last_turns: int = MAX_HISTORY_TURNS) -> list:
    """策略 A：按轮次裁剪历史。零额外调用，代价是丢掉旧信息"""
    turn_starts = _turn_starts(messages)

    if len(turn_starts) <= keep_last_turns:
        return messages

    cut = turn_starts[-keep_last_turns]

    # messages[:1] 保留 system 消息；被裁掉的旧摘要也一并丢弃
    return messages[:1] + messages[cut:]


def summarize_history(messages: list, keep_last_turns: int = MAX_HISTORY_TURNS) -> list:
    """策略 B：把旧轮次压缩成一条摘要消息，再保留最近若干轮。额外一次调用，信息有损"""
    turn_starts = _turn_starts(messages)

    if len(turn_starts) <= keep_last_turns:
        return messages
    cut = turn_starts[-keep_last_turns]

    old = messages[1:cut]
    recent = messages[cut:]

    # 上一次的摘要单独处理：它要参与本次摘要，但不能算作一轮对话
    previous = [message for message in old if _is_summary(message)]
    old_turns = [message for message in old if not _is_summary(message)]

    transcript = "\n".join(
        f"{message.get('role')}: {message.get('content') or '[tool call]'}"
        for message in old_turns
    )

    if previous:
        previous_text = "\n".join(str(message.get("content")) for message in previous)
        transcript = f"{previous_text}\n\n{transcript}"

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
        "content": f"{SUMMARY_PREFIX} {summary}"
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


def run_agent(messages: list, history_strategy: str = HISTORY_STRATEGY) -> None:
    for step in range(MAX_STEPS + 1):

        # 两种历史策略二选一，不是必须串联
        if history_strategy == "trim":
            messages[:] = trim_history(messages)
        elif history_strategy == "summarize":
            messages[:] = summarize_history(messages)
        else:
            raise ValueError(f"unknown history strategy: {history_strategy}")

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
    strategy = HISTORY_STRATEGY
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        # 运行时切换历史策略，方便对比两种方案
        if user_input.lower() in {"trim", "summarize"}:
            strategy = user_input.lower()
            print(f"history strategy -> {strategy}")
            continue

        if not user_input:
            continue
        messages.append({
            "role": "user",
            "content": user_input
        })

        run_agent(messages, strategy)


if __name__ == "__main__":
    main()
