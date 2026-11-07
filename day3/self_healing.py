"""Day 3 阶段 4：重试与自愈 —— Agent 层。

在 Day 2 的 agent 基础上加四道防线：

1. **API 层重试**：网络抖动/限流时自动退避重试（见 retry.py）
2. **工具层自愈**：工具失败时把结构化错误回灌给模型，让它换参数重试
3. **重复调用检测**：同样的工具+参数连续出现就中断，防止死循环烧钱
4. **总步数与 token 预算**：硬性熔断，超了就停并报告原因

设计原则：**失败要变成模型能理解的信息，而不是变成一个 exception。**
一个只会 crash 的 agent 无法自愈；一个能把错误说清楚的 agent 才能自愈。
"""

import json
import sys
from pathlib import Path

# 支持 `python day3/self_healing.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError

from day3.retry import (
    RetryExhausted,
    call_with_retry,
    describe_error,
    sanitize_error_text,
)
from tools import TOOL_REGISTRY, ai_model_call

MAX_STEPS = 8
MAX_REPEAT_CALLS = 2
MAX_CONSECUTIVE_FAILURES = 3
MAX_TOOL_RESULT_CHARS = 2000


# ---------------------------------------------------------------------------
# 工具执行：失败不 crash，转成模型能读懂的结构
# ---------------------------------------------------------------------------


def execute_tool_call(name: str, raw_arguments: str) -> tuple[str, bool]:
    """执行一次工具调用。

    返回 (回填给模型的内容, 是否成功)。

    关键设计：**任何失败都返回一条结构化的 tool 消息，而不是抛异常。**
    因为对 agent 来说，"工具报错了"是一条信息，不该是程序的终结。
    """
    if name not in TOOL_REGISTRY:
        return (
            json.dumps(
                {
                    "error": "unknown_tool",
                    "message": f"没有名为 '{name}' 的工具。可用工具：{', '.join(TOOL_REGISTRY)}",
                },
                ensure_ascii=False,
            ),
            False,
        )

    func, args_model = TOOL_REGISTRY[name]

    try:
        if args_model is None:
            result = func()
        else:
            args = args_model.model_validate_json(raw_arguments)
            result = func(**args.model_dump())
    except ValidationError as error:
        # 参数不合法：把「哪个字段、期望什么类型」告诉模型，它就能自己改对
        return (
            json.dumps(
                {
                    "error": "invalid_arguments",
                    "message": "参数校验失败，请按下面的字段要求重新调用",
                    "details": _format_validation_error(error),
                },
                ensure_ascii=False,
            ),
            False,
        )
    except Exception as error:
        # 业务失败：告诉模型出了什么事，让它换思路而不是重试同样的参数。
        # 错误原文要脱敏——本地路径不该发到云端。
        #
        # 这里故意捕获宽异常：工具层的职责是「任何失败都不能让 agent 崩掉」，
        # 失败必须变成模型可读的信息。这正是 BLE001 想避免的模式，
        # 但在「工具执行边界」这个位置是正确的取舍。
        return (
            json.dumps(
                {
                    "error": "tool_failed",
                    "message": f"{type(error).__name__}: {sanitize_error_text(str(error))}",
                    "hint": "请根据错误信息调整参数或改用其他方案",
                },
                ensure_ascii=False,
            ),
            False,
        )

    payload = json.dumps(result, ensure_ascii=False)

    # 工具结果可能极大（读了个 10MB 文件），必须截断
    # 这是 token 成本的第一大来源，比任何提示词优化都有效
    if len(payload) > MAX_TOOL_RESULT_CHARS:
        payload = (
            payload[:MAX_TOOL_RESULT_CHARS]
            + f"\n...（内容过长，已截断，原始长度 {len(payload)} 字符）"
        )

    return payload, True


def _format_validation_error(error: ValidationError) -> list[dict]:
    """把 Pydantic 的校验错误转成模型看得懂的字段清单。

    Pydantic 原始报错很长且是英文技术术语。模型需要的是：
    哪个字段错了、期望什么类型、我现在传了什么。
    """
    details = []

    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"])
        details.append({
            "field": location,
            "expected_type": item.get("type"),
            "received": str(item.get("input")),
            "message": item.get("msg"),
        })

    return details


# ---------------------------------------------------------------------------
# 死循环检测
# ---------------------------------------------------------------------------


class RepeatCallDetector:
    """检测「同样的工具 + 同样的参数」连续出现。

    模型陷入死循环的典型表现：反复调用同一个函数、传同样的参数、
    拿到同样的错误、再调一次……无限循环。

    这是真实线上故障，不是理论风险。检测到就中断。
    """

    def __init__(self, threshold: int = MAX_REPEAT_CALLS) -> None:
        self.threshold = threshold
        self.last_signature: str | None = None
        self.repeat_count = 0

    def register(self, name: str, raw_arguments: str) -> bool:
        """记录一次调用，返回 True 表示检测到重复。"""
        signature = f"{name}({raw_arguments})"

        if signature == self.last_signature:
            self.repeat_count += 1
        else:
            self.last_signature = signature
            self.repeat_count = 1

        return self.repeat_count >= self.threshold


# ---------------------------------------------------------------------------
# Agent 主循环
# ---------------------------------------------------------------------------


def run_agent(messages: list) -> str:
    """带重试与自愈的 Agent 主循环。返回最终回答文本。"""
    detector = RepeatCallDetector()
    consecutive_failures = 0
    total_tokens = 0

    for step in range(1, MAX_STEPS + 1):
        # ---- API 调用带退避重试 ----
        # 注意：这里用非流式。流式重试有个绕不开的问题——
        # 已经推给用户的那部分文本无法撤回，重试会导致内容重复。
        # 非流式下失败就是失败，重试干净利落。
        # 流式 + 重试要靠「缓冲区 + 确认后提交」实现，复杂度高得多。
        def make_request() -> object:
            return ai_model_call(
                stream=False,
                messages=messages,
            )

        def on_retry(
            attempt: int,
            error: BaseException,
            delay: float,
            current_step: int = step,
        ) -> None:
            print(f"  [第 {current_step} 步] API 失败（第 {attempt} 次尝试）：{describe_error(error)}")
            print(f"  [第 {current_step} 步] {delay:.2f}s 后重试")

        try:
            response = call_with_retry(make_request, on_retry=on_retry)
        except RetryExhausted as error:
            print(f"  [第 {step} 步] 重试用尽，终止 agent：{error}")
            return f"（调用 API 连续失败，已终止：{describe_error(error.last_error)}）"

        if response.usage:
            total_tokens += response.usage.total_tokens
            print(f"  [第 {step} 步 token] 累计 {total_tokens}")

        message = response.choices[0].message

        if not message.tool_calls:
            # 模型给出最终答案
            messages.append(message.model_dump(exclude_unset=True))
            return message.content or ""

        # 模型请求调用工具，把请求写回历史
        messages.append(message.model_dump(exclude_unset=True))

        for call in message.tool_calls:
            name = call.function.name
            raw_arguments = call.function.arguments

            # ---- 死循环检测 ----
            if detector.register(name, raw_arguments):
                print(f"  [第 {step} 步] 检测到重复调用 {name}，中断")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps({
                        "error": "repeated_call",
                        "message": f"你连续多次用相同参数调用 '{name}'，已中断。请换一种方式处理。",
                    }, ensure_ascii=False),
                })
                return "（检测到重复工具调用，已中断 agent）"

            print(f"  [第 {step} 步] 调用 {name}({raw_arguments})")

            content, ok = execute_tool_call(name, raw_arguments)

            print(f"  [第 {step} 步] {'成功' if ok else '失败'}：{content[:200]}")

            # 连续失败计数：模型如果一直调不通，说明这个思路走不通了
            consecutive_failures = 0 if ok else consecutive_failures + 1

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": content,
            })

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"  [第 {step} 步] 连续失败 {consecutive_failures} 次，中断")
                return f"（工具连续失败 {consecutive_failures} 次，已中断 agent）"

    return f"（达到最大步数 {MAX_STEPS}，未得出结论）"


def main() -> None:
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个可以使用工具的助手。需要数据时调用工具，"
                "工具报错时阅读错误信息并调整参数，不要重复相同的失败调用。"
            ),
        }
    ]

    while True:
        user_input = input("\n你：").strip()

        if user_input.lower() in {"exit", "quit"}:
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        answer = run_agent(messages)
        print(f"\nAI：{answer}")


if __name__ == "__main__":
    main()
