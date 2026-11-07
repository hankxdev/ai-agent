"""Day 3 阶段 4：端到端测试（不依赖网络）。

前面的 test_retry.py 测的是单个函数。本文件测的是**完整链路**：

    模型返回坏参数 → 工具失败 → 错误回灌 → 模型自我纠正 → 调用成功 → 得出答案

用假的模型客户端驱动，不需要 API Key、不需要网络。
这正是「把策略与模型解耦」带来的好处：链路逻辑可以被完整验证。

跑法：
    uv run python day3/test_self_healing_e2e.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import APIConnectionError, RateLimitError

import day3.self_healing as agent_module

# ---------------------------------------------------------------------------
# 假的模型响应
# ---------------------------------------------------------------------------


class FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = FakeFunction(name, arguments)
        self.type = "function"


class FakeMessage:
    def __init__(self, content: str | None, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_unset: bool = False) -> dict:
        payload: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in self.tool_calls
            ]
        return payload


class FakeChoice:
    def __init__(self, message: FakeMessage) -> None:
        self.message = message
        self.finish_reason = "tool_calls" if message.tool_calls else "stop"


class FakeUsage:
    def __init__(self, total: int = 100) -> None:
        self.prompt_tokens = total // 2
        self.completion_tokens = total - total // 2
        self.total_tokens = total


class FakeResponse:
    def __init__(self, message: FakeMessage) -> None:
        self.choices = [FakeChoice(message)]
        self.usage = FakeUsage()


class FakeRequest:
    method = "POST"
    url = "https://api.deepseek.com/chat/completions"


class FakeResponseObj:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.headers = {}
        self.request = FakeRequest()

    def json(self) -> dict:
        return {}

    def read(self) -> bytes:
        return b""


# ---------------------------------------------------------------------------
# 假客户端：按预设脚本依次返回响应
# ---------------------------------------------------------------------------


class ScriptedClient:
    """按脚本依次返回响应，可以混入错误以测试重试。"""

    def __init__(self, script: list) -> None:
        self.script = script
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1

        if not self.script:
            raise AssertionError("脚本已耗尽，但模型仍被调用")

        item = self.script.pop(0)

        if isinstance(item, BaseException):
            raise item

        return FakeResponse(item)


def patch_client(script: list) -> ScriptedClient:
    """把 agent 模块里的 ai_model_call 换成假的。"""
    fake = ScriptedClient(script)

    def fake_call(stream, messages, response_format=None, stream_options=None, use_tools=True):
        return fake.create(stream=stream, messages=messages)

    agent_module.ai_model_call = fake_call
    return fake


# ---------------------------------------------------------------------------
# 测试场景
# ---------------------------------------------------------------------------


def test_happy_path_single_tool_call() -> None:
    """一次工具调用成功，直接给出答案。"""
    fake = patch_client([
        FakeMessage(None, [FakeToolCall("call_1", "add", '{"a": 128, "b": 45}')]),
        FakeMessage("128 加 45 等于 173。"),
    ])

    messages = [{"role": "user", "content": "算一下"}]
    answer = agent_module.run_agent(messages)

    assert answer == "128 加 45 等于 173。"
    assert fake.calls == 2

    # 消息历史顺序必须正确：user -> assistant(tool_calls) -> tool -> assistant
    roles = [m.get("role") for m in messages]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_self_heal_after_invalid_arguments() -> None:
    """核心场景：模型第一次传错参数，错误回灌后模型自己改对了。

    这就是「自愈」——不需要人介入，agent 自己纠正。
    """
    fake = patch_client([
        # 第一次：参数类型错误
        FakeMessage(None, [FakeToolCall("call_1", "add", '{"a": "128", "b": "45"}')]),
        # 第二次：模型读到错误信息，改成数字
        FakeMessage(None, [FakeToolCall("call_2", "add", '{"a": 128, "b": 45}')]),
        # 第三次：给出最终答案
        FakeMessage("结果是 173。"),
    ])

    messages = [{"role": "user", "content": "算一下"}]
    answer = agent_module.run_agent(messages)

    assert answer == "结果是 173。"
    assert fake.calls == 3

    # 第二条消息必须是 tool 角色，且携带错误信息
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 2

    first_error = json.loads(tool_messages[0]["content"])
    assert first_error["error"] == "invalid_arguments"
    # 必须带字段级细节，模型才知道怎么改
    assert first_error["details"][0]["field"] == "a"

    second_result = json.loads(tool_messages[1]["content"])
    assert second_result["result"] == 173


def test_api_retry_then_success() -> None:
    """API 限流后自动重试并成功。"""
    fake = patch_client([
        RateLimitError("rate limited", response=FakeResponseObj(429), body=None),
        APIConnectionError(request=FakeRequest()),
        FakeMessage("正常回答"),
    ])

    messages = [{"role": "user", "content": "你好"}]
    answer = agent_module.run_agent(messages)

    assert answer == "正常回答"
    assert fake.calls == 3, "前两次失败应触发重试"


def test_repeated_call_is_interrupted() -> None:
    """模型死循环调用同一个工具，应被中断而不是烧光 token。"""
    same_call = FakeMessage(None, [FakeToolCall("call_1", "add", '{"a": 1, "b": 2}')])
    fake = patch_client([same_call, same_call, same_call, same_call, same_call])

    messages = [{"role": "user", "content": "算一下"}]
    answer = agent_module.run_agent(messages)

    assert "重复工具调用" in answer
    assert fake.calls <= 3, f"应在检测到重复后立刻停止，实际调用了 {fake.calls} 次"


def test_repeated_call_with_different_args_is_allowed() -> None:
    """参数不同就不算重复——模型可能在做有意义的连续计算。"""
    fake = patch_client([
        FakeMessage(None, [FakeToolCall("call_1", "add", '{"a": 1, "b": 2}')]),
        FakeMessage(None, [FakeToolCall("call_2", "add", '{"a": 3, "b": 4}')]),
        FakeMessage("都算完了。"),
    ])

    messages = [{"role": "user", "content": "算一下"}]
    answer = agent_module.run_agent(messages)

    assert answer == "都算完了。"
    assert fake.calls == 3


def test_unknown_tool_is_reported_to_model() -> None:
    """模型调用不存在的工具，错误要回灌并列出可用工具。"""
    patch_client([
        FakeMessage(None, [FakeToolCall("call_1", "nonexistent", '{}')]),
        FakeMessage("抱歉，我用错工具了。"),
    ])

    messages = [{"role": "user", "content": "算一下"}]
    answer = agent_module.run_agent(messages)

    assert "用错工具" in answer

    tool_message = next(m for m in messages if m.get("role") == "tool")
    payload = json.loads(tool_message["content"])
    assert payload["error"] == "unknown_tool"
    assert "add" in payload["message"]


def test_max_steps_limit() -> None:
    """步数超限要停止并报告，不能无限循环。"""
    script = [
        FakeMessage(None, [FakeToolCall(f"call_{i}", "multiply", f'{{"a": {i}, "b": 2}}')])
        for i in range(20)
    ]
    fake = patch_client(script)

    messages = [{"role": "user", "content": "一直算"}]
    answer = agent_module.run_agent(messages)

    assert "最大步数" in answer
    assert fake.calls <= agent_module.MAX_STEPS


def test_api_failure_returns_graceful_message() -> None:
    """API 彻底失败时，agent 返回可读消息而不是抛异常。"""
    script = [
        APIConnectionError(request=FakeRequest())
        for _ in range(10)
    ]
    patch_client(script)

    messages = [{"role": "user", "content": "你好"}]
    answer = agent_module.run_agent(messages)

    assert answer.startswith("（调用 API 连续失败")
    assert "终止" in answer or "失败" in answer


def test_tool_result_truncation() -> None:
    """超长工具结果必须截断——这是 token 成本的第一大来源。"""
    # 直接测试截断逻辑：构造一个超长返回值
    import day3.self_healing as module
    from day3.self_healing import execute_tool_call

    original = module.TOOL_REGISTRY["add"]
    module.TOOL_REGISTRY["add"] = (lambda a, b: {"result": "x" * 10000}, original[1])
    try:
        content, ok = execute_tool_call("add", '{"a": 1, "b": 2}')
        assert ok is True
        assert "已截断" in content
        assert len(content) < 10000
    finally:
        module.TOOL_REGISTRY["add"] = original


# ---------------------------------------------------------------------------
# 测试运行器
# ---------------------------------------------------------------------------


def _run_all() -> None:
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]

    passed = 0
    failed: list[tuple[str, BaseException]] = []

    for name, test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS  {name}")
        except BaseException as error:
            failed.append((name, error))
            print(f"  FAIL  {name}: {type(error).__name__}: {error}")

    print(f"\n{passed} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run_all()
