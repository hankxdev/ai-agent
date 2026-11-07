"""Day 3 阶段 4：重试与自愈的单元测试。

这些测试**不需要 API Key、不需要网络**，全部是纯逻辑验证。
跑法：

    uv run python -m pytest day3/test_retry.py -v

或者不装 pytest 直接跑：

    uv run python day3/test_retry.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from day3.retry import (
    RetryExhausted,
    call_with_retry,
    compute_delay,
    describe_error,
    is_transient,
)

# ---------------------------------------------------------------------------
# 构造异常实例
#
# SDK 的异常需要 request/response 对象才能构造，
# 这里用最简的假对象，只保留异常自己会读的属性。
# ---------------------------------------------------------------------------


class FakeRequest:
    def __init__(self) -> None:
        self.method = "POST"
        self.url = "https://api.deepseek.com/chat/completions"


class FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.headers = {}
        self.request = FakeRequest()

    def json(self) -> dict:
        return {}

    def read(self) -> bytes:
        return b""


# ---------------------------------------------------------------------------
# 错误分类测试
# ---------------------------------------------------------------------------


def test_rate_limit_is_transient() -> None:
    error = RateLimitError("rate limited", response=FakeResponse(429), body=None)
    assert is_transient(error) is True


def test_timeout_is_transient() -> None:
    error = APITimeoutError(request=FakeRequest())
    assert is_transient(error) is True


def test_connection_error_is_transient() -> None:
    error = APIConnectionError(request=FakeRequest())
    assert is_transient(error) is True


def test_server_error_is_transient() -> None:
    error = InternalServerError("boom", response=FakeResponse(500), body=None)
    assert is_transient(error) is True


def test_auth_error_is_permanent() -> None:
    """认证失败重试一万次也是同样的错，必须立刻失败。"""
    error = AuthenticationError("bad key", response=FakeResponse(401), body=None)
    assert is_transient(error) is False


def test_bad_request_is_permanent() -> None:
    error = BadRequestError("bad params", response=FakeResponse(400), body=None)
    assert is_transient(error) is False


def test_plain_exception_is_permanent() -> None:
    """工具内部抛的 ValueError 不该触发 API 级重试。"""
    assert is_transient(ValueError("工具炸了")) is False


def test_describe_error_does_not_leak_internals() -> None:
    """错误描述里不应该出现本地路径或异常类名等内部信息。

    因为这段文本会被回灌给模型，等于把本地实现细节送出去了。
    """
    error = ValueError("/home/user/secret/project/config.py 不存在")
    text = describe_error(error)

    assert "/home/user" not in text
    assert "secret" not in text
    assert "ValueError" not in text


def test_sanitize_error_text_removes_paths() -> None:
    """工具层错误要保留业务信息，但必须抹掉本地路径。"""
    from day3.retry import sanitize_error_text

    posix = sanitize_error_text("无法读取 /home/user/projects/secret/config.yaml")
    assert "/home/user" not in posix
    assert "secret" not in posix
    # 业务信息本身要保留，模型才知道怎么改
    assert "无法读取" in posix

    windows = sanitize_error_text(r"无法打开 C:\Users\hankx\private\data.txt")
    assert "hankx" not in windows
    assert "private" not in windows
    assert "无法打开" in windows


def test_sanitize_error_text_truncates_long_input() -> None:
    from day3.retry import sanitize_error_text

    result = sanitize_error_text("x" * 1000, max_length=50)
    assert len(result) <= 53


# ---------------------------------------------------------------------------
# 退避计算测试
# ---------------------------------------------------------------------------


def test_delay_grows_exponentially() -> None:
    """退避上限随尝试次数指数增长，但不超过 max_delay。"""
    delays = [compute_delay(attempt, base_delay=1.0, max_delay=100.0) for attempt in range(1, 6)]

    # 抖动是随机的，不能断言精确值，只能断言量级
    assert 0 <= delays[0] <= 1.0
    assert 0 <= delays[1] <= 2.0
    assert 0 <= delays[2] <= 4.0
    assert 0 <= delays[3] <= 8.0
    assert 0 <= delays[4] <= 16.0


def test_delay_respects_max_cap() -> None:
    delays = [compute_delay(attempt, base_delay=1.0, max_delay=3.0) for attempt in range(1, 10)]
    assert all(delay <= 3.0 for delay in delays)


def test_delay_has_jitter() -> None:
    """同一 attempt 多次计算应产生不同值——这是防惊群的关键。"""
    values = {compute_delay(3, base_delay=1.0) for _ in range(20)}
    assert len(values) > 1, "退避没有抖动，重试时刻会同步碰撞"


# ---------------------------------------------------------------------------
# 重试执行器测试
# ---------------------------------------------------------------------------


def test_retry_succeeds_after_transient_failures() -> None:
    """瞬时错误重试后成功。"""
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise APIConnectionError(request=FakeRequest())
        return "ok"

    result = call_with_retry(flaky, base_delay=0.001)

    assert result == "ok"
    assert attempts["count"] == 3


def test_retry_does_not_retry_permanent_error() -> None:
    """确定性错误立刻抛出，不重试。"""
    attempts = {"count": 0}

    def always_bad() -> str:
        attempts["count"] += 1
        raise AuthenticationError("bad key", response=FakeResponse(401), body=None)

    try:
        call_with_retry(always_bad, base_delay=0.001)
    except AuthenticationError:
        pass
    else:
        raise AssertionError("应该抛出 AuthenticationError")

    assert attempts["count"] == 1, f"确定性错误不应重试，实际尝试了 {attempts['count']} 次"


def test_retry_exhausted_raises_with_original_error() -> None:
    """重试用尽后抛 RetryExhausted，且保留最后一次异常。"""
    def always_timeout() -> str:
        raise APITimeoutError(request=FakeRequest())

    try:
        call_with_retry(always_timeout, max_retries=2, base_delay=0.001)
    except RetryExhausted as error:
        assert error.attempts == 2
        assert isinstance(error.last_error, APITimeoutError)
    else:
        raise AssertionError("应该抛出 RetryExhausted")


def test_on_retry_callback_is_called() -> None:
    """重试过程必须可见，否则线上排查无从下手。"""
    seen: list[tuple[int, float]] = []

    def flaky() -> str:
        if len(seen) < 2:
            raise APIConnectionError(request=FakeRequest())
        return "ok"

    def on_retry(attempt: int, error: BaseException, delay: float) -> None:
        seen.append((attempt, delay))

    call_with_retry(flaky, base_delay=0.001, on_retry=on_retry)

    assert len(seen) == 2
    assert seen[0][0] == 1
    assert seen[1][0] == 2


# ---------------------------------------------------------------------------
# 死循环检测测试
# ---------------------------------------------------------------------------


def test_repeat_detector_flags_identical_calls() -> None:
    from day3.self_healing import RepeatCallDetector

    detector = RepeatCallDetector(threshold=2)

    assert detector.register("add", '{"a":1,"b":2}') is False
    assert detector.register("add", '{"a":1,"b":2}') is True


def test_repeat_detector_resets_on_different_args() -> None:
    from day3.self_healing import RepeatCallDetector

    detector = RepeatCallDetector(threshold=2)

    assert detector.register("add", '{"a":1,"b":2}') is False
    assert detector.register("add", '{"a":3,"b":4}') is False
    assert detector.register("add", '{"a":1,"b":2}') is False


# ---------------------------------------------------------------------------
# 工具错误回灌测试
# ---------------------------------------------------------------------------


def test_unknown_tool_returns_structured_error() -> None:
    from day3.self_healing import execute_tool_call

    content, ok = execute_tool_call("nonexistent_tool", "{}")

    assert ok is False
    assert "unknown_tool" in content
    # 错误信息里要告诉模型有哪些工具可用
    assert "add" in content


def test_invalid_arguments_returns_field_details() -> None:
    """参数非法时，错误里必须带字段级信息，模型才能自己改对。"""
    import json

    from day3.self_healing import execute_tool_call

    content, ok = execute_tool_call("add", '{"a": "not_a_number", "b": 2}')

    assert ok is False
    payload = json.loads(content)
    assert payload["error"] == "invalid_arguments"
    assert len(payload["details"]) > 0
    assert "a" in payload["details"][0]["field"]


def test_successful_tool_call() -> None:
    import json

    from day3.self_healing import execute_tool_call

    content, ok = execute_tool_call("add", '{"a": 128, "b": 45}')

    assert ok is True
    assert json.loads(content)["result"] == 173


# ---------------------------------------------------------------------------
# 极简测试运行器（不装 pytest 也能跑）
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
