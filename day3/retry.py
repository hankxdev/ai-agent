"""Day 3 阶段 4：重试与自愈 —— 机制层。

这一层不依赖 LLM，只解决一件事：**哪些错误该重试、重试几次、间隔多久**。

把它单独拆出来有两个理由：
1. 可以在没有网络、没有 API Key 的情况下用 pytest / assert 完整验证。
2. 策略与 Agent 主流程解耦，换模型、换 API 时策略不用重写。

三种失败必须区别对待：

| 失败类型 | 例子 | 处理 |
|---|---|---|
| 瞬时故障 | 网络抖动、429 限流、5xx、连接超时 | 指数退避后重试 |
| 确定性错误 | 参数非法、认证失败、模型不存在 | 立刻失败，重试无意义 |
| 业务失败 | 工具执行报错（除以零、文件不存在） | 不重试，把错误信息回灌给模型 |

第三类最容易搞错。工具报错重试一万次也是同样的错，
正确做法是把错误**告诉模型**，让模型换个思路——这才是「自愈」。
"""

import time
from collections.abc import Callable

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)

# ---------------------------------------------------------------------------
# 错误分类
# ---------------------------------------------------------------------------

# 这些错误重试有意义：服务端状态是临时的，下一次可能就成功了
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)

# 这些错误重试没有意义，重试只是浪费时间和 token
PERMANENT_ERRORS: tuple[type[BaseException], ...] = (
    # AuthenticationError / PermissionDeniedError / NotFoundError / BadRequestError
    # 都继承自 APIStatusError，由下面的 status_code 判断区分
)


def is_transient(error: BaseException) -> bool:
    """判断一个异常是否值得重试。

    规则：
    1. 显式列出的瞬时异常 -> 可重试
    2. APIStatusError 且 status_code >= 500 -> 可重试（服务端问题）
    3. APIStatusError 且 status_code == 429 -> 可重试（限流）
    4. 其他 APIStatusError（4xx）-> 不可重试（客户端问题，改参数也没用）
    5. 非 API 异常（如工具内部抛的 ValueError）-> 不可重试
    """
    if isinstance(error, TRANSIENT_ERRORS):
        return True

    if isinstance(error, APIStatusError):
        status_code = error.status_code
        # 429 限流 + 5xx 服务端错误都属于「等一会儿再来」
        return status_code == 429 or status_code >= 500

    return False


def describe_error(error: BaseException) -> str:
    """把异常转成一句能直接回灌给模型的话。

    注意：**不要把异常原文直接塞给模型**。Python 的异常字符串经常
    包含内部路径、类名、对端地址，对模型没有指导意义，还会泄��本地信息。
    这里只提取「模型能据此纠正行为」的部分。
    """
    if isinstance(error, RateLimitError):
        return "请求被限流，请稍等一小段时间后再试，或减少单次请求的内容量。"

    if isinstance(error, APITimeoutError):
        return "请求超时，请用更简短的方式重述需求，或把任务拆成更小的步骤。"

    if isinstance(error, APIConnectionError):
        return "网络连接失败，请直接重试同一个请求。"

    if isinstance(error, APIStatusError):
        if error.status_code >= 500:
            return f"服务端暂时不可用（HTTP {error.status_code}），请重试。"
        return f"请求被服务端拒绝（HTTP {error.status_code}），请调整请求内容后重试。"

    if isinstance(error, OpenAIError):
        return "调用 API 时出错。请简化请求内容后重试。"

    # 兜底分支：这里**不能回显异常原文**。
    # 通用异常的消息里常含本地绝对路径、类名、内部标识符，
    # 而这段文本会被回灌给模型，等于把本地实现细节送到云端。
    return "执行时出错。请调整请求内容后重试。"


def sanitize_error_text(text: str, max_length: int = 300) -> str:
    """把异常原文脱敏后再交给模型。

    工具层的错误信息对模型自愈有价值（它需要知道「除以零了」才能改），
    但必须先抹掉本地路径——那是最容易泄露实现细节的部分。

    保留异常类型名：ValueError / KeyError 这类是通用知识，
    对模型理解错误类型有帮助，不构成信息泄露。
    """
    import re

    # 匹配 Windows 路径 C:\Users\... 和 POSIX 路径 /home/user/...
    sanitized = re.sub(r"[A-Za-z]:[\\/][^\s'\"]*", "<path>", text)
    sanitized = re.sub(r"(?<![\w<])/[^\s'\"]*/[^\s'\"]*", "<path>", sanitized)

    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."

    return sanitized


# ---------------------------------------------------------------------------
# 退避策略
# ---------------------------------------------------------------------------

# 默认参数，可按场景覆盖
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 0.5
DEFAULT_MAX_DELAY = 8.0


def compute_delay(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
) -> float:
    """指数退避 + 抖动。

    第 1 次失败等 base，第二次 2*base，第三次 4*base……上限 max_delay。

    **抖动（jitter）是必须的**：如果多个客户端同时被 429，它们的重试时刻
    若完全一致，就会形成「惊群」，每次重试都再次触发限流。
    抖动把重试时刻打散，避免同步碰撞。

    这里用「全抖动」策略：delay = random(0, 上限)，
    这是 AWS 论文里推荐的、对 thundering herd 最有效的一种。
    """
    import random

    ceiling = min(max_delay, base_delay * (2 ** (attempt - 1)))
    return random.uniform(0, ceiling)


# ---------------------------------------------------------------------------
# 重试执行器
# ---------------------------------------------------------------------------


class RetryExhausted(Exception):
    """重试用尽仍然失败。包装最后一次异常，保留原始原因。"""

    def __init__(self, attempts: int, last_error: BaseException) -> None:
        super().__init__(f"重试 {attempts} 次后仍然失败：{describe_error(last_error)}")
        self.attempts = attempts
        self.last_error = last_error


def call_with_retry[T](
    func: Callable[[], T],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """执行 func，失败时按指数退避重试。

    只对瞬时错误重试。确定性错误立刻抛出，不浪费一次调用。

    on_retry 回调签名：(第几次尝试, 异常, 将等待的秒数)，
    用来把重试过程打印出来——生产环境里「看不见的重试」是排查噩梦。

    这里用 Python 3.12 的 PEP 695 泛型语法（`def f[T](...)`），
    比传统的 TypeVar 写法简洁，类型检查器也支持得更好。
    """
    attempts = 0

    while True:
        attempts += 1

        try:
            return func()
        except BaseException as error:
            if not is_transient(error):
                # 确定性错误直接抛，不重试
                raise

            if attempts > max_retries:
                raise RetryExhausted(attempts - 1, error) from error

            delay = compute_delay(attempts, base_delay, max_delay)

            if on_retry is not None:
                on_retry(attempts, error, delay)

            time.sleep(delay)
