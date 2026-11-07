from datetime import datetime

from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel, Field

from config import API_KEY, BASE_URL, MODEL


class AddArgs(BaseModel):
    a: int = Field(description="Argument 1", strict=True)
    b: int = Field(description="Argument 2", strict=True)


class MultiplyArgs(BaseModel):
    a: int = Field(description="第一个乘数", strict=True)
    b: int = Field(description="第二个乘数", strict=True)


def add(a: int, b: int) -> dict:
    """add two numbers"""
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError(f"only int is accepted, got {type(a).__name__} and {type(b).__name__}")
    return {
        "result": a + b
    }


def get_current_time() -> str:
    return str(datetime.now())


def multiply(a: int, b: int) -> dict:
    return {
        "result": a * b
    }


TOOL_REGISTRY = {
    "add": (add, AddArgs),
    "get_current_time": (get_current_time, None),
    "multiply": (multiply, MultiplyArgs),
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "第一个加数"},
                    "b": {"type": "integer", "description": "第二个加数"},
                },
                "required": ["a", "b"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前时间。当用户问现在几点时使用。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "第一个乘数"},
                    "b": {"type": "integer", "description": "第二个乘数"},
                }
            },
            "required": ["a", "b"],
        }
    }
]

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def ai_model_call(stream: bool, messages: list, response_format=None, stream_options=None) -> ChatCompletion | Stream[ChatCompletionChunk]:
    kwargs: dict = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "stream": stream,
    }

    if response_format is not None:
        kwargs["response_format"] = response_format

    if stream_options is not None:
        kwargs["stream_options"] = stream_options

    return client.chat.completions.create(**kwargs)
