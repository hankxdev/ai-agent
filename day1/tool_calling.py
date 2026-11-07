import json
from datetime import datetime

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

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

messages = [
    {
        "role": "system",
        "content": "you are a helpful tool, call tools when needed, then answer with tool result"
    },
    {
        "role": "user",
        "content": "what's the time now? and help to calculate 101.1+202, and also 15 * '15'"

    }
]

request = client.chat.completions.create(  # type:ignore
    model=MODEL,
    messages=messages,
    tools=tools,
    stream=False
)

message = request.choices[0].message
print("finish_reason：", request.choices[0].finish_reason)

if not message.tool_calls:
    print("model answer without tool calls", message.content)
    raise SystemExit(1)

messages.append(message.model_dump(exclude_unset=True))

for call in message.tool_calls:
    name = call.function.name
    raw_args = call.function.arguments
    print(f"AI is trying to call {name},  call args: {raw_args}")

    if name not in TOOL_REGISTRY:
        result = {"error": f"未知工具：{name}"}
    else:
        func, args_model = TOOL_REGISTRY[name]
        try:
            if args_model is not None:
                args = args_model.model_validate_json(raw_args)
                result = func(**args.model_dump())
            else:
                result = func()
        except ValidationError as e:
            result = {"error": f"参数校验失败：{e}"}
        except (ValueError, TypeError) as e:
            result = {"error": f"工具执行失败：{e}"}

    print("tool call result:", result)

    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "content": json.dumps(result, ensure_ascii=False),
    })

final = client.chat.completions.create(  # type: ignore
    model=MODEL,
    messages=messages,
    tools=tools,
    stream=False
)

print("\n最终回答：")
print(final.choices[0].message.content)

print(final.choices[0].finish_reason)
