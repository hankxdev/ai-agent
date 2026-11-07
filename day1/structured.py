import json
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from config import API_KEY, BASE_URL, MODEL


class ReviewResult(BaseModel):
    summary: str = Field(description="Summary of the review")
    risks: list = Field(description="List of risks")
    severity: Literal["Low", "Medium", "High"] = Field(description="Severity of the review")
    score: int = Field(ge=0, le=10, description="Score of the review")


client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

schema = ReviewResult.model_json_schema()

response = client.chat.completions.create(  # type: ignore
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content":
                "你是一个风险评审员，以简短的答案给出评审意见。结果用json，不用markdown。 必须严格符合以下scheme \n"
                f"{json.dumps(schema, ensure_ascii=False)}"

        },
        {
            "role": "user",
            "content": "评审以下决策：为了加快产品开发速度，用Mendix来开发公司的核心应用"
        }
    ],
    stream=False,
    max_tokens=400,
    response_format={"type": "json_object"}
)

content = response.choices[0].message.content
if not content:
    raise RuntimeError("message content is empty")

try:
    result = ReviewResult.model_validate_json(content)
except ValidationError as e:
    print("Validation Error")
    print(e)
    raise SystemExit(1) from e

print("\n=== 校验通过 ===")
print("综述")
print(result.summary)
print("风险级别")
print(result.severity)
print("风险点")
for i in result.risks:
    print(f"- {i}")
print("风险评分")
print(result.score)

