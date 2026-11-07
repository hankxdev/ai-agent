
import json
from openai import OpenAI, OpenAIError
from pydantic import ValidationError
from config import API_KEY, BASE_URL, MODEL

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

request = client.chat.completions.create( # type: ignore[arg-type]
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content" : """
            You are a senior engineer, in charge of review changes. you need to output json, never markdown.
            json schema:
            {
               "summary": "change summary",
                "risks": ["risk 1", "risk2"],
                "severity": "low、medium or high"
            }
            """
        },
        {
            "role": "user",
            "content": "review below change: to test the api easily, disable auth temporarily"
        }
    ],
    stream=False,
    max_tokens=400,
    response_format= {"type": "json_object"}
)
content = request.choices[0].message.content
if not content:
    raise ValidationError("Content cannot be empty")

structedMsg =  json.loads(content)
print("ALL content")
print(structedMsg)
print("summary")
print(structedMsg["summary"])
print("severity")
print(structedMsg["severity"])
print("risks")
for risk in structedMsg["risks"]:
    print(f" - {risk}")