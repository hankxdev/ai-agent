import os

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv("../.env")

API_KEY = os.getenv("API_KEY")
MODEL_ID = "deepseek-flash"

if not API_KEY:
    raise SystemExit("API_KEY not set")
else:
    print("key is set, go")

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/")

messages = [
    {
        "role": "system",
        "content": "You are a engineer, answer in 20 words"
    }
]
try :
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        if not user_input:
            continue

        messages.append({
            "role": "user",
            "content": user_input
        })

        stream = client.chat.completions.create(  # type: ignore[arg-type]
            model=MODEL_ID,
            messages=messages,
            stream=True
        )

        print("AI：", end="", flush=True)
        answer_parts = []
        for chunk in stream:
            if not chunk.choices:
                continue

            part = chunk.choices[0].delta.content
            if part:
                print(part, end="", flush=True)
                answer_parts.append(part)

        print()

        messages.append(
            {
                "role": "assistant",
                "content": "".join(answer_parts)
            }
        )
except KeyboardInterrupt:
    print("Ctrl+C pressed")
except OpenAIError as e:
    print("API error", e)

print(messages)
