import json
from typing import Dict
from ollama import chat
from ollama import ChatResponse
from utils import load_system_prompt


def call_local_llm(user_content: str) -> str:
    system_prompt = load_system_prompt()

    try:
        response: ChatResponse = chat(
            model="llama3.1:8b",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            format="json",
        )

        total_tokens = response.get("prompt_eval_count", 0) + response.get(
            "eval_count", 0
        )

        print(response.message.content)

        parsed = safe_json_parse(response.message.content)
        tokens = total_tokens

        return {"summary": parsed, "tokens": tokens}

    except Exception as e:
        return f"LLM call failed: {e}"


def safe_json_parse(text: str) -> Dict:

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except Exception:
            print("Failed to parse JSON from LLM output.")
            return {
                "key_topics": [],
                "decisions": [],
                "important_mentions": [],
                "action_items": [],
            }
