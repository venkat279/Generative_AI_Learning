"""Structured output prompting (JSON) with validation + retry.

Calls Google's Gemini API (gemini-3.1-flash-lite) with JSON mode enabled, plus a
real validate-then-retry loop against the model's actual output (a
capable model in JSON mode usually succeeds on the first attempt, but
the retry path is still real and will engage if it doesn't).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import chat

SYSTEM_PROMPT = """You extract structured data from text.
Always respond with ONLY valid JSON matching this schema, no prose, no markdown fences:
{"name": string, "age": integer | null, "city": string | null}"""


def extract_json(text: str, max_attempts: int = 3) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    for attempt in range(1, max_attempts + 1):
        response = chat(messages, json_mode=True)
        raw = response["message"]["content"]
        try:
            data = json.loads(raw)
            required = {"name", "age", "city"}
            if not required.issubset(data.keys()):
                raise ValueError(f"missing keys: {required - data.keys()}")
            return data
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[attempt {attempt}] invalid JSON ({exc}), retrying...")
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"That was invalid: {exc}. Return ONLY corrected JSON."})

    raise RuntimeError("Model failed to produce valid JSON after retries")


if __name__ == "__main__":
    result = extract_json("Hi, I'm Samantha, 29 years old, based in Chennai.")
    print(result)
