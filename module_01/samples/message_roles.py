"""System / user / assistant message roles.

Calls Google's Gemini API (gemini-3.1-flash-lite) - needs GEMINI_API_KEY in common/.env.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import chat

messages = [
    {"role": "system", "content": "You are a terse Python tutor. Answer in <=2 sentences."},
    {"role": "user", "content": "What is a context window?"},
    {"role": "assistant", "content": "It's the maximum number of tokens (input + output) a model can consider at once."},
    {"role": "user", "content": "Give one practical consequence of a small context window."},
]

if __name__ == "__main__":
    response = chat(messages=messages)
    print(response["message"]["content"].strip())
