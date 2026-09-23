"""Streaming responses.

Calls Google's Gemini API (gemini-3.1-flash-lite) with streaming enabled and
prints each piece of the answer as it arrives - the same incremental
-output idea as OpenAI's SSE streaming, over Gemini's streaming API.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import stream_chat


def stream_answer(prompt: str) -> str:
    print(f"Prompt: {prompt}\n")

    full_text = []
    try:
        for piece in stream_chat([{"role": "user", "content": prompt}]):
            print(piece, end="", flush=True)
            full_text.append(piece)
    except KeyboardInterrupt:
        print("\n[stream interrupted by user]")

    return "".join(full_text)


if __name__ == "__main__":
    stream_answer("List three benefits of streaming an LLM response, briefly.")
