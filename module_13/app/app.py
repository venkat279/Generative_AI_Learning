"""Module 13 hands-on: trace a handful of chat requests, log them as
structured JSON with PII scrubbed out, deliberately trigger one real error 
to exercise error-rate tracking, then print a dashboard-style summary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import stream_chat  # noqa: E402

import requests
import tiktoken

import metrics
from tracer import LOG_PATH, Span

# Approximate tokenizer for prompt/completion counts
# (tiktoken's cl100k_base, not Gemini's own tokenizer)
ENCODING = tiktoken.get_encoding("cl100k_base")

REQUESTS = [
    {"user_id": "user_1", "feature": "chat", "message": "Give a one-sentence tagline for a coffee shop."},
    {"user_id": "user_2", "feature": "chat", "message": "Explain what a for loop is, in one sentence."},
    {
        "user_id": "user_1",
        "feature": "summarize",
        "message": "Summarize in one sentence: The quick brown fox jumps over the lazy dog on a sunny afternoon.",
    },
    {
        "user_id": "user_2",
        "feature": "chat",
        "message": "My email is jane.doe@example.com and my phone is 555-123-4567 - please remember that for next time.",
    },
]


def run_traced_chat(req: dict) -> None:
    with Span("chat", user_id=req["user_id"], feature=req["feature"]) as span:
        messages = [{"role": "user", "content": req["message"]}]
        pieces = []
        for i, piece in enumerate(stream_chat(messages)):
            if i == 0:
                span.mark_first_token()
            pieces.append(piece)
        response_text = "".join(pieces)

        entry = span.finish(
            prompt=req["message"],
            response=response_text,
            prompt_tokens=len(ENCODING.encode(req["message"])),
            completion_tokens=len(ENCODING.encode(response_text)),
        )
    print(
        f"[{entry['level']}] {req['feature']}/{req['user_id']} req={entry['request_id'][:8]} "
        f"- {entry['latency_ms']}ms total, {entry['ttft_ms']}ms to first token, ${entry['cost_usd']}"
    )
    print(f"    logged prompt: {entry['prompt']!r}")


def run_failing_call() -> None:
    """A deliberately unreachable port - a real connection failure (not
    simulated), to prove the tracer's ERROR path and error-rate metric work
    against a genuine failure, not a mocked one.
    """
    with Span("chat", user_id="user_2", feature="chat") as span:
        try:
            requests.post("http://localhost:19999/api/chat", json={}, timeout=3)
        except requests.exceptions.RequestException as e:
            entry = span.finish(level="ERROR", prompt="(connection attempt to a down backend)", error=str(e))
            print(f"[{entry['level']}] chat/user_2 req={entry['request_id'][:8]} - {entry['error'][:90]}")


def main() -> None:
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    print("Running traced requests...\n")
    for req in REQUESTS:
        run_traced_chat(req)
    run_failing_call()

    print(f"\nRaw trace log written to {LOG_PATH.name} - one JSON object per line, PII already scrubbed.")

    print("\n--- Dashboard (aggregated from traces.jsonl) ---")
    traces = metrics.load_traces()
    summary = metrics.summarize(traces)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
