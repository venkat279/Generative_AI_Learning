"""A hand-built local tracer standing in for LangSmith/Langfuse. 
`Span` is the manual spans-and-generations pattern Langfuse's SDK exposes; 
instead of a hosted dashboard, every finished span is appended as one 
structured JSON line to a local `traces.jsonl` file.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pii_scrub import scrub

LOG_PATH = Path(__file__).parent / "traces.jsonl"

# USD per 1M tokens - illustrative pricing for cost visibility (a fixed 
# mid-tier paid-API rate, not Gemini's actual price list).
PRICING_PER_1M = {"input": 2.50, "output": 10.00}


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    cost = (
        prompt_tokens / 1_000_000 * PRICING_PER_1M["input"]
        + completion_tokens / 1_000_000 * PRICING_PER_1M["output"]
    )
    return round(cost, 6)


class Span:
    """One traced unit of work. Usage:

        with Span("chat", user_id="u1", feature="chat") as span:
            ... do the LLM call, calling span.mark_first_token() on the
            ... first streamed piece ...
            span.finish(prompt=..., response=..., prompt_tokens=..., completion_tokens=...)
    """

    def __init__(self, name: str, user_id: str, feature: str, request_id: str | None = None):
        self.name = name
        self.user_id = user_id
        self.feature = feature
        self.request_id = request_id or str(uuid.uuid4())
        self._start = None
        self._first_token_at = None

    def __enter__(self):
        import time

        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def mark_first_token(self):
        """Call once, on the first streamed piece, to record time-to-first-token."""
        import time

        if self._first_token_at is None:
            self._first_token_at = time.monotonic()

    def finish(
        self,
        *,
        level: str = "INFO",
        prompt: str = "",
        response: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error: str | None = None,
    ) -> dict:
        import time

        end = time.monotonic()
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": self.request_id,
            "span": self.name,
            "user_id": self.user_id,
            "feature": self.feature,
            "level": level,
            "latency_ms": round((end - self._start) * 1000, 1),
            "ttft_ms": round((self._first_token_at - self._start) * 1000, 1) if self._first_token_at else None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": _estimate_cost(prompt_tokens, completion_tokens),
            "prompt": scrub(prompt),
            "response": scrub(response),
            "error": error,
        }
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry
