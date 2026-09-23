"""Aggregates traces.jsonl into the dashboard-style metrics 
(latency, token usage, cost per user/feature, error rate) - 
the local stand-in for a Langfuse/LangSmith dashboard.
"""

import json
from collections import defaultdict

from tracer import LOG_PATH


def load_traces() -> list[dict]:
    with LOG_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(traces: list[dict]) -> dict:
    total = len(traces)
    errors = [t for t in traces if t["level"] == "ERROR"]
    ok = [t for t in traces if t["level"] != "ERROR"]

    by_feature = defaultdict(lambda: {"count": 0, "cost_usd": 0.0})
    by_user = defaultdict(lambda: {"count": 0, "cost_usd": 0.0})
    for t in traces:
        by_feature[t["feature"]]["count"] += 1
        by_feature[t["feature"]]["cost_usd"] += t["cost_usd"]
        by_user[t["user_id"]]["count"] += 1
        by_user[t["user_id"]]["cost_usd"] += t["cost_usd"]
    for bucket in (by_feature, by_user):
        for stats in bucket.values():
            stats["cost_usd"] = round(stats["cost_usd"], 6)

    return {
        "total_requests": total,
        "error_rate": round(len(errors) / total, 3) if total else 0.0,
        "avg_latency_ms": round(sum(t["latency_ms"] for t in ok) / len(ok), 1) if ok else 0.0,
        "avg_ttft_ms": round(
            sum(t["ttft_ms"] for t in ok if t["ttft_ms"] is not None)
            / max(sum(1 for t in ok if t["ttft_ms"] is not None), 1),
            1,
        ),
        "total_tokens": sum(t["total_tokens"] for t in traces),
        "total_cost_usd": round(sum(t["cost_usd"] for t in traces), 6),
        "by_feature": dict(by_feature),
        "by_user": dict(by_user),
    }
