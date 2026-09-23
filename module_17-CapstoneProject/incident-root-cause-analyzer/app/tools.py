"""Agent tools: log/metric retrieval.

fetch_logs and fetch_metrics stand in for a real log/metrics platform
(CloudWatch, Splunk, ELK) - same interface (service + time window in,
matching entries out), reading local flat files instead of a real backend.

Correlate error patterns: the agent calls the tools for more than one 
service and compares timestamps to find which service failed first.
"""

import json
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

DATA_DIR = Path(__file__).parent / "data"
LOGS_DIR = DATA_DIR / "logs"
METRICS_DIR = DATA_DIR / "metrics"

KNOWN_SERVICES = ["api-gateway", "payments-service", "postgres-db"]


def _parse_ts(line: str) -> datetime | None:
    try:
        return datetime.fromisoformat(line.split(" ", 1)[0].replace("Z", "+00:00"))
    except (ValueError, IndexError):
        return None


def _in_window(ts: datetime, start: datetime, end: datetime) -> bool:
    return start <= ts <= end


def _read_log_lines(service: str, start_time: str, end_time: str) -> list[str]:
    path = LOGS_DIR / f"{service}.log"
    if not path.exists():
        return []
    start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

    matched = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ts = _parse_ts(line)
        if ts is not None and _in_window(ts, start, end):
            matched.append(line)
    return matched


@tool
def fetch_logs(service: str, start_time: str, end_time: str) -> str:
    """Fetch raw log lines for one service within a time window (stands in for a
    CloudWatch Logs Insights / Splunk / ELK query). service must be one of:
    api-gateway, payments-service, postgres-db.
    start_time and end_time are ISO 8601 UTC timestamps, e.g. 2026-08-30T09:10:00Z."""
    lines = _read_log_lines(service, start_time, end_time)
    if not lines:
        return f"No log entries found for {service} between {start_time} and {end_time}."
    return "\n".join(lines)


@tool
def fetch_metrics(service: str, start_time: str, end_time: str) -> str:
    """Fetch metric samples (error_rate_pct, p99_latency_ms, cpu_pct) for one
    service within a time window (stands in for a metrics/monitoring dashboard
    query). service must be one of: api-gateway, payments-service, postgres-db.
    start_time and end_time are ISO 8601 UTC timestamps, e.g. 2026-08-30T09:10:00Z."""
    path = METRICS_DIR / f"{service}.json"
    if not path.exists():
        return f"No metrics available for {service}."

    start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
    samples = json.loads(path.read_text(encoding="utf-8"))

    matched = [s for s in samples if _in_window(datetime.fromisoformat(s["timestamp"].replace("Z", "+00:00")), start, end)]
    if not matched:
        return f"No metric samples found for {service} between {start_time} and {end_time}."
    return "\n".join(
        f"{s['timestamp']} error_rate_pct={s['error_rate_pct']} p99_latency_ms={s['p99_latency_ms']} cpu_pct={s['cpu_pct']}"
        for s in matched
    )
