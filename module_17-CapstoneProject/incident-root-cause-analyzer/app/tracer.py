"""Tracing via Langfuse.

Credentials come from the shared common/.env (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
LANGFUSE_HOST) via python-dotenv. Run this file (`python tracer.py`) 
to verify credentials are wired up before running the full agent.

Every step of an incident investigation becomes one Langfuse observation,
nested under a root "incident_investigation" trace whose trace_id is
derived deterministically from incident_id (create_trace_id(seed=...)),
so get_trace(incident_id) can look it up later with no local state of its
own. Two observation types are used deliberately, not a generic "span"
for everything: log_tool() for each deterministic tool call, and
log_generation() for each real LLM call (with model name and token counts
attached) - this is what makes Langfuse's per-model token/cost dashboards
populate correctly.
"""

from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.api.core.api_error import ApiError

load_dotenv(Path(__file__).resolve().parents[3] / "common" / ".env")  # shared with every other module

_client = Langfuse()


class IncidentTracer:
    """One instance per incident investigation. Opens a root Langfuse span
    on construction; log_tool()/log_generation() add a child span per
    step; finish() closes the root span and flushes to Langfuse."""

    def __init__(self, incident_id: str, alert: str):
        self.incident_id = incident_id
        self.trace_id = _client.create_trace_id(seed=incident_id)
        self._step_index = 0
        self._root = _client.start_observation(
            trace_context={"trace_id": self.trace_id},
            name="incident_investigation",
            as_type="chain",
            input={"alert": alert},
        )

    def log_tool(self, name: str, output: str, input: dict | None = None) -> None:
        """Log one tool call+result (no LLM involved) as a "tool"
        observation under the root trace."""
        self._step_index += 1
        span = self._root.start_observation(name=name, as_type="tool", input=input, output=output)
        span.end()
        print(f"  [trace] step {self._step_index} [tool] {name}", flush=True)

    def log_generation(self, name: str, output: str, model: str, usage_details: dict) -> None:
        """Log one real LLM call as a "generation" observation - what
        populates Langfuse's model/token/cost dashboards, unlike a
        generic span. usage_details uses Langfuse's {"input", "output",
        "total"} keys."""
        self._step_index += 1
        span = self._root.start_observation(
            name=name, as_type="generation", output=output, model=model, usage_details=usage_details
        )
        span.end()
        print(f"  [trace] step {self._step_index} [generation] {name} (model={model}, tokens={usage_details})", flush=True)

    def finish(self, output: str | None = None) -> None:
        self._root.update(output=output)
        self._root.end()
        _client.flush()

    @property
    def step_count(self) -> int:
        return self._step_index

    def get_trace_url(self) -> str | None:
        return _client.get_trace_url(trace_id=self.trace_id)


def get_trace(incident_id: str) -> list[dict]:
    """Read back every logged step for one incident - the post-incident
    audit trail. Returns [] only for a genuinely unknown incident_id
    (404); other API errors (e.g. bad credentials) propagate, so a
    config problem doesn't look like a 404."""
    trace_id = _client.create_trace_id(seed=incident_id)
    try:
        trace = _client.api.trace.get(trace_id)
    except ApiError as exc:
        if exc.status_code == 404:
            return []
        raise

    observations = sorted(trace.observations, key=lambda o: o.start_time)
    return [
        {
            "name": obs.name,
            "type": obs.type,
            "input": obs.input,
            "output": obs.output,
            "start_time": obs.start_time.isoformat(),
        }
        for obs in observations
    ]


if __name__ == "__main__":
    try:
        _client.auth_check()
        print("Langfuse credentials OK")
    except Exception as exc:
        print(f"Langfuse credentials INVALID - check common/.env ({exc})")
