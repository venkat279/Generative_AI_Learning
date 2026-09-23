"""Incident Root Cause Analyzer

A FastAPI backend around the LangGraph agent in agent.py: investigates
an alert's logs/metrics via tool calls, matches the pattern to a runbook
via RAG, returns a structured RCA report, and traces every step to Langfuse.

Run a real server:
    uvicorn app:app --reload

Or drive this app object directly with no separate server process, via
the interactive UI:
    streamlit run streamlit_app.py
"""

import sys
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

sys.stdout.reconfigure(encoding="utf-8")

from agent import analyze_incident
from models import AnalyzeResponse, IncidentRequest, RCAReport
from tracer import get_trace

app = FastAPI(title="Module 17 - Incident Root Cause Analyzer")


@app.exception_handler(Exception)
def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Safety net: agent.analyze_incident() already handles the one known
    failure mode (LangGraph's recursion limit) gracefully; anything else
    still comes back as a clean 500 instead of a raw traceback."""
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


@app.get("/health")
def health() -> dict:
    """Sync endpoint - deliberately plain `def`."""
    return {"status": "ok"}


@app.post("/incidents/analyze", response_model=AnalyzeResponse)
def analyze(payload: IncidentRequest) -> AnalyzeResponse:
    """Sync endpoint wrapping the agent's tool-calling loop plus the
    structured-extraction call - both are blocking Gemini API calls, and
    FastAPI runs sync `def` route functions in a thread pool automatically,
    so this doesn't block the event loop."""
    incident_id = str(uuid.uuid4())
    report_dict, step_count, trace_url = analyze_incident(
        incident_id=incident_id,
        alert=payload.alert,
        services=payload.services,
        start_time=payload.start_time,
        end_time=payload.end_time,
    )
    report = RCAReport(**report_dict)
    return AnalyzeResponse(incident_id=incident_id, report=report, step_count=step_count, trace_url=trace_url)


@app.get("/incidents/{incident_id}/trace")
def trace(incident_id: str) -> list[dict]:
    """Post-incident audit: the full step-by-step trace for one past
    analysis, read back from Langfuse (also viewable via the trace_url
    returned by POST /incidents/analyze)."""
    entries = get_trace(incident_id)
    if not entries:
        raise HTTPException(status_code=404, detail="Unknown incident id")
    return entries


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
