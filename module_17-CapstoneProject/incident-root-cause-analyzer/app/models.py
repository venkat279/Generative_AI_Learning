"""Pydantic request/response models - FastAPI validates every incoming
request body against these, and uses them to generate the response schema.
"""

from pydantic import BaseModel


class IncidentRequest(BaseModel):
    alert: str  # e.g. "Elevated 502 errors on /checkout"
    services: list[str] = []  # which services the alert mentions
    start_time: str  # ISO 8601 UTC, e.g. "2026-08-30T09:10:00Z"
    end_time: str  # ISO 8601 UTC, e.g. "2026-08-30T09:30:00Z"


class TimelineEvent(BaseModel):
    timestamp: str
    service: str
    description: str


class RCAReport(BaseModel):
    incident_id: str
    summary: str
    timeline: list[TimelineEvent]
    root_cause: str
    evidence: list[str]
    recommended_fix: str
    confidence: str  # "low" | "medium" | "high"


class AnalyzeResponse(BaseModel):
    incident_id: str
    report: RCAReport
    step_count: int
    trace_url: str | None = None  # link to this incident's full trace in the Langfuse UI
