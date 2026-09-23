"""Pydantic request/response models - FastAPI validates every incoming
request body against these automatically, and uses them to generate
the response schema too.
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    tenant_id: str = "default"
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending" | "done" | "error"
    result: str | None = None


class UploadResponse(BaseModel):
    job_id: str
    filename: str
