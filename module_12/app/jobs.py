"""In-memory job store backing the background-task + status-polling
pattern used by both the report-generation and file-upload-processing
endpoints in app.py.
"""

import uuid

_jobs: dict[str, dict] = {}


def create_job() -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "result": None}
    return job_id


def set_result(job_id: str, result: str) -> None:
    _jobs[job_id] = {"status": "done", "result": result}


def set_error(job_id: str, error: str) -> None:
    _jobs[job_id] = {"status": "error", "result": error}


def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)
