"""Module 12 mini application: a FastAPI backend covering sync/async routes, SSE 
streaming, a WebSocket endpoint, background tasks with status polling, Pydantic models, 
dependency-injected auth/rate-limiting, CORS, a custom exception handler for LLM errors, 
session/state management, and file uploads with async processing.

Run a real server:
    uvicorn app:app --reload
    (or: python app.py, which does the same via uvicorn.run)

Or run the scripted walkthrough against this app object directly, with no server needed:
    python demo.py
"""

import sys
import tempfile
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from requests.exceptions import RequestException
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import chat, stream_chat

import jobs
from deps import enforce_rate_limit, session_store
from models import ChatRequest, ChatResponse, JobStatus, UploadResponse

MAX_UPLOAD_BYTES = 200_000
ALLOWED_UPLOAD_TYPES = {"text/plain", "text/markdown"}
SYSTEM_PROMPT = "You are a concise assistant."

app = FastAPI(title="Module 12 - GenAI Production API")

# CORS is real middleware; auth/rate limiting are dependencies below instead (§2).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(RequestException)
async def llm_unavailable_handler(request, exc: RequestException) -> JSONResponse:
    """Exception handler for LLM errors: turn a raw connection failure
    talking to the Gemini API into a clean 503 instead of an unhandled 500."""
    return JSONResponse(status_code=503, content={"detail": "The language model backend is unavailable."})


@app.get("/health")
def health() -> dict:
    """Sync endpoint - deliberately plain `def`, not `async def`: for a
    trivial check like this with no slow I/O to await, FastAPI running
    it in its thread pool costs nothing."""
    return {"status": "ok", "active_sessions": session_store.active_session_count()}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(enforce_rate_limit)])
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Sync endpoint wrapping a blocking LLM call - FastAPI runs sync
    `def` route functions in a thread pool automatically, so this
    doesn't block the event loop despite `common/llm_client.py`'s
    `chat()` being a plain blocking `requests` call. Session state
    (§3) is looked up/updated by (tenant_id, conversation_id); a new
    conversation id is generated with `uuid.uuid4()` if none was sent."""
    conversation_id = payload.conversation_id or str(uuid.uuid4())
    history = session_store.get_history(payload.tenant_id, conversation_id)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": payload.message}]
    response = chat(messages, temperature=0.4)
    reply = response["message"]["content"].strip()

    session_store.append_turn(payload.tenant_id, conversation_id, payload.message, reply)
    return ChatResponse(conversation_id=conversation_id, reply=reply)


@app.post("/chat/stream", dependencies=[Depends(enforce_rate_limit)])
async def chat_stream_endpoint(payload: ChatRequest) -> StreamingResponse:
    """Async endpoint + StreamingResponse: streams the model's reply as
    Server-Sent Events, piece by piece, instead of waiting for the full
    response (§1). `iterate_in_threadpool` adapts `stream_chat()`'s plain 
    (blocking) generator so pieces are still yielded to the client as they 
    arrive, without blocking the event loop while waiting on each one."""
    messages = [{"role": "user", "content": payload.message}]

    async def event_source():
        async for piece in iterate_in_threadpool(stream_chat(messages)):
            yield f"data: {piece}\n\n"
        yield "event: done\ndata: \n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, tenant_id: str = "default", conversation_id: str | None = None) -> None:
    """WebSocket endpoint (§1/§2): a persistent, two-way connection -
    the client can send several messages over one connection instead of
    a new POST per turn, unlike /chat or /chat/stream."""
    await websocket.accept()
    conversation_id = conversation_id or str(uuid.uuid4())
    await websocket.send_json({"conversation_id": conversation_id})

    try:
        while True:
            user_message = await websocket.receive_text()
            history = session_store.get_history(tenant_id, conversation_id)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": user_message}]
            response = await run_in_threadpool(chat, messages)
            reply = response["message"]["content"].strip()
            session_store.append_turn(tenant_id, conversation_id, user_message, reply)
            await websocket.send_text(reply)
    except WebSocketDisconnect:
        pass


def _generate_report(job_id: str, topic: str) -> None:
    """Runs after the /jobs/report response has already been sent."""
    try:
        response = chat([{"role": "user", "content": f"Write a 3-sentence status report about: {topic}"}], temperature=0.3)
        jobs.set_result(job_id, response["message"]["content"].strip())
    except Exception as exc:  # background job isolation: never let this crash the worker silently
        jobs.set_error(job_id, str(exc))


@app.post("/jobs/report", response_model=JobStatus, dependencies=[Depends(enforce_rate_limit)])
def start_report(topic: str, background_tasks: BackgroundTasks) -> JobStatus:
    """Background tasks and status polling (§1): kick off a slower job
    and return immediately with a job id, instead of making the caller
    wait for the whole thing synchronously."""
    job_id = jobs.create_job()
    background_tasks.add_task(_generate_report, job_id, topic)
    return JobStatus(job_id=job_id, status="pending")


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str) -> JobStatus:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return JobStatus(job_id=job_id, status=job["status"], result=job["result"])


def _process_uploaded_file(job_id: str, temp_path: Path) -> None:
    """Async file processing pipeline (§4): summarize the uploaded
    text, then clean up the temp file no matter what happens
    (temporary file storage and cleanup, §4)."""
    try:
        text = temp_path.read_text(encoding="utf-8", errors="replace")
        response = chat([{"role": "user", "content": f"Summarize this file in 2 sentences:\n\n{text[:4000]}"}], temperature=0.3)
        jobs.set_result(job_id, response["message"]["content"].strip())
    except Exception as exc:
        jobs.set_error(job_id, str(exc))
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/files/upload", response_model=UploadResponse, dependencies=[Depends(enforce_rate_limit)])
async def upload_file(file: UploadFile, background_tasks: BackgroundTasks) -> UploadResponse:
    """Accepting file uploads via multipart form + file type/size
    validation (§4), then handing the actual work off to a background
    task (§1) instead of blocking the upload response on an LLM call."""
    if file.content_type not in ALLOWED_UPLOAD_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(contents)
        temp_path = Path(tmp.name)

    job_id = jobs.create_job()
    background_tasks.add_task(_process_uploaded_file, job_id, temp_path)
    return UploadResponse(job_id=job_id, filename=file.filename)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
