# Module 17 — Capstone Project: Incident Root Cause Analyzer

An agentic system that ingests production logs, metrics, and runbooks to
autonomously diagnose incidents, identify its root cause, and recommend 
remediation steps.

## Tech stack

- **LangGraph** — a ReAct-style agent (`StateGraph`) that decides its own tool calls and loops until it has enough evidence.
- **Gemini (`gemini-3.1-flash-lite`, `gemini-embedding-001`)** — the hosted LLM and embedding model (via `langchain-google-genai` and a direct REST call, respectively); flash-lite for chat/reasoning, embedding-001 for embeddings.
- **Tool Use** — 3 tools (`fetch_metrics`, `fetch_logs` — per service/time-window, `search_runbooks` — RAG lookup), called by the model itself.
- **RAG** — the runbook knowledge base, chunked, embedded, and retrieved via **FAISS** (`IndexFlatIP` similarity search).
- **Langfuse** — traces every tool call and LLM generation per incident, with real token/cost data.
- **FastAPI** — the HTTP layer, with 3 endpoints (health, incident analyze, audit-trace).
- **Streamlit** — interactive UI on top of the FastAPI backend.

## Architecture

```
FastAPI (app.py) endpoints
  GET  /health                --> {"status": "ok"}
  POST /incidents/analyze     --> agent.analyze_incident()
  GET  /incidents/{id}/trace  --> tracer.get_trace()

agent.analyze_incident()  - a LangGraph StateGraph with 3 nodes:
  1. agent    - Gemini, bound to fetch_logs/fetch_metrics/search_runbooks,
                decides for itself which tool(s) to call and when it has
                enough evidence (a ReAct-style loop)
  2. tools    - LangGraph's prebuilt ToolNode; executes whatever tool
                call(s) the agent node just requested
       agent <-> tools loop until the model stops requesting tool calls
  3. extract  - a separate, tool-free call that turns the agent's final
                plain-language summary into a structured JSON RCA report

Tools the agent can call:
  fetch_logs(service, start, end)    -> tools.py   (local .log files)
  fetch_metrics(service, start, end) -> tools.py   (local .json files)
  search_runbooks(query)             -> runbook_rag.py:
                                           embed query (Gemini) -> FAISS
                                           IndexFlatIP top-K search over
                                           data/runbooks/*.md chunks

Tracing (tracer.py, every step of the loop above):
  IncidentTracer -> Langfuse Cloud
    log_generation()  - each real Gemini call, as a "generation" observation
    log_tool()        - each tool call/result, as a "tool" observation
  get_trace_url() / get_trace() - view or pull back the full trace after

Streamlit UI (streamlit_app.py)
  --> calls into the FastAPI app (app.py) above, in-process via TestClient
```

## The scenario: "ShopFast"

A fictional e-commerce system with 3 services - `api-gateway`,
`payments-service`, `postgres-db` - and 2 seeded incidents in
`data/logs/*.log` + `data/metrics/*.json`, each with a real, findable root
cause and a matching entry in `data/runbooks/*.md`:

| Incident | Root cause | Runbook |
|---|---|---|
| Checkout payment failures | `postgres-db` connection pool exhaustion (a leak after a deploy) cascades to `payments-service` timeouts, then `api-gateway` 502s | `db_connection_pool_exhaustion.md` |
| Checkout 500/504 spike | An external payment gateway call from `payments-service` times out repeatedly, exhausting its thread pool and cascading to `api-gateway` | `payment_gateway_timeout_cascade.md` |

One distractor runbook (`disk_space_alert.md`) is also indexed, unrelated
to either seeded incident.

## Files

| File | Role |
|---|---|
| `tools.py` | `fetch_logs`, `fetch_metrics` - read the local log/metric files. |
| `chunking.py` | Recursive character text splitter, used for chunking runbooks. |
| `runbook_rag.py` | Chunk -> embed (Gemini `gemini-embedding-001`, via `common/llm_client.py`'s `embed()`) -> FAISS `IndexFlatIP` index over `data/runbooks/*.md`, plus the `search_runbooks` tool. |
| `tracer.py` | `IncidentTracer` - Langfuse tracing; opens one root trace per incident, logs tool calls and LLM generations, `get_trace()` reads them back for audit. |
| `models.py` | Pydantic request/response schemas, including `RCAReport`. |
| `agent.py` | The LangGraph agent loop (Gemini chat model from `common/llm_client.py`'s `langchain_chat_model()`), prompts, and `analyze_incident()`. |
| `app.py` | FastAPI app: `POST /incidents/analyze`, `GET /incidents/{id}/trace`, `GET /health`. |
| `streamlit_app.py` | Interactive UI, via `TestClient` - pick a seeded incident or enter a custom one, view the RCA report and audit trace in a browser. |
| `data/logs/*.log`, `data/metrics/*.json`, `data/runbooks/*.md` | The synthetic ShopFast scenario. |
| `index_store/` | The runbook RAG index (`chunks.json` + the FAISS index `runbooks.faiss`). |

## Setup

Needs a [Google AI Studio](https://aistudio.google.com) API key for Gemini chat + embeddings:

```bash
cd module_17-CapstoneProject/incident-root-cause-analyzer/app
pip install -r ../requirements.txt
```

1. Generate an API key at aistudio.google.com (**Get API key**).
2. Edit `common/.env` (the shared folder at the repo root, used by every module) and fill in `GEMINI_API_KEY`.

Also needs a [Langfuse Cloud](https://cloud.langfuse.com) account:

1. Sign up at cloud.langfuse.com and create a project.
2. In the project's **Settings → API Keys**, generate a keypair.
3. Edit `common/.env` and fill in
   `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST`.
4. Verify the credentials: `python tracer.py` — prints `Langfuse credentials OK` on success.

Both sets of credentials live in the same `common/.env` file: the shared `common/llm_client.py` loads it for every Gemini call, and `tracer.py` loads it for Langfuse.

## Running

The interactive Streamlit UI (drives the FastAPI app via an in-process
`TestClient` - no separate `uvicorn` process needed):

```bash
streamlit run streamlit_app.py
```
(or `python -m streamlit run streamlit_app.py` if `streamlit` isn't on `PATH`.)

Opens at `http://localhost:8501`. Pick one of the 2 seeded incidents (or
enter a custom alert/services/time-window), click **Analyze incident**,
and see the structured RCA report (summary, timeline, root cause,
evidence, recommended fix, confidence) plus an expandable step-by-step
audit trace pulled from Langfuse. Past analyses from the session stay
selectable in a dropdown.

Or run the FastAPI app as a real server, with no UI:

```bash
uvicorn app:app --reload
```
(or `python -m uvicorn app:app --reload` if `uvicorn` isn't on `PATH`.)

Then open `http://127.0.0.1:8000/docs` for an interactive Swagger UI, or
`POST` to `/incidents/analyze` with a body like:

```json
{
  "alert": "Elevated 502 errors and payment failures on /checkout",
  "services": ["api-gateway", "payments-service"],
  "start_time": "2026-08-30T09:10:00Z",
  "end_time": "2026-08-30T09:30:00Z"
}
```

The response includes a `trace_url` linking directly into the Langfuse UI.
