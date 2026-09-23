# Module 17 — Capstone Project: Architecture and Concepts

The Incident Root Cause Analyzer
(`incident-root-cause-analyzer/`), described in two parts:

- **Part 1 — Architecture:** a brief high-level view (1.1), then a
  detailed breakdown of every component, the request flow, and the
  design decisions behind them (1.2).
- **Part 2 — Core GenAI concepts, with examples from this project:**
  fifteen foundational concepts, each explained in theory and then
  pointed at the exact piece of this project's code that implements it.

For setup and run instructions, see `incident-root-cause-analyzer/README.md`.
File paths below are relative to the repo root unless they start with
`app/`, which means `incident-root-cause-analyzer/app/`.

---

# Part 1 — Architecture

## 1.1 High-level architecture

**What it is:** an agentic system that investigates a production
incident the way an SRE would — given an alert, it pulls logs and
metrics, reasons about which service actually failed first, checks the
symptom against a knowledge base of known issues, and returns a
structured root-cause report with a recommended fix. Every step is
traced for audit.

```
 User (Streamlit UI or raw HTTP)
        |
        v
 FastAPI backend  --------------------------------------------+
        |                                                     |
        v                                                     |
 LangGraph agent (Gemini, via common/llm_client.py)           |
   decides its own tool calls, loops until it has evidence    |
        |                                                     |
        +--> log/metric fetch tools (local data)              |
        +--> runbook search tool --> RAG over a FAISS index   |
        |                                                     |
        v                                                     |
 Structured RCA report  <-------------------------------------+
        |
        v
 Langfuse (every step traced)
```

**Core pieces:**
- **LangGraph** — the agent's control flow: a model-driven tool-calling
  loop, not a scripted sequence.
- **Gemini** — the model behind that loop (`gemini-3.1-flash-lite` for
  reasoning, `gemini-embedding-001` for embeddings), reached through the
  repo's shared client, `common/llm_client.py`, the same one every other
  model-calling module in the repo uses.
- **Tool use** — 3 tools the agent calls itself: `fetch_logs`,
  `fetch_metrics`, `search_runbooks`.
- **RAG** — the runbook knowledge base, retrieved via FAISS.
- **FastAPI** — the service boundary (3 endpoints).
- **Streamlit** — the interactive front end.
- **Langfuse** — full observability into what the agent actually did.

## 1.2 Detailed architecture

### 1.2.1 Component breakdown

| Component | File | Role |
|---|---|---|
| Shared Gemini client | `common/llm_client.py` | `langchain_chat_model()` builds the agent's LangChain chat models; `embed()` produces the RAG embeddings. Loads `GEMINI_API_KEY` from `common/.env`. |
| HTTP layer | `app/app.py` | 3 routes (`GET /health`, `POST /incidents/analyze`, `GET /incidents/{id}/trace`) + a catch-all exception handler. |
| Agent | `app/agent.py` | The LangGraph state machine, prompts, retry logic, and `analyze_incident()` — the one entry point the API calls. |
| Tools | `app/tools.py` | `fetch_logs`/`fetch_metrics` — read local `.log`/`.json` fixture files for one service + time window. |
| RAG | `app/runbook_rag.py`, `app/chunking.py` | Chunk the runbooks → embed via the shared client → index with FAISS → `search_runbooks` tool. |
| Tracing | `app/tracer.py` | `IncidentTracer` — opens one Langfuse trace per incident and logs each step as a typed observation. Loads the `LANGFUSE_*` keys from `common/.env`. |
| Schemas | `app/models.py` | Pydantic request/response models (`IncidentRequest`, `RCAReport`, `AnalyzeResponse`). |
| UI | `app/streamlit_app.py` | The interactive front end, driving `app.py` in-process. |
| Data | `app/data/`, `app/index_store/` | Log/metric fixtures and runbooks for 3 services; the persisted FAISS index + chunk metadata. |

**Configuration.** All credentials live in one file, `common/.env`,
shared with every other module in the repo: `GEMINI_API_KEY`,
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.
`agent.py` and `runbook_rag.py` reach `common/` via
`sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))`
(`app/` → `incident-root-cause-analyzer/` → `module_17-CapstoneProject/`
→ repo root), and importing `llm_client` loads the key file.
`tracer.py` loads `common/.env` itself with `python-dotenv`, so
`python tracer.py` works standalone as a credentials check.

### 1.2.2 The agent loop, in detail

`agent.py` builds a `StateGraph(MessagesState)` with three nodes:

1. **`agent`** — Gemini (`gemini-3.1-flash-lite`), created by
   `langchain_chat_model(temperature=0.2)` — a `ChatGoogleGenerativeAI`
   preconfigured with the shared model name and key — and bound to all
   three tools via `bind_tools()`. Each turn it either requests one or
   more tool calls or, once it judges it has enough evidence, replies
   with a plain-language investigation summary.
2. **`tools`** — LangGraph's prebuilt `ToolNode`. Executes whatever the
   `agent` node just requested and appends the result(s) as
   `ToolMessage`s.
3. **`extract`** — a second, tool-free Gemini call
   (`langchain_chat_model(temperature=0.0, response_mime_type="application/json")`)
   that turns the `agent` node's final summary into the structured
   `RCAReport` shape.

A conditional edge (`_should_continue`) routes `agent → tools` whenever
the latest message carries `tool_calls`, and `agent → extract` once it
doesn't; `tools → agent` always loops back; `extract → END`. A run is
bounded by `recursion_limit=25` as a safety cap; the seeded incidents
typically finish in about 10-12 traced steps.

**Reliability details:**
- `_invoke_with_retry()` wraps every chat call with exponential backoff
  plus jitter (3 retries, `2.0 * 2**attempt` seconds + up to 1s) on
  `GoogleAPIError` (transient server errors, e.g. 503 "high demand") or
  `GoogleRateLimitError` (free-tier 429s).
- Embedding calls get the shared client's own retry: up to 5 retries on
  429/500/503 responses and request timeouts, waiting the `retryDelay`
  Gemini returns in a 429 response when present (it's a per-minute
  quota), exponential backoff otherwise. The client sends the key in the `x-goog-api-key`
  header, never in the URL, so it can't leak into error messages.
- The system prompt explicitly tells the model to give up and report
  "no evidence found" (low confidence) after checking 2-3 services with
  nothing found, rather than searching indefinitely.
- As a backstop, if LangGraph's `GraphRecursionError` still fires
  (evidence never converges), `analyze_incident()` catches it and
  returns a generic low-confidence `_INCONCLUSIVE_REPORT` instead of
  letting the exception (and a raw traceback) reach the caller.

### 1.2.3 Tools

| Tool | Signature | Behavior |
|---|---|---|
| `fetch_logs` | `(service, start_time, end_time)` | Reads `data/logs/{service}.log`, returns matching lines within the window. |
| `fetch_metrics` | `(service, start_time, end_time)` | Reads `data/metrics/{service}.json`, returns matching samples within the window. |
| `search_runbooks` | `(query)` | Embeds the query and returns the top-2 most similar runbook chunks. |

Known services: `api-gateway`, `payments-service`, `postgres-db`
(`KNOWN_SERVICES` in `tools.py`). There is deliberately no dedicated
cross-service "correlate" tool — the agent does that itself by calling
`fetch_logs`/`fetch_metrics` for more than one service and comparing
timestamps to find which one failed first.

The three tools' docstrings are not just developer comments —
LangChain's `@tool` decorator reads each function's real `__doc__` and
sends it to Gemini as that tool's schema description, which is exactly
what the model reads to decide which tool to call and when.

### 1.2.4 RAG pipeline

`runbook_rag.py`:
1. **Chunk** — `chunking.py`'s recursive character splitter
   (`CHUNK_SIZE=400`, `CHUNK_OVERLAP=60`) breaks each runbook markdown
   file into overlapping pieces, atomizing on paragraph/line/sentence/
   space boundaries and merging small pieces back up to the size limit.
2. **Embed** — each chunk is embedded by the shared client's `embed()`
   (Gemini `gemini-embedding-001` via `batchEmbedContents`, truncated to
   `EMBED_DIM=768`, `task_type="RETRIEVAL_DOCUMENT"`). Truncated Gemini
   embeddings aren't unit-length (a measured 768-dim vector had norm
   ~0.58), so `embed()` L2-normalizes every vector.
3. **Index** — vectors go into a FAISS `IndexFlatIP` (exact
   inner-product search, equivalent to cosine similarity for normalized
   vectors). FAISS is a similarity-search *library*, not a database: it
   has no store for chunk text or metadata, so `index_store/chunks.json`
   carries that alongside the index file (`index_store/runbooks.faiss`),
   with FAISS's row indices looked up against that same-order list. The
   index is built once and reused (`build_index()` is idempotent unless
   `force=True`).
4. **Retrieve** — `top_k_retrieve()` embeds the query with Gemini's
   `RETRIEVAL_QUERY` task type (the asymmetric counterpart of
   `RETRIEVAL_DOCUMENT`) and returns the top-`TOP_K=2` matches by score.

Knowledge base: 3 runbooks — `db_connection_pool_exhaustion.md`,
`payment_gateway_timeout_cascade.md`, and one unrelated distractor,
`disk_space_alert.md` — chunked into 15 pieces (5 + 6 + 4).

### 1.2.5 Tracing

`tracer.py`'s `IncidentTracer`, one instance per incident:
- Opens a root Langfuse observation (`as_type="chain"`) on construction,
  with a `trace_id` derived deterministically from `incident_id`
  (`create_trace_id(seed=incident_id)`), so the same incident always
  maps back to the same trace with no local state needed to look it up.
- `log_tool()` logs each tool call and result as a `"tool"` observation.
- `log_generation()` logs each real Gemini call as a `"generation"`
  observation, with the model name and real token counts from the
  response's `usage_metadata` — using this type rather than a generic
  `"span"` is what makes Langfuse's per-model token/cost dashboards
  populate.
- `agent.py`'s `_log_transcript()` walks the finished LangGraph message
  list once after the run and logs each step (model turns as
  `agent_turn`, the final extraction as `rca_extraction`), since the
  tool calls are decided by the model inside the loop rather than
  called explicitly in code.
- `finish()` closes the root observation and flushes to Langfuse;
  `get_trace_url()` and the module-level `get_trace()` let a caller view
  or pull back the full trace afterwards.

### 1.2.6 FastAPI layer

`app.py` exposes:
- **`GET /health`** — a deliberately shallow liveness check
  (`{"status": "ok"}`, no downstream calls). A `200` proves the process
  started and every import succeeded; it does not prove Gemini or
  Langfuse are reachable — that's a separate, on-demand check
  (`python tracer.py`).
- **`POST /incidents/analyze`** — generates an `incident_id`
  (`uuid.uuid4()`), calls `analyze_incident()`, validates the result as
  an `RCAReport`, and returns it with the step count and a Langfuse
  trace URL.
- **`GET /incidents/{incident_id}/trace`** — reads back the audit trail
  via `tracer.get_trace()`, `404` for an unknown (or not-yet-ingested)
  id.

All three routes are plain `def`, not `async def` — FastAPI runs plain
`def` routes on a thread pool, so `analyze()`'s sequence of *blocking*
Gemini calls (often 30-60+ seconds) doesn't freeze the event loop for
other concurrent requests.

A module-level `@app.exception_handler(Exception)` catches any
unexpected error (beyond the already-handled `GraphRecursionError`
case) and returns a clean 500 JSON response instead of a raw traceback.

### 1.2.7 Streamlit UI

`streamlit_app.py` drives `app.py` **in-process** via
`fastapi.testclient.TestClient(app, raise_server_exceptions=False)` — no
separate `uvicorn` process; one `streamlit run` command runs the UI and
the backend in the same Python process.

Flow: pick one of 2 seeded incidents (or fill in a custom alert,
services, and time window) → **Analyze incident** → results panel
(confidence badge, summary, timeline, root cause, evidence, recommended
fix, Langfuse trace link, raw JSON) → an expandable step-by-step audit
trace. A sidebar shows API health and the past analyses from the
current session.

One timing issue is handled explicitly: Langfuse's trace-*read* API lags
its own ingestion after an analysis completes (about 8s in testing; the
code allows for up to ~25s), so the trace expander retries up to 6 times, 4s apart,
instead of showing an error on the first 404.

### 1.2.8 End-to-end request flow

1. A user submits an alert, the services it mentions, and a time window
   (via the UI or a raw `POST`).
2. `app.py` generates an `incident_id` and calls
   `agent.analyze_incident(...)`, which opens a Langfuse trace
   (`IncidentTracer`) for it.
3. The `agent` node investigates: calls `fetch_logs`/`fetch_metrics` for
   the mentioned service(s) first and, per its system prompt, checks
   other known services too if the evidence suggests the real failure
   started upstream of the one that raised the alert.
4. Once it spots a clear symptom pattern, it calls `search_runbooks`,
   which embeds the query and retrieves the most similar runbook chunks.
5. Once satisfied, the agent stops calling tools and writes a
   plain-language investigation summary.
6. The `extract` node turns that summary into the structured JSON
   report.
7. `_log_transcript()` walks the full message transcript once and logs
   every model turn and tool call to Langfuse; `finish()` closes the
   trace.
8. `app.py` validates the report as an `RCAReport` and returns it with
   the step count and a direct Langfuse trace URL.
9. The caller can separately fetch the full step-by-step trace via
   `GET /incidents/{id}/trace` for post-incident audit.

### 1.2.9 Scenario data

A fictional e-commerce system ("ShopFast") with 3 services
(`api-gateway`, `payments-service`, `postgres-db`) and 2 seeded
incidents, each with a real, findable root cause and a matching runbook:

| Incident | Root cause | Runbook |
|---|---|---|
| Elevated 502 errors and payment failures on /checkout (2026-08-30, 09:10-09:30 UTC) | `postgres-db` connection pool exhaustion cascades to `payments-service` timeouts, then `api-gateway` 502s | `db_connection_pool_exhaustion.md` |
| Spike in 500/504 errors on /checkout (2026-09-01, 03:45-04:10 UTC) | An external payment gateway call from `payments-service` times out repeatedly, exhausting its thread pool | `payment_gateway_timeout_cascade.md` |

In each case the service that raises the alert is a downstream symptom,
not the root cause, so the agent has to look past the alerting service
to get the right answer. Both incidents reach the correct root cause at
`high` confidence.

---

# Part 2 — Core GenAI concepts, with examples from this project

Each concept below is covered in two parts: a theoretical explanation,
then how this project applies it. Sections 2.1-2.13 are implemented
here; 2.14 and 2.15 are roadmap topics this project does not cover
(their hands-on versions live in Modules 16 and 14).

## 2.1 Generative AI

**Theory.** Generative AI refers to models that produce new content —
text, images, audio, code — by learning the statistical patterns of
their training data and sampling from a learned probability
distribution, rather than retrieving a stored answer or classifying
input into a fixed set of labels. For text, this means predicting a
plausible continuation one token at a time; the same input can produce
different, equally valid outputs on different runs.

**In this project.** The whole investigation report is generated, not
templated or looked up. Given the same raw log lines and metric
samples, the model (`gemini-3.1-flash-lite`) composes a new
plain-language investigation summary — what happened, in what order,
across which services — and a second generation call turns that
narrative into a structured JSON report (`summary`, `timeline`,
`root_cause`, `evidence`, `recommended_fix`, `confidence`). No part of
the report text is a fill-in-the-blank template; it's produced fresh
from whatever evidence the tool calls surfaced for that incident.

## 2.2 LLMs (Large Language Models)

**Theory.** LLMs are transformer-based neural networks trained on vast
text corpora to predict the next token given prior context. At
sufficient scale they show capabilities beyond raw next-token
prediction — following natural-language instructions, multi-step
reasoning, and deciding when to invoke external tools. Behavior is tuned
per call with parameters like `temperature` (higher = more varied
sampling, lower = more deterministic).

**In this project.** `agent.py` uses one model, Gemini, in two roles
with two parameter settings, both built by the shared client's
`langchain_chat_model()`:
- `_agent_llm = langchain_chat_model(temperature=0.2).bind_tools(TOOLS)`
  — a low-but-nonzero temperature for the investigative reasoning step,
  where deciding "which service to check next" benefits from a little
  flexibility.
- `_extractor_llm = langchain_chat_model(temperature=0.0, response_mime_type="application/json")`
  — temperature 0 for turning the finished summary into strict JSON,
  where the same input should reliably produce the same structured
  shape.

The same underlying model plays two different roles purely through
prompt and parameter choices.

## 2.3 RAG (Retrieval-Augmented Generation)

**Theory.** RAG grounds an LLM's output in an external knowledge source
retrieved at query time, instead of relying only on what the model
memorized during training. The standard pipeline: chunk documents into
passages, embed each chunk into a vector, store the vectors, embed an
incoming query the same way, retrieve the most similar chunks
(nearest-neighbor search), and inject that text into the prompt before
generation ("augmentation"). This reduces hallucination and lets a model
reason over private or current data it was never trained on.

**In this project.** `runbook_rag.py` implements the full pipeline over
the runbook knowledge base (`app/data/runbooks/*.md`):
- **Chunk** — `chunking.py`'s recursive splitter, 400-character chunks
  with 60-character overlap.
- **Embed** — `gemini-embedding-001` via the shared client's `embed()`,
  with `task_type="RETRIEVAL_DOCUMENT"` for indexed chunks and
  `"RETRIEVAL_QUERY"` for a search query (Gemini's asymmetric-retrieval
  hint). Every vector is L2-normalized, which the next step relies on.
- **Store** — FAISS `IndexFlatIP`; because every vector is normalized,
  inner product equals cosine similarity. `chunks.json` carries the
  chunk text and source alongside the index.
- **Retrieve** — `top_k_retrieve()` / the `search_runbooks` tool, top-2
  chunks per query.
- **Augment + generate** — the agent calls `search_runbooks` as a tool
  once it has spotted a symptom pattern; the retrieved runbook text
  comes back as a tool result inside the same conversation, and the
  model's final summary cites that runbook's recommended fix.

## 2.4 Prompt Engineering

**Theory.** Prompt engineering is the deliberate design of the text sent
to a model to shape its behavior — assigning a role, spelling out
reasoning steps or constraints, defining an exact output format, and
giving explicit stop conditions — without changing the model itself.
Small wording changes can materially change reliability, especially for
structured output or bounded tasks.

**In this project.** Two purpose-built prompts in `agent.py`:
- `SYSTEM_PROMPT` shapes the agent's investigation: it assigns a role
  ("You are an SRE incident investigator"), names its exact tools and
  the known services, gives a domain-specific reasoning instruction
  ("the true root cause is often an *earlier* failure in a different,
  upstream service"), and — critically — an explicit stopping
  condition: after checking at most 2-3 services with no evidence, stop
  and report low confidence rather than searching indefinitely.
- `EXTRACTION_PROMPT` is built for structured-output reliability: it
  repeats the exact JSON schema inline, says "Respond with ONLY valid
  JSON," and adds a narrow but important instruction — "Copy every
  timestamp exactly as it appears... do not alter the year, date, or any
  digit" — because models otherwise tend to "normalize" timestamps when
  reformatting them into JSON.

## 2.5 Orchestration

**Theory.** Orchestration is the code that wires an LLM's calls together
with tools, state, and control flow into a working system — deciding
what gets called, in what order, with what data passed between steps.
It can be a fixed chain, a graph with branches and loops, or hand-written
control flow. Frameworks like LangChain/LangGraph provide reusable
primitives for this.

**In this project.** LangGraph's `StateGraph` (`build_agent()` in
`agent.py`) is the orchestration layer: an `agent` node, a `tools` node
(the prebuilt `ToolNode`), and an `extract` node, joined by one
conditional edge (`_should_continue` — to `tools` if the model requested
a tool call, otherwise to `extract`) and fixed edges (`tools → agent`,
`extract → END`). `recursion_limit=25` on `invoke(...)` is itself an
orchestration control — a structural cap on loop iterations. One level
up, FastAPI (`app.py`) is the outer orchestration layer, exposing
`analyze_incident()` as `POST /incidents/analyze` so a client (or the
Streamlit UI) can use it without knowing anything about the graph
underneath.

## 2.6 Agentic Workflows

**Theory.** An agentic workflow is what emerges when the *model itself*
decides which actions to take and when to stop, rather than code
dictating a fixed sequence of steps. The defining loop is think → act
(call a tool) → observe the result → decide again, repeated until the
model judges it has enough information — in contrast to a "chain,"
where the sequence and step count are fixed regardless of input.

**In this project.** The `agent` node is bound to three tools
(`fetch_logs`, `fetch_metrics`, `search_runbooks`) and is free to call
any of them, any number of times, in any order, based on what it sees.
For a given alert it decides which service to check first, whether the
evidence justifies checking a different, upstream service next (per
`SYSTEM_PROMPT`'s guidance to look past the alerting service), and when
the evidence is enough to stop and write a summary. The step count
varies per incident (e.g. 10 steps for the connection-pool incident, 12
for the gateway-timeout one), unlike a fixed pipeline. A fixed
`gather → search → synthesize` pipeline would have to fetch every
service's data every time; the agentic loop lets the model follow the
evidence instead.

## 2.7 Observability

**Theory.** Observability means instrumenting a system so its internal
behavior — latency, cost, token usage, which decisions were made, where
it failed — is visible after the fact, via tracing, structured logging,
and metrics/dashboards. This matters more for LLM applications than for
typical software because outputs are non-deterministic, and a model's
exact behavior for an input usually can't be predicted by reading the
code; you need to see what actually happened on a specific run.

**In this project.** `tracer.py` sends real traces to hosted Langfuse
(where Module 13 hand-builds a local tracer instead). `IncidentTracer`
opens one root trace per incident, with a `trace_id` derived
deterministically from the `incident_id` so it can be looked up later
with no local state. Two distinct observation types are logged, not one
generic "span" for everything:
- `log_tool()` for each tool call (`fetch_logs`/`fetch_metrics`/
  `search_runbooks`).
- `log_generation()` for each real Gemini call, carrying the model name
  and real token counts from the response's `usage_metadata` — this is
  what makes Langfuse's per-model token/cost dashboards populate.

`GET /incidents/{id}/trace` exposes this as a post-incident audit
endpoint, replaying the full step-by-step trace (tool calls, model
turns, final report) for any past analysis.

## 2.8 Guardrails

**Theory.** Guardrails are the safety and reliability mechanisms placed
around an LLM's input and output — validating structure, capping
runaway behavior, providing a safe fallback instead of crashing or
looping indefinitely, and handling upstream failures gracefully. (Module
15 builds a dedicated set — prompt-injection detection, PII redaction,
content moderation — as a standalone topic; this project applies several
of the same underlying patterns where the agent needs them.)

**In this project**, concretely:
- **A bounded stopping condition in the prompt** — `SYSTEM_PROMPT` tells
  the model to stop after checking at most 2-3 services with no
  evidence, rather than searching indefinitely.
- **A hard structural backstop behind it** — `recursion_limit=25` on the
  graph. If the prompt-level limit is ignored, LangGraph raises
  `GraphRecursionError` instead of looping forever, and
  `analyze_incident()` returns a graceful `_INCONCLUSIVE_REPORT`
  (`confidence: "low"`, an honest "could not reach a conclusion"
  message) instead of crashing or returning nothing.
- **Structured-output enforcement** — `EXTRACTION_PROMPT`'s strict
  JSON-only schema plus `response_mime_type="application/json"` forces
  the extraction call to return parseable JSON.
- **Retry-with-backoff around a flaky upstream** — transient Gemini
  failures (503s, 429s, timeouts) are retried instead of failing the
  whole analysis (see 2.13).
- **A catch-all failure boundary at the API layer** — `app.py`'s
  `@app.exception_handler(Exception)` turns any other unexpected error
  into a clean `500` JSON response instead of a raw traceback reaching
  the caller.
- **No secrets in error output** — the shared client sends the API key
  in a request header rather than the URL, so HTTP errors (which quote
  the URL) never contain it.

## 2.9 Tool Use / Function Calling

**Theory.** Function calling (tool use) lets a model do more than
generate text: the caller describes a set of functions (name,
parameters, a natural-language description), the model can respond with
a structured request to invoke one instead of plain text, the caller
executes that function in real code, and the result is fed back into the
conversation for the model to continue reasoning with. This is the
plumbing underneath any agentic behavior — without it a model can only
talk about taking an action, never take one.

**In this project.** `fetch_logs` and `fetch_metrics` (`tools.py`) and
`search_runbooks` (`runbook_rag.py`) are decorated with LangChain's
`@tool`, each with a docstring the model reads as that tool's schema
description (e.g. `fetch_logs`'s docstring spells out valid `service`
values and the expected timestamp format — that description, not the
code, is what the model sees).
`_agent_llm = langchain_chat_model(...).bind_tools(TOOLS)` registers
them with the model; when the model responds with `tool_calls`,
LangGraph's `ToolNode` executes the matching Python function and wraps
its return value in a `ToolMessage`, which flows back into
`state["messages"]` for the `agent` node's next turn. The model never
touches a file directly — it only sees text descriptions in and text
results back.

## 2.10 Structured Output Generation

**Theory.** Left alone, an LLM produces free-form prose — easy for a
person to read, unreliable for a program to parse. Structured output
generation constrains a model to emit data in an exact, predictable
shape (usually JSON matching a fixed schema), via a strict prompt
describing that schema, a provider's dedicated "JSON mode," or both.
This turns model output into data the rest of the system can consume
without a hand-rolled text-parsing step.

**In this project**, this is a separate model call from the
investigation itself — `_extract_node` in `agent.py` takes the finished
plain-language summary and runs it through `_extractor_llm`, which uses
`response_mime_type="application/json"` (Gemini's JSON mode, an
API-level constraint, not just a prompt request) on top of
`EXTRACTION_PROMPT` spelling out the exact schema inline. The result is
parsed with `json.loads(...)` in `analyze_incident()` and validated
again by Pydantic's `RCAReport` model in `app.py` — two independent
layers (the model's JSON mode, then schema validation) rather than
trusting either alone.

## 2.11 Vector Search / Embeddings

**Theory.** An embedding is a numeric vector representation of a piece
of text such that semantically similar text ends up close together in
vector space — this is what makes "search by meaning" possible instead
of exact keyword match. Vector search is the nearest-neighbor lookup
over stored embeddings; a dedicated vector index (FAISS, a vector
database, etc.) makes that lookup fast at scale. It's the retrieval half
of RAG, and also a reusable capability on its own (semantic search,
deduplication, clustering).

**In this project**, `runbook_rag.py` embeds through the shared client's
`embed()`, which calls `gemini-embedding-001` (`batchEmbedContents`,
768 dimensions) and L2-normalizes every returned vector, since truncated
Gemini embeddings aren't unit-length on their own. Vectors are indexed
with FAISS's `IndexFlatIP` (`build_index()`), which computes an exact
inner product between the query vector and every stored vector —
equivalent to cosine similarity because both sides are normalized.
`top_k_retrieve()` is the nearest-neighbor search: embed the query with
`task_type="RETRIEVAL_QUERY"`, call `index.search(...)`, and return the
top-2 chunks by score. FAISS stores only vectors and row positions —
`chunks.json` is the parallel metadata store that turns a row index back
into readable chunk text and its source runbook.

## 2.12 Building Production APIs

**Theory.** Turning a working script into a usable service means
wrapping it in a real API layer: typed request/response contracts,
input validation that rejects malformed requests before they reach
business logic, a health-check endpoint for uptime monitoring, and
centralized error handling so failures come back as clean, well-formed
responses instead of raw stack traces.

**In this project**, `app.py` and `models.py` are that layer, built on
FastAPI. `IncidentRequest`/`RCAReport`/`AnalyzeResponse` (`models.py`)
are Pydantic schemas — FastAPI validates every `POST
/incidents/analyze` body against `IncidentRequest` automatically and
uses `AnalyzeResponse` to document the response shape, so a malformed
request (missing `alert`, wrong field type) is rejected before
`analyze_incident()` runs. `GET /health` is a plain, synchronous
liveness check. `POST /incidents/analyze` is a sync `def` route — it
wraps a blocking chain of Gemini calls, and FastAPI runs sync routes in
a thread pool, so a slow investigation doesn't block the server's event
loop. `GET /incidents/{id}/trace` returns a `404` via `HTTPException`
for an unknown id, distinct from a `500` for a real server error. And
`@app.exception_handler(Exception)` is the centralized error boundary —
any unhandled exception becomes one consistent JSON error shape.

## 2.13 Reliability Patterns

**Theory.** Reliability patterns handle failures that are transient and
*not* the caller's fault — a flaky network, a rate limit, a temporarily
overloaded upstream — where the right response is to try again (with
backoff, so retries don't pile onto a struggling service) rather than
fail immediately. This is a different concern from Guardrails (which
handles misuse or unsafe input/output): reliability patterns apply even
to a perfectly valid request, because distributed systems fail
intermittently.

**In this project**, three layers handle three kinds of failure:
- **Chat calls** — `_invoke_with_retry()` in `agent.py` wraps every
  Gemini chat call and catches `GoogleAPIError` (e.g. a 503 "high
  demand") and `GoogleRateLimitError` (a 429), retrying 3 times with
  exponential backoff plus jitter (`base_delay * 2**attempt + random
  jitter`).
- **Embedding calls** — the shared `common/llm_client.py` retries 429,
  500, and 503 responses and request timeouts up to 5 times. For a 429 it
  waits the `retryDelay` Gemini returns (a per-minute quota needs a wait
  of up to a minute, not a couple of seconds); otherwise it uses
  exponential backoff.
- **Non-convergence** — `GraphRecursionError` handling in
  `analyze_incident()` (see 2.8) deals with a *structural* failure (the
  agent didn't converge) by degrading gracefully instead of retrying.

Transient upstream failures get a retry; a structural failure gets a
graceful fallback — different failure modes, deliberately different
responses.

## 2.14 Evaluation

**Theory.** LLM evaluation measures whether a model-backed system's
outputs are actually good — correct, relevant, grounded in the evidence,
and well-formed — and whether that quality holds as prompts, models, or
code change. Common approaches: reference-based metrics against a
golden dataset (exact match, ROUGE, embedding similarity), LLM-as-a-judge
scoring, hallucination checks (e.g. sampling consistency), human review,
and regression gates that fail a build when scores drop below a
threshold.

**In this project.**
- Not covered in this project — there is no automated evaluation
  (no golden dataset of incidents with expected root causes, no scoring,
  no regression gate). Module 16 builds these techniques hands-on.

## 2.15 Caching and Cost Optimization

**Theory.** LLM calls cost money and time, so production systems avoid
repeating them and match each request to the cheapest model that can
handle it. Common approaches: exact-match caching (a hash of the prompt
→ the stored response), semantic caching (reuse the answer to a
sufficiently similar earlier query, via embeddings), prompt compression
(send fewer tokens), and model routing (send simple requests to a cheap
model and escalate only when needed).

**In this project.**
- Not covered in this project — every analysis calls Gemini fresh, with
  no response cache, no prompt compression, and no model routing (the
  one exception to "fresh" is the runbook index, which is embedded once
  and reused). Module 14 builds these techniques hands-on.
