# Module 12 — Building Production APIs

Concept overview — brief explanations of every sub-topic in
Module 12 — Building Production APIs.

## 1. API Design for GenAI

- **Sync vs async endpoints** — a sync endpoint blocks the worker
  handling it until the work finishes; an async endpoint can yield
  control while waiting on I/O (like a slow LLM call), letting the
  same process serve other requests in the meantime. Frameworks like
  FastAPI can run a sync route in a thread pool automatically, so a
  blocking call doesn't have to freeze the whole event loop either way.
- **Streaming endpoints** — instead of waiting for a full LLM response
  before replying, the endpoint sends pieces of the answer to the
  client as they're generated — the same incremental output Module 2
  §2 covers, exposed over HTTP instead of just printed to a terminal.
- **WebSocket endpoints** — a persistent, two-way connection the client
  and server can both send messages over at any time, instead of one
  request producing one response — useful for a back-and-forth chat
  session without reconnecting for every turn.
- **Background tasks and status polling** — for work that takes too
  long to hold a request open for, the endpoint kicks off the work,
  returns immediately with an id, and the client separately polls a
  status endpoint (or gets notified) once it's done.
- **REST vs WebSocket vs SSE** — three ways to serve a GenAI response:
  plain REST (one request, one full response), Server-Sent Events /
  SSE (one request, a stream of response pieces over a single
  long-lived HTTP connection), or WebSocket (a persistent two-way
  connection for multiple exchanges). SSE fits one-shot streaming
  replies; WebSocket fits multi-turn, bidirectional interactions.

## 2. FastAPI for GenAI Backends

- **Project structure setup** — organizing routes, models, and shared
  dependencies into separate modules as an API grows, rather than one
  large file — this module's app splits `models.py` (schemas),
  `sessions.py`/`jobs.py` (state), `deps.py` (shared dependencies), and
  `app.py` (routes) along those lines.
- **Async route handlers** — FastAPI route functions declared
  `async def`, allowing them to `await` I/O (an LLM call, a database
  query) without blocking other requests being handled concurrently.
- **Pydantic request/response models** — declaring the exact shape of
  a request body or response as a `BaseModel` subclass; FastAPI
  validates incoming data against it automatically and rejects
  malformed requests before your route code even runs.
- **Dependency injection** — FastAPI's `Depends()` mechanism: shared
  logic (auth checks, rate limiting, a database session) is written
  once as a function and declared as a dependency on any route that
  needs it, instead of repeating the same checks inside every handler.
- **StreamingResponse** — FastAPI's response class for returning a
  generator/iterator instead of a fixed body, used to implement
  streaming (§1) and SSE endpoints.
- **WebSocket endpoint** — FastAPI's `@app.websocket(...)` decorator
  and `WebSocket` object (`accept()`, `receive_text()`, `send_text()`),
  the concrete implementation of a WebSocket endpoint (§1).
- **BackgroundTasks** — FastAPI's `BackgroundTasks` dependency: attach
  a function to run *after* the response has already been sent, the
  concrete mechanism behind background tasks and status polling (§1).
- **CORS, auth, rate limiting middleware** — cross-cutting concerns
  applied to every request: CORS middleware controls which browser
  origins may call the API; auth (often implemented as a dependency
  rather than true middleware in FastAPI) verifies who's calling; rate
  limiting caps how often a given caller may call in a time window.
- **Exception handlers for LLM errors** — a custom
  `@app.exception_handler(...)` that catches a specific exception type
  (e.g. the LLM backend being unreachable) and returns a clean,
  structured error response instead of letting it surface as a raw,
  unhandled 500.

## 3. Session and State Management

- **Stateless API design** — the server itself holds no client-specific
  state tied to a particular instance; instead, the client passes
  identifiers (like a conversation id) with every request, and any
  server instance can look up the associated state from a shared
  store — what makes an API horizontally scalable across multiple
  server processes/machines.
- **Redis for server-side session storage** — a common production
  choice for that shared store: fast, key-value, and external to any
  one API process, so all instances of the API see the same session
  state.
- **Conversation ID with UUID** — a unique identifier (typically a
  UUID) generated for a conversation, returned to the client, and sent
  back on every subsequent request in that conversation so the server
  can look up the right history.
- **Multi-tenant session isolation** — when one API serves multiple
  separate customers/tenants, session state must be scoped per tenant
  (not just per conversation id) so one tenant can never see another's
  data, even by an id collision.
- **Session expiry and cleanup** — old, inactive sessions need to be
  removed from the store after some period, both to free memory/storage
  and because very old context is often no longer relevant to inject
  into a new request.

## 4. File Handling

- **Accepting file uploads via multipart form** — the standard HTTP
  mechanism (`multipart/form-data`) for a client to send a binary file
  alongside a request, which the framework parses into an uploaded-file
  object rather than raw bytes in a JSON body.
- **File type and size validation** — checking an uploaded file's
  declared content type/extension and byte size against allowed limits
  *before* doing anything expensive with it, rejecting anything that
  doesn't qualify.
- **Async file processing pipelines** — once a file is accepted, doing
  the actual processing (parsing, chunking, embedding, summarizing) as
  a background task rather than making the upload request wait for all
  of it to finish synchronously.
- **Temporary file storage and cleanup** — writing an uploaded file to
  a temporary location for processing, then explicitly deleting it once
  processing is done (or fails) so temp storage doesn't grow
  unboundedly.

---

## 5. Coverage: roadmap sub-topic → implementation

Unlike earlier modules (which pick one practical technique per stage out
of many listed alternatives), most of this module's sub-topics *are*
individual, small pieces of API surface — so the hands-on app below
implements essentially all of them directly, each mapped to a specific
route or file:

| Sub-topic | Where it's implemented |
|---|---|
| Sync vs async endpoints | `GET /health`, `POST /chat` (sync `def`) vs. `POST /chat/stream`, `WS /ws/chat`, `POST /files/upload` (`async def`) |
| Streaming endpoints / REST vs WebSocket vs SSE | all three side by side: `/chat` (REST), `/chat/stream` (SSE), `/ws/chat` (WebSocket) |
| WebSocket endpoints | `WS /ws/chat` |
| Background tasks and status polling | `POST /jobs/report` + `GET /jobs/{job_id}`; also `POST /files/upload`'s processing |
| Project structure setup | `models.py` / `sessions.py` / `jobs.py` / `deps.py` / `app.py` |
| Pydantic request/response models | `models.py` |
| Dependency injection | `deps.py`'s `require_api_key` / `enforce_rate_limit`, wired in via `Depends(...)` |
| StreamingResponse | `/chat/stream` |
| BackgroundTasks | `/jobs/report`, `/files/upload` |
| CORS, auth, rate limiting middleware | `CORSMiddleware` in `app.py`; auth + rate limiting as dependencies in `deps.py` |
| Exception handlers for LLM errors | `llm_unavailable_handler` in `app.py` |
| Stateless API design | `tenant_id` + `conversation_id` passed explicitly by the client every call, not a server-side cookie/sticky session |
| Redis for server-side session storage | **not implemented** — an external service; `sessions.py`'s in-memory `SessionStore` stands in for it, same treatment Module 5 gave Qdrant and Module 11 gave Redis/PostgreSQL |
| Conversation ID with UUID | `uuid.uuid4()` in `/chat` and `/ws/chat` when the client doesn't supply one |
| Multi-tenant session isolation | `SessionStore` keyed by `(tenant_id, conversation_id)` |
| Session expiry and cleanup | `SessionStore._cleanup_expired()` — TTL-based, the same expiry pattern Module 11's long-term memory uses |
| Accepting file uploads via multipart form | `/files/upload`'s `UploadFile` parameter |
| File type and size validation | `/files/upload`'s content-type + size checks |
| Async file processing pipelines | `/files/upload`'s background summarization task |
| Temporary file storage and cleanup | `tempfile.NamedTemporaryFile` + a `finally:` block that deletes it |

---

## 6. Hands-on: a FastAPI backend covering every sub-topic

### 6.1 Files

| File | Role |
|---|---|
| `models.py` | Pydantic request/response models (§2). |
| `sessions.py` | `SessionStore` (per-tenant conversation history with TTL expiry) and `RateLimiter` (fixed-window, per API key) (§2-3). |
| `jobs.py` | In-memory job store backing background-task status polling (§1). |
| `deps.py` | Shared `Depends()` functions: API-key auth, rate limiting (§2). |
| `app.py` | The FastAPI app: every route, the CORS middleware, and the custom exception handler. |
| `demo.py` | A scripted walkthrough hitting every endpoint via FastAPI's `TestClient` — no real server process needed. |

### 6.2 Setup

```bash
cd app
pip install -r ../requirements.txt
python demo.py
```

Needs a Gemini API key in `common/.env` (the shared folder next to the
module folders) as `GEMINI_API_KEY=...` (keys
come from https://aistudio.google.com/apikey) — `python-dotenv` loads
it automatically. The model is `gemini-3.1-flash-lite`.

To run it as a real server instead (e.g. to try it from a browser or
`curl`):

```bash
uvicorn app:app --reload
```

then call it with the demo API key, e.g.
`curl -H "X-API-Key: demo-key" -X POST localhost:8000/chat -d '{"message": "hi"}' -H "Content-Type: application/json"`.

### 6.3 What `demo.py` walks through

1. `GET /health` — unauthenticated, sync.
2. `POST /chat` with no `X-API-Key` header → `401`.
3. Two `/chat` calls with the same `tenant_id`+`conversation_id` — the
   second correctly recalls a fact from the first (session state
   working).
4. The same `conversation_id` again, but a **different** `tenant_id` —
   correctly has no memory of that fact (multi-tenant isolation
   working).
5. `POST /chat/stream` — prints each SSE `data: ...` piece as it
   arrives.
6. `WS /ws/chat` — connects, receives a generated `conversation_id`,
   sends one message, receives one reply, over a single WebSocket
   connection.
7. `POST /jobs/report` then `GET /jobs/{job_id}` — starts a background
   report-generation job and polls its result.
8. `POST /files/upload` (a valid `.txt` file) then `GET /jobs/{job_id}`
   — uploads a file, polls until its background-generated summary is
   ready.
9. `POST /files/upload` with an unsupported content type → `415`,
   without ever reaching the LLM.
10. Six rapid `/files/upload` calls with a bad content type, against a
    5-requests-per-10-seconds limit.

### 6.4 What to expect

- Steps 3-4 print real, distinct model replies proving the session
  store actually separates tenants — the different-tenant reply
  correctly says it doesn't know the user's name.
- Step 10 prints a run of `415`s followed by `429`s (e.g. `415, 415,
  429, 429, 429, 429`) — the first calls reach the upload handler and fail its own content-type check;
  once the shared rate limiter's window fills up, later calls are
  rejected with `429` *before* that check ever runs — proof the
  dependency actually runs first. (The exact split between `415`s and
  `429`s depends on how many rate-limit hits earlier steps already
  used within the current window, since the limiter is a single shared
  in-memory counter across the whole `demo.py` run — running it fresh
  each time may shift it by a call or two.)
- Steps 7-8 show `"status": "done"` immediately on the very first poll,
  not `"pending"` — because `TestClient` runs a `BackgroundTasks`
  function to completion as part of finishing the request/response
  cycle, before control returns to the caller. Against a real running
  server (`uvicorn app:app`) handling concurrent traffic, polling
  `GET /jobs/{job_id}` immediately after the `POST` can genuinely
  observe `"pending"` before the job finishes — `demo.py`'s in-process
  `TestClient` just can't demonstrate that race by itself.
- Two real issues were caught and fixed while verifying this app:
  declaring the API-key header as `Header(...)` (required) made
  FastAPI reject a *missing* header with its own `422` before
  `require_api_key`'s body ever ran, instead of the intended `401` —
  fixed by making the header optional (`Header(default=None)`) and
  checking it manually, so both "missing" and "wrong" collapse to the
  same `401`. Separately, the first version of the rate-limit demo
  reused the slow, real-LLM `/chat` endpoint — with each call taking
  several real seconds, 5 sequential calls could take longer than the
  10-second window itself, so the limit never actually triggered;
  switched the demo to hit the fast, no-LLM-call invalid-upload path
  instead, which reliably lands multiple hits inside one window.

### 6.5 Simplifications

- Auth is a single hardcoded demo API key (`deps.API_KEY`), not real
  per-tenant credential storage.
- The WebSocket endpoint doesn't enforce the API-key dependency (kept
  auth on the REST endpoints only, to keep this "very simple" per the
  module's scope) — a production WebSocket endpoint would validate a
  token during the handshake before `accept()`.
- `SessionStore`, `RateLimiter`, and the job store are all plain
  in-memory Python dicts, reset every time the process restarts —
  Redis/a real database would be needed for state that survives a
  restart or is shared across multiple server processes (§3).
