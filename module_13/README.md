# Module 13 — Observability and Monitoring

Concept overview — brief explanations of every sub-topic in
Module 13 — Observability and Monitoring.

## 1. Why Observability Matters

- **Non-deterministic outputs need visibility** — the same prompt can
  produce different outputs from run to run, so a single "it worked once"
  test isn't enough; you need ongoing visibility into what a system is
  actually producing in production.
- **Debugging prompt issues** — when a response is wrong or malformed,
  observability is what lets you go back and inspect exactly which
  prompt, context, and parameters produced it, instead of guessing.
- **Cost visibility per user and feature** — token usage (and therefore
  cost) varies wildly by request; without per-user/per-feature breakdowns
  it's impossible to know what's actually driving spend.
- **Quality degradation detection** — a model, prompt, or upstream change
  can silently make outputs worse; observability is what surfaces that
  drift instead of it going unnoticed until a user complains.
- **Latency monitoring** — GenAI calls are slow relative to typical API
  calls, and latency varies a lot by prompt length and model load, so
  it needs to be tracked explicitly rather than assumed.

## 2. LangSmith

- **Setting up tracing** — instrumenting an application so every LLM
  call (and the surrounding logic) is recorded as a trace.
- **Automatic tracing with LangChain** — LangSmith can trace a LangChain
  pipeline (Module 8/9's chains and agents) with no extra code, since
  LangChain integrates with it directly.
- **Manual tracing for non-LangChain code** — wrapping arbitrary code
  (a raw API call, a custom pipeline) in LangSmith's tracing decorators/
  context managers when it isn't already going through LangChain.
- **Trace explorer** — LangSmith's UI for browsing individual traces.
- **Inspecting inputs, outputs, intermediate steps** — drilling into one
  trace to see the exact prompt sent, the exact response received, and
  any intermediate steps (tool calls, retrieved documents) in between.
- **Adding metadata to runs** — attaching extra tags (user id, feature
  name, experiment name) to a trace so it can be filtered/grouped later.
- **Creating datasets and running evaluations** — turning traced
  examples into a reusable evaluation dataset (Module 16 territory).
- **Comparing prompt versions with experiments** — running the same
  dataset against two prompt versions and comparing results side by
  side.

## 3. Langfuse

- **Self-hosting vs Langfuse Cloud** — Langfuse can run as a hosted
  cloud product or be self-hosted, unlike LangSmith which is
  cloud-only.
- **Tracing via Python SDK decorator** — wrapping a function with a
  decorator to automatically trace everything it does.
- **Manual tracing with spans and generations** — explicitly opening a
  "span" (a unit of work) and a "generation" (one LLM call within it),
  for finer control than the decorator gives.
- **User and session tracking** — associating traces with a specific
  user id and conversation/session id, so activity can be viewed per
  user or per session.
- **Cost tracking per model** — Langfuse computes cost per trace using
  each model's known pricing.
- **Dashboards — latency, cost, error rate** — pre-built aggregate views
  over all traced activity.
- **Prompt management** — storing and versioning prompts inside Langfuse
  itself rather than hardcoding them in application code.

## 4. Metrics to Track

- **Latency — TTFT and total generation time** — Time-To-First-Token
  (how long until the first piece of a streamed response arrives) and
  total generation time (how long until the full response is done) are
  both worth tracking, since a slow-to-start-but-fast-to-finish response
  feels very different from a slow-throughout one.
- **Token usage — input, output, total** — the raw quantity driving both
  cost and (indirectly) latency.
- **Cost per request, per user, per feature** — token usage translated
  into money, broken down by who/what is generating it.
- **Error rate — API errors, tool failures, timeouts** — the fraction of
  calls that didn't complete successfully, broken down by failure type.
- **Cache hit rate** — the fraction of requests served from a cache
  instead of a real model call (Module 14's territory — this module
  doesn't have a caching layer yet to measure).
- **Retrieval relevance scores** — for a RAG system (Module 6), how
  relevant the retrieved context actually was to the query.
- **User satisfaction scores** — explicit or implicit feedback (a
  thumbs-up/down, a follow-up "that's wrong") indicating whether a
  response actually helped the user.

## 5. Logging Best Practices

- **What to include in prompt/response logs** — enough to debug an issue
  later (the full prompt, the full response, model/parameters used,
  timestamps, identifiers) without over-logging.
- **PII scrubbing before logging** — redacting personal information
  (emails, phone numbers, etc.) from a prompt/response *before* it's
  written to a log, so the log itself doesn't become a PII liability.
- **Structured JSON logging** — logging each event as a JSON object with
  consistent fields, rather than a free-text line, so logs can be
  parsed, filtered, and aggregated programmatically.
- **Correlation IDs across services** — a single id generated for one
  logical request and threaded through every service/log line it
  touches, so all of that request's activity can be reassembled later.
- **Log levels — DEBUG, INFO, ERROR** — tagging each log line by
  severity, so noisy detail can be filtered out in normal operation
  while still being available when actively debugging.
- **Log storage — Elasticsearch, Loki, CloudWatch** — where structured
  logs actually get shipped and indexed for querying at scale, in a
  production deployment.

---

## 6. One sub-topic per topic

| Topic | Roadmap sub-topics available | Sub-topic used here |
|---|---|---|
| **Tracing / observability platform** | LangSmith (cloud, LangChain-native); Langfuse (self-host or cloud) | **A hand-built local tracer** — a `Span` context manager writing structured trace records to a local file, the same "hand-build a minimal version instead of reaching for the SaaS tool" approach Module 11 took for Mem0/Zep |
| **Metrics tracked** | latency (TTFT + total); token usage; cost; error rate; cache hit rate; retrieval relevance; user satisfaction | **Latency (TTFT + total generation time), token usage, cost per request/user/feature, error rate** — the metrics a single traced LLM call in this repo can genuinely produce; cache hit rate is deferred to Module 14 (no caching layer exists yet), retrieval relevance and user satisfaction are out of scope (this app has no retrieval step or feedback loop) |
| **Logging** | what to log; PII scrubbing; structured JSON; correlation IDs; log levels; log storage (Elasticsearch/Loki/CloudWatch) | **Structured JSON logging to a local `.jsonl` file**, with PII scrubbed before every write, a `request_id` correlation id per call, and `INFO`/`ERROR` log levels |

Everything else each topic lists (LangSmith/Langfuse themselves as hosted
products, dataset/experiment tooling, prompt management, retrieval-relevance
and user-satisfaction metrics) is intentionally out of scope for this mini
app — it hand-builds a minimal version of what those tools provide, the same
way Module 6 hand-built RAG and Module 11 hand-built memory instead of
reaching for a ready-made library.

## 7. Why each chosen technique, briefly

**A hand-built local tracer instead of LangSmith/Langfuse.** Both are real
SaaS/self-hosted products, not something to reimplement wholesale — but
this repo's whole convention is local, no-external-service tooling. The
`Span` class mirrors Langfuse's own "manual tracing with spans and
generations" pattern closely enough to demonstrate the actual mechanic
(open a span, do work, record structured attributes, close it) without
requiring an account or a running service.

**Latency, tokens, cost, error rate as the tracked metrics.** These four
are the ones a single local LLM call can genuinely produce evidence for.
Cache hit rate needs an actual cache (Module 14); retrieval relevance and
user satisfaction need a retrieval step and a feedback mechanism, neither
of which this app has.

**Local JSONL file instead of Elasticsearch/Loki/CloudWatch.** All three
are external log-aggregation services; a local file gets the "structured,
queryable, one record per line" property this module actually wants to
demonstrate, without needing a service to run.

**PII scrubbing as a regex step, not full detection.** Module 15 covers
PII detection/redaction properly (Microsoft Presidio, reversible
anonymization); here it's a lighter touch — just enough (email and phone
regexes) to keep obvious PII out of a log file, which is the actual
concern this module's "PII scrubbing before logging" bullet is about.

---

## 8. Hands-on: a local tracer with a dashboard rollup

### 8.1 Files

| File | Role |
|---|---|
| `pii_scrub.py` | `scrub()` — regex-based email/phone redaction, applied before anything is logged (§5). |
| `tracer.py` | `Span` — a manual tracing context manager that records latency, TTFT, token usage, illustrative cost, and writes one scrubbed, structured JSON line per finished call to `traces.jsonl` (§2-3, §5). |
| `metrics.py` | Aggregates `traces.jsonl` into a dashboard-style summary — total requests, error rate, average latency/TTFT, token/cost totals, and per-feature/per-user cost breakdowns (§4). |
| `app.py` | Runs four traced example requests plus one deliberate failure, then prints the aggregated dashboard. |

### 8.2 Setup

```bash
cd app
pip install -r ../requirements.txt
python app.py
```

Needs a Gemini API key in `common/.env` (the shared folder next to the
module folders) as `GEMINI_API_KEY=...` (keys
come from https://aistudio.google.com/apikey) — `python-dotenv` loads
it automatically. The model is `gemini-3.1-flash-lite`.

### 8.3 Step-by-step: what happens when you run `python app.py`

1. `traces.jsonl` (if left over from a previous run) is deleted, so each
   run's dashboard reflects only that run.
2. Four requests are sent, each tagged with a `user_id` and `feature`,
   streamed via the Gemini API so both time-to-first-token and total latency can
   be measured for real. One request's message deliberately contains a
   fake email and phone number.
3. A fifth call deliberately targets an unreachable port
   (`localhost:19999`) — a real `ConnectionError`, not a simulated one —
   so the tracer's `ERROR`-level path and the dashboard's error-rate
   metric are both exercised against a genuine failure.
4. Every call's `Span.finish()` writes one structured JSON line to
   `traces.jsonl`, with the prompt and response already passed through
   `scrub()`.
5. `metrics.summarize()` reads `traces.jsonl` back and prints the
   aggregated dashboard: total requests, error rate, average latency/TTFT,
   total tokens, total cost, and per-feature/per-user cost breakdowns.

### 8.4 What to expect

- Each successful request prints its own latency, time-to-first-token,
  and illustrative cost — real numbers from a real streamed call, not
  placeholders.
- The request containing a fake email/phone prints its *logged* prompt
  with `[EMAIL_REDACTED]`/`[PHONE_REDACTED]` in place of the real
  values — proof the PII scrub actually ran before the write, not just
  that it exists as a function.
- The deliberately unreachable call prints at `[ERROR]` with the real
  connection-error message, and the dashboard's `error_rate` reflects
  it (`0.2` for 1 error out of 5 total requests).
- The dashboard's `by_feature`/`by_user` breakdowns show real,
  different totals per tag — proof the metadata attached to each span
  (§2's "adding metadata to runs") is actually usable for filtering
  later, the same way LangSmith/Langfuse's dashboards let you slice by
  tag.

### 8.5 Runtime dependencies

`app.py` streams from Gemini's `gemini-3.1-flash-lite` via `common/llm_client.py`'s
`stream_chat()`. Token counts are approximated with `tiktoken` (same
caveat as Modules 2/11 — not Gemini's own tokenizer). Illustrative cost
uses a fixed $/1M-token pricing table
(`tracer.py`'s `PRICING_PER_1M`) — the same "illustrative paid-API
pricing math" convention as Module 2's `cost_calculator.py`, not
Gemini's actual price list. `traces.jsonl` is deleted and rewritten on
every run of `app.py`, rather than accumulated across runs.
