# Module 2 — Working with LLM APIs

Concept overview — brief explanations of every sub-topic in
Module 2 — Working with LLM APIs.

## 1. API Fundamentals

- **How REST API calls to LLMs work** — a chat completion is a plain
  HTTP POST: a JSON body describing the conversation and parameters
  goes to an endpoint, and a JSON body with the model's reply comes
  back — no persistent connection or special protocol required.
- **API key authentication** — hosted providers (OpenAI, Anthropic,
  Google) require a secret key sent in a request header to identify
  and bill the caller; this repo's scripts send a Gemini key
  (`GEMINI_API_KEY`) in the `x-goog-api-key` header.
- **The messages array structure** — the conversation is sent as an
  ordered list of `{role, content}` objects (§ Module 1's `system` /
  `user` / `assistant` roles), not a single flat prompt string.
- **Constructing a basic API request** — assembling the model name,
  the messages array, and any sampling parameters (Module 1 §5) into
  one JSON payload for the POST body.
- **Reading and parsing API responses** — extracting the generated
  text (and metadata like token usage) from the response JSON's
  nested structure, which differs slightly per provider.
- **HTTP status codes from LLM APIs** — standard HTTP semantics apply
  (`200` success, `401` bad key, `429` rate-limited, `5xx` server
  error), plus provider-specific error bodies with more detail.
- **Rate limits — RPM and TPM** — providers cap usage by Requests Per
  Minute and/or Tokens Per Minute; exceeding either returns a `429`.
- **Retry logic and exponential backoff** — on a transient failure
  (rate limit, server error), wait and retry rather than failing
  immediately, doubling (or similarly scaling) the wait time between
  attempts so repeated failures don't hammer the API.
- **Error types — rate limit, context exceeded, content policy, server
  error** — the main categories an API call can fail with, each
  needing different handling (retry vs. shorten input vs. rephrase vs.
  give up).
- **Timeout handling** — setting a maximum time to wait for a response
  before giving up, so a hung connection doesn't block the caller
  indefinitely.

## 2. Streaming Responses

- **What streaming is** — receiving the model's output incrementally,
  piece by piece, as it's generated, instead of waiting for the full
  response before anything is returned.
- **Server-Sent Events (SSE)** — the underlying HTTP mechanism most
  hosted providers use for streaming: a long-lived response where each
  chunk arrives as a separate `data: ...` line.
- **Implementing streaming in Python** — reading the response body
  incrementally (e.g. line-by-line) as it arrives instead of via a
  single blocking call, and parsing each chunk into text as it comes.
- **Handling partial responses** — each streamed piece is a fragment of
  the full answer, so a caller must accumulate/print pieces
  incrementally rather than assuming any one chunk is complete.
- **Handling stream interruptions** — a connection can drop mid-stream;
  a robust client needs to detect that and decide whether to retry or
  surface a partial result.

## 3. Token Management

- **Counting tokens before a call** — estimating how many tokens a
  request will use ahead of time, to stay within context limits and
  predict cost.
- **tiktoken for OpenAI** — OpenAI's own open-source tokenizer library,
  giving exact token counts for OpenAI models' specific encodings
  (e.g. `cl100k_base`).
- **Claude's token counter** — Anthropic provides its own tokenizer/
  counting endpoint, since Claude's vocabulary differs from OpenAI's.
- **Estimating cost per request (input vs output tokens)** — hosted
  providers price input and output tokens differently (output is
  usually pricier), so cost estimation needs both counts separately.
- **Context window limits per model** — each model has its own maximum
  token budget (input + output combined), which varies significantly
  across providers and model versions.
- **Truncation vs summarization vs chunking for long inputs** — three
  strategies when input exceeds the context window: cut off the excess
  (truncation, lossy), condense it with the model first
  (summarization), or split it into smaller pieces processed
  separately (chunking, the strategy Module 6's RAG pipeline builds
  on).
- **Building a cost calculator** — combining token counts with a
  provider's per-token pricing to estimate what a given request (or
  batch of requests) will cost before running it.

## 4. SDKs and Developer Tooling

- **OpenAI Python SDK** — OpenAI's official client library, wrapping
  the raw REST calls in typed Python methods.
- **Anthropic Python SDK** — Anthropic's equivalent official client for
  the Claude API.
- **Google Generative AI SDK** — Google's official client for the
  Gemini API.
- **LiteLLM for unified multi-provider access** — a library that
  exposes one consistent interface across many providers (OpenAI,
  Anthropic, Google, local models, etc.), so switching providers
  doesn't mean rewriting all the calling code.

## 5. Multi-turn Conversations

- **Maintaining conversation history** — keeping the growing list of
  past `messages` around so each new call includes the full
  conversation so far, since the API itself is stateless between
  calls.
- **Appending user and assistant turns** — after each exchange, both
  the user's message and the model's reply are appended to the history
  list before the next call.
- **Managing context length in long conversations** — history grows
  unbounded turn by turn, so at some point it must be trimmed or
  condensed to stay under the model's context window.
- **Summarization strategy** — periodically replacing older turns with
  a model-generated summary of them, preserving the gist while freeing
  up token budget.
- **Sliding window strategy** — dropping the oldest turns outright once
  a token/turn budget is exceeded, always keeping the system message
  and the most recent exchanges.

---

## 6. Hands-on: Module 2 sample scripts

### 6.1 Setup

```bash
pip install -r requirements.txt
```

### 6.2 Prerequisites: a Gemini API key

`streaming.py` and `multi_turn_conversation.py` call Google's Gemini API
(`gemini-3.1-flash-lite`) through `../../common/llm_client.py`. Before running
them:

1. Put a Gemini API key in `common/.env` (the shared folder next to
   the module folders) as `GEMINI_API_KEY=...`
   (keys come from https://aistudio.google.com/apikey).
2. `python-dotenv` (in `requirements.txt`) loads that `.env` file
   automatically - no environment variables to export by hand.

### 6.3 Files

All scripts live in `samples/`.

| Script | Topic covered | Needs API key? |
|---|---|---|
| `retry_with_backoff.py` | Error types, retry logic and exponential backoff | No |
| `streaming.py` | Streaming responses, handling partial chunks and interruption | Yes (Gemini) |
| `cost_calculator.py` | Token counting (`tiktoken`), estimating cost per request | No |
| `multi_turn_conversation.py` | Multi-turn conversations, sliding-window history management | Yes (Gemini) |

The two Gemini-backed scripts use `../../common/llm_client.py` —
shared Gemini client (`chat()` for a request, `stream_chat()` for
streaming).

### 6.4 Suggested run order

1. `samples/retry_with_backoff.py` and `samples/cost_calculator.py` first — pure local logic, no API call involved.
2. `samples/streaming.py`, then `samples/multi_turn_conversation.py`.

### 6.5 How these samples work

`streaming.py` prints the model's output incrementally as Gemini
streams it back, using its Server-Sent Events (`alt=sse`) streaming format.

`multi_turn_conversation.py`'s `ChatSession` keeps conversation
history under a token budget using a sliding window, trimming the
oldest turns first while always preserving the system message. Token
counts for this budget come from `tiktoken`'s `cl100k_base` encoding,
not Gemini's own tokenizer: counting locally avoids an extra API
round trip (Gemini's `countTokens` endpoint) on every turn.
`cl100k_base` is close enough to demonstrate the trimming behavior.

`retry_with_backoff.py` demonstrates retry logic and exponential
backoff against a simulated flaky function, so the retry behavior can
be exercised without depending on a real rate limit. 

`cost_calculator.py` estimates request cost from `tiktoken` token counts 
against an illustrative pricing table, for comparing per-request cost across
models.
