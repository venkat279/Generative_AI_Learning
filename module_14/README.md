# Module 14 — Caching and Cost Optimization

Concept overview — brief explanations of every sub-topic in
Module 14 — Caching and Cost Optimization.

## 1. Why Caching is Critical

- **LLM calls are expensive and slow** — every call costs real money (or,
  for a local model, real time and compute) and takes noticeably longer
  than a typical API call, so avoiding a redundant one is a direct win on
  both axes.
- **Semantically similar queries are common** — many real user queries are
  reworded versions of something already asked ("what's the capital of
  France?" vs "what's France's capital city?"), which an exact-match cache
  alone would never catch.
- **Caching reduces costs by 30–70% in production** — a commonly cited
  range for how much of a production LLM workload's traffic is
  cacheable, once both exact and semantic matches are accounted for.

## 2. Exact-Match Caching

- **Caching by exact prompt hash** — hashing the exact prompt text and
  using that hash as a cache key, so an identical repeated prompt is
  served from the cache instead of calling the model again.
- **Redis for fast lookup** — a common production choice for the
  key-value store backing this, valued for very low read/write latency.
- **TTL settings** — a cached entry expires after some time, so stale
  responses (to a prompt whose "correct" answer might change over time)
  don't get served indefinitely.
- **Cache invalidation strategies** — ways to proactively remove or
  refresh cached entries before their TTL expires, when the underlying
  data they depend on changes.

## 3. Semantic Caching

- **How semantic caching works** — instead of requiring an exact string
  match, embed the incoming query and compare it (by similarity) against
  the embeddings of previously cached queries; a close-enough match is
  served from the cache.
- **GPTCache — open-source semantic cache** — a purpose-built open-source
  library implementing this pattern behind a simple API.
- **Redis with vector extension** — using Redis's vector-search
  capability to store and query the query embeddings a semantic cache
  needs, instead of a separate vector database.
- **Similarity threshold tuning** — how close a match needs to be
  (typically a cosine-similarity cutoff) before it's treated as a cache
  hit; too low returns wrong answers for different questions, too high
  misses real paraphrases.
- **Cache hit rate measurement** — tracking what fraction of queries are
  actually served from the cache, to judge whether it's paying for
  itself.

## 4. LiteLLM

- **Unified interface to 100+ providers** — one consistent API surface
  that can call OpenAI, Anthropic, Google, and many others through the
  same function signature, instead of each provider's own SDK.
- **Automatic model routing** — LiteLLM can route a request to a specific
  provider/model based on configured rules.
- **Budget management per project/user** — setting a spending cap per
  project or user and having LiteLLM enforce it.
- **Fallback chains across models** — if a call to one model/provider
  fails, automatically retrying against a different one.
- **Load balancing across API keys** — spreading traffic across multiple
  API keys/deployments of the same model to stay under per-key rate
  limits.
- **Cost tracking and reporting** — LiteLLM computes and reports cost
  per call using each provider's known pricing, across all providers it
  routes to.

## 5. Prompt Compression

- **LLMLingua** — an open-source library that compresses a long prompt
  by removing less-informative tokens, using a small language model to
  judge which tokens matter, while trying to preserve the meaning an LLM
  needs to answer correctly.
- **LongLLMLingua** — a variant of LLMLingua tuned specifically for very
  long contexts (e.g. RAG's retrieved documents), where compression
  matters even more for cost/latency.
- **Extractive compression** — keeping whole original sentences/spans
  verbatim (just fewer of them), rather than rewriting the text.
- **Abstractive compression** — generating a shorter *rewritten* version
  of the text (via another LLM call), rather than only selecting from
  the original spans.
- **Compression ratio vs quality tradeoff** — the more aggressively a
  prompt is compressed, the more likely it is to lose information the
  model actually needed — compression ratio has to be chosen against
  how much quality loss is acceptable.

## 6. Model Routing Strategy

- **Cheap models for simple tasks** — routing straightforward requests
  (a factual lookup, a short classification) to a smaller/cheaper model
  that's more than capable of handling them correctly.
- **Powerful models for complex tasks** — reserving a larger/more
  expensive model for requests that actually need its extra reasoning
  capability.
- **Classifying task complexity before routing** — deciding which tier a
  given request belongs to, typically via a cheap classification step
  (a small model, or simple heuristics) run before the real call.
- **Cascading — small model first, escalate if needed** — trying the
  cheap tier first regardless, and only calling the powerful tier if the
  cheap tier's response looks insufficient (e.g., it got cut off, or
  failed some quality check) — routing on evidence rather than a
  prediction made in advance.

---

## 7. One sub-topic per topic

| Topic | Roadmap sub-topics available | Sub-topic used here |
|---|---|---|
| **Exact-match caching** | hash lookup; Redis; TTL; invalidation | **An in-memory dict keyed by a SHA-256 hash of the prompt, with TTL expiry** instead of Redis — no external service, the same expiry pattern Module 11's long-term memory uses |
| **Semantic caching** | GPTCache; Redis + vector extension; similarity threshold; hit-rate measurement | **Hand-built: embed each query via Gemini's `gemini-embedding-001`, compare cosine similarity against cached query embeddings, a tunable similarity threshold, and hit/miss counters** — the same embed-then-cosine mechanic Modules 4/6/11 use, applied to cache lookups instead of documents/facts |
| **Multi-provider abstraction (LiteLLM)** | unified interface; automatic routing; budget mgmt; fallback chains; load balancing; cost tracking | **Not implemented** — this repo deliberately calls one provider (the Gemini API) directly; a multi-*provider* router has nothing to route between here. Theory only, the same treatment Module 5 gave Qdrant |
| **Prompt compression** | LLMLingua; LongLLMLingua; extractive; abstractive; ratio/quality tradeoff | **A hand-built extractive compressor** — score each sentence by how many high-frequency, non-stopword terms it shares with the rest of the text, and keep only the top-scoring sentences. LLMLingua/LongLLMLingua need `transformers`/a small model runtime — the same compiled-ML-stack dependency that blocked FlashRank reranking (Module 6) |
| **Model routing strategy** | cheap vs powerful tiers; complexity classification; cascading | **Real classification and real cascade logic, with a simulated model swap** — both tiers call the same `gemini-3.1-flash-lite` model, one with a tight/cheap-style generation budget and one with a generous/powerful-style one — see §8 for why |

Everything else each topic lists (Redis and its vector extension as
external services, LiteLLM's multi-provider machinery, LLMLingua's model-
based token scoring, abstractive compression) is intentionally out of
scope for this mini app — it hand-builds a minimal version of what those
tools/patterns provide, the same way earlier modules hand-built RAG,
memory, and observability instead of reaching for a ready-made library.

## 8. Why each chosen technique, briefly

**In-memory dict for exact-match, hand-built embeddings for semantic.**
Both avoid Redis, keeping this module dependency-free and consistent with
the repo's local-only convention — the mechanics (hash lookup with TTL;
embed-and-compare with a threshold) are the real techniques, just backed
by a Python dict/list instead of an external store.

**Extractive compression via sentence scoring, not LLMLingua.**
LLMLingua needs a small model runtime (`transformers`) to judge token
importance — the same compiled-dependency chain (ultimately needing
`msvcp140.dll`, see repo root `ds-context.md`) that already blocks
`sentence-transformers` and blocked FlashRank in Module 6. A from-scratch
word-frequency sentence scorer is pure Python, needs nothing beyond
`re`/`collections`, and demonstrates the same underlying idea — the
compressed output above genuinely drops the two off-topic sentences (a
stray comment about weather, a comment about a team lunch) while keeping
the four sentences that actually carry the incident's content.

**Model routing simulated on one model, not a real second model.**
Both tiers use `gemini-3.1-flash-lite`, the model used everywhere else
in the repo. The alternative — a second, genuinely different model so
the two tiers call different models for real — was considered and
explicitly declined in favor of keeping this exercise to the model
already in use everywhere else in the repo. The classification (`classify_complexity`)
and the cascade trigger (escalate when the cheap tier's response is cut
off by its token budget — `done_reason == "length"`, which
`common/llm_client.py` sets when Gemini reports `finishReason: MAX_TOKENS`) are both
genuine logic, not simulated; only the "cheap model" vs "powerful model"
half of the distinction is simulated, via generation settings
(`temperature`/`num_predict`) rather than a real model swap.

---

## 9. Hands-on: caching, compression, and routing over the Gemini API

### 9.1 Files

| File | Role |
|---|---|
| `exact_cache.py` | `ExactCache` — SHA-256 hash lookup with TTL expiry and hit/miss counters (§2). |
| `semantic_cache.py` | `SemanticCache` — Gemini-embedding + cosine-similarity lookup with a tunable threshold, TTL expiry, and hit/miss counters (§3). |
| `compression.py` | `compress()` — word-frequency-based extractive sentence compression (§5). |
| `router.py` | `classify_complexity()` + `route()` — heuristic complexity classification, tiered generation settings, and cascade-on-truncation escalation (§6). |
| `app.py` | Three example demos: caching, compression, and routing/cascading, run back to back. |

### 9.2 Setup

```bash
cd app
pip install -r ../requirements.txt
python app.py
```

Needs a Gemini API key in `common/.env` (the shared folder next to the
module folders) as `GEMINI_API_KEY=...` (keys
come from https://aistudio.google.com/apikey) — `python-dotenv` loads
it automatically. Both models (`gemini-3.1-flash-lite` for chat,
`gemini-embedding-001` for embeddings) are called through that one key.

### 9.3 Step-by-step: what happens when you run `python app.py`

1. **Caching demo.** Four queries run against one `ExactCache` and one
   `SemanticCache`, in order: a fresh query (real LLM call, cached in
   both), an exact repeat (hits the exact cache), a paraphrase of the
   first query (misses the exact cache, hits the semantic cache), and a
   genuinely different query (misses both, real LLM call again).
2. **Compression demo.** An 8-sentence synthetic incident report — six
   sentences about a checkout-page bug and its fix, two unrelated asides
   about the weather and a team lunch — is compressed to roughly half
   its sentence count, and the before/after token counts are printed.
3. **Routing/cascading demo.** Three queries run through `route()`: a
   short factual question (classified `simple`, answered within the
   cheap tier's budget, no escalation), a short "define X" question
   whose natural answer is longer than the cheap tier's budget
   (classified `simple`, gets cut off, escalates to the `complex` tier
   for a full answer), and a long, open-ended planning question
   (classified `complex` directly).

### 9.4 What to expect

- The exact-repeat query returns instantly (0ms) from `ExactCache`,
  while the paraphrase returns from `SemanticCache` with a printed
  similarity score (e.g. `sim=0.993` for "What is the capital of
  France?" vs "What's the capital city of France?") — both real cache
  hits, not simulated ones. "What is the capital of Germany?" scores
  around `0.88` against the cached France query — below the `0.90`
  threshold — so it correctly misses and makes a real LLM call.
- The compression demo's output keeps the four sentences that share
  vocabulary with the rest of the text (customer, checkout, payment
  service, fix) and drops the two that don't (weather, team lunch) —
  the compressor is making a real content-based choice, not just
  truncating.
- The routing demo prints `[simple]` for the short factual question,
  `[complex (escalated)]` for the "define recursion" question — proof
  the cascade trigger (`done_reason == "length"`) fired for real, not
  just that the code path exists — and `[complex]` (no escalation tag)
  for the long planning question, since it was routed to the `complex`
  tier directly rather than cascading into it.

### 9.5 Runtime dependencies

`exact_cache.py` needs nothing beyond the standard library. `semantic_cache.py`
embeds via Gemini's `gemini-embedding-001` (`common/llm_client.py`'s `embed()`)
with the `SEMANTIC_SIMILARITY` task type — Gemini's embedding mode for
comparing two texts' meaning, which separates paraphrases (~`0.98`+)
from merely related queries (~`0.86`-`0.88`) widely enough for a single
threshold — and
uses `numpy` for the dot-product similarity — `embed()` returns
L2-normalized vectors, so dot product and cosine similarity are numerically
identical, the same convention Module 4 documents. `compression.py` needs
only the standard library. `router.py` and `app.py` call Gemini's
`gemini-3.1-flash-lite` via `common/llm_client.py`'s `chat()`; `app.py` also uses
`tiktoken` to report before/after token counts for the compression demo
(same approximate-tokenizer caveat as Modules 2/11/13).
