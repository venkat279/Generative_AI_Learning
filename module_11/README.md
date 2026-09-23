# Module 11 — Memory Systems

Concept overview — brief explanations of every sub-topic in
Module 11 — Memory Systems.

## 1. Types of Memory

- **In-context (short-term) memory** — whatever is currently sitting in
  the prompt/context window (the conversation so far, retrieved
  documents, tool results); it's what the model can "remember" for this
  call only, and it disappears once the context window is exceeded or
  the process ends.
- **External (long-term) memory** — information persisted *outside* the
  context window, in a database or file, and selectively pulled back
  into the prompt when relevant — the only way to remember anything
  across separate conversations or beyond what fits in one context
  window.
- **Episodic memory** — memory of specific past events/interactions
  (e.g. "the user asked about X yesterday, and I answered Y") — tied to
  a particular occurrence, not a general fact.
- **Semantic memory** — memory of general facts, independent of any
  specific conversation (e.g. "the user's timezone is IST," "the user
  prefers concise answers") — closer to a durable profile than a log of
  events.
- **Procedural memory** — memory of *how* to do something (a learned
  process, workflow, or preferred approach), as opposed to episodic
  memory (what happened) or semantic memory (what's true).

## 2. Short-Term Memory Management

- **Conversation history as messages list** — the same `system`/`user`/
  `assistant` messages array from Module 1 §5, growing one entry per
  turn; the simplest form of memory, already used throughout this repo
  (e.g. Module 2's `multi_turn_conversation.py`).
- **Buffer window — last N turns** — once history grows past a limit,
  keep only the most recent N turns and drop the rest outright — simple
  and cheap, at the cost of losing anything older.
- **Summarization of old turns** — instead of dropping old turns,
  periodically replace them with a model-generated summary, preserving
  the gist while freeing up token budget (Module 2 §5).
- **Token-aware truncation** — trimming history based on an actual
  token count against the model's context budget, rather than a fixed
  turn count — what Module 2's `ChatSession` sliding window already
  does with `tiktoken`.
- **Summary buffer — hybrid approach** — keeps the most recent turns
  verbatim (a buffer window) *and* a running summary of everything
  older than that, combining both techniques instead of choosing one.

## 3. Long-Term Memory with Vector Stores

- **Storing user facts as embeddings** — persisting individual pieces
  of semantic memory (§1) as embedded text, the same "embed text, store
  the vector" mechanic as Module 4/6's document indexing, just applied
  to facts about the user instead of document chunks.
- **Storing past conversation summaries** — persisting episodic memory
  (§1) the same way — each summarized past conversation becomes one
  more embedded, retrievable record.
- **Retrieving relevant memories at query time** — embedding the
  current query and running top-K similarity search (Module 4 §4) over
  stored memories, exactly like RAG's retrieval step (Module 6 §1), but
  retrieving *memories about the user/conversation* instead of
  *documents*.
- **Memory consolidation** — periodically merging, deduplicating, or
  summarizing accumulated memories so the store doesn't grow
  unboundedly with redundant or superseded facts.
- **Memory expiry and freshness** — giving stored memories an effective
  shelf life (via timestamps or explicit TTLs) so outdated facts (e.g.
  "the user is currently debugging issue #123") don't get retrieved and
  treated as still true long after they stopped being relevant.

## 4. External State Stores

- **Redis for session state** — an in-memory key-value store commonly
  used to hold active session/conversation state, valued for very low
  read/write latency; typically run as its own service.
- **PostgreSQL for structured memory** — a relational database for
  memory that has real structure (user profiles, fact tables) and
  benefits from SQL querying, joins, and transactional guarantees.
- **SQLite for local persistence** — a file-based, serverless database
  — no separate service to run, just a local `.db` file — well suited
  to this repo's fully-local, no-external-service convention (Python's
  standard library includes `sqlite3` directly).
- **LangGraph checkpointers** — LangGraph's built-in mechanism for
  persisting a graph's full state between calls, keyed by `thread_id`;
  already used in Module 10's human-in-the-loop pipeline via
  `MemorySaver()` (in-memory only) — LangGraph also ships a
  `SqliteSaver`/`PostgresSaver` for state that survives a process
  restart.

## 5. Memory in Practice

- **Injecting retrieved memories into system prompt** — the memory
  equivalent of RAG's augmentation step (Module 6 §8): once relevant
  memories are retrieved (§3), they're formatted into the system prompt
  so the model's response is grounded in what it "remembers," not just
  the current message.
- **User profile memory across sessions** — an accumulated, persistent
  semantic-memory (§1) record about a specific user, available at the
  start of *every* new conversation, not just recalled within one.
- **Project context memory** — the same idea scoped to a project/task
  instead of a user: durable facts about the codebase, decisions, or
  ongoing work that should persist across sessions working on it.
- **Mem0 — open-source memory layer** — a purpose-built open-source
  library that packages extraction, storage, and retrieval of long-term
  memories (§3) behind a simple API, instead of hand-building the
  embed/store/retrieve pipeline yourself.
- **Zep — LLM-specific memory store** — another purpose-built memory
  service for LLM applications, combining short-term conversation
  history with long-term fact/summary extraction and retrieval in one
  system.

---

## 6. One sub-topic per topic

Each area below uses one concrete mechanism, chosen for being the most
commonly used, practical default:

| Topic | Roadmap sub-topics available | Sub-topic used here |
|---|---|---|
| **Short-term memory** | buffer window; summarization of old turns; token-aware truncation; summary buffer | **Summary buffer** — a buffer window of recent turns kept verbatim, with anything older folded into a running summary instead of dropped |
| **Long-term memory** | storing facts/summaries as embeddings; retrieval at query time; consolidation; expiry | **Embeddings + top-K retrieval**, with a similarity-based duplicate check (consolidation) and a TTL-based cutoff (expiry) |
| **External state store** | Redis; PostgreSQL; SQLite; LangGraph checkpointers | **SQLite** — no separate service to run, consistent with this repo's local-only convention |
| **Memory in practice** | injecting retrieved memories into system prompt; user profile across sessions; project context memory | **Retrieved memories injected as a system message**, demonstrated across two separate chat sessions sharing one long-term store |

Everything else each topic lists (Redis/PostgreSQL as external services,
Mem0/Zep as ready-made memory libraries, procedural memory, project
context memory as a separate scope from user memory) is intentionally
out of scope for this mini app — it hand-builds a minimal version of
what those tools/patterns provide, the same way Module 6 hand-built RAG
instead of reaching for LlamaIndex.

## 7. Why each chosen technique, briefly

**Summary buffer.** A pure buffer window loses everything older than N
turns outright; pure summarization-from-the-start loses verbatim detail
even for the most recent exchange. The hybrid keeps the last few turns
exact (so the model can quote them precisely) while folding anything
older into a running summary (so the gist survives instead of vanishing)
— the same trade-off Module 2's sliding window makes, extended with a
summarization step instead of a hard drop.

**Embeddings + top-K retrieval, with consolidation and expiry.** The
same embed-then-cosine-similarity mechanic as Module 4/6, applied to
short factual statements instead of document chunks. Consolidation (skip
storing a near-duplicate of an existing memory) keeps the store from
accumulating the same fact restated many times across a conversation.
Expiry (a TTL on session-scoped facts) keeps transient state — "the user
is currently debugging X" — from being retrieved as if it were still
true long after it stopped being relevant.

**SQLite for external state.** Redis and PostgreSQL both require a
separate running service; SQLite is a single local file, needs no
install (`sqlite3` is in Python's standard library), and is more than
enough for a small memory store — consistent with every other module's
local-only, no-external-service convention.

**Injecting retrieved memories as a system message.** Retrieval alone
doesn't stop a model from also inventing details it wasn't actually told
— the same grounding problem Module 6's RAG prompt template solves for
documents. The memory block here is phrased the same way: list exactly
what's known, and instruct the model not to assume or invent anything
beyond it. When nothing relevant is retrieved, the app says so
explicitly instead of just omitting the block, since an *absent*
instruction is not the same as an *explicit* "you don't know this yet."

---

## 8. Hands-on: memory-augmented chat app

A chat loop that combines short-term (§2) and long-term (§3-4) memory,
demonstrating cross-session recall over two simulated conversations.

### 8.1 Files

| File | Role |
|---|---|
| `short_term_memory.py` | `ShortTermMemory` — buffer window + summary buffer (§2). |
| `long_term_memory.py` | `LongTermMemory` — SQLite-backed embeddings store: add (with consolidation), retrieve (top-K, with expiry) (§3-4). |
| `app.py` | Orchestration: fact extraction, memory-grounded system prompt, and the example run across two sessions (§5). |

### 8.2 Setup

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

### 8.3 Step-by-step: what happens when you run `python app.py`

1. **Session 1** starts with a fresh `ShortTermMemory(window_turns=2)`
   and runs 3 short, plain user turns — "Hi, I'm Sam.", "What's a good
   morning drink besides coffee?", "My favorite color is blue." (one
   fact, one bit of small talk, one more fact — deliberately simple and
   quick to read). For each turn, `respond()`:
   - Retrieves the top-3 long-term memories for the message and drops
     any below `MIN_RELEVANCE` (a similarity score threshold) before
     deciding what to inject.
   - Builds the system prompt: the surviving memories (explicitly
     framed as the *only* things to treat as true), or an explicit "no
     memories yet" message if none survived the threshold.
   - Calls `extract_fact()` — a separate, temperature-0 LLM call asking
     whether this message reveals a personal fact worth remembering —
     and stores it via `long_term.add()` if so (skipped if it's a
     near-duplicate of something already stored — consolidation).
   - Adds the turn to short-term memory, which folds the oldest turn
     into the running summary once more than `window_turns` turns are
     buffered.
2. **Memory expiry demo.** An ephemeral, short-TTL memory is stored
   directly (`ttl_seconds=2`), retrieved once immediately (present),
   then retrieved again after a 3-second sleep (gone) — showing expiry
   actually take effect within one run, not just described in theory.
3. **Session 2** starts a *brand-new* `ShortTermMemory` (no buffer, no
   summary carried over) but reuses the *same* `LongTermMemory`/SQLite
   file. It asks "What's my name, and what's my favorite color?" — the
   only way this can be answered correctly is via long-term retrieval,
   since nothing about the user exists in this session's short-term
   memory.

### 8.4 What to expect

- Session 1's first turn retrieves nothing (empty store) and the system
  prompt explicitly says so — the model should not invent a shared
  history with the user. An earlier version of this app's system
  prompt, without that explicit instruction, fabricated a fictional
  "we spoke last week..." backstory on the very first turn instead —
  the memory-system analogue of RAG hallucination (Module 6 §8):
  retrieval alone doesn't ground a model, the prompt has to say what
  *isn't* known too.
- Each of the two fact-bearing turns prints its own `[long-term]
  stored: ...` line ("The user's name is Sam." / "The user likes the
  color blue."); the small-talk turn prints nothing, since
  `extract_fact()` correctly returns nothing to store for it.
- **One thing to know about `extract_fact()`:** its prompt asks for
  *one* sentence, so a single message carrying two facts at once (e.g.
  "Hi, I'm Sam. I like tea.") can come back with only one of them
  extracted. The demo keeps one fact per turn, which is also a good
  reason to keep each turn's message focused on one thing.
- By Session 1's 3rd turn, `[short-term] folded oldest turn into
  summary` prints once, showing the summary buffer actually activating
  once `window_turns=2` is exceeded.
- The ephemeral fact shows up in the "right after storing" retrieval
  and is gone from the "3s later" retrieval.
- Session 2's answer correctly states both the name and the favorite
  color despite starting with zero short-term context — the
  cross-session recall this app exists to demonstrate.
- Retrieval scores sit well above `MIN_RELEVANCE` (`0.15`) for every
  stored memory (roughly `0.58`-`0.81`), since Gemini embeddings give
  even loosely related sentences a cosine similarity around `0.5` — so
  the threshold only screens out near-zero matches; the explicit "treat
  only these as true" framing is what keeps the reply grounded.

### 8.5 Runtime dependencies

`short_term_memory.py` uses `tiktoken` for token counting (same
approximate-tokenizer caveat as Module 2) and calls Gemini's `gemini-3.1-flash-lite`
(via `common/llm_client.py`) for summarization. `long_term_memory.py` uses Gemini's
`gemini-embedding-001` for embeddings and Python's standard-library `sqlite3` for storage — no
new external service. `app.py`'s `memory_store.db` persists in the
`app/` folder between runs; re-running `python app.py` reuses it, and
re-storing the same facts is skipped via consolidation rather than
duplicated.
