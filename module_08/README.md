# Module 8 — Orchestration Frameworks

Concept overview — brief explanations of every framework listed
in Module 8 — Orchestration Frameworks.

## What "orchestration framework" means here

Every module so far (1-7) called the Gemini API directly through a small
hand-rolled client (`common/llm_client.py`) — plain HTTP requests, no
abstraction layer. An orchestration framework is a library that
provides ready-made building blocks for common LLM-application
patterns instead — prompt templates, chained calls, memory,
retrieval, tool/agent integration — so you compose from existing
pieces rather than hand-rolling each one. The trade-off is always the
same: less code to write yourself, at the cost of an abstraction layer
between you and the raw API calls.

## 1. LangChain

The most widely used general-purpose LLM orchestration framework.
Provides: prompt templates, "chains" (composable sequences of an LLM
call plus other steps), memory classes for conversation history,
retriever interfaces for RAG, output parsers, and a very large
ecosystem of integrations (model providers, vector stores, document
loaders — including a first-party Gemini integration, `langchain-google-genai`).
Its newer graph-based extension, **LangGraph**, is what the roadmap's
later agent modules (9-10) build on. Known for being flexible and
broadly applicable, but also for having a lot of abstraction layers
that can make it harder to see exactly what's being sent to the model
underneath — the opposite trade-off from the hand-rolled
`common/llm_client.py` that modules 1-7 use.

## 2. LlamaIndex

Originally built specifically for connecting LLMs to your own data —
more RAG-centric than LangChain's broader "chain anything together"
scope. Provides high-level "engines" (a `QueryEngine` for one-shot
Q&A, a `ChatEngine` for conversational RAG) on top of the same
ingest → chunk → embed → index → retrieve pipeline this repo hand-built
in Module 6, plus connectors for many data sources (SQL databases,
APIs, structured/unstructured documents). Where LangChain is a
general toolkit, LlamaIndex's center of gravity is specifically
"get your data into an LLM-queryable index with minimal code."

## 3. CrewAI

A framework specifically for **multi-agent** systems: you define
several agents with distinct roles (e.g. "Researcher," "Writer,"
"Editor"), give each a goal and a set of tools, and a `Crew` +
`Process` coordinates them working together on a task. More
opinionated and higher-level than building the same thing directly in
LangGraph — it trades flexibility for a simpler role-based mental
model. This maps most directly onto Module 10 (Multi-Agent Systems)
later in the roadmap, rather than being a general orchestration tool
like LangChain/LlamaIndex.

## 4. Claude Coding Agentic Framework

The roadmap's least standardized entry — unlike the other three
(established, pip-installable open-source packages), this refers to
Anthropic's own tooling/patterns for building **coding agents**
specifically (e.g. the Claude Agent SDK, and the architecture patterns
behind tools like Claude Code itself: a model with file/shell/tool
access operating in a loop over a codebase). It's narrower in scope
than the other three — purpose-built for coding-assistant-style
agents rather than general LLM orchestration.

---

## 5. Hands-on: LangChain mini app

A two-step orchestrated pipeline over Google's Gemini model
(`gemini-3.1-flash-lite`): classify a support message, then draft a reply using that
classification — the same "classify, then act on it" shape as Module
3's structured-output work and Module 7's tool use, built this time
with LangChain's composition primitives instead of raw HTTP requests.

### Why LangChain?

- It's the general-purpose one — LlamaIndex would mostly re-demonstrate
  Module 6's RAG pipeline with a different library, and CrewAI's
  multi-agent focus belongs more naturally in Module 10.
- It has a first-party Gemini integration (`langchain-google-genai` /
  `ChatGoogleGenerativeAI`, the same one Module 17 uses) — no new
  external dependency beyond one more pip package.
- A small, honest demo — a prompt template piped into `ChatGoogleGenerativeAI`,
  maybe with LangChain's memory or output-parser abstraction — would
  directly show what the framework buys you over this repo's
  hand-rolled `common/llm_client.py`, which is exactly the point of an
  "orchestration frameworks" module.
- It sets up Module 9 cleanly, since LangGraph (used there for the
  ReAct agent pattern) is built on top of LangChain.

### 5.1 Prerequisites: a Gemini API key

Put a Gemini API key in `common/.env` (the shared folder next to the
module folders) as `GEMINI_API_KEY=...` (keys come from
https://aistudio.google.com/apikey). `app.py` gets its chat model from
`../../common/llm_client.py`'s `langchain_chat_model()` — a
`ChatGoogleGenerativeAI` preconfigured with the shared model name and
that key.

### 5.2 Setup

```bash
cd app
pip install -r ../requirements.txt
python app.py
```

### 5.3 Files

| File | Role |
|---|---|
| `app.py` | Defines both chains and runs 2 example support messages through them. |

### 5.4 How it works

- **`classify_chain = CLASSIFY_PROMPT | json_llm | JsonOutputParser()`**
  — a `ChatPromptTemplate` describing the JSON schema to return, piped
  into a `ChatGoogleGenerativeAI` from `langchain_chat_model()` (with
  Gemini's JSON mode enabled via `response_mime_type="application/json"`) and
  then a `JsonOutputParser()` that turns the raw string response into
  a Python dict, in one composed object.
- **`respond_chain = RESPOND_PROMPT | llm | StrOutputParser()`** — a
  second, independent chain that drafts a reply, parameterized by the
  category/urgency the first chain produced.
- **`handle_message()`** runs them in sequence: `classify_chain.invoke(...)`'s
  dict output is read directly into `respond_chain.invoke(...)`'s
  input — the two steps compose into one pipeline without any manual
  JSON parsing, request building, or response-shape handling, which is
  what callers of `common/llm_client.py`'s `chat()` all do by hand in modules 1-7.

### 5.5 What to expect

Both example messages get a real classification (e.g. the billing
double-charge message comes back `{"category": "billing", "urgency":
"high"}`) and a real, coherent drafted reply grounded in that
classification. Classification isn't always the most intuitive label
a human would pick (e.g. a return-policy question landing under
`"billing"` rather than `"shipping"`) — normal LLM classification
variance, not a bug in the chain.

Some of LangChain's transitive dependencies print a
`Microsoft Visual C++ Redistributable is not installed` warning to
stderr on import; this doesn't affect functionality here — the app
runs and produces real output regardless, since none of the parts
this app actually touches (`ChatGoogleGenerativeAI`, the prompt/parser classes)
need it. It's the same underlying machine-level gap noted for
module_05, just an unrelated dependency probing for it eagerly.
