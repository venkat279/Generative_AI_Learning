# Module 10 — Multi-Agent Systems

Concept overview — brief explanations of every sub-topic in
Module 10 — Multi-Agent Systems.

## 1. Core Concepts

- **Why multi-agent over single-agent** — a single agent (Module 9)
  juggling everything in one context/prompt gets unwieldy as a task
  grows; splitting responsibilities across several focused agents
  keeps each one's job (and prompt) simpler.
- **Task specialization** — each agent gets a narrower job and a
  system prompt tuned for it (e.g. one that only researches, one that
  only writes), rather than one agent trying to be good at everything.
- **Parallel execution across agents** — independent sub-tasks can run
  on separate agents at the same time instead of one agent working
  through everything sequentially.
- **Verification and self-correction** — one agent can check another's
  output (e.g. a critic reviewing a writer's draft) and send it back
  for revision, catching mistakes a single agent grading its own work
  wouldn't.
- **Scalability** — adding capability often means adding another
  specialized agent to the system, rather than growing one agent's
  prompt/toolset indefinitely.

## 2. Agent Roles

Common recurring roles a multi-agent system is composed from:

- **Planner agent** — breaks the overall goal into a sequence of
  sub-tasks for the other agents to execute.
- **Executor agent** — actually carries out a concrete sub-task
  (usually via tools), analogous to Module 9's single agent but scoped
  to one piece of the larger job.
- **Critic / Reviewer agent** — evaluates another agent's output
  against the goal and flags problems or requests revisions.
- **Researcher agent** — gathers information (search, retrieval) that
  other agents need as input.
- **Writer / Generator agent** — produces the final user-facing
  content from whatever the other agents assembled.
- **Orchestrator / Supervisor agent** — decides which agent acts next
  and routes information between them; the coordinator of the whole
  system.

## 3. Communication Patterns

How information actually flows between agents:

- **Hub-and-spoke** — every agent talks only to a central
  orchestrator, never directly to each other; the orchestrator routes
  everything.
- **Peer-to-peer** — agents message each other directly, with no
  central coordinator.
- **Hierarchical** — a tree of supervision (e.g. a top-level
  orchestrator delegating to team-lead agents, who delegate to worker
  agents), a layered version of hub-and-spoke.
- **Blackboard (shared state)** — agents don't message each other
  directly at all; they all read from and write to one shared state
  object, and react to what's there.
- **Pipeline (linear handoff)** — a fixed sequence, each agent's
  output becoming the next agent's input in a straight line (the
  simplest pattern — literally what Module 8's `classify_chain →
  respond_chain` already does with two chain steps instead of two
  agents).

## 4. Task Decomposition

- **Breaking a goal into sub-tasks** — turning one broad objective
  into a set of smaller, individually-assignable pieces — usually the
  planner agent's job.
- **Dependency graphs between tasks** — some sub-tasks need another
  sub-task's output before they can start; modeling that as a graph
  determines valid execution order.
- **Parallel vs sequential execution** — sub-tasks with no
  dependencies between them can run in parallel (§1); sub-tasks that
  depend on each other's output must run in order.
- **Merging results from multiple agents** — combining several
  agents' separate outputs back into one coherent final result, often
  the orchestrator's or a dedicated writer agent's job.

## 5. Reliability Patterns

- **Retry logic for failed agent steps** — the same exponential-backoff
  idea from Module 2's `retry_with_backoff.py`, applied to a whole
  agent step instead of a single API call.
- **Timeouts on tool calls** — same principle as Module 9's
  single-agent timeout, applied per tool call within a multi-agent run.
- **Fallback agents** — a backup agent (or a simpler, more reliable
  strategy) to fall back to when the primary agent for a role fails or
  produces unusable output.
- **Guardrails on agent outputs** — validating an agent's output
  against expected structure/constraints before passing it to the
  next agent, similar in spirit to Module 3's structured-output
  validation, but between agents rather than between you and one model.
- **Max iteration limits** — the multi-agent equivalent of Module 9's
  single-agent step cap, bounding how many total rounds of
  agent-to-agent handoff can happen.
- **Logging every agent step** — recording which agent did what, in
  what order, across the whole system — the multi-agent version of
  Module 9's per-step logging, but covering *inter*-agent handoffs too.

## 6. Human-in-the-Loop

- **When to pause for human approval** — deciding which actions are
  consequential enough (e.g. sending an email, spending money) that
  the system should stop and wait for a person before proceeding.
- **Interrupting a LangGraph workflow** — LangGraph's built-in support
  for pausing graph execution at a defined point and waiting for
  external input before resuming.
- **Presenting plan before execution** — showing the planner agent's
  proposed sub-tasks to a human up front, before any agent starts
  acting on them.
- **Accepting, rejecting, or editing agent steps** — the actual
  interaction a human has at a pause point: approve as-is, block it,
  or modify it before letting the system continue.
- **Audit trail of agent decisions** — a persistent record of what
  was proposed, what a human approved/changed, and what actually ran —
  accountability on top of the debugging log from §5.

---

## 7. Hands-on: multi-agent pipeline with human-in-the-loop

A `researcher` agent and a `writer` agent handing off work in sequence
(§2's Agent Roles, §3's Pipeline pattern), with a human-in-the-loop
checkpoint (§6) before the result is finalized — built with LangGraph
over Google's Gemini model (`gemini-3.1-flash-lite`).

**Why LangGraph instead of the originally recommended CrewAI:** every
CrewAI release available on this machine (checked all 13, `0.1.0`
through `0.11.2`) depends on `langchain<0.2.0`, which requires
`numpy<2` — and numpy's entire 1.x series has no prebuilt wheel for
this Python version, so installing it means compiling from source,
which needs a C compiler (MSVC/gcc/clang) that isn't installed on this
machine. Rather than install a full compiler toolchain, this was built
directly in LangGraph instead, continuing from Module 9's agent.

### 7.1 Prerequisites: a Gemini API key

Put a Gemini API key in `common/.env` (the shared folder next to the
module folders) as `GEMINI_API_KEY=...` (keys come from
https://aistudio.google.com/apikey). `app.py` gets its chat model from
`../../common/llm_client.py`'s `langchain_chat_model()` — a
`ChatGoogleGenerativeAI` preconfigured with the shared model name and
that key.

### 7.2 Setup

```bash
cd app
pip install -r ../requirements.txt
python app.py
```

This one is interactive — it pauses partway through and waits for you
to type a response in the terminal.

### 7.3 Files

| File | Role |
|---|---|
| `app.py` | Builds and runs the pipeline: `researcher` → `writer` → `human_review` (pauses here) → `finalize`. |

### 7.4 How it works

```
[researcher] --> [writer] --> [human_review] --> [finalize] --> END
                                    |
                          pauses the whole graph here,
                          waits for real input, then resumes
```

- **`researcher_node`** — one `llm.invoke(...)` call with a
  research-agent system framing, producing a few bullet points on the
  given topic.
- **`writer_node`** — a second `llm.invoke(...)` call, given *only*
  the researcher's notes, producing a short paragraph.
- **`human_review_node`** — calls `langgraph.types.interrupt(...)`,
  which pauses the *entire graph*, not just this node. The graph is
  compiled with `checkpointer=MemorySaver()`, which is what makes
  pausing and later resuming possible — without a checkpointer,
  `interrupt()` has no saved state to resume from. `app.py`'s `run()`
  calls `pipeline.invoke(...)` once (runs until the interrupt), prints
  the draft, reads a real decision from `input()`, then calls
  `pipeline.invoke(Command(resume=decision), config=...)` — the *same*
  `thread_id` in `config` is what tells LangGraph which paused run to
  resume.
- **`finalize_node`** — turns the human's decision into the final
  output: `approve` (or just pressing Enter) keeps the draft as-is,
  `reject` discards it, anything else is treated as replacement text
  the human typed in directly — covering all three of §6's "accepting,
  rejecting, or editing agent steps."

### 7.5 What to expect

Running it prints the researcher's notes, then the writer's draft,
then pauses on a prompt:

```
[human] Approve, reject, or type replacement text, then press Enter:
```

- Press Enter (or type `approve`) → the draft is used as the final output.
- Type `reject` → the final output is a fixed `"[rejected by human reviewer]"` marker, nothing else runs.
- Type anything else → that exact text becomes the final output, replacing the draft entirely.
