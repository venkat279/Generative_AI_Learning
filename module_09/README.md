# Module 9 — Single Agent Systems

Concept overview — brief explanations of every sub-topic in
Module 9 — Single Agent Systems.

## 1. Core Concepts

- **What is an AI agent** — an LLM that decides its own sequence of
  actions (which tools to call, in what order) to reach a goal, rather
  than following a path you hardcoded.
- **Agent vs a simple LLM call vs a chain** — a plain call answers
  once from a fixed prompt; a chain (Module 8) runs a fixed sequence
  of steps you defined; an agent decides *at runtime* which steps to
  take and how many, based on what it observes along the way.
- **The agent loop — Perceive → Think → Act → Observe** — the general
  cycle every agent runs: take in the current state, reason about
  what to do next, take an action (usually a tool call), observe the
  result, and repeat until done.
- **When to use an agent vs a fixed pipeline** — use a fixed
  chain/pipeline when the steps and their order are known in advance;
  reach for an agent only when the *path itself* needs to vary
  per-input in ways you can't enumerate ahead of time.
- **Limitations of single agents — reliability, latency, cost** —
  every extra reasoning/tool-call round trip is another chance for
  the model to go off track, adds real latency (multiple LLM calls
  instead of one), and multiplies token cost proportionally.

## 2. Agent Architecture

- **Components of an agent — LLM brain, tools, memory, executor** —
  the model that reasons, the functions it can call, the state it
  remembers across steps, and the code loop that actually runs
  everything and enforces stop conditions.
- **System prompt design for agents** — the system prompt has to
  establish not just persona/scope (Module 3) but the agent's
  operating procedure: what tools exist, when to use them, and when
  to stop and answer.
- **Tool registry — how agents discover and select tools** — the same
  tool-definition mechanism from Module 7 (name + description +
  schema), just with potentially many tools registered at once, that
  the model picks from at each step.
- **Scratchpad — how agents track intermediate reasoning** — a
  running log of the agent's own thoughts/actions/observations so far
  in the current task, fed back into the prompt each step so it has
  memory of what it already tried.
- **Stop conditions — when an agent decides it is done** — the model
  itself signals completion (e.g. by responding without another tool
  call), or an external limit (max steps, a timeout) forces a stop.

## 3. ReAct Agent Pattern

- **Reasoning and Acting interleaved** — the core idea behind "ReAct":
  make the model explicitly write out its reasoning *between* actions,
  not just before the first one, so each action is grounded in what
  was just observed.
- **Thought → Action → Observation loop** — the concrete repeating
  unit: the model writes a thought, takes an action (tool call), gets
  an observation (the tool's result), and writes the next thought
  based on it.
- **Implementing ReAct from scratch** — hand-rolling this loop with
  raw prompt text and string parsing (no framework) — closest in
  spirit to how this repo built modules 1-7 via `common/llm_client.py`.
- **ReAct with LangChain `AgentExecutor`** — LangChain's built-in
  runner for this loop: you give it a model, tools, and a prompt, and
  it manages the Thought/Action/Observation cycle for you.
- **ReAct with LlamaIndex `ReActAgent`** — the equivalent built-in
  agent runner in LlamaIndex, same pattern, different library.

## 4. OpenAI Tools Agent

- **How OpenAI function calling powers agents** — instead of a model
  writing "Thought: ... Action: ..." as free text (ReAct's original
  form), the model uses the structured tool-calling mechanism from
  Module 7 directly — the same mechanism Module 7 uses with
  Gemini (`gemini-3.1-flash-lite`).
- **`create_tool_calling_agent` in LangChain** — LangChain's
  constructor for an agent that uses structured tool calls (like
  Module 7's) instead of parsed ReAct text, paired with `AgentExecutor`
  to run it.
- **`OpenAIAgent` in LlamaIndex** — LlamaIndex's equivalent
  tool-calling agent class.
- **Structured tool outputs for agent decisions** — having the model's
  action be a real structured call (name + parsed arguments) rather
  than free text to parse is more reliable — this is exactly why
  Module 7's `calculate` tool used the native tool-calling API instead
  of asking the model to describe a function call in prose.

## 5. LangGraph Single Agent

- **Modeling an agent as a graph** — instead of a hidden loop inside
  a library's `AgentExecutor`, LangGraph makes each step an explicit
  node and each transition an explicit edge you define and can
  inspect.
- **State schema for agent scratchpad** — a typed object (e.g. a
  `TypedDict`) defining exactly what data flows between nodes —
  messages so far, tool results, whatever the agent needs to track.
- **Node — call LLM** — a graph node whose job is just "send the
  current state to the model, get its response."
- **Node — execute tool** — a graph node whose job is "run whatever
  tool the LLM node's last output asked for."
- **Conditional edge — continue loop or end** — a branch that
  inspects the state after the LLM node and decides: route to the
  tool node and loop back, or stop and return the final answer.
- **Compiling and running the agent graph** — `graph.compile()`
  turns the node/edge definition into a runnable object, invoked the
  same way as any LangChain chain from Module 8.

## 6. Agent Memory

- **Short-term memory — conversation history in context** — the
  running list of messages (same idea as Module 2's sliding-window
  history) kept in the prompt for the duration of one task.
- **Injecting tool results back into context** — each tool's output
  gets appended to the message history so the next reasoning step can
  see it — exactly what Module 7's `app.py` does by appending a
  `{"role": "tool", ...}` message.
- **Carrying state across agent steps** — the graph/executor's state
  object persists across every node visit within one run, not just
  the message list.
- **Connecting agent to a long-term memory store** — beyond a single
  run, persisting facts/history somewhere durable (a database, a
  vector store) so a *later*, separate run can recall them — this is
  Module 11's actual subject, just previewed here.

## 7. Error Handling and Reliability

- **Max iteration limit to prevent infinite loops** — a hard cap on
  how many Thought/Action cycles an agent can run, so a confused agent
  can't loop forever.
- **Handling tool execution failures gracefully** — a tool raising an
  exception should become an observation the agent can react to (e.g.
  "that failed, try something else"), not a crash — the same principle
  Module 7's tool description note already called out.
- **Fallback behavior when LLM produces invalid tool calls** — if the
  model's requested tool/arguments don't parse or don't exist, the
  agent needs a defined response (report the error back to the model,
  or abort) instead of crashing.
- **Logging every agent step for debugging** — recording each
  thought/action/observation so a multi-step run can be inspected
  after the fact, not just its final answer.
- **Timeout on individual tool calls** — bounding how long any single
  tool execution is allowed to run, independent of the overall
  iteration limit.

## 8. Agent Evaluation

- **Measuring task completion rate** — did the agent actually
  accomplish the goal, across a test set of tasks — the most basic
  agent metric.
- **Trajectory evaluation — were the right tools called in the right
  order** — judging the *path* the agent took, not just whether the
  final answer was correct.
- **Step efficiency — did the agent reach the answer in minimum
  steps** — penalizing an agent that gets there eventually but takes
  many more actions than necessary.
- **Human review of agent traces in LangSmith** — LangChain's tracing
  platform for inspecting real agent runs step-by-step; the practical
  tool for doing the trajectory/efficiency review above.

---

## 9. Hands-on: LangGraph single agent

A single agent, built as an explicit LangGraph graph over Google's
Gemini model (`gemini-3.1-flash-lite`) — the concrete version of §5 above. Reuses
the calculator tool concept from Module 7, reimplemented with
LangChain's own tool-calling mechanism this time.

### 9.1 Prerequisites: a Gemini API key

Put a Gemini API key in `common/.env` (the shared folder next to the
module folders) as `GEMINI_API_KEY=...` (keys come from
https://aistudio.google.com/apikey). `app.py` gets its chat model from
`../../common/llm_client.py`'s `langchain_chat_model()` — a
`ChatGoogleGenerativeAI` preconfigured with the shared model name and
that key.

### 9.2 Setup

```bash
cd app
pip install -r ../requirements.txt
python app.py
```

### 9.3 Files

| File | Role |
|---|---|
| `calculator_tool.py` | The custom tool: `calculate()`, decorated with LangChain's `@tool` — same AST-based safe evaluator as Module 7's version, not `eval()`. |
| `app.py` | Builds the LangGraph agent (`agent` node calls the LLM, `tools` node executes any requested tool, a conditional edge decides loop-vs-end) and runs 2 example questions through it. |

### 9.4 How it works

The graph has exactly two nodes and one decision point:

```
[agent] --tools_condition--> "tools" --> [tools] --> back to [agent]
                          --> END     --> stop, return final answer
```

- **`agent` node** (`call_model`) — sends the current message list to
  `langchain_chat_model(...).bind_tools([calculate])` (a
  `ChatGoogleGenerativeAI` from `common/llm_client.py`) and appends its response.
- **`tools_condition`** (LangGraph's prebuilt helper) — inspects the
  agent's latest message: if it contains a tool call, route to
  `"tools"`; otherwise route to `END`.
- **`tools` node** (LangGraph's prebuilt `ToolNode`) — runs whichever
  tool the agent's message requested and appends the result as a
  `ToolMessage`, then control flows back to `agent` so it can reason
  over that result.

This loop can run a different number of times per question — unlike
Module 8's chain, which always runs exactly the same two steps
regardless of input.

### 9.5 What to expect

`app.py` prints each step of the loop so the varying step count is
visible directly in the output:

- *"What is 847 times 293?"* takes **3 steps**: a tool call, the
  tool's exact result, then a final answer grounded in that result.
- *"What is the capital of France?"* takes **1 step** — the agent
  answers directly ("The capital of France is Paris.") without calling
  the tool at all, since nothing about it needs math, so
  `tools_condition` routes straight to `END` after the first `agent`
  turn.
