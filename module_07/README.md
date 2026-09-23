# Module 7 — Function Calling and Tool Use

Concept overview — brief explanations of every sub-topic 
in Module 7 — Function Calling and Tool Use.

## 1. Core Concepts

- **What function calling is** — a way for an LLM to say "call this
  function with these arguments" instead of just replying with text,
  so it can trigger real code (a database lookup, an API call, a
  calculation) as part of answering a prompt.
- **How it works: tool definition → model decision → execution → response**
 — you describe available functions ("tools") to the model up front; 
  the model decides whether to call one and with what arguments; 
  your code actually runs the function; the result is fed
  back to the model so it can produce a final answer.
- **OpenAI function calling specification** — tools are declared as
  JSON schemas in the `tools` parameter; the model returns a
  `tool_calls` object naming the function and arguments instead of
  (or alongside) normal text.
- **Anthropic tool use specification** — conceptually the same idea,
  declared via a `tools` parameter with `input_schema`; Claude returns
  a `tool_use` content block, and you reply with a `tool_result` block
  in the next turn.
- **Gemini function calling** — same pattern again, via
  `FunctionDeclaration`/`Tool` objects; Gemini returns a
  `functionCall` part the caller must execute and return.
- **Function calling vs RAG vs plain prompting** — plain prompting
  only uses what's in the prompt/training data; RAG injects retrieved
  *text* into the prompt; function calling lets the model trigger
  *actions* (fetch live data, perform a calculation, write to a
  system) — they're complementary, not competing techniques.

## 2. Defining Tools

- **JSON schema for tool definitions** — each tool's expected
  arguments are described using standard JSON Schema (types,
  required fields, etc.), the same format used to validate JSON
  elsewhere.
- **`name`, `description`, `parameters` fields** — the three parts of
  a tool definition: a unique identifier, a natural-language
  explanation the model uses to decide *when* to call it, and the
  JSON Schema for its arguments.
- **Writing effective tool descriptions** — the description is the
  model's only guide for picking the right tool; vague descriptions
  cause wrong or missed tool calls, so they should be specific about
  what the tool does and when to use it.
- **Required vs optional parameters** — JSON Schema's `required` list
  controls which arguments the model must always supply versus which
  it can omit.
- **Enum types for constrained inputs** — restricting a parameter to
  a fixed set of allowed values (e.g. `"unit": ["celsius",
  "fahrenheit"]`) so the model can't invent an invalid option.
- **Nested object schemas** — parameters can themselves be objects
  with their own sub-fields, for tools that need structured input
  (e.g. an `address` object with `street`/`city`/`zip`).
- **Arrays as parameter types** — a parameter can be a list of values
  or objects, for tools that take multiple items at once (e.g. a list
  of product IDs).

## 3. Executing Tool Calls

- **Detecting when the model calls a tool** — checking the API
  response for a tool-call field (`tool_calls`, `tool_use`,
  `functionCall`) instead of assuming the response is always plain
  text.
- **Extracting tool name and arguments** — the model's response
  contains the function name and a JSON string/object of arguments,
  which your code parses before doing anything.
- **Executing the tool in code** — actually running the
  corresponding Python function with the extracted arguments — this
  step is entirely your own code; the model never runs anything
  itself.
- **Returning tool results to the model** — the function's output is
  sent back to the model in a follow-up message (a `tool` role
  message for OpenAI, a `tool_result` block for Anthropic) so the
  model can use it.
- **Completing the conversation with the final response** — after
  seeing the tool result, the model generates its actual answer to
  the user, now grounded in the real data the tool returned.
- **Multi-turn tool use** — a single user question can trigger several
  rounds of "model calls tool → code executes → result returned →
  model calls another tool" before a final answer is produced.

## 4. Parallel Tool Calling

- **What parallel tool calling is** — the model requesting multiple
  tool calls in a single turn (e.g. checking weather in three cities
  at once) instead of one at a time.
- **OpenAI parallel tool calls implementation** — the response's
  `tool_calls` field is a list, potentially with more than one entry,
  each needing its own execution and result.
- **Executing multiple tools concurrently** — since the calls are
  independent, they can be run in parallel (e.g. with `asyncio` or a
  thread pool) rather than one after another, for speed.
- **Aggregating results from parallel calls** — all the individual
  tool results are collected and sent back together so the model can
  synthesize one final answer from all of them.

## 5. Building Custom Tools

Each of these is just "a Python function wrapped with a tool
definition so the model can call it" — the roadmap lists common
categories of what such a tool might do:

- **Web search tool** — Tavily, SerpAPI, DuckDuckGo: lets the model
  pull in current information beyond its training data.
- **Calculator / code execution tool** — offloads precise math or
  logic the model itself is unreliable at.
- **Database query tool** — lets the model answer questions against
  live, private data (e.g. "what's my order status").
- **REST API caller tool** — a generic wrapper for hitting any
  external HTTP API on the model's behalf.
- **File reader / writer tool** — lets the model read from or write
  to the local filesystem.
- **Email sender tool** — lets the model trigger sending an email as
  an action, not just draft text.
- **Calendar event creator tool** — lets the model schedule real
  calendar events.
- **Browser / web scraper tool** — lets the model fetch and read the
  content of a specific webpage.

## 6. Tool Use Patterns

- **Tool selection via descriptions** — with many tools registered,
  the model relies entirely on each tool's `description` to pick the
  right one, so description quality directly affects accuracy.
- **Tool chaining** — the output of one tool call becomes the input
  to the next (e.g. search for a city's coordinates, then call a
  weather tool with those coordinates).
- **Error handling when a tool fails** — a tool can raise an
  exception, time out, or return bad data; that failure needs to be
  surfaced back to the model (or handled in code) rather than
  crashing the whole interaction.
- **Tool result formatting** — how a tool's raw output (e.g. a
  database row, a JSON blob) is shaped into text the model can read
  and use effectively.
- **Role-based tool access control** — restricting which tools are
  even offered to the model based on who's asking (e.g. a
  "delete_record" tool only registered for admin users).

---

## 7. Hands-on: tool use mini app

Google's Gemini model (`gemini-3.1-flash-lite`) calling one custom tool: an
exact-arithmetic calculator. Covers the core tool-call lifecycle from
§1/§3 above end to end.

### 7.1 Prerequisites: a Gemini API key

Put a Gemini API key in `common/.env` (the shared folder next to the
module folders) as `GEMINI_API_KEY=...` (keys come
from https://aistudio.google.com/apikey) — `python-dotenv` loads it
automatically.

### 7.2 Setup

```bash
cd app
pip install -r ../requirements.txt
python app.py
```

### 7.3 Files

| File | Role |
|---|---|
| `calculator_tool.py` | The custom tool: `TOOL_SCHEMA` (the JSON-schema tool definition sent to the model) + `calculate()` (the actual Python function that executes it). |
| `app.py` | Wires the tool into a chat loop and runs 3 example questions. |

### 7.4 How it works

`common/llm_client.py`'s `chat()` takes an optional `tools` parameter
(a list of OpenAI-style function-tool definitions) and converts it into
Gemini's `functionDeclarations` for the `generateContent` call. `app.py`'s `ask(question)`
does the full round-trip:

1. **Send the question with the tool registered** — `chat(messages, tools=TOOLS)`.
2. **Check for a tool call** — if `message["tool_calls"]` is empty, the model answered directly; print that and stop.
3. **Otherwise, execute the tool** — for each call, read `function["name"]` and `function["arguments"]` (`common/llm_client.py` returns `arguments` as an already-parsed dict, not a JSON string) and run the matching Python function.
4. **Send the result back** — append the assistant's tool-call message, then a `{"role": "tool", "content": <result>}` message, to the conversation.
5. **Get the final answer** — call `chat()` again with the updated messages; the model's reply is now grounded in the tool's exact output instead of its own arithmetic.

### 7.5 The calculator tool

`calculator_tool.py`'s `calculate()` deliberately does **not** use
Python's `eval()` on the model-supplied expression string — that would
let a crafted expression execute arbitrary code. Instead it parses the
expression into an AST (`ast.parse(..., mode="eval")`) and walks it by
hand, only ever applying a fixed whitelist of arithmetic operators
(`+ - * / ** ` and unary minus). Anything else (attribute access,
function calls, imports) raises an error instead of running.

### 7.6 Example questions and what to expect

- *"What is 847 times 293?"* and the bill-splitting question both
  require exact multi-digit arithmetic — the model calls `calculate`,
  and the final answer reflects the tool's exact result rather than
  the model's own (often wrong) mental math.
- *"What is the capital of France?"* needs no tool at all — the model
  answers directly ("The capital of France is Paris.") without emitting
  a tool call, even though `calculate` is registered. Deciding *not* to
  call a tool is as much a part of tool use as calling one: the tool
  schema is only an offer, and the model picks it up only when the
  question actually needs it.
