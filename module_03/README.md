# Module 3 — Prompt Engineering

Concept overview — brief explanations of every sub-topic in
Module 3 — Prompt Engineering.

## 1. Prompt Design Foundations

- **Anatomy of a good prompt** — clear task instruction, relevant
  context, and (when it matters) a specified output format, kept as
  unambiguous as possible.
- **System prompt vs user prompt** — the system prompt sets persistent
  behavior/persona for the whole conversation; the user prompt is the
  specific request for this turn (Module 1 §5's message roles).
- **Being explicit vs implicit** — spelling out exactly what's wanted
  ("respond in exactly 3 bullet points") gets more reliable results
  than leaving it to be inferred from context.
- **Positive instructions vs negative instructions** — telling a model
  what *to* do ("write formally") is generally more reliable than only
  telling it what *not* to do ("don't write casually"), since negative
  instructions are easier for the model to drift from.
- **Output format control — plain text, JSON, Markdown, XML, tables** —
  explicitly stating the desired output shape, rather than assuming the
  model will infer it, dramatically improves format consistency.
- **Using delimiters — ` ``` `, `<tags>`, `---`** — visually fencing
  off distinct sections of a prompt (instructions vs. input data vs.
  examples) so the model doesn't conflate them.
- **Specifying output length** — stating a target length (word count,
  sentence count, "one paragraph") to avoid answers that are too terse
  or needlessly long.

## 2. Core Prompting & Reasoning Techniques

- **Zero-shot prompting** — asking the model to perform a task with
  only an instruction, no examples.
- **One-shot prompting** — including exactly one worked example before
  the actual request, to anchor the expected format/style.
- **Few-shot prompting** — including several worked examples, which
  usually improves consistency further than one-shot, at the cost of
  more prompt tokens.
- **Instruction hierarchy** — when instructions conflict (e.g. system
  vs. user), well-behaved models are trained to prioritize
  higher-authority instructions (system > user) over lower ones.
- **Chain-of-Thought (CoT)** — prompting the model to reason step by
  step before giving a final answer, which tends to improve accuracy on
  multi-step problems.
- **Zero-shot CoT** — triggering step-by-step reasoning with a simple
  instruction like "let's think step by step," no worked examples
  needed.
- **Few-shot CoT** — providing worked examples that themselves show
  step-by-step reasoning, not just the final answer, so the model
  imitates the reasoning pattern too.
- **Tree-of-Thought (ToT)** — extending CoT by exploring multiple
  reasoning branches in parallel and comparing/pruning them, rather
  than committing to one linear chain of reasoning.
- **ReAct prompting** — interleaving reasoning steps with actions (like
  tool calls) and their results, so the model can reason, act, observe,
  and reason again — the pattern behind Module 9's single-agent loop.
- **Meta-prompting** — using the model itself to help generate,
  refine, or critique a prompt, rather than a human hand-writing it
  from scratch.

## 3. Structured Output Prompting

- **Prompting for valid JSON** — explicitly instructing the model to
  respond with JSON matching a described shape, since without
  instruction it defaults to prose.
- **XML tags for structured sections** — wrapping distinct parts of a
  response in tags (e.g. `<answer>...</answer>`) as an alternative
  structuring method some models handle very reliably.
- **Prompting for Markdown** — asking for Markdown formatting
  (headers, lists, tables) when the output is meant for human reading
  rather than machine parsing.
- **Forcing consistent output with few-shot examples** — pairing a
  format instruction with example input/output pairs in the exact
  target shape, which reduces format drift versus instruction alone.
- **Handling malformed JSON and retry strategies** — since a model can
  still emit invalid JSON despite instructions, a robust caller
  validates the output and re-prompts (possibly showing the parse
  error) if it fails.
- **JSON mode (OpenAI)** — a provider-level flag that constrains
  decoding so the output is guaranteed to be syntactically valid JSON
  (though not necessarily matching your intended schema).
- **Structured outputs with schema** — some providers accept an actual
  JSON Schema and constrain generation to conform to it exactly, a
  stronger guarantee than JSON mode alone.
- **Tool use for structured extraction** — repurposing function-calling
  (Module 7) as a structured-extraction mechanism: define a "tool" that
  is really just the target schema, and read its arguments as the
  extracted data.

## 4. System Prompt Engineering

- **Writing robust system prompts for production** — system prompts
  used in a real product need to hold up under adversarial or unusual
  input, not just the happy path tested during development.
- **Defining persona, tone, scope, and limitations** — stating who the
  assistant is, how it should sound, what it's meant to help with, and
  what it explicitly should not do.
- **Injecting dynamic context at runtime** — templating the system
  prompt with request-specific values (user name, current date,
  retrieved data) rather than hardcoding a static string.
- **Handling out-of-scope inputs** — instructing the model on what to
  do when asked something outside its intended purpose (decline,
  redirect, give a fixed fallback response).
- **Versioning prompts as code** — tracking prompt changes in version
  control like any other code, since a prompt edit can change behavior
  as significantly as a code change.
- **A/B testing prompts** — running two prompt variants against real
  traffic (or a held-out test set) and comparing outcomes, rather than
  changing a production prompt on judgment alone.

## 5. Prompt Security

- **Prompt injection — direct and indirect** — direct injection is a
  user directly trying to override the system prompt's instructions;
  indirect injection is malicious instructions hidden inside
  third-party content the model reads (a webpage, a document) that it
  then follows as if they were legitimate instructions.
- **Prompt leaking prevention** — guarding against a user tricking the
  model into revealing its own (possibly sensitive) system prompt
  verbatim.
- **Jailbreaking — attack patterns and countermeasures** — techniques
  users try to get a model to bypass its safety/behavioral
  constraints (e.g. role-play framing), and the corresponding defenses.
- **Input validation before passing to LLM** — screening user input for
  known attack patterns before it ever reaches the model, this
  module's `prompt_injection_check.py`.
- **Output filtering** — checking the model's *response* before it
  reaches the end user or downstream system, catching problems that
  input-side validation alone would miss.

---

## 6. Hands-on: Module 3 sample scripts

### 6.1 Setup

```bash
pip install -r requirements.txt
```

### 6.2 Prerequisites: a Gemini API key

Four of these five scripts call Google's Gemini API
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
| `zero_vs_few_shot.py` | Zero-shot vs few-shot prompting | Yes (Gemini) |
| `chain_of_thought.py` | Direct answer vs. Chain-of-Thought reasoning | Yes (Gemini) |
| `structured_json_output.py` | Structured/JSON output prompting, schema validation, retry on malformed output | Yes (Gemini) |
| `system_prompt_engineering.py` | System prompt design — persona, scope, limitations, out-of-scope handling | Yes (Gemini) |
| `prompt_injection_check.py` | Prompt security — a simple input-validation heuristic | No |

The four Gemini-backed scripts use `../../common/llm_client.py` —
shared Gemini client (`chat()`, with an optional `json_mode=True` for
`structured_json_output.py`).

### 6.4 Suggested run order

1. `samples/prompt_injection_check.py` first — no API call needed.
2. `samples/zero_vs_few_shot.py`, `samples/chain_of_thought.py`, `samples/structured_json_output.py`, `samples/system_prompt_engineering.py` in order.

### 6.5 How these samples work

`zero_vs_few_shot.py` compares a bare zero-shot instruction against a
few-shot prompt anchored on labeled examples, classifying the same
review's sentiment under each style. `chain_of_thought.py` classifies
the same support-ticket message ("my card was charged twice") two
ways: a bare classification instruction (`direct_answer`) versus a
prompt that asks the model to identify the main issue and justify its
category choice before answering (`cot_answer`). `structured_json_output.py`
uses Gemini's JSON mode (`responseMimeType: "application/json"`) with a validate-then-retry
loop, re-prompting the model if its output isn't valid JSON matching
the expected schema. `system_prompt_engineering.py` uses a persona-
and scope-constrained system prompt, answering in-scope questions and
refusing out-of-scope ones with a fixed response.

`prompt_injection_check.py` is a regex heuristic that runs before any
text reaches an LLM, flagging suspicious patterns like "ignore
previous instructions."

### 6.6 Notes on scope

- All live-call scripts use the Gemini API (`gemini-3.1-flash-lite`); swapping in a different model is a one-line change to `MODEL` in `common/llm_client.py`.
- `prompt_injection_check.py` is a fully offline demo (a regex heuristic), so the input-validation logic can be verified without any API call.
