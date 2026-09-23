# Module 1 — Understanding the GenAI Landscape

Concept overview — brief explanations of every sub-topic in
Module 1 — Understanding the GenAI Landscape.

## 1. LLM Fundamentals

- **What is a Large Language Model (LLM)** — a neural network (almost
  always a Transformer) trained on huge amounts of text to predict the
  next token given everything before it; chaining that one-step
  prediction repeatedly is what produces full generated text.
- **What is a token** — the unit an LLM actually reads and writes, not
  quite a word and not quite a character — usually a word, sub-word
  piece, or punctuation mark, produced by a tokenizer (e.g. `tiktoken`
  for OpenAI models).
- **Context window and its limits** — the maximum number of tokens
  (input + output combined) a model can attend to in one call; anything
  beyond that limit is either truncated or must be summarized/chunked
  before it fits.
- **Temperature and randomness** — a sampling parameter controlling how
  much randomness is injected when picking the next token; low values
  make output near-deterministic, high values make it more varied and
  creative (see §5).
- **Hallucination — what it is and why it happens** — a model stating
  something false or made-up with full confidence, because it's
  generating the statistically most plausible next tokens, not
  querying a fact database; it has no built-in mechanism to know what
  it doesn't know.
- **Determinism vs creativity** — the trade-off `temperature`/`seed`
  control directly: `temperature=0` with a fixed `seed` gives
  near-identical output every run (useful for testing/reliability),
  higher temperature gives more varied, exploratory output.
- **Base model vs instruction-tuned model vs chat model** — a base
  model only continues text statistically (no notion of "answering a
  question"); an instruction-tuned model is further trained to follow
  explicit instructions; a chat model is instruction-tuned specifically
  around multi-turn `system`/`user`/`assistant` conversation structure.

## 2. Model Ecosystem

- **OpenAI GPT-4o, GPT-4o mini** — OpenAI's flagship multimodal model
  and its smaller, cheaper/faster sibling.
- **Anthropic Claude 3.5 Sonnet, Claude 3 Haiku** — Anthropic's
  mid-tier (balanced capability/cost) and smallest/fastest models.
- **Google Gemini 1.5 Pro, Gemini Flash** — Google's high-capability and
  low-latency/low-cost model tiers, notable for very large context
  windows.
- **Mistral, Mixtral, Phi-3** — smaller open-weight model families
  (Mistral AI's dense and mixture-of-experts models, Microsoft's
  Phi-3) that can be downloaded and run locally, without an API key.
- **Cohere Command R+** — a model family from Cohere tuned
  specifically for retrieval-augmented generation and tool use.
- **Closed-source vs open-source models** — closed models (GPT-4o,
  Claude, Gemini) are only accessible via a paid hosted API with no
  visible weights; open-weight models (Llama, Mistral, Phi) can be
  downloaded and run locally — this repo's hands-on scripts call the
  hosted Gemini API (`gemini-3.1-flash-lite`) instead.

## 3. Model Benchmarks

- **LMSYS Chatbot Arena** — a crowdsourced benchmark where humans
  compare two anonymous models' responses head-to-head and vote for the
  better one, aggregated into an Elo-style leaderboard.
- **MMLU (Massive Multitask Language Understanding)** — a
  multiple-choice benchmark spanning 57 academic subjects, used as a
  general knowledge/reasoning score.
- **HumanEval** — a benchmark of hand-written programming problems that
  scores a model on whether its generated code actually passes the
  given unit tests.
- **How to read a model card** — a model card documents a model's
  intended use, training data (at whatever detail the provider
  discloses), known limitations, and benchmark scores — the first place
  to check before picking a model for a given task.

## 4. Model Capabilities

- **Text generation** — the core capability every LLM has: producing
  free-form natural language continuations.
- **Code generation** — generating and completing source code,
  typically evaluated with benchmarks like HumanEval (§3).
- **Vision (image understanding)** — multimodal models that accept
  image input alongside text (e.g. GPT-4o, Gemini) and can describe,
  answer questions about, or reason over images.
- **Audio (speech-to-text, text-to-speech)** — converting spoken audio
  to text or generating spoken audio from text, sometimes via a
  dedicated model (e.g. Whisper) rather than the main LLM.
- **Embeddings** — a separate capability (covered fully in Module 4)
  where the model outputs a dense vector representing a text's meaning,
  instead of generating more text.
- **Function calling / tool use** — a model, instead of just replying
  in text, returns a structured request to call a named function with
  specific arguments, letting it take real actions or fetch real data
  (covered fully in Module 7).
- **Document understanding** — extracting structure and answering
  questions over long or structured documents (PDFs, forms, tables),
  often combined with OCR for scanned input.
- **Structured output generation** — constraining a model's output to a
  specific format (JSON, a schema) rather than free-form prose (covered
  fully in Module 3).

## 5. Key Model Parameters

- **temperature** — scales the randomness of next-token sampling; `0`
  is near-deterministic, higher values (e.g. `1.0`+) are more varied.
- **max_tokens** — a hard cap on how many tokens the response may
  contain; generation stops once it's hit, even mid-sentence.
- **top_p (nucleus sampling)** — instead of sampling from the full
  vocabulary, only samples from the smallest set of most-likely tokens
  whose cumulative probability reaches `p`.
- **top_k** — similarly restricts sampling, but to a fixed count of the
  `k` most-likely next tokens rather than a probability mass.
- **stop sequences** — one or more strings that, if generated,
  immediately end the response before `max_tokens` is reached.
- **seed** — fixes the random number generator's starting state, making
  otherwise-random sampling reproducible run-to-run (paired with low
  temperature for the most deterministic results).
- **frequency_penalty** — reduces the likelihood of tokens that have
  already appeared often in the output so far, discouraging repetition.
- **presence_penalty** — reduces the likelihood of any token that has
  appeared at all yet (regardless of how often), encouraging the model
  to introduce new topics/words.
- **system, user, assistant message roles** — the three roles that
  structure a chat-model conversation: `system` sets persona/behavior
  up front, `user` is the human's input, `assistant` is the model's
  prior replies — used to build multi-turn context (covered fully in
  Module 2).

---

## 6. Hands-on: Module 1 sample scripts

### 6.1 Setup

```bash
pip install -r requirements.txt
```

### 6.2 Prerequisites: a Gemini API key

`parameters.py` and `message_roles.py` call Google's Gemini API
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
| `tokens_and_context.py` | Tokens, tokenization, context window budgeting | No |
| `parameters.py` | Key model parameters — `temperature`, `seed`; determinism vs creativity | Yes (Gemini) |
| `message_roles.py` | `system` / `user` / `assistant` message roles | Yes (Gemini) |

Both scripts use `../../common/llm_client.py` — shared Gemini client.

### 6.4 Suggested run order

1. `samples/tokens_and_context.py` — builds intuition for tokens/context windows first, no API call needed.
2. `samples/parameters.py`, then
3. `samples/message_roles.py`.

### 6.5 How these samples work

`parameters.py` and `message_roles.py` call the real Gemini
`gemini-3.1-flash-lite` model — the responses you see are genuine
model output, not canned text. `parameters.py` demonstrates real
sampling behavior: at `temperature=0.0` with a fixed `seed`, Gemini
returns the (near-)identical answer both times; at `temperature=1.2`
you'll see genuinely different phrasings each run.
`tokens_and_context.py` explores tokenization with `tiktoken` and is not API-dependent.
