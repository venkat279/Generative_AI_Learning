# Module 16 — LLM Evaluation

Concept overview — brief explanations of every sub-topic in
Module 16 — LLM Evaluation.

## 1. Why Evaluation Matters

- **Non-deterministic outputs require systematic quality measurement** —
  the same prompt can produce different outputs run to run, so quality
  has to be measured deliberately rather than assumed from a few
  manual checks.
- **Catching regressions when prompts or models change** — a prompt edit
  or model swap can silently make outputs worse; a repeatable evaluation
  is what surfaces that instead of it going unnoticed.
- **Building confidence before shipping to production** — a systematic
  eval run is what turns "it seemed to work when I tried it" into an
  actual basis for deciding something is ready.
- **Measuring quality across diverse and adversarial inputs** — real
  traffic includes edge cases and hostile input, not just the friendly
  examples a developer happens to try by hand.

## 2. Evaluation Fundamentals

- **Reference-based vs reference-free evaluation** — reference-based
  evaluation compares an output against a known-correct answer;
  reference-free evaluation judges an output on its own merits (or
  against context) when no single correct answer exists.
- **Automated vs human evaluation** — automated metrics/judges scale
  cheaply but can miss nuance; human evaluation catches more but is slow
  and expensive.
- **Offline evaluation vs online (live) evaluation** — offline evaluation
  runs against a fixed dataset before shipping; online evaluation
  measures real production traffic after shipping.
- **Unit tests vs regression suites vs benchmark suites** — a unit test
  checks one specific behavior; a regression suite catches the system
  getting worse over time; a benchmark suite measures broad capability
  against a standard reference set.
- **The eval-driven development workflow** — writing the evaluation
  criteria alongside (or before) a prompt/pipeline change, the same
  spirit as test-driven development applied to LLM behavior.

## 3. Building Evaluation Datasets

- **What makes a good eval dataset** — representative of real usage,
  covers edge cases, has a clear notion of "correct" for each example.
- **Collecting seed examples from real usage** — building an initial
  dataset from actual logged interactions rather than only invented
  examples.
- **Synthetic data generation for edge cases** — using an LLM to generate
  additional plausible-but-rare examples a real dataset might be missing.
- **Golden dataset — expected inputs and outputs** — a curated set of
  input/expected-output pairs treated as ground truth for evaluation.
- **Adversarial examples — jailbreaks, ambiguous queries, out-of-scope inputs** 
  — deliberately difficult inputs included specifically to probe failure modes, 
  not just typical ones.
- **Dataset versioning and management** — tracking changes to the
  evaluation dataset itself over time, since the dataset is as much a
  moving target as the code being evaluated.

## 4. Metrics for Text Generation

- **Exact match** — the strictest possible metric: does the output match
  the reference character-for-character (after basic normalization).
- **BLEU score** — an n-gram precision metric originally built for
  machine translation, measuring how much of the candidate's n-grams
  appear in the reference.
- **ROUGE-1, ROUGE-2, ROUGE-L** — a family of recall-oriented n-gram/
  longest-common-subsequence metrics, originally built for summarization.
- **BERTScore — semantic similarity with embeddings** — comparing
  candidate and reference via contextual embeddings instead of surface
  n-grams, so a correct paraphrase scores well even with no words
  in common.
- **METEOR** — an n-gram metric that also accounts for synonyms and word
  order, aiming to correlate better with human judgment than BLEU alone.
- **Perplexity** — how "surprised" a language model is by a given text; a
  measure of fluency/likelihood under a model, not correctness against a
  reference.
- **When each metric is appropriate** — exact match for short,
  deterministic answers; BLEU/ROUGE for tasks with a fairly fixed
  expected phrasing; BERTScore/embedding similarity when correct
  paraphrases are common; perplexity for fluency, not factual accuracy.

## 5. LLM-as-a-Judge

- **Using a strong LLM to score another LLM's output** — instead of a
  fixed-formula metric, prompting a capable model to read a response and
  judge its quality directly.
- **Pointwise scoring — rating a single response** — the judge assigns a
  score (e.g. 1-5) to one response in isolation.
- **Pairwise comparison — choosing the better of two responses** — the
  judge is shown two candidate responses to the same input and picks
  the better one (or a tie).
- **Writing effective judge prompts** — being explicit about the rubric,
  the exact output format expected, and framing the judge as impartial.
- **Bias in LLM judges — position bias, verbosity bias, self-preference**
  — known failure modes: favoring whichever answer appears first/second
  (position bias), favoring longer answers regardless of quality
  (verbosity bias), or favoring output that resembles the judge's own
  style (self-preference).
- **Calibrating judge scores against human labels** — periodically
  checking that the automated judge's verdicts actually agree with real
  human judgment, and adjusting the judge prompt/setup if they drift.

## 6. Task-Specific Evaluation

- **Summarization — faithfulness, coverage, conciseness** — does the
  summary stay true to the source, cover its key points, and avoid
  padding.
- **Question answering — accuracy, groundedness, hallucination rate** —
  is the answer correct, is it actually supported by the given context,
  and how often does it invent unsupported claims.
- **Code generation — functional correctness, execution pass rate** — the
  strongest signal for code is usually whether it actually runs and
  passes tests, not surface similarity to a reference solution.
- **Classification — precision, recall, F1** — the standard
  classification metrics, applicable whenever an LLM's output can be
  reduced to a label.
- **Dialogue — coherence, engagingness, task completion** — whether a
  multi-turn conversation stays consistent, is engaging, and actually
  accomplishes what the user came for.
- **Instruction following — format compliance, constraint adherence** —
  whether the output actually obeys the specific constraints given (a
  requested format, length, or set of rules), independent of whether the
  content itself is otherwise good.

## 7. Hallucination Detection

- **Types of hallucination — factual, faithfulness, reasoning** —
  factual (the claim is simply untrue), faithfulness (the claim isn't
  supported by the given context even if it might be true elsewhere), and
  reasoning (a flawed chain of logic leading to a wrong conclusion).
- **Entailment-based detection** — using a natural-language-inference
  model to check whether a claim is actually logically entailed by its
  supporting context.
- **SelfCheckGPT — sampling-based consistency check** — sampling the same
  model multiple times for the same question and treating inconsistency
  across samples as evidence of hallucination, without needing any
  external reference at all.
- **FactScore — claim-level factuality scoring** — breaking a response
  into individual factual claims and scoring each one's factuality
  separately, rather than scoring the whole response as one unit.
- **Chainpoll — multi-sample polling for hallucination** — a related
  multi-sample approach that polls a judge across several generations to
  estimate a hallucination score.

## 8. Evaluation Frameworks

- **OpenAI Evals** — OpenAI's open-source framework for defining and
  running evaluation suites against its models.
- **EleutherAI LM Evaluation Harness** — a widely used open-source
  harness for running standardized academic benchmarks against language
  models.
- **DeepEval** — an open-source, pytest-style framework for writing LLM
  evaluation "unit tests" with built-in metrics.
- **Promptfoo** — an open-source CLI/library for testing and comparing
  prompts and models against a test-case dataset.
- **Braintrust** — a commercial platform for logging, evaluating, and
  comparing LLM outputs over time.
- **Weights & Biases Weave** — W&B's LLM-focused tracing/evaluation
  tooling, built on top of their existing experiment-tracking platform.

## 9. Human Evaluation

- **When human eval is necessary** — when quality is subjective, high-
  stakes, or not well captured by any automated metric or judge.
- **Designing annotation guidelines** — writing clear, specific
  instructions so different human raters apply the same standard.
- **Inter-annotator agreement — Cohen's Kappa, Krippendorff's Alpha** —
  statistical measures of how consistently multiple human raters agree
  with each other, used to judge whether the guidelines are actually
  well-specified.
- **Labeling platforms — Scale AI, Labelbox, Argilla** — services/tools
  for managing human annotation work at scale.
- **Sampling strategies for cost-efficient human review** — reviewing a
  representative subset of outputs rather than everything, to keep human
  evaluation affordable.
- **Preference data collection for RLHF** — collecting human preference
  judgments (A vs B, not just a score) specifically to train or fine-tune
  a model via reinforcement learning from human feedback.

## 10. Regression Testing and CI/CD

- **Running evals in CI pipelines** — running the evaluation suite
  automatically on every relevant code/prompt change, the same way a
  test suite runs in CI.
- **Thresholds and pass/fail gates** — defining a minimum acceptable
  score and treating anything below it as a build failure.
- **Tracking metric trends over time** — watching how eval scores move
  across changes, not just checking the latest one in isolation.
- **Alerting on quality regressions** — proactively notifying someone
  when a tracked metric drops, rather than waiting for it to be noticed
  by chance.
- **Prompt versioning tied to eval results** — recording which prompt
  version produced which eval results, so a regression can be traced back
  to the specific change that caused it.

---

## 11. What's implemented vs. theory-only

Per this session's explicit instruction, the hands-on app below covers
only the most important, most broadly applicable sub-topics rather than
all eight sections:

| Section | Implemented here | Theory-only |
|---|---|---|
| **Metrics for Text Generation** | Exact match, ROUGE-L (hand-built, LCS-based), embedding-based semantic similarity (a BERTScore stand-in) | BLEU, METEOR, perplexity — redundant with the three implemented metrics for this module's purposes (see §12) |
| **LLM-as-a-Judge** | Pointwise scoring, pairwise comparison, **and a real position-bias check** (swapping which answer is "A" vs "B" and confirming the judge's verdict doesn't flip) | Calibrating against human labels (no human raters here) |
| **Hallucination Detection** | SelfCheckGPT-style sampling consistency (hand-built: resample, embed, measure pairwise agreement) | Entailment-based detection, FactScore, Chainpoll — each needs either a dedicated NLI model or a heavier multi-step pipeline beyond this module's scope |
| **Regression Testing and CI/CD** | A minimal threshold-based pass/fail gate over the golden dataset's average similarity score, demonstrated against both a real (passing) run and a hypothetical (failing) regressed answer | Running in an actual CI pipeline, trend tracking over time, alerting |
| **Building Evaluation Datasets** | A tiny 2-example golden dataset, deliberately paired so one example makes exact match fail (correctly) and the other makes it pass | Seed collection from real usage, synthetic generation, adversarial examples, dataset versioning |
| Evaluation Fundamentals | *(conceptual only — no dedicated hands-on; the distinctions are demonstrated implicitly by what the app already does)* | — |
| Task-Specific Evaluation | *(QA is the task type exercised throughout; the other five task types aren't separately implemented)* | Summarization, code generation, classification, dialogue, instruction-following evaluation |
| Evaluation Frameworks | *(none)* | OpenAI Evals, LM Evaluation Harness, DeepEval, Promptfoo, Braintrust, W&B Weave — all real, installable tools, skipped so the underlying mechanics stay visible directly, the same choice Module 11/13/14/15 made for their own frameworks |
| Human Evaluation | *(none — no human raters in this repo)* | Everything in this section |

## 12. Why these three, briefly

**Exact match + ROUGE-L + semantic similarity, not BLEU/METEOR/perplexity.**
The three implemented metrics already span the useful range for this
module's purposes: strictest (exact match), a real n-gram/LCS-based
metric (ROUGE-L, hand-built rather than installing `rouge-score`), and a
paraphrase-tolerant metric (embedding cosine similarity, reusing the
same `embed()` mechanic Modules 4/6/11/14/16 all share). BLEU and METEOR
would mostly retread ROUGE-L's ground for this repo's short-answer
examples; perplexity measures fluency under a model, not correctness
against a reference, and isn't the kind of "is this answer right"
question this module's demo is actually asking.

**LLM-as-a-judge using gemini-3.1-flash-lite itself, with a genuine bias check.**
No second, "stronger" judge model is used here — `gemini-3.1-flash-lite` judges its
own kind of output, which is itself realistic (many real deployments
judge with the same tier of model they're evaluating, not a categorically
larger one). The position-bias check is a real, run experiment, not a
described-only concept: the same two candidate answers are compared
twice, in each order (A/B, then B/A), to check whether the verdict
depends on position rather than content.

**SelfCheckGPT-style consistency, not entailment/FactScore/Chainpoll.**
Consistency-via-resampling needs nothing beyond a chat model and an
embedding model - both already available through the Gemini API - while entailment-based
detection needs a dedicated NLI model, and FactScore/Chainpoll both need
either claim extraction or a separate judging pass on top. SelfCheckGPT's
core mechanic (resample, compare, treat disagreement as risk) is real
and it's the cheapest of the four to demonstrate honestly.

**A deliberately matched pair of dataset examples, not five similar
ones.** An earlier version of this dataset had 5 questions, all with
bare-word references compared against the model's natural full-sentence
answers - `exact_match` failed for *every* question, not just the one
case meant to illustrate that gap, which read as a mismatched dataset
rather than a deliberate lesson. The dataset now has exactly 2 questions,
chosen as a matched pair: one where a correct answer still fails exact
match (the bare-word-vs-full-sentence mismatch, kept deliberately this
time), and one phrased ("answer in one word") so a correct answer
matches exactly - giving a real example of each outcome instead of one
metric always failing across the board.

**A minimal pass/fail gate, not real CI.** The threshold-based gate is
the actual mechanic "regression testing" boils down to; wiring it into an
actual CI pipeline is an infrastructure concern this module doesn't need
to reproduce to demonstrate the idea.

---

## 13. Hands-on: metrics, judging, and hallucination detection over the Gemini API

### 13.1 Files

| File | Role |
|---|---|
| `metrics.py` | `exact_match()`, `rouge_l()` (hand-built LCS-based), `semantic_similarity()` (§4). |
| `judge.py` | `judge_pointwise()`, `judge_pairwise()` — LLM-as-a-judge over `gemini-3.1-flash-lite` (§5). |
| `hallucination.py` | `selfcheck_consistency()` — SelfCheckGPT-style sampling consistency check (§7). |
| `app.py` | `GOLDEN_DATASET` (§3) plus three demos: metrics + regression gate, LLM-as-a-judge (with a real bias check), and hallucination detection. |

### 13.2 Setup

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

The hallucination demo alone makes 10 real Gemini chat calls (5 samples
x 2 questions), so a full `python app.py` run takes a few minutes.

### 13.3 Step-by-step: what happens when you run `python app.py`

1. **Metrics + regression gate.** Each of the 2 golden-dataset questions
   is sent to Gemini at `temperature=0.0` with a fixed seed (for a
   reproducible run, not because every reference is expected to match
   verbatim); the real answer is scored with all three metrics against
   its reference. The average semantic similarity is compared against a
   threshold (`0.6`, calibrated from a real run — see §12) to produce a
   pass/fail gate, then the same check is run again against a hardcoded,
   deliberately wrong answer ("I don't know.") to prove the gate would
   actually catch a real regression, not just rubber-stamp whatever
   comes in.
2. **LLM-as-a-judge.** Gemini pointwise-scores a correct and an
   incorrect answer to the same question, then pairwise-compares the
   same two answers twice — once as (correct=A, incorrect=B), once with
   the order swapped — to check whether the verdict depends on which
   position an answer was shown in.
3. **Hallucination detection.** `selfcheck_consistency()` samples
   Gemini five times at `temperature=0.9` for a well-known fact
   ("capital of Japan") and for a genuinely obscure fact ("population of
   the smallest incorporated town in Wyoming"), embeds each set of
   answers, and reports the mean pairwise similarity for each.

### 13.4 What to expect

- The two questions deliberately land on opposite sides of exact match:
  the France question's reference is the bare phrase `"Paris."`, but
  the model's natural answer is a full sentence (`"The capital of
  France is Paris."`) — `exact_match=False` even though the answer is
  correct, while `rouge_l_f1` (`0.286`) and `semantic_sim` (`0.711`)
  both still give it partial-to-strong credit. The multiplication
  question explicitly asks the model to "answer in one word," so its
  natural answer (`"96"`) matches the reference exactly —
  `exact_match=True`, `rouge_l_f1=1.0`, `semantic_sim=1.0`. Seeing both
  outcomes side by side, on two genuinely correct answers, is the actual
  point: exact match's strictness is a property of the *metric*, not a
  sign the *answer* was wrong.
- The regression gate genuinely **passes** on the real run (avg
  similarity `0.855 >= 0.6`, pulled down from a perfect `1.0` by the
  France question's `0.711`) and genuinely **fails** on the hardcoded bad
  answer (`0.571`) — both outcomes are real, not scripted to always agree
  with each other. The margin on the bad answer is narrow: Gemini
  embeddings score even an unrelated short sentence around `0.5`-`0.6`
  against a reference, so a threshold for this metric has to sit above
  that floor rather than near zero.
- The judge's pointwise scores discriminate correctly (`5` for the
  correct answer, `1` for the incorrect one), and the pairwise
  comparison picked the correct answer both times regardless of which
  position it was shown in — **no position bias shows up for this
  clear-cut case**. That's an honest result, not evidence bias doesn't
  exist: §5 documents it as a known real risk, and it's far more likely
  to surface on closer, more subjective comparisons than on a
  factually unambiguous one like this.
- The well-known-fact question comes back **word-for-word identical**
  across all 5 samples (`mean_pairwise_similarity=1.0`, not flagged) —
  real evidence a grounded fact doesn't wobble under resampling. The
  "obscure" Wyoming question is *not* obscure to Gemini: all 5 samples
  name the same town (Lost Springs) and cite the same census figure, so
  its similarity also stays high (`~0.98`) and it is **not** flagged
  (`likely_hallucination=False`).
- **What that result teaches**: a SelfCheckGPT-style check measures
  *consistency*, not *correctness* — it only lights up when the model is
  genuinely unsure and its resamples diverge (a confident claim in one
  sample, a refusal or different number in another). A consistent answer
  can be consistently right, consistently wrong, or a consistent
  refusal, and this metric can't tell those apart. To see the
  low-similarity pattern, the test question has to sit beyond what the
  model reliably knows — which moves as models get stronger, so a good
  test question has to be chosen for the specific model under test.

### 13.5 Runtime dependencies

`metrics.py` and `hallucination.py` embed via Gemini's `gemini-embedding-001`
(`common/llm_client.py`'s `embed()`) and use `numpy` for cosine
similarity — `embed()` returns L2-normalized vectors, so dot product and
cosine similarity are numerically identical (Module 4's documented
convention). `judge.py` and `app.py` call `gemini-3.1-flash-lite` via
`common/llm_client.py`'s `chat()`, using `json_mode=True` for the judge's
structured score/verdict output.
