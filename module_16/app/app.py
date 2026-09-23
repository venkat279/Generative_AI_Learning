"""Module 16 hands-on: three demos over the Gemini API - reference-based
metrics against a golden dataset (with a regression-style pass/fail
gate), LLM-as-a-judge (pointwise + pairwise, including a real position-
bias check), and SelfCheckGPT-style hallucination detection.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import chat  # noqa: E402

from hallucination import selfcheck_consistency
from judge import judge_pairwise, judge_pointwise
from metrics import exact_match, rouge_l, semantic_similarity

# A tiny golden evaluation dataset - small enough to run against a
# live model on every call rather than needing to be versioned/stored
# separately. Two examples, deliberately chosen as a matched pair to show
# exact match's brittleness directly rather than by accident. The first
# question's reference is a bare phrase ("Paris.") while the model's
# natural answer is a full sentence ("The capital of France is Paris.")
# - exact_match is expected to be False here even though the answer is
# correct, while ROUGE-L and semantic similarity still give it credit.
# The second question explicitly asks the model to answer in one word,
# so its natural answer ("96") matches the reference exactly -
# exact_match is expected to be True here, giving the dataset one real
# example of each outcome instead of one metric always failing across
# the board.
GOLDEN_DATASET = [
    {"question": "What is the capital of France?", "reference": "Paris."},
    {"question": "What is 12 multiplied by 8? Answer in one word", "reference": "96"},
]

# Calibrated from a real run (per-question similarity: ~0.71 for the
# paraphrase case, 1.0 for the exact one-word case, averaging ~0.86) -
# below a healthy run's real average, and above the ~0.57 a non-answer
# like "I don't know." scores (see the regression check below); Gemini
# embeddings put even unrelated short sentences around 0.5-0.6.
REGRESSION_THRESHOLD = 0.6

# temperature=0 + a fixed seed makes each demo run reproducible, not
# because every reference is expected to match verbatim (only the
# "answer in one word" question is designed to do that) - just so the
# same run doesn't silently produce different scores on a re-run.
EVAL_TEMPERATURE = 0.0
EVAL_SEED = 42


def demo_metrics_eval() -> None:
    print("--- Reference-based metrics over a golden dataset ---")
    similarities = []
    for item in GOLDEN_DATASET:
        response = chat(
            [{"role": "user", "content": item["question"]}],
            temperature=EVAL_TEMPERATURE,
            seed=EVAL_SEED,
        )
        candidate = response["message"]["content"].strip()

        em = exact_match(candidate, item["reference"])
        rl = rouge_l(candidate, item["reference"])
        sim = semantic_similarity(candidate, item["reference"])
        similarities.append(sim)

        print(f"Q: {item['question']}")
        print(f"  reference: {item['reference']!r}")
        print(f"  model:     {candidate[:80]!r}")
        print(f"  exact_match={em}  rouge_l_f1={rl['f1']}  semantic_sim={sim}")

    avg_similarity = round(sum(similarities) / len(similarities), 3)
    gate = "PASS" if avg_similarity >= REGRESSION_THRESHOLD else "FAIL"
    print(f"\nRegression gate: avg semantic similarity {avg_similarity} >= {REGRESSION_THRESHOLD} -> {gate}")

    # Prove the gate actually discriminates, not just rubber-stamps: a
    # plausible regression (the model degrading to a non-answer) against
    # the same reference should fail it.
    regressed_candidate = "I don't know."
    regressed_reference = GOLDEN_DATASET[0]["reference"]
    regressed_score = semantic_similarity(regressed_candidate, regressed_reference)
    regressed_gate = "PASS" if regressed_score >= REGRESSION_THRESHOLD else "FAIL"
    print(
        f"Regression check (hypothetical regressed answer {regressed_candidate!r} vs "
        f"{regressed_reference!r}): similarity={regressed_score} -> {regressed_gate}"
    )


def demo_llm_judge() -> None:
    print("\n--- LLM-as-a-judge demo ---")
    question = "What is the capital of Australia?"
    good_answer = "The capital of Australia is Canberra."
    bad_answer = "The capital of Australia is Sydney."

    print(f"pointwise [correct answer]:   {judge_pointwise(question, good_answer)}")
    print(f"pointwise [incorrect answer]: {judge_pointwise(question, bad_answer)}")

    result_ab = judge_pairwise(question, good_answer, bad_answer)
    result_ba = judge_pairwise(question, bad_answer, good_answer)
    print(f"pairwise (good=A, bad=B): {result_ab}")
    print(f"pairwise (bad=A, good=B): {result_ba}")

    consistent = result_ab["winner"] == "A" and result_ba["winner"] == "B"
    print(f"judge picked the correct answer both times regardless of position: {consistent}")


def demo_hallucination_detection() -> None:
    print("\n--- Hallucination detection (SelfCheckGPT-style) demo ---")
    factual_question = "What is the capital of Japan?"
    obscure_question = "What is the exact population of the smallest incorporated town in Wyoming?"

    for label, question in [("well-known fact", factual_question), ("obscure fact", obscure_question)]:
        result = selfcheck_consistency(question, n_samples=5, temperature=0.9)
        print(f"\n[{label}] {question!r}")
        print(f"  mean_pairwise_similarity={result['mean_pairwise_similarity']} "
              f"likely_hallucination={result['likely_hallucination']}")
        for answer in result["answers"]:
            print(f"    - {answer[:90]!r}")


def main() -> None:
    demo_metrics_eval()
    demo_llm_judge()
    demo_hallucination_detection()


if __name__ == "__main__":
    main()
