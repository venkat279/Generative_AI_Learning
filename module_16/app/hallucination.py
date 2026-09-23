"""Hallucination detection via a hand-built SelfCheckGPT-style 
sampling consistency check: ask the same question multiple times
at a high temperature, embed each answer, and measure how much the
answers agree with each other. A well-grounded factual answer tends to
come back nearly identical across resamples; a question the model
doesn't reliably know the answer to - an obscure fact, or something
fabricated outright - tends to produce a mix of response *shapes*
(a confident specific claim here, a refusal or hedge there, a
clarifying question elsewhere) that vary sample to sample, since
there's no real fact anchoring the response, only whatever the model
improvises that particular sample.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import chat, embed  # noqa: E402

INCONSISTENCY_THRESHOLD = 0.8


def _sample_answers(question: str, n_samples: int, temperature: float) -> list[str]:
    answers = []
    for _ in range(n_samples):
        response = chat([{"role": "user", "content": question}], temperature=temperature)
        answers.append(response["message"]["content"].strip())
    return answers


def selfcheck_consistency(question: str, n_samples: int = 5, temperature: float = 0.9) -> dict:
    answers = _sample_answers(question, n_samples, temperature)
    vectors = [np.array(v) for v in embed(answers)]

    pair_scores = [
        float(np.dot(vectors[i], vectors[j])) for i in range(len(vectors)) for j in range(i + 1, len(vectors))
    ]
    mean_similarity = round(sum(pair_scores) / len(pair_scores), 3) if pair_scores else 1.0

    return {
        "answers": answers,
        "mean_pairwise_similarity": mean_similarity,
        "likely_hallucination": mean_similarity < INCONSISTENCY_THRESHOLD,
    }
