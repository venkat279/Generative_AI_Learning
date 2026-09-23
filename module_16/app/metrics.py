"""Reference-based metrics for text generation: exact match, a 
from-scratch ROUGE-L (LCS-based), and an embedding-based semantic 
similarity standing in for BERTScore. 
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import embed  # noqa: E402


def exact_match(candidate: str, reference: str) -> bool:
    return candidate.strip().lower() == reference.strip().lower()


def _lcs_length(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(a)][len(b)]


def rouge_l(candidate: str, reference: str) -> dict:
    """Precision/recall/F1 over the longest common (in-order) token
    subsequence - the actual ROUGE-L mechanic, not just a word-overlap
    ratio (Module 15's cheaper faithfulness proxy)."""
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()
    if not cand_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(cand_tokens, ref_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def semantic_similarity(candidate: str, reference: str) -> float:
    """Cosine similarity between embeddings - the same embed-then-compare
    mechanic Modules 4/6/11/14 use, applied to reference-based grading
    instead of retrieval. embed() returns unit-length vectors, so dot
    product == cosine similarity (documented in Module 4)."""
    vectors = embed([candidate, reference])
    a, b = np.array(vectors[0]), np.array(vectors[1])
    return round(float(np.dot(a, b)), 3)
