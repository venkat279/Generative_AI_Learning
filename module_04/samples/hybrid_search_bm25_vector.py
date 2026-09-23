"""Hybrid search: BM25 (sparse, keyword-based) + vector search (dense, semantic).

BM25 is great at exact keyword/acronym matches; dense vector search is
great at paraphrases and synonyms. Hybrid search combines both score
lists so a query benefits from whichever signal is stronger.
"""

import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import embed

from corpus import CORPUS


def bm25_scores(query: str) -> np.ndarray:
    tokenized_corpus = [doc.lower().split() for doc in CORPUS]
    bm25 = BM25Okapi(tokenized_corpus)
    return np.array(bm25.get_scores(query.lower().split()))


def vector_scores(query: str) -> np.ndarray:
    doc_embeddings = np.array(embed(CORPUS))
    query_embedding = np.array(embed(query)[0])
    # embeddings are already unit-length -> dot product IS cosine similarity
    return doc_embeddings @ query_embedding


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    low, high = scores.min(), scores.max()
    if high - low < 1e-9:
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


def hybrid_search(query: str, alpha: float = 0.5, k: int = 3) -> list[tuple[str, float, float, float]]:
    """alpha weights the vector score; (1 - alpha) weights BM25."""
    sparse = min_max_normalize(bm25_scores(query))
    dense = min_max_normalize(vector_scores(query))
    combined = alpha * dense + (1 - alpha) * sparse

    top_indices = np.argsort(-combined)[:k]
    return [(CORPUS[i], combined[i], dense[i], sparse[i]) for i in top_indices]


def main() -> None:
    query = "GIL threading limitation in Python"  # exact keyword overlap -> BM25 should shine here

    print(f"Query: {query!r}\n")
    print(f"{'doc':<75} {'hybrid':>7} {'dense':>7} {'bm25':>7}")
    for doc, combined, dense, sparse in hybrid_search(query, alpha=0.5, k=3):
        print(f"{doc:<75} {combined:>7.3f} {dense:>7.3f} {sparse:>7.3f}")

    print("\nTry alpha=1.0 (pure vector) vs alpha=0.0 (pure BM25) to see each")
    print("signal's contribution change the ranking.")


if __name__ == "__main__":
    main()
