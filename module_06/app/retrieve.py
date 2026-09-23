"""Retrieval pipeline: embed the query, then top-K retrieval.

Deliberately the simplest useful retrieval strategy - no similarity
threshold, no metadata pre/post-filtering, no multi-query or Self-RAG/CRAG
loops. Just: embed the query, rank every stored chunk by cosine
similarity, return the K best.
"""

import numpy as np

from embedder import embed
from ingest import load_index


def top_k_retrieve(query: str, k: int = 8) -> list[dict]:
    chunks, vectors = load_index()

    query_vector = embed([query])[0]
    scores = vectors @ query_vector  # vectors are L2-normalized -> dot product == cosine similarity

    top_indices = np.argsort(-scores)[:k]
    results = []
    for idx in top_indices:
        chunk = dict(chunks[idx])
        chunk["retrieval_score"] = float(scores[idx])
        results.append(chunk)
    return results
