"""Minimal semantic search with FAISS.

Same idea as semantic_search_numpy.py, but the top-K nearest-neighbor
search is delegated to FAISS's IndexFlatIP (exact inner-product search),
which is the pattern you'd scale up for larger corpora.
"""

import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import embed

from corpus import CORPUS


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # inner product; == cosine similarity for normalized vectors
    index.add(embeddings)
    return index


def main() -> None:
    doc_embeddings = np.array(embed(CORPUS), dtype="float32")  # already unit vectors

    index = build_faiss_index(doc_embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dimension {index.d}")

    query = "What happens near an object with extremely strong gravity?"
    query_embedding = np.array(embed(query), dtype="float32")

    k = 3
    scores, indices = index.search(query_embedding, k)

    print(f"\nQuery: {query!r}")
    print(f"Top-{k} results:\n")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        print(f"  {rank}. (score={score:.3f}) {CORPUS[idx]}")


if __name__ == "__main__":
    main()
