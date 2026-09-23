"""Building a semantic search system (NumPy version).

Embeds the document corpus once, stores the embeddings as a NumPy array
on disk, then embeds a user query and retrieves the top-K most similar
documents using cosine similarity -- no vector database required.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import embed

from corpus import CORPUS

EMBEDDINGS_PATH = Path(__file__).parent / "corpus_embeddings.npy"


def build_and_save_embeddings() -> np.ndarray:
    embeddings = np.array(embed(CORPUS))  # already unit vectors -> dot product == cosine similarity
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]} to {EMBEDDINGS_PATH.name}")
    return embeddings


def load_embeddings() -> np.ndarray:
    return np.load(EMBEDDINGS_PATH)


def top_k_search(query: str, embeddings: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    query_embedding = np.array(embed(query)[0])

    # normalized vectors -> dot product IS cosine similarity
    scores = embeddings @ query_embedding
    top_indices = np.argsort(-scores)[:k]

    return [(CORPUS[i], float(scores[i])) for i in top_indices]


def main() -> None:
    embeddings = build_and_save_embeddings()
    embeddings = load_embeddings()  # demonstrates reading back from disk

    query = "How do animals find their way over long distances?"
    print(f"\nQuery: {query!r}")
    print(f"Top-3 results:\n")
    for rank, (doc, score) in enumerate(top_k_search(query, embeddings, k=3), start=1):
        print(f"  {rank}. (score={score:.3f}) {doc}")


if __name__ == "__main__":
    main()
