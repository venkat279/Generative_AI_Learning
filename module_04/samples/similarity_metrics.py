"""Similarity and distance metrics.

Computes cosine similarity, dot product, and Euclidean (L2) distance by
hand with numpy (no black-box helper) for a "similar" pair and a
"dissimilar" pair, so you can see how each metric behaves.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import embed

from corpus import CORPUS, MODEL_NAME


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def report(label: str, a: np.ndarray, b: np.ndarray) -> None:
    print(f"{label}")
    print(f"  cosine similarity : {cosine_similarity(a, b):.4f}  (1 = identical direction, 0 = unrelated, -1 = opposite)")
    print(f"  dot product       : {dot_product(a, b):.4f}  (also grows with vector magnitude, not just direction)")
    print(f"  euclidean distance: {euclidean_distance(a, b):.4f}  (0 = identical, larger = farther apart)")
    print()


def main() -> None:
    embeddings = np.array(embed(CORPUS))

    similar_pair = (0, 1)     # two Python sentences
    different_pair = (0, 8)   # Python sentence vs dolphin sentence

    print(f"A: '{CORPUS[similar_pair[0]]}'")
    print(f"B: '{CORPUS[similar_pair[1]]}'")
    report("Similar pair (same topic)", embeddings[similar_pair[0]], embeddings[similar_pair[1]])

    print(f"A: '{CORPUS[different_pair[0]]}'")
    print(f"B: '{CORPUS[different_pair[1]]}'")
    report("Different pair (different topic)", embeddings[different_pair[0]], embeddings[different_pair[1]])

    print(f"Note: {MODEL_NAME} vectors are L2-normalized by llm_client.embed(), so they")
    print("come back unit-length, so dot product and cosine")
    print("similarity are numerically identical here -- euclidean distance still")
    print("differs since it measures absolute distance, not just direction.")


if __name__ == "__main__":
    main()
