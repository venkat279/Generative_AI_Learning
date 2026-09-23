"""Understanding Embeddings.

An embedding is a fixed-length vector of numbers that represents the
*meaning* of a piece of text. Sentences with similar meaning end up with
vectors that point in similar directions, even if they don't share any
words.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import embed

from corpus import MODEL_NAME


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    sentence_a = "The cat sat on the mat."
    sentence_b = "A feline was resting on the rug."   # same meaning, different words
    sentence_c = "The stock market fell sharply today."  # unrelated meaning

    embeddings = np.array(embed([sentence_a, sentence_b, sentence_c]))

    print(f"Model: {MODEL_NAME}")
    print(f"Embedding shape per sentence: {embeddings[0].shape}  <- this is the embedding dimension")
    print()

    sim_ab = cosine_similarity(embeddings[0], embeddings[1])
    sim_ac = cosine_similarity(embeddings[0], embeddings[2])

    print(f"'{sentence_a}'  vs  '{sentence_b}'")
    print(f"  -> cosine similarity = {sim_ab:.3f}  (paraphrase: expect HIGH)")
    print()
    print(f"'{sentence_a}'  vs  '{sentence_c}'")
    print(f"  -> cosine similarity = {sim_ac:.3f}  (unrelated topic: expect LOW)")


if __name__ == "__main__":
    main()
