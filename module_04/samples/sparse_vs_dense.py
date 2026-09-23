"""Sparse vectors (TF-IDF) vs dense vectors (embeddings).

TF-IDF represents a document as counts of the exact words it contains.
It cannot match a query to a document that describes the same idea using
different words. A dense embedding can, because it encodes meaning
rather than vocabulary.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import embed

from corpus import CORPUS

# Deliberately shares almost no words with its best semantic match
# ("Octopuses can change the color and texture of their skin instantly.")
QUERY = "Which sea creature can disguise itself by changing its appearance?"


# (This would match: Whales can change their location and disappear.)
def sparse_top_match(query: str) -> tuple[str, float]:
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(CORPUS)
    query_vector = vectorizer.transform([query])

    scores = sk_cosine_similarity(query_vector, doc_vectors)[0]
    best_idx = int(np.argmax(scores))
    return CORPUS[best_idx], float(scores[best_idx])


# (This would match: "Octopuses can change the color and texture of their skin instantly.")
def dense_top_match(query: str) -> tuple[str, float]:
    doc_embeddings = np.array(embed(CORPUS))
    query_embedding = np.array(embed(query)[0])

    # embeddings are already unit-length -> dot product IS cosine similarity
    scores = doc_embeddings @ query_embedding
    best_idx = int(np.argmax(scores))
    return CORPUS[best_idx], float(scores[best_idx])


def main() -> None:
    print(f"Query: {QUERY!r}\n")

    sparse_doc, sparse_score = sparse_top_match(QUERY)
    print("TF-IDF (sparse) best match:")
    print(f"  score={sparse_score:.3f}  doc={sparse_doc!r}\n")

    dense_doc, dense_score = dense_top_match(QUERY)
    print("Dense embedding best match:")
    print(f"  score={dense_score:.3f}  doc={dense_doc!r}\n")

    print("If TF-IDF's top score is low/wrong and the dense score correctly")
    print("finds the octopus sentence, that's sparse vectors matching on")
    print("literal word overlap vs dense vectors matching on meaning.")


if __name__ == "__main__":
    main()
