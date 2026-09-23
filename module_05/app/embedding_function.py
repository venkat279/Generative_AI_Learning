"""A minimal, dependency-light embedding function for the ChromaDB mini app.

ChromaDB's built-in default embedding function downloads and runs an ONNX
model (same family of problem as Module 4's sentence-transformers usage).
To keep this mini app's *logic* independent of that model-download/runtime
path, it instead uses a scikit-learn hashing vectorizer: deterministic,
no model download, no PyTorch/ONNX Runtime involved.

Trade-off: this is a bag-of-words hashing embedding, not a real semantic
model - two sentences that mean the same thing but share no words will
NOT score as similar. It exists to demonstrate ChromaDB's own features
(collections, metadata filtering, persistence, querying) rather than to
demonstrate embedding quality, which Module 4 already covers.
"""

from sklearn.feature_extraction.text import HashingVectorizer

from chromadb import Documents, EmbeddingFunction, Embeddings


class SimpleHashingEmbeddingFunction(EmbeddingFunction[Documents]):
    """A ChromaDB-compatible embedding function with no ML-runtime dependency."""

    def __init__(self, n_features: int = 512):
        self._n_features = n_features
        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
        )

    def __call__(self, input):
        return self._vectorizer.transform(input).toarray().tolist()

    def name(self) -> str:
        return "simple_hashing_embedding_function"
