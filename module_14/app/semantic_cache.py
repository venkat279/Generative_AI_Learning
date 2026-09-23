"""Semantic caching: embed each query via Gemini's gemini-embedding-001 and compare 
cosine similarity against previously cached queries, instead of requiring 
an exact string match - the same embed+cosine mechanic Modules 4/6/11 use, 
applied to cache lookups instead of documents/facts.
GPTCache/Redis-with-vector-extension are the named production tools; this
hand-builds the same idea locally.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import embed  # noqa: E402

# Gemini's embedding task type for comparing two texts' meaning - it spreads
# paraphrases (~0.98+) well apart from merely related queries (~0.86-0.88),
# which is what a single similarity threshold needs.
EMBED_TASK = "SEMANTIC_SIMILARITY"


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.90, ttl_seconds: int = 300):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self._entries: list[dict] = []
        self.hits = 0
        self.misses = 0

    def _evict_expired(self) -> None:
        now = time.time()
        self._entries = [e for e in self._entries if now - e["stored_at"] <= self.ttl_seconds]

    def get(self, prompt: str) -> tuple[str, float, str] | None:
        self._evict_expired()
        if not self._entries:
            self.misses += 1
            return None

        # embed() returns unit-length vectors, so dot product == cosine similarity.
        query_vec = np.array(embed(prompt, task_type=EMBED_TASK)[0])
        best = max(self._entries, key=lambda e: float(np.dot(query_vec, e["embedding"])))
        score = float(np.dot(query_vec, best["embedding"]))

        if score >= self.similarity_threshold:
            self.hits += 1
            return best["value"], score, best["prompt"]
        self.misses += 1
        return None

    def set(self, prompt: str, value: str) -> None:
        vec = np.array(embed(prompt, task_type=EMBED_TASK)[0])
        self._entries.append({"prompt": prompt, "embedding": vec, "value": value, "stored_at": time.time()})

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 3) if total else 0.0
