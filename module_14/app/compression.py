"""Extractive prompt compression: score each sentence by how many 
high-frequency, non-stopword terms it shares with the rest of the text, 
and keep only the top-scoring sentences, in original order - a minimal, 
from-scratch relative of LLMLingua's approach.
"""

import re
from collections import Counter

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "to", "is", "are",
    "was", "were", "it", "this", "that", "for", "with", "as", "at", "by", "be",
    "has", "have", "had", "its", "over", "into", "than", "then",
}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _content_words(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"\w+", text) if w.lower() not in STOPWORDS]


def compress(text: str, keep_ratio: float = 0.5) -> str:
    sentences = _sentences(text)
    if len(sentences) <= 1:
        return text

    freq = Counter(_content_words(text))

    def score(sentence: str) -> float:
        words = _content_words(sentence)
        return sum(freq[w] for w in words) / max(len(words), 1)

    keep_n = max(1, round(len(sentences) * keep_ratio))
    top_indices = sorted(range(len(sentences)), key=lambda i: score(sentences[i]), reverse=True)[:keep_n]
    kept_in_order = sorted(top_indices)
    return " ".join(sentences[i] for i in kept_in_order)
