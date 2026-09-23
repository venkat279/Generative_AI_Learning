"""Module 14 hands-on: three short demos over the Gemini API -
exact + semantic caching, extractive prompt compression, and
complexity-based model routing with cascade escalation.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import chat  # noqa: E402

import tiktoken

from compression import compress
from exact_cache import ExactCache
from router import route
from semantic_cache import SemanticCache

ENCODING = tiktoken.get_encoding("cl100k_base")  # approximate tokenizer, same caveat as Module 2/13


def answer(prompt: str) -> str:
    response = chat([{"role": "user", "content": prompt}])
    return response["message"]["content"]


def demo_caching() -> None:
    exact = ExactCache(ttl_seconds=300)
    semantic = SemanticCache(similarity_threshold=0.90, ttl_seconds=300)

    queries = [
        "What is the capital of France?",
        "What is the capital of France?",  # exact repeat -> exact-cache hit
        "What's the capital city of France?",  # paraphrase -> semantic-cache hit
        "What is the capital of Germany?",  # genuinely new -> real LLM call
    ]

    print("--- Caching demo ---")
    for q in queries:
        t0 = time.monotonic()
        cached = exact.get(q)
        if cached is not None:
            source, result = "exact-cache", cached
        else:
            sem = semantic.get(q)
            if sem is not None:
                value, score, matched_prompt = sem
                source, result = f"semantic-cache (sim={score:.3f} vs {matched_prompt!r})", value
            else:
                result = answer(q)
                exact.set(q, result)
                semantic.set(q, result)
                source = "llm call"
        elapsed_ms = (time.monotonic() - t0) * 1000
        print(f"[{source}] ({elapsed_ms:.0f}ms) {q!r} -> {result[:80]!r}")

    print(f"\nexact cache hit rate: {exact.hit_rate}")
    print(f"semantic cache hit rate: {semantic.hit_rate}")


def demo_compression() -> None:
    long_context = (
        "The customer reported that the checkout page fails intermittently. "
        "Our engineering team investigated the checkout page failure over several days. "
        "The root cause was traced to a race condition in the payment service. "
        "Weather in the office was pleasant this week, unrelated to the incident. "
        "The payment service race condition was fixed by adding a database lock. "
        "The fix for the payment service has been deployed to production. "
        "Team lunch on Friday was well attended by everyone in engineering. "
        "Monitoring now tracks checkout page failures in real time going forward."
    )
    compressed = compress(long_context, keep_ratio=0.5)

    print("\n--- Prompt compression demo ---")
    print(f"original:   {len(ENCODING.encode(long_context))} tokens")
    print(f"compressed: {len(ENCODING.encode(compressed))} tokens")
    print(f"compressed text: {compressed}")


def demo_routing() -> None:
    print("\n--- Model routing / cascading demo ---")
    queries = [
        "What is the capital of Japan?",
        "Define recursion in programming.",  # short enough to classify "simple", but the natural answer runs past the cheap tier's token cap -> triggers escalation
        "Give me a three-step plan to improve API latency for a GenAI backend, with tradeoffs for each step.",
    ]
    for q in queries:
        result = route(q)
        tag = result["tier"] + (" (escalated)" if result["escalated"] else "")
        print(f"[{tag}] {q!r} -> {result['answer'][:150]!r}")


def main() -> None:
    demo_caching()
    demo_compression()
    demo_routing()


if __name__ == "__main__":
    main()
