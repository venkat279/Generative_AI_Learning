"""Module 6 mini application: a local RAG pipeline over a small engineering
knowledge base (5 markdown docs + 1 PDF in docs/).

Pipeline for each question:
  index (once)  -> ingest -> chunk -> embed -> store        (ingest.py)
  retrieve      -> embed query -> top-K cosine similarity   (retrieve.py)
  augment       -> build a grounded, source-cited prompt    (generate.py)
  generate      -> local LLM answers from that prompt       (generate.py)

Run:
    python app.py
"""

from ingest import build_index
from retrieve import top_k_retrieve
from generate import build_prompt, generate_answer

RETRIEVE_K = 3

QUESTIONS = [
    "What should I do if there's a production incident?",
    "How do I set up my dev environment in my first week?",
    "Can I merge my own pull request without a review?",
    "How often are API keys rotated?",
    "How do whales communicate?"
]


def answer_question(query: str) -> None:
    print(f"\n{'=' * 70}\nQ: {query}")

    retrieved = top_k_retrieve(query, k=RETRIEVE_K)
    print(f"\nRetrieved (top {RETRIEVE_K} by embedding similarity):")
    for chunk in retrieved:
        print(f"  {chunk['retrieval_score']:.3f}  {chunk['id']}")

    prompt = build_prompt(query, retrieved)
    answer = generate_answer(prompt)

    sources = ", ".join(f"[{i}] {c['source']}" for i, c in enumerate(retrieved, start=1))
    print(f"\nAnswer: {answer}")
    print(f"Sources: {sources}")


def main() -> None:
    build_index()
    for question in QUESTIONS:
        answer_question(question)


if __name__ == "__main__":
    main()
