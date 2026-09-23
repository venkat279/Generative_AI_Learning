"""Augmentation and generation: RAG prompt template + a local LLM.

The "augmentation" step is the RAG prompt template itself: the retrieved
and reranked chunks are numbered and labeled with their source file, then
stuffed into a template that instructs the model to answer only from that
context and to cite which numbered chunk(s) it used - this is what turns
"passages that happen to be relevant" into "a grounded, checkable answer."

Generation calls Google's Gemini API (gemini-3.1-flash-lite).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import chat


def build_prompt(query: str, chunks: list[dict]) -> str:
    """RAG prompt template: numbered, source-labeled context + grounding
    instructions + the question."""
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(f"[{i}] (source: {chunk['source']})\n{chunk['text']}")
    context = "\n\n".join(context_blocks)

    return (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say you don't know. "
        "Cite the context items you used, like [1] or [2].\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def generate_answer(prompt: str, max_tokens: int = 200) -> str:
    response = chat(messages=[{"role": "user", "content": prompt}], num_predict=max_tokens)
    return response["message"]["content"].strip()
