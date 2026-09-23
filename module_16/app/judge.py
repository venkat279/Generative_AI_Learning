"""LLM-as-a-Judge: use gemini-3.1-flash-lite itself to score a single response 
(pointwise) or pick the better of two (pairwise) - the same model 
that generates the answers, not a separate "stronger" judge model.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import chat  # noqa: E402

POINTWISE_PROMPT = """You are an impartial judge evaluating the quality of an answer to a question.

Question: {question}
Answer: {answer}

Score the answer's correctness and helpfulness from 1 (very poor) to 5 (excellent).
Respond ONLY with JSON in this exact shape: {{"score": <1-5 integer>, "rationale": "<one sentence>"}}"""

PAIRWISE_PROMPT = """You are an impartial judge comparing two candidate answers to the same question.

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}

Decide which answer is more correct and helpful. If they are truly equal, say "tie".
Respond ONLY with JSON in this exact shape: {{"winner": "A" | "B" | "tie", "rationale": "<one sentence>"}}"""


def judge_pointwise(question: str, answer: str) -> dict:
    prompt = POINTWISE_PROMPT.format(question=question, answer=answer)
    response = chat([{"role": "user", "content": prompt}], temperature=0.0, json_mode=True)
    return json.loads(response["message"]["content"])


def judge_pairwise(question: str, answer_a: str, answer_b: str) -> dict:
    prompt = PAIRWISE_PROMPT.format(question=question, answer_a=answer_a, answer_b=answer_b)
    response = chat([{"role": "user", "content": prompt}], temperature=0.0, json_mode=True)
    return json.loads(response["message"]["content"])
