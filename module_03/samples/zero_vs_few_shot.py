"""Zero-shot vs few-shot prompting.

Calls Google's Gemini API (gemini-3.1-flash-lite) for both prompting styles, so
this shows a real model's actual behavior change between zero-shot and
few-shot prompting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import chat


def zero_shot(review: str) -> str:
    prompt = f"Classify the sentiment of this review as Positive, Negative, or Neutral:\n\n{review}"
    response = chat(messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()


def few_shot(review: str) -> str:
    prompt = f"""Classify the sentiment as Positive, Negative, or Neutral.

Review: "The battery lasts all day and the camera is amazing."
Sentiment: Positive

Review: "It stopped working after two days, total waste of money."
Sentiment: Negative

Review: "It does what it says on the box, nothing more."
Sentiment: Neutral

Review: "{review}"
Sentiment:"""
    response = chat(messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()


if __name__ == "__main__":
    review = "Shipping was fast but the product feels cheaply made."
    print("Zero-shot:", zero_shot(review))
    print("Few-shot :", few_shot(review))
