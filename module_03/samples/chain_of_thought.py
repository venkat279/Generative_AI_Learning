"""Chain-of-Thought prompting.

Calls Google's Gemini API (gemini-3.1-flash-lite) for both a direct answer 
and a Chain-of-Thought answer to the same question.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import chat

QUESTION = (
    "My card was charged twice for the same order."
    "Classify this as one of: billing, payment, account and shipping."
)

COT_QUESTION = """Classify the message.

Before choosing a category, identify the main issue and briefly justify
why the selected category fits.

Categories:
- billing
- payment
- account
- shipping

Message:
"My card was charged twice for the same order."

Return:
Category: <category>
Reason: <one-sentence justification>"""


def direct_answer(question: str) -> str:
    response = chat(messages=[{"role": "user", "content": f"{question}"}])
    return response["message"]["content"].strip()


def cot_answer(question: str) -> str:
    response = chat(messages=[{"role": "user", "content": f"{question}"}])
    return response["message"]["content"].strip()


if __name__ == "__main__":
    print("-- Direct answer --")
    print(direct_answer(QUESTION))

    print("\n-- Chain-of-Thought --")
    print(cot_answer(COT_QUESTION))
