"""System prompt engineering: persona, scope, out-of-scope handling.

Calls Google's Gemini API (gemini-3.1-flash-lite) with a persona/scope-constraining
system prompt, and shows the model's real behavior on in-scope vs
out-of-scope questions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import chat

SUPPORT_BOT_SYSTEM_PROMPT = """You are "OrderBot", a customer support assistant for an online bookstore.

Persona & tone: friendly, concise, professional.
Scope: order status, returns policy, and book recommendations for fiction/non-fiction.
Limitations:
- Do not discuss topics unrelated to the bookstore (politics, other companies, personal opinions).
- If asked something out of scope, reply exactly: "I can only help with bookstore orders, returns, and recommendations."
- Never invent an order status you don't have data for; say you'll escalate to a human instead.
"""


def ask_bot(user_message: str) -> str:
    response = chat(messages=[
        {"role": "system", "content": SUPPORT_BOT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])
    return response["message"]["content"].strip()


if __name__ == "__main__":
    print(ask_bot("What's your return policy on paperbacks?"))
    print(ask_bot("What do you think about the last election?"))  # should be refused
