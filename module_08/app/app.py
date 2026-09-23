"""Module 8 mini application: a LangChain-orchestrated pipeline over
Google's Gemini API (gemini-3.1-flash-lite).

Two composed steps for each support message:
  1. classify_chain - extract category + urgency as structured JSON.
  2. respond_chain  - draft a reply, using step 1's output as input.

This is the same "classify, then act on the classification" shape as
Module 3's structured-output work and Module 7's tool use, but built
with LangChain's building blocks (ChatPromptTemplate,
ChatGoogleGenerativeAI, an output parser, and the `|` pipe operator to
compose steps) instead of the raw HTTP `chat()` calls in
common/llm_client.py (only the model object comes from there, via
langchain_chat_model()) - the actual point of an "orchestration
frameworks" module: less glue code to compose two
model calls into one pipeline, with structured data flowing between
them.

Run:
    python app.py
"""

import sys
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import langchain_chat_model  # noqa: E402

llm = langchain_chat_model(temperature=0.3)
json_llm = langchain_chat_model(temperature=0.3, response_mime_type="application/json")

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You triage customer support messages. Respond with ONLY valid JSON "
            'matching this schema: {{"category": "billing" | "technical" | "account" | "shipping", '
            '"urgency": "low" | "medium" | "high"}}',
        ),
        ("user", "{message}"),
    ]
)
classify_chain = CLASSIFY_PROMPT | json_llm | JsonOutputParser()

RESPOND_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a support agent. Write a short, professional reply to the "
            "customer's message below. Category: {category}. Urgency: {urgency}.",
        ),
        ("user", "{message}"),
    ]
)
respond_chain = RESPOND_PROMPT | llm | StrOutputParser()


def handle_message(message: str) -> dict:
    classification = classify_chain.invoke({"message": message})
    reply = respond_chain.invoke(
        {
            "message": message,
            "category": classification.get("category", "unknown"),
            "urgency": classification.get("urgency", "unknown"),
        }
    )
    return {"classification": classification, "reply": reply}


MESSAGES = [
    "My card was charged twice for the same order and I need this fixed today.",
    "Just wondering what your return policy is on paperbacks.",
]


def main() -> None:
    for message in MESSAGES:
        print(f"\n{'=' * 70}\nMessage: {message}")
        result = handle_message(message)
        print(f"Classification: {result['classification']}")
        print(f"Reply: {result['reply']}")


if __name__ == "__main__":
    main()
