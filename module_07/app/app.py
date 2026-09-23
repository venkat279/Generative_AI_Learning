"""Module 7 mini application: function calling / tool use with
Google's Gemini API (gemini-3.1-flash-lite).

Demonstrates the full tool-call lifecycle:
  1. Register a tool (calculate) and send it alongside the user's question.
  2. Detect whether the model asked to call the tool.
  3. Execute the tool in code - the model never runs anything itself.
  4. Send the tool's result back as a `tool` role message.
  5. Get the model's final answer, now grounded in the exact result.

Run:
    python app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import chat

from calculator_tool import TOOL_SCHEMA, calculate

TOOLS = [TOOL_SCHEMA]

QUESTIONS = [
    "What is 847 times 293?",
    "If I split a bill of $138.50 evenly between 5 people, how much does each person pay?",
    "What is the capital of France?",
]


def ask(question: str) -> None:
    print(f"\n{'=' * 70}\nQ: {question}")

    messages = [{"role": "user", "content": question}]
    response = chat(messages, tools=TOOLS)
    message = response["message"]

    tool_calls = message.get("tool_calls")
    if not tool_calls:
        print(f"Answer (no tool needed): {message['content']}")
        return

    messages.append(message)
    for call in tool_calls:
        name = call["function"]["name"]
        args = call["function"]["arguments"]
        print(f"Tool call: {name}({args})")

        if name == "calculate":
            result = calculate(**args)
        else:
            result = f"error: unknown tool '{name}'"

        print(f"Tool result: {result}")
        messages.append({"role": "tool", "content": result})

    final = chat(messages, tools=TOOLS)
    print(f"Answer: {final['message']['content']}")


def main() -> None:
    for question in QUESTIONS:
        ask(question)


if __name__ == "__main__":
    main()
