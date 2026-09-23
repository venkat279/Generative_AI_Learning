"""Multi-turn conversation with a sliding window.

Calls Google's Gemini API (gemini-3.1-flash-lite) for real replies. The
sliding-window history trimming below uses tiktoken as an 
approximate token counter for budgeting purposes (it's not Gemini's 
exact tokenizer, but close enough to demonstrate the trimming behavior).
"""

import sys
from pathlib import Path

from tiktoken import get_encoding

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import chat

enc = get_encoding("cl100k_base")


class ChatSession:
    """Keeps history bounded by a token budget using a sliding window,
    always preserving the system message."""

    def __init__(self, system_prompt: str, max_history_tokens: int = 2000):
        self.system_prompt = system_prompt
        self.max_history_tokens = max_history_tokens
        self.history: list[dict] = []

    def _tokens_in(self, messages: list[dict]) -> int:
        return sum(len(enc.encode(m["content"])) for m in messages)

    def _trim_history(self) -> None:
        while self.history and self._tokens_in(self.history) > self.max_history_tokens:
            self.history.pop(0)  # drop oldest turn first

    def send(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        self._trim_history()

        messages = [{"role": "system", "content": self.system_prompt}, *self.history]
        response = chat(messages)
        reply = response["message"]["content"].strip()

        self.history.append({"role": "assistant", "content": reply})
        self._trim_history()
        return reply


if __name__ == "__main__":
    session = ChatSession(system_prompt="You are a concise data-science tutor.", max_history_tokens=500)
    print(session.send("What's the difference between precision and recall?"))
    print(session.send("Give a one-line example of when recall matters more."))
