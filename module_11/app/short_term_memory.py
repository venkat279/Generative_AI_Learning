"""Short-term memory management: conversation history as a messages
list, kept bounded with a buffer-window + summarization hybrid (a
"summary buffer").

Extends Module 2's `ChatSession` sliding window: instead of just
dropping the oldest turns once a budget is exceeded, the dropped turns
are folded into a running summary first via one extra LLM call, so
nothing is silently lost.
"""

import sys
from pathlib import Path

from tiktoken import get_encoding

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import chat

enc = get_encoding("cl100k_base")

SUMMARY_PROMPT = (
    "Update the running summary of this conversation with the new "
    "exchange below. Keep it to 2-3 sentences, preserving any durable "
    "facts or decisions.\n\nCurrent summary: {summary}\n\n"
    "New exchange:\nUser: {user}\nAssistant: {assistant}"
)


class ShortTermMemory:
    """Buffer window (last `window_turns` kept verbatim) + a running
    summary of anything older, bounded by both a turn count and a
    token budget (token-aware truncation)."""

    def __init__(self, window_turns: int = 3, max_buffer_tokens: int = 400):
        self.window_turns = window_turns
        self.max_buffer_tokens = max_buffer_tokens
        self.buffer: list[dict] = []  # verbatim recent messages
        self.summary: str = ""  # running summary of older turns

    def _tokens_in(self, messages: list[dict]) -> int:
        return sum(len(enc.encode(m["content"])) for m in messages)

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        self.buffer.append({"role": "user", "content": user_message})
        self.buffer.append({"role": "assistant", "content": assistant_message})

        turns = len(self.buffer) // 2
        while turns > self.window_turns or self._tokens_in(self.buffer) > self.max_buffer_tokens:
            oldest_user = self.buffer.pop(0)
            oldest_assistant = self.buffer.pop(0)
            self._fold_into_summary(oldest_user["content"], oldest_assistant["content"])
            turns = len(self.buffer) // 2

    def _fold_into_summary(self, user_message: str, assistant_message: str) -> None:
        prompt = SUMMARY_PROMPT.format(summary=self.summary or "(none yet)", user=user_message, assistant=assistant_message)
        response = chat([{"role": "user", "content": prompt}], temperature=0.2)
        self.summary = response["message"]["content"].strip()
        print(f"  [short-term] folded oldest turn into summary: {self.summary}")

    def as_context_messages(self) -> list[dict]:
        """This session's short-term memory, rendered as messages to
        prepend before the current user turn."""
        messages = []
        if self.summary:
            messages.append({"role": "system", "content": f"Summary of earlier conversation: {self.summary}"})
        messages.extend(self.buffer)
        return messages
