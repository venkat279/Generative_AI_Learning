"""Module 11 mini application: a memory-augmented chat loop.

  short-term memory  -> ShortTermMemory: buffer window + summary buffer   (short_term_memory.py)
  long-term memory   -> LongTermMemory: embeddings in SQLite, top-K retrieval, consolidation, expiry   (long_term_memory.py)
  memory in practice -> extract_fact() below + respond(): retrieved memories injected into the system prompt

Run:
    python app.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import chat

from long_term_memory import LongTermMemory
from short_term_memory import ShortTermMemory

SYSTEM_PROMPT = "You are a helpful assistant with memory of past conversations."
MIN_RELEVANCE = 0.15  # similarity score threshold - retrieved memories below this are treated as noise, not injected

FACT_EXTRACT_PROMPT = (
    'The user said: "{message}"\n\n'
    "If this message reveals a personal fact about the user (their "
    "name, a like/dislike, a role, or a project), rewrite that fact as "
    'one short third-person sentence starting with "The user". If it '
    "reveals nothing personal, reply with exactly NONE."
)


def extract_fact(user_message: str) -> str | None:
    """Memory in practice: decide whether a turn is worth storing
    long-term, and if so, extract it as one clean sentence."""
    response = chat(
        [{"role": "user", "content": FACT_EXTRACT_PROMPT.format(message=user_message)}],
        temperature=0.0,
    )
    fact = response["message"]["content"].strip()
    return None if fact.upper().startswith("NONE") else fact


def respond(short_term: ShortTermMemory, long_term: LongTermMemory, user_message: str) -> str:
    retrieved = long_term.retrieve(user_message, k=3)
    relevant = [m for m in retrieved if m["score"] >= MIN_RELEVANCE]  # similarity score threshold
    if retrieved:
        print("  [long-term] retrieved:")
        for m in retrieved:
            flag = "" if m in relevant else "  (ignored - below threshold)"
            print(f"    {m['score']:.3f}  ({m['kind']}) {m['text']}{flag}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if relevant:
        memory_block = "\n".join(f"- {m['text']}" for m in relevant)
        messages.append(
            {
                "role": "system",
                "content": (
                    "Relevant memories about this user (treat only these as true; "
                    "do not invent, assume, or reference any other prior history):\n" + memory_block
                ),
            }
        )
    else:
        messages.append(
            {"role": "system", "content": "You have no stored memories of this user yet - do not claim any prior history with them."}
        )
    messages.extend(short_term.as_context_messages())
    messages.append({"role": "user", "content": user_message})

    response = chat(messages, temperature=0.4)
    reply = response["message"]["content"].strip()

    fact = extract_fact(user_message)
    if fact:
        stored = long_term.add(fact)
        print(f"  [long-term] {'stored' if stored else 'skipped (duplicate)'}: {fact}")

    short_term.add_turn(user_message, reply)
    return reply


def run_session(label: str, short_term: ShortTermMemory, long_term: LongTermMemory, turns: list[str]) -> None:
    print(f"\n{'=' * 70}\n{label}")
    for user_message in turns:
        print(f"\nUser: {user_message}")
        reply = respond(short_term, long_term, user_message)
        print(f"Assistant: {reply}")


def main() -> None:
    long_term = LongTermMemory()

    # Session 1: a fresh short-term buffer. window_turns=2 means
    # summarization kicks in on the 3rd turn, folding the oldest turn
    # into a running summary instead of just dropping it. Turns 1 and 3
    # each carry one durable fact, which extract_fact() should catch
    # and long_term.add() should persist; turn 2 is small talk with
    # nothing worth remembering.
    session_1 = ShortTermMemory(window_turns=2)
    run_session(
        "Session 1",
        session_1,
        long_term,
        [
            "Hi, I'm Sam.",
            "What's a good morning drink besides coffee?",
            "My favorite color is blue.",
        ],
    )

    # Memory expiry and freshness: store a short-lived, session-scoped
    # fact and show it being retrieved immediately, then excluded once
    # its TTL has actually passed.
    long_term.add("The user is away from their desk right now.", kind="ephemeral", ttl_seconds=2)
    print("\n[long-term] retrieval right after storing the ephemeral fact:")
    for m in long_term.retrieve("Is the user around right now?", k=5):
        print(f"    {m['score']:.3f}  ({m['kind']}) {m['text']}")
    time.sleep(3)
    print("\n[long-term] same retrieval 3s later, after the ephemeral fact's TTL expired:")
    for m in long_term.retrieve("Is the user around right now?", k=5):
        print(f"    {m['score']:.3f}  ({m['kind']}) {m['text']}")

    # Session 2: a brand-new ShortTermMemory (nothing carried over in
    # the buffer/summary), but the SAME long-term store - so the model
    # can only know Sam's name/favorite color if long-term retrieval
    # actually works across sessions ("user profile memory across
    # sessions").
    session_2 = ShortTermMemory(window_turns=2)
    run_session(
        "Session 2 (new short-term memory, same long-term store)",
        session_2,
        long_term,
        ["What's my name, and what's my favorite color?"],
    )


if __name__ == "__main__":
    main()
