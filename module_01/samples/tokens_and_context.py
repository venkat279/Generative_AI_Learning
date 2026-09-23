"""Tokens and context windows.

LLMs don't see characters or words - they see tokens. tiktoken lets us
inspect exactly how a string gets tokenized, which is the first step to
reasoning about context windows and cost.
"""

import tiktoken


def explore_tokens(text: str, encoding_name: str = "cl100k_base") -> None:
    enc = tiktoken.get_encoding(encoding_name)
    token_ids = enc.encode(text)

    print(f"Text: {text!r}")
    print(f"Token count: {len(token_ids)}")
    print("Token breakdown:")
    for tid in token_ids:
        piece = enc.decode([tid])
        print(f"  {tid:>6}  ->  {piece!r}")


def context_budget(context_window: int, reserved_for_output: int, prompt_tokens: int) -> int:
    """How many tokens of conversation history/RAG context are left to fill."""
    available = context_window - reserved_for_output - prompt_tokens
    return max(available, 0)


if __name__ == "__main__":
    explore_tokens("Large Language Models tokenize subwords, not whole words!")

    # Example budgeting
    remaining = context_budget(context_window=200_000, reserved_for_output=1024, prompt_tokens=1500)
    print(f"\nTokens left for history/context: {remaining}")
