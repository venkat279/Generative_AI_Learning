"""Token counting and a cost calculator.

Counts tokens and multiplies by known pricing.
"""

import tiktoken

# USD per 1M tokens - illustrative pricing
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    encoding = tiktoken.encoding_for_model(model) if model in tiktoken.model.MODEL_TO_ENCODING else tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def estimate_cost(prompt: str, expected_output_tokens: int, model: str = "gpt-4o-mini") -> dict:
    input_tokens = count_tokens(prompt, model)
    rates = PRICING[model]
    input_cost = input_tokens / 1_000_000 * rates["input"]
    output_cost = expected_output_tokens / 1_000_000 * rates["output"]
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": expected_output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
    }


if __name__ == "__main__":
    long_prompt = "Summarize the following document in three bullet points.\n" + ("Lorem ipsum dolor sit amet. " * 200)

    for model in PRICING:
        estimate = estimate_cost(long_prompt, expected_output_tokens=150, model=model)
        print(estimate)
