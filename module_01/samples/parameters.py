"""Temperature and determinism.

Demonstrates the key model parameters section of the roadmap -
temperature and seed - and shows why temperature=0 is close to 
deterministic. Calls Google's Gemini API (gemini-3.1-flash-lite).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import chat

PROMPT = "In one sentence, describe what a large language model is."


def run(temperature: float, seed: int | None = None) -> str:
    response = chat(
        messages=[{"role": "user", "content": PROMPT}],
        temperature=temperature,
        seed=seed,
    )
    return response["message"]["content"].strip()


if __name__ == "__main__":
    print("-- temperature=0.0 (near-deterministic), run twice --")
    print(run(temperature=0.0, seed=42))
    print(run(temperature=0.0, seed=42))

    print("\n-- temperature=1.2 (creative/random), run three times --")
    for _ in range(3):
        print(run(temperature=1.2))
