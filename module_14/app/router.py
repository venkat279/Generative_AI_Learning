"""Model routing strategy: classify a query's complexity, route to 
a cheap or powerful tier, and cascade up if the cheap tier's answer
was cut off before finishing.

Both tiers call the same underlying Gemini model (`gemini-3.1-flash-lite`) -
"cheap" vs "powerful" is simulated via generation settings (a tight vs
generous `num_predict`/`temperature`), not a real model swap. See the
README for why a second real model isn't used for this exercise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import chat  # noqa: E402

TIERS = {
    "simple": {"temperature": 0.0, "num_predict": 40},
    "complex": {"temperature": 0.7, "num_predict": 200},
}

SIMPLE_MARKERS = ("what is", "what's", "define", "capital of", "how many", "spell")


def classify_complexity(query: str) -> str:
    """Heuristic complexity classifier - short, factual-sounding questions
    are 'simple'; anything else is 'complex'. A real system would likely use
    a small dedicated classifier model instead of keyword rules.
    """
    q = query.lower().strip()
    if len(q) < 60 and any(marker in q for marker in SIMPLE_MARKERS):
        return "simple"
    return "complex"


def route(query: str) -> dict:
    tier = classify_complexity(query)
    response = chat([{"role": "user", "content": query}], **TIERS[tier])
    escalated = False

    # Cascade: the "simple" tier's tight num_predict cut the answer off
    # before it finished (done_reason == "length") - a genuine "needed more
    # budget" signal, not a simulated one - so escalate for real.
    if tier == "simple" and response.get("done_reason") == "length":
        tier = "complex"
        escalated = True
        response = chat([{"role": "user", "content": query}], **TIERS["complex"])

    return {"tier": tier, "escalated": escalated, "answer": response["message"]["content"]}
