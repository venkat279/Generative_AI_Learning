"""Output validation: checks run on a model's response
before it's shown to a user.
"""

import json
import re

import jsonschema

CONTENT_POLICY_TERMS = ["kill yourself", "build a bomb"]  # demo-scale blocklist

CREDIT_CARD_RE = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b")


def check_json_schema(text: str, schema: dict) -> tuple[bool, str | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        return False, f"not valid JSON: {e}"
    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError as e:
        return False, f"schema violation: {e.message}"
    return True, None


def check_content_policy(text: str) -> tuple[bool, str | None]:
    lowered = text.lower()
    for term in CONTENT_POLICY_TERMS:
        if term in lowered:
            return False, f"matched disallowed content: {term!r}"
    return True, None


def strip_sensitive_patterns(text: str) -> str:
    """Regex/rule-based post-processing (§3): a last-resort safety net
    that redacts anything that looks like a raw credit-card number, even
    if it somehow made it into a response."""
    return CREDIT_CARD_RE.sub("[REDACTED]", text)


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\w+", text) if len(w) > 2}


def check_faithfulness(response: str, context: str, min_overlap: float = 0.4) -> tuple[bool, float]:
    """A lexical-overlap faithfulness proxy (§3): what fraction of the
    context's content words also show up in the response. This is much
    cheaper than a real entailment check (Module 16's entailment-based
    detection, SelfCheckGPT) and reasons about shared vocabulary, not
    meaning - it won't catch a response that negates the context using the
    same words, only one that's built from noticeably different facts.
    """
    context_words = _content_words(context)
    response_words = _content_words(response)
    if not context_words:
        return True, 1.0
    overlap = len(context_words & response_words) / len(context_words)
    return overlap >= min_overlap, round(overlap, 3)
