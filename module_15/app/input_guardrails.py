"""Input validation: a handful of independent checks run over 
a user message before it ever reaches the LLM.
"""

import re
import unicodedata

from langdetect import LangDetectException, detect

INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) instructions",
    r"disregard (all |the )?(previous|prior|above)",
    r"reveal (your |the )?(system prompt|instructions)",
    r"you are now (a|an) ",
    r"pretend (that )?you are",
    r"act as (if )?you (have no|are not) (restrictions|rules|guidelines)",
]

# Demo-scale blocklist
DISALLOWED_TOPICS = ["make a bomb", "synthesize nerve gas", "commit credit card fraud"]

MAX_INPUT_CHARS = 2000
ALLOWED_LANGUAGES = {"en"}


def check_prompt_injection(text: str) -> tuple[bool, str | None]:
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return False, f"matched injection pattern: {pattern!r}"
    return True, None


def check_disallowed_topics(text: str) -> tuple[bool, str | None]:
    lowered = text.lower()
    for topic in DISALLOWED_TOPICS:
        if topic in lowered:
            return False, f"matched disallowed topic: {topic!r}"
    return True, None


def check_length(text: str, max_chars: int = MAX_INPUT_CHARS) -> tuple[bool, str | None]:
    if len(text) > max_chars:
        return False, f"input is {len(text)} chars, over the {max_chars}-char limit"
    return True, None


def sanitize_encoding(text: str) -> str:
    """Normalize unicode to a canonical form (NFKC) and strip control
    characters - the two concrete pieces of "character encoding
    sanitization" a plain string can actually need."""
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in normalized if ch in "\n\t" or not unicodedata.category(ch).startswith("C"))


def check_language(text: str, allowed: set[str] = ALLOWED_LANGUAGES) -> tuple[bool, str | None]:
    try:
        lang = detect(text)
    except LangDetectException:
        return False, "could not detect a language"
    if lang not in allowed:
        return False, f"detected language {lang!r}, not in allowed set {allowed}"
    return True, None


def validate_input(text: str) -> dict:
    clean_text = sanitize_encoding(text)
    checks = {
        "prompt_injection": check_prompt_injection(clean_text),
        "disallowed_topic": check_disallowed_topics(clean_text),
        "length": check_length(clean_text),
        "language": check_language(clean_text),
    }
    violations = {name: reason for name, (ok, reason) in checks.items() if not ok}
    return {"clean_text": clean_text, "passed": not violations, "violations": violations}
