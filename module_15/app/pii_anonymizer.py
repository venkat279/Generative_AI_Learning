"""PII detection and reversible redaction - a hand-built stand-in for 
Microsoft Presidio: regex-based detection for a few PII types, 
replacing each match with a unique placeholder token and keeping
the real value in a mapping so it can be substituted back later
(Presidio's "reversing anonymization in the response"). Module 13's
`pii_scrub.py` does one-way redaction for logs only; this is the
two-way version, meant to sit in front of an LLM call - the model never
sees the real value, only the placeholder.
"""

import re
import uuid

# Order matters: most specific/narrow patterns first, so a broader one
# (CREDIT_CARD) doesn't consume digits a narrower one already claimed.
PATTERNS = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE": re.compile(r"\b\d{3}[\s.-]\d{3}[\s.-]\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"),
}


def anonymize(text: str) -> tuple[str, dict[str, str]]:
    mapping: dict[str, str] = {}
    result = text
    for pii_type, pattern in PATTERNS.items():

        def _replace(match: re.Match, pii_type: str = pii_type) -> str:
            token = f"<{pii_type}_{uuid.uuid4().hex[:6]}>"
            mapping[token] = match.group(0)
            return token

        result = pattern.sub(_replace, result)
    return result, mapping


def deanonymize(text: str, mapping: dict[str, str]) -> str:
    """Substitutes each placeholder token back to its real value. Also
    tries the token with its angle brackets stripped - a model asked to
    echo `<EMAIL_abc123>` inline will sometimes drop the brackets while
    still keeping the token text itself intact (observed in practice, not
    just a defensive guess), so an exact-string match alone would miss it.
    """
    for token, original in mapping.items():
        text = text.replace(token, original)
        text = text.replace(token.strip("<>"), original)
    return text
