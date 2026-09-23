"""Regex-based PII scrubbing, applied to prompts/responses before they're
written to a trace log ("PII scrubbing before logging").
A lighter touch than Module 15's dedicated PII-detection tooling - just
enough to keep obvious PII out of a log file.
"""

import re

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")


def scrub(text: str) -> str:
    if not text:
        return text
    text = EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    text = PHONE_RE.sub("[PHONE_REDACTED]", text)
    return text
