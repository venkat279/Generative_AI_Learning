"""A hand-built validator pipeline standing in for the Guardrails AI library: 
run a list of validators over a piece of text and apply one of Guardrails AI's 
own named on-fail actions - reask, filter, exception, noop.
"""

from collections.abc import Callable
from dataclasses import dataclass


class GuardrailViolation(Exception):
    pass


@dataclass
class Validator:
    name: str
    check: Callable[[str], tuple[bool, str | None]]
    on_fail: str  # "reask" | "filter" | "exception" | "noop"
    fix: Callable[[str, str], str] | None = None  # used by "filter": (text, reason) -> new text
    reask_fn: Callable[[str, str], str] | None = None  # used by "reask": (text, reason) -> new text


def run_validators(text: str, validators: list[Validator]) -> dict:
    current = text
    report = []
    for validator in validators:
        ok, reason = validator.check(current)
        if ok:
            report.append({"validator": validator.name, "passed": True})
            continue

        entry = {"validator": validator.name, "passed": False, "reason": reason}

        if validator.on_fail == "noop":
            report.append(entry)
            continue
        elif validator.on_fail == "exception":
            report.append(entry)
            raise GuardrailViolation(f"{validator.name} failed: {reason}")
        elif validator.on_fail == "filter" and validator.fix:
            current = validator.fix(current, reason)
        elif validator.on_fail == "reask" and validator.reask_fn:
            current = validator.reask_fn(current, reason)

        # Re-check after applying a fix, so the report reflects whether the
        # fix actually worked instead of just that a violation was found.
        fixed_ok, fixed_reason = validator.check(current)
        entry["fixed"] = fixed_ok
        if not fixed_ok:
            entry["reason_after_fix"] = fixed_reason
        report.append(entry)

    return {"final_text": current, "report": report}
