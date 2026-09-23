"""Content moderation via Gemini (`gemini-3.1-flash-lite`) prompted as a
Llama Guard 3-style safety classifier - used here instead of the OpenAI
Moderation API or Perspective API. The prompt carries Llama Guard 3's own
hazard taxonomy and output format, so the model returns "safe", or
"unsafe" followed by a line of comma-separated category codes (e.g.
"unsafe\\nS2" for a non-violent-crime suggestion aimed at an assistant
turn). Passing just a user message moderates that input; passing
`[user, assistant]` moderates the assistant's (last) turn - the same call
handles both directions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import chat  # noqa: E402

# Llama Guard 3's hazard taxonomy - sent to the model as the policy, and
# used to label the category codes it returns.
CATEGORY_LABELS = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}


MODERATION_PROMPT = """Task: Check if there is unsafe content in '{role}' messages in the conversation below according to our safety policy with the categories listed.

<BEGIN UNSAFE CONTENT CATEGORIES>
{categories}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

{conversation}

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST {role} message in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated category codes (e.g. S2).
Respond with nothing else."""


def moderate(messages: list[dict]) -> dict:
    """`messages` is a `[user]` or `[user, assistant]` list - the
    classifier evaluates whichever turn is last."""
    role = "Agent" if messages[-1]["role"] == "assistant" else "User"
    conversation = "\n\n".join(f"{'Agent' if m['role'] == 'assistant' else 'User'}: {m['content']}" for m in messages)
    categories = "\n".join(f"{code}: {label}." for code, label in CATEGORY_LABELS.items())
    prompt = MODERATION_PROMPT.format(role=role, categories=categories, conversation=conversation)

    response = chat([{"role": "user", "content": prompt}], temperature=0.0)
    content = response["message"]["content"].strip()
    lines = content.splitlines()
    flagged = lines[0].strip().lower() == "unsafe"
    codes = [c.strip() for line in lines[1:] for c in line.split(",") if c.strip()]
    categories = [CATEGORY_LABELS.get(c, c) for c in codes]
    return {"flagged": flagged, "categories": categories, "raw": content}


def moderate_input(user_message: str) -> dict:
    return moderate([{"role": "user", "content": user_message}])


def moderate_output(user_message: str, assistant_message: str) -> dict:
    return moderate(
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    )
