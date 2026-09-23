"""Shared embedding function - embeds through Google's Gemini API
(the `gemini-embedding-001` model) via the shared ../../common/llm_client.py
client. Vectors come back L2-normalized, so cosine similarity reduces
to a plain dot product.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import embed

MODEL_NAME = "gemini-embedding-001"
