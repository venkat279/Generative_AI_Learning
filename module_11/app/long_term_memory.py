"""Long-term memory: durable facts and conversation summaries, stored
as embeddings in a local SQLite database (the "External State Stores"
sub-topic used here - SQLite needs no separate service, unlike
Redis/PostgreSQL, consistent with this repo's local-only convention)
and retrieved by top-K cosine similarity - the same mechanic as
Module 6's retrieve.py, applied to memories instead of document
chunks.
"""

import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import embed

DB_PATH = Path(__file__).parent / "memory_store.db"
DUPLICATE_THRESHOLD = 0.95  # cosine similarity above this counts as "already remembered"


class LongTermMemory:
    def __init__(self, db_path: Path = DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                kind TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL
            )"""
        )
        self.conn.commit()

    def _all_rows(self):
        cursor = self.conn.execute("SELECT id, text, kind, embedding, created_at, expires_at FROM memories")
        return cursor.fetchall()

    def add(self, text: str, kind: str = "fact", ttl_seconds: float | None = None) -> bool:
        """Embed and store one memory - unless a near-duplicate already
        exists (memory consolidation). Returns True if actually stored."""
        vector = np.asarray(embed([text])[0], dtype=np.float32)

        for _, existing_text, _, blob, _, _ in self._all_rows():
            existing_vector = np.frombuffer(blob, dtype=np.float32)
            if float(vector @ existing_vector) >= DUPLICATE_THRESHOLD:
                return False

        now = time.time()
        expires_at = now + ttl_seconds if ttl_seconds is not None else None
        self.conn.execute(
            "INSERT INTO memories (text, kind, embedding, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
            (text, kind, vector.tobytes(), now, expires_at),
        )
        self.conn.commit()
        return True

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Top-K memories by cosine similarity, excluding anything past
        its expiry (memory expiry and freshness)."""
        now = time.time()
        query_vector = np.asarray(embed([query])[0], dtype=np.float32)

        scored = []
        for row_id, text, kind, blob, created_at, expires_at in self._all_rows():
            if expires_at is not None and expires_at < now:
                continue
            vector = np.frombuffer(blob, dtype=np.float32)
            scored.append({"id": row_id, "text": text, "kind": kind, "score": float(query_vector @ vector)})

        scored.sort(key=lambda m: m["score"], reverse=True)
        return scored[:k]
