"""RAG over the runbook knowledge base: chunk -> embed -> store -> retrieve.

Retrieval uses FAISS's IndexFlatIP (exact inner-product search). FAISS is
a similarity-search library, not a database - it has no store for chunk
text/metadata, so chunks.json carries that alongside the index (FAISS
returns row indices, looked up against this same-order list).

Embeddings come from the shared common/llm_client.py's embed()
(gemini-embedding-001, 768-dim), using Gemini's asymmetric-retrieval task
types: "RETRIEVAL_DOCUMENT" for indexed chunks, "RETRIEVAL_QUERY" for a
search query. embed() L2-normalizes every vector - this is what makes
IndexFlatIP's inner product equivalent to cosine similarity.
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from llm_client import EMBED_DIM, embed  # noqa: E402

from chunking import recursive_split  # noqa: E402

RUNBOOKS_DIR = Path(__file__).parent / "data" / "runbooks"
INDEX_DIR = Path(__file__).parent / "index_store"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
FAISS_INDEX_PATH = INDEX_DIR / "runbooks.faiss"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
TOP_K = 2

def _load_runbooks() -> list[dict]:
    return [{"source": path.name, "text": path.read_text(encoding="utf-8")} for path in sorted(RUNBOOKS_DIR.glob("*.md"))]


def _build_chunks(documents: list[dict]) -> list[dict]:
    chunks = []
    for doc in documents:
        pieces = recursive_split(doc["text"], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        for i, piece in enumerate(pieces):
            chunks.append({"id": f"{doc['source']}::{i}", "text": piece, "source": doc["source"]})
    return chunks


def build_index(force: bool = False) -> int:
    """Build (or reuse) the on-disk runbook index. Idempotent unless force=True."""
    INDEX_DIR.mkdir(exist_ok=True)

    if not force and CHUNKS_PATH.exists() and FAISS_INDEX_PATH.exists():
        return len(json.loads(CHUNKS_PATH.read_text(encoding="utf-8")))

    documents = _load_runbooks()
    chunks = _build_chunks(documents)
    vectors = np.array(embed([c["text"] for c in chunks], task_type="RETRIEVAL_DOCUMENT"), dtype="float32")

    index = faiss.IndexFlatIP(EMBED_DIM)  # inner product; == cosine similarity for L2-normalized vectors
    index.add(vectors)

    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    return len(chunks)


def _load_index() -> tuple[list[dict], faiss.Index]:
    if not CHUNKS_PATH.exists() or not FAISS_INDEX_PATH.exists():
        build_index()
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    return chunks, index


def top_k_retrieve(query: str, k: int = TOP_K) -> list[dict]:
    chunks, index = _load_index()
    query_vector = np.array(embed([query], task_type="RETRIEVAL_QUERY"), dtype="float32")
    scores, indices = index.search(query_vector, min(k, index.ntotal))

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:  # FAISS pads with -1 if k exceeds the number of stored vectors
            continue
        chunk = dict(chunks[idx])
        chunk["retrieval_score"] = float(score)
        results.append(chunk)
    return results


@tool
def search_runbooks(query: str) -> str:
    """Search the runbook knowledge base for entries matching known incident
    symptoms or root causes. Use this once you've observed a pattern in the
    logs/metrics (e.g. "connection pool exhausted", "cache miss rate spike")
    to check whether it matches a previously documented issue and fix."""
    results = top_k_retrieve(query, k=TOP_K)
    if not results:
        return "No matching runbook entries found."
    return "\n\n".join(f"[{r['source']}, score={r['retrieval_score']:.3f}]\n{r['text']}" for r in results)
