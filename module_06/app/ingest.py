"""Indexing pipeline: ingest -> parse -> chunk -> embed -> store.

  - Ingest/parse: read every .md file in docs/ as plain text, plus one
    .pdf file parsed with pypdf (pdf_loader.py) - a concrete example of
    "Document Ingestion and Parsing".
  - Chunk: recursive character text splitter (chunking.py).
  - Embed: Gemini `gemini-embedding-001` via embedder.py.
  - Store: chunk text + metadata as JSON, vectors as a NumPy array on
    disk - the same "store vectors + full document text" idea as
    Module 4/5, without pulling in a vector database.

Run standalone to (re)build the index:
    python ingest.py
"""

import json
from pathlib import Path

import numpy as np

from chunking import recursive_split
from embedder import embed
from pdf_loader import load_pdf

DOCS_DIR = Path(__file__).parent / "docs"
INDEX_DIR = Path(__file__).parent / "index_store"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60


def load_documents() -> list[dict]:
    """Ingest + parse: every markdown file as plain text, every PDF
    file's text extracted via pdf_loader.load_pdf."""
    documents = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        documents.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    for path in sorted(DOCS_DIR.glob("*.pdf")):
        documents.append({"source": path.name, "text": load_pdf(path)})
    return documents


def build_chunks(documents: list[dict]) -> list[dict]:
    """Chunk each document, keeping track of which source file and
    chunk-within-document each piece came from (metadata preservation)."""
    chunks = []
    for doc in documents:
        pieces = recursive_split(doc["text"], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "id": f"{doc['source']}::{i}",
                    "text": piece,
                    "source": doc["source"],
                    "chunk_index": i,
                }
            )
    return chunks


def build_index(force: bool = False) -> int:
    """Build (or reuse) the on-disk index. Returns the number of chunks
    in the index. Idempotent: running this twice does not re-embed and
    re-write unless force=True."""
    INDEX_DIR.mkdir(exist_ok=True)

    if not force and CHUNKS_PATH.exists() and EMBEDDINGS_PATH.exists():
        chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        print(f"Index already built ({len(chunks)} chunks) - reusing {INDEX_DIR}. Pass force=True to rebuild.")
        return len(chunks)

    documents = load_documents()
    chunks = build_chunks(documents)
    vectors = embed([c["text"] for c in chunks])

    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    np.save(EMBEDDINGS_PATH, np.asarray(vectors))

    print(f"Indexed {len(documents)} documents into {len(chunks)} chunks -> {INDEX_DIR}")
    return len(chunks)


def load_index() -> tuple[list[dict], np.ndarray]:
    """Load the persisted chunks + embeddings for retrieval."""
    if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        build_index()
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    vectors = np.load(EMBEDDINGS_PATH)
    return chunks, vectors


if __name__ == "__main__":
    build_index()
