"""ChromaDB mini application: a support-article knowledge base.

Covers the ChromaDB workflow from the roadmap in one small, runnable app:
  - installation and setup (PersistentClient)
  - creating a collection (with a custom embedding function + cosine space)
  - adding documents with metadata
  - querying with text
  - metadata filtering
  - persistent storage (re-running this script does not re-seed data)

Run:
    python app.py
"""

from pathlib import Path

import chromadb

from embedding_function import SimpleHashingEmbeddingFunction

DB_PATH = Path(__file__).parent / "chroma_store"
COLLECTION_NAME = "support_articles"

ARTICLES = [
    {"id": "a1", "category": "billing", "text": "How to update your credit card on file for monthly billing."},
    {"id": "a2", "category": "billing", "text": "Understanding prorated charges when you upgrade your subscription plan."},
    {"id": "a3", "category": "account", "text": "Steps to reset your password if you're locked out of your account."},
    {"id": "a4", "category": "account", "text": "How to enable two-factor authentication for extra account security."},
    {"id": "a5", "category": "technical", "text": "Troubleshooting steps when the mobile app crashes on startup."},
    {"id": "a6", "category": "technical", "text": "Fixing sync issues between the desktop and mobile applications."},
    {"id": "a7", "category": "shipping", "text": "How to track your order once it has shipped from our warehouse."},
    {"id": "a8", "category": "shipping", "text": "What to do if your package arrives damaged or items are missing."},
]


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=SimpleHashingEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def seed_if_empty(collection) -> None:
    if collection.count() > 0:
        print(f"Collection already has {collection.count()} articles - persisted from a previous run, skipping seed.")
        return

    collection.add(
        ids=[article["id"] for article in ARTICLES],
        documents=[article["text"] for article in ARTICLES],
        metadatas=[{"category": article["category"]} for article in ARTICLES],
    )
    print(f"Seeded {len(ARTICLES)} articles into '{COLLECTION_NAME}'.")


def search(collection, query: str, k: int = 3, category: str | None = None) -> None:
    where = {"category": category} if category else None
    results = collection.query(query_texts=[query], n_results=k, where=where)

    suffix = f"  (filtered to category={category!r})" if category else ""
    print(f"\nQuery: {query!r}{suffix}")
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"  [{meta['category']:9s}] (distance={distance:.3f}) {doc}")


def main() -> None:
    collection = get_collection()
    seed_if_empty(collection)

    search(collection, "I forgot my password")
    search(collection, "the app keeps crashing")
    search(collection, "my package never arrived", category="shipping")

    print(f"\nTotal articles stored: {collection.count()}")
    print(f"Persisted at: {DB_PATH}")


if __name__ == "__main__":
    main()
