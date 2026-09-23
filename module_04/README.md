# Module 4 — Embeddings and Semantic Search

Concept overview — brief explanations of every sub-topic in
Module 4 — Embeddings and Semantic Search.

## 1. Understanding Embeddings

- **What an embedding is** — a dense vector of floating-point numbers
  produced by a model to represent a piece of text (or image, audio,
  etc.), positioned in a high-dimensional space such that meaning is
  encoded as geometry.
- **Why embeddings capture semantic meaning** — the model producing
  them was trained so that texts with similar meaning end up close
  together in vector space, and dissimilar texts end up far apart —
  purely from patterns learned over training data, not hand-coded
  rules.
- **Embedding space and clustering** — because similar meanings sit
  close together, groups of related texts naturally form visible
  clusters when embeddings are plotted or clustered (e.g. with
  k-means).
- **Sparse vectors (TF-IDF, BM25) vs dense vectors** — sparse vectors
  are mostly zeros, one dimension per vocabulary word, capturing
  keyword overlap; dense embeddings are compact (a few hundred to a
  few thousand dimensions, all non-zero) and capture meaning beyond
  exact word matches — a synonym scores well against a dense embedding
  but not against a sparse one.
- **Use cases — search, clustering, classification, RAG** — embeddings
  underpin semantic search (find similar text), clustering (group
  similar texts), classification (compare against labeled examples),
  and RAG's retrieval step (Module 6).

## 2. Embedding Models

- **OpenAI text-embedding-3-small and text-embedding-3-large** —
  OpenAI's hosted embedding models, differing in vector dimensionality
  and quality/cost.
- **Cohere embed-english-v3.0 and embed-multilingual-v3.0** — Cohere's
  hosted embedding models, one English-only and one covering many
  languages.
- **Google text-embedding-004** — Google's hosted embedding model,
  part of the Gemini API family.
- **sentence-transformers — all-MiniLM-L6-v2, BAAI/bge-large-en** —
  open-weight embedding models runnable locally; `all-MiniLM-L6-v2` is
  small/fast, `bge-large-en` is larger and higher quality. This
  module's hands-on uses Google's hosted `gemini-embedding-001`
  instead, so no local PyTorch install is required.
- **Code embeddings** — embedding models trained specifically on source
  code, so semantically similar code (not just textually similar) ends
  up close together.
- **Image embeddings (CLIP)** — a model trained to embed images and
  text into the *same* vector space, enabling cross-modal search (find
  images by text query, or vice versa).
- **Embedding dimensions — 768, 1536, 3072** — common vector sizes
  across different models; higher dimensionality generally means more
  captured nuance at the cost of more storage/compute per vector.

## 3. Similarity and Distance Metrics

- **Cosine similarity** — measures the angle between two vectors,
  ignoring magnitude; ranges from -1 to 1, with 1 meaning identical
  direction (most common metric for text embeddings).
- **Dot product similarity** — the raw dot product of two vectors;
  equivalent to cosine similarity when both vectors are already
  unit-length (L2-normalized), which is the case for this module's
  embeddings.
- **Euclidean (L2) distance** — the straight-line distance between two
  vectors' endpoints; unlike cosine, it's sensitive to vector
  magnitude, not just direction.
- **Computing cosine similarity in Python** — a NumPy dot-product plus
  a norm calculation (or a plain dot product alone, if vectors are
  pre-normalized).
- **Interpreting similarity scores** — a higher cosine/dot-product
  score (or lower distance) means "more similar," but the absolute
  number's meaning is relative to the embedding model and corpus, not
  an intrinsic percentage.

## 4. Building a Semantic Search System

- **Embedding a document corpus** — running every document through the
  embedding model once, ahead of time, to build a searchable index.
- **Storing embeddings as NumPy arrays** — the simplest possible
  storage for a small corpus: one array of vectors kept in memory or
  saved to disk (`.npy`), searched with plain array operations, no
  database required.
- **Embedding the user query** — running the same embedding model over
  the search query at request time, so it lands in the same vector
  space as the indexed documents.
- **Top-K retrieval** — scoring the query vector against every stored
  document vector and returning the K highest-scoring matches.
- **Minimal semantic search with FAISS** — using Facebook's FAISS
  library's flat index (`IndexFlatIP`) for the same top-K search, which
  scales better than a naive NumPy loop as the corpus grows.
- **Hybrid search — BM25 + vector search** — combining keyword-based
  scoring (BM25, a sparse-vector technique) with dense-vector
  similarity, so a result needs to score well on either exact keyword
  match or semantic closeness — catches cases either method alone
  would miss.

---

## 5. Hands-on: Module 4 mini project

A small, self-contained project covering every topic above, using
Google's Gemini embedding model — `gemini-embedding-001`, truncated to
768 dimensions, called through the Gemini API.

### 5.1 Setup

```bash
pip install -r requirements.txt
```

### 5.2 Prerequisites: a Gemini API key

Every script in this module embeds text through Google's Gemini API
(`gemini-embedding-001`) via `../../common/llm_client.py`. Before running
them:

1. Put a Gemini API key in `common/.env` (the shared folder next to
   the module folders) as `GEMINI_API_KEY=...`
   (keys come from https://aistudio.google.com/apikey).
2. `python-dotenv` (in `requirements.txt`) loads that `.env` file
   automatically - no environment variables to export by hand.

### 5.3 Shared corpus

All scripts live in `samples/` and import `samples/corpus.py`: 17 short
sentences across 4 topics (Python, cooking, animals, space). Using one
small, obviously-clustered corpus everywhere makes it easy to eyeball
whether each technique is "working."

### 5.4 Files, mapped to roadmap topics

| Script | Module 4 topic covered |
|---|---|
| `embedding.py` | What an embedding is; why embeddings capture semantic meaning |
| `embedding_space_clustering.py` | Embedding space and clustering; use case — clustering |
| `similarity_metrics.py` | Cosine similarity, dot product, Euclidean (L2) distance; computing cosine similarity in Python; interpreting scores |
| `sparse_vs_dense.py` | Sparse vectors (TF-IDF) vs dense vectors |
| `semantic_search_numpy.py` | Embedding a corpus; storing embeddings as NumPy arrays; embedding a query; Top-K retrieval |
| `semantic_search_faiss.py` | Minimal semantic search with FAISS |
| `hybrid_search_bm25_vector.py` | Hybrid search — BM25 + vector search |

`corpus.py` — shared corpus + the model name constant (not a roadmap
topic by itself, just shared setup). All scripts embed text through
`../../common/llm_client.py`'s `embed()` function, the same shared
Gemini client used elsewhere in this repo.

### 5.5 Run order

```bash
cd samples
python embedding.py
python embedding_space_clustering.py
python similarity_metrics.py
python sparse_vs_dense.py
python semantic_search_numpy.py
python semantic_search_faiss.py
python hybrid_search_bm25_vector.py
```

Each script is independent and prints its own explanation of what to look
for in the output, so you can run them standalone in any order.

### 5.6 How these samples work

`embed()` L2-normalizes every vector before returning it (truncated
Gemini embeddings aren't unit-length on their own), so dot product and cosine similarity are numerically identical
throughout this module — `similarity_metrics.py` calls this out
directly, and `semantic_search_numpy.py`/`semantic_search_faiss.py`/
`hybrid_search_bm25_vector.py` all use a plain dot product for ranking
rather than a separate cosine-similarity step.

`embedding_space_clustering.py`'s KMeans clustering doesn't always
recover all 4 topics as cleanly separated clusters — occasional
mixing between two topics is expected, not a bug.

### 5.7 Notes on scope

- **Embedding model**: only Gemini's `gemini-embedding-001` is used — no OpenAI/Cohere embedding APIs. Swapping models means changing `EMBED_MODEL` (and `EMBED_DIM`) in `common/llm_client.py`.
- **Vector database**: Module 4 only asks for a "minimal" FAISS example — a real Chroma/Qdrant setup is Module 5's topic, not reproduced here.
- **Reranking / RRF**: Module 4 lists hybrid search as BM25 + vector search only; the more advanced Reciprocal Rank Fusion technique belongs to Module 6 (RAG) and isn't used here — `hybrid_search_bm25_vector.py` combines scores with simple min-max normalization + weighted sum instead.
