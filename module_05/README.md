# Module 5 — Vector Databases

Concept guide for Module 5 - Vector Databases, plus a
small hands-on ChromaDB mini app.

Module 4 showed how to embed text and search it with plain NumPy and a
minimal FAISS index. Module 5 is about what changes when that search
needs to run in production: at millions of vectors, with metadata
filters, persistence, and concurrent access — the job of a **vector
database**.

---

## 1. Core Concepts

- **Why relational databases are insufficient for vectors** — a
  relational database indexes *discrete, low-cardinality* values with
  structures like B-trees, built for equality (`WHERE id = 5`) and
  range queries. An embedding is a dense vector of a few hundred to a
  few thousand floats, and the question asked of it is "what is
  *closest* to this, across every dimension at once?" — a B-tree gives
  no leverage there, since no ordering of a high-dimensional point
  keeps "nearby" points adjacent. The only correct relational answer is
  to scan every row and compute a distance, which is too slow well
  before a production-sized corpus. Vector databases exist to answer
  nearest-neighbor questions without that full scan.
- **Approximate Nearest Neighbor (ANN) search** — computing the *exact*
  nearest neighbors always costs O(n) distance calculations per query —
  there's no way around it without giving something up. ANN algorithms
  give up a small, tunable amount of accuracy (they may occasionally
  miss the true single closest vector) in exchange for query times that
  are sub-linear, often near-constant, even across tens of millions of
  vectors — this speed/accuracy trade-off ("recall@k" vs latency) is
  the central design axis of every ANN index type below.
- **HNSW indexing algorithm (Hierarchical Navigable Small World)** — a
  **graph-based** ANN index and the default choice in most modern
  vector databases (Qdrant, Chroma, Milvus, Pinecone). It builds several
  stacked layers of a proximity graph: the top layer is sparse with a
  few long-range links, each layer below progressively denser. A search
  starts at the top layer, greedily walks toward the query vector, then
  drops down a layer and repeats — like using a highway system to get
  close to a destination before switching to local streets. Key tuning
  knobs: `M` (max connections per node — higher = better recall, more
  memory), `ef_construction` (how thoroughly the graph is built), and
  `ef_search` (how thoroughly a query searches it).
- **IVF (Inverted File Index)** — a **clustering-based** ANN approach.
  During indexing, vectors are grouped into `nlist` clusters (via
  k-means), each with a centroid. At query time, the query vector is
  compared only against the centroids, and only the `nprobe` closest
  clusters' vectors are actually scanned — the rest of the dataset is
  skipped entirely. This is fast, but can miss a true neighbor sitting
  near a cluster boundary (`nprobe` trades recall for speed too). IVF
  is frequently paired with **product quantization (PQ)** —
  compressing each vector into a small code — to shrink memory
  footprint at very large scale (`IVFPQ`).
- **Flat index — exact search** — no approximation at all: every query
  is compared against every stored vector (brute force), guaranteeing
  the true top-k result — the same idea as plain NumPy/FAISS
  `IndexFlatIP` in Module 4. Doesn't scale past maybe low millions of
  vectors on a single machine, but it's the correctness baseline used
  to sanity-check an ANN index's recall.
- **Metadata filtering alongside vector search** — real queries are
  rarely "find things similar to X" in isolation — usually "find things
  similar to X, *where* category = 'finance' and date > 2025-01-01."
  Vector databases store structured metadata (Chroma calls it
  `metadata`, Qdrant calls it `payload`) alongside each vector and let
  a query combine similarity search with filter conditions, via two
  strategies: **pre-filtering** (narrow to matching rows first, then
  search only within that subset — precise, can be slow on a large
  unindexed subset) or **post-filtering** (search first, then discard
  results failing the filter — fast, but risks returning fewer than
  `k` results). Most production vector databases (Qdrant in
  particular) implement smarter hybrid strategies internally so you
  don't have to choose manually.

---

## 2. ChromaDB

An open-source, developer-friendly vector database that can run entirely
**in-process** (no separate server needed) or against a running Chroma
server — the easiest of the three to get started with locally.

**Workflow, in the order the roadmap lists it:**

- **Installation and setup** — `pip install chromadb`; create a client. Chroma offers a few client modes: an ephemeral in-memory `chromadb.Client()`, a `PersistentClient(path=...)` that writes to a local folder, or an `HttpClient(...)` that talks to a separately running Chroma server.
- **Creating collections** — `client.create_collection(name=...)`, roughly Chroma's equivalent of a table/namespace. You can configure the distance metric (`cosine`, `l2`, `ip`) and, optionally, an embedding function the collection uses automatically.
- **Adding documents with embeddings and metadata** — `collection.add(ids=[...], documents=[...], metadatas=[...], embeddings=[...])`. You can either hand Chroma raw text and let its configured embedding function vectorize it, or supply pre-computed vectors yourself (e.g., from Module 4's embedding pipeline).
- **Querying with text or vectors** — `collection.query(query_texts=[...])` auto-embeds the query text, or `collection.query(query_embeddings=[...])` if you already have a vector. Returns top-k ids, documents, metadatas, and distances.
- **Metadata filtering** — pass `where={...}` to constrain by metadata fields, and `where_document={...}` to filter on the document text itself, combined with the similarity search in one call.
- **Persistent vs in-memory storage** — the ephemeral `Client()` keeps everything in RAM and loses it when the process exits; `PersistentClient(path="./chroma_db")` writes to disk so the collection survives restarts — the difference matters as soon as you want a corpus to persist between hands-on runs.

---

## 3. Qdrant

A production-grade vector database with a Rust core, exposed over
REST/gRPC, designed to be run as its own service (self-hosted via Docker
or a managed Qdrant Cloud cluster) rather than embedded in-process.

**Workflow, in the order the roadmap lists it:**

- **Installation — Docker and cloud** — self-host with `docker run -p 6333:6333 qdrant/qdrant`, or use a managed **Qdrant Cloud** cluster if you'd rather not run Docker locally. Either way, the Python app talks to it via the `qdrant-client` package pointed at a URL (and API key, for cloud).
- **Collections, points, and payloads** — a *collection* holds *points*; each point is an `id` + one or more `vector`s + a `payload` (arbitrary JSON metadata). Conceptually similar to Chroma's documents + metadatas, but Qdrant is vector-native — there's no built-in "document text" field, so you store text in the payload yourself if you want it returned.
- **Filtering with payload conditions** — `Filter` objects built from `must` / `should` / `must_not` clauses over payload fields (exact match, ranges, geo-radius, etc.), evaluated together with the vector search in a single request.
- **Sparse + dense hybrid search** — a point can carry both a dense vector (semantic) and a sparse vector (keyword-style, e.g. BM25/SPLADE), and Qdrant can fuse both signals server-side — the same "hybrid search" idea built manually in Module 4's BM25 + vector script, but handled natively by the database here.
- **Named vectors for multi-vector use cases** — a single point can store multiple *named* vectors (e.g. a `title` embedding and a `body` embedding, possibly from different models or with different dimensions), queried independently or together — useful for multi-field or multi-modal search.

---

## 4. FAISS

Facebook AI Similarity Search — already used for a minimal flat index in
Module 4 (`IndexFlatIP`). Unlike Chroma or Qdrant, FAISS isn't a
"database": no server, no built-in persistence format beyond explicit
save/load, no native metadata filtering. It's the algorithmic toolbox
that many vector databases build their indexing on internally, exposed
directly as a library you drive yourself.

**Index types and operations covered:**

- **`IndexFlatL2` and `IndexFlatIP`** — exact brute-force search using Euclidean distance or inner product respectively. The correctness baseline; what Module 4's FAISS script already used.
- **`IndexIVFFlat`** — IVF clustering wrapped around flat vectors. Unlike the flat indexes, this **requires a training step** (`index.train(vectors)`) to learn cluster centroids before any vectors can be added. Tuned via `nlist` (number of clusters) and `nprobe` (clusters searched per query).
- **`IndexHNSWFlat`** — the graph-based HNSW index from §1. No training step required; typically better recall/speed than IVF without `nprobe` tuning, at a higher memory cost per vector.
- **Saving and loading indexes from disk** — `faiss.write_index(index, path)` and `faiss.read_index(path)` persist a built index so it doesn't need to be rebuilt every run — FAISS's answer to Chroma's `PersistentClient`.
- **GPU-accelerated FAISS** — the `faiss-gpu` package plus `faiss.index_cpu_to_gpu(...)` moves index build/search onto a CUDA GPU for very large datasets. Not relevant at the scale of this project's small hands-on corpus, included here for completeness.

---

## 5. How the three compare

| | ChromaDB | Qdrant | FAISS |
|---|---|---|---|
| Runs as | in-process library (or optional server) | separate service (Docker/cloud) | in-process library |
| Persistence | built-in (`PersistentClient`) | built-in (server-managed storage) | manual (`write_index`/`read_index`) |
| Metadata filtering | yes (`where`) | yes (`Filter`, richer conditions) | no — bring your own (e.g. filter in Python) |
| Hybrid (sparse+dense) search | not native | native | no — manual, as done in Module 4 |
| Best fit here | fastest to prototype with locally | closest to a "real" production setup | lowest-level, most control, no extra service |

---

## 6. Setup status for hands-on

Hands-on for this module covers **ChromaDB** only.

### 6.1 Python packages

```
chromadb>=0.5.0
scikit-learn>=1.4.0
```

`qdrant-client` and `faiss-cpu` aren't part of this module's hands-on
— Qdrant stays theory-only here (§3), and FAISS is exercised hands-on
in Module 4.

### 6.2 ChromaDB needs no external service

It runs entirely in-process against the local filesystem — no Docker,
no account, no network call.

---

## 7. Hands-on: ChromaDB mini app

A small support-article knowledge base search tool in `app/`,
covering the ChromaDB workflow from §2 end to end.

| File | Role |
|---|---|
| `app/embedding_function.py` | A custom `EmbeddingFunction` built on scikit-learn's `HashingVectorizer` — deterministic bag-of-words vectors, no model download, no PyTorch/ONNX Runtime. Not a real semantic model (Module 4 already covers that); it exists so this app's ChromaDB *plumbing* doesn't depend on the same PyTorch/ONNX blocker as Module 4. |
| `app/app.py` | The mini app: gets/creates a persistent collection, seeds 8 short support articles (categories: billing, account, technical, shipping) if the collection is empty, then runs a few example queries — plain text search, and a metadata-filtered search restricted to `category="shipping"`. |

**What it demonstrates, mapped to §2's ChromaDB steps:**

1. Installation and setup → `chromadb.PersistentClient(path=...)`
2. Creating collections → `get_or_create_collection(name=..., embedding_function=..., metadata={"hnsw:space": "cosine"})`
3. Adding documents with embeddings and metadata → `collection.add(ids=..., documents=..., metadatas=...)`
4. Querying with text → `collection.query(query_texts=[...], n_results=k)`
5. Metadata filtering → `collection.query(..., where={"category": "shipping"})`
6. Persistent vs in-memory storage → data is written to `app/chroma_store/`; re-running `app.py` detects the collection is already seeded (`collection.count() > 0`) and skips re-adding, instead of duplicating rows.

### 7.1 Step-by-step: what happens when you run `python app.py`

`main()` runs four steps in order:

1. **`get_collection()` opens (or creates) the database and collection.**
   - `chromadb.PersistentClient(path=DB_PATH)` opens `app/chroma_store/` as the on-disk store. If that folder doesn't exist yet, Chroma creates it; if it already exists from a previous run, it's reopened as-is — this one line is what makes the data durable across runs.
   - `client.get_or_create_collection(name="support_articles", embedding_function=..., metadata={"hnsw:space": "cosine"})` either creates a brand-new, empty collection the first time, or hands back the existing one on subsequent runs. The embedding function and distance metric are only actually applied at creation time; on later runs they're just used consistently to interpret the collection that already exists.

2. **`seed_if_empty(collection)` loads the 8 sample articles — but only once.**
   - It calls `collection.count()` first. If that's `> 0` (a previous run already added data), it prints how many articles are already stored and returns immediately — nothing is re-added, so re-running the script never creates duplicates.
   - Otherwise, it calls `collection.add(ids=[...], documents=[...], metadatas=[...])` with all 8 `ARTICLES` in one batch. Internally, for each document string, Chroma calls `embedding_function(documents)` — i.e. `SimpleHashingEmbeddingFunction.__call__` — to turn the text into a 512-dimensional vector, then stores `(id, vector, document text, metadata)` together for that row.

3. **Three `search(collection, query, ...)` calls run example queries.**
   - Each call does `collection.query(query_texts=[query], n_results=k, where=...)`. Chroma embeds the query string the same way it embedded the documents (same embedding function → same vector space), then runs a nearest-neighbor search over the stored vectors using the collection's cosine distance metric.
   - Query 1, `"I forgot my password"` — no `where` filter, searches all 8 articles; expected to rank the `account` category password-reset article (`a3`) at or near the top, since it shares the most hashed word-features with the query.
   - Query 2, `"the app keeps crashing"` — again unfiltered; expected to rank the `technical` mobile-crash article (`a5`) highest.
   - Query 3, `"my package never arrived", category="shipping"` — this one passes `where={"category": "shipping"}`, so Chroma restricts the nearest-neighbor search to only the 2 articles tagged `shipping` (`a7`, `a8`) before ranking them, rather than ranking against all 8 and hoping the filter doesn't exclude the best matches.
   - Each result is printed as `[category] (distance=...) article text`, one line per returned neighbor — lower `distance` means a closer (more similar) match.

4. **Final summary.** `collection.count()` and `DB_PATH` are printed so you can see how many rows now exist and exactly where they're persisted on disk — running the script again will show the same count and the "already seeded" message from step 2, instead of doubling it.

### 7.2 Why the embedding function matters here

`SimpleHashingEmbeddingFunction.__call__` runs `HashingVectorizer.transform(input)` on whatever list of strings it's given (documents at add-time, the query at search-time) and returns dense vectors:

- **Feature hashing, not a learned vocabulary** — instead of building a word→index dictionary from a fitted corpus (like `TfidfVectorizer` needs), each word is hashed directly to one of `n_features` (512) buckets. This means it needs no fitting step and can embed a brand-new query string it has never seen, with no retraining.
- **`alternate_sign=False`** keeps bucket values non-negative (closer to raw word counts) rather than allowing hash collisions to cancel each other out.
- **`norm="l2"`** unit-normalizes every output vector, which is what makes the collection's `"hnsw:space": "cosine"` distance metric behave correctly (cosine similarity between unit vectors reduces to a plain dot product).
- The trade-off: this only matches on shared words/hash buckets, not meaning — it demonstrates ChromaDB's collection/query/filter/persistence mechanics without needing a PyTorch/ONNX-based embedding model, but it is not the semantic search from Module 4.

**To run:**

```bash
cd app
pip install -r ../requirements.txt
python app.py
```

Re-running `app.py` reuses the existing collection instead of reseeding it — see §7.1 step 2.
