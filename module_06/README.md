# Module 6 — Retrieval-Augmented Generation (RAG)

Concept overview — brief explanations of every sub-topic in
Module 6 — Retrieval-Augmented Generation (RAG) — plus a hands-on RAG
mini application.

## 1. RAG Architecture

- **The core problem RAG solves** — an LLM only knows what was in its
  training data, plus whatever text you put directly in its prompt.
  RAG solves the "the model doesn't know about *my* documents" problem
  by, at query time, searching your own document collection for the
  most relevant pieces and inserting them into the prompt, so the model
  generates an answer grounded in that retrieved text instead of (or in
  addition to) what it memorized during training.
- **RAG vs fine-tuning vs in-context learning** — three ways to get a
  model to use knowledge it wasn't trained on. **Fine-tuning** retrains
  the model's weights on your data — expensive, slow to update, and
  doesn't reliably teach a model *facts* so much as a *style*.
  **In-context learning** just pastes everything relevant into the
  prompt by hand — works for small amounts of text, but breaks down
  once your documents don't all fit in the context window. **RAG**
  keeps the model frozen but automatically fetches and injects only the
  relevant slice of a much larger document set per query — cheap to
  keep up to date (re-index a changed document, no retraining) and
  scales to far more content than the context window alone.
- **Indexing pipeline — ingest → parse → chunk → embed → store** — the
  one-time (or periodic) offline pipeline that prepares a document set
  for retrieval: load raw documents, extract their text, split that
  text into chunks, embed each chunk, and persist the chunks + vectors
  somewhere queryable.
- **Query pipeline — query → embed → retrieve → rerank → generate** —
  the per-request pipeline run at answer time: embed the incoming
  query, retrieve the most similar stored chunks, optionally rerank
  them for precision, then generate an answer grounded in the survivors.
- **Naive RAG vs Advanced RAG vs Modular RAG** — three levels of RAG
  sophistication. **Naive RAG** is the pipeline above run exactly once,
  straight through (embed → top-K retrieve → generate). **Advanced
  RAG** adds targeted improvements around that same core loop — query
  rewriting, reranking (§7), filtering — without changing its basic
  shape. **Modular RAG** goes further, treating retrieval/generation as
  swappable, potentially reorderable or repeatable modules (e.g. the
  retrieve-generate-critique loops of Self-RAG/CRAG in §6), rather than
  one fixed linear pipeline.

## 2. Document Ingestion and Parsing

- **Loading PDFs — PyMuPDF, pdfplumber, LlamaParse** — three common
  Python tools for extracting text (and sometimes layout/tables) from
  PDF files, differing in speed, layout-fidelity, and how well they
  handle complex/scanned PDFs.
- **Loading Word documents — python-docx** — reads `.docx` files'
  paragraphs, headings, and tables directly from their XML structure.
- **Loading HTML — BeautifulSoup, Unstructured** — BeautifulSoup parses
  raw HTML into a navigable tree for extracting text out of tags;
  Unstructured is a higher-level library that handles many document
  formats (including HTML) with built-in cleanup of boilerplate.
- **Loading CSV and Excel — pandas** — tabular data is loaded as a
  DataFrame, then usually converted to text (row-by-row or
  table-summarized) before it can be chunked and embedded like prose.
- **Loading code files — tree-sitter** — a parsing library that builds
  a real syntax tree for source code, enabling structure-aware
  splitting (e.g. by function/class) instead of treating code as plain
  text.
- **OCR for scanned documents — Tesseract, AWS Textract** — scanned
  documents have no extractable text layer, so OCR converts the image
  of each page into text first; Tesseract is a free local OCR engine,
  AWS Textract is a paid hosted service with stronger layout/table
  detection.
- **Table extraction strategies** — tables need special handling
  because naive text extraction loses their row/column structure;
  strategies range from preserving tables as Markdown, to extracting
  them as structured data separate from surrounding prose.

## 3. Text Chunking Strategies

- **Fixed-size chunking** — cuts text at a fixed character or token
  count, with no regard for structure — simplest to implement, but
  frequently slices a sentence or idea in half.
- **Recursive character text splitter** — tries the "biggest" separator
  first (blank lines between paragraphs), falling back to smaller
  separators (line breaks, sentence-ending periods, spaces) only for
  pieces still too large, so chunks stay as structurally coherent as
  possible while respecting a size limit.
- **Sentence-aware chunking** — splits strictly on sentence boundaries
  (via a sentence tokenizer), guaranteeing no chunk ever cuts a
  sentence in half, then groups sentences up to a size limit.
- **Semantic chunking** — splits based on where the *meaning* shifts
  (e.g. embedding consecutive sentences and cutting where similarity
  drops), rather than any fixed structural or size rule.
- **Markdown-aware chunking** — splits along Markdown structure
  (headers, list boundaries, code fences) so a chunk doesn't straddle,
  say, two unrelated sections under different headers.
- **Code-aware chunking** — splits source code along syntactic
  boundaries (functions, classes) using a real parser (§2's
  tree-sitter), rather than by character count.
- **Chunk size selection — 256, 512, 1024 tokens** — the target chunk
  size trades off two failure modes: too small loses surrounding
  context per chunk, too large dilutes a chunk's embedding with
  irrelevant content and wastes prompt budget at generation time; these
  three sizes are common practical defaults.
- **Chunk overlap — purpose and typical sizes** — repeating a small
  tail of one chunk at the start of the next (typically 10-20% of chunk
  size) so information sitting right at a chunk boundary isn't split
  across two chunks with neither containing the full context.
- **Metadata preservation per chunk** — carrying forward info like
  source filename, position within the document, and section/heading
  alongside each chunk's text, so retrieval results can be filtered,
  cited, and traced back to their origin.

## 4. Building the Indexing Pipeline

- **End-to-end pipeline architecture** — the concrete implementation of
  §1's ingest → parse → chunk → embed → store sequence: a series of
  discrete, ideally idempotent stages so any one stage can be rerun or
  swapped without redoing the others.
- **Batch embedding for efficiency** — sending many chunks to the
  embedding model in one batched call instead of one call per chunk,
  since embedding models are typically far more efficient (throughput
  per second) when given a batch.
- **Deduplication of chunks** — detecting and skipping chunks that are
  exact or near-duplicates of already-indexed content, avoiding wasted
  storage and redundant/repetitive retrieval results.
- **Incremental indexing** — adding or updating only the chunks
  belonging to new/changed documents instead of rebuilding the entire
  index from scratch every time the corpus changes.
- **Metadata schema design** — deciding upfront what metadata fields
  every chunk will carry (source, timestamp, category, etc.), since
  filtering and citation at query time can only use fields that were
  captured during indexing.
- **Storing vector and full document text** — persisting both the
  embedding (for similarity search) and the original chunk text (so a
  retrieved match can actually be shown/cited, not just scored).

## 5. Building the Retrieval Pipeline

- **Embedding the user query** — running the same embedding model used
  during indexing over the incoming query, so it lands in the same
  vector space as the stored chunks.
- **Top-K retrieval and choosing K** — scoring the query vector against
  every stored chunk vector and returning the K highest-scoring
  matches; too small a K risks missing relevant context, too large
  dilutes the generation prompt with marginally-relevant chunks.
- **Similarity score thresholds** — discarding retrieved chunks below a
  minimum similarity score, so a query with no genuinely relevant
  content doesn't still force-feed the model its "closest" (but
  irrelevant) matches.
- **Metadata pre-filtering** — narrowing the candidate set by metadata
  (Module 5 §1) before running similarity search, when the query has a
  known constraint (e.g. only search a specific document category).
- **Post-filtering retrieved results** — running similarity search
  first, then discarding results that fail a metadata or content
  condition afterward — simpler than pre-filtering but can return fewer
  than K usable results.

## 6. Advanced Retrieval Techniques

- **Multi-query retrieval** — having the LLM generate several
  reworded/rephrased versions of the user's query, retrieving for each,
  and merging the results — catches relevant chunks that the original
  query's exact phrasing alone would have missed.
- **Self-RAG** — the model reflects on whether the retrieved context
  actually supports answering the query, and can decide to retrieve
  again, skip retrieval, or flag low confidence, rather than always
  blindly generating from whatever was retrieved.
- **Corrective RAG (CRAG)** — explicitly grades retrieved chunks for
  relevance after retrieval, and if they're judged insufficient, falls
  back to another source (e.g. a web search or a broader re-retrieval)
  before generation instead of proceeding with poor context.

## 7. Reranking

- **Why reranking improves precision** — an initial retrieval pass
  (embedding similarity) is fast but approximate; a reranking step
  re-scores just the top candidates with a more expensive, more
  accurate relevance model, pushing the truly best matches to the top
  before they reach generation.
- **Cross-encoder vs bi-encoder** — a bi-encoder (used for the initial
  retrieval) embeds the query and each document independently, then
  compares vectors — fast, scales to millions of documents. A
  cross-encoder feeds the query and a candidate document into the model
  *together*, letting it directly attend to both — much more accurate,
  but too slow to run over an entire corpus, which is why it's used
  only to rerank a small shortlist.
- **Cohere Rerank API** — a hosted cross-encoder-style reranking
  service from Cohere.
- **BGE Reranker (open-source)** — an open-weight cross-encoder
  reranker model that can be run locally.
- **FlashRank — lightweight local reranker** — a small, fast,
  dependency-light open-source reranker designed to run locally without
  a GPU.
- **Reciprocal Rank Fusion (RRF)** — a rank-combination technique that
  merges multiple ranked result lists (e.g. from Module 4's BM25 +
  vector hybrid search) by scoring each item on `1 / (rank + constant)`
  summed across lists, favoring items that rank well across multiple
  methods without needing their raw scores to be comparable.

## 8. Augmentation and Generation

- **RAG prompt template structure** — retrieval only gets you
  *candidate* context; the prompt is what turns that into a grounded
  answer instead of the model just improvising. A typical template
  numbers each retrieved chunk, labels it with its source, and
  instructs the model on how to use that context.
- **Handling missing context gracefully** — explicitly instructing the
  model to say it doesn't know when the retrieved context doesn't
  actually answer the query, instead of falling back on its own
  (unverified) training knowledge or inventing an answer.
- **Citing sources in generated answers** — having the model reference
  which specific retrieved chunk(s) or source document(s) it drew each
  part of its answer from, so the answer can be checked against its
  sources.
- **Multi-document synthesis** — combining information spread across
  several retrieved chunks (possibly from different source documents)
  into one coherent answer, rather than only ever using a single chunk.
- **Grounding instructions to prevent hallucination** — explicit
  prompt-level rules (e.g. "answer only from the provided context")
  that constrain the model to the retrieved material instead of letting
  it blend in ungrounded, possibly incorrect claims.

## 9. RAG Evaluation

- **RAGAS framework** — an evaluation framework purpose-built for RAG
  pipelines, scoring both the retrieval and generation stages via LLM-
  judged metrics rather than requiring hand-labeled ground truth for
  every question:
  - **Faithfulness** — whether the generated answer's claims are
    actually supported by the retrieved context (the generation-side
    check against hallucination).
  - **Answer Relevancy** — whether the generated answer actually
    addresses the question asked, independent of whether it's factually
    grounded.
  - **Context Recall** — whether the retrieved context contains all the
    information actually needed to answer the question (a retrieval-side
    check for missed relevant chunks).
  - **Context Precision** — whether the retrieved context is free of
    irrelevant chunks that shouldn't have been retrieved at all.
- **Building a golden QA dataset** — a hand-curated set of
  question/expected-answer (and ideally expected-source) pairs, used as
  ground truth to measure retrieval and generation quality
  consistently over time.
- **TruLens evaluation framework** — another LLM-application evaluation
  library, providing tracing plus configurable feedback functions
  (including RAG-specific ones) similar in spirit to RAGAS.
- **Chunking ablation studies** — systematically varying chunking
  parameters (strategy, size, overlap) and re-measuring retrieval/
  answer quality, to find which chunking choice actually performs best
  for a given corpus instead of guessing.
- **Retrieval@K metrics** — measuring whether/how often a relevant
  chunk appears within the top K retrieved results (e.g. Recall@K,
  Precision@K), independent of what generation does with it.

---

## 10. One sub-topic per topic

Each stage below uses one sub-topic, chosen for being the most commonly used, practical default:

| Topic | Roadmap sub-topics available | Sub-topic used here |
|---|---|---|
| **Ingestion** | PDF (PyMuPDF/pdfplumber/LlamaParse); Word; HTML; CSV/Excel; code files; OCR; table extraction | **Loading PDFs** — one PDF file parsed with `pypdf` (`pdf_loader.py`), alongside the plain-Markdown files read directly |
| **Indexing** | fixed-size / recursive / sentence-aware / semantic / markdown / code-aware chunking; dedup; incremental indexing; batch embedding | **Recursive character text splitter** — split on paragraph → line → sentence → word boundaries, whichever keeps chunks intact; store chunk text + vector + metadata |
| **Retrieving** | similarity thresholds; metadata pre/post-filtering; multi-query retrieval; Self-RAG; Corrective RAG (CRAG) | **Top-K retrieval** — embed the query, rank all chunks by cosine similarity, take the K best |
| **Augmentation** | prompt template structure; handling missing context; citing sources; multi-document synthesis; grounding instructions | **RAG prompt template structure** — numbered, source-labeled context blocks + an explicit "answer only from this context, cite what you used" instruction |

## 11. Why each chosen technique, briefly

**Loading PDFs with pypdf.** One PDF file is included specifically to 
exercise the document ingestion step for real: `pdf_loader.py` 
extracts its text with `pypdf`, a pure-Python, open-source PDF library 
with no system dependency (unlike PyMuPDF, which needs a compiled 
extension, or LlamaParse, which is a hosted API) — consistent with this 
repo's local-tools-only convention.

**Recursive character text splitter.** A fixed-size splitter cuts text
at a character count with no regard for structure, frequently slicing a
sentence in half. The recursive splitter instead tries the "biggest"
separator first (blank lines between paragraphs), and only falls back to
smaller separators (line breaks, sentence-ending periods, spaces) for
pieces still too large — so chunks stay as semantically coherent as
possible while still respecting a size limit. A small overlap between
consecutive chunks (the tail of one chunk repeated at the start of the
next) avoids losing context for anything that happens to fall right at a
chunk boundary.

**Top-K retrieval.** The simplest way to turn "which chunks are relevant"
into "which chunks to hand the model": embed the query with the same
model used to embed the documents, score every stored chunk by cosine
similarity, and take the highest-scoring K. No filtering, no query
rewriting, no iterative correction loop — just the core mechanism every
fancier retrieval technique builds on top of.

**RAG prompt template structure.** Retrieval only gets you *candidate*
context — the prompt is what turns that into a grounded answer instead
of the model just improvising. The template used here numbers each
surviving chunk, labels it with its source file, and instructs the model
to answer only from that context, admit when the answer isn't there, and
cite which numbered chunk(s) it used — the minimum structure needed to
get an answer you can actually check against its sources.

---

## 12. Hands-on: local RAG mini app

A small "engineering knowledge base" over 5 short Markdown documents
plus 1 PDF in `app/docs/` (deployment process, incident response
runbook, onboarding checklist, coding standards, on-call rotation, and
a security policy as a PDF).

### 12.1 Files

| File | Role |
|---|---|
| `docs/*.md` | 5 of the 6 source documents to index — a small fictional engineering wiki. |
| `docs/security_policy.pdf` | The 6th source document — the same knowledge base, but as a PDF, to exercise real PDF ingestion (§2). |
| `pdf_loader.py` | Ingestion — extracts text from a PDF with `pypdf` (`load_pdf`). |
| `chunking.py` | Indexing — recursive character text splitter (`recursive_split`). |
| `embedder.py` | Shared embedding function — embeds through Gemini's `gemini-embedding-001` model via `../../common/llm_client.py`. |
| `ingest.py` | Indexing pipeline: load docs (Markdown + PDF) → chunk → embed → persist to `index_store/` (JSON + `.npy`). Idempotent — re-running reuses an existing index instead of rebuilding it. |
| `retrieve.py` | Retrieval pipeline: embed the query, top-K cosine similarity search over the stored chunk vectors. |
| `generate.py` | Augmentation + generation: builds the RAG prompt template, then answers it with Gemini's `gemini-3.1-flash-lite` model (`../../common/llm_client.py`). |
| `app.py` | Wires the whole pipeline together and runs the example questions in `QUESTIONS` end to end. |

### 12.2 Step-by-step: what happens when you run `python app.py`

1. **`build_index()` (indexing, runs once).**
   - `load_documents()` reads every `.md` file in `docs/` as plain text, and every `.pdf` file's text via `pdf_loader.load_pdf` (which runs `pypdf.PdfReader(...).pages[i].extract_text()` over every page and joins them) — two ingestion paths feeding the same downstream pipeline.
   - `build_chunks()` runs each document through `recursive_split(text, chunk_size=400, chunk_overlap=60)` and tags every resulting piece with an id (`"<filename>::<chunk index>"`), its source filename, and its position within the document. On the actual 6 sample docs this produces **20 chunks** total (verified: `coding_standards.md` → 3, `deployment_process.md` → 4, `incident_response.md` → 3, `onboarding_checklist.md` → 4, `oncall_rotation.md` → 3, `security_policy.pdf` → 3).
   - Each chunk's text is embedded with `embedder.embed(...)` (L2-normalized `gemini-embedding-001` vectors via the Gemini API), and the result is written to `index_store/chunks.json` (text + metadata) and `index_store/embeddings.npy` (the vectors) — "storing vector and full document text" from §4.
   - If `index_store/` already has both files, this step is skipped and the existing index is reused — running `app.py` twice does not re-embed or duplicate anything.

2. **For each example question in `QUESTIONS`, `answer_question(query)` runs the query pipeline:**
   - **Retrieve** — `top_k_retrieve(query, k=3)` embeds the query with the same model used for the documents, computes cosine similarity (a plain dot product, since every vector is already unit-length) against all 20 stored chunk vectors, and returns the 3 highest-scoring chunks with their similarity score.
   - **Augment** — `build_prompt(query, retrieved)` numbers those 3 chunks, labels each with its source filename, and wraps them in the grounding instructions described in §11.
   - **Generate** — `generate_answer(prompt)` sends that prompt to Gemini's `gemini-3.1-flash-lite` model and returns its answer.
   - The app prints, for each question: the retrieved chunk ids + similarity scores, the generated answer, and which source files it was allowed to cite — so you can see retrieval's effect on the final result, not just the end answer.

3. **Example questions used** (`app.py`'s `QUESTIONS` list) are deliberately answerable from the sample docs, except one that deliberately checks the "don't just agree with the user" grounding behavior and one that's off-topic entirely:
   - *"What should I do if there's a production incident?"* → should retrieve from `incident_response.md`.
   - *"How do I set up my dev environment in my first week?"* → should retrieve from `onboarding_checklist.md`.
   - *"Can I merge my own pull request without a review?"* → the docs say the opposite (every PR needs at least one approval); a good grounded answer says no and cites `coding_standards.md`, rather than hallucinating a "yes."
   - *"How often are API keys rotated?"* → answerable only from `security_policy.pdf`, the PDF-ingested document — a grounded answer citing it confirms text extracted from the PDF made it all the way through chunking, embedding, and retrieval intact.
   - *"How do whales communicate?"* → nothing in the engineering-wiki corpus covers this topic, so the top-3 retrieved chunks are all irrelevant; a properly grounded answer says it doesn't know rather than inventing one from unrelated context — this checks the "admit when the answer isn't there" instruction in the prompt template (§11), the same grounding behavior the pull-request question demonstrates, just via an off-topic question instead of a contradicted one.

### 12.3 Setup

```bash
cd app
pip install -r ../requirements.txt
python app.py
```

Needs a Gemini API key in `common/.env` (the shared folder next to the
module folders) as `GEMINI_API_KEY=...` (keys
come from https://aistudio.google.com/apikey) — `python-dotenv` loads
it automatically. Both models (`gemini-3.1-flash-lite` for generation,
`gemini-embedding-001` for embeddings) are called through that one key.

### 12.4 Runtime dependencies

Every piece of this pipeline runs through the Gemini API
(`../../common/llm_client.py`), `pypdf`, or the standard library —
`embedder.py` embeds through Gemini's `gemini-embedding-001` model, `generate.py`
generates through `gemini-3.1-flash-lite`, `pdf_loader.py` extracts PDF text with
`pypdf`, and `chunking.py`'s `recursive_split` is pure Python.
