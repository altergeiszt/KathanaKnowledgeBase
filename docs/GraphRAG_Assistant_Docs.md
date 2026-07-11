# GraphRAG Personal AI Assistant — Documentation

> Version 1.1 · June 2026 · Status: Draft
> _Reconciled 2026-07-03 against the implementation (embedded SurrealKV model). Full-source listings in TDD §1.3/§2/§3 are retained for design intent but carry ⚠️ banners where they diverge from the shipped code, which is authoritative._

---

## Table of Contents

1. [Software Requirements Specification (SRS)](#software-requirements-specification)
2. [Architecture Document](#architecture-document)
3. [Technical Design Document (TDD)](#technical-design-document)

---

# Software Requirements Specification

## 1. Introduction

### 1.1 Purpose

This document specifies the software requirements for a personal GraphRAG-powered AI assistant designed to index, retrieve, and reason over a private technical library. The system enables natural language querying over a curated collection of software development, mathematics, and self-help books stored as PDFs and EPUBs.

### 1.2 Scope

The system — referred to as the **GraphRAG Assistant** — encompasses:

- A document ingestion pipeline for PDF and EPUB sources
- Content-aware extraction and chunking strategies per content type
- A custom SurrealDB storage adapter for LightRAG
- A hybrid retrieval layer combining graph traversal and vector search
- A local LLM inference stack via Ollama
- A REST API bridge and OpenWebUI front-end
- A future integration path into the GraphNotes Tauri desktop application

### 1.3 Definitions and Acronyms

| Term | Definition |
|------|------------|
| **GraphRAG** | Graph-based Retrieval-Augmented Generation — a retrieval architecture that builds a knowledge graph from documents and uses it alongside vector search to answer queries |
| **LightRAG** | An open-source GraphRAG framework by HKUDS offering lower LLM call overhead than Microsoft GraphRAG |
| **SurrealDB** | A multi-model database supporting document, graph, vector, and key-value storage in a single engine, written in Rust |
| **KV Storage** | Key-Value storage used by LightRAG to cache entity summaries and LLM responses |
| **Vector Storage** | Storage for embedding vectors enabling semantic similarity search |
| **Graph Storage** | Storage for the entity-relationship knowledge graph built during ingestion |
| **EPUB** | Electronic Publication — a common ebook format with structured XML content |
| **CUDA** | Compute Unified Device Architecture — NVIDIA's GPU parallel computing platform |
| **Ollama** | A local LLM inference server that manages model loading and serves an OpenAI-compatible API |
| **OpenWebUI** | An open-source web interface for interacting with local LLM backends |
| **GraphNotes** | A personal Tauri 2.0 desktop application using Rust, React/TypeScript, and SurrealDB, currently under development |

### 1.4 Overview

This document is organized as follows: Section 2 covers the overall system description and constraints. Section 3 defines functional requirements. Section 4 covers non-functional requirements. Section 5 addresses future integration considerations with GraphNotes.

---

## 2. Overall Description

### 2.1 Product Perspective

The GraphRAG Assistant is a standalone personal knowledge system designed for a single local user. It is intentionally built as a self-contained pipeline — separate from GraphNotes — to validate the extraction, retrieval, and LLM interaction design before integration. The system runs entirely on local hardware except for optional API-based LLM calls during querying.

The SurrealDB storage adapter developed in this phase is a direct reuse artifact: the same adapter will serve as the storage backend when the assistant feature is integrated into GraphNotes. This is a primary architectural constraint.

### 2.2 Product Functions

At a high level, the system shall:

- Ingest and normalize PDFs and EPUBs into clean text
- Apply content-type-aware chunking and code block handling
- Deduplicate content across sources before indexing
- Generate embeddings locally using GPU-accelerated sentence transformers
- Build and persist a knowledge graph via LightRAG using a custom SurrealDB backend
- Accept natural language queries and return grounded, cited answers
- Support multiple query modes: local, global, hybrid, and naive

### 2.3 User Characteristics

The system has a single user: a technically proficient software developer with strong Rust and Python skills, familiarity with SurrealDB, and an interest in AI/ML tooling. The user is comfortable with CLI tooling and does not require a simplified UX. Preference is for local, cost-controlled infrastructure.

### 2.4 Constraints

| Constraint | Detail |
|------------|--------|
| **Hardware** | NVIDIA RTX 4080 Super (16 GB VRAM), local machine in Saskatchewan, Canada |
| **Cost** | Minimize LLM API costs; prefer local inference for ingestion; API calls only acceptable at query time |
| **Speed vs Cost** | Speed is a secondary concern; the pipeline may run slowly overnight to reduce cost |
| **Storage Backend** | SurrealDB must be used as the unified storage backend to ensure GraphNotes compatibility |
| **LLM (Ingestion)** | Qwen2.5:14B via Ollama — local inference only, no API calls during ingestion |
| **LLM (Query)** | Claude Haiku via API, or Qwen2.5:14B locally if cost is a concern |
| **No RAPIDS** | RAPIDS/cuGraph not used; CUDA acceleration limited to sentence-transformers embeddings |
| **Python Pipeline** | Ingestion pipeline implemented in Python using asyncio and multiprocessing |

### 2.5 Assumptions and Dependencies

- SurrealDB Python SDK `1.0.8` is used, providing the `AsyncSurreal()` factory and embedded `surrealkv://` connections (no separate SurrealDB server process). The originally-assumed v2.0+ SDK is **not** used — see Architecture §3 and TDD §1.
- LightRAG is pinned to `1.3.9`; its storage adapter interface (`BaseKVStorage`, `BaseVectorStorage`, `BaseGraphStorage`, `BaseDocStatusStorage`) is assumed stable only within that pin. Upgrades require re-validating the adapter.
- Ollama supports Qwen2.5:14B at Q4 quantization within 16 GB VRAM
- The ebook library (~14 GB) is accessible on local disk in EPUB and PDF format
- `docling` is used as the primary extraction tool for PDFs due to its superior layout handling

---

## 3. Functional Requirements

### 3.1 Ingestion Pipeline

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-I-01 | The system shall extract plain text from PDF files using docling, preserving section and paragraph structure | High | Core |
| FR-I-02 | The system shall extract chapter-level text from EPUB files using ebooklib and BeautifulSoup | High | Core |
| FR-I-03 | The system shall strip code blocks from software development books during extraction and index the surrounding prose only. Code blocks are detected and removed from the chunk text; they are **not** forwarded to the index as `[CODE]`/`[PROSE]` tagged metadata (that forwarding was dropped — the index is prose-only) | Medium | Core |
| FR-I-04 | The system shall extract prose-only content from mathematics books, skipping formula notation that fails to convert cleanly | Medium | Core |
| FR-I-05 | The system shall perform near-duplicate detection across all extracted text using MinHash LSH (datasketch) prior to indexing | High | Cost |
| FR-I-06 | The system shall use multiprocessing for PDF and EPUB extraction (CPU-bound) and asyncio for LLM API calls (I/O-bound) | Medium | Performance |
| FR-I-07 | The system shall checkpoint ingestion progress via per-document status tracking (`SurrealDBDocStatusStorage`), so that re-running the pipeline skips documents already marked processed and resumes the remainder | High | Reliability |
| FR-I-08 | The system shall support four text chunking strategies: Fix, Recursive, Vector, and Paragraph, configurable per content type | Medium | Core |

> **FR-I-08 status:** standing commitment, **not yet implemented**. The current pipeline always uses a single `semantic_text_splitter.TextSplitter`. Retained as a requirement per project decision (2026-07-03).

### 3.2 Embedding and Indexing

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-E-01 | The system shall generate text embeddings using a local sentence-transformers model with CUDA acceleration | High | Core |
| FR-E-02 | The system shall store all vector embeddings in SurrealDB using HNSW vector indexes | High | Core |
| FR-E-03 | The system shall build a LightRAG knowledge graph using Qwen2.5:14B via Ollama for entity and relationship extraction | High | Core |
| FR-E-04 | The system shall persist the knowledge graph (entities, relationships, community summaries) in SurrealDB | High | Core |
| FR-E-05 | The system shall store LLM response cache and document status in SurrealDB KV storage | Medium | Core |
| FR-E-06 | The system shall support incremental document insertion without full re-ingestion of the corpus | Medium | Core |

### 3.3 Query and Retrieval

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-Q-01 | The system shall support LightRAG query modes: local, global, hybrid, naive, and mix | High | Core |
| FR-Q-02 | The system shall route queries to Claude Haiku (API) or Qwen2.5:14B (local) based on configuration | Medium | Core |
| FR-Q-03 | The system shall return source citations alongside answers, indicating the originating document and section | Low | Future/Optional |
| FR-Q-04 | The system shall expose a REST API via FastAPI for integration with OpenWebUI | High | Core |
| FR-Q-05 | The system shall support streaming responses for real-time display in OpenWebUI | Medium | UX |
| FR-Q-06 | The system shall support independent configuration of the QUERY-role model (Ollama or Anthropic, selectable at query time). Fully independent per-role models for EXTRACT and KEYWORDS are **future/optional** — currently EXTRACT and KEYWORDS share one Ollama model | Low | Future/Optional |

### 3.4 SurrealDB Storage Adapter

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-S-01 | The adapter shall implement `BaseKVStorage` for LightRAG entity/summary and LLM cache storage | High | Core |
| FR-S-02 | The adapter shall implement `BaseVectorStorage` for embedding storage and HNSW similarity search | High | Core |
| FR-S-03 | The adapter shall implement `BaseGraphStorage` for knowledge graph node and edge operations | High | Core |
| FR-S-04 | The adapter shall implement `BaseDocStatusStorage` for document processing state tracking | Medium | Core |
| FR-S-05 | The adapter shall use the surrealdb Python SDK `1.0.8` (`AsyncSurreal()` factory) with async/await throughout, over an embedded `surrealkv://` connection | High | Core |
| FR-S-06 | The adapter shall support LightRAG workspace/namespace isolation for multi-tenant use (future GraphNotes) | Medium | GraphNotes |
| FR-S-07 | The adapter shall register four separately-named storage classes — `SurrealDBKVStorage`, `SurrealDBVectorStorage`, `SurrealDBGraphStorage`, `SurrealDBDocStatusStorage` — into LightRAG's storage registry. Registration is performed by a one-time setup script, `patch_lightrag.py`, which copies `surrealdb_impl.py` into the installed `lightrag/kg/` package and edits `kg/__init__.py` on disk (`STORAGES`, `STORAGE_IMPLEMENTATIONS`, `STORAGE_ENV_REQUIREMENTS`). It is idempotent and safe to re-run | High | Core |

### 3.5 Front-End and API

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-F-01 | OpenWebUI shall connect to the system via a FastAPI bridge implementing an Ollama-compatible API | High | Core |
| FR-F-02 | The FastAPI bridge shall handle conversation history and pass it into LightRAG retrieval (`QueryParam.conversation_history`, `history_turns=3`) on each turn, so follow-up queries are resolved against prior turns — not merely appended to the final generation prompt | High | Core |
| FR-F-03 | The system shall provide a LightRAG knowledge graph visualization endpoint compatible with the built-in LightRAG WebUI | Low | Optional |

---

## 4. Non-Functional Requirements

### 4.1 Performance

- Ingestion pipeline may run overnight; no real-time constraint on indexing speed
- Query response latency should be under 10 seconds for local Ollama queries
- Embedding generation should utilize CUDA; CPU fallback is acceptable for development

### 4.2 Cost

- LLM API usage during ingestion must be zero; all ingestion-phase LLM calls use Ollama
- Query-phase API costs (Claude Haiku) should be minimized by keeping retrieved context concise
- No RAPIDS or external cloud GPU services shall be used

### 4.3 Maintainability

- The ingestion pipeline shall be modular: extraction, chunking, deduplication, and indexing are independent stages
- The SurrealDB adapter shall be implemented in a single file (`surrealdb_impl.py`) following LightRAG's existing adapter conventions
- All configuration (model names, DB connection strings, chunking strategy) shall be driven by environment variables or a `.env` file

### 4.4 Portability

- The SurrealDB adapter shall be reusable in GraphNotes without modification to its core logic
- The ingestion pipeline shall be runnable on any machine with Python 3.11+ and CUDA optional

### 4.5 Security

- All data remains local; no telemetry or external data transmission except for optional API calls
- API keys shall be stored in environment variables, never hardcoded

---

## 5. Future GraphNotes Integration Considerations

The following design decisions are informed by the planned integration of this assistant into GraphNotes, a Tauri 2.0 desktop application using Rust, React/TypeScript, and SurrealDB.

### 5.1 Storage Layer Portability

SurrealDB is the shared persistence layer between the standalone pipeline and GraphNotes. The adapter developed here will be reused directly. When migrating, any PostgreSQL fallback used during development is replaced with SurrealDB — the rest of the pipeline is unchanged.

### 5.2 Migration Path

- **Phase 1 (current):** Standalone Python pipeline with SurrealDB backend
- **Phase 2:** FastAPI bridge wrapped as a sidecar service callable from GraphNotes Rust backend
- **Phase 3:** Native Rust integration using SurrealDB SDK directly, retiring the Python bridge

### 5.3 Data Schema Continuity

SurrealDB table names, field conventions, and HNSW index definitions used in this pipeline should be documented and kept stable, as GraphNotes will query the same database directly. Schema changes post-integration will require coordinated migrations.

---

# Architecture Document

## 1. Architecture Overview

The GraphRAG Assistant is structured as three loosely coupled subsystems: an ingestion pipeline, a storage and retrieval layer, and a query interface. This separation ensures that each subsystem can be developed, tested, and replaced independently — and critically, that the storage layer migrates cleanly into GraphNotes without rearchitecting the retrieval or query logic.

### 1.1 High-Level System Layers

| Layer | Subsystem | Primary Tools |
|-------|-----------|---------------|
| **Ingestion** | Extraction, Chunking, Deduplication, Embedding | docling, ebooklib, semantic-text-splitter, datasketch, sentence-transformers |
| **Storage & Retrieval** | KV Store, Vector Index, Knowledge Graph, LightRAG Engine | SurrealDB (custom adapter), LightRAG (HKUDS) |
| **Query Interface** | LLM Inference, Orchestration, API, Front-End | Ollama (Qwen2.5:14B), Claude Haiku, FastAPI, LlamaIndex, OpenWebUI |

### 1.2 Data Flow Summary

**Ingestion flow (one-time, batch):**

1. Raw EPUB/PDF files are read from disk
2. `docling` (PDF) and `ebooklib` (EPUB) extract structured text
3. Content-type rules strip or tag code blocks
4. `semantic-text-splitter` chunks prose into semantically coherent segments
5. `datasketch` MinHash deduplicates near-identical chunks across sources
6. `sentence-transformers` (CUDA) generates embeddings
7. LightRAG + Qwen2.5:14B via Ollama performs entity/relationship extraction
8. All artifacts (embeddings, graph, KV cache, doc status) are persisted to SurrealDB

**Query flow (per user turn):**

1. User submits query via OpenWebUI
2. FastAPI bridge receives request and passes to LlamaIndex orchestration layer
3. LightRAG retrieves relevant graph nodes and vector-similar chunks from SurrealDB
4. Retrieved context is assembled and sent to LLM (Haiku or Qwen2.5:14B)
5. Response streamed back to OpenWebUI with source citations

---

## 2. Ingestion Pipeline Architecture

### 2.1 Extraction Stage

#### 2.1.1 PDF Extraction

PDF extraction is CPU-bound. The pipeline uses Python multiprocessing to parallelize extraction across documents. `docling` is the primary tool due to its superior handling of complex PDF layouts, multi-column text, and mixed content.

Processing rules by content type:

| Content Type | Tool | Strategy | Code Blocks |
|--------------|------|----------|-------------|
| Software Dev Books | docling | Chapter/section chunking | Tagged `[CODE]`, kept as associated context |
| Math Books (~10) | docling | Prose-only, formula regions skipped | N/A — formulas excluded |
| Self-Help Books | docling | Full content, semantic chunking | N/A — no code content |

#### 2.1.2 EPUB Extraction

EPUBs are structured as XML internally. `ebooklib` reads the manifest and spine to extract chapters in order. BeautifulSoup strips HTML tags and normalizes whitespace. Chapter-level extraction is preferred over page-level, producing cleaner semantic chunks that respect the author's intended structure.

### 2.2 Chunking Stage

The system uses LightRAG's built-in chunking strategies, configurable per content type via environment variable:

- **Paragraph** — default for prose-heavy content (self-help, math prose)
- **Recursive** — for software dev books with mixed code/prose structure
- **Vector** — for content where semantic coherence is critical
- **Fix** — not used in production; available for debugging

Code blocks tagged as `[CODE]` during extraction are excluded from primary chunking but stored as metadata associated with the adjacent `[PROSE]` chunk. This allows the query layer to optionally surface code examples when the query semantics indicate the user wants implementation details.

### 2.3 Deduplication Stage

The 14 GB library contains significant textbook overlap (same subject, different publishers). Deduplication runs after extraction but before embedding generation to avoid wasting compute. `datasketch` MinHash LSH is used:

- Each chunk is shingled (character 5-grams)
- MinHash signatures are computed and indexed in an LSH forest
- Chunks with Jaccard similarity above 0.85 are deduplicated, keeping the first occurrence
- Estimated reduction: 30–50% fewer chunks after deduplication on the math library subset

### 2.4 Embedding Generation

`sentence-transformers` runs locally on the RTX 4080 Super using CUDA. The embedding model must be chosen before first ingestion and kept consistent — changing models requires re-indexing all vectors.

Recommended starting models:
- `all-MiniLM-L6-v2` (384-dim, fast)
- `nomic-embed-text` (768-dim, higher quality)

Embeddings are stored directly in SurrealDB with HNSW indexing. The embedding dimension is fixed at schema creation time. HNSW parameters (`ef_construction`, `m`) should be tuned for the expected corpus size (~500K–2M chunks estimated).

### 2.5 Concurrency Model

| Stage | Concurrency Model | Reason |
|-------|-------------------|--------|
| PDF/EPUB Extraction | `multiprocessing.Pool(processes=extraction_workers)` | CPU-bound; bypasses GIL. Worker count is **capped** by the `extraction_workers` config (default 8, override `EXTRACTION_WORKERS`) rather than `cpu_count()` — each worker runs a full docling+EasyOCR pipeline that can need several GB, so an uncapped pool risks OOM |
| Chunking | In-process (inside each extraction worker) | Chunking runs as part of `extract_document()` |
| Deduplication | Single process (MinHash LSH) | Runs once over the combined chunk set after extraction |
| Embedding Generation | Batched GPU inference | GPU parallelism via sentence-transformers |
| LightRAG Entity Extraction (Ollama) | `asyncio` + LightRAG internal async | I/O-bound; GIL releases on network wait |
| SurrealDB Writes | `asyncio`, **serialized** via an internal `_query_lock` | Embedded SurrealKV raises write-write transaction conflicts under concurrent writes on one connection; there is no network round-trip cost to serializing a local store |

---

## 3. Storage Architecture — SurrealDB

SurrealDB serves as the single unified storage backend, replacing the combination of Qdrant (vector), Neo4j (graph), and a KV store that many LightRAG deployments require. This unification is a core architectural decision driven by GraphNotes compatibility.

### 3.1 LightRAG Storage Interface

LightRAG defines four abstract base classes in `lightrag/base.py` that all storage backends must implement. The SurrealDB tables use short prefixes (`kv_`, `vec_`, `ent_`, `rel_`, `doc_status_`):

| Interface | Responsibility | SurrealDB Table(s) |
|-----------|---------------|-------------------|
| `BaseKVStorage` | Stores key-value pairs: entity summaries, LLM response cache, community reports | `kv_{namespace}` |
| `BaseVectorStorage` | Stores embedding vectors and supports HNSW ANN similarity search | `vec_{namespace}` |
| `BaseGraphStorage` | Stores knowledge graph: nodes (entities), edges (relationships), supports traversal | `ent_{namespace}`, `rel_{namespace}` |
| `BaseDocStatusStorage` | Tracks per-document ingestion state (pending, processing, processed, failed) | `doc_status_{namespace}` |

### 3.2 SurrealDB Schema Design

Tables are declared **`SCHEMALESS`** (not `SCHEMAFULL`), with `DEFINE FIELD ... IF NOT EXISTS` guiding the expected shape while allowing LightRAG to attach extra fields. `object`/metadata fields are declared `FLEXIBLE`. Record IDs are SurrealDB `RecordID`s; the adapter normalizes them to plain strings via `record::id(id)`. Examples below use the `default` namespace (the LightRAG workspace name).

> ⚠️ **Schema stability contract (SRS §5.3).** The table prefixes (`kv_`, `vec_`, `ent_`, `rel_`, `doc_status_`) and the field names below are the **frozen contract** for future GraphNotes integration, as of 2026-07-03. They intentionally differ from the earlier `kv_store_`/`vector_store_`/`entity_`/`relation_` draft names. Changing them post-integration requires a coordinated migration.

#### 3.2.1 KV Store Table

```sql
DEFINE TABLE IF NOT EXISTS kv_default SCHEMALESS;
DEFINE FIELD IF NOT EXISTS data ON kv_default FLEXIBLE TYPE object;
```

#### 3.2.2 Vector Store Table

```sql
DEFINE TABLE IF NOT EXISTS vec_default SCHEMALESS;
DEFINE FIELD IF NOT EXISTS content   ON vec_default TYPE string;
DEFINE FIELD IF NOT EXISTS embedding ON vec_default TYPE array<float>;
DEFINE FIELD IF NOT EXISTS metadata  ON vec_default FLEXIBLE TYPE option<object>;
DEFINE INDEX IF NOT EXISTS hnsw_idx  ON vec_default
  FIELDS embedding HNSW DIMENSION 384 DIST COSINE EFC 64 M 16;
```

#### 3.2.3 Entity (Graph Node) Table

```sql
DEFINE TABLE IF NOT EXISTS ent_default SCHEMALESS;
DEFINE FIELD IF NOT EXISTS entity_name ON ent_default TYPE string;
DEFINE FIELD IF NOT EXISTS entity_type ON ent_default TYPE string;
DEFINE FIELD IF NOT EXISTS description ON ent_default TYPE string;
DEFINE FIELD IF NOT EXISTS source_id   ON ent_default TYPE string;
DEFINE FIELD IF NOT EXISTS extra       ON ent_default FLEXIBLE TYPE option<object>;
```

#### 3.2.4 Relation (Graph Edge) Table

```sql
DEFINE TABLE IF NOT EXISTS rel_default SCHEMALESS;
DEFINE FIELD IF NOT EXISTS src_id      ON rel_default TYPE string;
DEFINE FIELD IF NOT EXISTS tgt_id      ON rel_default TYPE string;
DEFINE FIELD IF NOT EXISTS weight      ON rel_default TYPE float;
DEFINE FIELD IF NOT EXISTS description ON rel_default TYPE string;
DEFINE FIELD IF NOT EXISTS keywords    ON rel_default TYPE string;   -- NOTE: string, not array<string>
DEFINE FIELD IF NOT EXISTS source_id   ON rel_default TYPE string;
DEFINE INDEX IF NOT EXISTS idx_rel_src ON rel_default COLUMNS src_id;
DEFINE INDEX IF NOT EXISTS idx_rel_tgt ON rel_default COLUMNS tgt_id;
```

#### 3.2.5 Doc Status Table

```sql
DEFINE TABLE IF NOT EXISTS doc_status_default SCHEMALESS;
DEFINE FIELD IF NOT EXISTS status       ON doc_status_default TYPE string;
DEFINE FIELD IF NOT EXISTS content_hash ON doc_status_default TYPE option<string>;
DEFINE FIELD IF NOT EXISTS file_path    ON doc_status_default TYPE option<string>;
DEFINE FIELD IF NOT EXISTS track_id     ON doc_status_default TYPE option<string>;
DEFINE FIELD IF NOT EXISTS error_msg    ON doc_status_default TYPE option<string>;
DEFINE FIELD IF NOT EXISTS metadata     ON doc_status_default FLEXIBLE TYPE option<object>;
-- plus content_summary, content_length, chunks_count, chunks_list, created_at, updated_at
DEFINE INDEX IF NOT EXISTS idx_ds_status ON doc_status_default COLUMNS status;
DEFINE INDEX IF NOT EXISTS idx_ds_hash   ON doc_status_default COLUMNS content_hash;
DEFINE INDEX IF NOT EXISTS idx_ds_path   ON doc_status_default COLUMNS file_path;
```

### 3.3 Adapter Registration

Registration is performed by a **one-time setup script**, `patch_lightrag.py` (run explicitly as `python patch_lightrag.py`, e.g. via `setup.ps1`). It is not a runtime import — it physically modifies the installed LightRAG package on disk, and must be re-run after any reinstall/upgrade of `lightrag-hku`. Three idempotent steps:

1. **Copies** `surrealdb_impl.py` into the installed `lightrag/kg/` directory so it is importable as `lightrag.kg.surrealdb_impl`.
2. **Rewrites** `lightrag/kg/__init__.py` via string manipulation, adding the four class names to the `STORAGES` dict and to the per-role `STORAGE_IMPLEMENTATIONS[...]["implementations"]` lists (`KV_STORAGE`, `VECTOR_STORAGE`, `GRAPH_STORAGE`, `DOC_STATUS_STORAGE`).
3. **Adds** empty `STORAGE_ENV_REQUIREMENTS` entries for each class (the embedded adapter needs no env-var preconditions).

After patching, LightRAG resolves the classes by name exactly as for its built-in backends. Once registered, LightRAG is initialized with the SurrealDB backend:

```python
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_func,
    embedding_func=embedding_func,
    kv_storage="SurrealDBKVStorage",
    vector_storage="SurrealDBVectorStorage",
    graph_storage="SurrealDBGraphStorage",
    doc_status_storage="SurrealDBDocStatusStorage",
)
```

---

## 4. Query & Inference Architecture

### 4.1 LightRAG Query Modes

LightRAG provides five query modes, each using different retrieval strategies. The query layer selects mode based on query classification or explicit user selection:

| Mode | Strategy | Best For |
|------|----------|----------|
| `local` | Retrieves entity neighbors and directly connected relations from graph | Specific entity lookups: *'What does the Arc trait do in Rust?'* |
| `global` | Retrieves community summaries and high-level graph structure | Broad conceptual queries: *'What are the main themes across my ML books?'* |
| `hybrid` | Combines local + global graph retrieval | Multi-faceted technical questions |
| `naive` | Pure vector similarity search, no graph | Simple factual lookups where graph adds no value |
| `mix` | Graph retrieval + vector retrieval + reranker | Default production mode; highest quality |

### 4.2 LLM Role Configuration

LightRAG 2026 supports independent LLM configuration per role. The system uses this to optimize cost:

| Role | Function | Assigned Model |
|------|----------|---------------|
| `EXTRACT` | Entity/relationship extraction during ingestion | Qwen2.5:14B via Ollama (local, free) |
| `KEYWORDS` | Query keyword inference for graph lookup | Qwen2.5:14B via Ollama (shares the EXTRACT model — see note) |
| `QUERY` | Final answer generation from retrieved context | Claude Haiku via API, or Qwen2.5:14B locally (`QUERY_LLM_PROVIDER`) |
| `VLM` | Vision/multimodal (future use for diagram-heavy content) | Not configured initially |

> **Role-separation status.** Only the **QUERY** role is independently configurable today (via `QUERY_LLM_PROVIDER` = `ollama` | `anthropic` in `api.py`). EXTRACT and KEYWORDS both run through the single Ollama model built by `make_ollama_func()`; a separately-configured KEYWORDS model is **future/optional** (FR-Q-06).

### 4.3 API Bridge (FastAPI)

FastAPI exposes an Ollama-compatible REST API that OpenWebUI connects to. Key endpoints:

- `POST /api/chat` — accepts conversation history, routes to LightRAG, returns streamed response
- `GET /api/tags` — returns available 'models' (query mode configurations) to OpenWebUI
- `GET /health` — liveness check
- `GET /graph` — returns graph visualization data for LightRAG WebUI (optional)

The bridge maintains no persistent state; all conversation history is passed by the client on each request and forwarded to LightRAG as context.

### 4.4 Reranker (Future/Optional — not implemented)

> **Status:** deferred per project decision (2026-07-03). No reranker is wired in; `mix` mode currently returns LightRAG's combined graph+vector retrieval without a re-scoring pass.

The intended design: when using `mix` mode, a reranker model re-scores retrieved chunks before assembly into the context window. Recommended: `BAAI/bge-reranker-v2-m3`, run locally. This would improve relevance for mixed-content queries (e.g., a query requiring both a conceptual explanation and a code reference). Revisit if `mix`-mode answer quality proves insufficient.

---

## 5. GraphNotes Integration Architecture

The standalone pipeline is architecturally designed as Phase 1 of a two-phase delivery. Phase 2 integrates the assistant capability into GraphNotes as a native feature.

### 5.1 Phase 1 — Standalone (Current)

| Component | Technology | Notes |
|-----------|------------|-------|
| Ingestion | Python pipeline | Runs once; re-runs on library updates |
| Storage | SurrealDB — embedded SurrealKV file (`surrealkv://`) | Same on-disk database GraphNotes will open |
| LLM Inference | Ollama (local) | Shared with GraphNotes future use |
| Query API | FastAPI bridge | Replaced by native Rust bridge in Phase 2 |
| Front-End | OpenWebUI | Replaced by GraphNotes UI in Phase 2 |

### 5.2 Phase 2 — GraphNotes Integration

In Phase 2, the Python FastAPI bridge is wrapped as a sidecar process managed by the Tauri application lifecycle. The GraphNotes Rust backend calls the sidecar via HTTP on localhost. The SurrealDB instance is shared — GraphNotes manages it as a single embedded or server-mode database.

### 5.3 Phase 3 — Native Rust Integration (Optional)

If performance or distribution simplicity requires it, the Python pipeline can be replaced with a native Rust implementation using:

- `lopdf` or `pdf-extract` for PDF text extraction
- `epub` crate for EPUB parsing
- `surrealdb` Rust SDK for direct database interaction
- `candle` or `llm` crate for local embedding inference

This phase is optional and only warranted if the sidecar approach proves insufficient for the GraphNotes user experience requirements.

---

# Technical Design Document

## 1. SurrealDB Storage Adapter — `surrealdb_impl.py`

The SurrealDB adapter is the central technical deliverable. It bridges LightRAG's abstract storage interfaces and SurrealDB's multi-model capabilities, making SurrealDB a drop-in replacement for the Qdrant + Neo4j + Redis combination that production LightRAG deployments typically require.

The four classes are registered by the one-time `patch_lightrag.py` setup script (see Architecture §3.3), which copies the adapter into and edits the installed LightRAG package on disk. After patching, LightRAG's `kg/__init__.py` contains, in effect:

```python
STORAGES["SurrealDBKVStorage"]        = ".kg.surrealdb_impl"
STORAGES["SurrealDBVectorStorage"]    = ".kg.surrealdb_impl"
STORAGES["SurrealDBGraphStorage"]     = ".kg.surrealdb_impl"
STORAGES["SurrealDBDocStatusStorage"] = ".kg.surrealdb_impl"
# plus matching entries in STORAGE_IMPLEMENTATIONS[<role>]["implementations"]
```

### 1.1 File Structure

The adapter and its supporting files live at the **repo root** (not inside the installed `lightrag/` package); `patch_lightrag.py` copies the adapter into the package at setup time:

```
surrealdb_impl.py               # All four storage class implementations
patch_lightrag.py               # One-time setup: copy adapter into lightrag/kg/ + patch kg/__init__.py
ingest.py                       # Ingestion pipeline
api.py                          # FastAPI bridge
docID.py                        # stable_doc_id() helper
docling_to_content_list.py      # Future: RAG-Anything content_list converter (scaffold — see §1.4)
surreal.exe / setup.ps1         # Windows-native setup helpers
lightrag_data/graphrag.db       # Embedded SurrealKV database file (SURREALDB_PATH)
```

### 1.4 Future: RAG-Anything Integration (`docling_to_content_list.py`)

> **Status: FUTURE — scaffold only, not wired into the pipeline (as of 2026-07-03).**

`docling_to_content_list.py` is a scaffold for integrating this assistant with [RAG-Anything](https://github.com/HKUDS/RAG-Anything) (HKUDS). It converts docling output into RAG-Anything's `insert_content_list` schema (`text` / `image` / `table` / `equation` items with `page_idx`), which is a superset of the current prose-only pipeline — it would additionally capture tables, figures (extracted to image files), and math equations (retained as LaTeX rather than stripped). Its `_looks_like_code()` / formula-handling TODOs are placeholders for the existing content-type rules (FR-I-03, FR-I-04). It will be completed when the assistant is ported onto RAG-Anything's ingestion layer.

### 1.2 Environment Variables

The adapter uses an **embedded** `surrealkv://` database — there is no URL, username, or password. Auth-related variables from the earlier networked draft (`SURREALDB_URL`, `SURREALDB_USERNAME`, `SURREALDB_PASSWORD`) are **not used**.

| Variable | Default | Description |
|----------|---------|-------------|
| `SURREALDB_PATH` | `./lightrag_data/graphrag.db` | Path to the embedded SurrealKV database file (opened as `surrealkv://<path>`) |
| `SURREALDB_NAMESPACE` | `lightrag` | SurrealDB namespace |
| `SURREALDB_DATABASE` | `assistant` | SurrealDB database name |
| `SURREALDB_VECTOR_DIM` | `384` | Embedding dimension — must match model |
| `SURREALDB_HNSW_EF` | `64` | HNSW `ef_construction` (EFC) parameter |
| `SURREALDB_HNSW_M` | `16` | HNSW M parameter (connections per node) |

### 1.3 Full Implementation

> ⚠️ **The listing below is the ORIGINAL networked design and does not match the shipped code.** The authoritative implementation is [`surrealdb_impl.py`](surrealdb_impl.py) at the repo root. Read the code as the source of truth; the block below is retained only to show the original design intent. Key differences in the shipped adapter:
>
> - **Embedded, not networked:** `AsyncSurreal("surrealkv://<path>")` + `use(ns, db)` — **no** `Surreal(url)`, `connect()` over WebSocket, or `signin()`.
> - **Connection singleton:** a module-level `_CONNECTION` + `async get_connection()` / `close_connection()`, not a `config`-dict-carried `_get_connection(config)`. The storage dataclasses have **no** `config` field.
> - **Query serialization:** all queries run under an `asyncio.Lock` (`_query_lock`) to avoid embedded write-write conflicts; the SDK returns results already unwrapped, and a plain-string result is treated as an error and raised.
> - **RecordID handling:** IDs come back as `RecordID` objects; rows are normalized with `record::id(id)` and `_normalise_row()`.
> - **Signatures:** `filter_keys(set[str]) -> set[str]` (not `dict`); `drop() -> dict` status (not `None`); `SurrealDBVectorStorage.query(query: str, top_k, ...)` takes **raw text** and computes the embedding internally, adds a `cosine_better_than_threshold` filter; `upsert()` is itself batch-capable (no separate `upsert_many()`).
> - **Tables/fields:** `SCHEMALESS`/`FLEXIBLE`; prefixes `kv_`/`vec_`/`ent_`/`rel_`; entity fields `entity_name`/`entity_type`; `keywords` is `string`. Doc-status adds `content_hash`, `file_path`, `track_id`, `metadata`, `error_msg`, and seven query helpers (`get_docs_by_statuses`, `get_docs_by_track_id`, `get_docs_paginated`, `get_doc_by_file_path`/`_basename`/`_content_hash`, `get_all_status_counts`). Graph adds `get_all_labels`, `get_popular_labels`, `search_labels`, `remove_nodes`, `remove_edges`, and a BFS `get_knowledge_graph()`.

```python
"""
lightrag/kg/surrealdb_impl.py

SurrealDB storage adapter for LightRAG.
Implements BaseKVStorage, BaseVectorStorage, BaseGraphStorage,
and BaseDocStatusStorage using the SurrealDB Python SDK v2.x.

Register in lightrag/kg/__init__.py:
    STORAGE_IMPLEMENTATIONS["SurrealDBKVStorage"]        = "lightrag.kg.surrealdb_impl.SurrealDBKVStorage"
    STORAGE_IMPLEMENTATIONS["SurrealDBVectorStorage"]    = "lightrag.kg.surrealdb_impl.SurrealDBVectorStorage"
    STORAGE_IMPLEMENTATIONS["SurrealDBGraphStorage"]     = "lightrag.kg.surrealdb_impl.SurrealDBGraphStorage"
    STORAGE_IMPLEMENTATIONS["SurrealDBDocStatusStorage"] = "lightrag.kg.surrealdb_impl.SurrealDBDocStatusStorage"
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

from surrealdb import Surreal

from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    BaseDocStatusStorage,
    DocStatus,
)
from lightrag.utils import logger


# ---------------------------------------------------------------------------
# Shared connection pool
# ---------------------------------------------------------------------------

class SurrealDBDB:
    """
    Manages a single async SurrealDB connection per LightRAG workspace.
    All four storage classes share one instance of this via the storage
    config dict (keyed as '_connection'), following the postgres_impl.py
    singleton pattern.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.url       = config.get("url",       os.getenv("SURREALDB_URL",       "ws://localhost:8000/rpc"))
        self.namespace = config.get("namespace", os.getenv("SURREALDB_NAMESPACE", "lightrag"))
        self.database  = config.get("database",  os.getenv("SURREALDB_DATABASE",  "assistant"))
        self.username  = config.get("username",  os.getenv("SURREALDB_USERNAME",  "root"))
        self.password  = config.get("password",  os.getenv("SURREALDB_PASSWORD",  "root"))
        self._client: Surreal | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        async with self._lock:
            if self._client is not None:
                return
            client = Surreal(self.url)
            await client.connect()
            await client.signin({"user": self.username, "pass": self.password})
            await client.use(self.namespace, self.database)
            self._client = client
            logger.info(f"SurrealDB connected: {self.url} / {self.namespace}.{self.database}")

    async def query(self, sql: str, vars: dict[str, Any] | None = None) -> list[Any]:
        if self._client is None:
            raise RuntimeError("SurrealDBDB.connect() must be called before query()")
        result = await self._client.query(sql, vars or {})
        # SDK returns list of result objects; unwrap the first result's data
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict) and "result" in first:
                return first["result"] or []
            return first if isinstance(first, list) else []
        return []

    async def close(self) -> None:
        async with self._lock:
            if self._client:
                await self._client.close()
                self._client = None


def _get_connection(config: dict[str, Any]) -> SurrealDBDB:
    """Return the shared SurrealDBDB instance, creating it on first call."""
    if "_connection" not in config:
        config["_connection"] = SurrealDBDB(config)
    return config["_connection"]


# ---------------------------------------------------------------------------
# KV Storage
# ---------------------------------------------------------------------------

@dataclass
class SurrealDBKVStorage(BaseKVStorage):
    """
    Key-value storage backed by a SurrealDB SCHEMAFULL table.
    Used by LightRAG for entity summaries, community reports,
    and LLM response caching.
    """

    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._db = _get_connection(self.config)
        # namespace comes from LightRAG workspace name
        self._table = f"kv_store_{self.namespace}"

    async def initialize(self) -> None:
        await self._db.connect()
        await self._db.query(f"""
            DEFINE TABLE IF NOT EXISTS {self._table} SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id    ON {self._table} TYPE string;
            DEFINE FIELD IF NOT EXISTS value ON {self._table} TYPE any;
            DEFINE INDEX IF NOT EXISTS idx_id ON {self._table} COLUMNS id UNIQUE;
        """)

    async def finalize(self) -> None:
        await self._db.close()

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        rows = await self._db.query(
            f"SELECT * FROM {self._table} WHERE id = $id LIMIT 1",
            {"id": id},
        )
        return rows[0] if rows else None

    async def get_by_ids(self, ids: list[str], fields: list[str] | None = None) -> list[dict[str, Any] | None]:
        if not ids:
            return []
        field_clause = ", ".join(fields) if fields else "*"
        rows = await self._db.query(
            f"SELECT {field_clause} FROM {self._table} WHERE id IN $ids",
            {"ids": ids},
        )
        # Build id→row map; preserve ordering and fill None for misses
        row_map = {r["id"]: r for r in rows}
        return [row_map.get(id_) for id_ in ids]

    async def filter_keys(self, data: dict[str, Any]) -> set[str]:
        """Return keys from `data` that do NOT already exist in the table."""
        if not data:
            return set()
        existing = await self._db.query(
            f"SELECT id FROM {self._table} WHERE id IN $ids",
            {"ids": list(data.keys())},
        )
        existing_ids = {r["id"] for r in existing}
        return set(data.keys()) - existing_ids

    async def upsert(self, data: dict[str, Any]) -> None:
        """Upsert a single record. `data` must contain an 'id' key."""
        if not data:
            return
        await self._db.query(
            f"UPSERT type::thing($table, $id) CONTENT $data",
            {"table": self._table, "id": data["id"], "data": data},
        )

    async def upsert_many(self, items: list[dict[str, Any]]) -> None:
        """Batch upsert. Each item must contain an 'id' key."""
        for item in items:
            await self.upsert(item)

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        await self._db.query(
            f"DELETE {self._table} WHERE id IN $ids",
            {"ids": ids},
        )

    async def drop(self) -> None:
        await self._db.query(f"REMOVE TABLE {self._table}")

    async def index_done_callback(self) -> None:
        pass  # No-op; SurrealDB writes are immediate


# ---------------------------------------------------------------------------
# Vector Storage
# ---------------------------------------------------------------------------

@dataclass
class SurrealDBVectorStorage(BaseVectorStorage):
    """
    Vector storage backed by SurrealDB's native HNSW index.
    Stores chunk text + embedding + metadata and supports
    approximate nearest-neighbour (ANN) similarity search.
    """

    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._db = _get_connection(self.config)
        self._table = f"vector_store_{self.namespace}"
        self._dim = int(os.getenv("SURREALDB_VECTOR_DIM", "384"))
        self._ef  = int(os.getenv("SURREALDB_HNSW_EF",   "64"))
        self._m   = int(os.getenv("SURREALDB_HNSW_M",    "16"))

    async def initialize(self) -> None:
        await self._db.connect()
        # Use IF NOT EXISTS so re-runs are idempotent
        await self._db.query(f"""
            DEFINE TABLE IF NOT EXISTS {self._table} SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id        ON {self._table} TYPE string;
            DEFINE FIELD IF NOT EXISTS content   ON {self._table} TYPE string;
            DEFINE FIELD IF NOT EXISTS embedding ON {self._table} TYPE array<float>;
            DEFINE FIELD IF NOT EXISTS metadata  ON {self._table} TYPE object;
            DEFINE INDEX IF NOT EXISTS hnsw_idx  ON {self._table}
                FIELDS embedding
                HNSW DIMENSION {self._dim} DIST COSINE
                EFC {self._ef} M {self._m};
        """)

    async def finalize(self) -> None:
        await self._db.close()

    async def upsert(self, data: dict[str, Any]) -> None:
        """
        Upsert a single vector record.
        Expected keys: id (str), content (str), embedding (list[float]), metadata (dict).
        """
        await self._db.query(
            f"UPSERT type::thing($table, $id) CONTENT $data",
            {"table": self._table, "id": data["id"], "data": data},
        )

    async def upsert_many(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            await self.upsert(item)

    async def query(
        self,
        query_embedding: list[float],
        top_k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        ANN similarity search using SurrealDB's HNSW <|k,ef|> operator.
        Returns up to top_k records sorted by descending cosine similarity.
        filter is an optional dict of metadata key/value pairs to pre-filter.
        """
        where_clauses = [f"embedding <|{top_k},{self._ef}|> $vec"]
        if filter:
            for k, v in filter.items():
                where_clauses.append(f"metadata.{k} = ${k}")

        where = " AND ".join(where_clauses)
        bind: dict[str, Any] = {"vec": query_embedding}
        if filter:
            bind.update(filter)

        rows = await self._db.query(
            f"SELECT id, content, metadata, "
            f"vector::similarity::cosine(embedding, $vec) AS score "
            f"FROM {self._table} WHERE {where} "
            f"ORDER BY score DESC LIMIT {top_k}",
            bind,
        )
        return rows

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        await self._db.query(
            f"DELETE {self._table} WHERE id IN $ids",
            {"ids": ids},
        )

    async def drop(self) -> None:
        await self._db.query(f"REMOVE TABLE {self._table}")

    async def index_done_callback(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Graph Storage
# ---------------------------------------------------------------------------

@dataclass
class SurrealDBGraphStorage(BaseGraphStorage):
    """
    Knowledge graph storage using two flat SurrealDB tables:
    one for entity nodes and one for relation edges.

    LightRAG addresses edges by string ID (src___tgt), not by record
    links, so a flat relation table mirrors the postgres_impl.py / AGE
    approach rather than using SurrealDB's native RELATE edges.
    """

    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._db = _get_connection(self.config)
        self._entity_table   = f"entity_{self.namespace}"
        self._relation_table = f"relation_{self.namespace}"

    async def initialize(self) -> None:
        await self._db.connect()
        await self._db.query(f"""
            DEFINE TABLE IF NOT EXISTS {self._entity_table} SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id          ON {self._entity_table} TYPE string;
            DEFINE FIELD IF NOT EXISTS name        ON {self._entity_table} TYPE string;
            DEFINE FIELD IF NOT EXISTS type        ON {self._entity_table} TYPE string;
            DEFINE FIELD IF NOT EXISTS description ON {self._entity_table} TYPE string;
            DEFINE FIELD IF NOT EXISTS source_id   ON {self._entity_table} TYPE string;
            DEFINE FIELD IF NOT EXISTS extra       ON {self._entity_table} TYPE object;
            DEFINE INDEX IF NOT EXISTS idx_entity_name ON {self._entity_table} COLUMNS name UNIQUE;

            DEFINE TABLE IF NOT EXISTS {self._relation_table} SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id          ON {self._relation_table} TYPE string;
            DEFINE FIELD IF NOT EXISTS src_id      ON {self._relation_table} TYPE string;
            DEFINE FIELD IF NOT EXISTS tgt_id      ON {self._relation_table} TYPE string;
            DEFINE FIELD IF NOT EXISTS weight      ON {self._relation_table} TYPE float;
            DEFINE FIELD IF NOT EXISTS description ON {self._relation_table} TYPE string;
            DEFINE FIELD IF NOT EXISTS keywords    ON {self._relation_table} TYPE array<string>;
            DEFINE FIELD IF NOT EXISTS source_id   ON {self._relation_table} TYPE string;
            DEFINE INDEX IF NOT EXISTS idx_relation_id ON {self._relation_table} COLUMNS id UNIQUE;
            DEFINE INDEX IF NOT EXISTS idx_relation_src ON {self._relation_table} COLUMNS src_id;
            DEFINE INDEX IF NOT EXISTS idx_relation_tgt ON {self._relation_table} COLUMNS tgt_id;
        """)

    async def finalize(self) -> None:
        await self._db.close()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    async def has_node(self, node_id: str) -> bool:
        rows = await self._db.query(
            f"SELECT id FROM {self._entity_table} WHERE id = $id LIMIT 1",
            {"id": node_id},
        )
        return bool(rows)

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        rows = await self._db.query(
            f"SELECT * FROM {self._entity_table} WHERE id = $id LIMIT 1",
            {"id": node_id},
        )
        return rows[0] if rows else None

    async def upsert_node(self, node_id: str, node_data: dict[str, Any]) -> None:
        payload = {"id": node_id, **node_data}
        await self._db.query(
            f"UPSERT type::thing($table, $id) CONTENT $data",
            {"table": self._entity_table, "id": node_id, "data": payload},
        )

    async def delete_node(self, node_id: str) -> None:
        await self._db.query(
            f"DELETE {self._entity_table} WHERE id = $id",
            {"id": node_id},
        )
        # Also remove any edges referencing this node
        await self._db.query(
            f"DELETE {self._relation_table} WHERE src_id = $id OR tgt_id = $id",
            {"id": node_id},
        )

    async def node_degree(self, node_id: str) -> int:
        rows = await self._db.query(
            f"SELECT count() AS cnt FROM {self._relation_table} "
            f"WHERE src_id = $id OR tgt_id = $id GROUP ALL",
            {"id": node_id},
        )
        return rows[0]["cnt"] if rows else 0

    async def get_all_nodes(self) -> list[dict[str, Any]]:
        return await self._db.query(f"SELECT * FROM {self._entity_table}")

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    async def has_edge(self, src_id: str, tgt_id: str) -> bool:
        edge_id = f"{src_id}___{tgt_id}"
        rows = await self._db.query(
            f"SELECT id FROM {self._relation_table} WHERE id = $id LIMIT 1",
            {"id": edge_id},
        )
        return bool(rows)

    async def get_edge(self, src_id: str, tgt_id: str) -> dict[str, Any] | None:
        edge_id = f"{src_id}___{tgt_id}"
        rows = await self._db.query(
            f"SELECT * FROM {self._relation_table} WHERE id = $id LIMIT 1",
            {"id": edge_id},
        )
        return rows[0] if rows else None

    async def upsert_edge(self, src_id: str, tgt_id: str, edge_data: dict[str, Any]) -> None:
        edge_id = f"{src_id}___{tgt_id}"
        payload = {"id": edge_id, "src_id": src_id, "tgt_id": tgt_id, **edge_data}
        await self._db.query(
            f"UPSERT type::thing($table, $id) CONTENT $data",
            {"table": self._relation_table, "id": edge_id, "data": payload},
        )

    async def delete_edge(self, src_id: str, tgt_id: str) -> None:
        edge_id = f"{src_id}___{tgt_id}"
        await self._db.query(
            f"DELETE {self._relation_table} WHERE id = $id",
            {"id": edge_id},
        )

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Sum of both node degrees — used by LightRAG for edge weight scaling."""
        return await self.node_degree(src_id) + await self.node_degree(tgt_id)

    async def get_node_edges(self, node_id: str) -> list[tuple[str, str]]:
        """Return all (src_id, tgt_id) pairs where this node is src or tgt."""
        rows = await self._db.query(
            f"SELECT src_id, tgt_id FROM {self._relation_table} "
            f"WHERE src_id = $id OR tgt_id = $id",
            {"id": node_id},
        )
        return [(r["src_id"], r["tgt_id"]) for r in rows]

    async def get_all_edges(self) -> list[dict[str, Any]]:
        return await self._db.query(f"SELECT * FROM {self._relation_table}")

    async def drop(self) -> None:
        await self._db.query(f"REMOVE TABLE {self._entity_table}")
        await self._db.query(f"REMOVE TABLE {self._relation_table}")

    async def index_done_callback(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Doc Status Storage
# ---------------------------------------------------------------------------

@dataclass
class SurrealDBDocStatusStorage(BaseDocStatusStorage):
    """
    Tracks per-document ingestion state.
    Status values mirror LightRAG's DocStatus enum:
    pending | processing | done | failed
    """

    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._db = _get_connection(self.config)
        self._table = f"doc_status_{self.namespace}"

    async def initialize(self) -> None:
        await self._db.connect()
        await self._db.query(f"""
            DEFINE TABLE IF NOT EXISTS {self._table} SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id         ON {self._table} TYPE string;
            DEFINE FIELD IF NOT EXISTS status     ON {self._table} TYPE string;
            DEFINE FIELD IF NOT EXISTS content_summary ON {self._table} TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS content_length  ON {self._table} TYPE option<int>;
            DEFINE FIELD IF NOT EXISTS chunks_count    ON {self._table} TYPE option<int>;
            DEFINE FIELD IF NOT EXISTS created_at ON {self._table} TYPE datetime DEFAULT time::now();
            DEFINE FIELD IF NOT EXISTS updated_at ON {self._table} TYPE datetime DEFAULT time::now();
            DEFINE INDEX IF NOT EXISTS idx_doc_id     ON {self._table} COLUMNS id UNIQUE;
            DEFINE INDEX IF NOT EXISTS idx_doc_status ON {self._table} COLUMNS status;
        """)

    async def finalize(self) -> None:
        await self._db.close()

    async def get_status(self, doc_id: str) -> DocStatus | None:
        rows = await self._db.query(
            f"SELECT status FROM {self._table} WHERE id = $id LIMIT 1",
            {"id": doc_id},
        )
        if not rows:
            return None
        return DocStatus(rows[0]["status"])

    async def set_status(self, doc_id: str, status: DocStatus, **kwargs: Any) -> None:
        payload: dict[str, Any] = {
            "id": doc_id,
            "status": status.value,
            "updated_at": "time::now()",
            **kwargs,
        }
        await self._db.query(
            f"UPSERT type::thing($table, $id) MERGE $data",
            {"table": self._table, "id": doc_id, "data": payload},
        )

    async def get_docs_by_status(self, status: DocStatus) -> list[dict[str, Any]]:
        return await self._db.query(
            f"SELECT * FROM {self._table} WHERE status = $status",
            {"status": status.value},
        )

    async def get_status_counts(self) -> dict[str, int]:
        rows = await self._db.query(
            f"SELECT status, count() AS cnt FROM {self._table} GROUP BY status"
        )
        return {r["status"]: r["cnt"] for r in rows}

    async def filter_keys(self, doc_ids: list[str]) -> set[str]:
        """Return doc_ids whose status is NOT 'done' (i.e. need processing)."""
        if not doc_ids:
            return set()
        rows = await self._db.query(
            f"SELECT id FROM {self._table} WHERE id IN $ids AND status = 'done'",
            {"ids": doc_ids},
        )
        done_ids = {r["id"] for r in rows}
        return set(doc_ids) - done_ids

    async def drop(self) -> None:
        await self._db.query(f"REMOVE TABLE {self._table}")

    async def index_done_callback(self) -> None:
        pass
```

---

## 2. Ingestion Pipeline — `ingest.py`

> 📌 **Operational safeguards (crash resilience, preflight, idempotency, logging) are documented separately in [`Ingestion_Safeguards.md`](Ingestion_Safeguards.md), which is authoritative for the current LlamaIndex/Neo4j pipeline.** The listing in this section is SurrealDB-era and retained for design intent only.

Orchestrates file discovery, parallel extraction, deduplication, and LightRAG insertion. Run with:

```bash
python ingest.py --config config.yaml
python ingest.py --reset   # delete the embedded SurrealKV DB, then rebuild from scratch
```

Key design decisions baked in:
- `multiprocessing.Pool(processes=extraction_workers)` for CPU-bound extraction — worker count is **capped** (default 8) to avoid OOM, not `cpu_count()`
- `asyncio.Semaphore` caps concurrent LightRAG inserts to avoid overwhelming Ollama
- LightRAG's `DocStatusStorage` provides implicit checkpointing — re-running the pipeline skips documents already marked processed
- Content type is inferred from directory name fragments (configurable via `content_type_rules`)
- `--reset` deletes the embedded database file (via `reset_database()`); the storage classes recreate their schema on the next init. For an embedded file store this is the reliable equivalent of "drop and rebuild all tables."

> ⚠️ **The embedded `ingest.py` listing below is stale — read [`ingest.py`](ingest.py) as the source of truth.** The shipped pipeline differs: it imports `patch_lightrag`; uses a capped `extraction_workers` pool with `pool.imap_unordered` and Rich progress bars; calls `initialize_pipeline_status()` in `init_lightrag()` (fixes a "pipeline already busy" wedge); wraps the run in `try/finally` with `rag.finalize_storages()` to flush SurrealKV; inserts via `rag.ainsert(chunk.text, file_paths=chunk.source_path)` (**not** `metadata=`/`doc_id=`, so per-chunk `content_type`/`has_code` are not forwarded); `make_ollama_func(host)` drops the `model` parameter (the model name is supplied to LightRAG via `llm_model_name=`, and `ollama_model_complete()` derives it from LightRAG's injected global config); and `--reset` now actually clears data (see above).

```python
"""
ingest.py

GraphRAG Assistant — Document Ingestion Pipeline

Orchestrates:
  1. File discovery
  2. PDF/EPUB extraction (multiprocessing, CPU-bound)
  3. Deduplication (MinHash LSH)
  4. LightRAG insertion — embedding + entity extraction + SurrealDB persistence (asyncio)

Usage:
    python ingest.py [--config config.yaml] [--reset]

Environment variables (can also live in a .env file):
    SURREALDB_URL, SURREALDB_NAMESPACE, SURREALDB_DATABASE,
    SURREALDB_USERNAME, SURREALDB_PASSWORD, SURREALDB_VECTOR_DIM,
    LIGHTRAG_WORKING_DIR, OLLAMA_HOST, MAX_CONCURRENT_INSERTS
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import yaml
from bs4 import BeautifulSoup
from datasketch import MinHash, MinHashLSH
from dotenv import load_dotenv

# docling — pip install docling
from docling.document_converter import DocumentConverter

# ebooklib — pip install ebooklib
import ebooklib
from ebooklib import epub

# semantic-text-splitter — pip install semantic-text-splitter
from semantic_text_splitter import TextSplitter

# sentence-transformers — pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# LightRAG — pip install lightrag-hku
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single indexable text unit extracted from a source document."""
    text: str
    source_path: str
    content_type: str          # 'software' | 'math' | 'selfhelp'
    metadata: dict[str, Any] = field(default_factory=dict)
    # populated during chunk_software() — associated code blocks
    code_blocks: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    library_path: Path
    working_dir: Path
    max_concurrent_inserts: int = 4
    dedup_threshold: float = 0.85
    dedup_num_perm: int = 128
    dedup_shingle_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    vector_dim: int = 384
    chunk_max_chars: int = 1500   # for semantic splitter
    # Map directory name fragments → content type
    content_type_rules: dict[str, str] = field(default_factory=lambda: {
        "software": "software",
        "dev":      "software",
        "code":     "software",
        "math":     "math",
        "maths":    "math",
        "self":     "selfhelp",
        "help":     "selfhelp",
        "personal": "selfhelp",
    })


def load_config(path: str | None) -> PipelineConfig:
    defaults: dict[str, Any] = {
        "library_path": os.getenv("LIBRARY_PATH", "./library"),
        "working_dir":  os.getenv("LIGHTRAG_WORKING_DIR", "./lightrag_data"),
        "max_concurrent_inserts": int(os.getenv("MAX_CONCURRENT_INSERTS", "4")),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "ollama_host":  os.getenv("OLLAMA_HOST",  "http://localhost:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "qwen2.5:14b"),
        "vector_dim":   int(os.getenv("SURREALDB_VECTOR_DIM", "384")),
    }
    if path and Path(path).exists():
        with open(path) as f:
            file_cfg = yaml.safe_load(f) or {}
        defaults.update(file_cfg)
    return PipelineConfig(
        library_path=Path(defaults["library_path"]),
        working_dir=Path(defaults["working_dir"]),
        max_concurrent_inserts=defaults["max_concurrent_inserts"],
        embedding_model=defaults["embedding_model"],
        ollama_host=defaults["ollama_host"],
        ollama_model=defaults["ollama_model"],
        vector_dim=defaults["vector_dim"],
    )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_files(library_path: Path) -> list[Path]:
    """Recursively find all PDF and EPUB files under library_path."""
    files = []
    for ext in ("*.pdf", "*.epub"):
        files.extend(library_path.rglob(ext))
    logger.info(f"Discovered {len(files)} documents in {library_path}")
    return sorted(files)


# ---------------------------------------------------------------------------
# Content-type classification
# ---------------------------------------------------------------------------

_CONTENT_TYPE_RULES: dict[str, str] = {
    "software": "software",
    "dev":      "software",
    "code":     "software",
    "math":     "math",
    "maths":    "math",
    "self":     "selfhelp",
    "help":     "selfhelp",
    "personal": "selfhelp",
}

def classify(path: Path) -> str:
    """
    Infer content type from directory name fragments.
    Falls back to 'selfhelp' (full prose, no special handling).
    """
    parts = [p.lower() for p in path.parts]
    for part in parts:
        for keyword, ctype in _CONTENT_TYPE_RULES.items():
            if keyword in part:
                return ctype
    return "selfhelp"


# ---------------------------------------------------------------------------
# Extraction — PDF
# ---------------------------------------------------------------------------

def extract_pdf(path: Path) -> str:
    """
    Extract clean prose from a PDF using docling.
    Returns the full document text as a single string.
    """
    converter = DocumentConverter()
    result = converter.convert(str(path))
    return result.document.export_to_markdown()


# ---------------------------------------------------------------------------
# Extraction — EPUB
# ---------------------------------------------------------------------------

def extract_epub(path: Path) -> str:
    """
    Extract chapter-level text from an EPUB using ebooklib + BeautifulSoup.
    Chapters are joined with double newlines to preserve structure boundaries.
    """
    book = epub.read_epub(str(path))
    chapters: list[str] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        if text.strip():
            chapters.append(text)
    return "\n\n".join(chapters)


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

# Regex patterns for code block detection in software books
_CODE_FENCE_RE   = re.compile(r"```[\w]*\n([\s\S]*?)```", re.MULTILINE)
_INDENT_CODE_RE  = re.compile(r"(?m)^(?: {4}|\t)(.+)")
_FORMULA_RE      = re.compile(
    r"(\$\$[\s\S]+?\$\$"       # display LaTeX $$...$$
    r"|\$[^\$\n]+?\$"          # inline LaTeX $...$
    r"|\\begin\{[^}]+\}[\s\S]+?\\end\{[^}]+\})"  # \begin{} ... \end{}
)

_splitter: TextSplitter | None = None

def _get_splitter(max_chars: int = 1500) -> TextSplitter:
    global _splitter
    if _splitter is None:
        _splitter = TextSplitter(max_chars)
    return _splitter


def _semantic_split(text: str, max_chars: int = 1500) -> list[str]:
    """Split text into semantically coherent chunks via semantic-text-splitter."""
    splitter = _get_splitter(max_chars)
    return splitter.chunks(text)


def chunk_software(text: str, source_path: str, max_chars: int = 1500) -> list[Chunk]:
    """
    For software/dev books:
      1. Detect and extract fenced and indented code blocks.
      2. Replace each with a [CODE_N] placeholder in the prose stream.
      3. Semantically chunk the cleaned prose.
      4. Re-attach code blocks as metadata on the nearest chunk.
    """
    code_registry: dict[str, str] = {}

    def _replace_fence(m: re.Match) -> str:
        cid = f"CODE_{len(code_registry)}"
        code_registry[cid] = m.group(1).strip()
        return f"\n[{cid}]\n"

    def _replace_indent(m: re.Match) -> str:
        cid = f"CODE_{len(code_registry)}"
        code_registry[cid] = m.group(1).rstrip()
        return f"\n[{cid}]\n"

    prose = _CODE_FENCE_RE.sub(_replace_fence, text)
    prose = _INDENT_CODE_RE.sub(_replace_indent, prose)

    raw_chunks = _semantic_split(prose, max_chars)
    result: list[Chunk] = []
    for raw in raw_chunks:
        if not raw.strip():
            continue
        refs = re.findall(r"\[CODE_\d+\]", raw)
        blocks = [code_registry[r[1:-1]] for r in refs if r[1:-1] in code_registry]
        # Strip placeholder tokens from prose text
        clean = re.sub(r"\[CODE_\d+\]", "", raw).strip()
        if not clean:
            continue
        result.append(Chunk(
            text=clean,
            source_path=source_path,
            content_type="software",
            code_blocks=blocks,
            metadata={"has_code": bool(blocks)},
        ))
    return result


def chunk_math(text: str, source_path: str, max_chars: int = 1500) -> list[Chunk]:
    """
    For math books:
      Strip LaTeX formula regions (they don't convert cleanly to prose),
      then chunk the remaining prose.
    """
    clean = _FORMULA_RE.sub(" ", text)
    # Also drop lines that are mostly symbols / very short after formula removal
    lines = [l for l in clean.splitlines() if len(l.strip()) > 20]
    prose = "\n".join(lines)
    raw_chunks = _semantic_split(prose, max_chars)
    return [
        Chunk(text=c.strip(), source_path=source_path, content_type="math")
        for c in raw_chunks if c.strip()
    ]


def chunk_prose(text: str, source_path: str, content_type: str, max_chars: int = 1500) -> list[Chunk]:
    """
    Full semantic chunking with no special handling.
    Used for self-help / general prose.
    """
    raw_chunks = _semantic_split(text, max_chars)
    return [
        Chunk(text=c.strip(), source_path=source_path, content_type=content_type)
        for c in raw_chunks if c.strip()
    ]


def apply_content_rules(raw: str, path: Path) -> list[Chunk]:
    """Dispatch to the correct chunker based on classified content type."""
    source = str(path)
    ctype = classify(path)
    if ctype == "software":
        return chunk_software(raw, source)
    elif ctype == "math":
        return chunk_math(raw, source)
    else:
        return chunk_prose(raw, source, ctype)


# ---------------------------------------------------------------------------
# Top-level extraction (called inside multiprocessing.Pool.map)
# ---------------------------------------------------------------------------

def extract_document(path: Path) -> list[Chunk]:
    """
    Entry point for multiprocessing workers.
    Returns an empty list on any extraction failure (logged, not raised).
    """
    try:
        logger.info(f"Extracting: {path.name}")
        if path.suffix.lower() == ".pdf":
            raw = extract_pdf(path)
        elif path.suffix.lower() == ".epub":
            raw = extract_epub(path)
        else:
            return []
        return apply_content_rules(raw, path)
    except Exception as exc:
        logger.error(f"Extraction failed for {path}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def get_shingles(text: str, k: int = 5) -> list[str]:
    """Character k-gram shingles for MinHash."""
    text = text.lower()
    return [text[i:i + k] for i in range(len(text) - k + 1)]


def deduplicate(chunks: list[Chunk], threshold: float = 0.85, num_perm: int = 128, k: int = 5) -> list[Chunk]:
    """
    Remove near-duplicate chunks using MinHash LSH.
    Jaccard similarity >= threshold → keep first occurrence, discard later ones.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        m = MinHash(num_perm=num_perm)
        for shingle in get_shingles(chunk.text, k=k):
            m.update(shingle.encode("utf-8"))
        key = f"chunk_{i}"
        if not lsh.query(m):
            lsh.insert(key, m)
            kept.append(chunk)
    removed = len(chunks) - len(kept)
    logger.info(f"Deduplication: {len(chunks)} → {len(kept)} chunks ({removed} removed)")
    return kept


# ---------------------------------------------------------------------------
# LightRAG initialisation
# ---------------------------------------------------------------------------

def make_embedding_func(model_name: str, vector_dim: int) -> EmbeddingFunc:
    """
    Build a LightRAG EmbeddingFunc using a local sentence-transformers model.
    The model is loaded once and reused (CUDA if available).
    """
    _model = SentenceTransformer(model_name)

    async def _embed(texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        # Run GPU inference in a thread pool to avoid blocking the event loop
        embeddings = await loop.run_in_executor(
            None,
            lambda: _model.encode(texts, convert_to_numpy=True).tolist()
        )
        return embeddings

    return EmbeddingFunc(embedding_dim=vector_dim, max_token_size=512, func=_embed)


def make_ollama_func(host: str, model: str):  # NOTE: shipped code takes (host) only
    """Build an Ollama LLM func compatible with LightRAG's llm_model_func interface."""
    async def _llm(prompt: str, **kwargs) -> str:
        return await ollama_model_complete(
            prompt,
            host=host,
            model=model,
            **kwargs,
        )
    return _llm


async def init_lightrag(config: PipelineConfig) -> LightRAG:
    """Initialise LightRAG with the SurrealDB backend and local models."""
    config.working_dir.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(config.working_dir),
        llm_model_func=make_ollama_func(config.ollama_host, config.ollama_model),
        embedding_func=make_embedding_func(config.embedding_model, config.vector_dim),
        kv_storage="SurrealDBKVStorage",
        vector_storage="SurrealDBVectorStorage",
        graph_storage="SurrealDBGraphStorage",
        doc_status_storage="SurrealDBDocStatusStorage",
    )
    await rag.initialize_storages()  # NOTE: shipped code also calls initialize_pipeline_status()
    logger.info("LightRAG initialised with SurrealDB backend")
    return rag


# ---------------------------------------------------------------------------
# Insertion with checkpointing (via DocStatusStorage)
# ---------------------------------------------------------------------------

async def insert_with_semaphore(
    rag: LightRAG,
    chunk: Chunk,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    Insert a single chunk into LightRAG under the semaphore.
    LightRAG's DocStatusStorage provides implicit checkpointing —
    documents already marked 'done' are skipped on re-runs.
    """
    async with semaphore:
        try:
            # Pass source path as doc ID so status is tracked per document
            await rag.ainsert(
                chunk.text,
                metadata={
                    "source": chunk.source_path,
                    "content_type": chunk.content_type,
                    "has_code": bool(chunk.code_blocks),
                    **chunk.metadata,
                },
                doc_id=chunk.source_path,
            )
        except Exception as exc:
            logger.error(f"Insert failed for chunk from {chunk.source_path}: {exc}")


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

async def run_pipeline(config: PipelineConfig, reset: bool = False) -> None:
    # Stage 0: (Optional) reset existing data
    if reset:
        logger.warning("--reset flag set: existing SurrealDB data will be cleared on init")

    # Stage 1: Initialise LightRAG + SurrealDB
    rag = await init_lightrag(config)

    # Stage 2: File discovery
    files = discover_files(config.library_path)
    if not files:
        logger.warning(f"No PDF/EPUB files found under {config.library_path}")
        return

    # Stage 3: Parallel extraction (CPU-bound → multiprocessing)
    logger.info(f"Extracting {len(files)} documents using {cpu_count()} workers...")
    with Pool(processes=cpu_count()) as pool:
        results: list[list[Chunk]] = pool.map(extract_document, files)

    all_chunks: list[Chunk] = [chunk for doc_chunks in results for chunk in doc_chunks]
    logger.info(f"Extraction complete: {len(all_chunks)} raw chunks")

    # Stage 4: Deduplication
    chunks = deduplicate(
        all_chunks,
        threshold=config.dedup_threshold,
        num_perm=config.dedup_num_perm,
        k=config.dedup_shingle_k,
    )

    # Stage 5: Async insertion into LightRAG / SurrealDB
    logger.info(f"Inserting {len(chunks)} chunks (concurrency={config.max_concurrent_inserts})...")
    semaphore = asyncio.Semaphore(config.max_concurrent_inserts)
    tasks = [insert_with_semaphore(rag, chunk, semaphore) for chunk in chunks]
    await asyncio.gather(*tasks)

    logger.info("Ingestion pipeline complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG Assistant ingestion pipeline")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild SurrealDB tables")
    args = parser.parse_args()

    config = load_config(args.config)
    asyncio.run(run_pipeline(config, reset=args.reset))


if __name__ == "__main__":
    main()
```

---

## 3. FastAPI Bridge — `api.py`

Exposes an Ollama-compatible REST API so OpenWebUI connects with zero configuration. LightRAG query modes are surfaced as selectable "models" in OpenWebUI's model picker. Run with:

```bash
uvicorn api:app --host 0.0.0.0 --port 11435
```

Set `QUERY_LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=...` to route query-time generation through Claude Haiku instead of local Ollama. Ingestion-phase LLM calls (entity extraction, keyword inference) always use Ollama regardless of this setting.

> **Note (FR-F-02):** the shipped `/api/chat` passes conversation history **into retrieval** via `QueryParam(mode=mode, conversation_history=history, history_turns=3)`, so follow-up turns condition graph/vector selection — not only the final generation prompt. The embedded listing below shows the earlier `aquery(query, param=QueryParam(mode=mode))` form and is otherwise faithful to `api.py`.

```python
"""
api.py

GraphRAG Assistant — FastAPI Bridge

Exposes an Ollama-compatible REST API so OpenWebUI can connect without
any special configuration. Query modes (local, global, hybrid, naive, mix)
are surfaced as selectable 'models' in OpenWebUI's model picker.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 11435 --reload

Environment variables (can also live in a .env file):
    SURREALDB_URL, SURREALDB_NAMESPACE, SURREALDB_DATABASE,
    SURREALDB_USERNAME, SURREALDB_PASSWORD, SURREALDB_VECTOR_DIM,
    LIGHTRAG_WORKING_DIR, OLLAMA_HOST, OLLAMA_MODEL,
    QUERY_LLM_PROVIDER   ('ollama' | 'anthropic')
    ANTHROPIC_API_KEY    (required if QUERY_LLM_PROVIDER=anthropic)
    ANTHROPIC_MODEL      (default: claude-haiku-4-5-20251001)
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import anthropic
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_model_complete

from ingest import make_embedding_func  # reuse embedding func from pipeline

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WORKING_DIR         = os.getenv("LIGHTRAG_WORKING_DIR",  "./lightrag_data")
OLLAMA_HOST         = os.getenv("OLLAMA_HOST",           "http://localhost:11434")
OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL",          "qwen2.5:14b")
VECTOR_DIM          = int(os.getenv("SURREALDB_VECTOR_DIM", "384"))
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL",       "all-MiniLM-L6-v2")
QUERY_LLM_PROVIDER  = os.getenv("QUERY_LLM_PROVIDER",   "ollama")  # 'ollama' | 'anthropic'
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY",     "")
ANTHROPIC_MODEL     = os.getenv("ANTHROPIC_MODEL",       "claude-haiku-4-5-20251001")

QUERY_MODES = ["mix", "hybrid", "local", "global", "naive"]


# ---------------------------------------------------------------------------
# Pydantic request/response models (Ollama-compatible schema)
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "mix"
    messages: list[Message]
    stream: bool = True


class GenerateRequest(BaseModel):
    model: str = "mix"
    prompt: str
    stream: bool = True


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

async def _query_ollama(prompt: str) -> AsyncGenerator[str, None]:
    """Stream a response from Ollama using the OpenAI-compatible endpoint."""
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break


async def _query_anthropic(prompt: str) -> AsyncGenerator[str, None]:
    """Stream a response from Claude via the Anthropic API."""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    async with client.messages.stream(
        model=ANTHROPIC_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def stream_llm_response(prompt: str) -> AsyncGenerator[str, None]:
    """Route to the configured query-time LLM provider."""
    if QUERY_LLM_PROVIDER == "anthropic":
        async for token in _query_anthropic(prompt):
            yield token
    else:
        async for token in _query_ollama(prompt):
            yield token


# ---------------------------------------------------------------------------
# LightRAG initialisation (shared app state)
# ---------------------------------------------------------------------------

_rag: LightRAG | None = None


async def get_rag() -> LightRAG:
    if _rag is None:
        raise HTTPException(status_code=503, detail="LightRAG not yet initialised")
    return _rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise LightRAG on startup; shut down on exit."""
    global _rag
    logger.info("Initialising LightRAG with SurrealDB backend...")

    os.makedirs(WORKING_DIR, exist_ok=True)

    _rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=_make_ollama_func(),
        embedding_func=make_embedding_func(EMBEDDING_MODEL, VECTOR_DIM),
        kv_storage="SurrealDBKVStorage",
        vector_storage="SurrealDBVectorStorage",
        graph_storage="SurrealDBGraphStorage",
        doc_status_storage="SurrealDBDocStatusStorage",
    )
    await _rag.initialize_storages()
    logger.info(f"LightRAG ready. Query LLM: {QUERY_LLM_PROVIDER.upper()}")

    yield  # app runs here

    logger.info("Shutting down LightRAG...")
    # Storage finalization is handled internally by LightRAG


def _make_ollama_func():
    """Ollama LLM func used for ingestion-phase roles (EXTRACT, KEYWORDS)."""
    async def _fn(prompt: str, **kwargs) -> str:
        return await ollama_model_complete(prompt, host=OLLAMA_HOST, model=OLLAMA_MODEL, **kwargs)
    return _fn


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GraphRAG Assistant API",
    description="Ollama-compatible bridge over LightRAG + SurrealDB",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_chunk(content: str, done: bool = False) -> bytes:
    """Format a streaming chunk in Ollama NDJSON format."""
    payload = {
        "model": "graphrag",
        "created_at": "",
        "message": {"role": "assistant", "content": content},
        "done": done,
    }
    return (json.dumps(payload) + "\n").encode()


def _build_query(messages: list[Message]) -> tuple[str, list[dict]]:
    """
    Extract the latest user query and format prior turns as conversation history.
    Returns (query_string, history_list).
    """
    if not messages:
        raise HTTPException(status_code=400, detail="messages list is empty")
    query = messages[-1].content
    history = [{"role": m.role, "content": m.content} for m in messages[:-1]]
    return query, history


def _validate_mode(mode: str) -> str:
    """Normalise and validate the query mode. Falls back to 'mix'."""
    mode = mode.strip().lower()
    return mode if mode in QUERY_MODES else "mix"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "provider": QUERY_LLM_PROVIDER}


@app.get("/api/tags")
async def tags() -> dict[str, list[dict[str, Any]]]:
    """
    Return available 'models' (= LightRAG query modes) to OpenWebUI.
    OpenWebUI's model picker populates from this endpoint.
    """
    now = int(time.time())
    return {
        "models": [
            {
                "name": mode,
                "modified_at": now,
                "size": 0,
                "digest": mode,
                "details": {
                    "format": "graphrag",
                    "family": "lightrag",
                    "parameter_size": mode,
                    "quantization_level": "none",
                },
            }
            for mode in QUERY_MODES
        ]
    }


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """
    Ollama-compatible /api/chat endpoint.
    OpenWebUI passes the selected 'model' name as the query mode.
    Conversation history is forwarded to LightRAG on each turn.
    """
    rag = await get_rag()
    query, history = _build_query(request.messages)
    mode = _validate_mode(request.model)

    logger.info(f"[chat] mode={mode} query={query[:80]!r}")

    async def _stream() -> AsyncGenerator[bytes, None]:
        # 1. Retrieve context from LightRAG (graph + vector retrieval)
        try:
            context: str = await rag.aquery(
                query,
                param=QueryParam(mode=mode),
            )
        except Exception as exc:
            logger.error(f"LightRAG retrieval error: {exc}")
            yield _ollama_chunk(f"[Retrieval error: {exc}]", done=True)
            return

        # 2. Build grounded prompt with conversation history
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:]
        )
        prompt = (
            f"You are a knowledgeable assistant. Answer based on the context below.\n\n"
            f"Context:\n{context}\n\n"
            + (f"Conversation history:\n{history_text}\n\n" if history_text else "")
            + f"User: {query}\nAssistant:"
        )

        # 3. Stream LLM response token by token
        try:
            async for token in stream_llm_response(prompt):
                yield _ollama_chunk(token)
            yield _ollama_chunk("", done=True)
        except Exception as exc:
            logger.error(f"LLM streaming error: {exc}")
            yield _ollama_chunk(f"[LLM error: {exc}]", done=True)

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


@app.post("/api/generate")
async def generate(request: GenerateRequest) -> StreamingResponse:
    """
    Ollama-compatible /api/generate endpoint (single-turn, no history).
    Useful for quick testing outside of OpenWebUI.
    """
    rag = await get_rag()
    mode = _validate_mode(request.model)
    query = request.prompt

    logger.info(f"[generate] mode={mode} query={query[:80]!r}")

    async def _stream() -> AsyncGenerator[bytes, None]:
        try:
            context: str = await rag.aquery(query, param=QueryParam(mode=mode))
        except Exception as exc:
            yield _ollama_chunk(f"[Retrieval error: {exc}]", done=True)
            return

        prompt = (
            f"Answer the following question using only the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
        try:
            async for token in stream_llm_response(prompt):
                yield _ollama_chunk(token)
            yield _ollama_chunk("", done=True)
        except Exception as exc:
            yield _ollama_chunk(f"[LLM error: {exc}]", done=True)

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


@app.get("/graph")
async def graph_data() -> dict[str, Any]:
    """
    Return graph visualization data compatible with LightRAG's built-in WebUI.
    Optional endpoint — used if you run LightRAG's frontend separately.
    """
    rag = await get_rag()
    try:
        nodes = await rag.graph_storage.get_all_nodes()
        edges = await rag.graph_storage.get_all_edges()
        return {
            "nodes": [{"id": n["id"], "label": n.get("name", n["id"]), **n} for n in nodes],
            "edges": [{"from": e["src_id"], "to": e["tgt_id"], **e} for e in edges],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
```

---

## 4. Testing Strategy

> **Status: DEFERRED — not yet implemented (as of 2026-07-03).** No test files, no `pytest`/`pytest-asyncio`/`pytest-cov` dependencies, and no test fixtures exist in the repo. The strategy below is retained as the target design to implement when testing is prioritized. Since the storage backend is the embedded SurrealKV engine (no server), tests should point `SURREALDB_PATH` at a temporary file per test session and mock Ollama — there is no database container to manage.

### 4.1 Unit Tests — Adapter

Each storage class is tested in isolation against an embedded SurrealKV database pointed at a temporary file (`SURREALDB_PATH`). Tests cover:

- Connection and schema initialization
- Round-trip upsert and retrieval for KV storage
- Vector upsert and top-k similarity search with known embeddings
- Graph node and edge upsert, neighbor retrieval, degree calculation
- Doc status transitions: `pending` → `processing` → `done`
- Namespace isolation: two workspaces do not share data

### 4.2 Integration Tests — Ingestion Pipeline

- End-to-end test using a small synthetic corpus (10 documents)
- Verify deduplication removes known near-duplicate pairs
- Verify code block tagging correctly separates prose and code
- Verify LightRAG entity extraction produces non-empty graph in SurrealDB

### 4.3 Integration Tests — Query Layer

- Submit known queries against the synthetic corpus
- Verify each query mode (`local`, `global`, `hybrid`, `mix`, `naive`) returns non-empty results
- Verify citations reference real document IDs present in SurrealDB
- Verify FastAPI `/api/chat` streams a valid NDJSON response

### 4.4 Test Infrastructure

| Component | Tool | Notes |
|-----------|------|-------|
| Unit tests | `pytest` + `pytest-asyncio` | Async test support required throughout |
| SurrealDB (test) | Embedded SurrealKV (`surrealkv://` temp file) | Isolated per test session via fixtures; no server/container |
| Ollama (test) | Mock via `httpx` MockTransport | Avoid real GPU calls in unit tests |
| Coverage | `pytest-cov` | Target: >80% on `surrealdb_impl.py` |

---

## 5. Known Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LightRAG storage interface changes in a minor version update | Medium | High | Pin LightRAG version in `requirements.txt`; review CHANGELOG before upgrading |
| SurrealDB HNSW query syntax differs from expected | Medium | High | **Realized & mitigated.** With the pinned surrealdb `1.0.8` SDK, the ANN pattern is `embedding <|k,ef|> $vec` plus `vector::similarity::cosine(...)`; encoded directly in `SurrealDBVectorStorage.query()` with inline comments |
| Qwen2.5:14B entity extraction quality insufficient for technical content | Low | Medium | Evaluate on 50-document sample before full ingestion; fallback to Claude Haiku for `EXTRACT` role |
| docling formula extraction from math PDFs produces garbled text | High | Low | Math books use prose-only extraction (known limitation); formulas are excluded by design |
| SurrealDB HNSW performance insufficient for corpus size | Low | Medium | Tune `EFC` and `M` parameters; corpus is personal-scale and unlikely to stress HNSW |
| LightRAG does not officially merge SurrealDB adapter | High | Low | **Realized & mitigated.** Adapter is maintained as a standalone `surrealdb_impl.py`; the idempotent `patch_lightrag.py` setup script copies it into the installed `lightrag/kg/` package and patches `kg/__init__.py`. Must be re-run after any `lightrag-hku` reinstall/upgrade |

---

*GraphRAG Personal AI Assistant · All Documents · v1.0 · June 2026*
