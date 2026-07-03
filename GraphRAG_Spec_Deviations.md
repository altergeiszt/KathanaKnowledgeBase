# GraphRAG Assistant: Codebase vs. Spec Deviations

Audit date: 2026-07-03. Compares `GraphRAG_Assistant_Docs.md` against the current implementation in `KathanaKnowledgeBase`. Read-only audit — no source files were modified.

## Central finding

The spec assumes a **networked SurrealDB** (`ws://localhost:8000/rpc`, `signin()`, separate Docker service). The code implements an **embedded SurrealKV file-based database** (`surrealkv://./lightrag_data/graphrag.db`, `AsyncSurreal`, no auth). This single divergence cascades into most of the table-naming, schema, and method-signature deviations below.

## 1. Section-by-section deviations

| Spec section (line) | Spec says | Code reality | Severity |
|---|---|---|---|
| SRS §2.5 (L82) | SurrealDB SDK v2.0+ | `requirements.txt` pins `surrealdb==1.0.8` | Meaningful |
| FR-I-07/08 (L105, 613-620) | Per-document checkpointing; 4 chunking strategies (Fixed/Recursive/Vector/Paragraph) | `ingest.py insert_with_semaphore()` (~470-488) hashes per-chunk; always uses `semantic_text_splitter.TextSplitter` | Deliberate evolution — doc stale |
| FR-Q-03 (L122) | Citation formatting | No dedicated code; relies on raw LightRAG context | Meaningful gap |
| FR-Q-06 / Arch §4.2 (L127, 405-412) | Independent EXTRACT/QUERY/KEYWORDS models | Only QUERY is independently configurable; EXTRACT+KEYWORDS share one Ollama model (`make_ollama_func()`, ~424-438) | Meaningful |
| FR-S-05/07 (L135, 138) | SDK version claim; class-naming rule | Inconsistent with the four separately named storage classes actually implemented | Minor |
| FR-F-02 (L144) | Conversation history informs retrieval | Folded into prompt text (`api.py` ~282-291) but never passed to `rag.aquery()` (~273-276) — doesn't reach retrieval | Minor |
| Arch §2.5 (~L1547) | Worker pool size = `cpu_count()` | Capped via new `extraction_workers` config (`ingest.py` L111) to avoid OOM | Deliberate improvement — doc stale |
| Arch §3.1-3.2 (L288-332) | Table prefixes `kv_store_`, `vector_store_`, `entity_`, `relation_`; `SCHEMAFULL`; fields `name`/`type`; `keywords: array<string>` | Prefixes are `kv_`, `vec_`, `ent_`, `rel_`; tables are `SCHEMALESS`/`FLEXIBLE`; fields are `entity_name`/`entity_type`; `keywords: string` | Meaningful — spec (~L190) says these names must stay stable for future GraphNotes integration |
| Arch §3.3 (L334-360) | Simple dict-literal storage registration | `patch_lightrag.py` is a full runtime string-patching script | Code evolved beyond spec |
| Arch §4.4 (L423-425) | Reranker (`BAAI/bge-reranker-v2-m3`) for `mix` mode | Entirely unimplemented | Unimplemented |
| TDD §1 (L464-990) | Networked connection, `signin()`, `_get_connection(config)` singleton, `config: dict` on storage dataclasses, `filter_keys(dict)`, separate `upsert_many()`, `drop()` → `None`, vector `query()` takes an embedding | Embedded DB, no `signin()`, module-level `_CONNECTION`/`get_connection()`, no `config` field, `filter_keys(set[str])`, upsert is itself batch-capable, `drop()` returns a status dict, vector `query()` takes raw text and computes embeddings internally, adds `cosine_better_than_threshold`, entities keyed by sanitized `entity_name`, doc-status storage has 7 new query methods plus new fields (`content_hash`, `file_path`, `track_id`, `metadata`, `error_msg`) | All meaningful architectural deviations |
| — | `surrealdb_impl.py` docstring cites LightRAG 1.5.4 | `requirements.txt` pins LightRAG 1.3.9 | Meaningful version mismatch |
| TDD §2 `ingest.py` (L993-1414) | `--reset` flag rebuilds tables | No-op — only logs a warning (~L998) | **Bug**, meaningful |
| — | `make_ollama_func()` | Drops the `model` parameter (a deliberate, undocumented fix) | Doc stale |
| — | `init_lightrag()` | Adds an undocumented `initialize_pipeline_status()` call fixing a "pipeline already busy" bug | Undocumented addition |
| — | Insert uses `metadata=` / `doc_id=` | Uses `file_paths=` only; drops chunk-metadata (content_type, has_code) forwarding | Meaningful |
| — | Env vars (L1015-1018) | Missing docs for `EXTRACTION_WORKERS`, `SURREALDB_PATH`, `EMBEDDING_MODEL`, `OLLAMA_MODEL` | Doc gap |
| TDD §3 `api.py` (L1417-1764) | — | Near-perfect fidelity — no meaningful deviations found | None |
| TDD §4 Testing (L1766-1832) | Full testing strategy incl. Docker fixtures, Ollama mocks | Entirely unimplemented — no test files, no pytest deps | Unimplemented |
| TDD §5 Risks (L1834-1980) | HNSW query-syntax risk; LightRAG-not-merged risk | Both materialized as predicted; mitigations (`patch_lightrag.py`, inline comments) implemented as designed | Validated, no deviation |

## 2. Spec items with no implementation

1. Fixed / Recursive / Vector / Paragraph chunking strategies (spec L105, 613-620)
2. Reranker for `mix` mode (spec L423-425)
3. `.env.surrealdb.example`, `examples/lightrag_surrealdb_demo.py` (spec L483-485)
4. `_get_connection(config)` pattern (spec L566-572)
5. `upsert_many()` on any storage class
6. `SCHEMAFULL` / `UNIQUE` strict schemas (spec L591-598, 811-819)
7. `get_status` / `set_status` on doc-status storage (spec L962-981)
8. `--reset` actually dropping/rebuilding tables (spec L998) — functional bug, not just a doc gap
9. The entire testing strategy (spec L1766-1832)
10. Networked SurrealDB Docker service (README is also stale on this point)
11. Independent KEYWORDS-role model (FR-Q-06)
12. Explicit citation formatting (FR-Q-03)

## 3. Code additions with no mention in the spec

1. `docling_to_content_list.py` — an entire alternate, RAG-Anything-style ingestion pipeline
2. `docID.py` — entire file (`stable_doc_id()`)
3. Embedded SurrealKV model plus the `SURREALDB_PATH` env var
4. `patch_lightrag.py` — entire runtime-patching mechanism
5. `extraction_workers` / `EXTRACTION_WORKERS` config (`ingest.py` L111, 138 — absent from `config.yaml`)
6. Rich-based CLI progress bars (`ingest.py` ~38-48, 70-89, 517-522, 538-542)
7. `initialize_pipeline_status()` bugfix call (`ingest.py` L65, 455-461)
8. `rag.finalize_storages()` cleanup in a `finally` block (`ingest.py` 545-548)
9. `SurrealDBGraphStorage` extras: `get_all_labels`, `get_popular_labels`, `search_labels`, `remove_nodes`, `remove_edges`, `get_knowledge_graph` (BFS) — L445-594
10. `is_empty()` on KV / DocStatus storage (L224, 690)
11. `SurrealDBVectorStorage` extras: `get_by_id(s)`, `delete_entity`, `delete_entity_relation`, `get_vectors_by_ids`, `cosine_better_than_threshold` filtering (L305-358)
12. `SurrealDBDocStatusStorage` extras: `get_docs_by_statuses`, `get_docs_by_track_id`, `get_docs_paginated`, `get_doc_by_file_path`, `get_doc_by_file_basename`, `get_doc_by_content_hash`, `get_all_status_counts` (L611-628, 698-796)
13. `_normalise_row()` / `_id_str()` RecordID helpers (L45-59)
14. `_query_lock` serialization of SurrealDB queries (L88, 103-104)
15. `output.txt`, `setup.ps1`, `surreal.exe` at repo root — an undocumented Windows-native setup path outside Docker
16. `config.yaml` lacks the `extraction_workers` key despite its header claiming to be the primary override surface

## Overall assessment

`api.py` is essentially spec-perfect and can stay as-is (TDD §3). `ingest.py` has deliberate, well-reasoned reliability fixes but one real bug (`--reset` is a no-op) and silently drops chunk-metadata forwarding. `surrealdb_impl.py` diverges most heavily — driven by real SDK/LightRAG differences from what the spec assumed when written — and is the highest-priority area for either a doc rewrite or a code fix. `patch_lightrag.py`, `docling_to_content_list.py`, and `docID.py` are substantial components with zero documentation. TDD §4 (testing) is completely unimplemented.

Recommended priority: rewrite the SurrealDB Adapter (TDD §1) and Storage Architecture (Arch §3) sections to match the embedded SurrealKV model, update `.env.example`/README to drop the networked-DB assumption, fix the `--reset` bug, and decide whether to implement or formally drop the reranker, multi-model role separation, and testing strategy commitments.
