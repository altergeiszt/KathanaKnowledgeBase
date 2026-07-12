# Ingestion Pipeline — Safeguards & Operational Runbook

> Version 1.0 · July 2026 · Status: Current
> Covers the reliability safeguards added to [`ingest.py`](../ingest.py) on the `pipeline-tweaks` branch.
> Companion to the LlamaIndex/Neo4j migration design in [`Migration_LlamaIndex.md`](Migration_LlamaIndex.md). Where this document and the (SurrealDB-era) [`GraphRAG_Assistant_Docs.md`](GraphRAG_Assistant_Docs.md) disagree about ingestion, **this document and the code are authoritative.**

---

## 0. Why these exist

A full overnight ingest once completed "successfully" while writing **zero** graph entities. The library under test (`D:\IngestTest`) contained no books whose filename matched the curated slice (`classify.BOOK_LABELS`), so the LLM entity-extraction pass had nothing to do — but nothing in the run said so, and because logging was console-only, the terminal scrollback was the only record and it was gone by morning.

The safeguards below make that failure mode (and its neighbours — crashes, OOM-dropped documents, duplicate re-ingests, half-written checkpoints) **loud, recoverable, or impossible.**

---

## 1. The two-phase pipeline (context)

`ingest.py` runs in two halves, split at the post-dedup checkpoint:

1. **Extract half** — discover → parse (docling/EasyOCR, multiprocessing) → chunk → global MinHash dedup → write `chunks_checkpoint.json`. Slow, CPU/RAM-bound, crash-prone. Needs no database.
2. **Insert half** — load checkpoint → split by curated slice → embed-only insertion + curated-slice LLM entity extraction into Neo4j, in resumable batches (§2.7). Needs Neo4j + Ollama.

The halves can be run separately: `--extract-only` stops after phase 1; `--from-checkpoint` skips phase 1 and runs phase 2 from the saved checkpoint.

**Key architectural fact for the safeguards:** deduplication ([`deduplicate()`](../ingest.py)) is a *global, cross-book* MinHash pass (keep first occurrence across the whole corpus). A single book cannot be "deduplicated in isolation," so incremental checkpointing happens at the **pre-dedup** (raw) stage, and dedup remains one pass over the reassembled corpus.

---

## 2. Safeguards

### 2.1 Per-book raw cache (crash resilience)

**Problem:** a crash or OOM at hour 6 of extraction discarded all prior parsing.

**Fix:** each book's raw (pre-dedup) chunks are written to `lightrag_data/raw_chunks/<stem>.json` the moment its worker returns. On any re-run, already-parsed books are loaded from cache and only missing/stale books are re-parsed; dedup then runs once over the reassembled set.

- **Staleness:** each cache entry stores a `size:mtime` fingerprint of the source file. Edit a PDF/EPUB and its cache is invalidated automatically — it re-parses instead of serving stale chunks.
- **Failure-aware:** an extraction failure returns an empty chunk list, which is **not** cached — so a transient OOM is retried next run rather than frozen as "done."
- **Corruption-tolerant:** a half-written cache file (crash mid-write) fails to parse and is treated as absent.

Helpers: `raw_cache_dir` / `save_raw_book` / `load_raw_book` in [`ingest.py`](../ingest.py). Cache lives under `working_dir` and is gitignored.

### 2.2 Empty-curated guard + `--require-curated` (silent no-op)

**Problem:** the original incident — no library book matched `BOOK_LABELS`, so the graph got embeddings only and zero entities, silently.

**Fix:** after the hybrid split, if `curated_nodes` is empty the pipeline logs a loud warning naming the likely cause (filenames vs `BOOK_LABELS`) and listing the books it saw. Matching is **exact stem == title** — `clean_code.pdf` does *not* match `Clean Code`.

- Default: warn and continue (an embeddings-only corpus is a legitimate run).
- `--require-curated`: **abort** instead of warn — use this whenever a run is *meant* to extract entities, so it can never silently finish embeddings-only.

### 2.3 Preflight checks + `--skip-preflight` (fail fast)

**Problem:** a missing Ollama model or empty library only surfaced *after* extraction — hours in, at the first insert.

**Fix:** [`preflight()`](../ingest.py) runs before any expensive work:

- **Library:** `library_path` exists and contains at least one PDF/EPUB (fresh runs only).
- **Ollama:** the host is reachable and the configured extraction model (`ollama_model`, e.g. `qwen2.5:14b`) is actually pulled — checked via `GET {ollama_host}/api/tags`. Any run that will insert requires this.
- Neo4j reachability is already validated early by `init_stores()`, so preflight doesn't duplicate it.

Escape hatch: `--skip-preflight` bypasses the Ollama/library checks.

### 2.4 Dropped-document detection

**Problem:** the multiprocessing pool can silently drop a document via a worker `MemoryError`, yielding a near-empty result set instead of a clean failure.

**Fix:** after extraction, the pipeline compares discovered files against the set of source paths that actually produced chunks and **warns loudly**, listing any book that yielded zero chunks — the signature of an extraction failure or OOM.

### 2.5 Idempotency + resume guard (no duplicate re-ingests)

**Problem:** re-running the insert half without `--reset` stacks duplicate nodes. Note *why*: `TextNode` ids are **random per run**, so Neo4j's `MERGE`-by-id gives no cross-run idempotency — a re-run creates fresh-id copies rather than merging. And `delete_ref_doc` cannot purge the directly-upserted graph nodes/relations afterward (see [`Migration_LlamaIndex.md`](Migration_LlamaIndex.md) §7 and the `delete-ref-doc-blind-to-direct-upserts` finding), so there is no clean re-extract.

**Fix:** before insertion, if `--reset` was **not** passed, `count_book_nodes()` checks the book namespace. If it's already populated the run aborts telling you to re-run with `--reset` — **unless** a matching extraction-progress checkpoint (§2.7) marks this as a *resume* of a partial run, in which case it continues with the unfinished tail. With `--reset`, the namespace was already purged (count zero) and progress is cleared, so insertion proceeds fresh.

| State on re-run | Without `--reset` | With `--reset` |
|-----------------|-------------------|----------------|
| Empty namespace | Insert fresh | Insert fresh |
| Populated, **no** progress file (prior *completed* run) | **Abort** — would duplicate | Purge + insert fresh |
| Populated, **matching** progress file (prior *crashed* run) | **Resume** unfinished tail | Purge + insert fresh |
| Populated, **stale** progress file (corpus changed) | **Abort** — would duplicate | Purge + insert fresh |

### 2.7 Extraction resume checkpoint (the expensive LLM pass)

**Problem:** the curated-slice insertion runs `SimpleLLMPathExtractor` **per chunk** (seconds of LLM time each) inside a *single* `insert_nodes()` call over thousands of chunks. It has no resume point — a crash at chunk 800/1644 restarts the entire pass. `chunks_checkpoint.json` checkpoints the *input* to this step, not *progress through* it, and (per §2.5) random node ids mean a naive re-run duplicates rather than resumes.

**Fix:** insertion runs in **chunk-batches** (`EXTRACT_BATCH_CHUNKS`, default 50), and after each batch *commits* to Neo4j, the completed chunks are recorded in `lightrag_data/extract_progress.json` by a **stable content key** — `sha1(source_path + text)`, not the volatile `node_id`, so it survives re-building the nodes on resume. A resume loads the done-keys and skips them, re-running only the unfinished tail. Both the embeddings-only and curated passes are tracked (separate `embed` / `curated` buckets), so a crash anywhere in insertion resumes cleanly.

- **Corpus-bound:** the progress file stores a `size:mtime` fingerprint of `chunks_checkpoint.json`. If the checkpoint changes, progress is stale and ignored (you can't resume against a different corpus).
- **Commit-ordered:** progress is written *after* each batch's Neo4j commit and via `_atomic_write_json`, so a crash never marks un-committed work as done.
- **Self-clearing:** the file is deleted on clean completion and on `--reset`, so a *completed* run leaves a populated namespace with no progress file — which the §2.5 guard correctly treats as "don't re-ingest without `--reset`."

Helpers: `load_extract_progress` / `save_extract_progress` / `clear_extract_progress` / `insert_chunks_batched` in [`ingest.py`](../ingest.py). Bounds worst-case re-work on a crash to one batch (~50 chunks).

### 2.6 Atomic writes + file logging (durability & diagnosability)

- **Atomic checkpoint/cache writes:** `_atomic_write_json()` writes to a temp file, `fsync`s, then `os.replace()`s into place. A crash mid-write can never leave a truncated `chunks_checkpoint.json` that a later run would load as a valid (short) checkpoint.
- **File logging:** every run attaches a timestamped `FileHandler` writing to `lightrag_data/logs/ingest_YYYYMMDD_HHMMSS.log`, in addition to the Rich console. This is the direct fix for the original incident: a crashed or overnight run now always leaves an on-disk record.

### 2.8 Config precedence — single source of truth

**Problem:** `library_path` (and every other setting) could be set in *both* `.env` and `config.yaml`. The old `load_config` applied `defaults.update(file_cfg)`, so **config.yaml silently won over `.env`** — the opposite of config.yaml's own header comment ("can be overridden by environment variables in .env"). Worse, several config.yaml keys (`dedup_threshold`, `dedup_num_perm`, `dedup_shingle_k`, `chunk_max_chars`, `content_type_rules`) were never read into `PipelineConfig` at all, so setting them in config.yaml did nothing.

**Fix:** a single documented precedence, highest wins:

```
environment / .env   >   config.yaml   >   built-in (dataclass) defaults
```

- **Only env vars that are actually set** override — an unset var never clobbers a config.yaml value with a hardcoded fallback (the previous code's `os.getenv(K, default)` did exactly that).
- **No dropped keys:** the config is built from the full `PipelineConfig` field list, not a hand-picked subset, so every recognized config.yaml key applies. Unknown keys log a warning.
- **Conflicts are loud:** a key set in both `.env` and config.yaml with *different* values logs a `Config conflict on '<key>': ... (env wins)` warning at startup.
- **Visibility:** the resolved `library_path` / `working_dir` / `neo4j_db` / `ollama_model` are logged (and persisted to the file log) at the start of every run, so a wrong library/DB is visible up front instead of inferred from bad results.

The env↔config mapping lives in `_ENV_SPEC` in [`ingest.py`](../ingest.py). Secrets (e.g. `NEO4J_PASSWORD`) belong in `.env` only; see [`.env.example`](../.env.example).

**Extraction model split.** The curated-slice LLM extraction model is configurable independently of the general `OLLAMA_MODEL` via **`EXTRACT_MODEL`** (falls back to `OLLAMA_MODEL` when unset), so you can point extraction at a fast small model (e.g. `qwen3:4b-q8_0`) without disturbing a future query-role model. Preflight checks the *extraction* model is pulled.

**Thinking mode.** `EXTRACT_THINKING` = `auto` (default) | `on` | `off` | `none` controls hybrid-reasoning ("thinking") models. For high-volume structured extraction you want thinking **off** — it otherwise emits reasoning tokens per chunk, which is slow and can muddy output. `auto` disables thinking for thinking-capable models (qwen3, deepseek-r1, gpt-oss, …, matched by name substring) and **omits the flag for non-thinking models** like qwen2.5 (sending `think:false` to those can error). `none` **omits the flag entirely regardless of the model name** — the escape hatch when the name heuristic misjudges a model (e.g. a new family like `qwen3.5` that the `qwen3` substring flags as thinking-capable but which rejects the `think` param). Resolved by `_resolve_thinking()` in [`ingest.py`](../ingest.py).

> Note: `.env` is loaded with `load_dotenv(override=True)`, so within the environment layer the `.env` file also overrides a pre-existing *shell* environment variable.

---

## 3. CLI reference

| Flag | Effect |
|------|--------|
| `--config PATH` | Path to `config.yaml`. |
| `--reset` | Purge the Neo4j book namespace before inserting. Required to re-ingest cleanly (see §2.5). |
| `--extract-only` | Run phase 1 only (parse + dedup + checkpoint), then stop. No Neo4j/Ollama needed. |
| `--from-checkpoint` | Skip phase 1; run phase 2 from the saved `chunks_checkpoint.json`. |
| `--require-curated` | Abort (not warn) if no library book matches the curated slice (§2.2). |
| `--skip-preflight` | Bypass the Ollama-model / non-empty-library preflight checks (§2.3). |
| `--extract-all` | Send **every** book through LLM extraction (whole-corpus graph), bypassing the curated gate. Intended with the schema extractor. |

**Extractor & schema.** `EXTRACTOR` = `schema` (default) | `simple`. `schema` uses `SchemaLLMPathExtractor` constrained to a fixed entity/relation vocabulary enforced via the LLM's structured output, which keeps the graph traversable (the free-form `simple` extractor produced ~2 relations per type; a schema yields dozens per type). The vocabulary is loaded from a JSON file via `SCHEMA_PATH` (e.g. `schemas/target_schema.json` or `schemas/test_schema.json`) — shape `{"entity_types":[{"type":…}], "relation_types":[{"type":…}]}`; unset falls back to the built-in lists in [`ingest.py`](../ingest.py). See [`Schema_Vocabulary_Handoff.md`](Schema_Vocabulary_Handoff.md) for how the vocabulary is designed. `extract_strict` (config, default true) additionally prunes off-vocabulary triples. `--extract-all` + `schema` is the intended pairing for graphing the whole corpus.
| `--profile` | Per-stage timing via `pipeline_profiler` (if available). |

`--extract-only` and `--from-checkpoint` are mutually exclusive.

---

## 4. Working-directory layout

Everything lives under `working_dir` (default `./lightrag_data`, gitignored):

```
lightrag_data/
├── chunks_checkpoint.json     # post-dedup chunk list (phase-1 output, phase-2 input)
├── extract_progress.json      # insert-half resume state (§2.7); deleted on clean finish
├── raw_chunks/                # per-book pre-dedup cache (§2.1)
│   └── <book-stem>.json        #   { source, fingerprint, chunks[] }
├── logs/                      # per-run file logs (§2.6)
│   └── ingest_YYYYMMDD_HHMMSS.log
└── profile_results/           # --profile output
```

---

## 5. Runbook

**Fresh full ingest (recommended):**
```powershell
.venv\Scripts\Activate.ps1
ollama list                      # confirm qwen2.5:14b is pulled (preflight also checks)
python ingest.py --reset --require-curated
```

**Resume after a crash during extraction (phase 1):** just re-run the same command — cached books are skipped, only the unparsed remainder is re-parsed (§2.1).

**Resume after a crash during insertion/extraction (phase 2):** re-run **without** `--reset` (e.g. `python ingest.py --from-checkpoint`). The matching `extract_progress.json` lets it skip the chunks already committed and re-run only the unfinished tail (§2.7). Using `--reset` here instead would discard the completed work and restart the whole LLM pass.

**Split run (parse tonight, insert tomorrow):**
```powershell
python ingest.py --extract-only                       # phase 1, no DB needed
python ingest.py --from-checkpoint --reset --require-curated   # phase 2
```

**Re-ingest an existing graph:** always pass `--reset`. Without it, the idempotency guard (§2.5) aborts to prevent duplicate nodes.

**Verify a run actually produced entities** (Neo4j):
```cypher
MATCH (e:__Entity__) RETURN count(e);
MATCH ()-[r]->() RETURN count(r);
```
Zero on both after a run that was supposed to extract means the curated slice was empty — check filenames against `classify.BOOK_LABELS` (exact stem match). `--require-curated` turns this into an up-front abort.

---

## 6. Related

- [`Migration_LlamaIndex.md`](Migration_LlamaIndex.md) — LlamaIndex/Neo4j architecture (§5 hybrid insertion, §7 rebuild/purge, §11 curated slice).
- [`curated-slice.md`](curated-slice.md) — the hand-labeled book list behind `classify.BOOK_LABELS`.
- [`GraphRAG_Assistant_Docs.md`](GraphRAG_Assistant_Docs.md) — original SRS/Architecture/TDD (SurrealDB-era; superseded on ingestion specifics).
