# GraphRAG Assistant — Setup & Run Guide

A personal GraphRAG assistant over your ebook library, built on LightRAG with an
**embedded SurrealDB (SurrealKV) backend**. Everything runs locally — no Docker,
no database server, no cloud services (except optional Claude API calls at query time).

> **Storage model:** the assistant uses SurrealDB's embedded **SurrealKV** engine.
> The database is a single local file (`lightrag_data/graphrag.db`) opened in-process
> via `surrealkv://…`. There is **no SurrealDB server to run** and no `surreal.exe`
> required — the `surrealdb` Python SDK provides the embedded engine.

---

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| Python 3.11+ | Runs the pipeline and API | [python.org](https://www.python.org/downloads/) |
| Ollama | Runs LLMs locally on your GPU | [ollama.com](https://ollama.com) |
| `qwen2.5:14b` model | Entity extraction (+ optional query answering) | `ollama pull qwen2.5:14b` |

> **GPU note:** Ollama and `sentence-transformers` both use your RTX 4080 Super
> directly on the host — no container GPU setup required.

---

## Project Layout

```
KathanaKnowledgeBase/
├── surrealdb_impl.py      # Embedded SurrealDB (SurrealKV) storage adapter for LightRAG
├── patch_lightrag.py      # One-time setup: registers the adapter into installed LightRAG
├── ingest.py              # Ingestion pipeline (extract → dedup → index)
├── api.py                 # FastAPI bridge (Ollama-compatible; OpenWebUI/GraphNotes connect here)
├── docID.py               # stable_doc_id() helper
├── docling_to_content_list.py  # Future: RAG-Anything converter (scaffold, unused)
├── config.yaml            # Pipeline config
├── requirements.txt
├── setup.ps1              # Native Windows setup (venv + deps + patch + .env)
├── .env                   # copy from .env.example and edit
├── .env.example
└── lightrag_data/
    └── graphrag.db        # Embedded SurrealKV database (created on first run)
```

---

## Quick start (Windows / PowerShell)

The one-shot setup script creates the virtualenv, installs dependencies, patches
LightRAG, and creates your `.env`:

```powershell
pwsh ./setup.ps1
```

Then edit `.env` and set your library path (see Step 1 below), and skip to Step 3.

If you prefer to do it manually, follow all steps below.

---

## Step 1 — Configure your environment

```powershell
Copy-Item .env.example .env
```

Open `.env` and set the one required value:

```env
LIBRARY_PATH=C:\absolute\path\to\your\ebooks
```

Everything else works as-is for a local Ollama setup. To answer queries with
Claude Haiku instead of local Ollama, set `QUERY_LLM_PROVIDER=anthropic` and
`ANTHROPIC_API_KEY=…` (ingestion always uses Ollama regardless).

---

## Step 2 — Install dependencies and patch LightRAG

If you did **not** run `setup.ps1`:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

# One-time: copy the adapter into the installed LightRAG package and register it.
# Re-run this after any reinstall/upgrade of lightrag-hku.
.venv\Scripts\python.exe patch_lightrag.py
```

You'll see patch output like:

```
=== Patching lightrag-hku for SurrealDB adapter ===
  Copied surrealdb_impl.py → .../lightrag/kg/surrealdb_impl.py
  STORAGES: added SurrealDBKVStorage
  ...
  Verification passed — all classes registered correctly
=== Done ===
```

---

## Step 3 — Pull the Ollama model

```powershell
ollama pull qwen2.5:14b
```

This downloads ~8 GB. Make sure Ollama is running (`ollama serve` if it isn't
already running as a service).

---

## Step 4 — Run ingestion

Ingestion is a one-shot job: it extracts, deduplicates, indexes, then exits.
The embedded database file is created automatically on first run.

```powershell
.venv\Scripts\python.exe ingest.py --config config.yaml
```

You'll see progress like:

```
INFO  Discovered 247 documents in ./library
INFO  Extracting 247 documents using 8 workers...
INFO  Extraction complete: 142,331 raw chunks
INFO  Deduplication: 142,331 → 91,204 chunks (51,127 removed)
INFO  Inserting 91,204 chunks (concurrency=4)...
```

This will take a while — overnight is expected for a large library. If it's
interrupted, just re-run the same command: `SurrealDBDocStatusStorage` tracks
which documents are done and skips them automatically.

**To re-index from scratch** (deletes the embedded database file, then rebuilds):

```powershell
.venv\Scripts\python.exe ingest.py --config config.yaml --reset
```

---

## Step 5 — Start the API

```powershell
.venv\Scripts\uvicorn.exe api:app --host 0.0.0.0 --port 8000
```

Verify it's up:

```powershell
# Health
curl http://localhost:8000/health
# → {"status":"ok","provider":"ollama"}

# Query modes (surfaced to OpenWebUI as selectable "models")
curl http://localhost:8000/api/tags
# → {"models":[{"name":"mix"}, ...]}
```

The API is Ollama-compatible. GraphNotes (or any Ollama client) can connect to
`http://localhost:8000`.

---

## Step 6 — (Optional) OpenWebUI front-end

Docker has been removed from this project, so run OpenWebUI however you prefer
(e.g. `pip install open-webui` then `open-webui serve`, or the desktop app) and
point it at this API:

- Set OpenWebUI's Ollama base URL to `http://localhost:8000`
- In the model picker, choose one of: `mix`, `hybrid`, `local`, `global`, `naive`
- Ask a question about your library

---

## Troubleshooting

### Patch failed / storage classes not found

LightRAG may have restructured in a newer version. Inspect the installed
`__init__.py` and update `patch_lightrag.py`'s markers if needed:

```powershell
.venv\Scripts\python.exe -c "import importlib.util, pathlib; spec = importlib.util.find_spec('lightrag'); init = pathlib.Path(spec.origin).parent / 'kg' / '__init__.py'; print(init.read_text()[:2000])"
```

Remember to re-run `patch_lightrag.py` after every `pip install`/upgrade of
`lightrag-hku` — the patch modifies files inside the installed package.

### Inspect the embedded database

The database is a local SurrealKV file. Query it in-process (stop the API/ingest
first, since the embedded engine is single-writer):

```powershell
.venv\Scripts\python.exe -c "import asyncio; from surrealdb import AsyncSurreal; import os
async def check():
    db = AsyncSurreal('surrealkv://./lightrag_data/graphrag.db')
    await db.connect(); await db.use('lightrag', 'assistant')
    print(await db.query('SELECT count() AS n FROM doc_status_default GROUP ALL'))
    await db.close()
asyncio.run(check())"
```

### Check ingestion status

```powershell
.venv\Scripts\python.exe -c "import asyncio; from surrealdb_impl import SurrealDBDocStatusStorage
async def check():
    s = SurrealDBDocStatusStorage(); s.namespace = 'default'
    await s.initialize(); print(await s.get_status_counts())
asyncio.run(check())"
```

### Ollama not reachable

Confirm Ollama is running and the model is pulled:

```powershell
curl http://localhost:11434/api/tags
ollama list
```

---

## Useful notes

- **Reset everything:** delete `lightrag_data/graphrag.db` (or run ingestion with `--reset`).
- **Change embedding model:** update `EMBEDDING_MODEL` and `SURREALDB_VECTOR_DIM` in `.env`
  to match the model's output dimension, then re-ingest with `--reset` (vectors must be rebuilt).
- **Query-time LLM:** `QUERY_LLM_PROVIDER=ollama` (fully local) or `anthropic` (Claude Haiku via API).
