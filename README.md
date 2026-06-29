# GraphRAG Assistant — Setup & Testing Guide

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| Docker Desktop | Runs SurrealDB, the API, and OpenWebUI | [docker.com](https://www.docker.com/products/docker-desktop) |
| Ollama | Runs LLMs locally on your GPU | [ollama.com](https://ollama.com) |
| `qwen2.5:14b` model | Entity extraction + query answering | `ollama pull qwen2.5:14b` |

> **GPU note:** Ollama runs on your host machine (not in Docker), so it has direct access to your RTX 4080 Super without any NVIDIA Container Toolkit setup.

---

## Project Layout

Place all these files in one directory:

```
graphrag-assistant/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── patch_lightrag.py
├── surrealdb_impl.py
├── ingest.py
├── api.py
├── config.yaml
├── .env                ← copy from .env.example and edit
└── .env.example
```

---

## Step 1 — Configure your environment

```bash
cp .env.example .env
```

Open `.env` and set exactly one value:

```env
LIBRARY_PATH=/absolute/path/to/your/ebooks
```

Everything else works as-is for a local Ollama + Docker setup. SurrealDB runs on host port `8001` to avoid clashing with the API on `8000`.

---

## Step 2 — Pull the Ollama model

```bash
ollama pull qwen2.5:14b
```

This downloads ~8 GB. Make sure Ollama is running (`ollama serve` if not already started as a service).

---

## Step 3 — Build the Docker image

```bash
docker compose build
```

This installs all Python dependencies and runs `patch_lightrag.py` to copy `surrealdb_impl.py` into LightRAG and register the four storage classes. You'll see patch output in the build log:

```
=== Patching lightrag-hku to register SurrealDB adapter ===
  Copied surrealdb_impl.py → .../lightrag/kg/surrealdb_impl.py
  Patched .../lightrag/kg/__init__.py
    + STORAGE_IMPLEMENTATIONS["SurrealDBKVStorage"] = "..."
    ...
=== Patch complete ===
```

---

## Step 4 — Start SurrealDB and the API

```bash
docker compose up -d surrealdb api
```

Check both are healthy:

```bash
docker compose ps
# surrealdb should show "healthy"
# api should show "running"

# Check the API is up
curl http://localhost:8000/health
# → {"status":"ok","provider":"ollama"}

# Check query modes are visible
curl http://localhost:8000/api/tags
# → {"models":[{"name":"mix"}, ...]}
```

---

## Step 5 — Run ingestion

Ingestion is a one-shot job, not a long-running service. It runs, indexes everything, then exits.

```bash
docker compose --profile ingest run --rm ingest
```

You'll see progress logs like:

```
INFO  Discovered 247 documents in /library
INFO  Extracting 247 documents using 8 workers...
INFO  Extraction complete: 142,331 raw chunks
INFO  Deduplication: 142,331 → 91,204 chunks (51,127 removed)
INFO  Inserting 91,204 chunks (concurrency=4)...
```

This will take a while — overnight is expected for a 14 GB library. If it's interrupted, re-run the same command. LightRAG's `DocStatusStorage` tracks which documents are done and skips them automatically.

**To re-index from scratch:**
```bash
docker compose --profile ingest run --rm ingest python ingest.py --reset
```

---

## Step 6 — Start OpenWebUI

```bash
docker compose up -d openwebui
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

1. Create an account (local only, no external signup)
2. In the model picker, select one of: `mix`, `hybrid`, `local`, `global`, `naive`
3. Ask a question about your library

---

## Troubleshooting

### Patch failed / `STORAGE_IMPLEMENTATIONS` not found

LightRAG restructured in a newer version. Check what's in the installed `__init__.py`:

```bash
docker compose run --rm api python -c "
import importlib.util, pathlib
spec = importlib.util.find_spec('lightrag')
init = pathlib.Path(spec.origin).parent / 'kg' / '__init__.py'
print(init.read_text()[:2000])
"
```

Look for the dict name and update `patch_lightrag.py` accordingly.

### SurrealDB SDK response unwrap fails

The `SurrealDBDB.query()` method in `surrealdb_impl.py` unwraps SDK responses. If you see `KeyError: 'result'` or empty results where you expect data, print the raw SDK response to see its shape:

```bash
docker compose run --rm api python -c "
import asyncio
from surrealdb import Surreal

async def check():
    db = Surreal('ws://surrealdb:8000/rpc')
    await db.connect()
    await db.signin({'user': 'root', 'pass': 'root'})
    await db.use('lightrag', 'assistant')
    raw = await db.query('SELECT 1 AS test')
    print(type(raw), raw)

asyncio.run(check())
"
```

Adjust the unwrap logic in `SurrealDBDB.query()` to match the actual structure.

### Ollama not reachable from container

`host.docker.internal` resolves to your host machine on Docker Desktop (Windows/Mac). On Linux it may not be set automatically. Add this to the `api` and `ingest` services in `docker-compose.yml`:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

### Check ingestion status

```bash
docker compose run --rm api python -c "
import asyncio
from surrealdb_impl import SurrealDBDocStatusStorage

async def check():
    storage = SurrealDBDocStatusStorage()
    storage.namespace = 'default'
    await storage.initialize()
    counts = await storage.get_status_counts()
    print(counts)

asyncio.run(check())
"
```

---

## Useful commands

```bash
# View API logs
docker compose logs -f api

# View SurrealDB logs
docker compose logs -f surrealdb

# Open SurrealDB query console (runs surreal sql in the container)
docker compose exec surrealdb surreal sql \
  --conn ws://localhost:8000 --user root --pass root \
  --ns lightrag --db assistant

# Stop everything
docker compose down

# Stop and delete all data volumes (full reset)
docker compose down -v
```
