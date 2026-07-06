"""
delete_completeness_harness.py

GraphRAG Assistant — Delete-Completeness Harness (current LightRAG + SurrealKV stack)

This is the §7 "delete-verify" harness from Migration_LlamaIndex.md, adapted to
MEASURE THE CURRENT STATE of the live ingest path *before* migrating off LightRAG.

Why this exists
---------------
Spaced-repetition means an edited note is re-ingested repeatedly, and each update is a
delete-old-then-insert-new. The load-bearing risk (§7) is that a delete may clear the
document/docstore layer but leave stale nodes behind in the *graph* / *vector* stores,
so old content keeps answering queries. The migration doc raises this for the future
target framework; the same `adelete_by_doc_id` completeness question already applies to
the CURRENT LightRAG + embedded-SurrealKV stack.

Running this against today's stack gives the empirical baseline: does
`rag.adelete_by_doc_id()` actually purge every store, or does it leave orphans? That is
the "gauge the current state of our ingest points" deliverable.

What it does
------------
1.  Spins up LightRAG on an ISOLATED throwaway SurrealKV database (never touches your
    real ./lightrag_data/graphrag.db).
2.  Inserts note v1 carrying a unique sentinel token/entity (ALPHA).
3.  Confirms v1 is present (raw-store scan + retrieval query).
4.  Deletes v1 via rag.adelete_by_doc_id(<doc_id>).
5.  Inserts note v2 with a different sentinel (BETA), same doc_id.
6.  THE CRITICAL CHECK: scans EVERY table in the embedded store for surviving ALPHA
    orphans (chunks, vector rows, graph entities, graph relations, doc_status, KV,
    LLM cache), and confirms retrieval now surfaces BETA and not ALPHA.
7.  Prints a per-store PASS/FAIL breakdown and an overall verdict.

Requirements
------------
- Ollama running (entity extraction happens inside ainsert; that's what populates the
  graph store, which is the whole thing we're testing). Uses OLLAMA_MODEL from env.
- The sentence-transformers embedding model (all-MiniLM-L6-v2 by default).
- The lightrag/kg SurrealDB storage classes already registered on disk
  (patch_lightrag.py has been run — it has, since ingest.py/api.py work).

Usage
-----
    python delete_completeness_harness.py
    python delete_completeness_harness.py --keep-db     # don't wipe the test DB after
    python delete_completeness_harness.py --db-dir ./surrealdb_data_test

Exit code is 0 if the delete purged cleanly (harness PASSES), 1 otherwise — so this can
gate CI / the migration go/no-go.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Sentinels — coined proper-noun tokens that will not collide with real corpus
# content and that an LLM extractor is likely to lift out as named entities, so
# we exercise BOTH the raw-chunk stores and the graph (entity/relation) stores.
# ---------------------------------------------------------------------------
ALPHA = "Zanderalpha"   # lives only in note v1
BETA = "Zanderbeta"     # lives only in note v2
DOC_ID = "doc-harness-note-concept-x"  # fixed id so delete-by-doc-id is deterministic
SOURCE_PATH = "harness://Concept X.md"


def note_body(sentinel: str) -> str:
    """A note-shaped document. Phrased so the sentinel reads as a named entity
    (proper noun + explicit relation) to give LLM entity extraction something to
    pull into the graph store — that's the layer §7 warns can leak on delete."""
    return (
        f"# Concept X\n\n"
        f"## What is Concept X?\n\n"
        f"Concept X is fundamentally about {sentinel}. "
        f"{sentinel} is the single defining idea of Concept X, and Concept X "
        f"cannot be understood without {sentinel}.\n\n"
        f"## How does Concept X relate to other ideas?\n\n"
        f"Concept X relates directly to {sentinel}. The relationship between "
        f"Concept X and {sentinel} is the core of this note.\n"
    )


# ---------------------------------------------------------------------------
# Environment isolation
#
# ingest.py runs load_dotenv(override=True) at import, which would clobber any
# env we set beforehand with the values in .env. So we import ingest FIRST, then
# override the env to point at a throwaway DB. SurrealDBDB reads SURREALDB_PATH
# lazily (at get_connection(), during initialize_storages()), so setting it here
# — before init — is picked up, and the real ./lightrag_data store is untouched.
# ---------------------------------------------------------------------------

def isolate_env(db_dir: Path) -> tuple[Path, Path]:
    import ingest  # noqa: F401 — imported for its side-effect (load_dotenv)

    db_dir = db_dir.resolve()
    db_path = db_dir / "harness.db"
    working_dir = db_dir / "working"
    working_dir.mkdir(parents=True, exist_ok=True)

    os.environ["SURREALDB_PATH"] = str(db_path).replace("\\", "/")
    os.environ["LIGHTRAG_WORKING_DIR"] = str(working_dir)
    # Keep namespace/database at defaults; isolation is by file path.
    os.environ.setdefault("SURREALDB_NAMESPACE", "lightrag")
    os.environ.setdefault("SURREALDB_DATABASE", "assistant")
    return db_path, working_dir


# ---------------------------------------------------------------------------
# Raw-store inspection — reuse LightRAG's OWN embedded connection
#
# Embedded SurrealKV is single-opener: a second process/handle on the file can
# lock or miss buffered writes. So we scan through surrealdb_impl.get_connection()
# (the same singleton the storages use), guaranteeing we see exactly what the
# storages just wrote.
# ---------------------------------------------------------------------------

async def _list_tables(db) -> list[str]:
    rows = await db.query("INFO FOR DB")
    if not rows:
        return []
    first = rows[0]
    tables = first.get("tables", {}) if isinstance(first, dict) else {}
    return sorted(tables.keys())


async def scan_for_token(db, token: str) -> dict[str, int]:
    """Return {table_name: hit_count} for every table containing `token` anywhere
    in any record (each record serialized to JSON so we catch the token in content,
    metadata, entity names, relation endpoints, or cached LLM output — anywhere)."""
    hits: dict[str, int] = {}
    for table in await _list_tables(db):
        try:
            rows = await db.query(f"SELECT * FROM {table}")
        except Exception as exc:  # a table we can't read shouldn't abort the scan
            console.print(f"[yellow]  (could not scan {table}: {exc})[/yellow]")
            continue
        count = 0
        for row in rows or []:
            blob = json.dumps(row, default=str, ensure_ascii=False)
            if token.lower() in blob.lower():
                count += 1
        if count:
            hits[table] = count
    return hits


async def table_row_counts(db) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in await _list_tables(db):
        try:
            rows = await db.query(f"SELECT count() AS c FROM {table} GROUP ALL")
            counts[table] = rows[0]["c"] if rows else 0
        except Exception:
            counts[table] = -1
    return counts


# ---------------------------------------------------------------------------
# Retrieval probe — does a query still surface the stale content?
# ---------------------------------------------------------------------------

async def retrieval_contains(rag, token: str) -> bool:
    """Query the RAG for the note's concept and check whether the retrieved
    context still contains the sentinel. Uses only_need_context=True to get the
    raw retrieved chunks/graph context without spending an LLM synthesis call —
    this is the true 'does stale content still get retrieved' signal."""
    from lightrag import QueryParam

    try:
        context = await rag.aquery(
            "What is Concept X about?",
            param=QueryParam(mode="naive", only_need_context=True),
        )
    except Exception as exc:
        console.print(f"[yellow]  (retrieval probe failed, mode=naive: {exc})[/yellow]")
        return False
    return token.lower() in (context or "").lower()


# ---------------------------------------------------------------------------
# Harness scenario
# ---------------------------------------------------------------------------

async def run_harness(db_dir: Path, keep_db: bool) -> bool:
    # Wipe any prior test DB for a clean slate BEFORE anyone opens the file.
    if db_dir.exists():
        shutil.rmtree(db_dir, ignore_errors=True)
    db_path, working_dir = isolate_env(db_dir)
    console.rule("[bold]Delete-Completeness Harness — current LightRAG + SurrealKV stack")
    console.print(f"Isolated test DB : [cyan]{db_path}[/cyan]")
    console.print(f"Working dir      : [cyan]{working_dir}[/cyan]")
    console.print(f"Sentinels        : v1=[green]{ALPHA}[/green]  v2=[green]{BETA}[/green]")
    console.print(f"doc_id           : [cyan]{DOC_ID}[/cyan]\n")

    # Imports deferred until AFTER isolate_env so env overrides are in effect.
    from ingest import load_config, init_lightrag
    from surrealdb_impl import get_connection, close_connection

    config = load_config(None)
    rag = await init_lightrag(config)

    results: dict[str, bool] = {}
    try:
        db = await get_connection()

        # ── Step 1: insert v1 (ALPHA) ────────────────────────────────────
        console.print("[bold]Step 1[/bold]  Insert note v1 (ALPHA)…")
        await rag.ainsert(note_body(ALPHA), ids=DOC_ID, file_paths=SOURCE_PATH)

        v1_hits = await scan_for_token(db, ALPHA)
        console.print(f"  ALPHA present in {len(v1_hits)} table(s): {v1_hits or '∅'}")
        results["v1 ingested (ALPHA lands in ≥1 store)"] = bool(v1_hits)
        results["v1 retrievable (query surfaces ALPHA)"] = await retrieval_contains(rag, ALPHA)

        # ── Step 2: delete v1 ────────────────────────────────────────────
        console.print("\n[bold]Step 2[/bold]  Delete v1 via adelete_by_doc_id()…")
        deletion = await rag.adelete_by_doc_id(DOC_ID)
        status = getattr(deletion, "status", deletion)
        message = getattr(deletion, "message", "")
        console.print(f"  DeletionResult: status=[cyan]{status}[/cyan] {message}")
        results["delete reported success"] = (status == "success")

        # ── Step 3: insert v2 (BETA), same doc_id ────────────────────────
        console.print("\n[bold]Step 3[/bold]  Insert note v2 (BETA) under same doc_id…")
        await rag.ainsert(note_body(BETA), ids=DOC_ID, file_paths=SOURCE_PATH)

        # ── Step 4: THE CRITICAL CHECK — orphan scan ─────────────────────
        console.print("\n[bold]Step 4[/bold]  Scan every store for surviving ALPHA orphans…")
        alpha_orphans = await scan_for_token(db, ALPHA)
        beta_hits = await scan_for_token(db, BETA)

        _print_store_breakdown(await table_row_counts(db), alpha_orphans, beta_hits)

        results["no ALPHA orphans in ANY store"] = not alpha_orphans
        results["v2 present (BETA in ≥1 store)"] = bool(beta_hits)

        # ── Step 5: retrieval reflects the update ────────────────────────
        console.print("\n[bold]Step 5[/bold]  Retrieval probe after update…")
        beta_retrievable = await retrieval_contains(rag, BETA)
        alpha_retrievable = await retrieval_contains(rag, ALPHA)
        console.print(f"  query surfaces BETA : {beta_retrievable}")
        console.print(f"  query surfaces ALPHA: {alpha_retrievable}  (want False)")
        results["v2 retrievable (query surfaces BETA)"] = beta_retrievable
        results["stale v1 NOT retrievable (no ALPHA)"] = not alpha_retrievable

    finally:
        # Flush SurrealKV's write buffer and close the shared connection, then
        # LightRAG's own storages, so nothing is left holding the embedded file.
        try:
            await rag.finalize_storages()
        except Exception:
            pass
        try:
            await close_connection()
        except Exception:
            pass

    passed = _report(results)

    if not keep_db:
        shutil.rmtree(db_dir, ignore_errors=True)
        console.print(f"\nCleaned up test DB at {db_dir} (use --keep-db to retain).")
    else:
        console.print(f"\nRetained test DB at {db_dir} for inspection.")

    return passed


def _print_store_breakdown(counts, alpha_orphans, beta_hits) -> None:
    table = Table(title="Per-store state after delete-then-reinsert", show_lines=False)
    table.add_column("Table", style="cyan", no_wrap=True)
    table.add_column("Rows", justify="right")
    table.add_column("ALPHA orphans", justify="right", style="red")
    table.add_column("BETA (v2)", justify="right", style="green")
    for tbl in sorted(counts):
        a = alpha_orphans.get(tbl, 0)
        b = beta_hits.get(tbl, 0)
        table.add_row(
            tbl,
            str(counts[tbl]),
            f"[bold red]{a}[/bold red]" if a else "0",
            str(b),
        )
    console.print(table)


def _report(results: dict[str, bool]) -> bool:
    console.rule("[bold]Verdict")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Result", justify="center")
    for name, ok in results.items():
        table.add_row(name, "[green]PASS[/green]" if ok else "[red]FAIL[/red]")
    console.print(table)

    # The harness PASSES only if the delete purged cleanly AND the update took.
    # (The "v1 retrievable" / "v1 ingested" checks are sanity preconditions: if
    # they fail the test setup was invalid — Ollama down, extraction produced
    # nothing — rather than a delete-completeness verdict.)
    critical = [
        "no ALPHA orphans in ANY store",
        "v2 present (BETA in ≥1 store)",
        "v2 retrievable (query surfaces BETA)",
        "stale v1 NOT retrievable (no ALPHA)",
    ]
    passed = all(results.get(k, False) for k in critical)
    preconditions_ok = results.get("v1 ingested (ALPHA lands in ≥1 store)", False)

    if not preconditions_ok:
        console.print(
            "\n[bold yellow]INCONCLUSIVE[/bold yellow] — v1 never landed in any store. "
            "Check that Ollama is running and extraction succeeded; the delete verdict "
            "is meaningless without a real v1 to delete."
        )
        return False

    if passed:
        console.print(
            "\n[bold green]HARNESS PASSED[/bold green] — adelete_by_doc_id() purged "
            "every store cleanly on the current stack. The delete path is sound; the "
            "note-update loop can be built on it (§7)."
        )
    else:
        console.print(
            "\n[bold red]HARNESS FAILED[/bold red] — the current delete path leaves "
            "graph/vector orphans (see the red column above). This is the §7 risk, "
            "present TODAY. Fallbacks (§7): manual per-source purge, or rebuild the "
            "note namespace on each update. Carry this into the migration criteria."
        )
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument(
        "--db-dir",
        default="./surrealdb_data_test/delete_harness",
        help="Throwaway directory for the isolated test DB "
             "(default: ./surrealdb_data_test/delete_harness)",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Keep the isolated test DB after the run (for manual inspection).",
    )
    args = parser.parse_args()

    passed = asyncio.run(run_harness(Path(args.db_dir), args.keep_db))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
