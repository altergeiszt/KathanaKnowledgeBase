"""
inspect_chunks.py

Diagnostic script for the GraphRAG Assistant pipeline.

Purpose: pull raw stored chunks straight out of the embedded SurrealKV
database (vec_default table) so we can visually inspect:

  1. Whether chunk-metadata (content_type, has_code, source) actually made
     it into storage, or was dropped at the rag.ainsert(file_paths=...) step
     (see GraphRAG_Spec_Deviations.md — this is a documented deviation).
  2. Whether the *text* itself looks clean — intact code fences, intact
     markdown tables, sane paragraph boundaries — or whether docling /
     semantic-text-splitter already mangled structure before insertion.

This script does NOT use LightRAG or your surrealdb_impl.py adapter at all.
It talks to the embedded SurrealKV file directly via the raw `surrealdb`
Python SDK, read-only. Nothing is modified.

Usage:
    python inspect_chunks.py
    python inspect_chunks.py --db-path ./lightrag_data/graphrag.db --limit 5
    python inspect_chunks.py --source-contains "python_crash_course"
    python inspect_chunks.py --dump-full   # print full chunk text, not truncated

Requires: pip install surrealdb  (already in your requirements.txt)
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path
from typing import Any

try:
    from surrealdb import AsyncSurreal
except ImportError:
    print("ERROR: `surrealdb` package not found. Run: pip install surrealdb")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Heuristic checks — flag likely damage patterns for eyeballing
# ---------------------------------------------------------------------------

def flag_issues(text: str) -> list[str]:
    """Cheap heuristics to flag likely chunking/extraction damage.
    These are hints to look at, not definitive verdicts — use your own eyes too."""
    issues = []

    # Unbalanced code fences (odd number of ``` means a fence got cut mid-block)
    fence_count = text.count("```")
    if fence_count % 2 != 0:
        issues.append(f"UNBALANCED_CODE_FENCE (found {fence_count} ``` markers)")

    # Markdown table rows present but header separator missing/broken
    table_rows = re.findall(r"^\|.*\|$", text, re.MULTILINE)
    if table_rows:
        has_separator = any(re.match(r"^\|[\s\-:|]+\|$", row) for row in table_rows)
        if not has_separator:
            issues.append(f"TABLE_ROWS_NO_SEPARATOR ({len(table_rows)} pipe-rows, no header separator found)")

    # Chunk starts or ends mid-sentence (lowercase start, no terminal punctuation)
    stripped = text.strip()
    if stripped and stripped[0].islower():
        issues.append("STARTS_MID_SENTENCE (first char is lowercase)")
    if stripped and stripped[-1] not in ".!?\"'`)]}":
        issues.append("ENDS_MID_SENTENCE (no terminal punctuation)")

    # Suspiciously short chunk (may indicate over-aggressive splitting)
    if 0 < len(stripped) < 40:
        issues.append(f"VERY_SHORT_CHUNK ({len(stripped)} chars)")

    # Code-like content NOT wrapped in a fence at all (possible missed [CODE] tag)
    code_signals = ("def ", "class ", "import ", "    return ", "=>", "};")
    has_code_signal = any(sig in text for sig in code_signals)
    has_fence = "```" in text
    has_code_tag = "[CODE]" in text
    if has_code_signal and not has_fence and not has_code_tag:
        issues.append("POSSIBLE_UNTAGGED_CODE (code-like tokens, no fence, no [CODE] tag)")

    return issues


# ---------------------------------------------------------------------------
# Main inspection routine
# ---------------------------------------------------------------------------

async def inspect(
    db_path: str,
    limit: int,
    source_contains: str | None,
    dump_full: bool,
) -> None:
    db_file = Path(db_path).resolve()
    print(f"Connecting to embedded SurrealKV store at: {db_file}")
    if not db_file.exists() and not db_file.with_suffix("").exists():
        print(f"WARNING: path does not appear to exist on disk. Double-check --db-path.")

    # Windows paths (e.g. H:\KathanaKnowledgeBase\...) break the surrealkv://
    # URL parser: the drive-letter colon gets misread as a host:port separator
    # ("H" as host, everything after as port), raising a ValueError. Forward
    # slashes avoid that ambiguity. This mirrors why your .env.example default
    # (a relative, forward-slash path) has always worked without hitting this.
    db_url_path = str(db_file).replace("\\", "/")
    db = AsyncSurreal(f"surrealkv://{db_url_path}")
    await db.use("default", "default")  # namespace, database — adjust if yours differ

    try:
        # 1. Overall counts, so we know the scale of what we're sampling
        count_rows = await db.query("SELECT count() AS cnt FROM vec_default GROUP ALL")
        total = count_rows[0]["cnt"] if count_rows else 0
        print(f"\nTotal chunks in vec_default: {total}\n")
        if total == 0:
            print("No chunks found. Is the DB path correct, or has ingestion not run yet?")
            return

        # 2. Pull a sample. Filter by source substring if given (metadata.source
        #    or content_type — whichever survived; we check both since one of
        #    them may be the very thing that's missing).
        if source_contains:
            rows = await db.query(
                f"SELECT record::id(id) AS id, content, metadata "
                f"FROM vec_default "
                f"WHERE string::contains(metadata.source, $s) "
                f"   OR string::contains(content, $s) "
                f"LIMIT {limit}",
                {"s": source_contains},
            )
        else:
            rows = await db.query(
                f"SELECT record::id(id) AS id, content, metadata "
                f"FROM vec_default LIMIT {limit}"
            )

        if not rows:
            print("No matching rows. Try without --source-contains, or check the filter string.")
            return

        for i, row in enumerate(rows, 1):
            content: str = row.get("content", "") or ""
            metadata: dict[str, Any] | None = row.get("metadata")

            print("=" * 100)
            print(f"CHUNK {i} / {len(rows)}   —   id: {row.get('id')}")
            print("-" * 100)

            # --- Metadata check -------------------------------------------------
            print("METADATA:")
            if metadata is None or metadata == {}:
                print("  ⚠️  EMPTY/MISSING — this matches the documented ainsert(file_paths=...) "
                      "deviation. content_type/has_code did not survive to storage.")
            else:
                for k, v in metadata.items():
                    print(f"  {k}: {v}")
                missing = [f for f in ("content_type", "has_code", "source") if f not in metadata]
                if missing:
                    print(f"  ⚠️  Present but missing expected fields: {missing}")

            # --- Text check ------------------------------------------------------
            print("\nCONTENT:")
            shown = content if dump_full else (content[:600] + (" …[truncated]" if len(content) > 600 else ""))
            print(shown)

            # --- Heuristic flags ---------------------------------------------
            issues = flag_issues(content)
            print("\nFLAGS:")
            if issues:
                for issue in issues:
                    print(f"  🚩 {issue}")
            else:
                print("  (none of the cheap heuristics tripped — doesn't mean it's clean, just no red flags)")

            print()

    finally:
        await db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect stored chunks in the embedded SurrealKV store.")
    parser.add_argument(
        "--db-path",
        default="./lightrag_data/graphrag.db",
        help="Path to the embedded SurrealKV database file (default: ./lightrag_data/graphrag.db)",
    )
    parser.add_argument("--limit", type=int, default=8, help="Number of chunks to sample (default: 8)")
    parser.add_argument(
        "--source-contains",
        default=None,
        help="Substring to filter by (matches metadata.source or content) — e.g. a book filename fragment",
    )
    parser.add_argument(
        "--dump-full",
        action="store_true",
        help="Print full chunk text instead of truncating to 600 chars",
    )
    args = parser.parse_args()

    asyncio.run(inspect(args.db_path, args.limit, args.source_contains, args.dump_full))


if __name__ == "__main__":
    main()
