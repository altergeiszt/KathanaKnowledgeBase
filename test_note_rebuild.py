"""
test_note_rebuild.py — validates §7 Fallback #2 (note-namespace rebuild).

Context: test_delete_completeness.py proved that `index.delete_ref_doc()` alone
leaves directly-upserted note entities/relations orphaned in the Neo4j property
graph (RESULT: FAIL — delete_ref_doc is blind to anything written via
`property_graph_store.upsert_nodes()/upsert_relations()`, which is exactly how the
§6 note ingester writes structure). This harness proves the documented Fallback #2
fixes it:

    To update a note, wipe the entire NOTE namespace from the graph store and
    re-ingest the notes. Cheap — notes have no LLM extraction cost (§6).

Deliberately reuses the SAME `ingest_note_version()` shape that failed under
delete_ref_doc (imported from test_delete_completeness), so a PASS here isolates
the fix to the rebuild *strategy* — not to any change in how notes are written.

It also inserts a BOOK-namespaced control node that mentions ALPHA and asserts the
node SURVIVES the rebuild — proving the purge is scoped to `source_type='note'` /
the `note::` ref_doc_id namespace (§3) and never touches the expensive book graph.

USAGE
-----
    # Neo4j Desktop running, 'llamaindex' database active, neo4j://127.0.0.1:7687
    python test_note_rebuild.py
"""

from __future__ import annotations

import sys

from llama_index.core import PropertyGraphIndex
from llama_index.core.schema import TextNode, RelatedNodeInfo, NodeRelationship
from llama_index.core.indices.property_graph import ImplicitPathExtractor

# Reuse the connection, embedder, and the exact note-writing shape from the
# delete-completeness gate. Importing this module loads the HuggingFace embedder
# (module-level) but does NOT run its main() — that's guarded by __main__.
from test_delete_completeness import make_store, ingest_note_version, _EMBED, NOTE_ID

BOOK_REF_DOC = "book::sample"
BOOK_CHUNK_ID = "book::sample::ch1"


def build_index(store) -> PropertyGraphIndex:
    # kg_extractors must be a non-empty LLM-free list (see test_delete_completeness
    # for why []/None falls back to an OpenAI-backed SimpleLLMPathExtractor).
    return PropertyGraphIndex.from_existing(
        property_graph_store=store,
        embed_model=_EMBED,
        kg_extractors=[ImplicitPathExtractor()],
    )


def insert_book_control(index: PropertyGraphIndex) -> None:
    """A book-namespaced node mentioning ALPHA. It must SURVIVE a note rebuild —
    this is the guard proving the rebuild is scoped to notes, not a global wipe."""
    node = TextNode(
        text="This textbook chapter explains ALPHA thoroughly.",
        id_=BOOK_CHUNK_ID,
        metadata={"source_type": "book", "ref_doc_id": BOOK_REF_DOC},
    )
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=BOOK_REF_DOC)
    index.insert_nodes([node])


# ---------------------------------------------------------------------------
# Fallback #2: note-namespace rebuild
# ---------------------------------------------------------------------------

# Scoped strictly to the note namespace (§3): source_type='note', OR a node whose
# id / ref_doc_id lives in the 'note::' namespace (catches the bare doc/source
# node LlamaIndex creates for a SOURCE relationship, which has no source_type
# property of its own). Book nodes use a different namespace, so they're untouched.
_NOTE_NAMESPACE_PURGE = """
MATCH (n)
WHERE n.source_type = 'note'
   OR n.id STARTS WITH 'note::'
   OR n.ref_doc_id STARTS WITH 'note::'
DETACH DELETE n
"""


def rebuild_note_namespace(store, index: PropertyGraphIndex, notes: list[dict]) -> None:
    """Purge the whole note namespace from the graph store, then re-ingest `notes`.

    This is the operation delete_ref_doc could not do: it reaches directly-upserted
    entities/relations (the RELATES_TO edges and target Concept nodes) that
    delete_ref_doc leaves orphaned. Re-ingesting all notes rebuilds every needed
    concept/relation from scratch, so shared wikilink targets are recreated by
    whichever surviving notes still reference them — correct by construction.
    """
    store.structured_query(_NOTE_NAMESPACE_PURGE)
    for note in notes:
        ingest_note_version(index, **note)


# ---------------------------------------------------------------------------
# Verification helpers — direct Cypher, source-aware (§7 stale check)
# ---------------------------------------------------------------------------

def note_source_mentions(store, term: str) -> list[str]:
    """NOTE-sourced graph nodes mentioning `term`. This is the §7 stale check:
    stale = nodes where term appears AND node_source == 'note'. Book nodes are
    intentionally excluded — the rebuild is only responsible for note content."""
    rows = store.structured_query(
        """
        MATCH (n)
        WHERE (n.source_type = 'note'
               OR n.id STARTS WITH 'note::'
               OR n.ref_doc_id STARTS WITH 'note::')
          AND any(k IN keys(n) WHERE apoc.meta.cypher.type(n[k]) = 'STRING'
                                     AND n[k] CONTAINS $term)
        RETURN labels(n) AS labels, apoc.map.removeKey(properties(n), 'embedding') AS props
        """,
        param_map={"term": term},
    )
    return [f"{r['labels']} {r['props']}" for r in rows]


def book_source_mentions(store, term: str) -> list[str]:
    """BOOK-sourced graph nodes mentioning `term` — used to assert the rebuild
    left the book graph untouched."""
    rows = store.structured_query(
        """
        MATCH (n)
        WHERE (n.source_type = 'book'
               OR n.id STARTS WITH 'book::'
               OR n.ref_doc_id STARTS WITH 'book::')
          AND any(k IN keys(n) WHERE apoc.meta.cypher.type(n[k]) = 'STRING'
                                     AND n[k] CONTAINS $term)
        RETURN labels(n) AS labels, apoc.map.removeKey(properties(n), 'embedding') AS props
        """,
        param_map={"term": term},
    )
    return [f"{r['labels']} {r['props']}" for r in rows]


def clean_test_namespaces(store) -> None:
    """Reset this harness's own test data (note:: namespace + the book control) so
    the run starts clean and is repeatable. Scoped — leaves all other data alone."""
    store.structured_query(
        """
        MATCH (n)
        WHERE n.source_type = 'note'
           OR n.id STARTS WITH 'note::'
           OR n.ref_doc_id STARTS WITH 'note::'
           OR n.id STARTS WITH 'book::sample'
           OR n.ref_doc_id STARTS WITH 'book::sample'
        DETACH DELETE n
        """
    )


def main() -> None:
    print("Connecting to Neo4j...")
    store = make_store()
    index = build_index(store)

    print("[0] Cleaning test namespaces for a repeatable run...")
    clean_test_namespaces(store)

    # --- book control: a book node that mentions ALPHA, must survive everything ---
    print("[1] Inserting BOOK control node (mentions ALPHA, source_type=book)...")
    insert_book_control(index)
    book_alpha_before = book_source_mentions(store, "ALPHA")
    assert book_alpha_before, "setup broken: book control node didn't write"
    print(f"    book-sourced ALPHA present: {len(book_alpha_before)} node(s)")

    # --- note v1: references ALPHA ---
    print("\n[2] Ingesting note v1 (references ALPHA)...")
    ingest_note_version(
        index, NOTE_ID,
        concept="Concept X",
        body_text="Concept X is fundamentally about ALPHA and its properties.",
        wikilink_target="ALPHA",
    )
    note_alpha_v1 = note_source_mentions(store, "ALPHA")
    print(f"    note-sourced ALPHA present after v1: {len(note_alpha_v1)} node(s)")
    assert note_alpha_v1, "FAIL: v1 didn't write note ALPHA — ingestion broken, stop"

    # --- edit via Fallback #2: rebuild the note namespace with v2 (BETA) ---
    print("\n[3] Editing note via NOTE-NAMESPACE REBUILD (v2 references BETA)...")
    rebuild_note_namespace(store, index, [dict(
        ref_doc_id=NOTE_ID,
        concept="Concept X",
        body_text="Concept X is fundamentally about BETA and its properties.",
        wikilink_target="BETA",
    )])

    # --- THE CHECKS ---
    print("\n[4] Verifying graph store directly...")
    note_alpha_v2 = note_source_mentions(store, "ALPHA")
    note_beta_v2 = note_source_mentions(store, "BETA")
    book_alpha_v2 = book_source_mentions(store, "ALPHA")

    print(f"    stale note-sourced ALPHA remaining: {len(note_alpha_v2)}")
    for h in note_alpha_v2:
        print(f"      leftover: {h}")
    print(f"    note-sourced BETA present:          {len(note_beta_v2)}")
    print(f"    book-sourced ALPHA (control):       {len(book_alpha_v2)}  (must be >=1)")

    print("\n" + "=" * 68)
    ok = (not note_alpha_v2) and bool(note_beta_v2) and bool(book_alpha_v2)
    if ok:
        print("RESULT: PASS — Fallback #2 works.")
        print("  - stale note ALPHA fully purged from the GRAPH STORE")
        print("  - new note BETA present")
        print("  - book graph untouched (book ALPHA survived the rebuild)")
        print("Note updates via namespace rebuild are trustworthy. Safe to build the")
        print("note ingester (§6) + spaced-repetition loop on this strategy.")
        print("=" * 68)
        return
    print("RESULT: FAIL — rebuild did not produce a clean note namespace.")
    if note_alpha_v2:
        print("  - stale ALPHA survived: the purge missed note-namespaced nodes.")
    if not note_beta_v2:
        print("  - BETA missing: re-ingestion didn't write v2.")
    if not book_alpha_v2:
        print("  - book control gone: the purge was TOO broad and hit the book graph!")
    print("=" * 68)
    sys.exit(1)


if __name__ == "__main__":
    main()
