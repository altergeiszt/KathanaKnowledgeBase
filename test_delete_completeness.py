"""
test_delete_completeness.py — the §7 gate from Migration_LlamaIndex.md.

Proves (or disproves) that editing a note — delete old ref_doc_id, re-insert new
version — actually purges the OLD content from the Neo4j property graph, not just
from LlamaIndex's docstore. This is the documented LlamaIndex failure mode: deleting
a doc can clear the docstore while leaving graph nodes/relations orphaned, so stale
content keeps answering queries.

Do not build the note ingester (§6) or the spaced-repetition update loop until this
passes. If it fails, see the two fallbacks documented in §7 of the migration doc:
  1. Manual purge by ref_doc_id (enumerate + delete explicitly)
  2. Full note-namespace rebuild each update cycle (cheap since notes have no
     LLM extraction cost)

USAGE
-----
    pip install llama-index-core llama-index-graph-stores-neo4j \
                llama-index-llms-ollama llama-index-embeddings-huggingface
    # Neo4j Desktop running, 'llamaindex' database active, bolt://localhost:7687
    python test_delete_completeness.py

Uses a real EntityNode + Relation (per §6 — notes get structure read directly from
wikilinks, no LLM extraction), matching how the note ingester will actually write
to the graph. Also runs a TextNode through the vector side, since a note update
needs to clear BOTH the graph entity and the embedded text node.

If any API below doesn't match your installed version, the traceback will name the
exact missing attribute/method — fix that call site and re-run. Everything here is
built against the documented LlamaIndex property-graph API shape but not verified
against your exact installed version (no network access to the actual package from
where this was written) — treat method names as "likely correct, confirm on first
run," not gospel.
"""

from __future__ import annotations

import sys

from llama_index.core import PropertyGraphIndex
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.schema import TextNode, RelatedNodeInfo, NodeRelationship
from llama_index.core.embeddings import BaseEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import ImplicitPathExtractor

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    _EMBED = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
except Exception as exc:
    print(f"Could not load HuggingFaceEmbedding ({exc}); falling back to a mock "
          f"embedder so this harness can still test GRAPH delete-completeness "
          f"even if the real embedding model isn't installed yet.")
    import random

    class _MockEmbedding(BaseEmbedding):
        def _get_query_embedding(self, query: str) -> list[float]:
            random.seed(hash(query) % (2**32))
            return [random.random() for _ in range(384)]
        def _get_text_embedding(self, text: str) -> list[float]:
            return self._get_query_embedding(text)
        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)
        async def _aget_text_embedding(self, text: str) -> list[float]:
            return self._get_text_embedding(text)

    _EMBED = _MockEmbedding()


NEO4J_URL = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "v1rthcr4ft"   # <-- set to your actual local password
NEO4J_DATABASE = "llamaindex"

NOTE_ID = "note::concept-x"   # stable ref_doc_id, mirrors the content-hash scheme


def make_store() -> Neo4jPropertyGraphStore:
    return Neo4jPropertyGraphStore(
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        url=NEO4J_URL,
        database=NEO4J_DATABASE,
    )


def ingest_note_version(index: PropertyGraphIndex, ref_doc_id: str,
                        concept: str, body_text: str, wikilink_target: str) -> None:
    """
    Mirrors §6's planned note ingester: title -> EntityNode, wikilink -> Relation,
    no LLM call. Also inserts a TextNode with the body so the note is vector-
    retrievable, matching the real design (graph structure + embedded text
    together per note).
    """
    entity = EntityNode(
        label="Concept",
        name=concept,
        properties={"source_type": "note", "ref_doc_id": ref_doc_id},
    )
    target = EntityNode(
        label="Concept",
        name=wikilink_target,
        properties={"source_type": "note"},
    )
    relation = Relation(
        label="RELATES_TO",
        source_id=entity.id,
        target_id=target.id,
        properties={"ref_doc_id": ref_doc_id},
    )
    index.property_graph_store.upsert_nodes([entity, target])
    index.property_graph_store.upsert_relations([relation])

    text_node = TextNode(
        text=body_text,
        id_=f"{ref_doc_id}::body",
        metadata={"source_type": "note", "ref_doc_id": ref_doc_id},
    )
    # ref_doc_id is a read-only property (derived from the SOURCE relationship)
    # in llama-index 0.14.x — it has no setter. Set the SOURCE relation instead;
    # this is also what delete_ref_doc() keys off to find/remove this node.
    text_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=ref_doc_id)
    index.insert_nodes([text_node])


def graph_mentions(store: Neo4jPropertyGraphStore, term: str) -> list[str]:
    """Direct Cypher check against the graph store — NOT via the docstore/query
    engine — because the docstore can look clean while the graph store is not.
    This is the check that actually answers the §7 question."""
    # Only test STRING-typed properties: inserted Chunk nodes carry an `embedding`
    # FloatArray, and Neo4j's toString() throws CypherTypeError on an array — the
    # original `toString(n[k])` over every key blew up as soon as real embeddings
    # were present. apoc.meta.cypher.type() lets us skip non-string props entirely,
    # and we drop `embedding` from the returned props so the output isn't flooded
    # with a 384-dim vector. (APOC confirmed enabled on this instance.)
    query = """
    MATCH (n)
    WHERE any(k IN keys(n) WHERE apoc.meta.cypher.type(n[k]) = 'STRING'
                                 AND n[k] CONTAINS $term)
    RETURN labels(n) AS labels, apoc.map.removeKey(properties(n), 'embedding') AS props
    """
    results = store.structured_query(query, param_map={"term": term})
    return [f"{r['labels']} {r['props']}" for r in results]


def vector_node_exists(index: PropertyGraphIndex, node_id: str) -> bool:
    try:
        index.docstore.get_node(node_id)
        return True
    except Exception:
        return False


def main() -> None:
    print("Connecting to Neo4j...")
    store = make_store()

    # kg_extractors MUST be a non-empty list of LLM-free extractors. If omitted (or
    # passed as []), from_existing() falls back to SimpleLLMPathExtractor(Settings.llm),
    # which resolves to OpenAI and errors out — and we want NO LLM extraction here
    # anyway (notes get their structure directly from wikilinks, per §6). The `or`
    # in base.py:124 treats [] as falsy, so we pass a real extractor: ImplicitPathExtractor
    # reads relationships already present on inserted nodes, no model call.
    index = PropertyGraphIndex.from_existing(
        property_graph_store=store,
        embed_model=_EMBED,
        kg_extractors=[ImplicitPathExtractor()],
    )

    # --- v1: note references ALPHA ---
    print(f"\n[1] Ingesting note v1 (references ALPHA)...")
    ingest_note_version(
        index, NOTE_ID,
        concept="Concept X",
        body_text="Concept X is fundamentally about ALPHA and its properties.",
        wikilink_target="ALPHA",
    )

    alpha_hits_v1 = graph_mentions(store, "ALPHA")
    print(f"    ALPHA present in graph after v1 insert: {len(alpha_hits_v1)} node(s)")
    assert alpha_hits_v1, "FAIL: v1 insert didn't even write ALPHA — ingestion itself is broken, stop here"

    # --- edit: delete v1, insert v2 referencing BETA instead ---
    print(f"\n[2] Deleting note (ref_doc_id={NOTE_ID}) via delete_ref_doc...")
    index.delete_ref_doc(NOTE_ID, delete_from_docstore=True)

    print(f"[3] Ingesting note v2 (references BETA instead)...")
    ingest_note_version(
        index, NOTE_ID,
        concept="Concept X",
        body_text="Concept X is fundamentally about BETA and its properties.",
        wikilink_target="BETA",
    )

    # --- THE ACTUAL CHECK: is ALPHA really gone from the GRAPH STORE? ---
    print(f"\n[4] Checking graph store directly for leftover ALPHA nodes...")
    alpha_hits_v2 = graph_mentions(store, "ALPHA")
    beta_hits_v2 = graph_mentions(store, "BETA")

    print(f"    ALPHA nodes remaining: {len(alpha_hits_v2)}")
    for h in alpha_hits_v2:
        print(f"      leftover: {h}")
    print(f"    BETA nodes present:    {len(beta_hits_v2)}")

    print("\n" + "=" * 68)
    if alpha_hits_v2:
        print("RESULT: FAIL — stale ALPHA content survived delete_ref_doc().")
        print("The graph store has orphaned nodes/relations. Do NOT build the")
        print("note-update loop on this yet. See §7 fallbacks:")
        print("  1. Manual purge by ref_doc_id (enumerate + delete explicitly)")
        print("  2. Full note-namespace rebuild each update cycle")
        sys.exit(1)
    elif not beta_hits_v2:
        print("RESULT: INCONCLUSIVE — v2 (BETA) never got written either.")
        print("Something is wrong with ingest_note_version() or the connection,")
        print("not necessarily with delete completeness. Check the traceback /")
        print("Neo4j Browser (`MATCH (n) RETURN n LIMIT 25`) before concluding")
        print("anything about delete behavior.")
        sys.exit(2)
    else:
        print("RESULT: PASS — old content fully purged, new content present.")
        print("delete_ref_doc() is trustworthy for the note-update loop.")
        print("Safe to proceed with §6 (note ingester) and the spaced-repetition")
        print("update design as originally planned.")
    print("=" * 68)


if __name__ == "__main__":
    main()
