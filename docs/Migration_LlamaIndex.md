# Migration: LightRAG → LlamaIndex PropertyGraphIndex

Working migration plan for moving the KathanaKnowledgeBase RAG off LightRAG + custom
SurrealDB onto LlamaIndex's `PropertyGraphIndex`. Written against the actual
`chunks_checkpoint.json` (4648 chunks, 7 source files) and two representative notes.

Status: design / not yet implemented. The delete-verify harness (§7) is the
load-bearing risk and should be validated before building on the rest.

---

## 0. Why move

The workarounds accumulated around LightRAG were mostly *storage* workarounds —
consequences of forcing LightRAG's pluggable storage onto embedded SurrealKV, which
was itself chosen only to share a backend with GraphNotes. That coupling turned out to
be unnecessary (GraphNotes' interaction is read-only / interface-level, not
shared-storage — see §1), which removes the storage constraint and, with it, most of
the reason the LightRAG integration was painful.

What deletes on migration:

- `surrealdb_impl.py` (custom adapter) and `patch_lightrag.py` (on-disk package patch)
- The `_query_lock` write serialization
- Windows `surrealkv://` path normalization
- The `initialize_pipeline_status()` wedge workaround
- The bounded-teardown / `insert_timeout` / `--debug-finalize` scaffolding — all of it
  existed to compensate for LightRAG's opaque `ainsert()` and undocumented daemon
  worker pool, which do not exist in this design.

What survives unchanged (the actual value):

- docling extraction, semantic chunking, MinHash-LSH dedup
- `chunks_checkpoint.json` as the ingestion seam
- content-hash stable-ID scheme (becomes `ref_doc_id`)

---

## 1. GraphNotes coupling — confirmed interface-level

Neither GraphNotes use case requires shared storage:

| Flow | Direction | Requirement |
|---|---|---|
| "Feed ebooks, query them" | GraphNotes reads RAG output | A query interface (function call / small API) |
| "Surface note↔book relationships" | RAG reads notes *from* GraphNotes; ingests them | Read-only access to notes + ingestion |

At no point do the two write into the same store. The RAG stores its unified graph in
its own backend; GraphNotes reads results back. This is why the framework choice is now
unconstrained by storage, and why an embedded no-server store is viable again (without
SurrealDB specifically).

---

## 2. Checkpoint findings that shape the design

Two things surfaced from the real `chunks_checkpoint.json` that change the plan.

### 2.1 `content_type` is unreliable in current data — fix classification

Observed distribution: **selfhelp 3441, software 1207, math 0.**

| Source file | Assigned | Should be | Why it misfired |
|---|---|---|---|
| c13andnet9…fundamentals.pdf | software | software | accidental: "de**velop**ment" contains "dev" |
| dotNET10-1 Fundamentals.pdf | selfhelp | software | no path fragment matched |
| A First Course in Linear Algebra.pdf | selfhelp | math | no path fragment matched |
| linalgebra.pdf | selfhelp | math | "linalgebra" ≠ "math" |
| personalfinance*.epub | selfhelp | selfhelp | "personal" → selfhelp (ok) |
| freelancetofreedom.epub | selfhelp | selfhelp | default (ok) |

`classify()` matches directory-name substrings, but `D:\IngestTest\` is flat, so almost
everything defaulted to `selfhelp`. Downstream consequence (independent of LlamaIndex):
math books skipped LaTeX handling and the dotNET book **skipped code extraction** — its
code is inline in prose chunks, un-separated. All 540 code-bearing chunks trace to the
one file that accidentally matched `software`.

**Action:** replace path-based `classify()` with content-based classification (inspect
text: fenced-code density → software, LaTeX/`\begin{}` density → math, else prose), OR
require typed subfolders. Do this *before* trusting `content_type` as node metadata.

### 2.2 Notes are already a graph — do not LLM-extract them

`Rust Data Types.md` is an atomic concept note with authored structure:

```markdown
---
tags: [rust, variables, data-types]
---
# Rust Data Types
## What is Rust Data Types?
- ... Since Rust is a [[statically typed language]], the type ...
## How does Rust Data Types relate to other ideas?
```

- Title **is** the concept → one `EntityNode` per note.
- `tags:` → node properties.
- `[[statically typed language]]` → an **authored edge**, no extraction needed.
- The "How does X relate to other ideas?" section is a relationship slot by design.

Running an LLM path-extractor over these would be lossy and strictly worse than reading
the wikilinks/tags directly. **Books and notes therefore need different ingestion**
(§5 vs §6).

Caveats from the sample files:
- `new-study-notes.md` is a **Foam template** (`${FOAM_TITLE_SAFE}` placeholders) —
  skip template files or the graph fills with garbage entities.
- Both notes end with a stray `ø` — strip export artifacts before ingest.

---

## 3. Target architecture — one store, source-namespaced, bridged at retrieval

The tested "middle ground": **one physical property-graph store, two source-namespaced
node sets, no hard-wired cross-edges at ingestion.**

- Every node tagged `source_type ∈ {note, book}`.
- Notes and books use **separate `ref_doc_id` namespaces**.
- Books → LLM entity extraction; notes → structure read directly (§6).
- Note↔book relationships emerge at **query time**, two ways:
  1. **Shared entity names** — a note's `[[statically typed language]]` and a Rust
     chapter mentioning the same concept resolve to the same entity.
  2. **Vector similarity** spanning both node sets.

Why this shape:

- **Testable.** Retrieve over notes-only / books-only / both by changing retriever
  config, and measure which surfaces relationships best — no schema change. Defers the
  unified-vs-separate decision to an empirical test.
- **Clean churn.** Re-ingesting an edited note touches only note-namespaced
  `ref_doc_id`s, never the expensive book graph (critical for spaced-repetition updates).

### Store choice — decided: Neo4j (see §11)

Originally scoped as embedded (`SimplePropertyGraphStore` / Kuzu) before the scale
finding in §10 and the maintenance discussion in §11 settled it on **Neo4j Community
Edition, local via Neo4j Desktop**, connected through `Neo4jPropertyGraphStore` over
Bolt. Trade accepted: a running local server instead of an embedded file — in
exchange for active maintenance, first-class LlamaIndex + ecosystem support, and the
Neo4j Browser for visually exploring note↔book relationships (the actual point of
this project). Community Edition's single-active-database limit is a non-issue here:
notes/books separation is via `source_type` labels and namespaced `ref_doc_id`s within
one database (§3), not two physical databases.

---

## 4. Node mapping: `Chunk` → LlamaIndex `TextNode`

Checkpoint entry shape (confirmed):

```json
{ "text": str, "source_path": str, "content_type": str,
  "metadata": {"has_code": bool}?, "code_blocks": [str] }
```

Mapping:

| Chunk field | LlamaIndex target | Notes |
|---|---|---|
| `text` | `TextNode.text` | as-is |
| `source_path` | `ref_doc_id` (via content-hash) | stable ID; keep the existing MD5-of-path scheme |
| `content_type` | `metadata["content_type"]` | **only after classification fix (§2.1)** |
| `metadata.has_code` | `metadata["has_code"]` | books only |
| — | `metadata["source_type"] = "book"` | namespacing (§3) |
| `code_blocks` | separate linked `TextNode`s | see below |

**`code_blocks` decision (knob to test):** keep each code block as its *own*
retrievable `TextNode`, linked to its prose parent, rather than flattening into
metadata. Rationale: a code query should hit code, a concept query should hit prose;
merging them dilutes both. Testable against the alternative.

Sketch:

```python
from llama_index.core.schema import TextNode

def chunk_to_nodes(chunk: dict) -> list[TextNode]:
    doc_id = content_hash_id(chunk["source_path"])   # existing scheme
    base = TextNode(
        text=chunk["text"],
        metadata={
            "source_type": "book",
            "source_path": chunk["source_path"],
            "content_type": chunk["content_type"],   # post-fix
            "has_code": bool(chunk.get("metadata", {}).get("has_code")),
        },
    )
    base.ref_doc_id = doc_id
    nodes = [base]
    for i, code in enumerate(chunk.get("code_blocks") or []):
        cn = TextNode(
            text=code,
            metadata={"source_type": "book", "source_path": chunk["source_path"],
                      "content_type": "code", "parent": base.node_id},
        )
        cn.ref_doc_id = doc_id
        nodes.append(cn)
    return nodes
```

---

## 5. Book ingestion — `run_pipeline` after migration

The front half is unchanged. Only Stage 5 changes.

**Before:** Stage 5 = `rag.ainsert(chunk.text, file_paths=...)` (opaque; embedding +
extraction + SurrealDB write bundled).

**After:** Stage 5 = build `TextNode`s from the checkpoint → run `kg_extractors` on book
nodes → insert into `PropertyGraphIndex`.

```python
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

llm   = Ollama(model="qwen2.5:14b", request_timeout=300.0)
embed = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")  # keep current model

graph_store = Neo4jPropertyGraphStore(
    username="neo4j", password="<local password>",
    url="bolt://localhost:7687",   # Neo4j Desktop, local
)

nodes = [n for c in load_checkpoint() for n in chunk_to_nodes(c)]

index = PropertyGraphIndex(
    nodes=nodes,
    llm=llm,
    embed_model=embed,
    kg_extractors=[SimpleLLMPathExtractor(llm=llm)],   # runs per chunk; you control it
    property_graph_store=graph_store,
    show_progress=True,
)
```

Note: unlike the embedded-store sketch this replaces, Neo4j persists server-side —
no `index.storage_context.persist(...)` step needed; writes land in the running
Neo4j instance directly via Bolt.

Notes vs LightRAG:

- Extraction is **inspectable and swappable** (`kg_extractors` list) instead of opaque.
- The per-insert timeout / runaway-chunk problem is gone: extraction runs in a
  controllable batch you own. If a chunk is pathological, cap it with LlamaIndex's own
  `request_timeout` on the LLM, not external `wait_for` cancellation.
- No daemon-worker pool, no teardown scaffolding.

---

## 6. Note ingestion — separate, structure-first, no LLM

A small ingester, not part of the book pipeline. Reads notes from GraphNotes
(read-only), one note = one `ref_doc_id`.

Steps per note:

1. **Skip templates** — reject files containing `${FOAM_...}` placeholders / empty
   section bodies.
2. **Strip artifacts** — trailing `ø`, etc.
3. **Frontmatter** `tags:` → node properties.
4. **Title** → the concept `EntityNode` (name = note title).
5. **Wikilinks** `[[target]]` → explicit `Relation` edges inserted directly into the
   property graph store — **no LLM call**.
6. Tag every node `source_type="note"`, namespaced `ref_doc_id`.

```python
import re, frontmatter  # python-frontmatter

WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")
TEMPLATE = re.compile(r"\$\{FOAM_")

def ingest_note(path, graph_store):
    post = frontmatter.load(path)
    body = post.content.rstrip("ø \n")
    if TEMPLATE.search(body):
        return  # skip template
    title = extract_title(body)                 # "# Rust Data Types" -> concept
    note_id = content_hash_id(str(path))
    concept = EntityNode(name=title, label="Concept",
                         properties={"source_type": "note",
                                     "tags": post.get("tags", [])})
    rels = []
    for target in WIKILINK.findall(body):       # authored edges
        rels.append(Relation(source_id=title, target_id=target.strip(),
                             label="RELATES_TO"))
    graph_store.upsert_nodes([concept])
    graph_store.upsert_relations(rels)
    # also index the body text as a TextNode (source_type=note, ref_doc_id=note_id)
    # so notes are vector-retrievable and share entities with books.
```

(API names above are indicative — confirm exact `EntityNode` / `Relation` /
`upsert_*` signatures against the installed LlamaIndex version.)

---

## 7. Delete-verify harness — BUILD AND PASS THIS FIRST

The load-bearing risk. Spaced-repetition means edited notes are re-ingested repeatedly,
each update = **delete-old-then-insert-new**. There is a documented LlamaIndex issue
where deleting a doc clears the *docstore* but leaves nodes in the *property-graph
store*, so stale note content keeps answering queries. This is the same
`adelete_by_doc_id` completeness question already flagged for the RAG-Anything eval.

**Do not build the note-update loop until this harness passes.**

```python
def test_delete_completeness(graph_store, index):
    note_v1 = make_note("Concept X", body="X is about ALPHA. [[ALPHA]]")
    ingest_note(note_v1, graph_store); persist(index)
    assert query_mentions(index, "ALPHA")            # v1 present

    # edit: ALPHA -> BETA
    note_v2 = make_note("Concept X", body="X is about BETA. [[BETA]]")
    index.delete_ref_doc(ref_doc_id_of(note_v2), delete_from_docstore=True)
    ingest_note(note_v2, graph_store); persist(index)

    # CRITICAL: verify the GRAPH STORE, not just the docstore
    remaining = graph_store.get(ref_doc_id=ref_doc_id_of(note_v2))
    stale = [n for n in all_nodes(graph_store)
             if "ALPHA" in node_text(n) and node_source(n) == "note"]
    assert not stale, f"stale ALPHA nodes survived delete: {stale}"
    assert query_mentions(index, "BETA")             # v2 present
    assert not query_mentions(index, "ALPHA")        # v1 fully gone
```

- If it **passes**: the whole design is sound; proceed with §5/§6.
- If it **fails**: the delete leaves graph-store orphans. Fallbacks, in order:
  1. Manual purge — enumerate graph-store nodes by `ref_doc_id`/`source_path` and
     delete explicitly (the docstore-vs-graph-store divergence means you may need to
     hit the graph store directly).
  2. Note-subgraph rebuild — since notes are cheap (no LLM extraction), rebuild the
     entire *note* namespace on each update cycle; the book graph is untouched.

---

## 8. GraphNotes query interface (later)

Once ingestion is stable, expose retrieval to GraphNotes as a read-only interface —
a thin function/API, mirroring the current `api.py` role. PropertyGraphIndex composes
retrievers you can pick per query:

- `LLMSynonymRetriever` — keyword/synonym node lookup
- `VectorContextRetriever` — vector similarity (spans notes + books → the cross-source
  bridge from §3)
- graph-traversal retrieval — follow authored/extracted edges

The notes-only / books-only / both A/B (§3) is a matter of filtering retrievers on
`source_type` — no reindex.

---

---

## 10. Measured extraction cost (benchmark, qwen2.5:14b, RTX 4080 Super)

Ran `extraction_benchmark.py` against a stratified 40-chunk sample of the real
checkpoint. Results (serial, concurrency=1):

| Condition | median s/chunk | notes |
|---|---|---|
| open-ended, single pass | 3.67s | baseline |
| schema-constrained, single pass | 3.88s | **6% slower**, not faster — schema prompt produced *more* output tokens (245 vs 235 median), not fewer. Kills the "schema extraction is cheaper" argument for neo4j-graphrag. |
| open-ended + gleaning (2nd pass) | 6.42s | **+75%** for +1 entity/+2 relations median. Not worth it at this scale — **use single-pass, no gleaning** as the default. |

Full-corpus build time at ~3.9s/chunk, serial, full (non-hybrid) extraction:

| Chunks | Time |
|---|---|
| 80k (conservative) | 3.6 days |
| 300k (mid) | 13.5 days |
| 1M (text-interpretation of "1.30 GB") | 44.9 days |

**This is why full-corpus extraction is out and the hybrid design (embed everything,
LLM-extract only notes + a curated book slice, §3) is load-bearing, not optional.**
A curated slice of a few thousand chunks turns this into hours, re-runnable overnight.

---

## 11. Decision: LlamaIndex as framework, Neo4j as store

**Correction from an earlier draft of this doc**, which briefly recorded
"neo4j-graphrag owns ingestion end-to-end" — that was a mixup between two separate
axes (which *framework* orchestrates ingestion, vs. which *engine* the hybrid split
avoids full-corpus extraction on) and has been reverted. The actual decision:

- **LlamaIndex** is the orchestration framework — `TextNode`s, `kg_extractors`,
  `PropertyGraphIndex` (§4, §5). This is what preserves the docling/dedup front-half
  and `chunks_checkpoint.json` seam (§0), and what makes the hybrid split (§3) a
  natural per-node choice rather than something fought against a pipeline that wants
  to own parsing-through-writing as one motion.
- **Neo4j** (Community Edition, local via Desktop) is the storage engine underneath —
  `Neo4jPropertyGraphStore` in place of the generic/embedded store referenced
  elsewhere in this doc. Chosen for active maintenance, documentation quality, and
  the Neo4j Browser visualization of note↔book relationships.
- `neo4j-graphrag` (the first-party Neo4j library) was evaluated and set aside for
  this project specifically because it wants to own parsing → chunking → extraction →
  writing itself, which would mean giving up the custom front-half. Its
  schema-constrained extraction was *not* faster on measured hardware either (§10),
  removing the one technical argument that might have offset that cost. It remains a
  reasonable individual-component option later (e.g. its entity-resolution or
  Text2Cypher retriever against the same Neo4j instance), just not as the ingestion
  owner.
- Single-pass extraction, gleaning **off** (§10), regardless of framework.
- Hybrid split (§3) stays as originally designed: embed every `TextNode`; run
  `kg_extractors` only on notes + a curated book slice.

Update §5's code sketch and §7's harness to target `Neo4jPropertyGraphStore`
specifically (Bolt URI/auth) rather than a generic property-graph store.

---

## 12. Just in case: cloud GPU for the extraction batch (not a database migration)

If even the curated hybrid slice is too slow locally, or extraction schema iteration
needs to be faster than same-day, cloud GPU is the lever — but rent **GPU throughput**,
not "compute-optimized" instances (those are CPU-heavy; useless for a GPU-bound
token-generation bottleneck).

### Instance choice

AWS P-series, but not the naive "oldest = cheapest" read:

| Family | GPU | Verdict |
|---|---|---|
| P2 | K80 (2014) | **Avoid.** Older/slower than the local 4080 Super for this workload — likely *worse* than running locally. Not a real option despite lowest sticker price. |
| P3 | V100 | Reasonable floor — genuine generational upgrade over local hardware, decent price especially on Spot. |
| **P4d** | A100 (40/80GB) | **Likely sweet spot.** Higher hourly on-demand rate than P3, but ~2.5x throughput — often cheaper *per chunk* despite higher $/hr. |
| P5 | H100 | Overkill for 14B-class batch entity extraction; built for frontier training. |

### Use Spot, not on-demand

Spot = up to 90% off on-demand, in exchange for possible reclamation. This fits the
job specifically: it's a non-time-sensitive batch, and the pipeline already
checkpoints (`chunks_checkpoint.json`, plus LlamaIndex's own per-node insert tracking) —
interruption just means resume from the last completed chunk, not restart. Spot is
built for exactly this shape of job.

**EC2 Capacity Blocks for ML** is a middle ground worth knowing about: reserve GPU
capacity for a fixed short window (a day or two) at a set rate — no interruption risk,
cheaper than full on-demand, good fit for "burn through a bounded batch once."

### What NOT to do: don't run the database in the cloud

The cloud box is a **stateless extraction worker**, not a database host. Avoid the
"how do I migrate data off a cloud database" problem entirely by never putting one
there:

1. Cloud GPU instance runs Ollama/vLLM + the extraction pipeline against the curated
   slice.
2. It writes results to **files** (CSV/Parquet: entities, relations, embeddings) —
   not into a live Neo4j.
3. Download the files (small — extracted graph is far smaller than source corpus;
   minutes, not a data-transfer project).
4. Bulk-load locally via `neo4j-admin database import` / batched `LOAD CSV` into the
   local Neo4j.

Gotcha: pin the exact embedding model + version on both sides. A vector index is only
coherent if every vector came from the same embedder — mixing a cloud-generated
embedding with a locally-generated query embedding from a different model/version
silently breaks retrieval. (Currently `all-MiniLM-L6-v2` — keep this pinned.)

**Try local-hybrid first.** If the curated slice finishes locally in a tolerable
window (e.g. under ~12h), cloud may not be needed at all for an MVP. Reach for cloud
when even the curated build is too slow locally, or extraction-schema iteration speed
matters enough that day-plus-per-run is intolerable.

---

## 13. Suggested order of work

1. **Delete-verify harness (§7)** — gate. Prove note updates purge cleanly against
   `Neo4jPropertyGraphStore` specifically (§11).
2. Content-based classification fix (§2.1) — so `content_type` is trustworthy.
3. Define the curated hybrid slice (§3, §11) — which books (or chapters) get full
   `kg_extractors` treatment vs. embeddings-only.
4. Book ingestion via `PropertyGraphIndex` + `kg_extractors` (§5, §11) over the
   curated slice, single-pass, no gleaning (§10) — delete the LightRAG/SurrealDB
   scaffolding once this is green.
5. Note ingester (§6) — structure-first, wikilinks as edges, feeding the same Neo4j.
6. Retrieval interface + notes/books A/B (§3, §8).
7. **If needed** — cloud GPU batch per §12, only after confirming the local curated
   build is genuinely too slow.

Open knobs to test empirically: code-blocks-as-separate-nodes vs metadata (§4);
unified vs namespaced retrieval (§3); size/composition of the curated slice (§3, §10);
`SimplePropertyGraphStore`-equivalent local persistence vs scale needs.
