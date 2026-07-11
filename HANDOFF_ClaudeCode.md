# KathanaKnowledgeBase — handoff context for Claude Code

Personal GraphRAG knowledge base. Mid-migration from LightRAG+SurrealDB to
LlamaIndex+Neo4j. This doc is the "why" behind the code — read alongside
`Migration_LlamaIndex.md` (full design reference, in-repo) and
`test_delete_completeness.py` (the immediate next thing to run).

## Where this stands right now

Just finished writing `test_delete_completeness.py` — **not yet run**. This is the
current blocking task. Everything else is designed but not implemented pending its
result.

## Stack

- Local, self-hosted. Windows, RTX 4080 Super (16GB), Ollama (`qwen2.5:14b` for
  extraction), `all-MiniLM-L6-v2` (sentence-transformers) for embeddings.
- Orchestration: **LlamaIndex** (`llama-index-core` 0.14.x). Confirmed choice — see
  "Decisions already made and reversed" below, this one flip-flopped twice.
- Graph store: **Neo4j Community Edition**, local via Neo4j Desktop. Instance
  `RAG-Pipeline-Base`, database `llamaindex`, `bolt://localhost:7687`. APOC plugin
  enabled; GenAI plugin deliberately NOT installed (embeddings are computed locally
  in Python before writing to Neo4j — GenAI plugin solves a different architecture
  where Neo4j itself calls out to a cloud embedding API, not applicable here).
  Vector index support confirmed working (`db.index.vector.*` procedures present,
  test index created at real 384-dim shape and dropped cleanly).
- Migrating FROM: LightRAG + a hand-written SurrealDB/SurrealKV adapter. That whole
  stack is being deleted, not preserved — see rationale below.

## The corpus and the hybrid architecture (this is the load-bearing decision)

~69-71 books, ~1.3GB. A 7-file/6MB test slice (`chunks_checkpoint.json`, in repo)
produced 4648 chunks; extrapolated full-corpus is ~80k-300k+ chunks depending on
what "1.3GB" measures (source bytes vs. extracted text).

**Measured, not estimated:** ran `extraction_benchmark.py` against real chunks on
the actual local Ollama setup. Median LLM entity-extraction time is **3.7-3.9s per
chunk**, regardless of open-ended vs. schema-constrained prompting (schema was
*6% slower*, not faster — a real finding, don't assume schema-constrained extraction
is cheaper without re-testing if the model changes). Gleaning (2nd LLM pass) costs
**+75% time for marginal entity gain** — decided against; single-pass extraction only.

At ~3.9s/chunk serial, full-corpus extraction is **3.6-13.5+ days**. This is why the
architecture is NOT "run LLM extraction over everything." It's:

- **Embed every chunk** (cheap, fast — minutes for the whole corpus on local GPU).
- **LLM-extract entities/relations only for**: (a) all personal notes (via wikilinks,
  see below — no LLM call needed for these anyway), and (b) a hand-picked curated
  slice of 18 books (`curated-slice.md`, in repo) — one or two per corpus "Phase"
  folder, chosen by the user, not automatically.
- Everything else in the corpus is vector-searchable but has no graph structure.

This hybrid split is why LlamaIndex (not a KG-pipeline-owns-everything framework)
was the right call — see below.

## Two note-worthy corpus findings (already fixed in code)

1. **`content_type` classification must be chunk-level, not book-level or
   folder-level.** The user's own curated-slice labels prove single books
   legitimately carry multiple types (e.g. "Stats and Calculus Workshop with Python"
   → {MATH, PYTHON}). `classify.py` (in repo) implements this as
   `frozenset[ContentType]` per chunk, two-tier:
   - Tier 1: `BOOK_LABELS` dict, exact match against the 18 hand-labeled curated-slice
     titles. This dict ALSO defines curated-slice membership (`is_curated()`) — a
     book being in `BOOK_LABELS` means both "use this content_type" AND "run full
     kg_extractors on this book." Keep these coupled; don't split into two lists that
     can drift.
   - Tier 2: regex/keyword density scoring, no LLM (cheap heuristic is fine here
     specifically because Tier 2 books are embeddings-only — a wrong tag costs a
     mistagged metadata filter, not a corrupted expensive extraction).
   - Validated against real checkpoint data: ~0.5-1.8% cross-contamination rate
     against known-domain test books, down from a much higher rate on the first
     (too-permissive) threshold. Good enough for Tier 2; don't over-tune further.

2. **Docling wraps equation/matrix blocks in triple-backtick fences during PDF→
   markdown conversion**, same as code blocks. This broke the classifier twice
   (bare fence, then brace-inside-fence both false-positived on math notation) before
   landing on requiring genuine code-shaped constructs (function-call syntax,
   statement-terminating semicolons) inside a fence, not just any fence or any brace.
   **This same blind spot likely exists in the OLD `ingest.py`'s `has_code` metadata
   and `chunk_software()` routing** (LightRAG-era code, not yet checked/fixed) — if
   any of that logic survives into the new pipeline, re-audit it against this finding
   before trusting `has_code` on math-heavy books.

## Personal notes (GraphNotes project — separate app, read-only dependency)

Notes are Foam-style markdown: title = concept, YAML frontmatter `tags:`, body has a
"How does X relate to other ideas?" section using `[[wikilink]]` syntax as *authored*
relationships. Key design decision: **notes do NOT go through LLM extraction at all**
— their structure is already explicit (title → EntityNode, wikilinks → Relations,
tags → node properties), parsed directly, no Ollama call. Running an LLM extractor
over already-structured authored content would be strictly worse than reading it
directly, and it's free compared to the book pipeline's LLM cost.

GraphNotes' relationship to this RAG is read-only / interface-level — NOT shared
storage. (Early design had considered a shared embedded SurrealDB specifically so
GraphNotes and the RAG could share a backend; that requirement dissolved once it was
clarified GraphNotes only reads notes into the RAG and reads results back out — no
process needs to write into the other's store. This is *why* SurrealDB stopped being
a constraint and Neo4j became viable.)

Notes update on a **spaced-repetition interval** — occasional, not high-frequency,
but genuinely recurring edits over each note's life. This is precisely why the
delete-completeness question (next section) matters: every note edit is a
delete-old-then-reinsert cycle, not an append.

## The immediate blocking task: delete-completeness

There's a documented LlamaIndex issue where deleting a doc clears the docstore but
can leave the property graph store with orphaned nodes/relations — stale content
keeps answering queries. Given notes update repeatedly via spaced repetition, this
must be proven true or false BEFORE building the note ingester or any update loop.

`test_delete_completeness.py` (in repo, just written, not yet executed):
- Ingests a note "Concept X" wikilinking to `ALPHA` (as EntityNode/Relation, mirroring
  the real planned note-ingestion shape, no LLM).
- Calls `delete_ref_doc(ref_doc_id, delete_from_docstore=True)`.
- Re-ingests the same `ref_doc_id` as a v2 wikilinking to `BETA` instead.
- **The actual check**: raw Cypher `MATCH` query directly against the Neo4j store
  (bypassing LlamaIndex's own query APIs entirely) searching for any surviving
  mention of `ALPHA`. This bypass is deliberate — querying through LlamaIndex's own
  layer could paper over exactly the docstore-vs-graph-store divergence this test
  exists to catch.
- Three possible outcomes are handled explicitly in the script's output: PASS (old
  content purged, safe to proceed), FAIL (orphans found — do not build the note
  ingester, use one of two documented fallbacks), INCONCLUSIVE (neither version
  wrote correctly — likely an API mismatch or connection issue, not a real answer
  about delete behavior; check `MATCH (n) RETURN n LIMIT 25` in Neo4j Browser first).

**Known unverified surface**: the script was written without network access to
actually import/inspect the installed `llama-index-graph-stores-neo4j` (0.7.0) or
`llama-index-core` (0.14.23) packages. Class/method names (`EntityNode`, `Relation`,
`delete_ref_doc`, `structured_query`, `PropertyGraphIndex.from_existing`) are built
against the documented API shape but may not match exactly — expect to fix a few
call sites against real `AttributeError`/`TypeError` tracebacks on first run. This is
expected friction, not a sign the design is wrong.

**Fallbacks if it fails** (documented, don't need to be rediscovered):
1. Manual purge — enumerate graph-store nodes by `ref_doc_id` and delete explicitly,
   since the bug is specifically that the automatic path misses some.
2. Full note-namespace rebuild on each update cycle — since notes have no LLM
   extraction cost, rebuilding the entire note subgraph per spaced-repetition cycle
   is cheap; the (expensive) book graph is untouched either way.

## Decisions already made and reversed (context for why some things look indecisive)

- Storage: SurrealDB (shared with GraphNotes) → dropped once GraphNotes' dependency
  was clarified as read-only, not shared-storage. Neo4j chosen instead: actively
  maintained, well-documented, has a browser UI for visually exploring note↔book
  relationships (the actual point of the project). Community Edition's
  single-active-database limit is a non-issue — notes/books coexist in one database
  via `source_type` node labels + namespaced `ref_doc_id`s, not two physical DBs.
- Graph store engine: Kuzu was seriously considered (embedded, no server, fast) but
  ruled out — KuzuDB was archived on GitHub Oct 2025 after Apple's acquisition,
  development stopped. Neo4j chosen for active maintenance over Kuzu's raw
  embedded-performance edge.
- **Framework: flip-flopped once, now settled.** LlamaIndex-orchestrates-custom-
  front-half was the original plan → briefly changed to "neo4j-graphrag's
  SimpleKGPipeline owns ingestion end-to-end" (chosen for path-of-least-resistance to
  a first MVP) → reverted back to LlamaIndex after realizing that decision had been
  conflated with a DIFFERENT axis (hybrid-vs-full-extraction). neo4j-graphrag wants
  to own parsing→chunking→extraction→writing as one motion, which is incompatible
  with preserving the custom docling/dedup front-half AND makes the hybrid
  embed-everything/extract-a-slice split awkward to express. LlamaIndex's node-based
  model (`TextNode`s in, `kg_extractors` as a pluggable list) makes hybrid a natural
  per-node choice instead. **Current state: LlamaIndex framework, Neo4j store,
  neo4j-graphrag not used** (though individual neo4j-graphrag components like its
  entity-resolution or Text2Cypher retriever remain reasonable to bolt on later
  against the same Neo4j instance, since nothing prevents mixing).
- Cloud GPU (AWS): scoped as "just in case," not committed to. If the local hybrid
  build (curated slice only, NOT full corpus) turns out to be too slow even after
  scoping down to ~18 books, the plan is: rent GPU throughput (P3 or P4d, NOT P2 —
  P2's K80 GPUs are slower than the local 4080 Super, a real trap in initial
  research), use Spot pricing (job is checkpointable/resumable, fits Spot's
  interruption model), and treat the cloud instance as a stateless extraction worker
  that emits CSV/Parquet artifacts — NEVER stand up a networked Neo4j in the cloud.
  Pull the (small) extracted-graph artifacts back and bulk-load locally via
  `neo4j-admin database import`. Gotcha flagged: pin the embedding model version
  identically on both sides if this is ever used, or the vector index becomes
  incoherent.

## What Claude Code should actually do first

1. Run `test_delete_completeness.py` against the live local Neo4j (`llamaindex` DB).
   Fix any API mismatches that surface (expected, see above) rather than treating
   them as a sign to redesign.
2. Read the PASS/FAIL/INCONCLUSIVE output and act accordingly — do not proceed to
   step 3 on a FAIL or INCONCLUSIVE result without first resolving it (see
   fallbacks above).
3. Only after a genuine PASS: begin `Migration_LlamaIndex.md`'s §13 order-of-work —
   Stage 5 swap in `ingest.py` (LightRAG→LlamaIndex+Neo4jPropertyGraphStore,
   wiring `classify.py`'s `is_curated()` to route the 18-book curated slice through
   `kg_extractors` and everything else to embeddings-only), then the note ingester
   (§6), then retrieval.

Full design detail, code sketches, and the complete reasoning chain for every
decision above live in `Migration_LlamaIndex.md` — treat that as the source of
truth for implementation specifics; this doc is oriented context, not a spec.
