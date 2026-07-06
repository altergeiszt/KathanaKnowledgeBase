"""
Example: how to wire the profiler into your actual GraphRAG ingestion pipeline.

This is a TEMPLATE — swap the placeholder function calls for your real
docling / semantic-text-splitter / sentence-transformers / LightRAG+Ollama /
SurrealDB calls. The point is just showing WHERE the `with profiler.stage(...)`
blocks go relative to your existing code.

Run for a small batch first (5-10 books spanning your three content tiers:
software/prose/self-help) to get a representative sample before committing to
a full 14GB run.
"""

from pathlib import Path
from pipeline_profiler import Profiler


def run_pipeline_for_book(profiler: Profiler, book_path: Path, content_tier: str):
    """
    content_tier: "software" (vector-only), "math" (prose-only),
                  or "self_help" (full GraphRAG w/ Qwen extraction)
    """

    # --- Stage 1: Parse ---------------------------------------------------
    with profiler.stage("parse_docling", book=book_path.name, tier=content_tier):
        # doc = docling_convert(book_path)
        pass  # <- replace with real call

    # --- Stage 2: Chunk -----------------------------------------------------
    with profiler.stage("chunk_semantic", book=book_path.name):
        # chunks = semantic_splitter.chunks(doc.text)
        pass  # <- replace with real call

    # --- Stage 3: Embed (always happens, all tiers) -------------------------
    with profiler.stage("embed_sentence_transformers", book=book_path.name):
        # vectors = embedder.encode(chunks, device="cuda")
        pass  # <- replace with real call

    # --- Stage 4: Entity extraction (ONLY for full-GraphRAG tier) ----------
    # This is almost certainly your heaviest stage. Isolating it like this
    # is the key thing you want data on.
    if content_tier == "self_help":
        with profiler.stage("extract_entities_qwen_ollama", book=book_path.name,
                             n_chunks="<fill in>"):
            # entities = lightrag_extract_via_ollama(chunks, model="qwen2.5:14b")
            pass  # <- replace with real call

    # --- Stage 5: MinHash dedup check ---------------------------------------
    with profiler.stage("minhash_dedup_check", book=book_path.name):
        # is_dup = lsh.query(minhash)
        pass  # <- replace with real call

    # --- Stage 6: Write to SurrealDB -----------------------------------------
    with profiler.stage("write_surrealdb", book=book_path.name):
        # db.insert(vectors=vectors, entities=entities if content_tier == "self_help" else None)
        pass  # <- replace with real call


if __name__ == "__main__":
    profiler = Profiler(out_dir="./profile_results", gpu_poll_interval=0.25)

    # Sample a handful of books across your three tiers — don't run the
    # full 14GB library for this first pass, you just need representative
    # numbers per tier.
    sample_books = [
        (Path("sample_software_book.epub"), "software"),
        (Path("sample_math_book.pdf"), "math"),
        (Path("sample_selfhelp_book.epub"), "self_help"),
    ]

    for book_path, tier in sample_books:
        run_pipeline_for_book(profiler, book_path, tier)

    profiler.report()
