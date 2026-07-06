"""
ingest.py

GraphRAG Assistant — Document Ingestion Pipeline

Orchestrates:
  1. File discovery
  2. PDF/EPUB extraction (multiprocessing, CPU-bound)
  3. Deduplication (MinHash LSH)
  4. LightRAG insertion — embedding + entity extraction + SurrealDB persistence (asyncio)

Usage:
    python ingest.py [--config config.yaml] [--reset]

Environment variables (can also live in a .env file):
    SURREALDB_URL, SURREALDB_NAMESPACE, SURREALDB_DATABASE,
    SURREALDB_USERNAME, SURREALDB_PASSWORD, SURREALDB_VECTOR_DIM,
    LIGHTRAG_WORKING_DIR, OLLAMA_HOST, MAX_CONCURRENT_INSERTS
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field, asdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import yaml
from bs4 import BeautifulSoup
from datasketch import MinHash, MinHashLSH
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# docling — pip install docling
from docling.document_converter import DocumentConverter

# ebooklib — pip install ebooklib
import ebooklib
from ebooklib import epub

# semantic-text-splitter — pip install semantic-text-splitter
from semantic_text_splitter import TextSplitter

# sentence-transformers — pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# LightRAG — pip install lightrag-hku
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

from pipeline_profiler import Profiler

load_dotenv(override=True)
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, show_path=False)],
)
logger = logging.getLogger(__name__)


def make_progress() -> Progress:
    """Standard progress bar layout shared by all pipeline stages."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single indexable text unit extracted from a source document."""
    text: str
    source_path: str
    content_type: str          # 'software' | 'math' | 'selfhelp'
    metadata: dict[str, Any] = field(default_factory=dict)
    # populated during chunk_software() — associated code blocks
    code_blocks: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    library_path: Path
    working_dir: Path
    extraction_workers: int = 8
    max_concurrent_inserts: int = 4
    # Per-chunk insertion timeout (seconds). Guards against a single chunk whose
    # Ollama entity-extraction "gleaning" loop runs away and wedges the whole run
    # (the DB write path is already serialized by the adapter's _query_lock, so a
    # straggler is far more likely to be a slow LLM step than DB contention).
    # A chunk that exceeds this is logged with its source path and skipped; re-run
    # with --from-checkpoint afterwards to re-attempt any that were skipped.
    # 0 disables the timeout.
    insert_timeout: int = 300
    dedup_threshold: float = 0.85
    dedup_num_perm: int = 128
    dedup_shingle_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    vector_dim: int = 384
    chunk_max_chars: int = 1500   # for semantic splitter
    # Map directory name fragments → content type
    content_type_rules: dict[str, str] = field(default_factory=lambda: {
        "software": "software",
        "dev":      "software",
        "code":     "software",
        "math":     "math",
        "maths":    "math",
        "self":     "selfhelp",
        "help":     "selfhelp",
        "personal": "selfhelp",
    })


def load_config(path: str | None) -> PipelineConfig:
    defaults: dict[str, Any] = {
        "library_path": os.getenv("LIBRARY_PATH", "./library"),
        "working_dir":  os.getenv("LIGHTRAG_WORKING_DIR", "./lightrag_data"),
        "extraction_workers": int(os.getenv("EXTRACTION_WORKERS", "8")),
        "max_concurrent_inserts": int(os.getenv("MAX_CONCURRENT_INSERTS", "4")),
        "insert_timeout": int(os.getenv("INSERT_TIMEOUT", "300")),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "ollama_host":  os.getenv("OLLAMA_HOST",  "http://localhost:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "qwen2.5:14b"),
        "vector_dim":   int(os.getenv("SURREALDB_VECTOR_DIM", "384")),
    }
    if path and Path(path).exists():
        with open(path) as f:
            file_cfg = yaml.safe_load(f) or {}
        defaults.update(file_cfg)
    return PipelineConfig(
        library_path=Path(defaults["library_path"]),
        working_dir=Path(defaults["working_dir"]),
        extraction_workers=defaults["extraction_workers"],
        max_concurrent_inserts=defaults["max_concurrent_inserts"],
        insert_timeout=defaults["insert_timeout"],
        embedding_model=defaults["embedding_model"],
        ollama_host=defaults["ollama_host"],
        ollama_model=defaults["ollama_model"],
        vector_dim=defaults["vector_dim"],
    )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_files(library_path: Path) -> list[Path]:
    """Recursively find all PDF and EPUB files under library_path."""
    files = []
    for ext in ("*.pdf", "*.epub"):
        files.extend(library_path.rglob(ext))
    logger.info(f"Discovered {len(files)} documents in {library_path}")
    return sorted(files)


# ---------------------------------------------------------------------------
# Content-type classification
# ---------------------------------------------------------------------------

_CONTENT_TYPE_RULES: dict[str, str] = {
    "software": "software",
    "dev":      "software",
    "code":     "software",
    "math":     "math",
    "maths":    "math",
    "self":     "selfhelp",
    "help":     "selfhelp",
    "personal": "selfhelp",
}

def classify(path: Path) -> str:
    """
    Infer content type from directory name fragments.
    Falls back to 'selfhelp' (full prose, no special handling).
    """
    parts = [p.lower() for p in path.parts]
    for part in parts:
        for keyword, ctype in _CONTENT_TYPE_RULES.items():
            if keyword in part:
                return ctype
    return "selfhelp"


# ---------------------------------------------------------------------------
# Extraction — PDF
# ---------------------------------------------------------------------------

def extract_pdf(path: Path) -> str:
    """
    Extract clean prose from a PDF using docling.
    Returns the full document text as a single string.
    """
    converter = DocumentConverter()
    result = converter.convert(str(path))
    return result.document.export_to_markdown()


# ---------------------------------------------------------------------------
# Extraction — EPUB
# ---------------------------------------------------------------------------

def extract_epub(path: Path) -> str:
    """
    Extract chapter-level text from an EPUB using ebooklib + BeautifulSoup.
    Chapters are joined with double newlines to preserve structure boundaries.
    """
    book = epub.read_epub(str(path))
    chapters: list[str] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        if text.strip():
            chapters.append(text)
    return "\n\n".join(chapters)


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

# Regex patterns for code block detection in software books
_CODE_FENCE_RE   = re.compile(r"```[\w]*\n([\s\S]*?)```", re.MULTILINE)
_INDENT_CODE_RE  = re.compile(r"(?m)^(?: {4}|\t)(.+)")
_FORMULA_RE      = re.compile(
    r"(\$\$[\s\S]+?\$\$"       # display LaTeX $$...$$
    r"|\$[^\$\n]+?\$"          # inline LaTeX $...$
    r"|\\begin\{[^}]+\}[\s\S]+?\\end\{[^}]+\})"  # \begin{} ... \end{}
)

_splitter: TextSplitter | None = None

def _get_splitter(max_chars: int = 1500) -> TextSplitter:
    global _splitter
    if _splitter is None:
        _splitter = TextSplitter(max_chars)
    return _splitter


def _semantic_split(text: str, max_chars: int = 1500) -> list[str]:
    """Split text into semantically coherent chunks via semantic-text-splitter."""
    splitter = _get_splitter(max_chars)
    return splitter.chunks(text)


def chunk_software(text: str, source_path: str, max_chars: int = 1500) -> list[Chunk]:
    """
    For software/dev books:
      1. Detect and extract fenced and indented code blocks.
      2. Replace each with a [CODE_N] placeholder in the prose stream.
      3. Semantically chunk the cleaned prose.
      4. Re-attach code blocks as metadata on the nearest chunk.
    """
    code_registry: dict[str, str] = {}

    def _replace_fence(m: re.Match) -> str:
        cid = f"CODE_{len(code_registry)}"
        code_registry[cid] = m.group(1).strip()
        return f"\n[{cid}]\n"

    def _replace_indent(m: re.Match) -> str:
        cid = f"CODE_{len(code_registry)}"
        code_registry[cid] = m.group(1).rstrip()
        return f"\n[{cid}]\n"

    prose = _CODE_FENCE_RE.sub(_replace_fence, text)
    prose = _INDENT_CODE_RE.sub(_replace_indent, prose)

    raw_chunks = _semantic_split(prose, max_chars)
    result: list[Chunk] = []
    for raw in raw_chunks:
        if not raw.strip():
            continue
        refs = re.findall(r"\[CODE_\d+\]", raw)
        blocks = [code_registry[r[1:-1]] for r in refs if r[1:-1] in code_registry]
        # Strip placeholder tokens from prose text
        clean = re.sub(r"\[CODE_\d+\]", "", raw).strip()
        if not clean:
            continue
        result.append(Chunk(
            text=clean,
            source_path=source_path,
            content_type="software",
            code_blocks=blocks,
            metadata={"has_code": bool(blocks)},
        ))
    return result


def chunk_math(text: str, source_path: str, max_chars: int = 1500) -> list[Chunk]:
    """
    For math books:
      Strip LaTeX formula regions (they don't convert cleanly to prose),
      then chunk the remaining prose.
    """
    clean = _FORMULA_RE.sub(" ", text)
    # Also drop lines that are mostly symbols / very short after formula removal
    lines = [l for l in clean.splitlines() if len(l.strip()) > 20]
    prose = "\n".join(lines)
    raw_chunks = _semantic_split(prose, max_chars)
    return [
        Chunk(text=c.strip(), source_path=source_path, content_type="math")
        for c in raw_chunks if c.strip()
    ]


def chunk_prose(text: str, source_path: str, content_type: str, max_chars: int = 1500) -> list[Chunk]:
    """
    Full semantic chunking with no special handling.
    Used for self-help / general prose.
    """
    raw_chunks = _semantic_split(text, max_chars)
    return [
        Chunk(text=c.strip(), source_path=source_path, content_type=content_type)
        for c in raw_chunks if c.strip()
    ]


def apply_content_rules(raw: str, path: Path) -> list[Chunk]:
    """Dispatch to the correct chunker based on classified content type."""
    source = str(path)
    ctype = classify(path)
    if ctype == "software":
        return chunk_software(raw, source)
    elif ctype == "math":
        return chunk_math(raw, source)
    else:
        return chunk_prose(raw, source, ctype)


# ---------------------------------------------------------------------------
# Top-level extraction (called inside multiprocessing.Pool.map)
# ---------------------------------------------------------------------------

def extract_document(path: Path) -> tuple[str, list[Chunk]]:
    """
    Entry point for multiprocessing workers.
    Returns an empty list on any extraction failure (logged, not raised).
    """
    try:
        logger.debug(f"Extracting: {path.name}")
        if path.suffix.lower() == ".pdf":
            raw = extract_pdf(path)
        elif path.suffix.lower() == ".epub":
            raw = extract_epub(path)
        else:
            return path.name, []
        return path.name, apply_content_rules(raw, path)
    except Exception as exc:
        logger.error(f"Extraction failed for {path}: {exc}")
        return path.name, []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def get_shingles(text: str, k: int = 5) -> list[str]:
    """Character k-gram shingles for MinHash."""
    text = text.lower()
    return [text[i:i + k] for i in range(len(text) - k + 1)]


def deduplicate(chunks: list[Chunk], threshold: float = 0.85, num_perm: int = 128, k: int = 5) -> list[Chunk]:
    """
    Remove near-duplicate chunks using MinHash LSH.
    Jaccard similarity >= threshold → keep first occurrence, discard later ones.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        m = MinHash(num_perm=num_perm)
        for shingle in get_shingles(chunk.text, k=k):
            m.update(shingle.encode("utf-8"))
        key = f"chunk_{i}"
        if not lsh.query(m):
            lsh.insert(key, m)
            kept.append(chunk)
    removed = len(chunks) - len(kept)
    logger.info(f"Deduplication: {len(chunks)} → {len(kept)} chunks ({removed} removed)")
    return kept

# ---------------------------------------------------------------------------
# Chunk checkpointing (post-dedup, pre-insertion)
# ---------------------------------------------------------------------------

def checkpoint_path(config: PipelineConfig) -> Path:
    return config.working_dir / "chunks_checkpoint.json"


def save_chunks_checkpoint(chunks: list[Chunk], config: PipelineConfig) -> None:
    """Persist the post-dedup chunk list to disk before insertion begins."""
    path = checkpoint_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f)
    logger.info(f"Checkpoint written: {len(chunks)} chunks -> {path}")


def load_chunks_checkpoint(config: PipelineConfig) -> list[Chunk]:
    """Load a previously saved chunk list, skipping extraction/dedup."""
    path = checkpoint_path(config)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    chunks = [Chunk(**c) for c in raw]
    logger.info(f"Checkpoint loaded: {len(chunks)} chunks <- {path}")
    return chunks




# ---------------------------------------------------------------------------
# LightRAG initialisation
# ---------------------------------------------------------------------------

def make_embedding_func(model_name: str, vector_dim: int) -> EmbeddingFunc:
    """
    Build a LightRAG EmbeddingFunc using a local sentence-transformers model.
    The model is loaded once and reused (CUDA if available).
    """
    _model = SentenceTransformer(model_name)

    async def _embed(texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        # Run GPU inference in a thread pool to avoid blocking the event loop
        # show_progress_bar=False: SentenceTransformer.encode() defaults to
        # showing a tqdm bar whenever the logger is at INFO or below. Since
        # this fires on every single embedding call (i.e. per chunk, at
        # whatever concurrency the insertion semaphore allows), it was
        # spamming a fresh "Batches" bar per call instead of updating one —
        # tqdm's cursor control and Rich's Live display don't share a line.
        # The outer "Inserting chunks" Rich bar already tracks real progress.
        embeddings = await loop.run_in_executor(
            None,
            lambda: _model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()
        )
        return embeddings

    return EmbeddingFunc(embedding_dim=vector_dim, max_token_size=512, func=_embed)


def make_ollama_func(host: str):
    """
    Build an Ollama LLM func compatible with LightRAG's llm_model_func interface.
    ollama_model_complete() doesn't take a `model` kwarg — it derives the model
    name internally from kwargs["hashing_kv"].global_config["llm_model_name"],
    which LightRAG always injects. Passing model= here leaks through into
    _ollama_model_if_cache's **kwargs and collides with its positional model arg.
    """
    async def _llm(prompt: str, **kwargs) -> str:
        return await ollama_model_complete(
            prompt,
            host=host,
            **kwargs,
        )
    return _llm


async def init_lightrag(config: PipelineConfig) -> LightRAG:
    """Initialise LightRAG with the SurrealDB backend and local models."""
    config.working_dir.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(config.working_dir),
        llm_model_func=make_ollama_func(config.ollama_host),
        llm_model_name=config.ollama_model,
        embedding_func=make_embedding_func(config.embedding_model, config.vector_dim),
        kv_storage="SurrealDBKVStorage",
        vector_storage="SurrealDBVectorStorage",
        graph_storage="SurrealDBGraphStorage",
        doc_status_storage="SurrealDBDocStatusStorage",
    )
    await rag.initialize_storages()
    # Required setup step LightRAG normally does during FastAPI lifespan —
    # without it, pipeline_status["history_messages"] doesn't exist and the
    # first ainsert() crashes after setting busy=True but before the
    # try/finally that would reset it, permanently wedging every later
    # insert into a silent no-op ("pipeline already busy") for the rest of
    # the process.
    await initialize_pipeline_status()
    logger.info("LightRAG initialised with SurrealDB backend")
    return rag


# ---------------------------------------------------------------------------
# Insertion with checkpointing (via DocStatusStorage)
# ---------------------------------------------------------------------------

async def insert_with_semaphore(
    rag: LightRAG,
    chunk: Chunk,
    semaphore: asyncio.Semaphore,
    progress: Progress | None = None,
    task_id: Any = None,
    timeout: float | None = None,
) -> None:
    """
    Insert a single chunk into LightRAG under the semaphore.
    Each chunk is inserted as its own LightRAG "document" — do not pass a
    shared `ids` value across chunks from the same source file: LightRAG's
    checkpoint dedup treats a repeated id as "already processed" regardless
    of content, which would silently drop every chunk after the first one
    for a given book. Content-hash IDs (the default when `ids` is omitted)
    keep each chunk independently tracked and idempotent across re-runs.

    If a Rich progress + task_id are passed, the bar description is updated
    to the source file *on semaphore acquire* (i.e. when this chunk actually
    starts inserting, not when it finishes). This surfaces what's in flight:
    with N concurrent inserts the description flickers among the N active
    files (last-writer-wins), and when the run collapses to a single slow
    straggler the description stabilises on that file's name — which is the
    only way to see which chunk is hanging, since ainsert() itself is opaque.

    `timeout` (seconds, None/0 = disabled) caps a single ainsert(). A chunk
    that exceeds it — most likely a runaway Ollama gleaning loop, since the
    adapter already serializes DB writes — is cancelled, logged with its
    source path, and skipped so it can't wedge the whole run. Cancellation is
    safe here: the adapter's _query_lock is released on CancelledError (it's
    held only inside an `async with` around each individual query), and chunk
    upserts are content-hash idempotent, so a --from-checkpoint re-run cleanly
    re-attempts anything skipped. A skipped chunk is simply absent from the
    graph until then, not half-written in a corrupt state.
    """
    async with semaphore:
        if progress is not None and task_id is not None:
            progress.update(task_id, description=f"Inserting {Path(chunk.source_path).name}")
        try:
            coro = rag.ainsert(chunk.text, file_paths=chunk.source_path)
            if timeout:
                await asyncio.wait_for(coro, timeout=timeout)
            else:
                await coro
        except asyncio.TimeoutError:
            logger.error(
                f"Insert TIMED OUT after {timeout}s for a chunk from "
                f"{chunk.source_path} — skipped. Re-run with --from-checkpoint "
                f"to re-attempt. (Likely a runaway LLM entity-extraction step.)"
            )
        except Exception as exc:
            logger.error(f"Insert failed for chunk from {chunk.source_path}: {exc}")


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def reset_database(config: PipelineConfig) -> None:
    """
    Delete the embedded SurrealKV database so the next init rebuilds it from
    scratch. For an embedded file-based store this is the reliable equivalent
    of "drop and rebuild all tables" — it clears every namespace/table in one
    step, and the storage classes recreate their schema via
    `DEFINE TABLE IF NOT EXISTS ...` on the next `initialize()`.

    Must run BEFORE init_lightrag() connects, since connecting opens the file.
    """
    db_path = Path(os.getenv("SURREALDB_PATH", str(config.working_dir / "graphrag.db")))
    if db_path.exists():
        if db_path.is_dir():
            shutil.rmtree(db_path)
        else:
            db_path.unlink()
        logger.warning(f"--reset: removed embedded database at {db_path}")
    else:
        logger.info(f"--reset: no existing database found at {db_path}")


async def run_pipeline(config: PipelineConfig, reset: bool = False, from_checkpoint: bool = False, profile: bool = False) -> None:
    # Stage 0: (Optional) reset existing data — must happen before we connect
    if reset:
        reset_database(config)

    profiler = Profiler(out_dir=str(config.working_dir / "profile_results")) if profile else None

    def stage(name: str, **meta):
        return profiler.stage(name, **meta) if profiler else contextlib.nullcontext()

    # Stage 1: Initialise LightRAG + SurrealDB
    rag = await init_lightrag(config)
    try:
        if from_checkpoint and checkpoint_path(config).exists():
            chunks = load_chunks_checkpoint(config)
        else:
            if from_checkpoint:
                logger.warning(f"--from-checkpoint set but no checkpoint found at {checkpoint_path(config)}; running full pipeline")

            files = discover_files(config.library_path)
            if not files:
                logger.warning(f"No PDF/EPUB files found under {config.library_path}")
                return

            # Stage 3: Parallel extraction (CPU-bound → multiprocessing)
            # NOTE: worker count is capped well below cpu_count() — each worker runs a full
            # docling+EasyOCR pipeline that can need several GB for large PDFs, so a high
            # process count risks OOM (workers silently drop documents via MemoryError,
            # producing a near-empty result set instead of a clean failure).
            logger.info(f"Extracting {len(files)} documents using {config.extraction_workers} workers...")
            all_chunks: list[Chunk] = []
            with stage("extraction", n_files=len(files)):
                with make_progress() as progress, Pool(processes=config.extraction_workers) as pool:
                    task = progress.add_task("Extracting documents", total=len(files))
                    for name, doc_chunks in pool.imap_unordered(extract_document, files):
                        all_chunks.extend(doc_chunks)
                        progress.update(task, description=f"Extracted {name}")
                        progress.advance(task)

            logger.info(f"Extraction complete: {len(all_chunks)} raw chunks")

            # Stage 4: Deduplication
            with stage("deduplication", n_chunks=len(all_chunks)):
                chunks = deduplicate(
                    all_chunks,
                    threshold=config.dedup_threshold,
                    num_perm=config.dedup_num_perm,
                    k=config.dedup_shingle_k,
                )

            # Checkpoint: persist the post-dedup chunk list before insertion begins
            save_chunks_checkpoint(chunks, config)

        # Stage 5: Async insertion into LightRAG / SurrealDB — runs on BOTH the
        # checkpoint-resume path and the fresh-run path, not just the latter.
        # NOTE: this single stage covers embedding + entity extraction + SurrealDB
        # write together — LightRAG's ainsert() bundles all three into one opaque
        # async call, so there's no hook point to time them individually.
        logger.info(f"Inserting {len(chunks)} chunks (concurrency={config.max_concurrent_inserts})...")
        with stage("lightrag_insertion", n_chunks=len(chunks), concurrency=config.max_concurrent_inserts):
            semaphore = asyncio.Semaphore(config.max_concurrent_inserts)
            with make_progress() as progress:
                task = progress.add_task("Inserting chunks", total=len(chunks))
                tasks = [
                    insert_with_semaphore(rag, chunk, semaphore, progress, task, config.insert_timeout)
                    for chunk in chunks
                ]
                for coro in asyncio.as_completed(tasks):
                    await coro
                    progress.advance(task)

        logger.info("Ingestion pipeline complete.")
        if profiler:
            profiler.report()
    finally:
        # Embedded SurrealKV buffers writes — without this, data can be lost
        # if the process exits without an explicit flush/close. On a large
        # graph this flush is not instant, so give it its own visible phase
        # AND an elapsed-time heartbeat.
        #
        # Why not rely on the spinner: Rich drives spinner animation from a
        # background refresh thread on wall-clock time, so it keeps spinning
        # even if finalize is wedged — it can't distinguish progress from a
        # hang. The heartbeat below only logs while the event loop is actually
        # being serviced, so:
        #   - ticking heartbeat  → finalize is progressing / async-waiting
        #   - frozen heartbeat   → the call is blocking the loop in native
        #                          code (busy or deadlocked); attach with
        #                          `py-spy dump --pid <PID>` from another
        #                          terminal, or drop in
        #                          faulthandler.dump_traceback_later(30, repeat=True)
        #                          before this block, to see exactly where.
        # shield() keeps the 5s poll timeout from cancelling the flush —
        # cancelling finalize mid-write is exactly the data loss it prevents.
        with make_progress() as progress:
            progress.add_task("Flushing to SurrealDB (finalizing storages)", total=None)
            finalize_task = asyncio.create_task(rag.finalize_storages())
            waited = 0
            while not finalize_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(finalize_task), timeout=5)
                except asyncio.TimeoutError:
                    waited += 5
                    logger.info(f"finalize_storages() still running… {waited}s elapsed")
            await finalize_task  # surface any exception raised by finalize


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _shutdown_loop(loop: asyncio.AbstractEventLoop, timeout: float = 10.0) -> None:
    """
    Bounded replacement for the teardown asyncio.run() does after the main
    coroutine finishes.

    asyncio.run()'s built-in _cancel_all_tasks() cancels leftover tasks and then
    waits on an UNBOUNDED `gather(*leftover, return_exceptions=True)`. If a
    third-party async resource leaves a task parked on a native Windows IOCP read
    — e.g. LightRAG's Ollama/httpx client, or LLM sub-tasks orphaned when a
    runaway ainsert() is cancelled by the per-chunk insert_timeout — that gather
    never returns, and the process hangs in `GetQueuedCompletionStatus` at the
    "finalizing storages" line with all data already flushed.

    This does the same cancel-and-drain but:
      (a) logs the repr() of every leftover task first, so the actual leak can be
          identified and closed properly (the real fix is usually an explicit
          client .aclose() before shutdown), and
      (b) waits only `timeout` seconds before abandoning any task that refuses to
          cancel and forcing the loop shut. wait_for schedules a loop callback, so
          on the proactor loop this is guaranteed to wake and return rather than
          blocking in the completion port forever.
    Abandoning leftovers is safe here: finalize_storages() already flushed
    SurrealKV, so the orphaned tasks are just network connections we don't need.
    """
    pending = list(asyncio.all_tasks(loop))
    if pending:
        logger.warning(f"{len(pending)} task(s) still pending at shutdown; cancelling:")
        for t in pending:
            logger.warning(f"  leftover task: {t.get_coro()!r}")
            t.cancel()
        try:
            loop.run_until_complete(
                asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=timeout,
                )
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Leftover task(s) did not cancel within {timeout}s — abandoning them "
                "and forcing shutdown. Data was already flushed by finalize_storages(), "
                "so this is safe; the repr(s) above identify what to close explicitly."
            )
    loop.run_until_complete(loop.shutdown_asyncgens())


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG Assistant ingestion pipeline")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild SurrealDB tables")
    parser.add_argument(
        "--from-checkpoint",
        action="store_true",
        help="Skip discovery/extraction/dedup and reuse the last saved chunks_checkpoint.json",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-stage timing via pipeline_profiler (writes to working_dir/profile_results)",
    )
    parser.add_argument(
        "--debug-finalize",
        action="store_true",
        help=(
            "Diagnose end-of-run hangs. Arms faulthandler to dump every thread's "
            "stack to finalize_traceback.log on a timer (default 30s, override with "
            "DEBUG_FINALIZE_INTERVAL). The timer runs on a dedicated C thread, so it "
            "fires even when the main thread is fully blocked — capturing a hang in "
            "finalize_storages() OR in asyncio.run()'s _cancel_all_tasks teardown "
            "(which a finalize-only probe would miss). Off by default; only pass it "
            "when actively chasing a hang. Dumps go to a file, not the console, so "
            "they don't disturb the Rich progress UI."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # --debug-finalize: arm faulthandler for the WHOLE asyncio.run(), not just the
    # finalize block. The py-spy dump that motivated this flag showed the process
    # parked in asyncio.run's _cancel_all_tasks teardown — i.e. *after* run_pipeline
    # returned — so a probe scoped only to finalize_storages() would never see it.
    # Wrapping the whole run covers finalize, the teardown sweep, and any insertion
    # straggler too. Line-buffered file so the last dump survives a hard kill.
    dump_fh = None
    if args.debug_finalize:
        import faulthandler
        interval = int(os.getenv("DEBUG_FINALIZE_INTERVAL", "30"))
        dump_fh = open("finalize_traceback.log", "w", buffering=1)
        faulthandler.enable(file=dump_fh)
        faulthandler.dump_traceback_later(interval, repeat=True, file=dump_fh)
        logger.info(
            f"--debug-finalize: dumping all thread stacks every {interval}s "
            f"to finalize_traceback.log"
        )

    # Manage the loop manually instead of asyncio.run() so we control teardown:
    # asyncio.run()'s built-in _cancel_all_tasks waits forever on leftover tasks,
    # which is the observed hang. _shutdown_loop() bounds and logs that instead.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_pipeline(
            config,
            reset=args.reset,
            from_checkpoint=args.from_checkpoint,
            profile=args.profile,
        ))
    finally:
        _shutdown_loop(loop)
        if dump_fh is not None:
            import faulthandler
            faulthandler.cancel_dump_traceback_later()
            dump_fh.close()
        loop.close()

if __name__ == "__main__":
    main()