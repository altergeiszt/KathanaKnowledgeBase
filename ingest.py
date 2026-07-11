"""
ingest.py

GraphRAG Assistant — Document Ingestion Pipeline (LlamaIndex + Neo4j)

Orchestrates:
  1. File discovery
  2. PDF/EPUB extraction (multiprocessing, CPU-bound) — docling
  3. Deduplication (MinHash LSH)
  4. Node building + HYBRID insertion into a Neo4j PropertyGraphIndex:
       - curated-slice books (classify.is_curated) → LLM entity extraction
       - everything else                          → embeddings-only
     Full-corpus LLM extraction is infeasible (~3.9s/chunk → days; §10 of
     Migration_LlamaIndex.md), so only a hand-picked curated slice gets
     kg_extractors while every chunk is still embedded and vector-searchable (§3).

Usage:
    python ingest.py [--config config.yaml] [--reset] [--from-checkpoint] [--profile]

Environment variables (can also live in a .env file):
    LIBRARY_PATH, LIGHTRAG_WORKING_DIR (working dir / checkpoint location),
    EMBEDDING_MODEL, OLLAMA_HOST, OLLAMA_MODEL, SURREALDB_VECTOR_DIM (embed dim),
    NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict, fields, MISSING
from datetime import datetime
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

# LlamaIndex + Neo4j — the post-migration insertion stack (§5, §11 of Migration_LlamaIndex.md)
from llama_index.core import PropertyGraphIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import (
    TextNode,
    RelatedNodeInfo,
    NodeRelationship,
    TransformComponent,
)
from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Two-tier, chunk-level content-type + curated-slice classification (classify.py).
# classify_content (Tier1+Tier2) → per-chunk content_type metadata (chunk_to_nodes);
# route_signals → document-level content densities for chunker routing (§2.1).
from classify import classify as classify_content, route_signals, is_curated, ContentType

# Optional per-stage profiler (moved to .archived_code during migration housekeeping).
# --profile degrades to a warning if it isn't importable, rather than hard-failing.
try:
    from pipeline_profiler import Profiler
except ModuleNotFoundError:
    Profiler = None

load_dotenv(override=True)
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, show_path=False)],
)
logger = logging.getLogger(__name__)


def add_file_logging(working_dir: Path) -> Path:
    """Attach a timestamped FileHandler to the root logger so every run leaves an
    on-disk record. Console logging (RichHandler) is console-only, so a crashed or
    overnight run otherwise leaves no trace once the terminal closes — which is
    exactly how a silent zero-entity run went undiagnosed. Returns the log path."""
    log_dir = working_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ingest_{datetime.now():%Y%m%d_%H%M%S}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    logger.info(f"Logging to {log_path}")
    return log_path


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
    # Repurposed post-migration: the Ollama LLM request timeout (seconds) for the
    # curated-slice entity extraction. A single extraction call runs several seconds
    # (§10 ~3.9s median), so the LLM gets real headroom instead of the 30s default.
    insert_timeout: int = 300
    dedup_threshold: float = 0.85
    dedup_num_perm: int = 128
    dedup_shingle_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    # Extraction LLM: the model SimpleLLMPathExtractor uses for the curated slice.
    # Empty → falls back to ollama_model (so existing single-model setups are unchanged).
    # Split out so extraction can use a fast small model independent of any future
    # query-role model.
    extract_model: str = ""
    # Thinking mode for the extraction LLM: "auto" | "on" | "off". "auto" disables
    # thinking for thinking-capable models (qwen3, deepseek-r1, gpt-oss, …) — which is
    # what you want for high-volume structured extraction — and leaves it unset for
    # non-thinking models like qwen2.5 (sending think:false to those can error).
    extract_thinking: str = "auto"
    vector_dim: int = 384
    chunk_max_chars: int = 1500   # for semantic splitter
    # Neo4j PropertyGraphStore connection (§11). Password comes from the env only —
    # never commit it to config.yaml.
    neo4j_url: str = "neo4j://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "llamaindex"
    # Curated-slice LLM extraction knobs (§10 — single pass, no gleaning).
    max_paths_per_chunk: int = 10
    extract_num_workers: int = 4
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


# Config key -> (environment variable name(s), parser). Multiple names = fallbacks,
# first-set wins. Keys NOT listed here (dedup_*, chunk_max_chars, content_type_rules)
# are configurable via config.yaml only — they have no env override.
_ENV_SPEC: dict[str, tuple[list[str], Any]] = {
    "library_path":           (["LIBRARY_PATH"], str),
    "working_dir":            (["WORKING_DIR", "LIGHTRAG_WORKING_DIR"], str),
    "extraction_workers":     (["EXTRACTION_WORKERS"], int),
    "max_concurrent_inserts": (["MAX_CONCURRENT_INSERTS"], int),
    "insert_timeout":         (["INSERT_TIMEOUT"], int),
    "embedding_model":        (["EMBEDDING_MODEL"], str),
    "ollama_host":            (["OLLAMA_HOST"], str),
    "ollama_model":           (["OLLAMA_MODEL"], str),
    "extract_model":          (["EXTRACT_MODEL"], str),
    "extract_thinking":       (["EXTRACT_THINKING"], str),
    "vector_dim":             (["VECTOR_DIM", "SURREALDB_VECTOR_DIM"], int),
    "neo4j_url":              (["NEO4J_URL"], str),
    "neo4j_user":             (["NEO4J_USER"], str),
    "neo4j_password":         (["NEO4J_PASSWORD"], str),
    "neo4j_database":         (["NEO4J_DATABASE"], str),
    "max_paths_per_chunk":    (["MAX_PATHS_PER_CHUNK"], int),
    "extract_num_workers":    (["EXTRACT_NUM_WORKERS"], int),
}


def load_config(path: str | None) -> PipelineConfig:
    """Resolve config with a single, documented precedence (highest wins):

        environment / .env   >   config.yaml   >   dataclass defaults

    Only env vars that are ACTUALLY SET override — an unset var never clobbers a
    config.yaml value with a hardcoded fallback. Building the config from the full
    field list (not a hand-picked subset) means no config.yaml key is silently dropped.
    """
    known = {f.name for f in fields(PipelineConfig)}

    # Layer 1 — dataclass field defaults (the single source of truth for defaults).
    resolved: dict[str, Any] = {}
    for f in fields(PipelineConfig):
        if f.default is not MISSING:
            resolved[f.name] = f.default
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            resolved[f.name] = f.default_factory()  # type: ignore[misc]
    resolved.setdefault("library_path", "./library")       # required fields: seed a fallback
    resolved.setdefault("working_dir", "./lightrag_data")

    # Layer 2 — config.yaml (all recognized keys, so dedup_*/chunk_max_chars/etc. apply).
    file_cfg: dict[str, Any] = {}
    if path and Path(path).exists():
        with open(path) as f:
            file_cfg = yaml.safe_load(f) or {}
    for k, v in file_cfg.items():
        if k in known:
            resolved[k] = v
        else:
            logger.warning(f"config.yaml: ignoring unknown key '{k}'")

    # Layer 3 — environment / .env, only for vars actually present. Warn loudly when an
    # env var overrides a DIFFERENT config.yaml value so the conflict is never silent.
    for key, (env_names, parser) in _ENV_SPEC.items():
        raw = next((os.environ[n] for n in env_names if n in os.environ), None)
        if raw is None:
            continue
        val = parser(raw)
        if key in file_cfg and str(file_cfg[key]) != str(val):
            logger.warning(
                f"Config conflict on '{key}': config.yaml={file_cfg[key]!r} overridden by "
                f"env {env_names[0]}={val!r} (env wins)."
            )
        resolved[key] = val

    # extract_model defaults to the general ollama_model when left unset.
    if not resolved.get("extract_model"):
        resolved["extract_model"] = resolved["ollama_model"]

    resolved["library_path"] = Path(resolved["library_path"])
    resolved["working_dir"] = Path(resolved["working_dir"])
    return PipelineConfig(**resolved)


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
# Chunker routing — content-based (§2.1)
# ---------------------------------------------------------------------------

# Document-scaled routing thresholds. classify.py's per-CHUNK threshold (2.5
# matches/1000 chars) is far too high for whole-document routing — measured book-level
# densities run ~0.02–2.1 — so these are separate, PROVISIONAL values fit to the 7-file
# checkpoint sample; re-validate on a larger sample before trusting them broadly.
#   math ~2.1 (real math book) vs ≤0.4 (prose)  -> a 1.0 floor separates cleanly
#   software density is weak everywhere; genuine fenced code is the reliable signal
_MATH_DENSITY_FLOOR = 1.0
_SOFT_DENSITY_FLOOR = 0.3   # well above prose (~0.02); catches dense inline code w/o fences
_FENCE_FLOOR = 1           # any single genuine (code-token-bearing) fenced block


def _route_chunker(raw: str, path: Path) -> str:
    """Pick the chunker for a document from its CONTENT, not its path (§2.1).

    The old path-based classify() keyed off directory-name fragments, but the library
    is flat, so almost everything defaulted to 'selfhelp' — math books skipped LaTeX
    handling and code books skipped code-block extraction. This combines two signals:

      1. Curated hand-labels (classify_content Tier 1) — authoritative for the 19
         curated books the user labeled by hand.
      2. Document-scaled content densities (classify.route_signals) — for everything
         else, and as a supplement (a curated {ALGORITHMS, ML} label says nothing about
         whether the book physically contains code worth extracting).

    Code takes priority over math for books that are both (e.g. a Python+math workshop):
    un-extracted code pollutes prose chunks more than residual LaTeX does.

    KNOWN GAP: books whose code docling leaves INLINE (no ``` fences) score ~0 on every
    signal and fall through to prose — see the .NET fundamentals case in §2.1. The
    typed-subfolder fallback (§2.1) is the real fix for those; not handled here.
    """
    labels = classify_content(raw, str(path))
    sig = route_signals(raw)
    has_code = (
        ContentType.SOFTWARE in labels or ContentType.PYTHON in labels
        or sig["fences"] >= _FENCE_FLOOR
        or sig["software"] >= _SOFT_DENSITY_FLOOR
    )
    has_math = (
        ContentType.MATH in labels or ContentType.DISCRETE_MATH in labels
        or sig["math"] >= _MATH_DENSITY_FLOOR
    )
    if has_code:
        return "software"
    if has_math:
        return "math"
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
    """Dispatch to the correct chunker based on the document's content (§2.1)."""
    source = str(path)
    ctype = _route_chunker(raw, path)
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
    Returns (source_path, chunks); an empty list on any extraction failure
    (logged, not raised). The first element is the FULL source path (not just the
    basename) so the per-book raw cache can fingerprint the source file.
    """
    try:
        logger.debug(f"Extracting: {path.name}")
        if path.suffix.lower() == ".pdf":
            raw = extract_pdf(path)
        elif path.suffix.lower() == ".epub":
            raw = extract_epub(path)
        else:
            return str(path), []
        return str(path), apply_content_rules(raw, path)
    except Exception as exc:
        logger.error(f"Extraction failed for {path}: {exc}")
        return str(path), []


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
# Atomic JSON write — temp file in the same dir + os.replace, so a crash mid-write
# never leaves a truncated file that a later run would load as a valid (short)
# checkpoint. os.replace is atomic on the same filesystem.
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Per-book raw chunk cache (pre-dedup crash resilience — Option A)
# ---------------------------------------------------------------------------
# Extraction (docling + EasyOCR) is the slow, memory-hungry, crash-prone phase.
# Each book's raw chunks are cached the moment its worker returns, so a crash or
# OOM mid-run loses at most the in-flight book: a re-run reloads the cached books
# and re-parses only what is missing or stale. Dedup stays a single GLOBAL pass
# over the reassembled corpus (it is inherently cross-book), so these cache
# entries are PRE-dedup — the post-dedup checkpoint is written once, downstream.

def raw_cache_dir(config: PipelineConfig) -> Path:
    return config.working_dir / "raw_chunks"


def _file_fingerprint(path: Path) -> str:
    """Size+mtime signature so an edited source file invalidates its stale cache."""
    st = path.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def raw_cache_path(config: PipelineConfig, source: str | Path) -> Path:
    return raw_cache_dir(config) / f"{Path(source).stem}.json"


def save_raw_book(source: str | Path, chunks: list[Chunk], config: PipelineConfig) -> None:
    """Persist one book's raw (pre-dedup) chunks, keyed by a size+mtime fingerprint.

    Callers should skip empty chunk lists (an extraction failure returns []): caching
    an empty result would make a transient OOM look like a permanently-parsed book and
    suppress the retry on the next run.
    """
    src = Path(source)
    payload = {
        "source": str(src),
        "fingerprint": _file_fingerprint(src) if src.exists() else "",
        "chunks": [asdict(c) for c in chunks],
    }
    _atomic_write_json(raw_cache_path(config, source), payload)


def load_raw_book(source: str | Path, config: PipelineConfig) -> list[Chunk] | None:
    """Return cached raw chunks for a book, or None if absent, stale, or corrupt.

    Stale = the source file's size+mtime no longer matches the cached fingerprint,
    so an edited PDF/EPUB is re-parsed instead of served from a stale cache. A
    half-written cache (crash mid-write) fails to parse and is treated as absent.
    """
    path = raw_cache_path(config, source)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None  # corrupt / half-written → re-parse
    src = Path(source)
    if src.exists() and payload.get("fingerprint") != _file_fingerprint(src):
        return None
    return [Chunk(**c) for c in payload["chunks"]]


# ---------------------------------------------------------------------------
# Chunk checkpointing (post-dedup, pre-insertion)
# ---------------------------------------------------------------------------

def checkpoint_path(config: PipelineConfig) -> Path:
    return config.working_dir / "chunks_checkpoint.json"


def save_chunks_checkpoint(chunks: list[Chunk], config: PipelineConfig) -> None:
    """Persist the post-dedup chunk list to disk before insertion begins."""
    path = checkpoint_path(config)
    _atomic_write_json(path, [asdict(c) for c in chunks])
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
# Extraction progress checkpoint (INSERT-half resume — the expensive LLM pass)
# ---------------------------------------------------------------------------
# The curated-slice LLM extraction runs the model per chunk (~seconds each), and a
# single insert_nodes() call over thousands of chunks has no resume point: a crash
# restarts the whole pass. Worse, TextNode ids are random per run, so a re-run
# duplicates rather than MERGE-ing — Neo4j gives no cross-run idempotency here.
#
# So we insert in chunk-batches and, after each batch COMMITS, record the completed
# chunks by a STABLE content key (hash of source_path+text — not the volatile
# node_id) in extract_progress.json. A resume skips done chunks, re-running only the
# unfinished tail. The file is fingerprinted to the checkpoint it was built against,
# so a changed corpus invalidates stale progress. Cleared on clean completion / --reset.

EXTRACT_BATCH_CHUNKS = 50  # chunks per commit; bounds re-work on crash to one batch


def extract_progress_path(config: PipelineConfig) -> Path:
    return config.working_dir / "extract_progress.json"


def _checkpoint_fingerprint(config: PipelineConfig) -> str:
    """size:mtime of chunks_checkpoint.json — ties progress to a specific corpus."""
    st = checkpoint_path(config).stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def _chunk_key(chunk: Chunk) -> str:
    """Stable per-chunk key that survives re-running chunk_to_nodes (unlike node_id)."""
    h = hashlib.sha1()
    h.update(chunk.source_path.encode("utf-8"))
    h.update(b"\x00")
    h.update(chunk.text.encode("utf-8"))
    return h.hexdigest()


def load_extract_progress(config: PipelineConfig) -> dict:
    """Return {'fingerprint', 'embed': set, 'curated': set}. Absent, unreadable, or
    stale (checkpoint changed) progress yields a fresh empty record."""
    fresh = {"fingerprint": _checkpoint_fingerprint(config), "embed": set(), "curated": set()}
    path = extract_progress_path(config)
    if not path.exists():
        return fresh
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return fresh
    if data.get("fingerprint") != fresh["fingerprint"]:
        logger.warning("extract_progress.json is stale (checkpoint changed); ignoring it")
        return fresh
    return {
        "fingerprint": data["fingerprint"],
        "embed": set(data.get("embed", [])),
        "curated": set(data.get("curated", [])),
    }


def save_extract_progress(progress: dict, config: PipelineConfig) -> None:
    _atomic_write_json(extract_progress_path(config), {
        "fingerprint": progress["fingerprint"],
        "embed": sorted(progress["embed"]),
        "curated": sorted(progress["curated"]),
    })


def clear_extract_progress(config: PipelineConfig) -> None:
    extract_progress_path(config).unlink(missing_ok=True)


def insert_chunks_batched(
    index: PropertyGraphIndex,
    chunks_subset: list[Chunk],
    progress: dict,
    bucket: str,               # "embed" | "curated"
    config: PipelineConfig,
    *,
    label: str,
    batch_chunks: int = EXTRACT_BATCH_CHUNKS,
) -> int:
    """Insert nodes for chunks_subset in chunk-batches, skipping any already recorded
    done in progress[bucket], checkpointing progress after each committed batch. Returns
    the node count inserted this run."""
    done = progress[bucket]
    pending = [c for c in chunks_subset if _chunk_key(c) not in done]
    skipped = len(chunks_subset) - len(pending)
    if skipped:
        logger.info(f"{label}: skipping {skipped} chunk(s) already done (resume)")
    if not pending:
        return 0

    total_batches = (len(pending) + batch_chunks - 1) // batch_chunks
    inserted = 0
    for bi, i in enumerate(range(0, len(pending), batch_chunks), start=1):
        batch = pending[i : i + batch_chunks]
        nodes: list[TextNode] = []
        for c in batch:
            nodes.extend(chunk_to_nodes(c))
        logger.info(f"{label}: batch {bi}/{total_batches} - {len(nodes)} node(s) from {len(batch)} chunk(s)...")
        index.insert_nodes(nodes)                 # commits to Neo4j
        for c in batch:                           # mark done only AFTER the commit
            done.add(_chunk_key(c))
        save_extract_progress(progress, config)   # atomic; survives a crash on the next batch
        inserted += len(nodes)
    return inserted


# ---------------------------------------------------------------------------
# Stable IDs — content-hash ref_doc_id (§0/§4), preserved MD5-of-path scheme
# ---------------------------------------------------------------------------

def content_hash_id(source_path: str) -> str:
    """One stable ref_doc_id per source book: the 'book::' namespace + MD5 of the
    source path. All chunks of a book share this id, so a book is a single ref_doc
    (§3/§4) and --reset can purge the whole book namespace by prefix."""
    digest = hashlib.md5(str(source_path).encode("utf-8")).hexdigest()
    return f"book::{digest}"


# ---------------------------------------------------------------------------
# Chunk → LlamaIndex TextNode(s)  (§4)
# ---------------------------------------------------------------------------

def _content_type_labels(text: str, source_path: str) -> list[str]:
    """Two-tier content-type classification (classify.py): exact-title book labels
    for the curated slice, else per-chunk density heuristics. Returns a sorted list
    of type values — multi-label, since a chunk can legitimately be e.g.
    ['math', 'python'] (§2.1). Replaces the unreliable path-based content_type."""
    return sorted(ct.value for ct in classify_content(text, source_path))


def chunk_to_nodes(chunk: Chunk) -> list[TextNode]:
    """Build the base prose TextNode plus one linked TextNode per code block (§4).

    ref_doc_id is set via the SOURCE relationship — NOT `node.ref_doc_id = ...`,
    which is a read-only property in llama-index 0.14.x (see §7 Resolution). This is
    also what delete/rebuild logic keys off, so it must be set correctly here.
    """
    doc_id = content_hash_id(chunk.source_path)
    labels = _content_type_labels(chunk.text, chunk.source_path)
    base = TextNode(
        text=chunk.text,
        metadata={
            "source_type": "book",
            "source_path": chunk.source_path,
            "content_type": labels,
            "has_code": bool(chunk.metadata.get("has_code") or chunk.code_blocks),
        },
    )
    # This metadata is for retrieval FILTERING only, not semantic content. By default
    # LlamaIndex prepends it into the text seen by both the embedder and the LLM
    # extractor — which made SimpleLLMPathExtractor mine entities out of
    # "source_type=book, has_code=False, source_path=..." instead of the prose. Exclude
    # it from both representations so extraction/embedding see only the chunk text.
    _exclude_metadata_from_content(base)
    base.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    nodes: list[TextNode] = [base]
    # code_blocks-as-separate-nodes (§4 knob): a code query should hit code, a
    # concept query should hit prose — keep them as distinct retrievable nodes
    # linked to the prose parent, not flattened into metadata.
    for code in (chunk.code_blocks or []):
        if not code.strip():
            continue
        cn = TextNode(
            text=code,
            metadata={
                "source_type": "book",
                "source_path": chunk.source_path,
                "content_type": ["code"],
                "parent": base.node_id,
            },
        )
        _exclude_metadata_from_content(cn)
        cn.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
        nodes.append(cn)
    return nodes


def _exclude_metadata_from_content(node: TextNode) -> None:
    """Keep node metadata out of the text the embedder and LLM extractor see — it's
    filter metadata, not content (see chunk_to_nodes)."""
    keys = list(node.metadata.keys())
    node.excluded_llm_metadata_keys = keys
    node.excluded_embed_metadata_keys = keys


# ---------------------------------------------------------------------------
# Embed-only extractor — the hybrid split's cheap side (§3)
# ---------------------------------------------------------------------------

class EmbedOnlyExtractor(TransformComponent):
    """A no-op kg_extractor for the embeddings-only path.

    PropertyGraphIndex._insert_nodes asserts every node carries KG node/relation
    metadata keys (normally set by an extractor), so an EMPTY kg_extractors list is
    not allowed — it would AssertionError. This stamps empty lists: the node is
    embedded and stored as a vector-searchable Chunk but contributes no
    entities/relations, which is exactly the embed-everything / extract-only-a-slice
    hybrid (§3/§10)."""

    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.metadata[KG_NODES_KEY] = []
            node.metadata[KG_RELATIONS_KEY] = []
        return nodes

    async def acall(self, nodes, **kwargs):
        return self.__call__(nodes, **kwargs)


# ---------------------------------------------------------------------------
# Neo4j PropertyGraphIndex setup (§5, §11)
# ---------------------------------------------------------------------------

@dataclass
class Stores:
    graph_store: Neo4jPropertyGraphStore
    extract_index: PropertyGraphIndex   # curated slice → LLM entity extraction
    embed_index: PropertyGraphIndex     # everything else → embeddings only


# Substrings that mark a hybrid-reasoning ("thinking") model. Used only by the "auto"
# thinking policy — not exhaustive, just the families likely to be used for extraction.
_THINKING_MODEL_HINTS = ("qwen3", "deepseek-r1", "deepseek-v3", "gpt-oss", "magistral", "-r1", "reasoning")


def _is_thinking_model(model: str) -> bool:
    name = model.lower()
    return any(hint in name for hint in _THINKING_MODEL_HINTS)


def _resolve_thinking(model: str, setting: str) -> bool | None:
    """Map the extract_thinking policy to the Ollama `thinking` arg.

    Returns False (disable) / True (enable) / None (omit the flag — the model default).
    None is important for non-thinking models (e.g. qwen2.5): sending think:false to a
    model that doesn't support thinking can error, so "auto" omits it for those.
    """
    setting = (setting or "auto").strip().lower()
    if setting == "on":
        return True
    if setting == "off":
        return False
    # auto: disable thinking for thinking-capable models, leave others untouched.
    return False if _is_thinking_model(model) else None


def init_stores(config: PipelineConfig) -> Stores:
    """Connect to Neo4j and build two PropertyGraphIndex views over the SAME store
    and storage context: one runs SimpleLLMPathExtractor (curated slice), the other
    runs the EmbedOnlyExtractor (rest). Routing the hybrid split by *index* — rather
    than mutating one index's private `_kg_extractors` between batches — keeps it on
    the public API. Neo4j persists server-side, so there is no explicit persist()."""
    if not config.neo4j_password:
        raise SystemExit(
            "NEO4J_PASSWORD is not set (env or .env). Refusing to connect to Neo4j."
        )

    graph_store = Neo4jPropertyGraphStore(
        username=config.neo4j_user,
        password=config.neo4j_password,
        url=config.neo4j_url,
        database=config.neo4j_database,
    )
    storage = StorageContext.from_defaults(property_graph_store=graph_store)

    thinking = _resolve_thinking(config.extract_model, config.extract_thinking)
    llm = Ollama(
        model=config.extract_model,
        base_url=config.ollama_host,
        request_timeout=float(config.insert_timeout) if config.insert_timeout else 600.0,
        thinking=thinking,
    )
    logger.info(
        f"Extraction LLM: {config.extract_model} "
        f"(thinking={'default' if thinking is None else thinking}; policy={config.extract_thinking})"
    )
    embed = HuggingFaceEmbedding(model_name=config.embedding_model)

    common = dict(
        property_graph_store=graph_store,
        embed_model=embed,
        storage_context=storage,
        show_progress=True,
    )
    extract_index = PropertyGraphIndex.from_existing(
        llm=llm,
        kg_extractors=[SimpleLLMPathExtractor(
            llm=llm,
            max_paths_per_chunk=config.max_paths_per_chunk,
            num_workers=config.extract_num_workers,
        )],
        **common,
    )
    embed_index = PropertyGraphIndex.from_existing(
        kg_extractors=[EmbedOnlyExtractor()],
        **common,
    )
    logger.info(
        f"Neo4j PropertyGraphIndex ready at {config.neo4j_url} (db={config.neo4j_database}); "
        f"extract_model={config.extract_model}, embed={config.embedding_model}"
    )
    return Stores(graph_store, extract_index, embed_index)


def reset_book_namespace(graph_store: Neo4jPropertyGraphStore) -> None:
    """--reset: purge ONLY the book namespace from Neo4j (source_type='book' or the
    'book::' id/ref_doc_id namespace), leaving notes untouched. Mirrors the note
    rebuild purge (§7) scoped to books. Replaces the old SurrealKV file delete."""
    graph_store.structured_query(
        """
        MATCH (n)
        WHERE n.source_type = 'book'
           OR n.id STARTS WITH 'book::'
           OR n.ref_doc_id STARTS WITH 'book::'
        DETACH DELETE n
        """
    )
    logger.warning("--reset: purged the book namespace (source_type='book') from Neo4j")


# ---------------------------------------------------------------------------
# Preflight & idempotency guards
# ---------------------------------------------------------------------------

# WHERE clause identifying the book namespace, shared by the purge and the count so
# the idempotency guard sees exactly what --reset would delete.
_BOOK_NS_WHERE = (
    "n.source_type = 'book' OR n.id STARTS WITH 'book::' "
    "OR n.ref_doc_id STARTS WITH 'book::'"
)


def count_book_nodes(graph_store: Neo4jPropertyGraphStore) -> int:
    """How many nodes already live in the book namespace (see reset_book_namespace)."""
    rows = graph_store.structured_query(
        f"MATCH (n) WHERE {_BOOK_NS_WHERE} RETURN count(n) AS c"
    )
    return rows[0]["c"] if rows else 0


def _check_ollama(config: PipelineConfig) -> None:
    """Fail fast if Ollama is unreachable or the extraction model isn't pulled —
    otherwise this only surfaces after hours of extraction, at the first insert."""
    tags_url = config.ollama_host.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=5) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"Preflight: Ollama unreachable at {config.ollama_host} ({exc}). "
            f"Start it (`ollama serve`) or set OLLAMA_HOST. Use --skip-preflight to bypass."
        )
    names = {m.get("name", "") for m in payload.get("models", [])}
    # Ollama tags carry a ':tag' suffix; accept an exact match or a bare-name match.
    bare = {n.split(":", 1)[0] for n in names}
    # Check the EXTRACTION model — that's the one the insert half actually calls.
    model = config.extract_model
    if model not in names and model.split(":", 1)[0] not in bare:
        raise SystemExit(
            f"Preflight: extraction model '{model}' is not pulled in Ollama "
            f"(have: {sorted(names) or 'none'}). Run `ollama pull {model}`. "
            f"Use --skip-preflight to bypass."
        )
    logger.info(f"Preflight OK: Ollama reachable at {config.ollama_host}, extraction model '{model}' present")


def preflight(config: PipelineConfig, *, check_ollama: bool, check_library: bool) -> None:
    """Fail-fast checks before any expensive work. Neo4j reachability is already
    validated by init_stores; this covers the two gaps that bite late: a missing
    Ollama model (only hit at first insert) and an empty library."""
    if check_library:
        if not config.library_path.exists():
            raise SystemExit(f"Preflight: library_path does not exist: {config.library_path}")
        if not discover_files(config.library_path):
            raise SystemExit(
                f"Preflight: no PDF/EPUB files under {config.library_path}. "
                f"Use --skip-preflight to bypass."
            )
    if check_ollama:
        _check_ollama(config)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(config: PipelineConfig, reset: bool = False, from_checkpoint: bool = False, profile: bool = False, extract_only: bool = False, require_curated: bool = False, skip_preflight: bool = False) -> None:
    # Every run leaves an on-disk log (console logging alone is lost when the terminal
    # closes — the reason a silent overnight run once went undiagnosed).
    add_file_logging(config.working_dir)
    # Log the RESOLVED config so a wrong library/DB is visible up front (and persisted to
    # the file log), not silently inferred from bad results later.
    logger.info(
        f"Effective config: library_path={config.library_path} | working_dir={config.working_dir} "
        f"| neo4j_db={config.neo4j_database} @ {config.neo4j_url} | ollama_model={config.ollama_model}"
    )

    # --extract-only and --from-checkpoint are opposite halves of the split run and
    # cannot both apply: one does extraction-then-stop, the other skips extraction.
    if extract_only and from_checkpoint:
        logger.warning("--extract-only and --from-checkpoint are mutually exclusive; ignoring --from-checkpoint")
        from_checkpoint = False

    # Preflight fail-fast (before any expensive work). A fresh run needs a non-empty
    # library; any run that will insert needs the Ollama model pulled — otherwise that
    # only surfaces at the first insert, after extraction has already run.
    will_extract = not (from_checkpoint and checkpoint_path(config).exists())
    if skip_preflight:
        logger.warning("--skip-preflight: skipping Ollama/library preflight checks")
    else:
        preflight(config, check_ollama=not extract_only, check_library=will_extract)

    # Stage 1: connect to Neo4j and build the two index views. --extract-only stops
    # before insertion, so it needs no DB connection at all (Neo4j need not be up).
    stores = None if extract_only else init_stores(config)

    # Stage 0: (optional) purge the book namespace before re-ingesting.
    if reset:
        if extract_only:
            logger.warning("--reset has no effect with --extract-only (no DB is touched); ignoring")
        else:
            reset_book_namespace(stores.graph_store)

    if profile and Profiler is None:
        logger.warning("--profile requested but pipeline_profiler is unavailable (archived); continuing without profiling")
    profiler = Profiler(out_dir=str(config.working_dir / "profile_results")) if (profile and Profiler is not None) else None

    def stage(name: str, **meta):
        return profiler.stage(name, **meta) if profiler else contextlib.nullcontext()

    # Stage 2–4: obtain the deduped chunk list (from checkpoint, or fresh).
    if from_checkpoint and checkpoint_path(config).exists():
        chunks = load_chunks_checkpoint(config)
    else:
        if from_checkpoint:
            logger.warning(f"--from-checkpoint set but no checkpoint found at {checkpoint_path(config)}; running full pipeline")

        files = discover_files(config.library_path)
        if not files:
            logger.warning(f"No PDF/EPUB files found under {config.library_path}")
            return

        # Per-book raw cache (Option A crash resilience): split discovered files into
        # those already parsed by a prior (possibly crashed) run and those still needing
        # extraction. Only missing/stale books are re-parsed; each is cached the moment
        # its worker returns, so a crash mid-extraction loses at most the in-flight book.
        all_chunks: list[Chunk] = []
        to_parse: list[Path] = []
        for f in files:
            cached = load_raw_book(f, config)
            if cached is None:
                to_parse.append(f)
            else:
                all_chunks.extend(cached)
        if all_chunks:
            logger.info(
                f"Resuming from raw cache: {len(all_chunks)} chunk(s) from "
                f"{len(files) - len(to_parse)} cached book(s); {len(to_parse)} to parse"
            )

        # Parallel extraction (CPU-bound → multiprocessing).
        # NOTE: worker count is capped well below cpu_count() — each worker runs a full
        # docling+EasyOCR pipeline that can need several GB for large PDFs, so a high
        # process count risks OOM (workers silently drop documents via MemoryError,
        # producing a near-empty result set instead of a clean failure).
        if to_parse:
            logger.info(f"Extracting {len(to_parse)} documents using {config.extraction_workers} workers...")
            with stage("extraction", n_files=len(to_parse)):
                with make_progress() as progress, Pool(processes=config.extraction_workers) as pool:
                    task = progress.add_task("Extracting documents", total=len(to_parse))
                    for source, doc_chunks in pool.imap_unordered(extract_document, to_parse):
                        # Cache before extending: persist each book's work before we
                        # could crash on a later one. Skip empty results (extraction
                        # failure) so a transient OOM is retried, not cached as done.
                        if doc_chunks:
                            save_raw_book(source, doc_chunks, config)
                        all_chunks.extend(doc_chunks)
                        progress.update(task, description=f"Extracted {Path(source).name}")
                        progress.advance(task)

        logger.info(f"Extraction complete: {len(all_chunks)} raw chunks")

        # Dropped-document detection: books that yielded 0 chunks — an extraction
        # failure or a silent worker OOM (see the multiprocessing note above). With the
        # full source path now on every chunk, these are exactly the discovered files
        # absent from the produced set. Warn loudly rather than let a near-empty corpus
        # sail through as if it were complete.
        produced = {c.source_path for c in all_chunks}
        dropped = [f for f in files if str(f) not in produced]
        if dropped:
            logger.warning(
                f"{len(dropped)}/{len(files)} document(s) produced 0 chunks "
                f"(extraction failure or OOM): {[f.name for f in dropped]}"
            )

        # Deduplication
        with stage("deduplication", n_chunks=len(all_chunks)):
            chunks = deduplicate(
                all_chunks,
                threshold=config.dedup_threshold,
                num_perm=config.dedup_num_perm,
                k=config.dedup_shingle_k,
            )

        # Checkpoint: persist the post-dedup chunk list before insertion begins
        save_chunks_checkpoint(chunks, config)

    # --extract-only: chunks are now parsed, deduped, and on disk. Stop before the
    # embed/insert half — resume it later (e.g. another night) with --from-checkpoint.
    if extract_only:
        logger.info(f"Extract-only complete: {len(chunks)} chunks checkpointed to "
                    f"{checkpoint_path(config)}; stopping before embedding/insertion.")
        if profiler:
            profiler.report()
        return

    # Load any partial-insertion progress (empty on a fresh run). A --reset wipes the
    # namespace, so it must also discard stale progress and start from a clean slate.
    if reset:
        clear_extract_progress(config)
    progress = load_extract_progress(config)
    resuming = bool(progress["embed"] or progress["curated"])

    # Idempotency + resume guard (§ delete_ref_doc blocker): TextNode ids are random per
    # run, so inserting into an already-populated namespace without --reset DUPLICATES
    # nodes (Neo4j MERGE can't dedup them cross-run), and delete_ref_doc can't purge the
    # directly-upserted graph nodes afterward. Refuse UNLESS either --reset ran (namespace
    # already empty) or a matching progress file marks this as a resume of a partial run.
    if not reset:
        existing = count_book_nodes(stores.graph_store)
        if existing and not resuming:
            raise SystemExit(
                f"Refusing to insert: {existing} node(s) already exist in the book namespace "
                f"and no matching in-progress checkpoint was found. Re-ingesting without --reset "
                f"would duplicate them (random node ids → no cross-run MERGE). Re-run with --reset "
                f"to purge first."
            )
        if existing and resuming:
            logger.warning(
                f"Resuming a partial insertion (no --reset): "
                f"{len(progress['embed'])} embed + {len(progress['curated'])} curated chunk(s) "
                f"already committed; continuing with the unfinished tail."
            )

    # Stage 5: hybrid split by curated-slice membership. Curated books get full LLM
    # entity extraction; everything else is embeddings-only but still vector-searchable.
    # Split by CHUNK (not pre-built nodes) so the resumable inserter can rebuild each
    # batch's nodes and key progress on a stable content hash.
    curated_chunks: list[Chunk] = []
    embed_chunks: list[Chunk] = []
    curated_books: set[str] = set()
    embed_books: set[str] = set()
    for c in chunks:
        stem = Path(c.source_path).stem
        if is_curated(c.source_path):
            curated_chunks.append(c)
            curated_books.add(stem)
        else:
            embed_chunks.append(c)
            embed_books.add(stem)

    logger.info(
        f"Hybrid split: {len(curated_chunks)} curated chunk(s) from {len(curated_books)} book(s) "
        f"→ LLM extraction; {len(embed_chunks)} chunk(s) from {len(embed_books)} book(s) → embeddings-only"
    )

    # Silent-no-op guard: if NOTHING matched the curated slice, the graph gets
    # embeddings only and ZERO entities/relations — a run that "succeeds" while
    # writing no graph. That exact mismatch (library filenames vs classify.BOOK_LABELS)
    # cost a full overnight run once. Warn loudly rather than fail, since an
    # embeddings-only corpus is a legitimate run; escalate to a hard stop with --require-curated.
    if not curated_chunks:
        msg = (
            "No books matched the curated slice (classify.BOOK_LABELS) — the graph will "
            "have embeddings ONLY, 0 entities and 0 relations. If you expected entity "
            "extraction, check that library filenames' stems exactly match a BOOK_LABELS "
            "title (matching is exact: 'clean_code' != 'Clean Code'). "
            f"Books seen: {sorted(embed_books)}"
        )
        if require_curated:
            raise SystemExit(f"--require-curated: {msg}")
        logger.warning(msg)

    # Resumable, checkpointed insertion: embed-only first (fast, no LLM), then the curated
    # slice (slow per-chunk LLM extraction). Each commits in batches and records progress,
    # so a crash resumes from the last committed batch instead of restarting.
    if embed_chunks:
        with stage("embed_only_insertion", n_chunks=len(embed_chunks)):
            n = insert_chunks_batched(
                stores.embed_index, embed_chunks, progress, "embed", config,
                label="Embeddings-only insert",
            )
            logger.info(f"Embeddings-only insertion complete ({n} node(s) inserted this run).")
    if curated_chunks:
        with stage("curated_extraction_insertion", n_chunks=len(curated_chunks)):
            n = insert_chunks_batched(
                stores.extract_index, curated_chunks, progress, "curated", config,
                label="Curated LLM extraction",
            )
            logger.info(f"Curated extraction complete ({n} node(s) inserted this run).")

    # Clean completion → drop the progress file so the next run doesn't try to resume.
    clear_extract_progress(config)
    logger.info("Ingestion pipeline complete.")
    if profiler:
        profiler.report()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG Assistant ingestion pipeline (LlamaIndex + Neo4j)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--reset", action="store_true", help="Purge the book namespace in Neo4j before ingesting")
    parser.add_argument(
        "--from-checkpoint",
        action="store_true",
        help="Skip discovery/extraction/dedup and reuse the last saved chunks_checkpoint.json",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Parse + dedup + write chunks_checkpoint.json, then stop before embedding/insertion "
             "(no Neo4j needed). Resume the insert half later with --from-checkpoint.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-stage timing via pipeline_profiler (writes to working_dir/profile_results)",
    )
    parser.add_argument(
        "--require-curated",
        action="store_true",
        help="Abort (instead of warn) if no library book matches the curated slice, so a "
             "run that was meant to extract entities can't silently finish embeddings-only.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the Ollama-model / non-empty-library preflight checks.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(
        config,
        reset=args.reset,
        from_checkpoint=args.from_checkpoint,
        profile=args.profile,
        extract_only=args.extract_only,
        require_curated=args.require_curated,
        skip_preflight=args.skip_preflight,
    )


if __name__ == "__main__":
    main()
