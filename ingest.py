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


def load_config(path: str | None) -> PipelineConfig:
    defaults: dict[str, Any] = {
        "library_path": os.getenv("LIBRARY_PATH", "./library"),
        "working_dir":  os.getenv("WORKING_DIR", os.getenv("LIGHTRAG_WORKING_DIR", "./lightrag_data")),
        "extraction_workers": int(os.getenv("EXTRACTION_WORKERS", "8")),
        "max_concurrent_inserts": int(os.getenv("MAX_CONCURRENT_INSERTS", "4")),
        "insert_timeout": int(os.getenv("INSERT_TIMEOUT", "300")),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "ollama_host":  os.getenv("OLLAMA_HOST",  "http://localhost:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "qwen2.5:14b"),
        "vector_dim":   int(os.getenv("VECTOR_DIM", os.getenv("SURREALDB_VECTOR_DIM", "384"))),
        "neo4j_url":      os.getenv("NEO4J_URL", "neo4j://127.0.0.1:7687"),
        "neo4j_user":     os.getenv("NEO4J_USER", "neo4j"),
        "neo4j_password": os.getenv("NEO4J_PASSWORD", ""),
        "neo4j_database": os.getenv("NEO4J_DATABASE", "llamaindex"),
        "max_paths_per_chunk": int(os.getenv("MAX_PATHS_PER_CHUNK", "10")),
        "extract_num_workers": int(os.getenv("EXTRACT_NUM_WORKERS", "4")),
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
        neo4j_url=defaults["neo4j_url"],
        neo4j_user=defaults["neo4j_user"],
        neo4j_password=defaults["neo4j_password"],
        neo4j_database=defaults["neo4j_database"],
        max_paths_per_chunk=defaults["max_paths_per_chunk"],
        extract_num_workers=defaults["extract_num_workers"],
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

    llm = Ollama(
        model=config.ollama_model,
        base_url=config.ollama_host,
        request_timeout=float(config.insert_timeout) if config.insert_timeout else 600.0,
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
        f"LLM={config.ollama_model}, embed={config.embedding_model}"
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
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(config: PipelineConfig, reset: bool = False, from_checkpoint: bool = False, profile: bool = False, extract_only: bool = False) -> None:
    # --extract-only and --from-checkpoint are opposite halves of the split run and
    # cannot both apply: one does extraction-then-stop, the other skips extraction.
    if extract_only and from_checkpoint:
        logger.warning("--extract-only and --from-checkpoint are mutually exclusive; ignoring --from-checkpoint")
        from_checkpoint = False

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

        # Parallel extraction (CPU-bound → multiprocessing).
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

    # Stage 5: hybrid node build + insertion (§5).
    # Split chunks by curated-slice membership: curated books get full LLM entity
    # extraction; everything else is embeddings-only but still vector-searchable.
    curated_nodes: list[TextNode] = []
    embed_nodes: list[TextNode] = []
    curated_books: set[str] = set()
    embed_books: set[str] = set()
    for c in chunks:
        stem = Path(c.source_path).stem
        if is_curated(c.source_path):
            curated_nodes.extend(chunk_to_nodes(c))
            curated_books.add(stem)
        else:
            embed_nodes.extend(chunk_to_nodes(c))
            embed_books.add(stem)

    logger.info(
        f"Hybrid split: {len(curated_nodes)} node(s) from {len(curated_books)} curated book(s) "
        f"→ LLM extraction; {len(embed_nodes)} node(s) from {len(embed_books)} book(s) → embeddings-only"
    )

    # Embed-only first (fast, no LLM), then the curated slice (slow LLM extraction).
    if embed_nodes:
        with stage("embed_only_insertion", n_nodes=len(embed_nodes)):
            logger.info(f"Inserting {len(embed_nodes)} embeddings-only node(s)...")
            stores.embed_index.insert_nodes(embed_nodes)
    if curated_nodes:
        with stage("curated_extraction_insertion", n_nodes=len(curated_nodes)):
            logger.info(f"Extracting + inserting {len(curated_nodes)} curated-slice node(s) via LLM...")
            stores.extract_index.insert_nodes(curated_nodes)

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
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(
        config,
        reset=args.reset,
        from_checkpoint=args.from_checkpoint,
        profile=args.profile,
        extract_only=args.extract_only,
    )


if __name__ == "__main__":
    main()
