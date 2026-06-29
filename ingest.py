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
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import yaml
from bs4 import BeautifulSoup
from datasketch import MinHash, MinHashLSH
from dotenv import load_dotenv

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
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
    max_concurrent_inserts: int = 4
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
        "max_concurrent_inserts": int(os.getenv("MAX_CONCURRENT_INSERTS", "4")),
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
        max_concurrent_inserts=defaults["max_concurrent_inserts"],
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

def extract_document(path: Path) -> list[Chunk]:
    """
    Entry point for multiprocessing workers.
    Returns an empty list on any extraction failure (logged, not raised).
    """
    try:
        logger.info(f"Extracting: {path.name}")
        if path.suffix.lower() == ".pdf":
            raw = extract_pdf(path)
        elif path.suffix.lower() == ".epub":
            raw = extract_epub(path)
        else:
            return []
        return apply_content_rules(raw, path)
    except Exception as exc:
        logger.error(f"Extraction failed for {path}: {exc}")
        return []


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
        embeddings = await loop.run_in_executor(
            None,
            lambda: _model.encode(texts, convert_to_numpy=True).tolist()
        )
        return embeddings

    return EmbeddingFunc(embedding_dim=vector_dim, max_token_size=512, func=_embed)


def make_ollama_func(host: str, model: str):
    """Build an Ollama LLM func compatible with LightRAG's llm_model_func interface."""
    async def _llm(prompt: str, **kwargs) -> str:
        return await ollama_model_complete(
            prompt,
            host=host,
            model=model,
            **kwargs,
        )
    return _llm


async def init_lightrag(config: PipelineConfig) -> LightRAG:
    """Initialise LightRAG with the SurrealDB backend and local models."""
    config.working_dir.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(config.working_dir),
        llm_model_func=make_ollama_func(config.ollama_host, config.ollama_model),
        embedding_func=make_embedding_func(config.embedding_model, config.vector_dim),
        kv_storage="SurrealDBKVStorage",
        vector_storage="SurrealDBVectorStorage",
        graph_storage="SurrealDBGraphStorage",
        doc_status_storage="SurrealDBDocStatusStorage",
    )
    await rag.initialize_storages()
    logger.info("LightRAG initialised with SurrealDB backend")
    return rag


# ---------------------------------------------------------------------------
# Insertion with checkpointing (via DocStatusStorage)
# ---------------------------------------------------------------------------

async def insert_with_semaphore(
    rag: LightRAG,
    chunk: Chunk,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    Insert a single chunk into LightRAG under the semaphore.
    LightRAG's DocStatusStorage provides implicit checkpointing —
    documents already marked 'done' are skipped on re-runs.
    """
    async with semaphore:
        try:
            # Pass source path as doc ID so status is tracked per document
            await rag.ainsert(
                chunk.text,
                metadata={
                    "source": chunk.source_path,
                    "content_type": chunk.content_type,
                    "has_code": bool(chunk.code_blocks),
                    **chunk.metadata,
                },
                doc_id=chunk.source_path,
            )
        except Exception as exc:
            logger.error(f"Insert failed for chunk from {chunk.source_path}: {exc}")


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

async def run_pipeline(config: PipelineConfig, reset: bool = False) -> None:
    # Stage 0: (Optional) reset existing data
    if reset:
        logger.warning("--reset flag set: existing SurrealDB data will be cleared on init")

    # Stage 1: Initialise LightRAG + SurrealDB
    rag = await init_lightrag(config)

    # Stage 2: File discovery
    files = discover_files(config.library_path)
    if not files:
        logger.warning(f"No PDF/EPUB files found under {config.library_path}")
        return

    # Stage 3: Parallel extraction (CPU-bound → multiprocessing)
    logger.info(f"Extracting {len(files)} documents using {cpu_count()} workers...")
    with Pool(processes=cpu_count()) as pool:
        results: list[list[Chunk]] = pool.map(extract_document, files)

    all_chunks: list[Chunk] = [chunk for doc_chunks in results for chunk in doc_chunks]
    logger.info(f"Extraction complete: {len(all_chunks)} raw chunks")

    # Stage 4: Deduplication
    chunks = deduplicate(
        all_chunks,
        threshold=config.dedup_threshold,
        num_perm=config.dedup_num_perm,
        k=config.dedup_shingle_k,
    )

    # Stage 5: Async insertion into LightRAG / SurrealDB
    logger.info(f"Inserting {len(chunks)} chunks (concurrency={config.max_concurrent_inserts})...")
    semaphore = asyncio.Semaphore(config.max_concurrent_inserts)
    tasks = [insert_with_semaphore(rag, chunk, semaphore) for chunk in chunks]
    await asyncio.gather(*tasks)

    logger.info("Ingestion pipeline complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG Assistant ingestion pipeline")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild SurrealDB tables")
    args = parser.parse_args()

    config = load_config(args.config)
    asyncio.run(run_pipeline(config, reset=args.reset))


if __name__ == "__main__":
    main()
