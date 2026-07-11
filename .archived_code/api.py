# =============================================================================
# ⚠️ ARCHIVED — LightRAG/SurrealDB era, NON-FUNCTIONAL. Do not run.
#
# This was the query/serving bridge for the pre-migration stack. It no longer
# imports (`from ingest import make_embedding_func` was removed in the migration)
# and targets LightRAG + the SurrealDB storage adapter, both retired in the move
# to LlamaIndex + Neo4j.
#
# Kept as a REFERENCE for the eventual query interface — see Migration_LlamaIndex.md
# §8 "GraphNotes query interface (later)". That rewrite mirrors this file's role
# (OpenWebUI-compatible endpoints, NDJSON streaming, conversation history, Ollama/
# Anthropic provider routing) but replaces the retrieval guts with PropertyGraphIndex
# retrievers (LLMSynonymRetriever / VectorContextRetriever / graph traversal) over Neo4j.
# =============================================================================
"""
api.py

GraphRAG Assistant — FastAPI Bridge

Exposes an Ollama-compatible REST API so OpenWebUI can connect without
any special configuration. Query modes (local, global, hybrid, naive, mix)
are surfaced as selectable 'models' in OpenWebUI's model picker.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 11435 --reload

Environment variables (can also live in a .env file):
    SURREALDB_URL, SURREALDB_NAMESPACE, SURREALDB_DATABASE,
    SURREALDB_USERNAME, SURREALDB_PASSWORD, SURREALDB_VECTOR_DIM,
    LIGHTRAG_WORKING_DIR, OLLAMA_HOST, OLLAMA_MODEL,
    QUERY_LLM_PROVIDER   ('ollama' | 'anthropic')
    ANTHROPIC_API_KEY    (required if QUERY_LLM_PROVIDER=anthropic)
    ANTHROPIC_MODEL      (default: claude-haiku-4-5-20251001)
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import anthropic
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_model_complete

from ingest import make_embedding_func  # reuse embedding func from pipeline

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WORKING_DIR         = os.getenv("LIGHTRAG_WORKING_DIR",  "./lightrag_data")
OLLAMA_HOST         = os.getenv("OLLAMA_HOST",           "http://localhost:11434")
OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL",          "qwen2.5:14b")
VECTOR_DIM          = int(os.getenv("SURREALDB_VECTOR_DIM", "384"))
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL",       "all-MiniLM-L6-v2")
QUERY_LLM_PROVIDER  = os.getenv("QUERY_LLM_PROVIDER",   "ollama")  # 'ollama' | 'anthropic'
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY",     "")
ANTHROPIC_MODEL     = os.getenv("ANTHROPIC_MODEL",       "claude-haiku-4-5-20251001")

QUERY_MODES = ["mix", "hybrid", "local", "global", "naive"]


# ---------------------------------------------------------------------------
# Pydantic request/response models (Ollama-compatible schema)
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "mix"
    messages: list[Message]
    stream: bool = True


class GenerateRequest(BaseModel):
    model: str = "mix"
    prompt: str
    stream: bool = True


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

async def _query_ollama(prompt: str) -> AsyncGenerator[str, None]:
    """Stream a response from Ollama using the OpenAI-compatible endpoint."""
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break


async def _query_anthropic(prompt: str) -> AsyncGenerator[str, None]:
    """Stream a response from Claude via the Anthropic API."""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    async with client.messages.stream(
        model=ANTHROPIC_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def stream_llm_response(prompt: str) -> AsyncGenerator[str, None]:
    """Route to the configured query-time LLM provider."""
    if QUERY_LLM_PROVIDER == "anthropic":
        async for token in _query_anthropic(prompt):
            yield token
    else:
        async for token in _query_ollama(prompt):
            yield token


# ---------------------------------------------------------------------------
# LightRAG initialisation (shared app state)
# ---------------------------------------------------------------------------

_rag: LightRAG | None = None


async def get_rag() -> LightRAG:
    if _rag is None:
        raise HTTPException(status_code=503, detail="LightRAG not yet initialised")
    return _rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise LightRAG on startup; shut down on exit."""
    global _rag
    logger.info("Initialising LightRAG with SurrealDB backend...")

    os.makedirs(WORKING_DIR, exist_ok=True)

    _rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=_make_ollama_func(),
        embedding_func=make_embedding_func(EMBEDDING_MODEL, VECTOR_DIM),
        kv_storage="SurrealDBKVStorage",
        vector_storage="SurrealDBVectorStorage",
        graph_storage="SurrealDBGraphStorage",
        doc_status_storage="SurrealDBDocStatusStorage",
    )
    await _rag.initialize_storages()
    logger.info(f"LightRAG ready. Query LLM: {QUERY_LLM_PROVIDER.upper()}")

    yield  # app runs here

    logger.info("Shutting down LightRAG...")
    # Storage finalization is handled internally by LightRAG


def _make_ollama_func():
    """Ollama LLM func used for ingestion-phase roles (EXTRACT, KEYWORDS)."""
    async def _fn(prompt: str, **kwargs) -> str:
        return await ollama_model_complete(prompt, host=OLLAMA_HOST, model=OLLAMA_MODEL, **kwargs)
    return _fn


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GraphRAG Assistant API",
    description="Ollama-compatible bridge over LightRAG + SurrealDB",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_chunk(content: str, done: bool = False) -> bytes:
    """Format a streaming chunk in Ollama NDJSON format."""
    payload = {
        "model": "graphrag",
        "created_at": "",
        "message": {"role": "assistant", "content": content},
        "done": done,
    }
    return (json.dumps(payload) + "\n").encode()


def _build_query(messages: list[Message]) -> tuple[str, list[dict]]:
    """
    Extract the latest user query and format prior turns as conversation history.
    Returns (query_string, history_list).
    """
    if not messages:
        raise HTTPException(status_code=400, detail="messages list is empty")
    query = messages[-1].content
    history = [{"role": m.role, "content": m.content} for m in messages[:-1]]
    return query, history


def _validate_mode(mode: str) -> str:
    """Normalise and validate the query mode. Falls back to 'mix'."""
    mode = mode.strip().lower()
    return mode if mode in QUERY_MODES else "mix"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "provider": QUERY_LLM_PROVIDER}


@app.get("/api/tags")
async def tags() -> dict[str, list[dict[str, Any]]]:
    """
    Return available 'models' (= LightRAG query modes) to OpenWebUI.
    OpenWebUI's model picker populates from this endpoint.
    """
    now = int(time.time())
    return {
        "models": [
            {
                "name": mode,
                "modified_at": now,
                "size": 0,
                "digest": mode,
                "details": {
                    "format": "graphrag",
                    "family": "lightrag",
                    "parameter_size": mode,
                    "quantization_level": "none",
                },
            }
            for mode in QUERY_MODES
        ]
    }


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """
    Ollama-compatible /api/chat endpoint.
    OpenWebUI passes the selected 'model' name as the query mode.
    Conversation history is forwarded to LightRAG on each turn.
    """
    rag = await get_rag()
    query, history = _build_query(request.messages)
    mode = _validate_mode(request.model)

    logger.info(f"[chat] mode={mode} query={query[:80]!r}")

    async def _stream() -> AsyncGenerator[bytes, None]:
        # 1. Retrieve context from LightRAG (graph + vector retrieval).
        #    Conversation history is passed into retrieval (not just the final
        #    prompt) so follow-up turns resolve pronouns/ellipsis against prior
        #    context when selecting graph nodes and vector chunks (FR-F-02).
        try:
            context: str = await rag.aquery(
                query,
                param=QueryParam(
                    mode=mode,
                    conversation_history=history,
                    history_turns=3,
                ),
            )
        except Exception as exc:
            logger.error(f"LightRAG retrieval error: {exc}")
            yield _ollama_chunk(f"[Retrieval error: {exc}]", done=True)
            return

        # 2. Build grounded prompt with conversation history
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:]
        )
        prompt = (
            f"You are a knowledgeable assistant. Answer based on the context below.\n\n"
            f"Context:\n{context}\n\n"
            + (f"Conversation history:\n{history_text}\n\n" if history_text else "")
            + f"User: {query}\nAssistant:"
        )

        # 3. Stream LLM response token by token
        try:
            async for token in stream_llm_response(prompt):
                yield _ollama_chunk(token)
            yield _ollama_chunk("", done=True)
        except Exception as exc:
            logger.error(f"LLM streaming error: {exc}")
            yield _ollama_chunk(f"[LLM error: {exc}]", done=True)

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


@app.post("/api/generate")
async def generate(request: GenerateRequest) -> StreamingResponse:
    """
    Ollama-compatible /api/generate endpoint (single-turn, no history).
    Useful for quick testing outside of OpenWebUI.
    """
    rag = await get_rag()
    mode = _validate_mode(request.model)
    query = request.prompt

    logger.info(f"[generate] mode={mode} query={query[:80]!r}")

    async def _stream() -> AsyncGenerator[bytes, None]:
        try:
            context: str = await rag.aquery(query, param=QueryParam(mode=mode))
        except Exception as exc:
            yield _ollama_chunk(f"[Retrieval error: {exc}]", done=True)
            return

        prompt = (
            f"Answer the following question using only the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
        try:
            async for token in stream_llm_response(prompt):
                yield _ollama_chunk(token)
            yield _ollama_chunk("", done=True)
        except Exception as exc:
            yield _ollama_chunk(f"[LLM error: {exc}]", done=True)

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


@app.get("/graph")
async def graph_data() -> dict[str, Any]:
    """
    Return graph visualization data compatible with LightRAG's built-in WebUI.
    Optional endpoint — used if you run LightRAG's frontend separately.
    """
    rag = await get_rag()
    try:
        nodes = await rag.graph_storage.get_all_nodes()
        edges = await rag.graph_storage.get_all_edges()
        return {
            "nodes": [{"id": n["id"], "label": n.get("name", n["id"]), **n} for n in nodes],
            "edges": [{"from": e["src_id"], "to": e["tgt_id"], **e} for e in edges],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
