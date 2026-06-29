"""
lightrag/kg/surrealdb_impl.py

SurrealDB storage adapter for LightRAG.
Implements BaseKVStorage, BaseVectorStorage, BaseGraphStorage,
and BaseDocStatusStorage using the SurrealDB Python SDK v2.x.

Register in lightrag/kg/__init__.py:
    STORAGE_IMPLEMENTATIONS["SurrealDBKVStorage"]        = "lightrag.kg.surrealdb_impl.SurrealDBKVStorage"
    STORAGE_IMPLEMENTATIONS["SurrealDBVectorStorage"]    = "lightrag.kg.surrealdb_impl.SurrealDBVectorStorage"
    STORAGE_IMPLEMENTATIONS["SurrealDBGraphStorage"]     = "lightrag.kg.surrealdb_impl.SurrealDBGraphStorage"
    STORAGE_IMPLEMENTATIONS["SurrealDBDocStatusStorage"] = "lightrag.kg.surrealdb_impl.SurrealDBDocStatusStorage"
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

from surrealdb import Surreal

from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    BaseDocStatusStorage,
    DocStatus,
)
from lightrag.utils import logger


# ---------------------------------------------------------------------------
# Shared connection pool
# ---------------------------------------------------------------------------

class SurrealDBDB:
    """
    Manages a single async SurrealDB connection per LightRAG workspace.
    All four storage classes share one instance of this via the storage
    config dict (keyed as '_connection'), following the postgres_impl.py
    singleton pattern.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.url       = config.get("url",       os.getenv("SURREALDB_URL",       "ws://localhost:8000/rpc"))
        self.namespace = config.get("namespace", os.getenv("SURREALDB_NAMESPACE", "lightrag"))
        self.database  = config.get("database",  os.getenv("SURREALDB_DATABASE",  "assistant"))
        self.username  = config.get("username",  os.getenv("SURREALDB_USERNAME",  "root"))
        self.password  = config.get("password",  os.getenv("SURREALDB_PASSWORD",  "root"))
        self._client: Surreal | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        async with self._lock:
            if self._client is not None:
                return
            client = Surreal(self.url)
            await client.connect()
            await client.signin({"user": self.username, "pass": self.password})
            await client.use(self.namespace, self.database)
            self._client = client
            logger.info(f"SurrealDB connected: {self.url} / {self.namespace}.{self.database}")

    async def query(self, sql: str, vars: dict[str, Any] | None = None) -> list[Any]:
        if self._client is None:
            raise RuntimeError("SurrealDBDB.connect() must be called before query()")
        result = await self._client.query(sql, vars or {})
        # SDK returns list of result objects; unwrap the first result's data
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict) and "result" in first:
                return first["result"] or []
            return first if isinstance(first, list) else []
        return []

    async def close(self) -> None:
        async with self._lock:
            if self._client:
                await self._client.close()
                self._client = None


def _get_connection(config: dict[str, Any]) -> SurrealDBDB:
    """Return the shared SurrealDBDB instance, creating it on first call."""
    if "_connection" not in config:
        config["_connection"] = SurrealDBDB(config)
    return config["_connection"]


# ---------------------------------------------------------------------------
# KV Storage
# ---------------------------------------------------------------------------

@dataclass
class SurrealDBKVStorage(BaseKVStorage):
    """
    Key-value storage backed by a SurrealDB SCHEMAFULL table.
    Used by LightRAG for entity summaries, community reports,
    and LLM response caching.
    """

    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._db = _get_connection(self.config)
        # namespace comes from LightRAG workspace name
        self._table = f"kv_store_{self.namespace}"

    async def initialize(self) -> None:
        await self._db.connect()
        await self._db.query(f"""
            DEFINE TABLE IF NOT EXISTS {self._table} SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id    ON {self._table} TYPE string;
            DEFINE FIELD IF NOT EXISTS value ON {self._table} TYPE any;
            DEFINE INDEX IF NOT EXISTS idx_id ON {self._table} COLUMNS id UNIQUE;
        """)

    async def finalize(self) -> None:
        await self._db.close()

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        rows = await self._db.query(
            f"SELECT * FROM {self._table} WHERE id = $id LIMIT 1",
            {"id": id},
        )
        return rows[0] if rows else None

    async def get_by_ids(self, ids: list[str], fields: list[str] | None = None) -> list[dict[str, Any] | None]:
        if not ids:
            return []
        field_clause = ", ".join(fields) if fields else "*"
        rows = await self._db.query(
            f"SELECT {field_clause} FROM {self._table} WHERE id IN $ids",
            {"ids": ids},
        )
        # Build id→row map; preserve ordering and fill None for misses
        row_map = {r["id"]: r for r in rows}
        return [row_map.get(id_) for id_ in ids]

    async def filter_keys(self, data: dict[str, Any]) -> set[str]:
        """Return keys from `data` that do NOT already exist in the table."""
        if not data:
            return set()
        existing = await self._db.query(
            f"SELECT id FROM {self._table} WHERE id IN $ids",
            {"ids": list(data.keys())},
        )
        existing_ids = {r["id"] for r in existing}
        return set(data.keys()) - existing_ids

    async def upsert(self, data: dict[str, Any]) -> None:
        """Upsert a single record. `data` must contain an 'id' key."""
        if not data:
            return
        await self._db.query(
            f"UPSERT type::thing($table, $id) CONTENT $data",
            {"table": self._table, "id": data["id"], "data": data},
        )

    async def upsert_many(self, items: list[dict[str, Any]]) -> None:
        """Batch upsert. Each item must contain an 'id' key."""
        for item in items:
            await self.upsert(item)

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        await self._db.query(
            f"DELETE {self._table} WHERE id IN $ids",
            {"ids": ids},
        )

    async def drop(self) -> None:
        await self._db.query(f"REMOVE TABLE {self._table}")

    async def index_done_callback(self) -> None:
        pass  # No-op; SurrealDB writes are immediate


# ---------------------------------------------------------------------------
# Vector Storage
# ---------------------------------------------------------------------------

@dataclass
class SurrealDBVectorStorage(BaseVectorStorage):
    """
    Vector storage backed by SurrealDB's native HNSW index.
    Stores chunk text + embedding + metadata and supports
    approximate nearest-neighbour (ANN) similarity search.
    """

    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._db = _get_connection(self.config)
        self._table = f"vector_store_{self.namespace}"
        self._dim = int(os.getenv("SURREALDB_VECTOR_DIM", "384"))
        self._ef  = int(os.getenv("SURREALDB_HNSW_EF",   "64"))
        self._m   = int(os.getenv("SURREALDB_HNSW_M",    "16"))

    async def initialize(self) -> None:
        await self._db.connect()
        # Use IF NOT EXISTS so re-runs are idempotent
        await self._db.query(f"""
            DEFINE TABLE IF NOT EXISTS {self._table} SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id        ON {self._table} TYPE string;
            DEFINE FIELD IF NOT EXISTS content   ON {self._table} TYPE string;
            DEFINE FIELD IF NOT EXISTS embedding ON {self._table} TYPE array<float>;
            DEFINE FIELD IF NOT EXISTS metadata  ON {self._table} TYPE object;
            DEFINE INDEX IF NOT EXISTS hnsw_idx  ON {self._table}
                FIELDS embedding
                HNSW DIMENSION {self._dim} DIST COSINE
                EFC {self._ef} M {self._m};
        """)

    async def finalize(self) -> None:
        await self._db.close()

    async def upsert(self, data: dict[str, Any]) -> None:
        """
        Upsert a single vector record.
        Expected keys: id (str), content (str), embedding (list[float]), metadata (dict).
        """
        await self._db.query(
            f"UPSERT type::thing($table, $id) CONTENT $data",
            {"table": self._table, "id": data["id"], "data": data},
        )

    async def upsert_many(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            await self.upsert(item)

    async def query(
        self,
        query_embedding: list[float],
        top_k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        ANN similarity search using SurrealDB's HNSW <|k,ef|> operator.
        Returns up to top_k records sorted by descending cosine similarity.
        filter is an optional dict of metadata key/value pairs to pre-filter.
        """
        where_clauses = [f"embedding <|{top_k},{self._ef}|> $vec"]
        if filter:
            for k, v in filter.items():
                where_clauses.append(f"metadata.{k} = ${k}")

        where = " AND ".join(where_clauses)
        bind: dict[str, Any] = {"vec": query_embedding}
        if filter:
            bind.update(filter)

        rows = await self._db.query(
            f"SELECT id, content, metadata, "
            f"vector::similarity::cosine(embedding, $vec) AS score "
            f"FROM {self._table} WHERE {where} "
            f"ORDER BY score DESC LIMIT {top_k}",
            bind,
        )
        return rows

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        await self._db.query(
            f"DELETE {self._table} WHERE id IN $ids",
            {"ids": ids},
        )

    async def drop(self) -> None:
        await self._db.query(f"REMOVE TABLE {self._table}")

    async def index_done_callback(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Graph Storage
# ---------------------------------------------------------------------------

@dataclass
class SurrealDBGraphStorage(BaseGraphStorage):
    """
    Knowledge graph storage using two flat SurrealDB tables:
    one for entity nodes and one for relation edges.

    LightRAG addresses edges by string ID (src___tgt), not by record
    links, so a flat relation table mirrors the postgres_impl.py / AGE
    approach rather than using SurrealDB's native RELATE edges.
    """

    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._db = _get_connection(self.config)
        self._entity_table   = f"entity_{self.namespace}"
        self._relation_table = f"relation_{self.namespace}"

    async def initialize(self) -> None:
        await self._db.connect()
        await self._db.query(f"""
            DEFINE TABLE IF NOT EXISTS {self._entity_table} SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id          ON {self._entity_table} TYPE string;
            DEFINE FIELD IF NOT EXISTS name        ON {self._entity_table} TYPE string;
            DEFINE FIELD IF NOT EXISTS type        ON {self._entity_table} TYPE string;
            DEFINE FIELD IF NOT EXISTS description ON {self._entity_table} TYPE string;
            DEFINE FIELD IF NOT EXISTS source_id   ON {self._entity_table} TYPE string;
            DEFINE FIELD IF NOT EXISTS extra       ON {self._entity_table} TYPE object;
            DEFINE INDEX IF NOT EXISTS idx_entity_name ON {self._entity_table} COLUMNS name UNIQUE;

            DEFINE TABLE IF NOT EXISTS {self._relation_table} SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id          ON {self._relation_table} TYPE string;
            DEFINE FIELD IF NOT EXISTS src_id      ON {self._relation_table} TYPE string;
            DEFINE FIELD IF NOT EXISTS tgt_id      ON {self._relation_table} TYPE string;
            DEFINE FIELD IF NOT EXISTS weight      ON {self._relation_table} TYPE float;
            DEFINE FIELD IF NOT EXISTS description ON {self._relation_table} TYPE string;
            DEFINE FIELD IF NOT EXISTS keywords    ON {self._relation_table} TYPE array<string>;
            DEFINE FIELD IF NOT EXISTS source_id   ON {self._relation_table} TYPE string;
            DEFINE INDEX IF NOT EXISTS idx_relation_id ON {self._relation_table} COLUMNS id UNIQUE;
            DEFINE INDEX IF NOT EXISTS idx_relation_src ON {self._relation_table} COLUMNS src_id;
            DEFINE INDEX IF NOT EXISTS idx_relation_tgt ON {self._relation_table} COLUMNS tgt_id;
        """)

    async def finalize(self) -> None:
        await self._db.close()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    async def has_node(self, node_id: str) -> bool:
        rows = await self._db.query(
            f"SELECT id FROM {self._entity_table} WHERE id = $id LIMIT 1",
            {"id": node_id},
        )
        return bool(rows)

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        rows = await self._db.query(
            f"SELECT * FROM {self._entity_table} WHERE id = $id LIMIT 1",
            {"id": node_id},
        )
        return rows[0] if rows else None

    async def upsert_node(self, node_id: str, node_data: dict[str, Any]) -> None:
        payload = {"id": node_id, **node_data}
        await self._db.query(
            f"UPSERT type::thing($table, $id) CONTENT $data",
            {"table": self._entity_table, "id": node_id, "data": payload},
        )

    async def delete_node(self, node_id: str) -> None:
        await self._db.query(
            f"DELETE {self._entity_table} WHERE id = $id",
            {"id": node_id},
        )
        # Also remove any edges referencing this node
        await self._db.query(
            f"DELETE {self._relation_table} WHERE src_id = $id OR tgt_id = $id",
            {"id": node_id},
        )

    async def node_degree(self, node_id: str) -> int:
        rows = await self._db.query(
            f"SELECT count() AS cnt FROM {self._relation_table} "
            f"WHERE src_id = $id OR tgt_id = $id GROUP ALL",
            {"id": node_id},
        )
        return rows[0]["cnt"] if rows else 0

    async def get_all_nodes(self) -> list[dict[str, Any]]:
        return await self._db.query(f"SELECT * FROM {self._entity_table}")

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    async def has_edge(self, src_id: str, tgt_id: str) -> bool:
        edge_id = f"{src_id}___{tgt_id}"
        rows = await self._db.query(
            f"SELECT id FROM {self._relation_table} WHERE id = $id LIMIT 1",
            {"id": edge_id},
        )
        return bool(rows)

    async def get_edge(self, src_id: str, tgt_id: str) -> dict[str, Any] | None:
        edge_id = f"{src_id}___{tgt_id}"
        rows = await self._db.query(
            f"SELECT * FROM {self._relation_table} WHERE id = $id LIMIT 1",
            {"id": edge_id},
        )
        return rows[0] if rows else None

    async def upsert_edge(self, src_id: str, tgt_id: str, edge_data: dict[str, Any]) -> None:
        edge_id = f"{src_id}___{tgt_id}"
        payload = {"id": edge_id, "src_id": src_id, "tgt_id": tgt_id, **edge_data}
        await self._db.query(
            f"UPSERT type::thing($table, $id) CONTENT $data",
            {"table": self._relation_table, "id": edge_id, "data": payload},
        )

    async def delete_edge(self, src_id: str, tgt_id: str) -> None:
        edge_id = f"{src_id}___{tgt_id}"
        await self._db.query(
            f"DELETE {self._relation_table} WHERE id = $id",
            {"id": edge_id},
        )

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Sum of both node degrees — used by LightRAG for edge weight scaling."""
        return await self.node_degree(src_id) + await self.node_degree(tgt_id)

    async def get_node_edges(self, node_id: str) -> list[tuple[str, str]]:
        """Return all (src_id, tgt_id) pairs where this node is src or tgt."""
        rows = await self._db.query(
            f"SELECT src_id, tgt_id FROM {self._relation_table} "
            f"WHERE src_id = $id OR tgt_id = $id",
            {"id": node_id},
        )
        return [(r["src_id"], r["tgt_id"]) for r in rows]

    async def get_all_edges(self) -> list[dict[str, Any]]:
        return await self._db.query(f"SELECT * FROM {self._relation_table}")

    async def drop(self) -> None:
        await self._db.query(f"REMOVE TABLE {self._entity_table}")
        await self._db.query(f"REMOVE TABLE {self._relation_table}")

    async def index_done_callback(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Doc Status Storage
# ---------------------------------------------------------------------------

@dataclass
class SurrealDBDocStatusStorage(BaseDocStatusStorage):
    """
    Tracks per-document ingestion state.
    Status values mirror LightRAG's DocStatus enum:
    pending | processing | done | failed
    """

    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._db = _get_connection(self.config)
        self._table = f"doc_status_{self.namespace}"

    async def initialize(self) -> None:
        await self._db.connect()
        await self._db.query(f"""
            DEFINE TABLE IF NOT EXISTS {self._table} SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id         ON {self._table} TYPE string;
            DEFINE FIELD IF NOT EXISTS status     ON {self._table} TYPE string;
            DEFINE FIELD IF NOT EXISTS content_summary ON {self._table} TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS content_length  ON {self._table} TYPE option<int>;
            DEFINE FIELD IF NOT EXISTS chunks_count    ON {self._table} TYPE option<int>;
            DEFINE FIELD IF NOT EXISTS created_at ON {self._table} TYPE datetime DEFAULT time::now();
            DEFINE FIELD IF NOT EXISTS updated_at ON {self._table} TYPE datetime DEFAULT time::now();
            DEFINE INDEX IF NOT EXISTS idx_doc_id     ON {self._table} COLUMNS id UNIQUE;
            DEFINE INDEX IF NOT EXISTS idx_doc_status ON {self._table} COLUMNS status;
        """)

    async def finalize(self) -> None:
        await self._db.close()

    async def get_status(self, doc_id: str) -> DocStatus | None:
        rows = await self._db.query(
            f"SELECT status FROM {self._table} WHERE id = $id LIMIT 1",
            {"id": doc_id},
        )
        if not rows:
            return None
        return DocStatus(rows[0]["status"])

    async def set_status(self, doc_id: str, status: DocStatus, **kwargs: Any) -> None:
        payload: dict[str, Any] = {
            "id": doc_id,
            "status": status.value,
            "updated_at": "time::now()",
            **kwargs,
        }
        await self._db.query(
            f"UPSERT type::thing($table, $id) MERGE $data",
            {"table": self._table, "id": doc_id, "data": payload},
        )

    async def get_docs_by_status(self, status: DocStatus) -> list[dict[str, Any]]:
        return await self._db.query(
            f"SELECT * FROM {self._table} WHERE status = $status",
            {"status": status.value},
        )

    async def get_status_counts(self) -> dict[str, int]:
        rows = await self._db.query(
            f"SELECT status, count() AS cnt FROM {self._table} GROUP BY status"
        )
        return {r["status"]: r["cnt"] for r in rows}

    async def filter_keys(self, doc_ids: list[str]) -> set[str]:
        """Return doc_ids whose status is NOT 'done' (i.e. need processing)."""
        if not doc_ids:
            return set()
        rows = await self._db.query(
            f"SELECT id FROM {self._table} WHERE id IN $ids AND status = 'done'",
            {"ids": doc_ids},
        )
        done_ids = {r["id"] for r in rows}
        return set(doc_ids) - done_ids

    async def drop(self) -> None:
        await self._db.query(f"REMOVE TABLE {self._table}")

    async def index_done_callback(self) -> None:
        pass
