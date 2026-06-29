"""
lightrag/kg/surrealdb_impl.py

SurrealDB storage adapter for LightRAG 1.5.4.
Implements BaseKVStorage, BaseVectorStorage, BaseGraphStorage,
and DocStatusStorage using the SurrealDB Python SDK v2.x (async).

Registered via patch_lightrag.py — do not edit lightrag source directly.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from surrealdb import Surreal

from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocStatus,
    DocStatusStorage,
    DocProcessingStatus,
    StorageNameSpace,
)
from lightrag.types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from lightrag.utils import EmbeddingFunc, logger


# ---------------------------------------------------------------------------
# Shared connection — one per LightRAG workspace
# ---------------------------------------------------------------------------

class SurrealDBDB:
    """
    Manages a single async SurrealDB connection shared across all four
    storage classes in one LightRAG workspace.
    """

    def __init__(self) -> None:
        self.url       = os.getenv("SURREALDB_URL",       "ws://localhost:8000/rpc")
        self.namespace = os.getenv("SURREALDB_NAMESPACE", "lightrag")
        self.database  = os.getenv("SURREALDB_DATABASE",  "assistant")
        self.username  = os.getenv("SURREALDB_USERNAME",  "root")
        self.password  = os.getenv("SURREALDB_PASSWORD",  "root")
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
            raise RuntimeError("Call connect() first")
        result = await self._client.query(sql, vars or {})
        # SDK v2 returns a list of statement results; unwrap the first one
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict) and "result" in first:
                return first["result"] or []
            if isinstance(first, list):
                return first
        return []

    async def close(self) -> None:
        async with self._lock:
            if self._client:
                await self._client.close()
                self._client = None


# One connection per workspace, keyed by (namespace, database)
_CONNECTIONS: dict[tuple[str, str], SurrealDBDB] = {}
_CONN_LOCK = asyncio.Lock()


async def get_connection() -> SurrealDBDB:
    ns  = os.getenv("SURREALDB_NAMESPACE", "lightrag")
    db  = os.getenv("SURREALDB_DATABASE",  "assistant")
    key = (ns, db)
    async with _CONN_LOCK:
        if key not in _CONNECTIONS:
            conn = SurrealDBDB()
            await conn.connect()
            _CONNECTIONS[key] = conn
        return _CONNECTIONS[key]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# KV Storage
# ---------------------------------------------------------------------------

@dataclass
class SurrealDBKVStorage(BaseKVStorage):
    """
    Key-value storage for LightRAG entity summaries, community reports,
    and LLM response cache.
    """

    def __post_init__(self) -> None:
        self._table = f"kv_{self.namespace}"
        self._db: SurrealDBDB | None = None

    async def initialize(self) -> None:
        self._db = await get_connection()
        await self._db.query(f"""
            DEFINE TABLE IF NOT EXISTS {self._table} SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id    ON {self._table} TYPE string;
            DEFINE FIELD IF NOT EXISTS data  ON {self._table} FLEXIBLE TYPE object;
            DEFINE INDEX IF NOT EXISTS idx_kv_id ON {self._table} COLUMNS id UNIQUE;
        """)

    async def finalize(self) -> None:
        pass  # connection is shared; closed globally

    async def index_done_callback(self) -> None:
        pass

    async def drop(self) -> dict[str, str]:
        try:
            await self._db.query(f"REMOVE TABLE {self._table}")
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        rows = await self._db.query(
            f"SELECT data FROM {self._table} WHERE id = $id LIMIT 1", {"id": id}
        )
        return rows[0]["data"] if rows else None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        rows = await self._db.query(
            f"SELECT id, data FROM {self._table} WHERE id IN $ids", {"ids": ids}
        )
        row_map = {r["id"]: r["data"] for r in rows}
        return [row_map.get(i) for i in ids]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        rows = await self._db.query(
            f"SELECT id FROM {self._table} WHERE id IN $ids", {"ids": list(keys)}
        )
        existing = {r["id"] for r in rows}
        return keys - existing

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        # LightRAG passes {id: {fields...}, id2: {fields2...}, ...}
        for record_id, record_data in data.items():
            await self._db.query(
                f"UPSERT type::thing($t, $id) CONTENT {{id: $id, data: $data}}",
                {"t": self._table, "id": record_id, "data": record_data},
            )

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        await self._db.query(
            f"DELETE {self._table} WHERE id IN $ids", {"ids": ids}
        )

    async def is_empty(self) -> bool:
        rows = await self._db.query(f"SELECT count() AS cnt FROM {self._table} GROUP ALL")
        return (rows[0]["cnt"] if rows else 0) == 0


# ---------------------------------------------------------------------------
# Vector Storage
# ---------------------------------------------------------------------------

@dataclass
class SurrealDBVectorStorage(BaseVectorStorage):
    """
    Vector storage backed by SurrealDB's native HNSW index.
    """

    def __post_init__(self) -> None:
        self._validate_embedding_func()
        self._table = f"vec_{self.namespace}"
        self._dim = int(os.getenv("SURREALDB_VECTOR_DIM", "384"))
        self._ef  = int(os.getenv("SURREALDB_HNSW_EF",   "64"))
        self._m   = int(os.getenv("SURREALDB_HNSW_M",    "16"))
        self._db: SurrealDBDB | None = None

    async def initialize(self) -> None:
        self._db = await get_connection()
        await self._db.query(f"""
            DEFINE TABLE IF NOT EXISTS {self._table} SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id        ON {self._table} TYPE string;
            DEFINE FIELD IF NOT EXISTS content   ON {self._table} TYPE string;
            DEFINE FIELD IF NOT EXISTS embedding ON {self._table} TYPE array<float>;
            DEFINE FIELD IF NOT EXISTS metadata  ON {self._table} FLEXIBLE TYPE object;
            DEFINE INDEX IF NOT EXISTS hnsw_idx  ON {self._table}
                FIELDS embedding
                HNSW DIMENSION {self._dim} DIST COSINE
                EFC {self._ef} M {self._m};
        """)

    async def finalize(self) -> None:
        pass

    async def index_done_callback(self) -> None:
        pass

    async def drop(self) -> dict[str, str]:
        try:
            await self._db.query(f"REMOVE TABLE {self._table}")
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        # LightRAG passes {id: {content, embedding, ...metadata}}
        for record_id, record_data in data.items():
            embedding = record_data.get("embedding", [])
            content   = record_data.get("content", "")
            metadata  = {k: v for k, v in record_data.items()
                         if k not in ("embedding", "content")}
            await self._db.query(
                f"UPSERT type::thing($t, $id) CONTENT "
                f"{{id: $id, content: $c, embedding: $e, metadata: $m}}",
                {"t": self._table, "id": record_id,
                 "c": content, "e": embedding, "m": metadata},
            )

    async def query(
        self,
        query: str,
        top_k: int,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        if query_embedding is None:
            query_embedding = (await self.embedding_func([query]))[0]
        rows = await self._db.query(
            f"SELECT id, content, metadata, "
            f"vector::similarity::cosine(embedding, $vec) AS score "
            f"FROM {self._table} "
            f"WHERE embedding <|{top_k},{self._ef}|> $vec "
            f"AND vector::similarity::cosine(embedding, $vec) >= $threshold "
            f"ORDER BY score DESC LIMIT {top_k}",
            {"vec": query_embedding,
             "threshold": self.cosine_better_than_threshold},
        )
        return rows

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        rows = await self._db.query(
            f"SELECT * FROM {self._table} WHERE id = $id LIMIT 1", {"id": id}
        )
        return rows[0] if rows else None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        rows = await self._db.query(
            f"SELECT * FROM {self._table} WHERE id IN $ids", {"ids": ids}
        )
        row_map = {r["id"]: r for r in rows}
        return [row_map.get(i) for i in ids]

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        await self._db.query(
            f"DELETE {self._table} WHERE id IN $ids", {"ids": ids}
        )

    async def delete_entity(self, entity_name: str) -> None:
        await self._db.query(
            f"DELETE {self._table} WHERE metadata.entity_name = $name",
            {"name": entity_name},
        )

    async def delete_entity_relation(self, entity_name: str) -> None:
        await self._db.query(
            f"DELETE {self._table} WHERE metadata.src_id = $name OR metadata.tgt_id = $name",
            {"name": entity_name},
        )

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        rows = await self._db.query(
            f"SELECT id, embedding FROM {self._table} WHERE id IN $ids",
            {"ids": ids},
        )
        return {r["id"]: r["embedding"] for r in rows}


# ---------------------------------------------------------------------------
# Graph Storage
# ---------------------------------------------------------------------------

@dataclass
class SurrealDBGraphStorage(BaseGraphStorage):
    """
    Knowledge graph storage using two flat SurrealDB tables.
    Edges are stored as a flat relation table (not RELATE) to match
    LightRAG's string-ID edge addressing pattern.
    """

    def __post_init__(self) -> None:
        self._ent = f"ent_{self.namespace}"   # entity nodes
        self._rel = f"rel_{self.namespace}"   # relation edges
        self._db: SurrealDBDB | None = None

    async def initialize(self) -> None:
        self._db = await get_connection()
        await self._db.query(f"""
            DEFINE TABLE IF NOT EXISTS {self._ent} SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id          ON {self._ent} TYPE string;
            DEFINE FIELD IF NOT EXISTS entity_name ON {self._ent} TYPE string;
            DEFINE FIELD IF NOT EXISTS entity_type ON {self._ent} TYPE string;
            DEFINE FIELD IF NOT EXISTS description ON {self._ent} TYPE string;
            DEFINE FIELD IF NOT EXISTS source_id   ON {self._ent} TYPE string;
            DEFINE FIELD IF NOT EXISTS extra       ON {self._ent} FLEXIBLE TYPE object;
            DEFINE INDEX IF NOT EXISTS idx_ent_name ON {self._ent} COLUMNS entity_name UNIQUE;

            DEFINE TABLE IF NOT EXISTS {self._rel} SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id          ON {self._rel} TYPE string;
            DEFINE FIELD IF NOT EXISTS src_id      ON {self._rel} TYPE string;
            DEFINE FIELD IF NOT EXISTS tgt_id      ON {self._rel} TYPE string;
            DEFINE FIELD IF NOT EXISTS weight      ON {self._rel} TYPE float;
            DEFINE FIELD IF NOT EXISTS description ON {self._rel} TYPE string;
            DEFINE FIELD IF NOT EXISTS keywords    ON {self._rel} TYPE array<string>;
            DEFINE FIELD IF NOT EXISTS source_id   ON {self._rel} TYPE string;
            DEFINE FIELD IF NOT EXISTS extra       ON {self._rel} FLEXIBLE TYPE object;
            DEFINE INDEX IF NOT EXISTS idx_rel_src ON {self._rel} COLUMNS src_id;
            DEFINE INDEX IF NOT EXISTS idx_rel_tgt ON {self._rel} COLUMNS tgt_id;
        """)

    async def finalize(self) -> None:
        pass

    async def index_done_callback(self) -> None:
        pass

    async def drop(self) -> dict[str, str]:
        try:
            await self._db.query(f"REMOVE TABLE {self._ent}")
            await self._db.query(f"REMOVE TABLE {self._rel}")
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ── Nodes ──────────────────────────────────────────────────────────

    async def has_node(self, node_id: str) -> bool:
        rows = await self._db.query(
            f"SELECT id FROM {self._ent} WHERE entity_name = $id LIMIT 1",
            {"id": node_id},
        )
        return bool(rows)

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        rows = await self._db.query(
            f"SELECT * FROM {self._ent} WHERE entity_name = $id LIMIT 1",
            {"id": node_id},
        )
        return rows[0] if rows else None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        record_id = node_id.replace(" ", "_").replace("/", "_")
        payload = {"id": record_id, "entity_name": node_id, **node_data}
        await self._db.query(
            f"UPSERT type::thing($t, $id) CONTENT $data",
            {"t": self._ent, "id": record_id, "data": payload},
        )

    async def delete_node(self, node_id: str) -> None:
        await self._db.query(
            f"DELETE {self._ent} WHERE entity_name = $id", {"id": node_id}
        )
        await self._db.query(
            f"DELETE {self._rel} WHERE src_id = $id OR tgt_id = $id", {"id": node_id}
        )

    async def remove_nodes(self, nodes: list[str]) -> None:
        for node in nodes:
            await self.delete_node(node)

    async def node_degree(self, node_id: str) -> int:
        rows = await self._db.query(
            f"SELECT count() AS cnt FROM {self._rel} "
            f"WHERE src_id = $id OR tgt_id = $id GROUP ALL",
            {"id": node_id},
        )
        return rows[0]["cnt"] if rows else 0

    async def get_all_labels(self) -> list[str]:
        rows = await self._db.query(f"SELECT entity_name FROM {self._ent}")
        return sorted(r["entity_name"] for r in rows)

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        rows = await self._db.query(
            f"SELECT entity_name, count() AS deg FROM {self._rel} "
            f"GROUP BY src_id, tgt_id LIMIT {limit}"
        )
        # Simpler fallback: just return top labels by degree
        all_nodes = await self._db.query(f"SELECT entity_name FROM {self._ent} LIMIT {limit}")
        return [r["entity_name"] for r in all_nodes]

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        rows = await self._db.query(
            f"SELECT entity_name FROM {self._ent} "
            f"WHERE string::contains(string::lowercase(entity_name), $q) LIMIT {limit}",
            {"q": query.lower()},
        )
        return [r["entity_name"] for r in rows]

    async def get_all_nodes(self) -> list[dict]:
        return await self._db.query(f"SELECT * FROM {self._ent}")

    # ── Edges ──────────────────────────────────────────────────────────

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        edge_id = f"{source_node_id}___{target_node_id}"
        rows = await self._db.query(
            f"SELECT id FROM {self._rel} WHERE id = $id LIMIT 1", {"id": edge_id}
        )
        return bool(rows)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        edge_id = f"{source_node_id}___{target_node_id}"
        rows = await self._db.query(
            f"SELECT * FROM {self._rel} WHERE id = $id LIMIT 1", {"id": edge_id}
        )
        return rows[0] if rows else None

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        edge_id = f"{source_node_id}___{target_node_id}"
        payload = {
            "id": edge_id,
            "src_id": source_node_id,
            "tgt_id": target_node_id,
            **edge_data,
        }
        await self._db.query(
            f"UPSERT type::thing($t, $id) CONTENT $data",
            {"t": self._rel, "id": edge_id, "data": payload},
        )

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        for src, tgt in edges:
            edge_id = f"{src}___{tgt}"
            await self._db.query(
                f"DELETE {self._rel} WHERE id = $id", {"id": edge_id}
            )

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return await self.node_degree(src_id) + await self.node_degree(tgt_id)

    async def get_node_edges(
        self, source_node_id: str
    ) -> list[tuple[str, str]] | None:
        rows = await self._db.query(
            f"SELECT src_id, tgt_id FROM {self._rel} "
            f"WHERE src_id = $id OR tgt_id = $id",
            {"id": source_node_id},
        )
        if not rows:
            return None
        return [(r["src_id"], r["tgt_id"]) for r in rows]

    async def get_all_edges(self) -> list[dict]:
        return await self._db.query(f"SELECT * FROM {self._rel}")

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        """BFS subgraph from node_label up to max_depth hops."""
        kg = KnowledgeGraph()
        visited_nodes: set[str] = set()
        visited_edges: set[tuple[str, str]] = set()

        if node_label == "*":
            nodes = await self.get_all_nodes()
            for n in nodes[:max_nodes]:
                name = n.get("entity_name", n.get("id", ""))
                kg.nodes.append(KnowledgeGraphNode(id=name, labels=[name], properties=n))
                visited_nodes.add(name)
            edges = await self.get_all_edges()
            for e in edges:
                pair = (e["src_id"], e["tgt_id"])
                if pair not in visited_edges:
                    visited_edges.add(pair)
                    kg.edges.append(KnowledgeGraphEdge(
                        id=e.get("id", f"{e['src_id']}___{e['tgt_id']}"),
                        source=e["src_id"], target=e["tgt_id"],
                        type="RELATED", properties=e,
                    ))
            kg.is_truncated = len(nodes) >= max_nodes
            return kg

        frontier = {node_label}
        for _ in range(max_depth):
            if not frontier or len(visited_nodes) >= max_nodes:
                break
            next_frontier: set[str] = set()
            for node in frontier:
                if node in visited_nodes:
                    continue
                node_data = await self.get_node(node)
                if node_data:
                    kg.nodes.append(KnowledgeGraphNode(
                        id=node, labels=[node], properties=node_data
                    ))
                visited_nodes.add(node)
                edges = await self.get_node_edges(node) or []
                for src, tgt in edges:
                    pair = (src, tgt)
                    if pair not in visited_edges:
                        visited_edges.add(pair)
                        edge_data = await self.get_edge(src, tgt) or {}
                        kg.edges.append(KnowledgeGraphEdge(
                            id=edge_data.get("id", f"{src}___{tgt}"),
                            source=src, target=tgt,
                            type="RELATED", properties=edge_data,
                        ))
                    neighbor = tgt if src == node else src
                    if neighbor not in visited_nodes:
                        next_frontier.add(neighbor)
            frontier = next_frontier

        kg.is_truncated = len(visited_nodes) >= max_nodes
        return kg


# ---------------------------------------------------------------------------
# Doc Status Storage
# ---------------------------------------------------------------------------

@dataclass
class SurrealDBDocStatusStorage(DocStatusStorage):
    """
    Document ingestion status storage.
    Inherits from DocStatusStorage which itself inherits from BaseKVStorage.
    """

    def __post_init__(self) -> None:
        self._table = f"doc_status_{self.namespace}"
        self._db: SurrealDBDB | None = None

    async def initialize(self) -> None:
        self._db = await get_connection()
        await self._db.query(f"""
            DEFINE TABLE IF NOT EXISTS {self._table} SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id              ON {self._table} TYPE string;
            DEFINE FIELD IF NOT EXISTS status          ON {self._table} TYPE string;
            DEFINE FIELD IF NOT EXISTS content_summary ON {self._table} TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS content_length  ON {self._table} TYPE option<int>;
            DEFINE FIELD IF NOT EXISTS content_hash    ON {self._table} TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS file_path       ON {self._table} TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS chunks_count    ON {self._table} TYPE option<int>;
            DEFINE FIELD IF NOT EXISTS chunks_list     ON {self._table} TYPE option<array<string>>;
            DEFINE FIELD IF NOT EXISTS error_msg       ON {self._table} TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS track_id        ON {self._table} TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS created_at      ON {self._table} TYPE string;
            DEFINE FIELD IF NOT EXISTS updated_at      ON {self._table} TYPE string;
            DEFINE FIELD IF NOT EXISTS metadata        ON {self._table} FLEXIBLE TYPE object;
            DEFINE INDEX IF NOT EXISTS idx_ds_id       ON {self._table} COLUMNS id UNIQUE;
            DEFINE INDEX IF NOT EXISTS idx_ds_status   ON {self._table} COLUMNS status;
            DEFINE INDEX IF NOT EXISTS idx_ds_hash     ON {self._table} COLUMNS content_hash;
            DEFINE INDEX IF NOT EXISTS idx_ds_path     ON {self._table} COLUMNS file_path;
        """)

    async def finalize(self) -> None:
        pass

    async def index_done_callback(self) -> None:
        pass

    async def drop(self) -> dict[str, str]:
        try:
            await self._db.query(f"REMOVE TABLE {self._table}")
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ── BaseKVStorage interface ────────────────────────────────────────

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        rows = await self._db.query(
            f"SELECT * FROM {self._table} WHERE id = $id LIMIT 1", {"id": id}
        )
        return rows[0] if rows else None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        rows = await self._db.query(
            f"SELECT * FROM {self._table} WHERE id IN $ids", {"ids": ids}
        )
        row_map = {r["id"]: r for r in rows}
        return [row_map.get(i) for i in ids]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        rows = await self._db.query(
            f"SELECT id FROM {self._table} WHERE id IN $ids AND status = 'processed'",
            {"ids": list(keys)},
        )
        done = {r["id"] for r in rows}
        return keys - done

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        for doc_id, doc_data in data.items():
            payload = {"id": doc_id, "updated_at": _now_iso(), **doc_data}
            if "created_at" not in payload:
                payload["created_at"] = _now_iso()
            await self._db.query(
                f"UPSERT type::thing($t, $id) CONTENT $data",
                {"t": self._table, "id": doc_id, "data": payload},
            )

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        await self._db.query(
            f"DELETE {self._table} WHERE id IN $ids", {"ids": ids}
        )

    async def is_empty(self) -> bool:
        rows = await self._db.query(
            f"SELECT count() AS cnt FROM {self._table} GROUP ALL"
        )
        return (rows[0]["cnt"] if rows else 0) == 0

    # ── DocStatusStorage interface ─────────────────────────────────────

    def _row_to_status(self, row: dict) -> DocProcessingStatus:
        return DocProcessingStatus(
            content_summary=row.get("content_summary", ""),
            content_length=row.get("content_length", 0),
            file_path=row.get("file_path", "unknown_source"),
            status=DocStatus(row.get("status", "pending")),
            created_at=row.get("created_at", _now_iso()),
            updated_at=row.get("updated_at", _now_iso()),
            track_id=row.get("track_id"),
            chunks_count=row.get("chunks_count"),
            chunks_list=row.get("chunks_list", []),
            error_msg=row.get("error_msg"),
            metadata=row.get("metadata", {}),
            content_hash=row.get("content_hash"),
        )

    async def get_status_counts(self) -> dict[str, int]:
        rows = await self._db.query(
            f"SELECT status, count() AS cnt FROM {self._table} GROUP BY status"
        )
        return {r["status"]: r["cnt"] for r in rows}

    async def get_all_status_counts(self) -> dict[str, int]:
        return await self.get_status_counts()

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        rows = await self._db.query(
            f"SELECT * FROM {self._table} WHERE status = $s", {"s": status.value}
        )
        return {r["id"]: self._row_to_status(r) for r in rows}

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus]
    ) -> dict[str, DocProcessingStatus]:
        vals = [s.value for s in statuses]
        rows = await self._db.query(
            f"SELECT * FROM {self._table} WHERE status IN $s", {"s": vals}
        )
        return {r["id"]: self._row_to_status(r) for r in rows}

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        rows = await self._db.query(
            f"SELECT * FROM {self._table} WHERE track_id = $tid", {"tid": track_id}
        )
        return {r["id"]: self._row_to_status(r) for r in rows}

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        status_filters: list[DocStatus] | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        where = ""
        bind: dict[str, Any] = {}
        filter_vals = DocStatusStorage.resolve_status_filter_values(
            status_filter, status_filters
        )
        if filter_vals:
            where = "WHERE status IN $statuses"
            bind["statuses"] = list(filter_vals)

        offset = (page - 1) * page_size
        order  = f"ORDER BY {sort_field} {'DESC' if sort_direction == 'desc' else 'ASC'}"

        rows = await self._db.query(
            f"SELECT * FROM {self._table} {where} {order} LIMIT {page_size} START {offset}",
            bind,
        )
        count_rows = await self._db.query(
            f"SELECT count() AS cnt FROM {self._table} {where} GROUP ALL", bind
        )
        total = count_rows[0]["cnt"] if count_rows else 0
        return [(r["id"], self._row_to_status(r)) for r in rows], total

    async def get_doc_by_file_path(self, file_path: str) -> dict[str, Any] | None:
        rows = await self._db.query(
            f"SELECT * FROM {self._table} WHERE file_path = $fp LIMIT 1",
            {"fp": file_path},
        )
        return rows[0] if rows else None

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> tuple[str, dict[str, Any]] | None:
        rows = await self._db.query(
            f"SELECT * FROM {self._table} WHERE file_path = $fp LIMIT 1",
            {"fp": basename},
        )
        if rows:
            return rows[0]["id"], rows[0]
        return None

    async def get_doc_by_content_hash(
        self, content_hash: str
    ) -> tuple[str, dict[str, Any]] | None:
        rows = await self._db.query(
            f"SELECT * FROM {self._table} WHERE content_hash = $h LIMIT 1",
            {"h": content_hash},
        )
        if rows:
            return rows[0]["id"], rows[0]
        return None
