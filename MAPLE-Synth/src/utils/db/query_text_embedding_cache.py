from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
import sqlalchemy


QUERY_TEXT_EMBEDDING_CACHE_TABLE = "query_text_embedding_cache"


def parse_embedding_value(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("Cannot parse NULL embedding.")
    if isinstance(value, str):
        items = [float(v) for v in value.strip("[]").split(",") if v.strip()]
        return np.asarray(items, dtype=np.float32)
    return np.asarray([float(v) for v in value], dtype=np.float32)


def vector_str(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.10g}" for v in vec) + "]"


def ensure_query_cache_table(engine) -> None:
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(
            sqlalchemy.text(
                f"""
                CREATE TABLE IF NOT EXISTS {QUERY_TEXT_EMBEDDING_CACHE_TABLE} (
                    cache_key TEXT PRIMARY KEY,
                    query_id TEXT NOT NULL,
                    query_text_sha1 TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    instruction_sha1 TEXT NOT NULL DEFAULT '',
                    instruction TEXT,
                    query_text TEXT NOT NULL,
                    embedding vector NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        conn.execute(
            sqlalchemy.text(
                f"""
                CREATE INDEX IF NOT EXISTS idx_query_text_embedding_cache_model
                ON {QUERY_TEXT_EMBEDDING_CACHE_TABLE} (model_name, endpoint)
                """
            )
        )


def build_query_cache_key(
    *,
    query_id: str,
    query_text: str,
    model_name: str,
    instruction: str | None,
) -> tuple[str, str, str]:
    query_text_sha1 = hashlib.sha1(query_text.encode("utf-8")).hexdigest()
    instruction_sha1 = hashlib.sha1(str(instruction or "").encode("utf-8")).hexdigest()
    payload = {
        "model": model_name,
        "instruction_sha1": instruction_sha1,
        "query_id": query_id,
        "query_text_sha1": query_text_sha1,
    }
    cache_key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return cache_key, query_text_sha1, instruction_sha1


def fetch_cached_embeddings(
    engine,
    *,
    cache_keys: Sequence[str],
) -> dict[str, np.ndarray]:
    if not cache_keys:
        return {}
    stmt = (
        sqlalchemy.text(
            f"""
            SELECT cache_key, embedding
            FROM {QUERY_TEXT_EMBEDDING_CACHE_TABLE}
            WHERE cache_key IN :keys
            """
        ).bindparams(sqlalchemy.bindparam("keys", expanding=True))
    )
    cached: dict[str, np.ndarray] = {}
    with engine.connect() as conn:
        for start in range(0, len(cache_keys), 512):
            chunk = list(cache_keys[start : start + 512])
            for row in conn.execute(stmt, {"keys": chunk}):
                cached[str(row.cache_key)] = parse_embedding_value(row.embedding)
    return cached


def insert_cached_embeddings(engine, rows: Sequence[dict[str, Any]]) -> int:
    if not rows:
        return 0
    insert_sql = sqlalchemy.text(
        f"""
        INSERT INTO {QUERY_TEXT_EMBEDDING_CACHE_TABLE} (
            cache_key,
            query_id,
            query_text_sha1,
            model_name,
            endpoint,
            instruction_sha1,
            instruction,
            query_text,
            embedding
        )
        VALUES (
            :cache_key,
            :query_id,
            :query_text_sha1,
            :model_name,
            :endpoint,
            :instruction_sha1,
            :instruction,
            :query_text,
            CAST(:embedding AS vector)
        )
        ON CONFLICT (cache_key) DO NOTHING
        """
    )
    inserted = 0
    with engine.begin() as conn:
        result = conn.execute(insert_sql, list(rows))
        if result.rowcount is not None and result.rowcount > 0:
            inserted = int(result.rowcount)
    return inserted
