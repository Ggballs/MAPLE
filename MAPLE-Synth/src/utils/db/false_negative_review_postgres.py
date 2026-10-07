from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    BigInteger,
    Column,
    Index,
    Integer,
    MetaData,
    String,
    TIMESTAMP,
    Table,
    Text,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, insert as postgres_insert
from sqlalchemy.engine import Engine


FALSE_NEGATIVE_REVIEW_TABLE = "false_negative_review"
DEFAULT_TASK = "false-negative-human-check-gritlm-top5-v1"
logger = logging.getLogger(__name__)

metadata = MetaData()
_ENGINE_CACHE: Dict[str, Engine] = {}

false_negative_review = Table(
    FALSE_NEGATIVE_REVIEW_TABLE,
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("task", String, nullable=False),
    Column("pair_id", String, nullable=False),
    Column("query_id", String, nullable=False),
    Column("paper_id", String, nullable=False),
    Column("reviewer_username", String, nullable=False),
    Column("choice", String, nullable=False),
    Column("confidence", String, nullable=True),
    Column("note", Text, nullable=True),
    Column("ordering_seed", Integer, nullable=False),
    Column("pair_snapshot", JSONB, nullable=False),
    Column(
        "created_at",
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column(
        "updated_at",
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        server_onupdate=text("CURRENT_TIMESTAMP"),
    ),
    Index("idx_false_negative_review_task", "task"),
    Index("idx_false_negative_review_query_id", "query_id"),
    Index("idx_false_negative_review_paper_id", "paper_id"),
    Index("idx_false_negative_review_reviewer_username", "reviewer_username"),
    Index("uq_false_negative_review_task_pair_reviewer", "task", "pair_id", "reviewer_username", unique=True),
)


def get_engine(db_url: Optional[str] = None) -> Engine:
    url = db_url or os.getenv("HUMAN_FEEDBACK_DB_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("Set HUMAN_FEEDBACK_DB_URL or DATABASE_URL to use false-negative review storage.")
    if url not in _ENGINE_CACHE:
        _ENGINE_CACHE[url] = create_engine(
            url,
            pool_pre_ping=True,
            pool_size=int(os.getenv("POSTGRES_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("POSTGRES_MAX_OVERFLOW", "0")),
            pool_recycle=int(os.getenv("POSTGRES_POOL_RECYCLE", "1800")),
        )
    return _ENGINE_CACHE[url]


def ensure_schema(*, engine: Optional[Engine] = None, db_url: Optional[str] = None) -> Engine:
    db = engine or get_engine(db_url)
    metadata.create_all(db, tables=[false_negative_review])
    return db


def build_pair_id(query_id: str, paper_id: str) -> str:
    raw = f"{query_id}\n{paper_id}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def save_review(
    *,
    task: str,
    query_id: str,
    paper_id: str,
    reviewer_username: str,
    choice: str,
    ordering_seed: int,
    pair_snapshot: Dict[str, Any],
    confidence: Optional[str] = None,
    note: Optional[str] = None,
    pair_id: Optional[str] = None,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> None:
    db = ensure_schema(engine=engine, db_url=db_url)
    resolved_pair_id = str(pair_id or build_pair_id(query_id, paper_id))
    row = {
        "task": str(task or DEFAULT_TASK).strip() or DEFAULT_TASK,
        "pair_id": resolved_pair_id,
        "query_id": str(query_id),
        "paper_id": str(paper_id),
        "reviewer_username": str(reviewer_username).strip(),
        "choice": str(choice).strip(),
        "confidence": str(confidence).strip() if confidence else None,
        "note": str(note) if note not in (None, "") else None,
        "ordering_seed": int(ordering_seed),
        "pair_snapshot": pair_snapshot,
    }
    stmt = postgres_insert(false_negative_review).values(row)
    update_columns = {
        col.name: stmt.excluded[col.name]
        for col in false_negative_review.columns
        if col.name not in {"id", "created_at", "updated_at"}
    }
    update_columns["updated_at"] = text("CURRENT_TIMESTAMP")
    stmt = stmt.on_conflict_do_update(
        index_elements=["task", "pair_id", "reviewer_username"],
        set_=update_columns,
    )
    with db.begin() as conn:
        conn.execute(stmt)


def load_review(
    *,
    task: str,
    pair_id: str,
    reviewer_username: str,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    db = ensure_schema(engine=engine, db_url=db_url)
    task_name = str(task or DEFAULT_TASK).strip() or DEFAULT_TASK
    stmt = (
        select(false_negative_review)
        .where(false_negative_review.c.task == task_name)
        .where(false_negative_review.c.pair_id == str(pair_id))
        .where(false_negative_review.c.reviewer_username == str(reviewer_username).strip())
    )
    with db.begin() as conn:
        row = conn.execute(stmt).mappings().first()
    return dict(row) if row else None


def load_reviews_for_reviewer(
    *,
    task: str,
    reviewer_username: str,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    db = ensure_schema(engine=engine, db_url=db_url)
    task_name = str(task or DEFAULT_TASK).strip() or DEFAULT_TASK
    stmt = (
        select(false_negative_review)
        .where(false_negative_review.c.task == task_name)
        .where(false_negative_review.c.reviewer_username == str(reviewer_username).strip())
        .order_by(false_negative_review.c.query_id, false_negative_review.c.paper_id)
    )
    with db.begin() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [dict(row) for row in rows]


def load_progress(
    *,
    task: str,
    reviewer_username: Optional[str] = None,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> Dict[str, int]:
    db = ensure_schema(engine=engine, db_url=db_url)
    task_name = str(task or DEFAULT_TASK).strip() or DEFAULT_TASK
    stmt = select(func.count(false_negative_review.c.id).label("cnt")).where(
        false_negative_review.c.task == task_name
    )
    if reviewer_username:
        stmt = stmt.where(false_negative_review.c.reviewer_username == str(reviewer_username).strip())
    with db.begin() as conn:
        count = int(conn.execute(stmt).scalar() or 0)
    return {"reviewed_pairs": count}


def export_reviews_json(
    *,
    task: str,
    reviewer_username: Optional[str] = None,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> str:
    rows = load_reviews_for_reviewer(
        task=task,
        reviewer_username=str(reviewer_username or "").strip(),
        engine=engine,
        db_url=db_url,
    ) if reviewer_username else []
    return json.dumps(rows, ensure_ascii=False, indent=2)
