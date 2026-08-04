from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    query_text: str
    target_paper_id: str
    aspect_group: str


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    representation_type: str
    text: str
    source_paths: tuple[str, ...]


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    paper_id: str
    representation_type: str
    chunk_index: int
    text: str
    source_path: str
    token_count: int
    overlap_tokens: int


@dataclass(frozen=True)
class RankedResult:
    query_id: str
    ranked_ids: list[str]
    scores: list[float]

