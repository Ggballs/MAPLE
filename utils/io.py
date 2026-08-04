from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from utils.constants import CANONICAL_VIEWS, REPRESENTATION_TO_DIRNAME
from utils.types import PaperRecord, QueryRecord

QUERY_ID_FIELDS = ("query_id", "query_key", "id")
QUERY_TEXT_FIELDS = ("query_text", "query", "text")
TARGET_PAPER_FIELDS = ("target_paper_id", "paper_id")
ASPECT_FIELDS = ("aspect_group", "source_view", "view")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_aspect_group(value: str) -> str:
    raw = str(value or "").strip().lower()
    normalized = raw.replace("_", "/")
    if normalized == "experiment":
        normalized = "experiment/result"
    if normalized not in CANONICAL_VIEWS:
        raise ValueError(f"Unsupported aspect group: {value!r}")
    return normalized


def resolve_representation_dir(representations_root: str | Path, representation: str) -> Path:
    if representation not in REPRESENTATION_TO_DIRNAME:
        raise ValueError(f"Unsupported representation: {representation}")
    return Path(representations_root).expanduser().resolve() / REPRESENTATION_TO_DIRNAME[representation]


def list_representation_shards(representations_root: str | Path, representation: str) -> list[Path]:
    rep_dir = resolve_representation_dir(representations_root, representation)
    if not rep_dir.is_dir():
        raise FileNotFoundError(f"representation directory not found: {rep_dir}")
    return sorted(rep_dir.glob("*.jsonl"))


def load_paper_records(
    representations_root: str | Path,
    representation: str,
    *,
    max_papers: int | None = None,
) -> tuple[list[PaperRecord], list[str], int]:
    shard_paths = list_representation_shards(representations_root, representation)
    paper_texts: dict[str, list[str]] = defaultdict(list)
    paper_source_paths: dict[str, list[str]] = defaultdict(list)
    total_rows = 0
    limit = int(max_papers) if max_papers and max_papers > 0 else None
    for shard in shard_paths:
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                total_rows += 1
                paper_id = str(row.get("paper_id") or "").strip()
                text = str(row.get("text") or "")
                source_path = str(row.get("source_path") or "")
                if not paper_id:
                    continue
                if text:
                    paper_texts[paper_id].append(text)
                if source_path:
                    paper_source_paths[paper_id].append(source_path)
                if limit is not None and len(paper_texts) >= limit:
                    break
        if limit is not None and len(paper_texts) >= limit:
            break
    papers = [
        PaperRecord(
            paper_id=paper_id,
            representation_type=representation,
            text="\n\n".join(part for part in parts if part).strip(),
            source_paths=tuple(paper_source_paths.get(paper_id, [])),
        )
        for paper_id, parts in sorted(paper_texts.items())
    ]
    if limit is not None:
        papers = papers[:limit]
    return papers, [str(path) for path in shard_paths], total_rows


def _first_str(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def load_query_records_from_rows(rows: Iterable[dict[str, Any]]) -> list[QueryRecord]:
    queries: list[QueryRecord] = []
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        query_id = _first_str(row, QUERY_ID_FIELDS) or f"line-{line_number:06d}"
        if query_id in seen:
            continue
        query_text = _first_str(row, QUERY_TEXT_FIELDS)
        target_paper_id = _first_str(row, TARGET_PAPER_FIELDS)
        aspect_group_raw = _first_str(row, ASPECT_FIELDS)
        if not query_text or not target_paper_id or not aspect_group_raw:
            continue
        try:
            aspect_group = normalize_aspect_group(aspect_group_raw)
        except ValueError:
            continue
        seen.add(query_id)
        queries.append(
            QueryRecord(
                query_id=query_id,
                query_text=query_text,
                target_paper_id=target_paper_id,
                aspect_group=aspect_group,
            )
        )
    return queries


def load_queries(query_path: str | Path) -> list[QueryRecord]:
    path = Path(query_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"query file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return load_query_records_from_rows(rows)


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return slug or "item"
