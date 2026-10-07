from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import text

from openreview_pipeline.stage5_worker.core import _slugify
from openreview_pipeline.stage5_worker.runtime import _download_pdf_to_path
from utils.db.paper_text_embeddings import ensure_table, existing_embeddings, get_engine, insert_embeddings
from utils.docling_parse import parse_pdf_text_only


DEFAULT_QUEUE_TABLE = "stage5_candidate_queue"
DEFAULT_MODEL_NAME = "qwen3-embed-8b"
DEFAULT_MODEL_PATH = "Qwen/Qwen3-Embedding-8B"
DEFAULT_DEVICE = "cpu"
DEFAULT_MAX_CHARS = 32000


@dataclass
class HardNegativeRecord:
    row_id: int
    paper_id: str
    candidate_title: str
    candidate_arxiv_id: Optional[str]
    candidate_pdf_url: Optional[str]
    candidate_full_text_path: Optional[str]
    candidate_pdf_path: Optional[str]
    candidate_abstract: Optional[str]


def _derive_paper_id(candidate_title: str, candidate_arxiv_id: Optional[str]) -> str:
    if candidate_arxiv_id:
        return str(candidate_arxiv_id).replace("/", "_")
    return hashlib.sha1(candidate_title.encode("utf-8")).hexdigest()[:12]


def _pdf_url_hash(pdf_url: str) -> str:
    return hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()[:10]


def _load_hard_negative_rows(engine, queue_table_name: str) -> list[HardNegativeRecord]:
    canonical_view_name = f"{queue_table_name}_canonical" if queue_table_name != DEFAULT_QUEUE_TABLE else "stage5_candidate_queue_canonical"
    sql = text(
        f"""
        SELECT
            id,
            candidate_title,
            candidate_arxiv_id,
            candidate_pdf_url,
            candidate_abstract,
            review_payload
        FROM {canonical_view_name}
        WHERE review_status = 'completed'
          AND review_label = 'hard_negative'
        ORDER BY id
        """
    )
    rows: list[HardNegativeRecord] = []
    seen_paper_ids: set[str] = set()
    with engine.connect() as conn:
        for row in conn.execute(sql):
            review_payload = row.review_payload or {}
            paper_id = _derive_paper_id(str(row.candidate_title or ""), row.candidate_arxiv_id)
            if paper_id in seen_paper_ids:
                continue
            seen_paper_ids.add(paper_id)
            rows.append(
                HardNegativeRecord(
                    row_id=int(row.id),
                    paper_id=paper_id,
                    candidate_title=str(row.candidate_title or ""),
                    candidate_arxiv_id=str(row.candidate_arxiv_id) if row.candidate_arxiv_id else None,
                    candidate_pdf_url=str(row.candidate_pdf_url) if row.candidate_pdf_url else None,
                    candidate_full_text_path=str(review_payload.get("candidate_full_text_path") or "") or None,
                    candidate_pdf_path=str(review_payload.get("candidate_pdf_path") or "") or None,
                    candidate_abstract=str(row.candidate_abstract or "") or None,
                )
            )
    return rows


def _resolve_pdf_path(record: HardNegativeRecord, pdf_output_dir: Path) -> Optional[Path]:
    if record.candidate_pdf_path:
        candidate = Path(record.candidate_pdf_path)
        if candidate.exists():
            return candidate
    if record.candidate_pdf_url:
        cached = pdf_output_dir / f"{_slugify(record.candidate_title, 80)}-{_pdf_url_hash(record.candidate_pdf_url)}.pdf"
        if cached.exists():
            return cached
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            _download_pdf_to_path(record.candidate_pdf_url, cached, timeout_seconds=45)
            if cached.exists() and cached.stat().st_size > 0:
                return cached
        except Exception:
            return None
    return None


def _ensure_full_text(record: HardNegativeRecord, pdf_output_dir: Path) -> tuple[Optional[Path], Optional[str]]:
    if record.candidate_full_text_path:
        full_text_path = Path(record.candidate_full_text_path)
        if full_text_path.exists():
            return full_text_path, full_text_path.read_text(encoding="utf-8")

    pdf_path = _resolve_pdf_path(record, pdf_output_dir)
    if pdf_path is None:
        return None, None

    payload = parse_pdf_text_only(pdf_path, disable_table_structure=False)
    markdown = str(payload.get("markdown") or "") if payload.get("ok") else ""
    if not markdown.strip():
        retry_payload = parse_pdf_text_only(pdf_path, disable_table_structure=True)
        markdown = str(retry_payload.get("markdown") or "") if retry_payload.get("ok") else ""
    if not markdown.strip():
        return None, None

    full_text_dir = pdf_output_dir / "parsed_full_text"
    full_text_dir.mkdir(parents=True, exist_ok=True)
    full_text_path = full_text_dir / f"{pdf_path.stem}.md"
    full_text_path.write_text(markdown, encoding="utf-8")
    return full_text_path, markdown


def _truncate(text: str, max_chars: int) -> str:
    return text[:max_chars]


def _embed_texts(
    texts: list[str],
    *,
    model_path: str,
    device: str,
    batch_size: int,
) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=False, show_progress_bar=True)
    return vectors.tolist()


def _update_review_payload_paths(engine, queue_table_name: str, row_id: int, pdf_path: Optional[Path], full_text_path: Optional[Path]) -> None:
    sql = text(
        f"""
        UPDATE {queue_table_name}
        SET
            review_payload = COALESCE(review_payload, '{{}}'::jsonb)
                || jsonb_strip_nulls(
                    jsonb_build_object(
                        'candidate_pdf_path', CAST(:candidate_pdf_path AS text),
                        'candidate_full_text_path', CAST(:candidate_full_text_path AS text)
                    )
                ),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = :row_id
        """
    )
    with engine.begin() as conn:
        conn.execute(
            sql,
            {
                "row_id": row_id,
                "candidate_pdf_path": str(pdf_path) if pdf_path else None,
                "candidate_full_text_path": str(full_text_path) if full_text_path else None,
            },
        )


def _prepare_record(record: HardNegativeRecord, pdf_output_dir: Path) -> dict[str, Any]:
    pdf_path = _resolve_pdf_path(record, pdf_output_dir)
    full_text_path, markdown = _ensure_full_text(record, pdf_output_dir)
    reused = bool(record.candidate_full_text_path and full_text_path and Path(record.candidate_full_text_path).exists())
    return {
        "record": record,
        "pdf_path": pdf_path,
        "full_text_path": full_text_path,
        "markdown": markdown,
        "reused": reused,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse and embed Stage5 hard negatives with Qwen3.")
    parser.add_argument("--queue-table-name", default=DEFAULT_QUEUE_TABLE)
    parser.add_argument("--pdf-output-dir", required=True)
    parser.add_argument("--db-url", default="")
    parser.add_argument("--embed-model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--embed-device", default=DEFAULT_DEVICE)
    parser.add_argument("--embed-batch-size", type=int, default=4)
    parser.add_argument("--embed-max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--parse-workers", type=int, default=8)
    parser.add_argument("--manifest-path", default="")
    args = parser.parse_args()

    engine = get_engine(args.db_url or None)
    ensure_table(engine)

    pdf_output_dir = Path(args.pdf_output_dir).expanduser().resolve()
    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path
        else (pdf_output_dir.parent / "hard_negative_embedding_manifest.jsonl")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_hard_negative_rows(engine, str(args.queue_table_name))
    embedded_pairs = existing_embeddings(engine)

    papers_to_embed: list[HardNegativeRecord] = []
    texts: list[str] = []
    char_map: dict[str, int] = {}
    manifest_lines: list[str] = []
    reused_full_text = 0
    reparsed_full_text = 0
    missing_full_text = 0

    parse_workers = max(1, int(args.parse_workers))
    with ThreadPoolExecutor(max_workers=parse_workers) as executor:
        futures = [executor.submit(_prepare_record, record, pdf_output_dir) for record in rows]
        for future in as_completed(futures):
            prepared = future.result()
            record = prepared["record"]
            pdf_path = prepared["pdf_path"]
            full_text_path = prepared["full_text_path"]
            markdown = prepared["markdown"]
            reused = bool(prepared["reused"])
            key = (record.paper_id, DEFAULT_MODEL_NAME)
            if full_text_path is not None:
                _update_review_payload_paths(engine, str(args.queue_table_name), record.row_id, pdf_path, full_text_path)
            if not markdown:
                missing_full_text += 1
                continue
            if reused:
                reused_full_text += 1
            else:
                reparsed_full_text += 1
            manifest_lines.append(
                json.dumps(
                    {
                        "row_id": record.row_id,
                        "paper_id": record.paper_id,
                        "candidate_title": record.candidate_title,
                        "candidate_arxiv_id": record.candidate_arxiv_id,
                        "candidate_pdf_url": record.candidate_pdf_url,
                        "candidate_pdf_path": str(pdf_path) if pdf_path else None,
                        "candidate_full_text_path": str(full_text_path) if full_text_path else None,
                        "reused_full_text": reused,
                    },
                    ensure_ascii=False,
                )
            )
            if key in embedded_pairs:
                continue
            papers_to_embed.append(record)
            text_value = _truncate(markdown, int(args.embed_max_chars))
            texts.append(text_value)
            char_map[record.paper_id] = len(text_value)

    if manifest_lines:
        manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    inserted = 0
    if papers_to_embed:
        vectors = _embed_texts(
            texts,
            model_path=str(args.embed_model_path),
            device=str(args.embed_device),
            batch_size=max(1, int(args.embed_batch_size)),
        )
        inserted = insert_embeddings(
            engine,
            [record.paper_id for record in papers_to_embed],
            DEFAULT_MODEL_NAME,
            vectors,
            char_map,
        )

    summary = {
        "queue_table_name": str(args.queue_table_name),
        "hard_negative_unique_candidates": len(rows),
        "reused_full_text": reused_full_text,
        "reparsed_full_text": reparsed_full_text,
        "missing_full_text": missing_full_text,
        "new_embeddings_inserted": inserted,
        "embedding_model": DEFAULT_MODEL_NAME,
        "embedding_device": str(args.embed_device),
        "parse_workers": parse_workers,
        "manifest_path": str(manifest_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
