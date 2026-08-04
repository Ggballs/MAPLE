from __future__ import annotations

import gzip
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from utils.embed import l2_normalize

LOGGER = logging.getLogger(__name__)


def canonicalize_paper_id(value: str) -> str:
    paper_id = str(value or "").strip()
    for suffix in (".restored", ".docling"):
        if paper_id.endswith(suffix):
            paper_id = paper_id[: -len(suffix)]
    return paper_id


def parse_embedding_string(value: str) -> np.ndarray:
    return np.fromstring(str(value).strip().strip("[]"), sep=",", dtype=np.float32)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_tsv_gz(path: Path) -> Iterable[dict[str, str]]:
    with _open_text(path) as handle:
        header = None
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if header is None:
                header = line.split("\t")
                continue
            values = line.split("\t")
            if len(values) < len(header):
                values.extend([""] * (len(header) - len(values)))
            yield {key: values[index] for index, key in enumerate(header)}


def load_bundle_query_rows(query_path: Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(query_path))
    for row in rows:
        if "target_paper_id" not in row and "paper_id" in row:
            row["target_paper_id"] = row["paper_id"]
        if "aspect_group" not in row and "source_view" in row:
            row["aspect_group"] = row["source_view"]
    return rows


def load_jsonl_embedding_map(path: Path, *, id_field: str) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for row in read_jsonl(path):
        key = str(row.get(id_field) or "").strip()
        if not key:
            continue
        result[key] = parse_embedding_string(str(row.get("embedding") or "[]"))
    return result


def load_tsv_embedding_map(path: Path, *, id_field: str) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for row in read_tsv_gz(path):
        key = str(row.get(id_field) or "").strip()
        if not key:
            continue
        result[key] = parse_embedding_string(str(row.get("embedding") or "[]"))
    return result


def collect_tsv_ids(path: Path, *, id_field: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in read_tsv_gz(path):
        value = canonicalize_paper_id(str(row.get(id_field) or "").strip())
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def load_bundle_dense_arrays(
    *,
    paper_embeddings_path: Path,
    query_embeddings_path: Path,
    query_rows: list[dict[str, Any]],
    paper_id_field: str,
    query_id_field: str,
    query_target_field: str = "target_paper_id",
) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    if paper_embeddings_path.suffix == ".jsonl":
        paper_map = load_jsonl_embedding_map(paper_embeddings_path, id_field=paper_id_field)
    else:
        paper_map = load_tsv_embedding_map(paper_embeddings_path, id_field=paper_id_field)

    if query_embeddings_path.suffix == ".jsonl":
        query_map = load_jsonl_embedding_map(query_embeddings_path, id_field=query_id_field)
    else:
        query_map = load_tsv_embedding_map(query_embeddings_path, id_field=query_id_field)

    paper_ids = [canonicalize_paper_id(paper_id) for paper_id in paper_map.keys()]
    paper_vectors = l2_normalize(np.stack(list(paper_map.values()), axis=0))

    query_ids: list[str] = []
    query_vectors: list[np.ndarray] = []
    filtered_query_rows: list[dict[str, Any]] = []
    for row in query_rows:
        query_id = str(row.get("query_id") or "").strip()
        vector = query_map.get(query_id)
        if vector is None:
            continue
        cloned = dict(row)
        cloned[query_target_field] = canonicalize_paper_id(str(cloned.get(query_target_field) or ""))
        filtered_query_rows.append(cloned)
        query_ids.append(query_id)
        query_vectors.append(vector)
    return paper_ids, paper_vectors, query_ids, l2_normalize(np.stack(query_vectors, axis=0))


def compute_dense_rankings(
    *,
    paper_ids: list[str],
    paper_vectors: np.ndarray,
    query_ids: list[str],
    query_vectors: np.ndarray,
    top_k: int,
) -> list[tuple[str, list[str], list[float]]]:
    scores = query_vectors @ paper_vectors.T
    rows: list[tuple[str, list[str], list[float]]] = []
    for index, query_id in enumerate(query_ids):
        top_indices = np.argsort(-scores[index])[:top_k]
        ranked_ids = [paper_ids[pos] for pos in top_indices]
        ranked_scores = [float(scores[index][pos]) for pos in top_indices]
        rows.append((query_id, ranked_ids, ranked_scores))
    return rows


def collect_chunk_paper_ids(chunk_embeddings_path: Path) -> list[str]:
    paper_ids: list[str] = []
    seen: set[str] = set()
    for row in read_tsv_gz(chunk_embeddings_path):
        paper_id = canonicalize_paper_id(str(row.get("paper_id") or ""))
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        paper_ids.append(paper_id)
    return paper_ids


def reduce_max_scores_by_paper(
    scores: np.ndarray,
    *,
    paper_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if scores.size == 0 or paper_indices.size == 0:
        return (
            np.zeros((scores.shape[0], 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    order = np.argsort(paper_indices, kind="stable")
    sorted_papers = paper_indices[order]
    sorted_scores = scores[:, order]
    unique_papers, start_indices = np.unique(sorted_papers, return_index=True)
    reduced_scores = np.maximum.reduceat(sorted_scores, start_indices, axis=1)
    return reduced_scores.astype(np.float32, copy=False), unique_papers.astype(np.int64, copy=False)


def reduce_topk_scores_by_paper(
    scores: np.ndarray,
    *,
    paper_indices: np.ndarray,
    topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    if scores.size == 0 or paper_indices.size == 0:
        return (
            np.zeros((scores.shape[0], 0, topk), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    if topk <= 0:
        raise ValueError("topk must be positive")
    order = np.argsort(paper_indices, kind="stable")
    sorted_papers = paper_indices[order]
    sorted_scores = scores[:, order]
    unique_papers, start_indices = np.unique(sorted_papers, return_index=True)
    end_indices = np.concatenate([start_indices[1:], np.asarray([sorted_papers.shape[0]])])
    reduced = np.full((scores.shape[0], len(unique_papers), topk), -np.inf, dtype=np.float32)
    for group_index, (start, end) in enumerate(zip(start_indices.tolist(), end_indices.tolist(), strict=True)):
        group_scores = sorted_scores[:, start:end]
        width = group_scores.shape[1]
        keep = min(topk, width)
        if keep <= 0:
            continue
        if width == 1:
            top_values = group_scores
        else:
            partition_index = width - keep
            top_idx = np.argpartition(group_scores, partition_index, axis=1)[:, -keep:]
            top_values = np.take_along_axis(group_scores, top_idx, axis=1)
        top_values = np.sort(top_values, axis=1)[:, ::-1].astype(np.float32, copy=False)
        reduced[:, group_index, :keep] = top_values
    return reduced, unique_papers.astype(np.int64, copy=False)


def collapse_paper_scores(
    scores: np.ndarray,
    *,
    pooling: str,
) -> np.ndarray:
    if pooling == "max":
        return scores
    if pooling == "avg-topk":
        finite_mask = np.isfinite(scores)
        valid_counts = finite_mask.sum(axis=1)
        safe_denominator = np.maximum(valid_counts, 1)
        return np.where(finite_mask, scores, 0.0).sum(axis=1) / safe_denominator
    raise ValueError(f"Unsupported chunk pooling: {pooling!r}")


def _compute_streaming_chunk_paper_rankings_single(
    *,
    chunk_embeddings_path: Path,
    paper_ids: list[str],
    query_ids: list[str],
    query_vectors: np.ndarray,
    top_k: int,
    chunk_batch_size: int,
    chunk_pooling: str,
    chunk_pooling_topk: int,
    total_chunk_count: int | None = None,
    progress_log_every_batches: int = 50,
    progress_label: str = "chunk-dense",
) -> list[tuple[str, list[str], list[float]]]:
    paper_to_index = {paper_id: index for index, paper_id in enumerate(paper_ids)}
    if not paper_ids:
        return [(query_id, [], []) for query_id in query_ids]

    if chunk_pooling == "avg-topk":
        best_scores = np.full((len(query_ids), len(paper_ids), chunk_pooling_topk), -np.inf, dtype=np.float32)
    else:
        best_scores = np.full((len(query_ids), len(paper_ids)), -np.inf, dtype=np.float32)

    started_at = multiprocessing.time.time() if hasattr(multiprocessing, "time") else None
    if started_at is None:
        import time as _time
        started_at = _time.time()
    processed_chunks = 0
    batch_index = 0

    def flush_batch(batch_vectors: list[np.ndarray], batch_paper_ids: list[str]) -> None:
        nonlocal processed_chunks, batch_index
        if not batch_vectors:
            return
        batch_index += 1
        paper_indices = np.asarray([paper_to_index[paper_id] for paper_id in batch_paper_ids], dtype=np.int64)
        chunk_matrix = l2_normalize(np.stack(batch_vectors, axis=0))
        score_matrix = np.matmul(query_vectors, chunk_matrix.T).astype(np.float32, copy=False)
        if chunk_pooling == "avg-topk":
            reduced_scores, unique_papers = reduce_topk_scores_by_paper(
                score_matrix,
                paper_indices=paper_indices,
                topk=chunk_pooling_topk,
            )
            if unique_papers.size:
                existing = best_scores[:, unique_papers, :]
                combined = np.concatenate([existing, reduced_scores], axis=2)
                best_scores[:, unique_papers, :] = np.sort(combined, axis=2)[:, :, -chunk_pooling_topk:]
        else:
            reduced_scores, unique_papers = reduce_max_scores_by_paper(
                score_matrix,
                paper_indices=paper_indices,
            )
            if unique_papers.size:
                best_scores[:, unique_papers] = np.maximum(best_scores[:, unique_papers], reduced_scores)
        processed_chunks += len(batch_vectors)
        if batch_index % max(1, progress_log_every_batches) == 0:
            import time as _time

            elapsed = max(1e-6, _time.time() - started_at)
            throughput = processed_chunks / elapsed
            if total_chunk_count:
                remaining = max(0, total_chunk_count - processed_chunks)
                eta = remaining / throughput if throughput > 0 else -1.0
                pct = 100.0 * processed_chunks / total_chunk_count
                LOGGER.info(
                    "progress label=%s processed_chunks=%s total_chunks=%s pct=%.2f elapsed=%.1fs throughput=%.1f_chunks_s eta=%.1fs",
                    progress_label,
                    processed_chunks,
                    total_chunk_count,
                    pct,
                    elapsed,
                    throughput,
                    eta,
                )
            else:
                LOGGER.info(
                    "progress label=%s processed_chunks=%s elapsed=%.1fs throughput=%.1f_chunks_s",
                    progress_label,
                    processed_chunks,
                    elapsed,
                    throughput,
                )

    batch_vectors: list[np.ndarray] = []
    batch_paper_ids: list[str] = []
    for row in read_tsv_gz(chunk_embeddings_path):
        embedding_value = str(row.get("embedding") or "").strip()
        if not embedding_value:
            continue
        paper_id = canonicalize_paper_id(str(row.get("paper_id") or ""))
        if not paper_id:
            continue
        vector = parse_embedding_string(embedding_value)
        if vector.size == 0:
            continue
        batch_vectors.append(vector)
        batch_paper_ids.append(paper_id)
        if len(batch_vectors) >= chunk_batch_size:
            flush_batch(batch_vectors, batch_paper_ids)
            batch_vectors = []
            batch_paper_ids = []
    flush_batch(batch_vectors, batch_paper_ids)

    rows: list[tuple[str, list[str], list[float]]] = []
    for query_index, query_id in enumerate(query_ids):
        query_scores = collapse_paper_scores(best_scores[query_index], pooling=chunk_pooling)
        valid_mask = np.isfinite(query_scores)
        if not np.any(valid_mask):
            rows.append((query_id, [], []))
            continue
        valid_indices = np.flatnonzero(valid_mask)
        valid_scores = query_scores[valid_mask]
        order = np.argsort(-valid_scores)[:top_k]
        top_indices = valid_indices[order]
        ranked_ids = [paper_ids[int(pos)] for pos in top_indices]
        ranked_scores = [float(query_scores[int(pos)]) for pos in top_indices]
        rows.append((query_id, ranked_ids, ranked_scores))
    return rows


def _chunk_query_worker(payload: dict[str, Any]) -> list[tuple[str, list[str], list[float]]]:
    return _compute_streaming_chunk_paper_rankings_single(
        chunk_embeddings_path=Path(payload["chunk_embeddings_path"]),
        paper_ids=list(payload["paper_ids"]),
        query_ids=list(payload["query_ids"]),
        query_vectors=np.asarray(payload["query_vectors"], dtype=np.float32),
        top_k=int(payload["top_k"]),
        chunk_batch_size=int(payload["chunk_batch_size"]),
        chunk_pooling=str(payload["chunk_pooling"]),
        chunk_pooling_topk=int(payload["chunk_pooling_topk"]),
        total_chunk_count=int(payload["total_chunk_count"]) if payload.get("total_chunk_count") is not None else None,
        progress_log_every_batches=int(payload.get("progress_log_every_batches", 50)),
        progress_label=str(payload.get("progress_label", "chunk-dense")),
    )


def compute_streaming_chunk_paper_rankings(
    *,
    chunk_embeddings_path: Path,
    query_ids: list[str],
    query_vectors: np.ndarray,
    top_k: int,
    chunk_batch_size: int = 4096,
    worker_count: int = 1,
    paper_ids: list[str] | None = None,
    chunk_pooling: str = "max",
    chunk_pooling_topk: int = 3,
    total_chunk_count: int | None = None,
    progress_log_every_batches: int = 50,
    progress_label: str = "chunk-dense",
) -> list[tuple[str, list[str], list[float]]]:
    paper_ids = list(paper_ids) if paper_ids is not None else collect_chunk_paper_ids(chunk_embeddings_path)
    if not paper_ids:
        return [(query_id, [], []) for query_id in query_ids]

    shard_count = max(1, min(int(worker_count), len(query_ids)))
    if shard_count == 1:
        return _compute_streaming_chunk_paper_rankings_single(
            chunk_embeddings_path=chunk_embeddings_path,
            paper_ids=paper_ids,
            query_ids=query_ids,
            query_vectors=query_vectors,
            top_k=top_k,
            chunk_batch_size=chunk_batch_size,
            chunk_pooling=chunk_pooling,
            chunk_pooling_topk=chunk_pooling_topk,
            total_chunk_count=total_chunk_count,
            progress_log_every_batches=progress_log_every_batches,
            progress_label=progress_label,
        )

    payloads: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        shard_query_ids = query_ids[shard_index::shard_count]
        if not shard_query_ids:
            continue
        shard_query_vectors = query_vectors[shard_index::shard_count]
        payloads.append(
            {
                "chunk_embeddings_path": str(chunk_embeddings_path),
                "paper_ids": paper_ids,
                "query_ids": shard_query_ids,
                "query_vectors": shard_query_vectors,
                "top_k": top_k,
                "chunk_batch_size": chunk_batch_size,
                "chunk_pooling": chunk_pooling,
                "chunk_pooling_topk": chunk_pooling_topk,
                "total_chunk_count": total_chunk_count,
                "progress_log_every_batches": progress_log_every_batches,
                "progress_label": f"{progress_label}:shard{len(payloads)}",
            }
        )

    rows: list[tuple[str, list[str], list[float]]] = []
    with ProcessPoolExecutor(
        max_workers=len(payloads),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        for shard_rows in executor.map(_chunk_query_worker, payloads):
            rows.extend(shard_rows)
    return rows
