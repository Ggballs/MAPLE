from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from utils.bm25 import BM25Index
from utils.embed import ServiceTextEmbedder, l2_normalize
from utils.io import ensure_dir, read_json, write_json, write_jsonl


def write_meta_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def save_dense_index(
    *,
    output_dir: Path,
    meta_rows: list[dict[str, Any]],
    embeddings: np.ndarray,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> None:
    ensure_dir(output_dir)
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "config.json", config)
    write_meta_jsonl(output_dir / "corpus_meta.jsonl", meta_rows)
    np.save(output_dir / "embeddings.npy", embeddings)


def save_bm25_index(
    *,
    output_dir: Path,
    docs: list[dict[str, str]],
    meta_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> None:
    ensure_dir(output_dir)
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "config.json", config)
    write_meta_jsonl(output_dir / "corpus_meta.jsonl", meta_rows)
    bm25_dir = ensure_dir(output_dir / "bm25_index")
    BM25Index(bm25_dir).build(docs)


def load_meta_rows(index_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (index_dir / "corpus_meta.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_embeddings(index_dir: Path) -> np.ndarray:
    return np.load(index_dir / "embeddings.npy")


def dense_search(
    *,
    index_dir: Path,
    query_texts: list[str],
    service_url: str,
    batch_size: int,
    instruction: str | None,
    top_k: int,
) -> list[tuple[list[str], list[float]]]:
    meta_rows = load_meta_rows(index_dir)
    embeddings = l2_normalize(load_embeddings(index_dir))
    embedder = ServiceTextEmbedder(
        service_url=service_url,
        batch_size=batch_size,
        instruction=instruction,
    )
    query_embeddings = embedder.embed_texts(query_texts)
    scores = query_embeddings @ embeddings.T
    id_key = "paper_id" if meta_rows and "paper_id" in meta_rows[0] else "chunk_id"
    results: list[tuple[list[str], list[float]]] = []
    for row_scores in scores:
        top_indices = np.argsort(-row_scores)[:top_k]
        ids = [str(meta_rows[index][id_key]) for index in top_indices]
        values = [float(row_scores[index]) for index in top_indices]
        results.append((ids, values))
    return results


def bm25_search(
    *,
    index_dir: Path,
    query_texts: list[str],
    top_k: int,
) -> list[tuple[list[str], list[float]]]:
    bm25 = BM25Index(index_dir / "bm25_index")
    return [bm25.search(query_text, top_k) for query_text in query_texts]


def load_manifest(index_dir: Path) -> dict[str, Any]:
    return read_json(index_dir / "manifest.json")

