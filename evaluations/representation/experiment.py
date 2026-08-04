from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from utils.chunking import build_chunks
from utils.bundles import (
    canonicalize_paper_id,
    collapse_paper_scores,
    collect_tsv_ids,
    compute_dense_rankings,
    compute_streaming_chunk_paper_rankings,
    load_tsv_embedding_map,
)
from utils.constants import (
    DEFAULT_DATA_ROOT,
    DEFAULT_EXPERIMENT_OUTPUT_ROOT,
    DEFAULT_REPRESENTATION_BUNDLE_ROOT,
    DEFAULT_REPRESENTATION_QUERY_CANDIDATES,
    DEFAULT_REPRESENTATIONS_ROOT,
    resolve_first_existing_path,
)
from utils.embed import ServiceTextEmbedder, resolve_query_instruction
from utils.hf_assets import ensure_hf_cached_index, parse_bool
from utils.index import bm25_search, dense_search, save_bm25_index, save_dense_index
from utils.io import load_paper_records, load_queries, load_query_records_from_rows, safe_slug
from utils.metrics import build_query_rows, compute_metrics, write_experiment_outputs
from utils.policies import truncate_texts_for_embedding
from utils.types import ChunkRecord, PaperRecord, RankedResult

LOGGER = logging.getLogger(__name__)
RETRIEVAL_TOP_K = [5, 20]

REPRESENTATION_EXPERIMENT_DEFAULTS = {
    "backend": None,
    "level": None,
    "representation": None,
    "model_name": None,
    "chunk_query_workers": 32,
    "chunk_batch_size": 4096,
    "chunk_pooling": "max",
    "chunk_pooling_topk": 3,
    "progress_log_every_batches": 50,
}

REPRESENTATION_PRESETS: dict[str, dict[str, str | int]] = {
    "paper-bm25-abstract": {"backend": "bm25", "level": "paper", "representation": "abstract"},
    "paper-bm25-fulltext": {"backend": "bm25", "level": "paper", "representation": "fulltext"},
    "paper-bm25-ocr": {"backend": "bm25", "level": "paper", "representation": "interleaved_ocr"},
    "paper-gritlm-abstract": {"backend": "dense", "level": "paper", "representation": "abstract", "model_name": "gritlm-7b"},
    "paper-gritlm-fulltext": {"backend": "dense", "level": "paper", "representation": "fulltext", "model_name": "gritlm-7b"},
    "paper-gritlm-ocr": {"backend": "dense", "level": "paper", "representation": "interleaved_ocr", "model_name": "gritlm-7b"},
    "paper-scincl-abstract": {"backend": "dense", "level": "paper", "representation": "abstract", "model_name": "scincl"},
    "paper-scincl-fulltext": {"backend": "dense", "level": "paper", "representation": "fulltext", "model_name": "scincl"},
    "paper-scincl-ocr": {"backend": "dense", "level": "paper", "representation": "interleaved_ocr", "model_name": "scincl"},
    "paper-qwen-abstract": {"backend": "dense", "level": "paper", "representation": "abstract", "model_name": "qwen3-vl-embed-8b"},
    "paper-qwen-fulltext": {"backend": "dense", "level": "paper", "representation": "fulltext", "model_name": "qwen3-vl-embed-8b"},
    "paper-qwen-ocr": {"backend": "dense", "level": "paper", "representation": "interleaved_ocr", "model_name": "qwen3-vl-embed-8b"},
    "chunk-bm25-fulltext-max": {"backend": "bm25", "level": "chunk", "representation": "fulltext", "chunk_pooling": "max"},
    "chunk-bm25-fulltext-avg3": {"backend": "bm25", "level": "chunk", "representation": "fulltext", "chunk_pooling": "avg-topk", "chunk_pooling_topk": 3},
    "chunk-bm25-ocr-max": {"backend": "bm25", "level": "chunk", "representation": "interleaved_ocr", "chunk_pooling": "max"},
    "chunk-bm25-ocr-avg3": {"backend": "bm25", "level": "chunk", "representation": "interleaved_ocr", "chunk_pooling": "avg-topk", "chunk_pooling_topk": 3},
    "chunk-gritlm-fulltext-max": {"backend": "dense", "level": "chunk", "representation": "fulltext", "model_name": "gritlm-7b", "chunk_pooling": "max"},
    "chunk-gritlm-fulltext-avg3": {"backend": "dense", "level": "chunk", "representation": "fulltext", "model_name": "gritlm-7b", "chunk_pooling": "avg-topk", "chunk_pooling_topk": 3},
    "chunk-gritlm-ocr-max": {"backend": "dense", "level": "chunk", "representation": "interleaved_ocr", "model_name": "gritlm-7b", "chunk_pooling": "max"},
    "chunk-gritlm-ocr-avg3": {"backend": "dense", "level": "chunk", "representation": "interleaved_ocr", "model_name": "gritlm-7b", "chunk_pooling": "avg-topk", "chunk_pooling_topk": 3},
    "chunk-scincl-fulltext-max": {"backend": "dense", "level": "chunk", "representation": "fulltext", "model_name": "scincl", "chunk_pooling": "max"},
    "chunk-scincl-fulltext-avg3": {"backend": "dense", "level": "chunk", "representation": "fulltext", "model_name": "scincl", "chunk_pooling": "avg-topk", "chunk_pooling_topk": 3},
    "chunk-scincl-ocr-max": {"backend": "dense", "level": "chunk", "representation": "interleaved_ocr", "model_name": "scincl", "chunk_pooling": "max"},
    "chunk-scincl-ocr-avg3": {"backend": "dense", "level": "chunk", "representation": "interleaved_ocr", "model_name": "scincl", "chunk_pooling": "avg-topk", "chunk_pooling_topk": 3},
    "chunk-qwen-fulltext-max": {"backend": "dense", "level": "chunk", "representation": "fulltext", "model_name": "qwen3-vl-embed-8b", "chunk_pooling": "max"},
    "chunk-qwen-fulltext-avg3": {"backend": "dense", "level": "chunk", "representation": "fulltext", "model_name": "qwen3-vl-embed-8b", "chunk_pooling": "avg-topk", "chunk_pooling_topk": 3},
    "chunk-qwen-ocr-max": {"backend": "dense", "level": "chunk", "representation": "interleaved_ocr", "model_name": "qwen3-vl-embed-8b", "chunk_pooling": "max"},
    "chunk-qwen-ocr-avg3": {"backend": "dense", "level": "chunk", "representation": "interleaved_ocr", "model_name": "qwen3-vl-embed-8b", "chunk_pooling": "avg-topk", "chunk_pooling_topk": 3},
}




def _apply_representation_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "representations_root": DEFAULT_REPRESENTATIONS_ROOT,
        "backend": None,
        "instruction": None,
        "batch_size": 32,
        "chunk_size_tokens": 512,
        "chunk_overlap_tokens": 128,
        "query_path": resolve_first_existing_path(DEFAULT_REPRESENTATION_QUERY_CANDIDATES),
        "bundle_root": DEFAULT_REPRESENTATION_BUNDLE_ROOT,
        "hf_dataset_id": "kai-02/MAPLE",
        "hf_local_dir": DEFAULT_DATA_ROOT,
        "level": getattr(args, "level", None),
        "representation": getattr(args, "representation", None),
        "model_name": getattr(args, "model_name", None),
        "chunk_query_workers": 32,
        "chunk_batch_size": 4096,
        "chunk_pooling": "max",
        "chunk_pooling_topk": 3,
        "progress_log_every_batches": 50,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)


def _resolve_representation_preset(args: argparse.Namespace) -> None:
    preset_name = str(getattr(args, "preset", "") or "").strip()
    if not preset_name:
        return
    preset = REPRESENTATION_PRESETS.get(preset_name)
    if preset is None:
        supported = ", ".join(sorted(REPRESENTATION_PRESETS))
        raise ValueError(f"Unsupported representation preset {preset_name!r}. Supported presets: {supported}")
    for key, value in preset.items():
        current_value = getattr(args, key, None)
        default_value = REPRESENTATION_EXPERIMENT_DEFAULTS.get(key, None)
        if current_value in (None, "") or current_value == default_value:
            setattr(args, key, value)
    if not getattr(args, "output_dir", None):
        setattr(
            args,
            "output_dir",
            str(Path(DEFAULT_EXPERIMENT_OUTPUT_ROOT) / "representation" / safe_slug(preset_name)),
        )


def _resolve_representation_query_path(args: argparse.Namespace) -> None:
    if args.query_path and Path(args.query_path).expanduser().exists():
        return
    bundle_root = getattr(args, "bundle_root", None)
    if bundle_root:
        candidate = Path(bundle_root).expanduser() / "queries" / "raw" / "queries_2095.jsonl"
        if candidate.exists():
            args.query_path = str(candidate)
            return
    args.query_path = resolve_first_existing_path(DEFAULT_REPRESENTATION_QUERY_CANDIDATES)


def _read_query_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _representation_query_paths(args: argparse.Namespace) -> list[Path]:
    candidates: list[Path] = []
    if getattr(args, "query_path", None):
        candidates.append(Path(args.query_path).expanduser())
    bundle_root = getattr(args, "bundle_root", None)
    if bundle_root:
        candidates.append(Path(bundle_root).expanduser() / "queries" / "raw" / "queries_2095.jsonl")
    data_root = Path(getattr(args, "hf_local_dir", DEFAULT_DATA_ROOT)).expanduser()
    candidates.append(data_root / "queries" / "queries_2095.jsonl")
    for candidate in candidates:
        if candidate.exists():
            return [candidate]
    split_paths = [
        data_root / "queries" / "text_grounded.jsonl",
        data_root / "queries" / "multimodal_grounded.jsonl",
    ]
    existing_split_paths = [path for path in split_paths if path.exists()]
    if existing_split_paths:
        return existing_split_paths
    raise FileNotFoundError(
        "No representation query file found. Expected queries_2095.jsonl or "
        "queries/text_grounded.jsonl plus queries/multimodal_grounded.jsonl."
    )


def _load_representation_queries(args: argparse.Namespace):
    paths = _representation_query_paths(args)
    if len(paths) == 1:
        return load_queries(paths[0])
    return load_query_records_from_rows(_read_query_rows(paths))


def _run_bundle_representation_experiment(args: argparse.Namespace) -> None:
    bundle_root = Path(args.bundle_root).expanduser().resolve()
    if not args.backend or not args.level or not args.representation:
        raise ValueError("Cached representation experiments require a known preset.")
    queries = _load_representation_queries(args)
    if getattr(args, "max_queries", None):
        queries = queries[: int(args.max_queries)]
    max_top_k = max(RETRIEVAL_TOP_K)
    if args.backend == "dense":
        if not args.model_name:
            raise ValueError("--model-name is required for dense bundle experiment")
        query_file = _find_query_bundle_file(bundle_root, args.model_name)
        if args.level == "paper":
            paper_file = _find_holistic_bundle_file(bundle_root, args.model_name, args.representation)
            paper_map = load_tsv_embedding_map(paper_file, id_field="paper_id")
            query_map = load_tsv_embedding_map(query_file, id_field="query_id")
            import numpy as np

            from utils.embed import l2_normalize

            paper_ids = [canonicalize_paper_id(paper_id) for paper_id in paper_map.keys()]
            paper_vectors = l2_normalize(np.stack(list(paper_map.values()), axis=0))
            query_ids = [query.query_id for query in queries if query.query_id in query_map]
            query_vectors = l2_normalize(np.stack([query_map[query_id] for query_id in query_ids], axis=0))
            dense_rows = compute_dense_rankings(
                paper_ids=paper_ids,
                paper_vectors=paper_vectors,
                query_ids=query_ids,
                query_vectors=query_vectors,
                top_k=max_top_k,
            )
            lookup = {query_id: (ranked_ids, scores) for query_id, ranked_ids, scores in dense_rows}
            results = [
                RankedResult(
                    query_id=query.query_id,
                    ranked_ids=lookup.get(query.query_id, ([], []))[0],
                    scores=lookup.get(query.query_id, ([], []))[1],
                )
                for query in queries
            ]
        else:
            chunk_file = _find_chunk_bundle_file(bundle_root, args.model_name, args.representation)
            holistic_file = _find_holistic_bundle_file(bundle_root, args.model_name, args.representation)
            _, bm25_meta_path = _find_bm25_bundle_paths(bundle_root, "chunk", args.representation)
            query_map = load_tsv_embedding_map(query_file, id_field="query_id")
            import numpy as np

            from utils.embed import l2_normalize

            query_ids = [query.query_id for query in queries if query.query_id in query_map]
            query_vectors = l2_normalize(np.stack([query_map[query_id] for query_id in query_ids], axis=0))
            paper_ids = collect_tsv_ids(holistic_file, id_field="paper_id")
            total_chunk_count = sum(1 for _ in bm25_meta_path.open(encoding="utf-8"))
            dense_rows = compute_streaming_chunk_paper_rankings(
                chunk_embeddings_path=chunk_file,
                paper_ids=paper_ids,
                query_ids=query_ids,
                query_vectors=query_vectors,
                top_k=max_top_k,
                chunk_batch_size=int(args.chunk_batch_size),
                worker_count=int(args.chunk_query_workers),
                chunk_pooling=str(args.chunk_pooling),
                chunk_pooling_topk=int(args.chunk_pooling_topk),
                total_chunk_count=total_chunk_count,
                progress_log_every_batches=int(args.progress_log_every_batches),
                progress_label=f"{args.model_name}:{args.representation}:{args.chunk_pooling}",
            )
            lookup = {query_id: (ranked_ids, scores) for query_id, ranked_ids, scores in dense_rows}
            results = [
                RankedResult(
                    query_id=query.query_id,
                    ranked_ids=lookup.get(query.query_id, ([], []))[0],
                    scores=lookup.get(query.query_id, ([], []))[1],
                )
                for query in queries
            ]
        model_label = args.model_name
    else:
        index_dir, meta_path = _find_bm25_bundle_paths(bundle_root, args.level, args.representation)
        meta_rows = _load_bundle_meta_rows(meta_path)
        ranked = [BM25Index(index_dir).search(query.query_text, max_top_k * (8 if args.level == "chunk" else 1)) for query in queries]
        if args.level == "chunk":
            chunk_to_paper = {str(row["chunk_id"]): canonicalize_paper_id(str(row["paper_id"])) for row in meta_rows}
            results = []
            for query, (ranked_ids, scores) in zip(queries, ranked):
                paper_ids, paper_scores = _aggregate_chunk_rankings(
                    ranked_ids,
                    scores,
                    chunk_to_paper,
                    max_top_k,
                    chunk_pooling=str(args.chunk_pooling),
                    chunk_pooling_topk=int(args.chunk_pooling_topk),
                )
                results.append(RankedResult(query_id=query.query_id, ranked_ids=paper_ids, scores=paper_scores))
        else:
            results = [
                RankedResult(
                    query_id=query.query_id,
                    ranked_ids=[canonicalize_paper_id(value) for value in ranked_ids],
                    scores=scores,
                )
                for query, (ranked_ids, scores) in zip(queries, ranked)
            ]
        model_label = "BM25"

    query_rows = build_query_rows(queries, results)
    metrics = compute_metrics(query_rows, RETRIEVAL_TOP_K)
    metrics["model_name"] = model_label
    rankings = [
        {
            "query_id": result.query_id,
            "paper_id": paper_id,
            "rank": rank,
            "score": score,
        }
        for result in results
        for rank, (paper_id, score) in enumerate(zip(result.ranked_ids, result.scores), start=1)
    ]
    write_experiment_outputs(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        metrics=metrics,
        query_rows=query_rows,
        rankings=rankings,
    )




def _aggregate_chunk_rankings(
    ranked_ids: list[str],
    scores: list[float],
    chunk_to_paper: dict[str, str],
    top_k: int,
    chunk_pooling: str = "max",
    chunk_pooling_topk: int = 3,
) -> tuple[list[str], list[float]]:
    if chunk_pooling == "max":
        paper_best: dict[str, float] = {}
        for chunk_id, score in zip(ranked_ids, scores):
            paper_id = chunk_to_paper.get(chunk_id)
            if not paper_id:
                continue
            previous = paper_best.get(paper_id)
            if previous is None or score > previous:
                paper_best[paper_id] = score
        ranked = sorted(paper_best.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [paper_id for paper_id, _ in ranked], [float(score) for _, score in ranked]

    if chunk_pooling != "avg-topk":
        raise ValueError(f"Unsupported chunk pooling: {chunk_pooling!r}")

    paper_scores: dict[str, list[float]] = {}
    for chunk_id, score in zip(ranked_ids, scores):
        paper_id = chunk_to_paper.get(chunk_id)
        if not paper_id:
            continue
        bucket = paper_scores.setdefault(paper_id, [])
        bucket.append(float(score))
    reduced = []
    for paper_id, values in paper_scores.items():
        top_values = sorted(values, reverse=True)[:chunk_pooling_topk]
        reduced.append((paper_id, float(sum(top_values) / max(len(top_values), 1))))
    ranked = sorted(reduced, key=lambda item: item[1], reverse=True)[:top_k]
    return [paper_id for paper_id, _ in ranked], [float(score) for _, score in ranked]


def _find_holistic_bundle_file(bundle_root: Path, model_name: str, representation: str) -> Path:
    modern = {
        ("gritlm-7b", "abstract"): bundle_root / "doc_embeddings" / "holistic" / "gritlm-7b" / "maple-abstract.tsv.gz",
        ("gritlm-7b", "fulltext"): bundle_root / "doc_embeddings" / "holistic" / "gritlm-7b" / "maple-full-text.tsv.gz",
        ("gritlm-7b", "interleaved_ocr"): bundle_root / "doc_embeddings" / "holistic" / "gritlm-7b" / "interleaved-ocr.tsv.gz",
        ("scincl", "abstract"): bundle_root / "doc_embeddings" / "holistic" / "scincl" / "maple-abstract-scincl.tsv.gz",
        ("scincl", "fulltext"): bundle_root / "doc_embeddings" / "holistic" / "scincl" / "full-text.tsv.gz",
        ("scincl", "interleaved_ocr"): bundle_root / "doc_embeddings" / "holistic" / "scincl" / "interleaved-ocr.tsv.gz",
        ("qwen3-vl-embed-8b", "abstract"): bundle_root / "doc_embeddings" / "holistic" / "qwen3-vl-embed-8b" / "maple-abstract-qwen-vl.tsv.gz",
        ("qwen3-vl-embed-8b", "fulltext"): bundle_root / "doc_embeddings" / "holistic" / "qwen3-vl-embed-8b" / "maple-full-text-qwen-vl.tsv.gz",
        ("qwen3-vl-embed-8b", "interleaved_ocr"): bundle_root / "doc_embeddings" / "holistic" / "qwen3-vl-embed-8b" / "maple-interleaved-ocr-qwen-vl.tsv.gz",
        ("qwen3-vl-embed-8b", "screenshot"): bundle_root / "doc_embeddings" / "holistic" / "qwen3-vl-embed-8b" / "maple-screenshot-qwen-vl.tsv.gz",
        ("qwen3-vl-embed-8b", "interleaved_multimodal"): bundle_root / "doc_embeddings" / "holistic" / "qwen3-vl-embed-8b" / "interleaved-multimodal.tsv.gz",
    }
    compat = {
        ("gritlm-7b", "abstract"): bundle_root / "doc_embeddings" / "holistic" / "paper_text_embeddings.gritlm-7b.maple-abstract.tsv.gz",
        ("gritlm-7b", "fulltext"): bundle_root / "doc_embeddings" / "holistic" / "paper_text_embeddings.gritlm-7b.maple-full-text.tsv.gz",
        ("gritlm-7b", "interleaved_ocr"): bundle_root / "doc_embeddings" / "holistic" / "paper_text_embeddings.gritlm-7b.interleaved-ocr.tsv.gz",
        ("scincl", "abstract"): bundle_root / "doc_embeddings" / "holistic" / "paper_text_embeddings.scincl.maple-abstract-scincl.tsv.gz",
        ("scincl", "fulltext"): bundle_root / "doc_embeddings" / "holistic" / "paper_text_embeddings.scincl.full-text.tsv.gz",
        ("scincl", "interleaved_ocr"): bundle_root / "doc_embeddings" / "holistic" / "paper_text_embeddings.scincl.interleaved-ocr.tsv.gz",
        ("qwen3-vl-embed-8b", "abstract"): bundle_root / "doc_embeddings" / "holistic" / "paper_text_embeddings.qwen3-vl-embed-8b.maple-abstract-qwen-vl.tsv.gz",
        ("qwen3-vl-embed-8b", "fulltext"): bundle_root / "doc_embeddings" / "holistic" / "paper_text_embeddings.qwen3-vl-embed-8b.maple-full-text-qwen-vl.tsv.gz",
        ("qwen3-vl-embed-8b", "interleaved_ocr"): bundle_root / "doc_embeddings" / "holistic" / "paper_text_embeddings.qwen3-vl-embed-8b.maple-interleaved-ocr-qwen-vl.tsv.gz",
        ("qwen3-vl-embed-8b", "screenshot"): bundle_root / "doc_embeddings" / "holistic" / "paper_text_embeddings.qwen3-vl-embed-8b.maple-screenshot-qwen-vl.tsv.gz",
        ("qwen3-vl-embed-8b", "interleaved_multimodal"): bundle_root / "doc_embeddings" / "holistic" / "paper_text_embeddings.qwen3-vl-embed-8b.interleaved-multimodal.tsv.gz",
    }
    for mapping in (modern, compat):
        path = mapping.get((model_name, representation))
        if path and path.exists():
            return path
    raise ValueError(f"Unsupported holistic bundle pair: model={model_name!r}, representation={representation!r}")


def _find_chunk_bundle_file(bundle_root: Path, model_name: str, representation: str) -> Path:
    modern = {
        ("gritlm-7b", "fulltext"): bundle_root / "doc_embeddings" / "chunks" / "gritlm-7b" / "text.maple-ft-gritlm-c512-o128.tsv.gz",
        ("gritlm-7b", "interleaved_ocr"): bundle_root / "doc_embeddings" / "chunks" / "gritlm-7b" / "text.maple-ocr-gritlm-c512-o128.tsv.gz",
        ("scincl", "fulltext"): bundle_root / "doc_embeddings" / "chunks" / "scincl" / "text.maple-ft-scincl-c512-o128.tsv.gz",
        ("scincl", "interleaved_ocr"): bundle_root / "doc_embeddings" / "chunks" / "scincl" / "text.maple-ocr-scincl-c512-o128.tsv.gz",
        ("qwen3-vl-embed-8b", "fulltext"): bundle_root / "doc_embeddings" / "chunks" / "qwen3-vl-embed-8b" / "multimodal.maple-ft-qwen-vl-chunk-c512-o128.tsv.gz",
        ("qwen3-vl-embed-8b", "interleaved_ocr"): bundle_root / "doc_embeddings" / "chunks" / "qwen3-vl-embed-8b" / "multimodal.maple-ocr-qwen-vl-chunk-c512-o128.tsv.gz",
    }
    compat = {
        ("gritlm-7b", "fulltext"): bundle_root / "doc_embeddings" / "chunks" / "paper_text_embedding_chunks.gritlm-7b.maple-ft-gritlm-c512-o128.tsv.gz",
        ("gritlm-7b", "interleaved_ocr"): bundle_root / "doc_embeddings" / "chunks" / "paper_text_embedding_chunks.gritlm-7b.maple-ocr-gritlm-c512-o128.tsv.gz",
        ("scincl", "fulltext"): bundle_root / "doc_embeddings" / "chunks" / "paper_text_embedding_chunks.scincl.maple-ft-scincl-c512-o128.tsv.gz",
        ("scincl", "interleaved_ocr"): bundle_root / "doc_embeddings" / "chunks" / "paper_text_embedding_chunks.scincl.maple-ocr-scincl-c512-o128.tsv.gz",
        ("qwen3-vl-embed-8b", "fulltext"): bundle_root / "doc_embeddings" / "chunks" / "paper_multimodal_embedding_chunks.qwen3-vl-embed-8b.maple-ft-qwen-vl-chunk-c512-o128.tsv.gz",
        ("qwen3-vl-embed-8b", "interleaved_ocr"): bundle_root / "doc_embeddings" / "chunks" / "paper_multimodal_embedding_chunks.qwen3-vl-embed-8b.maple-ocr-qwen-vl-chunk-c512-o128.tsv.gz",
    }
    for mapping in (modern, compat):
        path = mapping.get((model_name, representation))
        if path and path.exists():
            return path
    raise ValueError(f"Unsupported chunk bundle pair: model={model_name!r}, representation={representation!r}")


def _find_query_bundle_file(bundle_root: Path, model_name: str) -> Path:
    modern = bundle_root / "queries" / "embeddings" / model_name / "query_embeddings.tsv.gz"
    if modern.exists():
        return modern
    compat = {
        "gritlm-7b": bundle_root / "queries" / "gritlm-7b.query_embeddings.q2095.paper-search-instruction.tsv.gz",
        "qwen3-vl-embed-8b": bundle_root / "queries" / "qwen3-vl-embed-8b.query_embeddings.q2095.paper-search-instruction.tsv.gz",
        "scincl": bundle_root / "queries" / "scincl.query_embeddings.q2095.no-instruction.tsv.gz",
    }.get(model_name)
    if compat and compat.exists():
        return compat
    raise ValueError(f"Unsupported query bundle model: {model_name!r}")


def _find_bm25_bundle_paths(bundle_root: Path, level: str, representation: str) -> tuple[Path, Path]:
    if level == "paper":
        modern = {
            "abstract": (bundle_root / "bm25" / "papers" / "abstract" / "index_dir", bundle_root / "bm25" / "papers" / "abstract" / "meta.jsonl"),
            "fulltext": (bundle_root / "bm25" / "papers" / "fulltext" / "index_dir", bundle_root / "bm25" / "papers" / "fulltext" / "meta.jsonl"),
            "interleaved_ocr": (bundle_root / "bm25" / "papers" / "fulltext_ocr" / "index_dir", bundle_root / "bm25" / "papers" / "fulltext_ocr" / "meta.jsonl"),
        }
        compat = {
            "fulltext": (bundle_root / "bm25" / "papers" / "fulltext_from_chunks.index", bundle_root / "bm25" / "papers" / "fulltext_from_chunks.meta.jsonl"),
            "interleaved_ocr": (bundle_root / "bm25" / "papers" / "ocr_from_chunks.index", bundle_root / "bm25" / "papers" / "ocr_from_chunks.meta.jsonl"),
        }
    else:
        modern = {
            "fulltext": (bundle_root / "bm25" / "chunks" / "fulltext_gritlm" / "index_dir", bundle_root / "bm25" / "chunks" / "fulltext_gritlm" / "meta.jsonl"),
            "interleaved_ocr": (bundle_root / "bm25" / "chunks" / "ocr_gritlm" / "index_dir", bundle_root / "bm25" / "chunks" / "ocr_gritlm" / "meta.jsonl"),
        }
        compat = {
            "fulltext": (bundle_root / "bm25" / "chunks" / "fulltext_gritlm.index", bundle_root / "bm25" / "chunks" / "fulltext_gritlm.meta.jsonl"),
            "interleaved_ocr": (bundle_root / "bm25" / "chunks" / "ocr_gritlm.index", bundle_root / "bm25" / "chunks" / "ocr_gritlm.meta.jsonl"),
        }
    for mapping in (modern, compat):
        paths = mapping.get(representation)
        if paths and paths[0].exists() and paths[1].exists():
            return paths
    raise ValueError(f"Unsupported BM25 bundle paths: level={level!r}, representation={representation!r}")


def _load_bundle_meta_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_experiment(args: argparse.Namespace) -> None:
    if getattr(args, "online_cached_index", False):
        args.bundle_root = str(
            ensure_hf_cached_index(
                domain="representation",
                dataset_id=str(args.hf_dataset_id),
                local_dir=args.hf_local_dir,
            )
        )
    _resolve_representation_preset(args)
    _resolve_representation_query_path(args)
    if not args.output_dir:
        target = args.model_name or args.backend or "run"
        scope = f"{args.level or 'level'}-{args.representation or 'repr'}-{target}"
        if getattr(args, "chunk_pooling", None):
            scope = f"{scope}-{args.chunk_pooling}"
        args.output_dir = str(Path(DEFAULT_EXPERIMENT_OUTPUT_ROOT) / "representation" / safe_slug(scope))

    if not args.index_dir and args.bundle_root:
        _run_bundle_representation_experiment(args)
        return

    if not args.index_dir:
        raise ValueError("--index-dir is required unless --bundle-root is provided")

    index_dir = Path(args.index_dir).expanduser().resolve()
    config_payload = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    backend = str(config_payload["backend"])
    level = str(config_payload["level"])
    queries = _load_representation_queries(args)
    if args.max_queries:
        queries = queries[: int(args.max_queries)]
    query_texts = [query.query_text for query in queries]
    max_top_k = max(RETRIEVAL_TOP_K)

    if backend == "dense":
        service_url = args.service_url or config_payload.get("service_url")
        if not service_url:
            raise ValueError("Dense experiment requires service URL from CLI or index config")
        ranked = dense_search(
            index_dir=index_dir,
            query_texts=query_texts,
            service_url=str(service_url),
            batch_size=int(args.batch_size),
            instruction=resolve_query_instruction(
                str(config_payload.get("model_name") or args.model_name or ""),
                args.instruction
                if args.instruction is not None
                else config_payload.get("query_instruction", config_payload.get("instruction")),
            ),
            top_k=max_top_k * (8 if level == "chunk" else 1),
        )
    else:
        ranked = bm25_search(
            index_dir=index_dir,
            query_texts=query_texts,
            top_k=max_top_k * (8 if level == "chunk" else 1),
        )

    import json as _json

    meta_rows = []
    with (index_dir / "corpus_meta.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                meta_rows.append(_json.loads(line))

    if level == "chunk":
        chunk_to_paper = {str(row["chunk_id"]): str(row["paper_id"]) for row in meta_rows}
        collapsed = [
            _aggregate_chunk_rankings(
                ranked_ids,
                scores,
                chunk_to_paper,
                max_top_k,
                chunk_pooling=str(args.chunk_pooling),
                chunk_pooling_topk=int(args.chunk_pooling_topk),
            )
            for ranked_ids, scores in ranked
        ]
        results = [
            RankedResult(query_id=query.query_id, ranked_ids=paper_ids, scores=paper_scores)
            for query, (paper_ids, paper_scores) in zip(queries, collapsed)
        ]
    else:
        results = [
            RankedResult(query_id=query.query_id, ranked_ids=ranked_ids, scores=scores)
            for query, (ranked_ids, scores) in zip(queries, ranked)
        ]

    query_rows = build_query_rows(queries, results)
    metrics = compute_metrics(query_rows, RETRIEVAL_TOP_K)
    metrics["model_name"] = config_payload.get("model_name") or safe_slug(index_dir.name)
    rankings = [
        {
            "query_id": result.query_id,
            "paper_id": paper_id,
            "rank": rank,
            "score": score,
        }
        for result in results
        for rank, (paper_id, score) in enumerate(zip(result.ranked_ids, result.scores), start=1)
    ]
    write_experiment_outputs(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        metrics=metrics,
        query_rows=query_rows,
        rankings=rankings,
    )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluations.representation.experiment")
    parser.add_argument("preset", nargs="?", help="Cached preset name, such as paper-gritlm-fulltext.")
    parser.add_argument("--index-dir", help="Local index directory. Omit when using a cached preset.")
    parser.add_argument("--output-dir", help="Directory for metrics and rankings.")
    parser.add_argument("--service-url", help="HTTP embedding endpoint for dense local indexes.")
    parser.add_argument("--online-cached-index", nargs="?", const=True, default=False, type=parse_bool, help="Download and use released cached indexes.")
    parser.add_argument("--max-queries", type=int, help="Optional query limit for quick tests.")
    return parser


def main_with_argv(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    _apply_representation_defaults(args)
    run_experiment(args)


def main() -> None:
    main_with_argv()


if __name__ == "__main__":
    main()
