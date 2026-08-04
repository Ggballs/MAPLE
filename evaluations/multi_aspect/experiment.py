from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from utils.bm25 import BM25Index
from utils.bundles import canonicalize_paper_id, compute_dense_rankings, load_bundle_dense_arrays
from utils.constants import (
    DEFAULT_DATA_ROOT,
    DEFAULT_EXPERIMENT_OUTPUT_ROOT,
    DEFAULT_MULTI_ASPECT_BUNDLE_ROOT,
    DEFAULT_MULTI_ASPECT_QUERY_CANDIDATES,
    DEFAULT_REPRESENTATIONS_ROOT,
    DEFAULT_SCREENSHOT_ROOT,
    resolve_first_existing_path,
)
from utils.embed import ServiceMultimodalEmbedder, ServiceTextEmbedder, resolve_query_instruction
from utils.hf_assets import ensure_hf_cached_index, parse_bool
from utils.index import bm25_search, dense_search, save_bm25_index, save_dense_index
from utils.io import load_paper_records, load_queries, load_query_records_from_rows, safe_slug, write_jsonl
from utils.metrics import build_query_rows, compute_metrics, write_experiment_outputs
from utils.policies import prepare_paper_texts, should_normalize
from utils.screenshots import ensure_screenshots_ready
from utils.types import PaperRecord, RankedResult

LOGGER = logging.getLogger(__name__)

FULLTEXT_MODEL_NAMES = {
    "bm25",
    "bge-m3",
    "specter2",
    "scincl",
    "instructor-xl",
    "gritlm-7b",
    "qwen3-embed-8b",
}
SCREENSHOT_MODEL_NAMES = {"ops-mm-embed-7b", "qwen3-vl-embed-8b"}
RETRIEVAL_TOP_K = [5, 20]

MULTI_ASPECT_PRESETS: dict[str, dict[str, str]] = {
    "fulltext-bm25": {"backend": "bm25", "representation": "fulltext"},
    "fulltext-bge-m3": {"backend": "dense", "representation": "fulltext", "model_name": "bge-m3"},
    "fulltext-specter2": {"backend": "dense", "representation": "fulltext", "model_name": "specter2"},
    "fulltext-scincl": {"backend": "dense", "representation": "fulltext", "model_name": "scincl"},
    "fulltext-instructor-xl": {"backend": "dense", "representation": "fulltext", "model_name": "instructor-xl"},
    "fulltext-gritlm-7b": {"backend": "dense", "representation": "fulltext", "model_name": "gritlm-7b"},
    "fulltext-qwen3-embed-8b": {"backend": "dense", "representation": "fulltext", "model_name": "qwen3-embed-8b"},
    "screenshot-ops-mm-embed-7b": {"backend": "dense", "representation": "screenshot", "model_name": "ops-mm-embed-7b"},
    "screenshot-qwen3-vl-embed-8b": {"backend": "dense", "representation": "screenshot", "model_name": "qwen3-vl-embed-8b"},
}

def _apply_multi_aspect_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "representations_root": DEFAULT_REPRESENTATIONS_ROOT,
        "screenshot_root": DEFAULT_SCREENSHOT_ROOT,
        "maple_root": DEFAULT_DATA_ROOT,
        "screenshot_preprocess_workers": 64,
        "instruction": None,
        "batch_size": 32,
        "query_path": resolve_first_existing_path(DEFAULT_MULTI_ASPECT_QUERY_CANDIDATES),
        "bundle_root": DEFAULT_MULTI_ASPECT_BUNDLE_ROOT,
        "hf_dataset_id": "kai-02/MAPLE",
        "hf_local_dir": DEFAULT_DATA_ROOT,
        "backend": None,
        "representation": None,
        "model_name": getattr(args, "model_name", None),
        "max_queries": getattr(args, "max_queries", None),
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)


def _read_query_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _multi_aspect_query_paths(args: argparse.Namespace) -> list[Path]:
    candidates: list[Path] = []
    if getattr(args, "query_path", None):
        candidates.append(Path(args.query_path).expanduser())
    bundle_root = getattr(args, "bundle_root", None)
    if bundle_root:
        root = Path(bundle_root).expanduser()
        candidates.extend(
            [
                root / "shared" / "queries_2095_all415mm.jsonl",
                root / "queries" / "raw" / "queries_2095_all415mm.jsonl",
            ]
        )
    data_root = Path(getattr(args, "maple_root", DEFAULT_DATA_ROOT)).expanduser()
    candidates.append(data_root / "queries" / "queries_2095_all415mm.jsonl")
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
        "No multi_aspect query file found. Expected queries_2095_all415mm.jsonl or "
        "queries/text_grounded.jsonl plus queries/multimodal_grounded.jsonl."
    )


def _load_multi_aspect_query_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    return _read_query_rows(_multi_aspect_query_paths(args))


def _load_multi_aspect_queries(args: argparse.Namespace):
    paths = _multi_aspect_query_paths(args)
    if len(paths) == 1:
        return load_queries(paths[0])
    return load_query_records_from_rows(_read_query_rows(paths))


def _resolve_multi_aspect_preset(args: argparse.Namespace) -> None:
    preset_name = str(getattr(args, "preset", "") or "").strip()
    if not preset_name:
        return
    preset = MULTI_ASPECT_PRESETS.get(preset_name)
    if preset is None:
        supported = ", ".join(sorted(MULTI_ASPECT_PRESETS))
        raise ValueError(f"Unsupported multi_aspect preset {preset_name!r}. Supported presets: {supported}")
    for key, value in preset.items():
        if getattr(args, key, None) in (None, ""):
            setattr(args, key, value)
    if not getattr(args, "output_dir", None):
        setattr(
            args,
            "output_dir",
            str(Path(DEFAULT_EXPERIMENT_OUTPUT_ROOT) / "multi_aspect" / safe_slug(preset_name)),
        )


def _resolve_multi_aspect_query_path(args: argparse.Namespace) -> None:
    if args.query_path and Path(args.query_path).expanduser().exists():
        return
    bundle_root = getattr(args, "bundle_root", None)
    if bundle_root:
        root = Path(bundle_root).expanduser()
        for candidate in (
            root / "shared" / "queries_2095_all415mm.jsonl",
            root / "queries" / "raw" / "queries_2095_all415mm.jsonl",
        ):
            if candidate.exists():
                args.query_path = str(candidate)
                return
    args.query_path = resolve_first_existing_path(DEFAULT_MULTI_ASPECT_QUERY_CANDIDATES)


def _run_bundle_experiment(args: argparse.Namespace) -> None:
    bundle_root = Path(args.bundle_root).expanduser().resolve()
    if not args.backend:
        raise ValueError("Cached multi_aspect experiments require a known preset.")
    queries_raw = _load_multi_aspect_query_rows(args)
    queries = _load_multi_aspect_queries(args)
    if getattr(args, "max_queries", None):
        queries = queries[: int(args.max_queries)]
        allowed_query_ids = {query.query_id for query in queries}
        queries_raw = [
            row
            for row in queries_raw
            if str(row.get("query_id") or row.get("query_key") or row.get("id") or "").strip() in allowed_query_ids
        ]
    query_texts = [query.query_text for query in queries]
    if args.backend == "dense":
        if not args.representation or not args.model_name:
            raise ValueError("Dense cached experiments require a preset with representation and model_name.")
        cached_view = _cached_view_name(args.representation)
        try:
            paper_path, query_path = _find_multi_aspect_dense_bundle_files(bundle_root, cached_view, args.model_name)
        except FileNotFoundError:
            if not getattr(args, "online_cached_index", False):
                raise
            bundle_root = ensure_hf_cached_index(
                domain="multi_aspect_retrieval",
                dataset_id=str(args.hf_dataset_id),
                local_dir=args.hf_local_dir,
                force_download=True,
                allow_patterns=[
                    f"cached_index/multi_aspect_retrieval/{cached_view}/{args.model_name}/**",
                    f"cached_index/multi_aspect_retrieval/doc_embeddings/{cached_view}/{args.model_name}/**",
                    "cached_index/multi_aspect_retrieval/shared/queries_2095_all415mm.jsonl",
                    "cached_index/multi_aspect_retrieval/queries/raw/queries_2095_all415mm.jsonl",
                ],
            )
            paper_path, query_path = _find_multi_aspect_dense_bundle_files(bundle_root, cached_view, args.model_name)
        paper_ids, paper_vectors, query_ids, query_vectors = load_bundle_dense_arrays(
            paper_embeddings_path=paper_path,
            query_embeddings_path=query_path,
            query_rows=queries_raw,
            paper_id_field="paper_id",
            query_id_field="query_id",
        )
        dense_rows = compute_dense_rankings(
            paper_ids=paper_ids,
            paper_vectors=paper_vectors,
            query_ids=query_ids,
            query_vectors=query_vectors,
            top_k=max(RETRIEVAL_TOP_K),
        )
        ranked_lookup = {query_id: (paper_ids, scores) for query_id, paper_ids, scores in dense_rows}
        results = [
            RankedResult(
                query_id=query.query_id,
                ranked_ids=ranked_lookup.get(query.query_id, ([], []))[0],
                scores=ranked_lookup.get(query.query_id, ([], []))[1],
            )
            for query in queries
        ]
        model_label = args.model_name
    else:
        try:
            index_dir = _find_multi_aspect_bm25_index_dir(bundle_root)
        except FileNotFoundError:
            if not getattr(args, "online_cached_index", False):
                raise
            bundle_root = ensure_hf_cached_index(
                domain="multi_aspect_retrieval",
                dataset_id=str(args.hf_dataset_id),
                local_dir=args.hf_local_dir,
                force_download=True,
                allow_patterns=[
                    "cached_index/multi_aspect_retrieval/bm25/full-text/**",
                    "cached_index/multi_aspect_retrieval/bm25/papers/full-text/**",
                    "cached_index/multi_aspect_retrieval/shared/queries_2095_all415mm.jsonl",
                    "cached_index/multi_aspect_retrieval/queries/raw/queries_2095_all415mm.jsonl",
                ],
            )
            index_dir = _find_multi_aspect_bm25_index_dir(bundle_root)
        searcher = BM25Index(index_dir)
        ranked = [searcher.search(text, max(RETRIEVAL_TOP_K)) for text in query_texts]
        results = []
        for query, (ranked_ids, scores) in zip(queries, ranked):
            canonical_ids = [canonicalize_paper_id(paper_id) for paper_id in ranked_ids]
            results.append(RankedResult(query_id=query.query_id, ranked_ids=canonical_ids, scores=scores))
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




def _cached_view_name(representation: str) -> str:
    return {"fulltext": "full-text"}.get(representation, representation)


def _find_multi_aspect_dense_bundle_files(bundle_root: Path, cached_view: str, model_name: str) -> tuple[Path, Path]:
    candidates = [
        (
            bundle_root / cached_view / model_name / "paper_embeddings.jsonl",
            bundle_root / cached_view / model_name / "query_embeddings.jsonl",
        ),
        (
            bundle_root / "doc_embeddings" / cached_view / model_name / "paper_embeddings.jsonl",
            bundle_root / "doc_embeddings" / cached_view / model_name / "query_embeddings.jsonl",
        ),
    ]
    for paper_path, query_path in candidates:
        if paper_path.exists() and query_path.exists():
            return paper_path, query_path
    checked = ", ".join(str(pair[0].parent) for pair in candidates)
    raise FileNotFoundError(f"Cached dense files not found for representation={cached_view!r}, model={model_name!r}. Checked: {checked}")


def _find_multi_aspect_bm25_index_dir(bundle_root: Path) -> Path:
    candidates = [
        bundle_root / "bm25" / "full-text" / "index_dir",
        bundle_root / "bm25" / "papers" / "full-text" / "index_dir",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cached BM25 index_dir not found under {bundle_root}")


def run_experiment(args: argparse.Namespace) -> None:
    if getattr(args, "online_cached_index", False):
        args.bundle_root = str(
            ensure_hf_cached_index(
                domain="multi_aspect_retrieval",
                dataset_id=str(args.hf_dataset_id),
                local_dir=args.hf_local_dir,
            )
        )
    _resolve_multi_aspect_preset(args)
    _resolve_multi_aspect_query_path(args)
    if not args.output_dir:
        target = args.model_name or args.backend or "run"
        representation = args.representation or "default"
        args.output_dir = str(Path(DEFAULT_EXPERIMENT_OUTPUT_ROOT) / "multi_aspect" / safe_slug(f"{representation}-{target}"))

    if not args.index_dir and args.bundle_root:
        _run_bundle_experiment(args)
        return

    if not args.index_dir:
        raise ValueError("--index-dir is required unless --bundle-root is provided")

    index_dir = Path(args.index_dir).expanduser().resolve()
    queries = _load_multi_aspect_queries(args)
    if args.max_queries:
        queries = queries[: int(args.max_queries)]
    query_texts = [query.query_text for query in queries]
    config_payload = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    backend = str(config_payload["backend"])
    if backend == "dense":
        service_url = args.service_url or config_payload.get("service_url")
        if not service_url:
            raise ValueError("Dense experiment requires service URL from CLI or index config")
        model_name = str(config_payload.get("model_name") or args.model_name or "")
        query_instruction = resolve_query_instruction(
            model_name,
            args.instruction
            if args.instruction is not None
            else config_payload.get("query_instruction", config_payload.get("instruction")),
        )
        ranked = dense_search(
            index_dir=index_dir,
            query_texts=query_texts,
            service_url=str(service_url),
            batch_size=int(args.batch_size),
            instruction=query_instruction,
            top_k=max(RETRIEVAL_TOP_K),
        )
    else:
        ranked = bm25_search(index_dir=index_dir, query_texts=query_texts, top_k=max(RETRIEVAL_TOP_K))

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
    parser = argparse.ArgumentParser(prog="python -m evaluations.multi_aspect.experiment")
    parser.add_argument("preset", nargs="?", help="Cached preset name, such as fulltext-gritlm-7b.")
    parser.add_argument("--index-dir", help="Local index directory. Omit when using a cached preset.")
    parser.add_argument("--output-dir", help="Directory for metrics and rankings.")
    parser.add_argument("--service-url", help="HTTP embedding endpoint for dense local indexes.")
    parser.add_argument(
        "--online-cached-index",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Download and use released cached indexes.",
    )
    parser.add_argument("--max-queries", type=int, help="Optional query limit for quick tests.")
    return parser


def main_with_argv(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _apply_multi_aspect_defaults(args)
    run_experiment(args)


def main() -> None:
    main_with_argv()


if __name__ == "__main__":
    main()
