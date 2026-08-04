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


def _resolve_index_mode(args: argparse.Namespace) -> tuple[str, str]:
    model_name = str(args.model_name or "").strip()
    if model_name in SCREENSHOT_MODEL_NAMES:
        return ("screenshot", "dense")
    if model_name == "bm25":
        return ("fulltext", "bm25")
    if model_name in FULLTEXT_MODEL_NAMES:
        return ("fulltext", "dense")
    raise ValueError(f"Unsupported multi_aspect model_name: {model_name!r}")


def _cached_view_name(representation: str) -> str:
    return {"fulltext": "full-text"}.get(representation, representation)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _embedding_to_string(vector) -> str:
    values = [float(value) for value in vector]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _write_cached_dense_files(
    *,
    output_dir: Path,
    representation: str,
    model_name: str,
    service_url: str,
    instruction: str | None,
    papers,
    paper_embeddings,
    query_rows_raw: list[dict],
    query_embeddings,
) -> None:
    created_at = _now_iso()
    paper_rows = []
    for paper, vector in zip(papers, paper_embeddings):
        paper_rows.append(
            {
                "paper_id": paper.paper_id,
                "task": _cached_view_name(representation),
                "model_name": model_name,
                "embedding": _embedding_to_string(vector),
                "embedding_dim": int(len(vector)),
                "created_at": created_at,
            }
        )
    write_jsonl(output_dir / "paper_embeddings.jsonl", paper_rows)

    query_rows = []
    for row, vector in zip(query_rows_raw, query_embeddings):
        query_id = str(row.get("query_id") or row.get("query_key") or row.get("id") or "").strip()
        query_text = str(row.get("query_text") or row.get("query") or row.get("text") or "").strip()
        target_paper_id = str(row.get("target_paper_id") or row.get("paper_id") or "").strip()
        source_view = str(row.get("source_view") or row.get("aspect_group") or row.get("view") or "").strip()
        is_multimodal = bool(row.get("is_multimodal", False))
        cache_material = "\n".join(
            [
                model_name,
                service_url,
                instruction or "",
                query_text,
            ]
        )
        query_rows.append(
            {
                "cache_key": _sha1_text(cache_material),
                "query_id": query_id,
                "query_text": query_text,
                "query_text_sha1": _sha1_text(query_text),
                "model_name": model_name,
                "endpoint": service_url,
                "instruction_sha1": _sha1_text(instruction or ""),
                "instruction": instruction,
                "embedding": _embedding_to_string(vector),
                "embedding_dim": int(len(vector)),
                "created_at": created_at,
                "paper_id": target_paper_id,
                "source_view": source_view,
                "is_multimodal": is_multimodal,
            }
        )
    write_jsonl(output_dir / "query_embeddings.jsonl", query_rows)


def _write_cached_bm25_files(*, output_dir: Path, papers) -> None:
    bm25_src = output_dir / "bm25_index"
    bm25_dst = output_dir / "index_dir"
    if bm25_dst.exists():
        if bm25_dst.is_symlink() or bm25_dst.is_file():
            bm25_dst.unlink()
        elif bm25_dst.is_dir():
            shutil.rmtree(bm25_dst)
    try:
        os.symlink(bm25_src, bm25_dst, target_is_directory=True)
    except OSError:
        shutil.copytree(bm25_src, bm25_dst)

    meta_rows = []
    for paper in papers:
        meta_rows.append(
            {
                "paper_id": paper.paper_id,
                "source_path": paper.source_paths[0] if paper.source_paths else "",
                "markdown_chars": len(paper.text),
            }
        )
    write_jsonl(output_dir / "meta.jsonl", meta_rows)


def _load_screenshot_papers(root: Path) -> tuple[list[PaperRecord], list[str], int]:
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    papers: list[PaperRecord] = []
    total_rows = 0
    for paper_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        page_paths = [
            path
            for path in sorted(paper_dir.iterdir())
            if path.is_file() and path.suffix.lower() in image_suffixes
        ]
        if not page_paths:
            continue
        total_rows += len(page_paths)
        papers.append(
            PaperRecord(
                paper_id=paper_dir.name,
                representation_type="screenshot",
                text="",
                source_paths=tuple(str(path) for path in page_paths),
            )
        )
    return papers, [str(root)], total_rows


def _average_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("Expected a non-empty 2D embedding matrix to average.")
    return matrix.mean(axis=0).astype(np.float32, copy=False)


def _build_dense_screenshot_index(
    *,
    papers: list[PaperRecord],
    query_rows_raw: list[dict[str, Any]],
    service_url: str,
    model_name: str,
    instruction: str | None,
    batch_size: int,
    output_dir: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> None:
    query_instruction = resolve_query_instruction(model_name, instruction)
    paper_embedder = ServiceMultimodalEmbedder(
        service_url=service_url,
        batch_size=batch_size,
        instruction=None,
        normalize_output=should_normalize(representation="screenshot", model_name=model_name, kind="paper"),
    )
    query_embedder = ServiceMultimodalEmbedder(
        service_url=service_url,
        batch_size=batch_size,
        instruction=query_instruction,
        normalize_output=should_normalize(representation="screenshot", model_name=model_name, kind="query"),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "screenshot_progress.json"
    page_items: list[tuple[int, str]] = []
    for paper_index, paper in enumerate(papers):
        page_items.extend((paper_index, image_path) for image_path in paper.source_paths)

    vector_sums: np.ndarray | None = None
    vector_counts = np.zeros(len(papers), dtype=np.int32)
    failed_pages: list[dict[str, Any]] = []
    completed_pages = 0
    started_at = time.time()

    def write_progress() -> None:
        elapsed = max(time.time() - started_at, 1e-6)
        done_per_second = completed_pages / elapsed
        remaining = max(len(page_items) - completed_pages, 0)
        eta_seconds = remaining / done_per_second if done_per_second > 0 else None
        progress_path.write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "content": "screenshot",
                    "completed_pages": completed_pages,
                    "total_pages": len(page_items),
                    "completed_papers_with_pages": int((vector_counts > 0).sum()),
                    "total_papers": len(papers),
                    "failed_pages": len(failed_pages),
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta_seconds,
                    "updated_at": _now_iso(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def embed_batch(batch_paths: list[str]) -> np.ndarray:
        try:
            return paper_embedder.embed_images(batch_paths)
        except Exception:
            if len(batch_paths) == 1:
                raise
            midpoint = max(len(batch_paths) // 2, 1)
            left = embed_batch(batch_paths[:midpoint])
            right = embed_batch(batch_paths[midpoint:])
            if left.size == 0:
                return right
            if right.size == 0:
                return left
            return np.concatenate([left, right], axis=0)

    for start in range(0, len(page_items), batch_size):
        batch_items = page_items[start : start + batch_size]
        batch_paths = [path for _, path in batch_items]
        try:
            page_vectors = embed_batch(batch_paths)
        except Exception as exc:
            for paper_index, image_path in batch_items:
                try:
                    page_vectors = paper_embedder.embed_images([image_path])
                except Exception as single_exc:
                    failed_pages.append(
                        {
                            "paper_id": papers[paper_index].paper_id,
                            "image_path": image_path,
                            "error": repr(single_exc),
                        }
                    )
                    completed_pages += 1
                    continue
                if vector_sums is None:
                    vector_sums = np.zeros((len(papers), page_vectors.shape[1]), dtype=np.float32)
                vector_sums[paper_index] += page_vectors[0].astype(np.float32, copy=False)
                vector_counts[paper_index] += 1
                completed_pages += 1
            LOGGER.warning("Recovered screenshot batch after failure: %r", exc)
        else:
            if page_vectors.ndim != 2 or page_vectors.shape[0] != len(batch_items):
                raise ValueError(
                    "Expected "
                    f"{len(batch_items)} page embeddings, got shape {tuple(page_vectors.shape)}."
                )
            if vector_sums is None:
                vector_sums = np.zeros((len(papers), page_vectors.shape[1]), dtype=np.float32)
            for (paper_index, _), vector in zip(batch_items, page_vectors):
                vector_sums[paper_index] += vector.astype(np.float32, copy=False)
                vector_counts[paper_index] += 1
            completed_pages += len(batch_items)

        if completed_pages == len(page_items) or completed_pages % max(batch_size * 50, 1) == 0:
            write_progress()

    if vector_sums is None:
        raise ValueError("No screenshot embeddings were produced.")

    kept_indices = np.flatnonzero(vector_counts > 0)
    if len(kept_indices) != len(papers):
        LOGGER.warning(
            "Dropping %d screenshot papers with no successful page embeddings.",
            len(papers) - len(kept_indices),
        )
    paper_matrix = (
        vector_sums[kept_indices] / vector_counts[kept_indices, None].astype(np.float32)
    ).astype(np.float32, copy=False)
    kept_papers = [papers[int(index)] for index in kept_indices]
    write_progress()
    if failed_pages:
        write_jsonl(output_dir / "failed_pages.jsonl", failed_pages)

    save_dense_index(
        output_dir=output_dir,
        meta_rows=[
            {
                "paper_id": paper.paper_id,
                "representation_type": paper.representation_type,
                "source_paths": list(paper.source_paths),
                "text_chars": 0,
            }
            for paper in kept_papers
        ],
        embeddings=paper_matrix,
        manifest=manifest,
        config=config,
    )

    filtered_query_rows: list[dict[str, Any]] = []
    query_texts: list[str] = []
    for row in query_rows_raw:
        query_text = str(row.get("query_text") or row.get("query") or row.get("text") or "").strip()
        query_id = str(row.get("query_id") or row.get("query_key") or row.get("id") or "").strip()
        if not query_id or not query_text:
            continue
        filtered_query_rows.append(row)
        query_texts.append(query_text)
    query_matrix = query_embedder.embed_texts(query_texts)

    _write_cached_dense_files(
        output_dir=output_dir,
        representation="screenshot",
        model_name=model_name,
        service_url=service_url,
        instruction=query_instruction,
        papers=kept_papers,
        paper_embeddings=paper_matrix,
        query_rows_raw=filtered_query_rows,
        query_embeddings=query_matrix,
    )


def run_index(args: argparse.Namespace) -> None:
    if not args.output_dir:
        args.output_dir = str(Path("indexes") / "multi_aspect" / safe_slug(str(args.model_name)))
    output_dir = Path(args.output_dir).expanduser().resolve()
    representation, backend = _resolve_index_mode(args)
    if representation == "screenshot":
        screenshot_root = Path(args.screenshot_root).expanduser().resolve()
        preprocess_result = ensure_screenshots_ready(
            screenshot_root=screenshot_root,
            maple_root=Path(args.maple_root).expanduser().resolve(),
            output_root=screenshot_root.parent,
            workers=int(args.screenshot_preprocess_workers),
        )
        if preprocess_result:
            LOGGER.info("Triggered screenshot preprocessing: %s", preprocess_result)
        papers, shard_paths, total_rows = _load_screenshot_papers(screenshot_root)
    else:
        papers, shard_paths, total_rows = load_paper_records(
            args.representations_root,
            representation,
            max_papers=args.max_papers,
        )
    prepared_paper_texts, prepared_paper_lengths = prepare_paper_texts(
        papers=papers,
        representation=representation,
        model_name=str(args.model_name or ""),
    )
    meta_rows = [
        {
            "paper_id": paper.paper_id,
            "representation_type": paper.representation_type,
            "source_paths": list(paper.source_paths),
            "text_chars": prepared_paper_lengths[index],
        }
        for index, paper in enumerate(papers)
    ]
    manifest = {
        "experiment": "multi_aspect",
        "representation": representation,
        "level": "paper",
        "backend": backend,
        "model_name": args.model_name,
        "service_url": args.service_url,
        "instruction": args.instruction,
        "query_instruction": resolve_query_instruction(args.model_name, args.instruction),
        "corpus_row_count": total_rows,
        "distinct_paper_count": len(papers),
        "chunk_count": 0,
        "source_shards": shard_paths,
        "build_timestamp": time.time(),
    }
    config = {
        "representation": representation,
        "backend": backend,
        "service_url": args.service_url,
        "model_name": args.model_name,
        "instruction": args.instruction,
        "query_instruction": resolve_query_instruction(args.model_name, args.instruction),
        "batch_size": args.batch_size,
    }

    if backend == "dense":
        if not args.service_url or not args.model_name:
            raise ValueError("--service-url and --model-name are required for dense indexing")
        query_rows_raw = _load_multi_aspect_query_rows(args)
        query_instruction = resolve_query_instruction(args.model_name, args.instruction)
        queries = _load_multi_aspect_queries(args)
        if args.max_queries:
            queries = queries[: int(args.max_queries)]
        query_map = {query.query_id: query for query in queries}
        filtered_query_rows = []
        for row in query_rows_raw:
            query_id = str(row.get("query_id") or row.get("query_key") or row.get("id") or "").strip()
            if query_id and query_id in query_map:
                filtered_query_rows.append(row)

        if representation == "screenshot":
            _build_dense_screenshot_index(
                papers=papers,
                query_rows_raw=filtered_query_rows,
                service_url=args.service_url,
                model_name=args.model_name,
                instruction=query_instruction,
                batch_size=args.batch_size,
                output_dir=output_dir,
                manifest=manifest,
                config=config,
            )
            return

        paper_embedder = ServiceTextEmbedder(
            service_url=args.service_url,
            batch_size=args.batch_size,
            instruction=None,
            normalize_output=should_normalize(representation=representation, model_name=args.model_name, kind="paper"),
        )
        query_embedder = ServiceTextEmbedder(
            service_url=args.service_url,
            batch_size=args.batch_size,
            instruction=query_instruction,
            normalize_output=should_normalize(representation=representation, model_name=args.model_name, kind="query"),
        )
        paper_embeddings = paper_embedder.embed_texts(prepared_paper_texts)
        query_embeddings = query_embedder.embed_texts(
            [str(row.get("query_text") or row.get("query") or row.get("text") or "").strip() for row in filtered_query_rows]
        )
        save_dense_index(
            output_dir=output_dir,
            meta_rows=meta_rows,
            embeddings=paper_embeddings,
            manifest=manifest,
            config=config,
        )
        _write_cached_dense_files(
            output_dir=output_dir,
            representation=representation,
            model_name=args.model_name,
            service_url=args.service_url,
            instruction=query_instruction,
            papers=papers,
            paper_embeddings=paper_embeddings,
            query_rows_raw=filtered_query_rows,
            query_embeddings=query_embeddings,
        )
        return

    docs = [
        {
            "id": paper.paper_id,
            "contents": paper.text,
        }
        for paper in papers
    ]
    save_bm25_index(
        output_dir=output_dir,
        docs=docs,
        meta_rows=meta_rows,
        manifest=manifest,
        config=config,
    )
    _write_cached_bm25_files(output_dir=output_dir, papers=papers)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluations.multi_aspect.index")
    parser.add_argument(
        "--model-name",
        required=True,
        choices=sorted(FULLTEXT_MODEL_NAMES | SCREENSHOT_MODEL_NAMES),
        help="Retriever to index. Text models use fulltext; screenshot models use screenshots.",
    )
    parser.add_argument("--service-url", help="HTTP embedding endpoint. Not needed for bm25.")
    parser.add_argument("--output-dir", help="Directory for the built index.")
    parser.add_argument("--max-papers", type=int, help="Optional small-corpus limit for quick tests.")
    parser.add_argument("--max-queries", type=int, help="Optional query limit for quick tests.")
    return parser


def main_with_argv(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _apply_multi_aspect_defaults(args)
    run_index(args)


def main() -> None:
    main_with_argv()


if __name__ == "__main__":
    main()
