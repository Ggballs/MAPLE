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
from utils.io import load_paper_records, load_queries, safe_slug
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


def _infer_index_backend(args: argparse.Namespace) -> str:
    if args.backend:
        return str(args.backend)
    if args.model_name or args.service_url:
        return "dense"
    return "bm25"


def _default_index_output_dir(args: argparse.Namespace) -> str:
    backend = _infer_index_backend(args)
    label = str(args.model_name or backend)
    return str(Path("indexes") / "representation" / safe_slug(f"{args.level}-{args.representation}-{label}"))


def _build_chunk_records(
    papers: list[PaperRecord],
    *,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    model_name: str | None,
    tokenizer_path: str | None,
) -> list[ChunkRecord]:
    rows: list[ChunkRecord] = []
    for paper in papers:
        rows.extend(
            build_chunks(
                paper,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                model_name=model_name,
                tokenizer_path=tokenizer_path,
            )
        )
    return rows


def run_index(args: argparse.Namespace) -> None:
    args.backend = _infer_index_backend(args)
    if not args.output_dir:
        args.output_dir = _default_index_output_dir(args)
    papers, shard_paths, total_rows = load_paper_records(
        args.representations_root,
        args.representation,
        max_papers=args.max_papers,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.level == "chunk" and args.representation == "abstract":
        raise ValueError("abstract does not support chunk-level indexing in this minimal experiment layer")

    if args.level == "paper":
        meta_rows = [
            {
                "paper_id": paper.paper_id,
                "representation_type": paper.representation_type,
                "source_paths": list(paper.source_paths),
                "text_chars": len(paper.text),
            }
            for paper in papers
        ]
        texts = [paper.text for paper in papers]
        docs = [{"id": paper.paper_id, "contents": paper.text} for paper in papers]
        chunk_count = 0
    else:
        chunk_rows = _build_chunk_records(
            papers,
            chunk_size_tokens=args.chunk_size_tokens,
            chunk_overlap_tokens=args.chunk_overlap_tokens,
            model_name=args.model_name,
            tokenizer_path=None,
        )
        meta_rows = [
            {
                "chunk_id": row.chunk_id,
                "paper_id": row.paper_id,
                "representation_type": row.representation_type,
                "chunk_index": row.chunk_index,
                "source_path": row.source_path,
                "token_count": row.token_count,
                "overlap_tokens": row.overlap_tokens,
                "text_chars": len(row.text),
            }
            for row in chunk_rows
        ]
        texts = [row.text for row in chunk_rows]
        docs = [{"id": row.chunk_id, "contents": row.text} for row in chunk_rows]
        chunk_count = len(chunk_rows)

    manifest = {
        "experiment": "representation",
        "representation": args.representation,
        "level": args.level,
        "backend": args.backend,
        "model_name": args.model_name,
        "service_url": args.service_url,
        "instruction": args.instruction,
        "corpus_row_count": total_rows,
        "distinct_paper_count": len(papers),
        "chunk_count": chunk_count,
        "source_shards": shard_paths,
        "build_timestamp": time.time(),
        "tokenizer_path": None,
    }
    config = {
        "representation": args.representation,
        "level": args.level,
        "backend": args.backend,
        "service_url": args.service_url,
        "model_name": args.model_name,
        "instruction": args.instruction,
        "batch_size": args.batch_size,
        "chunk_size_tokens": args.chunk_size_tokens,
        "chunk_overlap_tokens": args.chunk_overlap_tokens,
        "tokenizer_path": None,
    }

    if args.backend == "dense":
        if not args.service_url or not args.model_name:
            raise ValueError("--service-url and --model-name are required for dense indexing")
        embedder = ServiceTextEmbedder(
            service_url=args.service_url,
            batch_size=args.batch_size,
            instruction=args.instruction,
        )
        embeddings = embedder.embed_texts(truncate_texts_for_embedding(texts, args.model_name))
        save_dense_index(
            output_dir=output_dir,
            meta_rows=meta_rows,
            embeddings=embeddings,
            manifest=manifest,
            config=config,
        )
        return

    save_bm25_index(
        output_dir=output_dir,
        docs=docs,
        meta_rows=meta_rows,
        manifest=manifest,
        config=config,
    )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluations.representation.index")
    parser.add_argument("--representation", choices=["abstract", "fulltext", "interleaved_ocr"], required=True, help="Representation view to index.")
    parser.add_argument("--level", choices=["paper", "chunk"], required=True, help="Index paper embeddings or text chunks.")
    parser.add_argument("--model-name", help="Dense model name. Omit for BM25.")
    parser.add_argument("--service-url", help="HTTP embedding endpoint. Omit for BM25.")
    parser.add_argument("--output-dir", help="Directory for the built index.")
    parser.add_argument("--max-papers", type=int, help="Optional small-corpus limit for quick tests.")
    return parser


def main_with_argv(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    _apply_representation_defaults(args)
    run_index(args)


def main() -> None:
    main_with_argv()


if __name__ == "__main__":
    main()
