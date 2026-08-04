from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from utils.constants import (
    DEFAULT_EXPERIMENT_OUTPUT_ROOT,
    DEFAULT_MAPLE_1Q_QUERY_CANDIDATES,
    DEFAULT_MAPLE_1Q_RESULTS_ROOT_CANDIDATES,
    resolve_first_existing_path,
)
from utils.io import ensure_dir, safe_slug

HF_MAPLE_1Q_QUERIES_URL = "https://huggingface.co/datasets/kai-02/MAPLE/raw/main/MAPLE-1Q/queries.jsonl"
CANONICAL_VIEWS = ("motivation", "method", "experiment/result")
MAPLE_1Q_RECALL_K = 20

RESULT_SPECS = [
    {
        "model_name": "BM25",
        "paper_view": "full-text",
        "dir_candidates": ("fulltext-bm25", "bm25_full_text"),
    },
    {
        "model_name": "BGE-M3",
        "paper_view": "full-text",
        "dir_candidates": ("fulltext-bge-m3", "full_text__bge-m3"),
    },
    {
        "model_name": "SPECTER2",
        "paper_view": "full-text",
        "dir_candidates": ("fulltext-specter2", "full_text__specter2"),
    },
    {
        "model_name": "SCiNCL",
        "paper_view": "full-text",
        "dir_candidates": ("fulltext-scincl", "full_text__scincl"),
    },
    {
        "model_name": "Instructor-XL",
        "paper_view": "full-text",
        "dir_candidates": ("fulltext-instructor-xl", "full_text__instructor-xl"),
    },
    {
        "model_name": "GritLM-7B",
        "paper_view": "full-text",
        "dir_candidates": ("fulltext-gritlm-7b", "full_text__gritlm-7b"),
    },
    {
        "model_name": "Qwen3-Embed-8B",
        "paper_view": "full-text",
        "dir_candidates": ("fulltext-qwen3-embed-8b", "full_text__qwen3-embed-8b"),
    },
    {
        "model_name": "Ops-MM-Embedding-v1-7B",
        "paper_view": "screenshot",
        "dir_candidates": ("screenshot-ops-mm-embed-7b", "screenshot__ops-mm-embed-7b"),
    },
    {
        "model_name": "Qwen3-VL-Embedding-8B",
        "paper_view": "screenshot",
        "dir_candidates": ("screenshot-qwen3-vl-embed-8b", "screenshot__qwen3-vl-embed-8b"),
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluations.maple_1q.experiment")
    parser.add_argument(
        "--results-root",
        default=resolve_first_existing_path(DEFAULT_MAPLE_1Q_RESULTS_ROOT_CANDIDATES),
        help="Directory containing completed multi-aspect experiment outputs.",
    )
    parser.add_argument(
        "--query-path",
        default=resolve_first_existing_path(DEFAULT_MAPLE_1Q_QUERY_CANDIDATES),
        help="Local MAPLE-1Q query JSONL.",
    )
    parser.add_argument("--output-dir", help="Directory for MAPLE-1Q summary outputs.")
    return parser


def _apply_maple_1q_defaults(args: argparse.Namespace) -> None:
    if not hasattr(args, "query_url"):
        args.query_url = HF_MAPLE_1Q_QUERIES_URL
    if not hasattr(args, "force_download"):
        args.force_download = False


def normalize_source_view(value: str) -> str:
    raw = str(value or "").strip().lower().replace("_", "/")
    if raw == "experiment":
        raw = "experiment/result"
    return raw


def download_if_needed(target_path: Path, url: str, force: bool = False) -> Path:
    if target_path.exists() and not force:
        return target_path
    ensure_dir(target_path.parent)
    with urllib.request.urlopen(url) as response:  # nosec - fixed public dataset URL
        target_path.write_bytes(response.read())
    return target_path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_maple_1q_queries(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            query_id = str(row.get("query_id") or f"line-{line_number:06d}").strip()
            if query_id in seen:
                continue
            query_text = str(row.get("query_text") or row.get("query") or row.get("text") or "").strip()
            target_paper_id = str(row.get("paper_id") or row.get("target_paper_id") or "").strip()
            source_view = normalize_source_view(str(row.get("source_view") or row.get("aspect_group") or ""))
            if not query_text or not target_paper_id or source_view not in CANONICAL_VIEWS:
                continue
            seen.add(query_id)
            rows.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "target_paper_id": target_paper_id,
                    "source_view": source_view,
                    "is_multimodal": bool(row.get("is_multimodal", False)),
                }
            )
    return rows


def load_rankings(path: Path) -> dict[str, list[str]]:
    by_query: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            query_id = str(row.get("query_id") or "").strip()
            paper_id = str(row.get("paper_id") or "").strip()
            try:
                rank = int(row.get("rank"))
            except (TypeError, ValueError):
                continue
            if query_id and paper_id:
                by_query[query_id].append((rank, paper_id))
    resolved: dict[str, list[str]] = {}
    for query_id, pairs in by_query.items():
        pairs.sort(key=lambda item: item[0])
        resolved[query_id] = [paper_id for _, paper_id in pairs]
    return resolved


def _model_aliases(model_name: str) -> set[str]:
    normalized = safe_slug(model_name).replace("_", "-")
    aliases = {normalized}
    if normalized == "bm25":
        aliases.add("bm25")
    if normalized == "scincl":
        aliases.add("scincl")
    if normalized == "specter2":
        aliases.add("specter2")
    if normalized == "instructor-xl":
        aliases.add("instructor-xl")
    if normalized == "gritlm-7b":
        aliases.add("gritlm-7b")
    if normalized == "qwen3-embed-8b":
        aliases.add("qwen3-embed-8b")
    return aliases


def _metrics_model_name(run_dir: Path) -> str | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return str(payload.get("model_name") or "").strip() or None
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return str(payload[0].get("model_name") or "").strip() or None
    return None


def resolve_result_dir(results_root: Path, spec: dict[str, Any]) -> Path | None:
    dir_candidates = tuple(spec["dir_candidates"])
    for dirname in dir_candidates:
        candidate = results_root / dirname
        if (candidate / "rankings.jsonl").exists():
            return candidate
    aliases = _model_aliases(str(spec["model_name"]))
    for candidate in sorted(path for path in results_root.iterdir() if path.is_dir()):
        if not (candidate / "rankings.jsonl").exists():
            continue
        model_name = _metrics_model_name(candidate)
        if model_name and safe_slug(model_name).replace("_", "-") in aliases:
            return candidate
    return None


def compute_per_query_rows(
    *,
    queries: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    model_name: str,
    paper_view: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        ranked_ids = rankings.get(query["query_id"], [])
        target_rank = None
        for index, paper_id in enumerate(ranked_ids, start=1):
            if paper_id == query["target_paper_id"]:
                target_rank = index
                break
        row: dict[str, Any] = {
            "dataset_name": "MAPLE-1Q",
            "query_id": query["query_id"],
            "query_text": query["query_text"],
            "target_paper_id": query["target_paper_id"],
            "source_view": query["source_view"],
            "is_multimodal": query["is_multimodal"],
            "retriever_name": model_name,
            "paper_view": paper_view,
            "target_rank": target_rank,
            "ranked_paper_ids": ranked_ids,
            "ndcg_at_10": (1.0 / math.log2(target_rank + 1)) if (target_rank is not None and target_rank <= 10) else 0.0,
            "hit_at_20": int(target_rank is not None and target_rank <= MAPLE_1Q_RECALL_K),
        }
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_order = [spec["model_name"] for spec in RESULT_SPECS]
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["retriever_name"])
        bucket = grouped.setdefault(
            key,
            {
                "model_name": key,
                "query_count": 0,
                "ndcg_at_10_sum": 0.0,
                "hit_at_20_sum": 0,
            },
        )
        bucket["query_count"] += 1
        bucket["ndcg_at_10_sum"] += float(row["ndcg_at_10"])
        bucket["hit_at_20_sum"] += int(row["hit_at_20"])

    summaries: list[dict[str, Any]] = []
    for _, bucket in sorted(grouped.items(), key=lambda item: model_order.index(item[0])):
        q = int(bucket["query_count"])
        summaries.append(
            {
                "model_name": bucket["model_name"],
                "ndcg_at_10": (bucket["ndcg_at_10_sum"] / q) if q else 0.0,
                "recall_at_20": (bucket["hit_at_20_sum"] / q) if q else 0.0,
                "ndcg_at_10_percent": 100.0 * ((bucket["ndcg_at_10_sum"] / q) if q else 0.0),
                "recall_at_20_percent": 100.0 * ((bucket["hit_at_20_sum"] / q) if q else 0.0),
                "query_count": q,
            }
        )
    return summaries


def write_table_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model_name", "ndcg_at_10_percent", "recall_at_20_percent", "query_count"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model_name": row["model_name"],
                    "ndcg_at_10_percent": f"{float(row['ndcg_at_10_percent']):.2f}",
                    "recall_at_20_percent": f"{float(row['recall_at_20_percent']):.2f}",
                    "query_count": row["query_count"],
                }
            )


def print_summary_table(rows: list[dict[str, Any]]) -> None:
    print("Model\tnDCG@10\tRecall@20\tQueries")
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["model_name"]),
                    f"{float(row['ndcg_at_10_percent']):.2f}",
                    f"{float(row['recall_at_20_percent']):.2f}",
                    str(row["query_count"]),
                ]
            )
        )


def run_experiment(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root).expanduser().resolve()
    query_path = Path(args.query_path).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path(DEFAULT_EXPERIMENT_OUTPUT_ROOT) / "maple_1q" / safe_slug(results_root.name or "default")).resolve()
    )
    ensure_dir(output_dir)

    download_if_needed(query_path, args.query_url, force=args.force_download)
    queries = load_maple_1q_queries(query_path)

    all_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for spec in RESULT_SPECS:
        run_dir = resolve_result_dir(results_root, spec)
        if run_dir is None:
            status_rows.append(
                {
                    "retriever_name": spec["model_name"],
                    "paper_view": spec["paper_view"],
                    "status": "missing_rankings",
                    "dir_candidates": list(spec["dir_candidates"]),
                }
            )
            continue
        rankings = load_rankings(run_dir / "rankings.jsonl")
        rows = compute_per_query_rows(
            queries=queries,
            rankings=rankings,
            model_name=str(spec["model_name"]),
            paper_view=str(spec["paper_view"]),
        )
        all_rows.extend(rows)
        status_rows.append(
            {
                "retriever_name": spec["model_name"],
                "paper_view": spec["paper_view"],
                "status": "ok",
                "result_dir": str(run_dir),
                "query_count": len(rows),
            }
        )

    summary_rows = summarize_rows(all_rows)
    write_csv(output_dir / "metrics.csv", summary_rows)
    write_table_tsv(output_dir / "metrics.tsv", summary_rows)
    (output_dir / "metrics.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_summary_table(summary_rows)

    manifest = {
        "dataset_name": "MAPLE-1Q",
        "query_path": str(query_path),
        "query_url": args.query_url,
        "query_count": len(queries),
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "models_found": [row["retriever_name"] for row in status_rows if row["status"] == "ok"],
        "missing_models": [row["retriever_name"] for row in status_rows if row["status"] != "ok"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not manifest["models_found"]:
        raise RuntimeError(
            f"No standard multi-aspect result directories with rankings.jsonl were found under {results_root}. "
            "Run the standard multi_aspect experiments first, or point --results-root to their output directory."
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _apply_maple_1q_defaults(args)
    run_experiment(args)


if __name__ == "__main__":
    main()
