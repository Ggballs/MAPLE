from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from utils.constants import CANONICAL_VIEWS
from utils.io import ensure_dir, write_json, write_jsonl
from utils.types import QueryRecord, RankedResult


def build_query_rows(queries: list[QueryRecord], results: list[RankedResult]) -> list[dict[str, Any]]:
    result_lookup = {result.query_id: result for result in results}
    rows: list[dict[str, Any]] = []
    for query in queries:
        result = result_lookup.get(query.query_id)
        ranked_ids = result.ranked_ids if result else []
        target_rank = None
        for index, paper_id in enumerate(ranked_ids, start=1):
            if paper_id == query.target_paper_id:
                target_rank = index
                break
        rows.append(
            {
                "query_id": query.query_id,
                "query_text": query.query_text,
                "target_paper_id": query.target_paper_id,
                "aspect_group": query.aspect_group,
                "target_rank": target_rank,
                "ranked_paper_ids": ranked_ids,
                "scores": result.scores if result else [],
            }
        )
    return rows


def compute_metrics(query_rows: list[dict[str, Any]], top_ks: list[int]) -> dict[str, Any]:
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        by_paper[str(row["target_paper_id"])].append(row)
        by_view[str(row["aspect_group"])].append(row)

    metrics: dict[str, Any] = {"top_k": top_ks, "query_count": len(query_rows), "paper_count": len(by_paper)}
    for k in top_ks:
        any_values: list[float] = []
        all_values: list[float] = []
        coverage_values: list[float] = []
        for rows in by_paper.values():
            hits = [row["target_rank"] is not None and int(row["target_rank"]) <= k for row in rows]
            any_values.append(1.0 if any(hits) else 0.0)
            all_values.append(1.0 if hits and all(hits) else 0.0)
            coverage_values.append((sum(1 for hit in hits if hit) / len(hits)) if hits else 0.0)
        metrics[f"AnyAspect@{k}"] = sum(any_values) / len(any_values) if any_values else 0.0
        metrics[f"AllAspect@{k}"] = sum(all_values) / len(all_values) if all_values else 0.0
        metrics[f"AspectCoverage@{k}"] = sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
        for view in CANONICAL_VIEWS:
            rows = by_view.get(view, [])
            values = [
                1.0 if row["target_rank"] is not None and int(row["target_rank"]) <= k else 0.0
                for row in rows
            ]
            label = "Experiment/Result" if view == "experiment/result" else view.title()
            metrics[f"{label} R@{k}"] = sum(values) / len(values) if values else 0.0
    return metrics


def write_experiment_outputs(
    *,
    output_dir: Path,
    metrics: dict[str, Any],
    query_rows: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
) -> None:
    ensure_dir(output_dir)
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "query_rows.jsonl", query_rows)
    write_jsonl(output_dir / "rankings.jsonl", rankings)
    table_headers = [
        "Model",
        "AllAspect@5",
        "AllAspect@20",
        "AnyAspect@5",
        "AnyAspect@20",
        "AspectCoverage@5",
        "AspectCoverage@20",
        "Motivation R@5",
        "Motivation R@20",
        "Method R@5",
        "Method R@20",
        "Experiment/Result R@5",
        "Experiment/Result R@20",
    ]
    model_name = str(metrics.get("model_name") or "model")
    row = [
        model_name,
        f"{100.0 * float(metrics.get('AllAspect@5', 0.0)):.2f}",
        f"{100.0 * float(metrics.get('AllAspect@20', 0.0)):.2f}",
        f"{100.0 * float(metrics.get('AnyAspect@5', 0.0)):.2f}",
        f"{100.0 * float(metrics.get('AnyAspect@20', 0.0)):.2f}",
        f"{100.0 * float(metrics.get('AspectCoverage@5', 0.0)):.2f}",
        f"{100.0 * float(metrics.get('AspectCoverage@20', 0.0)):.2f}",
        f"{100.0 * float(metrics.get('Motivation R@5', 0.0)):.2f}",
        f"{100.0 * float(metrics.get('Motivation R@20', 0.0)):.2f}",
        f"{100.0 * float(metrics.get('Method R@5', 0.0)):.2f}",
        f"{100.0 * float(metrics.get('Method R@20', 0.0)):.2f}",
        f"{100.0 * float(metrics.get('Experiment/Result R@5', 0.0)):.2f}",
        f"{100.0 * float(metrics.get('Experiment/Result R@20', 0.0)):.2f}",
    ]
    table_path = output_dir / "table.tsv"
    table_path.write_text("\t".join(table_headers) + "\n" + "\t".join(row) + "\n", encoding="utf-8")

