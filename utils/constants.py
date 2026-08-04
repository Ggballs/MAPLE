from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HF_DATASET_ID = "kai-02/MAPLE"
DEFAULT_DATA_ROOT = os.environ.get("MAPLE_DATA_ROOT", "data/MAPLE")
DEFAULT_PREPROCESSED_ROOT = os.environ.get("MAPLE_PREPROCESSED_ROOT", "data/MAPLE_preprocessed")
DEFAULT_CACHE_ROOT = os.environ.get("MAPLE_CACHE_ROOT", "cached_index")

DEFAULT_REPRESENTATIONS_ROOT = str(Path(DEFAULT_DATA_ROOT) / "representations")
DEFAULT_SCREENSHOT_ROOT = str(Path(DEFAULT_PREPROCESSED_ROOT) / "screenshots")
DEFAULT_MULTI_ASPECT_BUNDLE_ROOT = str(Path(DEFAULT_CACHE_ROOT) / "multi_aspect_retrieval")
DEFAULT_REPRESENTATION_BUNDLE_ROOT = str(Path(DEFAULT_CACHE_ROOT) / "representation")
DEFAULT_EXPERIMENT_OUTPUT_ROOT = "outputs/experiments"
DEFAULT_MAPLE_1Q_RESULTS_ROOT_CANDIDATES = (
    "outputs/experiments/multi_aspect",
)

DEFAULT_MULTI_ASPECT_QUERY_CANDIDATES = (
    str(Path(DEFAULT_DATA_ROOT) / "queries" / "queries_2095_all415mm.jsonl"),
    str(Path(DEFAULT_MULTI_ASPECT_BUNDLE_ROOT) / "shared" / "queries_2095_all415mm.jsonl"),
)

DEFAULT_MAPLE_1Q_QUERY_CANDIDATES = (
    str(Path(DEFAULT_DATA_ROOT) / "MAPLE-1Q" / "queries.jsonl"),
    "outputs/MAPLE-1Q/queries/queries.jsonl",
)

DEFAULT_REPRESENTATION_QUERY_CANDIDATES = (
    str(Path(DEFAULT_DATA_ROOT) / "queries" / "queries_2095.jsonl"),
    str(Path(DEFAULT_REPRESENTATION_BUNDLE_ROOT) / "queries" / "raw" / "queries_2095.jsonl"),
)

REPRESENTATION_TO_DIRNAME = {
    "abstract": "abstract",
    "fulltext": "fulltext",
    "interleaved_ocr": "interleaved_ocr",
    "screenshot": "screenshot",
}

CANONICAL_VIEWS = ("motivation", "method", "experiment/result")


def resolve_first_existing_path(candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return candidates[0]
