from __future__ import annotations

from pathlib import Path

from utils.constants import DEFAULT_DATA_ROOT, DEFAULT_HF_DATASET_ID


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def ensure_hf_cached_index(
    *,
    domain: str,
    dataset_id: str = DEFAULT_HF_DATASET_ID,
    local_dir: str | Path = DEFAULT_DATA_ROOT,
    force_download: bool = False,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Download one cached-index subtree from the MAPLE HF dataset if needed."""
    if domain not in {"multi_aspect_retrieval", "representation"}:
        raise ValueError(f"Unsupported cached-index domain: {domain!r}")
    root = Path(local_dir).expanduser()
    target = root / "cached_index" / domain
    if target.exists() and not force_download:
        return target

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install huggingface-hub or run `huggingface-cli download kai-02/MAPLE --repo-type dataset --local-dir data/MAPLE` first."
        ) from exc

    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=str(root),
        allow_patterns=allow_patterns or [f"cached_index/{domain}/**"],
        force_download=force_download,
    )
    if not target.exists():
        raise FileNotFoundError(f"HF cached index download finished, but expected path is missing: {target}")
    return target
