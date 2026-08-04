from __future__ import annotations

import concurrent.futures as cf
import json
import os
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Iterator

from utils.pdf_screenshots import export_screenshots

DEFAULT_MAPLE_ROOT = Path(os.environ.get("MAPLE_DATA_ROOT", "data/MAPLE"))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("MAPLE_PREPROCESSED_ROOT", "data/MAPLE_preprocessed"))
DEFAULT_WORKERS = 64


def iter_corpus_rows(corpus_dir: Path) -> Iterator[dict]:
    for path in sorted(corpus_dir.glob("corpus-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def existing_shard_set(maple_root: Path) -> set[str]:
    shard_root = maple_root / "pdf_shards"
    return {str(path.relative_to(maple_root)) for path in shard_root.glob("*.tar.gz")}


def screenshot_job_rows(
    *,
    maple_root: Path,
    screenshot_root: Path,
) -> list[tuple[str, str, str]]:
    corpus_dir = maple_root / "corpus"
    existing_shards = existing_shard_set(maple_root)
    jobs: list[tuple[str, str, str]] = []
    for row in iter_corpus_rows(corpus_dir):
        paper_id = str(row.get("paper_id") or "").strip()
        member = str(row.get("pdf_path") or "").strip()
        shard_rel = str(row.get("belonging_shard") or "").strip()
        if not paper_id or not member or not shard_rel:
            continue
        if shard_rel not in existing_shards:
            continue
        out_dir = screenshot_root / paper_id
        if (out_dir / "_SUCCESS.json").exists():
            continue
        jobs.append((paper_id, shard_rel, member))
    return jobs


def _write_status(status_root: Path, paper_id: str, payload: dict) -> None:
    status_root.mkdir(parents=True, exist_ok=True)
    path = status_root / f"{paper_id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_summary(
    summary_path: Path,
    *,
    total: int,
    done: int,
    failed: int,
    processing: int,
    pending: int,
    elapsed: float,
    recent_rate: float | None,
    workers: int,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "total": total,
        "completed": done,
        "failed": failed,
        "processing": processing,
        "pending": pending,
        "workers": workers,
        "elapsed_sec": round(elapsed, 3),
        "done_per_min": round(recent_rate, 3) if recent_rate is not None else None,
        "updated_at": time.time(),
    }
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(summary_path)


def _process_one(
    *,
    maple_root: Path,
    output_root: Path,
    paper_id: str,
    shard_rel: str,
    member: str,
) -> dict:
    screenshot_root = output_root / "screenshots"
    status_root = output_root / "status"
    shard_path = maple_root / shard_rel
    out_dir = screenshot_root / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    payload = {
        "paper_id": paper_id,
        "shard_path": str(shard_path),
        "member": member,
        "status": "processing",
        "started_at": start,
    }
    _write_status(status_root, paper_id, payload)
    try:
        tmp_parent = os.environ.get("MAPLE_TMP_DIR")
        with tempfile.TemporaryDirectory(
            prefix=f"maple_ss_{paper_id}_",
            dir=tmp_parent if tmp_parent else None,
        ) as tmpdir:
            pdf_tmp = Path(tmpdir) / f"{paper_id}.pdf"
            with tarfile.open(shard_path, "r:gz") as tar:
                src = tar.extractfile(member)
                if src is None:
                    raise FileNotFoundError(f"missing member in shard: {member}")
                pdf_tmp.write_bytes(src.read())
            page_count = export_screenshots(pdf_tmp, out_dir)
        done = {
            "paper_id": paper_id,
            "shard_path": str(shard_path),
            "member": member,
            "status": "completed",
            "page_count": int(page_count),
            "elapsed_sec": round(time.time() - start, 3),
            "output_dir": str(out_dir),
            "finished_at": time.time(),
        }
        _write_status(status_root, paper_id, done)
        (out_dir / "_SUCCESS.json").write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
        return done
    except Exception as exc:  # noqa: BLE001
        fail = {
            "paper_id": paper_id,
            "shard_path": str(shard_path),
            "member": member,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": round(time.time() - start, 3),
            "finished_at": time.time(),
        }
        _write_status(status_root, paper_id, fail)
        return fail


def export_missing_screenshots(
    *,
    maple_root: Path = DEFAULT_MAPLE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    screenshot_root = output_root / "screenshots"
    summary_path = output_root / "screenshot_progress.json"
    screenshot_root.mkdir(parents=True, exist_ok=True)
    (output_root / "status").mkdir(parents=True, exist_ok=True)
    jobs = screenshot_job_rows(maple_root=maple_root, screenshot_root=screenshot_root)
    total = len(jobs)
    started = time.time()
    done = 0
    failed = 0
    completed_times: list[float] = []
    processing = 0

    _write_summary(
        summary_path,
        total=total,
        done=0,
        failed=0,
        processing=0,
        pending=total,
        elapsed=0.0,
        recent_rate=None,
        workers=workers,
    )
    if not jobs:
        return {"total": 0, "completed": 0, "failed": 0, "workers": workers, "output_root": str(output_root)}

    with cf.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_one,
                maple_root=maple_root,
                output_root=output_root,
                paper_id=paper_id,
                shard_rel=shard_rel,
                member=member,
            ): (paper_id, shard_rel, member)
            for paper_id, shard_rel, member in jobs
        }
        processing = len(futures)
        _write_summary(
            summary_path,
            total=total,
            done=done,
            failed=failed,
            processing=processing,
            pending=total - done - failed - processing,
            elapsed=time.time() - started,
            recent_rate=None,
            workers=workers,
        )
        for future in cf.as_completed(futures):
            result = future.result()
            processing -= 1
            if result.get("status") == "completed":
                done += 1
                completed_times.append(time.time())
            else:
                failed += 1
            cutoff = time.time() - 300
            completed_times = [t for t in completed_times if t >= cutoff]
            recent_rate = (len(completed_times) / 5.0) if completed_times else None
            _write_summary(
                summary_path,
                total=total,
                done=done,
                failed=failed,
                processing=processing,
                pending=total - done - failed - processing,
                elapsed=time.time() - started,
                recent_rate=recent_rate,
                workers=workers,
            )

    _write_summary(
        summary_path,
        total=total,
        done=done,
        failed=failed,
        processing=0,
        pending=total - done - failed,
        elapsed=time.time() - started,
        recent_rate=None,
        workers=workers,
    )
    return {
        "total": total,
        "completed": done,
        "failed": failed,
        "workers": workers,
        "output_root": str(output_root),
        "summary_path": str(summary_path),
    }


def ensure_screenshots_ready(
    *,
    screenshot_root: Path,
    maple_root: Path = DEFAULT_MAPLE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    workers: int = DEFAULT_WORKERS,
) -> dict | None:
    has_any_pages = False
    if screenshot_root.is_dir():
        for paper_dir in screenshot_root.iterdir():
            if not paper_dir.is_dir():
                continue
            if any(path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} for path in paper_dir.iterdir()):
                has_any_pages = True
                break
    if has_any_pages:
        return None
    return export_missing_screenshots(maple_root=maple_root, output_root=output_root, workers=workers)
