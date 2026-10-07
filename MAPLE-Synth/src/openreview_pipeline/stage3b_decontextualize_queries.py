from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class QueryDecontextualizer:
    """Compatibility shim for the decontextualization stage.

    The original implementation is currently absent from the refactored tree.
    Keep the stage importable so the rest of the pipeline can run; for now the
    stage behaves as a pass-through copy of the generated query dataset.
    """

    def __init__(self, llm=None, max_concurrent_papers: int = 1) -> None:
        self.llm = llm
        self.max_concurrent_papers = max(1, int(max_concurrent_papers))

    def run(self, summarized_path: Path | str, queries_path: Path | str, output_path: Path | str) -> Path:
        _ = Path(summarized_path).expanduser().resolve()
        source = Path(queries_path).expanduser().resolve()
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "stage3b_decontextualize_queries_missing_impl source=%s target=%s action=passthrough_copy",
            source,
            target,
        )
        shutil.copy2(source, target)
        return target
