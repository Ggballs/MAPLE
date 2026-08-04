from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from utils.io import ensure_dir


def _prepare_pyserini_import() -> None:
    # Pyserini 1.x imports optional OpenAI encoders while importing Lucene search.
    # BM25 does not use them, but the import path still expects a key.
    os.environ.setdefault("OPENAI_API_KEY", "unused-for-maple-bm25")


class BM25Index:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir

    def build(self, rows: Iterable[dict[str, str]]) -> None:
        _prepare_pyserini_import()
        try:
            from pyserini.index.lucene import LuceneIndexer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("BM25 indexing requires Pyserini/Lucene. Install pyserini and a working Java runtime.") from exc

        docs_dir = ensure_dir(self.index_dir / "_docs")
        docs_path = docs_dir / "docs.jsonl"
        with docs_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        indexer = LuceneIndexer(index_dir=str(self.index_dir), threads=4)
        with docs_path.open(encoding="utf-8") as handle:
            batch = [json.loads(line) for line in handle if line.strip()]
        indexer.add_batch_dict(batch)
        indexer.close()

    def search(self, query_text: str, top_k: int) -> tuple[list[str], list[float]]:
        _prepare_pyserini_import()
        try:
            from pyserini.search.lucene import LuceneSearcher
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("BM25 search requires Pyserini/Lucene. Install pyserini and a working Java runtime.") from exc

        segments = list(self.index_dir.glob("segments*"))
        if not segments:
            raise RuntimeError(f"Lucene BM25 index not found at {self.index_dir}")

        searcher = LuceneSearcher(str(self.index_dir))
        hits = searcher.search(query_text, k=top_k)
        return [hit.docid for hit in hits], [float(hit.score) for hit in hits]
