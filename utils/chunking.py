from __future__ import annotations

import os
from functools import lru_cache

from utils.types import ChunkRecord, PaperRecord

DEFAULT_FALLBACK_TOKENIZER_PATH = os.environ.get(
    "MAPLE_CHUNK_TOKENIZER_PATH",
    "Qwen/Qwen3-Embedding-8B",
)

DEFAULT_TOKENIZER_PATHS = {
    "qwen3-vl-embed-8b": os.environ.get("QWEN3_EMBED_TOKENIZER_PATH", "Qwen/Qwen3-Embedding-8B"),
    "qwen3-embed-8b": os.environ.get("QWEN3_EMBED_TOKENIZER_PATH", "Qwen/Qwen3-Embedding-8B"),
    "gritlm-7b": os.environ.get("GRITLM_TOKENIZER_PATH", "GritLM/GritLM-7B"),
    "gritlm-full": os.environ.get("GRITLM_TOKENIZER_PATH", "GritLM/GritLM-7B"),
    "bge-m3": os.environ.get("BGE_M3_TOKENIZER_PATH", "BAAI/bge-m3"),
    "instructor-xl": os.environ.get("INSTRUCTOR_XL_TOKENIZER_PATH", "hkunlp/instructor-xl"),
    "specter2": os.environ.get("SPECTER2_TOKENIZER_PATH", "allenai/specter2_base"),
    "scincl": os.environ.get("SCINCL_TOKENIZER_PATH", "malteos/scincl"),
}


def clean_text(value: str) -> str:
    return value.replace("\x00", "")


@lru_cache(maxsize=16)
def load_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def resolve_tokenizer_path(*, model_name: str | None, tokenizer_path: str | None) -> str:
    if tokenizer_path:
        return str(tokenizer_path)
    normalized = str(model_name or "").strip().lower()
    if normalized in DEFAULT_TOKENIZER_PATHS:
        return DEFAULT_TOKENIZER_PATHS[normalized]
    return DEFAULT_FALLBACK_TOKENIZER_PATH


def build_chunks(
    paper: PaperRecord,
    *,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    model_name: str | None = None,
    tokenizer_path: str | None = None,
) -> list[ChunkRecord]:
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if chunk_overlap_tokens < 0 or chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must satisfy 0 <= overlap < chunk_size")

    resolved_tokenizer_path = resolve_tokenizer_path(model_name=model_name, tokenizer_path=tokenizer_path)
    try:
        tokenizer = load_tokenizer(resolved_tokenizer_path)
    except OSError:
        if tokenizer_path or resolved_tokenizer_path == DEFAULT_FALLBACK_TOKENIZER_PATH:
            raise
        tokenizer = load_tokenizer(DEFAULT_FALLBACK_TOKENIZER_PATH)
    text = clean_text(paper.text)
    if not text.strip():
        return []

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    stride = chunk_size_tokens - chunk_overlap_tokens
    chunks: list[ChunkRecord] = []
    source_path = paper.source_paths[0] if paper.source_paths else ""
    chunk_index = 0
    for start in range(0, len(token_ids), stride):
        end = min(start + chunk_size_tokens, len(token_ids))
        window_ids = token_ids[start:end]
        chunk_text = clean_text(tokenizer.decode(window_ids, skip_special_tokens=True).strip())
        if chunk_text:
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{paper.representation_type}::{paper.paper_id}::chunk::{chunk_index:04d}",
                    paper_id=paper.paper_id,
                    representation_type=paper.representation_type,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    source_path=source_path,
                    token_count=len(window_ids),
                    overlap_tokens=0 if chunk_index == 0 else chunk_overlap_tokens,
                )
            )
            chunk_index += 1
        if end >= len(token_ids):
            break
    return chunks
