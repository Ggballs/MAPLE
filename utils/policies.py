from __future__ import annotations

from typing import Iterable, Protocol


class TextPaper(Protocol):
    text: str


PAPER_CHAR_LIMITS: dict[tuple[str, str], int] = {
    # Matches the released cached qwen3 paper embeddings.
    ("full-text", "qwen3-embed-8b"): 32768,
    ("full-text", "bge-m3"): 2048 * 4,
    ("full-text", "gritlm-7b"): 32768 * 4,
    ("full-text", "gritlm-full"): 32768 * 4,
    ("full-text", "instructor-xl"): 512 * 4,
    ("full-text", "specter2"): 512 * 4,
    ("full-text", "scincl"): 512 * 4,
}

EMBED_TEXT_CHAR_LIMITS: dict[str, int] = {
    "qwen3-embed-8b": 32768 * 4,
    "qwen3-vl-embed-8b": 32768 * 4,
    "bge-m3": 2048 * 4,
    "gritlm-7b": 32768 * 4,
    "gritlm-full": 32768 * 4,
    "instructor-xl": 512 * 4,
    "specter2": 512 * 4,
    "scincl": 512 * 4,
}

NORMALIZE_POLICY: dict[tuple[str, str, str], bool] = {
    ("full-text", "bge-m3", "paper"): True,
    ("full-text", "bge-m3", "query"): True,
    ("full-text", "gritlm-7b", "paper"): True,
    ("full-text", "gritlm-7b", "query"): True,
    ("full-text", "instructor-xl", "paper"): True,
    ("full-text", "instructor-xl", "query"): True,
    ("full-text", "qwen3-embed-8b", "paper"): True,
    ("full-text", "qwen3-embed-8b", "query"): True,
    ("full-text", "scincl", "paper"): False,
    ("full-text", "scincl", "query"): False,
    ("full-text", "specter2", "paper"): False,
    ("full-text", "specter2", "query"): False,
    ("screenshot", "ops-mm-embed-7b", "paper"): False,
    ("screenshot", "ops-mm-embed-7b", "query"): True,
    ("screenshot", "qwen3-vl-embed-8b", "paper"): False,
    ("screenshot", "qwen3-vl-embed-8b", "query"): True,
}


def _policy_key(representation: str) -> str:
    return {"fulltext": "full-text"}.get(representation, representation)


def should_normalize(*, representation: str, model_name: str, kind: str) -> bool:
    return NORMALIZE_POLICY.get((_policy_key(representation), model_name, kind), True)


def paper_text_limit(*, representation: str, model_name: str) -> int | None:
    return PAPER_CHAR_LIMITS.get((_policy_key(representation), model_name))


def prepare_paper_texts(*, papers: Iterable[TextPaper], representation: str, model_name: str) -> tuple[list[str], list[int]]:
    char_limit = paper_text_limit(representation=representation, model_name=model_name)
    texts: list[str] = []
    lengths: list[int] = []
    for paper in papers:
        text = paper.text
        if char_limit is not None and len(text) > char_limit:
            text = text[:char_limit]
        texts.append(text)
        lengths.append(len(text))
    return texts, lengths


def truncate_texts_for_embedding(texts: list[str], model_name: str | None) -> list[str]:
    char_limit = EMBED_TEXT_CHAR_LIMITS.get(str(model_name or "").strip())
    if char_limit is None:
        return texts
    return [text[:char_limit] if len(text) > char_limit else text for text in texts]
