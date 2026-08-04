from __future__ import annotations

import json
from urllib import request
from urllib.parse import urlparse
from typing import Optional, Protocol


class TextEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class BGEM3Embedder:
    def __init__(self, model_path: str, device: str = "cuda:2"):
        self.model_path = model_path
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_path,
                device=self.device,
                trust_remote_code=True,
            )
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load_model().encode(texts)
        return [list(map(float, vector)) for vector in vectors]


def _is_local_service_url(url: str) -> bool:
    host = urlparse(url).hostname
    return host in {"127.0.0.1", "localhost", "::1"}


def _urlopen_service(req: request.Request, *, timeout: float):
    url = req.full_url
    if _is_local_service_url(url):
        return request.build_opener(request.ProxyHandler({})).open(req, timeout=timeout)
    return request.urlopen(req, timeout=timeout)


class HttpTextEmbedder:
    def __init__(self, service_url: str, timeout_seconds: float = 120.0):
        self.service_url = service_url
        self.timeout_seconds = float(timeout_seconds)
        self._batch_size: int = 32
        self._instruction: str | None = None
        self._text_prefix: str | None = None

    def set_batch_size(self, batch_size: int) -> None:
        self._batch_size = batch_size

    def set_instruction(self, instruction: str | None) -> None:
        """Set GritLM-style instruction for the embed request. None = document mode."""
        self._instruction = instruction

    def set_text_prefix(self, prefix: str | None) -> None:
        """Set text prefix for models like Instructor-XL that prepend instruction to text."""
        self._text_prefix = prefix

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        bs = self._batch_size
        # Apply text prefix if set (Instructor-XL style: "{prefix} {text}")
        if self._text_prefix:
            texts = [f"{self._text_prefix} {t}" for t in texts]
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            body: dict[str, object] = {"texts": batch}
            if self._instruction is not None:
                body["instruction"] = self._instruction
            payload = json.dumps(body).encode("utf-8")
            req = request.Request(
                self.service_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with _urlopen_service(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list):
                raise ValueError("Embedding service response must contain an 'embeddings' list.")
            if len(embeddings) != len(batch):
                raise ValueError(
                    "Embedding service returned "
                    f"{len(embeddings)} vectors for {len(batch)} input texts."
                )
            all_embeddings.extend([[float(value) for value in vector] for vector in embeddings])
        return all_embeddings


def build_text_embedder(
    *,
    model_path: str,
    device: str,
    service_url: Optional[str] = None,
    timeout_seconds: float = 120.0,
) -> TextEmbedder:
    if service_url:
        return HttpTextEmbedder(service_url=service_url, timeout_seconds=timeout_seconds)
    return BGEM3Embedder(model_path=model_path, device=device)
