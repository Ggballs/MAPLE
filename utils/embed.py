from __future__ import annotations

import json
from urllib import request
from typing import Iterable

import numpy as np

from retrievers.models.http import HttpTextEmbedder, _urlopen_service


DEFAULT_QUERY_INSTRUCTION = "Represent this paper search query for retrieval of relevant scientific papers"

DEFAULT_QUERY_INSTRUCTION_MODELS = frozenset(
    {
        "instructor-xl",
        "gritlm-7b",
        "qwen3-embed-8b",
        "qwen3-vl-embed-8b",
        "ops-mm-embed-7b",
    }
)


def resolve_query_instruction(model_name: str | None, instruction: str | None) -> str | None:
    if instruction is not None:
        return instruction
    normalized_model_name = str(model_name or "").strip()
    if normalized_model_name in DEFAULT_QUERY_INSTRUCTION_MODELS:
        return DEFAULT_QUERY_INSTRUCTION
    return None


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return array / norms


class ServiceTextEmbedder:
    def __init__(
        self,
        *,
        service_url: str,
        batch_size: int,
        instruction: str | None = None,
        timeout_seconds: float = 120.0,
        normalize_output: bool = True,
    ) -> None:
        self.service_url = service_url
        self.batch_size = int(batch_size)
        self.instruction = instruction
        self.timeout_seconds = float(timeout_seconds)
        self.normalize_output = bool(normalize_output)
        self._client = HttpTextEmbedder(service_url, timeout_seconds=timeout_seconds)
        self._client.set_batch_size(self.batch_size)
        self._client.set_instruction(instruction)

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._client.embed_texts(values)
        array = np.asarray(vectors, dtype=np.float32)
        if self.normalize_output:
            return l2_normalize(array)
        return array


class ServiceMultimodalEmbedder:
    def __init__(
        self,
        *,
        service_url: str,
        batch_size: int,
        instruction: str | None = None,
        timeout_seconds: float = 120.0,
        normalize_output: bool = True,
    ) -> None:
        self.service_url = service_url
        self.batch_size = int(batch_size)
        self.instruction = instruction
        self.timeout_seconds = float(timeout_seconds)
        self.normalize_output = bool(normalize_output)

    def _request(self, payload: dict[str, object]) -> np.ndarray:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.service_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _urlopen_service(req, timeout=self.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise ValueError("Embedding service response must contain an 'embeddings' list.")
        array = np.asarray(embeddings, dtype=np.float32)
        if self.normalize_output and array.size:
            return l2_normalize(array)
        return array

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        values = [str(text) for text in texts if str(text)]
        if not values:
            return np.zeros((0, 0), dtype=np.float32)
        batches: list[np.ndarray] = []
        for index in range(0, len(values), self.batch_size):
            payload: dict[str, object] = {"texts": values[index : index + self.batch_size]}
            if self.instruction is not None:
                payload["instruction"] = self.instruction
            batches.append(self._request(payload))
        return np.concatenate(batches, axis=0)

    def embed_images(self, image_paths: Iterable[str]) -> np.ndarray:
        values = [str(path) for path in image_paths if str(path)]
        if not values:
            return np.zeros((0, 0), dtype=np.float32)
        batches: list[np.ndarray] = []
        for index in range(0, len(values), self.batch_size):
            payload = {"images": values[index : index + self.batch_size]}
            batches.append(self._request(payload))
        return np.concatenate(batches, axis=0)
