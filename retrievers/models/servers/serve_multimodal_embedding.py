"""Serve a single multimodal embedding model over HTTP.

Usage:
  python serve_multimodal_embedding.py --model qwen3-vl-embed-8b --device cuda:0 --port 18083
  python serve_multimodal_embedding.py --model ops-mm-embed-7b     --device cuda:0 --port 18086

Each instance loads ONE model.

POST /embed  body: {"texts": [...], "images": [...]}
  texts[i] and images[i] are paired (either or both can be provided).
  images entries can be local file paths or base64 data URIs.
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel, Field
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CHOICES = [
    "qwen3-vl-embed-8b",
    "ops-mm-embed-7b",
]


class EmbedRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list,
                              description="File paths or base64 data URIs per item")
    instruction: str | None = Field(default=None,
                                    description="Task instruction for models that support it (Qwen3-VL)")


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _resolve_images(image_specs: list[str]) -> list[str]:
    """Convert base64 data URIs to temp file paths; pass local paths through."""
    resolved: list[str] = []
    for spec in image_specs:
        if spec.startswith("data:") and ";base64," in spec:
            _, b64 = spec.split(";base64,", 1)
            fd, path = tempfile.mkstemp(suffix=".png", prefix="mmembed_")
            os.close(fd)
            Path(path).write_bytes(base64.b64decode(b64))
            resolved.append(path)
        else:
            resolved.append(spec)
    return resolved


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------

class Qwen3VLEmbedder:
    """Qwen3-VL-Embedding via SentenceTransformer — needs transformers >= 5.x for qwen3_vl support.

    Use the `maple-text` environment or another environment with Qwen3-VL-capable transformers.
    """

    def __init__(self, model_path: str, device: str):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading Qwen3-VL-Embedding from %s on %s ...", model_path, device)
        self._model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
        self._dim = 4096
        logger.info("Loaded. Embedding dim: %d", self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str], images: list[str], instruction: str | None = None) -> list[list[float]]:
        n = max(len(texts), len(images))
        if n == 0:
            return []
        inputs = []
        for i in range(n):
            item: dict[str, object] = {}
            if i < len(texts) and texts[i]:
                item["text"] = texts[i]
            if i < len(images) and images[i]:
                item["image"] = images[i]
            inputs.append(item)
        kwargs = {}
        if instruction:
            kwargs["prompt"] = instruction
        vectors = self._model.encode(inputs, show_progress_bar=False, **kwargs)
        return [list(map(float, v)) for v in vectors]


class OpsMMEmbedder:
    """Ops-MM-embedding-v1-7B — custom OpsMMEmbeddingV1 class from HF."""

    def __init__(self, model_path: str, device: str):
        logger.info("Loading Ops-MM-Embedding from %s on %s ...", model_path, device)
        self._model = None  # lazy load
        self._path = model_path
        self._device = device
        self._dim = 3584
        self._loaded = False

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_loaded(self):
        if self._loaded:
            return
        import sys
        candidate_roots = [
            os.environ.get("OPS_MM_CODE_DIR", ""),
            str(Path(self._path).resolve()),
            str(Path(self._path).resolve().parent),
        ]
        for root in candidate_roots:
            if root and root not in sys.path:
                sys.path.insert(0, root)
        from ops_mm_embedding_v1 import OpsMMEmbeddingV1
        self._model = OpsMMEmbeddingV1(
            self._path, device=self._device, attn_implementation="flash_attention_2",
        )
        self._loaded = True
        logger.info("Loaded. Embedding dim: %d", self._dim)

    def encode(self, texts: list[str], images: list[str], instruction: str | None = None) -> list[list[float]]:
        n = max(len(texts), len(images))
        if n == 0:
            return []
        self._ensure_loaded()

        return [
            list(map(float, v))
            for v in self._model.get_fused_embeddings(
                texts=texts or None, images=images or None, instruction=instruction,
            )
        ]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_CONFIG: dict[str, dict] = {
    "qwen3-vl-embed-8b": {
        "default_path": os.environ.get("QWEN3_VL_MODEL_PATH", "Qwen/Qwen3-VL-Embedding-8B"),
        "display_name": "Qwen3-VL-Embedding-8B",
        "embedder_class": Qwen3VLEmbedder,
    },
    "ops-mm-embed-7b": {
        "default_path": os.environ.get("OPS_MM_MODEL_PATH", "OpenSearch-AI/Ops-MM-embedding-v1-7B"),
        "display_name": "Ops-MM-Embedding-7B",
        "embedder_class": OpsMMEmbedder,
    },
}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def build_app(model_name: str, model_path: str, device: str):
    from fastapi import FastAPI

    cfg = MODEL_CONFIG[model_name]
    embedder: Optional[object] = None

    app = FastAPI(title=f"SciFullMMBench Multimodal Embedding — {cfg['display_name']}")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": model_name,
            "display_name": cfg["display_name"],
            "embed_dim": embedder.dim if embedder else None,
            "device": device,
        }

    @app.post("/embed", response_model=EmbedResponse)
    def embed(payload: EmbedRequest) -> EmbedResponse:
        nonlocal embedder
        if embedder is None:
            cls = cfg["embedder_class"]
            embedder = cls(model_path, device)
        texts = payload.texts or []
        images = payload.images or []
        if images:
            images = _resolve_images(images)
        vectors = embedder.encode(texts, images, instruction=payload.instruction)
        return EmbedResponse(embeddings=vectors)

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a single multimodal embedding model over HTTP."
    )
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override default model path.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18083)
    return parser


def main() -> int:
    import uvicorn

    args = build_parser().parse_args()
    model_path = args.model_path or MODEL_CONFIG[args.model]["default_path"]
    cfg = MODEL_CONFIG[args.model]
    logger.info("Model: %s (%s)", args.model, cfg["display_name"])
    logger.info("Path: %s", model_path)
    logger.info("Device: %s, Port: %d", args.device, args.port)

    app = build_app(args.model, model_path, str(args.device))
    uvicorn.run(app, host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
