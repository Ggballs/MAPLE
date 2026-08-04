"""Standalone Qwen3-Embed-8B embedding service (fp16, max 32K tokens)."""
from __future__ import annotations

import argparse
import logging
import os

import torch
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("QWEN3_EMBED_MODEL_PATH", "Qwen/Qwen3-Embedding-8B")


class EmbedRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)
    batch_size: int | None = Field(default=None, ge=1)
    instruction: str | None = Field(
        default=None,
        description="Qwen3 query instruction. Formats as 'Instruct: {instruction}\\nQuery:{text}'"
    )


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


def build_app(device: str):
    from fastapi import FastAPI
    from sentence_transformers import SentenceTransformer

    app = FastAPI(title="Qwen3-Embed-8B Service")
    model: SentenceTransformer | None = None

    def _ensure_model():
        nonlocal model
        if model is None:
            logger.info("Loading Qwen3-Embed-8B on %s (fp16) ...", device)
            model = SentenceTransformer(
                MODEL_PATH, device=device, trust_remote_code=True,
                model_kwargs={"torch_dtype": torch.float16},
            )
            logger.info("Qwen3-Embed-8B loaded (max_seq_length=%d).", model.max_seq_length)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "device": device, "model_loaded": model is not None}

    @app.post("/embed", response_model=EmbedResponse)
    def embed(payload: EmbedRequest) -> EmbedResponse:
        if not payload.texts:
            return EmbedResponse(embeddings=[])
        _ensure_model()
        bs = payload.batch_size or 1
        texts = payload.texts
        # Qwen3 query format: "Instruct: {task}\nQuery:{query}"
        if payload.instruction:
            texts = [f"Instruct: {payload.instruction}\nQuery:{t}" for t in texts]
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            vectors = model.encode(batch, show_progress_bar=False)
            all_vectors.extend([list(map(float, v)) for v in vectors])
        return EmbedResponse(embeddings=all_vectors)

    return app


def main() -> int:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18087)
    args = parser.parse_args()

    app = build_app(device=str(args.device))
    uvicorn.run(app, host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
