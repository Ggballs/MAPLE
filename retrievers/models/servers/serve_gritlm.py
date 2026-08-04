"""Standalone GritLM-7B embedding service using the official gritlm package."""
from __future__ import annotations

import argparse
import logging
import os

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("GRITLM_MODEL_PATH", "GritLM/GritLM-7B")

DEFAULT_INSTRUCTION = "Represent this text for retrieval of relevant scientific papers"


class EmbedRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)
    batch_size: int | None = Field(default=None, ge=1)
    instruction: str | None = Field(
        default=None,
        description="Query instruction. If None, uses document mode (empty instruction)."
    )


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


def build_app(device: str):
    from fastapi import FastAPI

    app = FastAPI(title="GritLM-7B Embedding Service")
    model = None  # GritLM instance

    def _ensure_model():
        nonlocal model
        if model is None:
            from gritlm import GritLM
            logger.info("Loading GritLM-7B on %s ...", device)
            model = GritLM(
                MODEL_PATH,
                torch_dtype="auto",
                mode="embedding",  # saves memory by omitting LM head
                pooling_method="mean",
                device=device,
            )
            logger.info("GritLM-7B loaded.")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "device": device, "model_loaded": model is not None}

    @app.post("/embed", response_model=EmbedResponse)
    def embed(payload: EmbedRequest) -> EmbedResponse:
        if not payload.texts:
            return EmbedResponse(embeddings=[])
        _ensure_model()
        instruction = payload.instruction or ""
        bs = payload.batch_size or len(payload.texts)
        all_vectors: list[list[float]] = []
        for i in range(0, len(payload.texts), bs):
            batch = payload.texts[i : i + bs]
            vectors = model.encode(
                batch,
                instruction=instruction,
                batch_size=len(batch),
            )
            all_vectors.extend([list(map(float, v)) for v in vectors])
        return EmbedResponse(embeddings=all_vectors)

    return app


def main() -> int:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18082)
    args = parser.parse_args()

    app = build_app(device=str(args.device))
    uvicorn.run(app, host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
