from __future__ import annotations

import argparse
import gc
import logging
import os
from typing import Optional

import torch
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbedRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)
    batch_size: int | None = Field(default=None, ge=1)
    instruction: str | None = Field(
        default=None,
        description="For Instructor-XL: formats as [instruction, text] pairs"
    )


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


# ---- Model registry ---------------------------------------------------------

MODEL_REGISTRY = {
    # qwen3-embed-8b served separately on port 18087 (GPU 3)
    "instructor-xl": {
        "path": os.environ.get("INSTRUCTOR_XL_MODEL_PATH", "hkunlp/instructor-xl"),
        "display_name": "Instructor-XL",
    },
    # gritlm-7b served separately on port 18082 (needs transformers 4.x)
    "specter2": {
        "path": os.environ.get("SPECTER2_MODEL_PATH", "allenai/specter2_base"),
        "display_name": "SPECTER2",
    },
    "scincl": {
        "path": os.environ.get("SCINCL_MODEL_PATH", "malteos/scincl"),
        "display_name": "SCiNCL",
    },
}


# ---- Embedders --------------------------------------------------------------

class Specter2Embedder:
    """SPECTER2 uses vanilla BERT — load with AutoModel + mean pooling."""

    def __init__(self, model_path: str, device: str):
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModel.from_pretrained(model_path).to(device)
        self._model.eval()
        self._device = device

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        bs = batch_size or 64
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            max_len = self._tokenizer.model_max_length
            if max_len is None or max_len > 512:
                max_len = 512  # SPECTER2 is BERT-based
            inputs = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(self._device)
            outputs = self._model(**inputs)
            # mean pooling over token dimension (excluding padding)
            attention_mask = inputs["attention_mask"]
            hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            summed = (hidden * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            all_embeddings.extend(
                [list(map(float, vec)) for vec in pooled.cpu().numpy()]
            )
        return all_embeddings


class SentenceTransformerEmbedder:
    def __init__(self, model_path: str, device: str):
        import torch
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            model_path, device=device, trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16},
        )

    def encode(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        if not texts:
            return []
        bs = batch_size or len(texts)
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            vectors = self._model.encode(batch, show_progress_bar=False)
            all_vectors.extend([list(map(float, vec)) for vec in vectors])
        return all_vectors



def _build_embedder(model_name: str, device: str):
    entry = MODEL_REGISTRY[model_name]
    path = entry["path"]
    if model_name == "specter2":
        return Specter2Embedder(path, device)
    return SentenceTransformerEmbedder(path, device)


# ---- FastAPI app ------------------------------------------------------------

def build_app(device: str):
    import threading
    from fastapi import FastAPI

    app = FastAPI(title="SciFullMMBench Multi-Model Embedding Service")
    embedders: dict[str, object] = {}
    _lock = threading.Lock()

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok", "device": device,
            "loaded_models": list(embedders.keys()),
            "available_models": list(MODEL_REGISTRY.keys()),
        }

    @app.post("/embed/{model_name}", response_model=EmbedResponse)
    def embed(model_name: str, payload: EmbedRequest) -> EmbedResponse:
        if model_name not in MODEL_REGISTRY:
            return EmbedResponse(embeddings=[])
        if not payload.texts:
            return EmbedResponse(embeddings=[])

        if model_name not in embedders:
            with _lock:
                if model_name not in embedders:  # double-check
                    logger.info("Loading '%s' ...", model_name)
                    embedders[model_name] = _build_embedder(model_name, device)
                    logger.info("'%s' loaded.", model_name)

        encoder = embedders[model_name]
        texts = payload.texts
        # Instructor-XL uses [instruction, text] pairs format
        if payload.instruction and model_name == "instructor-xl":
            texts = [[payload.instruction, t] for t in texts]
        vectors = encoder.encode(texts, batch_size=payload.batch_size)
        if device.startswith("cuda"):
            import gc; gc.collect()
            torch.cuda.empty_cache()
        return EmbedResponse(embeddings=vectors)

    return app


# ---- CLI --------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve multiple embedding models over HTTP."
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--no-eager", action="store_true", help="Disable pre-loading all models at startup")
    return parser


def main() -> int:
    import uvicorn

    args = build_parser().parse_args()
    app = build_app(device=str(args.device))
    uvicorn.run(app, host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
