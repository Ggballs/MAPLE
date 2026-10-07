#!/usr/bin/env python3
"""Run Stage 1 through Stage 5 from an existing Stage 0 JSON artifact."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import yaml


class RetryEmptyTextBackend:
    """Retry empty text responses without changing the underlying model behavior."""

    def __init__(self, backend, retries: int = 3):
        self.backend = backend
        self.retries = max(1, retries)

    def generate(self, prompt: str, **kwargs) -> str:
        for attempt in range(self.retries):
            response = self.backend.generate(prompt, **kwargs)
            if isinstance(response, str) and response.strip():
                return response
            if attempt + 1 < self.retries:
                print(f"Empty LLM text response; retrying ({attempt + 2}/{self.retries})", flush=True)
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM returned an empty text response {self.retries} times")

    def generate_json(self, prompt: str, **kwargs):
        return self.backend.generate_json(prompt, **kwargs)

    def generate_with_pdf_url(self, prompt: str, pdf_url: str, **kwargs):
        return self.backend.generate_with_pdf_url(prompt, pdf_url, **kwargs)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloaded-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=root / "configs/config.yaml")
    parser.add_argument("--rules", type=Path, default=root / "configs/rules.yaml")
    parser.add_argument("--api-base-url", default=None, help="Overrides LLM_API_BASE_URL and llm.base_url in config")
    parser.add_argument("--model", default=None, help="Overrides LLM_MODEL and llm.model in config")
    parser.add_argument("--api-key-env", default="LLM_API_KEY", help="Environment variable for a single API key")
    parser.add_argument("--analysis-modes", nargs="+", choices=["retrieval", "style", "embedding"])
    parser.add_argument("--scholar-max-results", type=int)
    parser.add_argument("--skip-stage5-analysis-filter", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(bundle_root / "src"))

    input_path = args.downloaded_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    rules_path = args.rules.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Copy configs/config.example.yaml to {config_path} and configure it first")
    from openreview_pipeline.runner import (
        run_filter_stage,
        run_generate_queries_stage,
        run_hard_negative_mining_stage,
        run_query_analysis_stage,
        run_summarize_stage,
    )
    from utils.llm import OpenAICompatibleBackend

    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    llm_config = config.get("llm", {}) if isinstance(config.get("llm"), dict) else {}
    api_key = os.getenv(args.api_key_env, "").strip()
    configured_tokens = llm_config.get("api_tokens", [])
    if isinstance(configured_tokens, str):
        configured_tokens = [configured_tokens]
    api_tokens = [api_key] if api_key else [str(token).strip() for token in configured_tokens if str(token).strip()]
    if not api_tokens:
        raise RuntimeError(
            f"Add an API key to llm.api_tokens in {config_path} or set {args.api_key_env}."
        )
    api_base_url = (
        args.api_base_url
        or os.getenv("LLM_API_BASE_URL")
        or str(llm_config.get("base_url") or "").strip()
    )
    model_name = args.model or os.getenv("LLM_MODEL") or str(llm_config.get("model") or "").strip()
    if not api_base_url or not model_name:
        raise RuntimeError("Set llm.base_url and llm.model in the config, or pass the corresponding CLI options.")

    stages = config.setdefault("stages", {})
    generate = stages.setdefault("generate_queries", {})
    if os.getenv("MAPLE_GOLDEN_DB_URL"):
        generate["golden_embedding_db_url"] = os.environ["MAPLE_GOLDEN_DB_URL"]
    if os.getenv("BGE_M3_MODEL_PATH"):
        generate["bge_model_path"] = os.environ["BGE_M3_MODEL_PATH"]
        stages.setdefault("query_analysis", {}).setdefault("embedding_analysis", {})[
            "bge_model_path"
        ] = os.environ["BGE_M3_MODEL_PATH"]
    if os.getenv("BGE_DEVICE"):
        generate["bge_device"] = os.environ["BGE_DEVICE"]
        stages.setdefault("query_analysis", {}).setdefault("embedding_analysis", {})[
            "bge_device"
        ] = os.environ["BGE_DEVICE"]
    if os.getenv("BGE_EMBEDDING_SERVICE_URL"):
        generate["embedding_service_url"] = os.environ["BGE_EMBEDDING_SERVICE_URL"]

    # Keep any runtime DB URL out of the bundle; the temporary config is mode 0600 and deleted.
    fd, runtime_config_name = tempfile.mkstemp(prefix="maple-stage1-5-", suffix=".yaml")
    runtime_config = Path(runtime_config_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        backend = RetryEmptyTextBackend(
            OpenAICompatibleBackend(
                base_url=api_base_url,
                api_tokens=api_tokens,
                model=model_name,
                max_tokens=int(config.get("llm", {}).get("max_tokens", 12000)),
                temperature=float(config.get("llm", {}).get("temperature", 0.0)),
                max_retries=int(config.get("llm", {}).get("max_retries", 3)),
                retry_backoff_seconds=float(config.get("llm", {}).get("retry_backoff_seconds", 3.0)),
            )
        )

        paths = {
            "filtered": output_dir / "01_filtered.json",
            "summarized": output_dir / "02_summarized.json",
            "queries": output_dir / "03_queries.json",
            "analysis": output_dir / "04_query_analysis",
            "mining": output_dir / "05_hard_negatives.json",
        }
        print("Stage 1: filter", flush=True)
        run_filter_stage(input_path=input_path, output_path=paths["filtered"], rules_config_path=rules_path)
        print("Stage 2: summarize", flush=True)
        run_summarize_stage(input_path=paths["filtered"], output_path=paths["summarized"], config_path=runtime_config, llm_backend=backend)
        print("Stage 3: generate queries", flush=True)
        run_generate_queries_stage(input_path=paths["summarized"], output_path=paths["queries"], config_path=runtime_config, llm_backend=backend)
        print("Stage 4: query analysis", flush=True)
        run_query_analysis_stage(
            summarized_path=paths["summarized"],
            queries_path=paths["queries"],
            output_dir=paths["analysis"],
            config_path=runtime_config,
            downloaded_path=input_path,
            analysis_modes=args.analysis_modes,
            llm_backend=backend,
        )

        print("Stage 5: hard-negative mining", flush=True)
        previous_cwd = Path.cwd()
        os.chdir(output_dir)
        try:
            run_hard_negative_mining_stage(
                input_path=paths["queries"],
                output_path=paths["mining"],
                query_analysis_output_dir=paths["analysis"],
                config_path=runtime_config,
                scholar_max_results=args.scholar_max_results,
                llm_backend=backend,
                apply_query_analysis_filter=not args.skip_stage5_analysis_filter,
            )
        finally:
            os.chdir(previous_cwd)
        print(f"Pipeline output: {output_dir}", flush=True)
        print(f"Stage 5 result: {paths['mining']}", flush=True)
    finally:
        runtime_config.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
