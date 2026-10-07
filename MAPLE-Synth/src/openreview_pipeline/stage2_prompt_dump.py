from __future__ import annotations

from pathlib import Path
from typing import Optional

from openreview_pipeline.schemas.schemas_filter import FilterResult, FilteredPapersDataset
from openreview_pipeline.stage2_summarize import Summarizer
from utils import MockLLMBackend, load_json


def _resolve_filter_result(
    dataset: FilteredPapersDataset,
    *,
    paper_id: Optional[str] = None,
) -> FilterResult:
    if paper_id:
        for result in dataset.results:
            if result.paper.paper.id == paper_id:
                return result
        raise ValueError(f"Paper id not found in filtered dataset: {paper_id}")

    for result in dataset.results:
        if result.passed:
            return result
    if dataset.results:
        return dataset.results[0]
    raise ValueError("Filtered dataset is empty")


def render_stage2_text_prompt(filter_result: FilterResult) -> str:
    paper = filter_result.paper
    summarizer = Summarizer(
        llm=MockLLMBackend(""),
        include_multimodal_evidence=False,
    )
    openreview_content = summarizer._build_openreview_content(paper)
    return summarizer._get_prompt_template().format(
        paper_title=paper.paper.title,
        paper_abstract=paper.paper.abstract,
        full_openreview_content=openreview_content,
    )


def write_stage2_text_prompt(
    *,
    filtered_input_path: Path,
    output_path: Path,
    paper_id: Optional[str] = None,
) -> Path:
    dataset = load_json(filtered_input_path, FilteredPapersDataset)
    filter_result = _resolve_filter_result(dataset, paper_id=paper_id)
    prompt = render_stage2_text_prompt(filter_result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")
    return output_path
