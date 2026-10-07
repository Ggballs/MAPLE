# MAPLE-Synth

Generate multi-aspect queries and mine hard-negative papers from OpenReview (Stages 0–5). Downstream embedding/indexing experiments are not included.

## 1. Install

Requirements: Conda, PostgreSQL with pgvector installed, and an OpenAI-compatible LLM API. The PostgreSQL account must be able to create a database and enable the `vector` extension.

```bash
git clone https://github.com/Ggballs/MAPLE.git
cd MAPLE/MAPLE-Synth
conda create -n maple-synth python=3.11 -y
conda activate maple-synth
python -m pip install -r requirements.txt
python -m pip install -e .
cp configs/config.example.yaml configs/config.yaml
chmod 600 configs/config.yaml
```

Run all subsequent commands from `MAPLE-Synth/`. The local `configs/config.yaml`, models, and outputs are ignored by Git; never commit credentials.

## 2. Configure

Edit these fields in `configs/config.yaml`; leave the other example settings unchanged:

| Field | Value |
| --- | --- |
| `openreview.token` | Your OpenReview token; alternatively set `username` and `password`. |
| `llm.base_url`, `llm.model`, `llm.api_tokens` | Your API endpoint, supported model, and API key list. |
| `stages.generate_queries.golden_embedding_db_url` | `postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME` |
| `stages.generate_queries.bge_model_path` | `models/bge-m3` |
| `stages.generate_queries.bge_device` | `cpu` or `cuda:0` |
| `stages.query_analysis.embedding_analysis.bge_model_path` | `models/bge-m3` |
| `stages.query_analysis.embedding_analysis.bge_device` | Same device as above. |

The example generates IR queries only and uses the configured LLM for Stage 5. For GPU inference, install a CUDA-compatible PyTorch build. If needed, set your own `HTTP_PROXY`/`HTTPS_PROXY`; optionally use `HF_ENDPOINT=https://hf-mirror.com` for model downloads.

## 3. Download BGE-M3

```bash
python -m pip install --upgrade huggingface_hub
export HF_HUB_DISABLE_XET=1
hf download BAAI/bge-m3 --include '*.json' --exclude 'onnx/*' --local-dir models/bge-m3
hf download BAAI/bge-m3 pytorch_model.bin sentencepiece.bpe.model --local-dir models/bge-m3
python - <<'PY'
from evaluations.embedding.embeddings import BGEM3Embedder
vector = BGEM3Embedder("models/bge-m3", device="cpu").embed_texts(["A paper search query."])[0]
assert len(vector) == 1024
print("BGE-M3 ready")
PY
```

Allow at least 3 GB of disk space for BGE-M3.

## 4. Prepare the ICL Database

Create a dedicated ICL database. Replace `USER`, `PASSWORD`, `HOST`, and `DBNAME` below with your database settings, and use the same connection in `configs/config.yaml`.

```bash
createdb --host HOST --port 5432 --username USER DBNAME
psql 'postgresql://USER:PASSWORD@HOST:5432/DBNAME' -c 'CREATE EXTENSION IF NOT EXISTS vector;'
DB_URL='postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME'
openreview-pipeline init-golden-query-embeddings --db-url "$DB_URL"
openreview-pipeline bootstrap-golden-query-embeddings --db-url "$DB_URL"
```

Bootstrap embeds and imports the bundled 84 IR exemplars (`specific=1`: Motivation 11, Method 40, Experiment/Result 33). It skips an already-populated table; use an empty table for this setup.

## 5. Run Stages 0–5

First download a small sample and verify OpenReview access:

```bash
mkdir -p outputs/smoke
openreview-pipeline download --venue ICLR --year 2025 --max-papers 20 --output outputs/smoke/00_downloaded.json
python - <<'PY'
import json
papers = json.load(open("outputs/smoke/00_downloaded.json"))["papers"]
assert papers, "No papers downloaded: check OpenReview credentials and network access."
print(f"Downloaded {len(papers)} papers")
PY
openreview-pipeline run-pipeline --stages 1-5 --downloaded-input outputs/smoke/00_downloaded.json --output-dir outputs/smoke --summarize-limit 1
```

If no papers pass Stage 1, increase `--max-papers` or change venue/year. Use a separate output directory for each run.

To resume after Stage 2:

```bash
openreview-pipeline run-pipeline --stages 3-5 --input-path outputs/smoke/02_summarized.json --downloaded-input outputs/smoke/00_downloaded.json --output-dir outputs/smoke
```

## Outputs

`outputs/smoke/` contains `00_downloaded.json`, `01_filtered.json`, `02_summarized.json`, `03_queries.json`, `04_query_analysis/`, `05_hard_negatives.json`, and `final_pipeline_output.json`.

Candidate PDFs and parsed text are cached separately in `outputs/hard_negative_pdfs/`. See `openreview-pipeline --help` for additional options.
