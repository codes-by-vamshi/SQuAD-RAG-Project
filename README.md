# SQuAD-RAG-Project

Raw dataset source (Kaggle):
https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset?select=train-v1.1.json

## Environment Setup

Recommended (Conda):

```bash
conda env create -f environment/environment.yml
conda activate squad-rag-project
```

Alternative (pip + venv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements.txt
```

Quick check:

```bash
python3 -c "import faiss, openai, yaml; print('environment ready')"
```

Set OpenAI API key (`.env` in project root):

```bash
echo "OPENAI_API_KEY=your_api_key_here" >> .env
```

## Step 1: Data Pipeline

Run Step 1 to process the raw SQuAD file and generate processed outputs.

It reads only:
- `data/raw/train-v1.1.json`

It writes processed files to:
- `data/documents/`
- `data/qa/`

Run:

```bash
python3 Step1_Data_Pipeline.py
```

## Step 2: Data Splits (Chunking)

Run Step 2 to split each document into overlapping text chunks for retrieval.

Default config:
- `configs/baseline.yaml` (`chunk_size`, `overlap_size`)

It reads:
- `data/documents/*.txt`

It writes:
- `data/splits/chunks.jsonl`

Run:

```bash
python3 "Step2_Data_Splits(Chunking).py"
```

## Step 3: Create Embeddings

Run Step 3 to create OpenAI embeddings from chunked text.

Default config:
- `configs/baseline.yaml`
  - `embedding_model`
  - `embedding_batch_size`
  - `flush_every_batches`

It reads:
- `data/splits/chunks.jsonl`

It writes:
- `embeddings/embeddings.jsonl`

Required:
- `OPENAI_API_KEY` in environment or `.env`

Run:

```bash
python3 Step3_Create_Embeddings.py --overwrite
```

Optional smoke test (small sample):

```bash
python3 Step3_Create_Embeddings.py --limit 20 --overwrite
```

Optional output flush control:

```bash
python3 Step3_Create_Embeddings.py --overwrite --flush-every-batches 5
```

## Step 4: Create Indexes

Run Step 4 to build FAISS indexes from embedding vectors.

Default config:
- `configs/baseline.yaml`
  - `ivf_nlist_divisor`
  - `ivf_nlist_min`
  - `ivf_nprobe`
  - `hnsw_m`
  - `hnsw_ef_construction`
  - `hnsw_ef_search`
  - `flat_l2_index_filename`
  - `ivf_flat_index_filename`
  - `hnsw_index_filename`

It reads:
- `embeddings/embeddings.jsonl`

It writes:
- `indexes/*.index`

Run:

```bash
python3 Step4_Create_Indexes.py --overwrite
```

## Step 5: Retrieve Chunks

Run Step 5 to embed a question and retrieve top-k chunks from all FAISS indexes.

Default config:
- `configs/baseline.yaml`
  - `retrieval_top_k`
  - `retrieval_preview_chars`
  - `retrieval_seed`
  - `ivf_nprobe`
  - `hnsw_ef_search`
  - `flat_l2_index_filename`
  - `ivf_flat_index_filename`
  - `hnsw_index_filename`
  - `embedding_model`

It reads:
- `embeddings/embeddings.jsonl`
- `data/qa/*.json` (when `--question` is not provided)
- `indexes/*.index`

Run (random QA sample):

```bash
python3 Step5_Retrieve.py
```

Run (custom question):

```bash
python3 Step5_Retrieve.py --question "Who founded the Roman Republic?"
```

## Step 6: Evaluate Index Quality

Run Step 6 to evaluate ANN fidelity and latency for each index.

Method:
- Sample 100 questions per seed.
- Compute exact top-k with `flat_l2`.
- Compute ANN recall for `ivf_flat` and `hnsw` as overlap against `flat_l2` top-k.
- Report latency per query (`avg`, `p50`, `p95`, `p99`) for all indexes.
- Report per-seed and combined (across seeds) summaries.

Default config:
- `configs/baseline.yaml`
  - `index_eval_sample_size`
  - `index_eval_top_k`
  - `index_eval_seeds`
  - `index_eval_output_dir`
  - `embedding_model`
  - `embedding_batch_size`
  - `ivf_nprobe`
  - `hnsw_ef_search`
  - `flat_l2_index_filename`
  - `ivf_flat_index_filename`
  - `hnsw_index_filename`

Run:

```bash
python3 Step6_Evaluate_Index_Quality.py
```

Example with custom seeds:

```bash
python3 Step6_Evaluate_Index_Quality.py --seeds 101,202,303,404,505,606
```

Outputs:
- `results/index_eval/sampled_queries_by_seed.json`
- `results/index_eval/query_results.jsonl`
- `results/index_eval/per_seed_summary.json`
- `results/index_eval/per_seed_summary.csv`
- `results/index_eval/combined_summary.json`
- `results/index_eval/combined_summary.csv`
- `results/index_eval/report.md`

## Latest Results (Step 6)

Source:
- `results/index_eval/report.md`
- `results/index_eval/combined_summary.csv`

Experiment setup:
- Sample size per seed: `100`
- Top-k: `5`
- Seeds: `11, 22, 33, 44, 55, 66`

### Combined Across 6 Seeds

| index | recall_mean | recall_std | avg_ms_mean | avg_ms_std | p50_ms_mean | p95_ms_mean | p99_ms_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat_l2 | 1.0000 | 0.0000 | 2.6015 | 0.0604 | 2.5329 | 2.9327 | 3.1149 |
| ivf_flat | 0.9593 | 0.0072 | 0.2574 | 0.0258 | 0.2331 | 0.4234 | 0.5606 |
| hnsw | 0.9800 | 0.0043 | 0.1388 | 0.0039 | 0.1292 | 0.2093 | 0.2776 |

### Per-Seed Results

| seed | index | recall@5 | avg_ms | p50_ms | p95_ms | p99_ms |
|---:|---|---:|---:|---:|---:|---:|
| 11 | flat_l2 | 1.0000 | 2.5251 | 2.4555 | 2.7349 | 2.9133 |
| 11 | ivf_flat | 0.9500 | 0.3094 | 0.2572 | 0.5051 | 0.7641 |
| 11 | hnsw | 0.9720 | 0.1376 | 0.1270 | 0.1604 | 0.1786 |
| 22 | flat_l2 | 1.0000 | 2.5472 | 2.4941 | 2.8007 | 2.9391 |
| 22 | ivf_flat | 0.9540 | 0.2300 | 0.2296 | 0.2575 | 0.2834 |
| 22 | hnsw | 0.9860 | 0.1324 | 0.1296 | 0.1638 | 0.1838 |
| 33 | flat_l2 | 1.0000 | 2.6021 | 2.5039 | 3.0471 | 3.3272 |
| 33 | ivf_flat | 0.9560 | 0.2445 | 0.2252 | 0.4870 | 0.5625 |
| 33 | hnsw | 0.9780 | 0.1429 | 0.1302 | 0.2533 | 0.3618 |
| 44 | flat_l2 | 1.0000 | 2.5841 | 2.4950 | 3.0091 | 3.2085 |
| 44 | ivf_flat | 0.9720 | 0.2520 | 0.2287 | 0.4792 | 0.5358 |
| 44 | hnsw | 0.9820 | 0.1414 | 0.1303 | 0.2248 | 0.2904 |
| 55 | flat_l2 | 1.0000 | 2.7063 | 2.6467 | 3.0489 | 3.1272 |
| 55 | ivf_flat | 0.9600 | 0.2415 | 0.2320 | 0.3056 | 0.5148 |
| 55 | hnsw | 0.9800 | 0.1357 | 0.1299 | 0.1811 | 0.2885 |
| 66 | flat_l2 | 1.0000 | 2.6442 | 2.6023 | 2.9556 | 3.1741 |
| 66 | ivf_flat | 0.9640 | 0.2671 | 0.2262 | 0.5060 | 0.7028 |
| 66 | hnsw | 0.9820 | 0.1428 | 0.1284 | 0.2726 | 0.3627 |

Observations:
- `flat_l2` is exact and stable (`Recall@5 = 1.0000`) but slowest.
- `hnsw` gives the best speed/quality tradeoff here (`Recall@5 ~ 0.98`, lowest latency).
- `ivf_flat` is faster than `flat_l2` but lower recall than `hnsw` with current settings.
