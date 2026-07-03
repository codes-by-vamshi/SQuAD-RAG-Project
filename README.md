# SQuAD-RAG-Project

Raw dataset source (Kaggle):
https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset?select=train-v1.1.json

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
