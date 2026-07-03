import argparse
import csv
import json
import math
import os
import random
import re
import string
from collections import Counter
from pathlib import Path
from time import perf_counter_ns

import faiss
import numpy as np
import yaml
from dotenv import load_dotenv


def parse_seed_list(value) -> list[int]:
    if isinstance(value, list):
        seeds = [int(v) for v in value]
    elif isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        seeds = [int(part) for part in parts]
    else:
        raise ValueError("Seed list must be a list[int] or comma-separated string.")

    if not seeds:
        raise ValueError("Seed list cannot be empty.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed list must not contain duplicates.")
    return seeds


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = {
        "embedding_model": str(raw.get("embedding_model", "text-embedding-3-small")),
        "embedding_batch_size": int(raw.get("embedding_batch_size", 100)),
        "ivf_nprobe": int(raw.get("ivf_nprobe", 32)),
        "hnsw_ef_search": int(raw.get("hnsw_ef_search", 32)),
        "flat_l2_index_filename": str(raw.get("flat_l2_index_filename", "flat_l2.index")),
        "ivf_flat_index_filename": str(raw.get("ivf_flat_index_filename", "ivf_flat.index")),
        "hnsw_index_filename": str(raw.get("hnsw_index_filename", "hnsw.index")),
        "generation_model": str(raw.get("generation_model", "gpt-4o-mini")),
        "generation_temperature": float(raw.get("generation_temperature", 0.0)),
        "generation_max_tokens": int(raw.get("generation_max_tokens", 80)),
        "generation_eval_sample_size": int(raw.get("generation_eval_sample_size", 100)),
        "generation_eval_seeds": parse_seed_list(
            raw.get("generation_eval_seeds", [11, 22, 33, 44, 55, 66])
        ),
        "generation_eval_retrieval_top_k": int(raw.get("generation_eval_retrieval_top_k", 5)),
        "generation_eval_context_max_chars": int(raw.get("generation_eval_context_max_chars", 4000)),
        "generation_eval_index": str(raw.get("generation_eval_index", "hnsw")),
        "generation_eval_output_dir": str(
            raw.get("generation_eval_output_dir", "results/generation_eval")
        ),
    }

    if cfg["embedding_batch_size"] <= 0:
        raise ValueError("`embedding_batch_size` must be > 0.")
    if cfg["ivf_nprobe"] <= 0:
        raise ValueError("`ivf_nprobe` must be > 0.")
    if cfg["hnsw_ef_search"] <= 0:
        raise ValueError("`hnsw_ef_search` must be > 0.")
    if cfg["generation_max_tokens"] <= 0:
        raise ValueError("`generation_max_tokens` must be > 0.")
    if cfg["generation_eval_sample_size"] <= 0:
        raise ValueError("`generation_eval_sample_size` must be > 0.")
    if cfg["generation_eval_retrieval_top_k"] <= 0:
        raise ValueError("`generation_eval_retrieval_top_k` must be > 0.")
    if cfg["generation_eval_context_max_chars"] <= 0:
        raise ValueError("`generation_eval_context_max_chars` must be > 0.")
    if cfg["generation_eval_index"] not in {"flat_l2", "ivf_flat", "hnsw"}:
        raise ValueError("`generation_eval_index` must be one of: flat_l2, ivf_flat, hnsw.")
    if cfg["generation_temperature"] < 0:
        raise ValueError("`generation_temperature` must be >= 0.")

    return cfg


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            yield line_no, record


def load_embedding_metadata(path: Path) -> tuple[list[dict], int]:
    metadata: list[dict] = []
    expected_dim = None

    for line_no, record in iter_jsonl(path):
        embedding = record.get("embedding")
        if embedding is None:
            raise ValueError(f"Missing `embedding` field at {path}:{line_no}")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"`embedding` must be a non-empty list at {path}:{line_no}")

        if expected_dim is None:
            expected_dim = len(embedding)
        elif len(embedding) != expected_dim:
            raise ValueError(
                f"Inconsistent embedding dimensions at {path}:{line_no}: "
                f"expected {expected_dim}, got {len(embedding)}"
            )

        metadata.append(
            {
                "doc_name": str(record.get("doc_name", "")),
                "doc_id": record.get("doc_id"),
                "chunk_id": str(record.get("chunk_id", "")),
                "chunk_index": record.get("chunk_index"),
                "text": str(record.get("text", "")),
            }
        )

    if not metadata:
        raise ValueError(f"No embeddings found in: {path}")

    return metadata, expected_dim


def load_question_pool(qa_dir: Path) -> list[dict]:
    qa_files = sorted(qa_dir.glob("*.json"))
    if not qa_files:
        raise FileNotFoundError(f"No QA files found in directory: {qa_dir}")

    pool = []
    for qa_file in qa_files:
        with qa_file.open("r", encoding="utf-8") as f:
            qa_pairs = json.load(f)

        if not isinstance(qa_pairs, list):
            raise ValueError(f"QA file must contain a list: {qa_file}")

        for qa_index, pair in enumerate(qa_pairs):
            question = str(pair.get("q", "")).strip()
            answer = str(pair.get("a", "")).strip()
            if not question or not answer:
                continue

            query_id = f"{qa_file.stem}:{qa_index}"
            pool.append(
                {
                    "query_id": query_id,
                    "question": question,
                    "answer": answer,
                    "qa_file": qa_file.name,
                    "qa_index": qa_index,
                }
            )

    if not pool:
        raise ValueError(f"No valid QA question-answer pairs found in: {qa_dir}")

    return pool


def sample_query_ids(pool: list[dict], sample_size: int, seeds: list[int]) -> dict[int, list[str]]:
    if sample_size > len(pool):
        raise ValueError(
            f"Sample size ({sample_size}) is larger than available QA pairs ({len(pool)})."
        )

    query_ids = [item["query_id"] for item in pool]
    sampled = {}
    for seed in seeds:
        rng = random.Random(seed)
        sampled[seed] = rng.sample(query_ids, sample_size)
    return sampled


def batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def embed_questions(client, model: str, questions: list[dict], batch_size: int) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    total = len(questions)

    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def create_embeddings(input_texts: list[str]):
        return client.embeddings.create(model=model, input=input_texts)

    for batch_no, batch in enumerate(batched(questions, batch_size), start=1):
        inputs = [item["question"] for item in batch]
        response = create_embeddings(inputs)

        if len(response.data) != len(batch):
            raise RuntimeError(
                f"Embedding response size mismatch in batch {batch_no}: "
                f"expected {len(batch)}, got {len(response.data)}"
            )

        for item, row in zip(batch, response.data):
            vectors[item["query_id"]] = np.array([row.embedding], dtype=np.float32)

        done = min(batch_no * batch_size, total)
        if batch_no % 5 == 0 or done == total:
            print(f"Embedded {done}/{total} sampled questions")

    return vectors


def set_index_runtime_params(index, ivf_nprobe: int, hnsw_ef_search: int) -> None:
    if hasattr(index, "nprobe"):
        nlist = getattr(index, "nlist", ivf_nprobe)
        index.nprobe = min(ivf_nprobe, max(1, nlist))
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = hnsw_ef_search


def retrieve_context(
    index,
    query_vector: np.ndarray,
    metadata: list[dict],
    top_k: int,
    context_max_chars: int,
) -> tuple[list[int], list[float], str]:
    distances, ids = index.search(query_vector, top_k)
    retrieved_ids = [int(v) for v in ids[0]]
    retrieved_distances = [float(v) for v in distances[0]]

    pieces = []
    used = 0
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id < 0 or chunk_id >= len(metadata):
            continue

        chunk_text = " ".join(metadata[chunk_id]["text"].split())
        if not chunk_text:
            continue

        prefix = f"[{rank}] "
        remaining = context_max_chars - used
        if remaining <= len(prefix):
            break

        available = remaining - len(prefix)
        clipped = chunk_text if len(chunk_text) <= available else chunk_text[:available]
        pieces.append(prefix + clipped)
        used += len(prefix) + len(clipped)

        if used >= context_max_chars:
            break

    context = "\n\n".join(pieces)
    return retrieved_ids, retrieved_distances, context


def generate_answer(
    client,
    model: str,
    question: str,
    context: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, float]:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _call():
        return client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a factual QA assistant. Answer the question strictly from the "
                        "provided context. If the context does not contain the answer, say "
                        "'I don't know'. Keep answers concise."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}",
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    started = perf_counter_ns()
    response = _call()
    latency_ms = (perf_counter_ns() - started) / 1_000_000

    message = response.choices[0].message.content if response.choices else ""
    return (message or "").strip(), latency_ms


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def tokenize_for_metric(text: str) -> list[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def exact_match_score(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_for_metric(prediction)
    ref_tokens = tokenize_for_metric(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0

    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[j]))
        prev = curr
    return prev[-1]


def rouge_l_f1_score(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_for_metric(prediction)
    ref_tokens = tokenize_for_metric(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ngram_counts(tokens: list[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_for_metric(prediction)
    ref_tokens = tokenize_for_metric(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    max_n = min(4, len(pred_tokens))
    precisions = []
    for n in range(1, max_n + 1):
        pred_counts = ngram_counts(pred_tokens, n)
        ref_counts = ngram_counts(ref_tokens, n)

        clipped = 0
        total = 0
        for ngram, count in pred_counts.items():
            clipped += min(count, ref_counts.get(ngram, 0))
            total += count

        # Add-1 smoothing for stability on short QA answers.
        precisions.append((clipped + 1) / (total + 1))

    if len(pred_tokens) > len(ref_tokens):
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1))

    weight = 1.0 / max_n
    return brevity_penalty * math.exp(sum(weight * math.log(p) for p in precisions))


def summarize_values(values: list[float]) -> dict:
    arr = np.array(values, dtype=np.float64)
    return {
        "avg": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def create_markdown_report(
    path: Path,
    per_seed_rows: list[dict],
    combined_summary: dict,
    sample_size: int,
    seeds: list[int],
    index_name: str,
    generation_model: str,
    top_k: int,
) -> None:
    lines = [
        "# Generation Evaluation",
        "",
        f"- Sample size per seed: {sample_size}",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Retrieval index: {index_name}",
        f"- Retrieval top-k context: {top_k}",
        f"- Generation model: {generation_model}",
        "",
        "## Per-Seed Results",
        "",
        "| seed | EM | F1 | ROUGE-L | BLEU | avg_gen_ms | p50_gen_ms | p95_gen_ms | p99_gen_ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in per_seed_rows:
        lines.append(
            f"| {row['seed']} | {row['exact_match']:.4f} | {row['f1']:.4f} | "
            f"{row['rouge_l']:.4f} | {row['bleu']:.4f} | {row['avg_gen_ms']:.4f} | "
            f"{row['p50_gen_ms']:.4f} | {row['p95_gen_ms']:.4f} | {row['p99_gen_ms']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Combined Across Seeds",
            "",
            "| metric | mean | std |",
            "|---|---:|---:|",
            f"| EM | {combined_summary['exact_match_mean']:.4f} | {combined_summary['exact_match_std']:.4f} |",
            f"| F1 | {combined_summary['f1_mean']:.4f} | {combined_summary['f1_std']:.4f} |",
            f"| ROUGE-L | {combined_summary['rouge_l_mean']:.4f} | {combined_summary['rouge_l_std']:.4f} |",
            f"| BLEU | {combined_summary['bleu_mean']:.4f} | {combined_summary['bleu_std']:.4f} |",
            f"| Avg Gen Latency (ms) | {combined_summary['avg_gen_ms_mean']:.4f} | {combined_summary['avg_gen_ms_std']:.4f} |",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 7: Evaluate generation quality (EM/F1/ROUGE-L/BLEU) with multi-seed sampling."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./configs/baseline.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--qa-dir",
        type=Path,
        default=Path("./data/qa"),
        help="Directory with QA JSON files.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("./embeddings/embeddings.jsonl"),
        help="Embeddings JSONL file from Step 3.",
    )
    parser.add_argument(
        "--indexes-dir",
        type=Path,
        default=Path("./indexes"),
        help="Directory with FAISS indexes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override sample size per seed.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed list (example: 11,22,33,44,55,66).",
    )
    parser.add_argument(
        "--index",
        type=str,
        choices=["flat_l2", "ivf_flat", "hnsw"],
        default=None,
        help="Retrieval index used to build generation context.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=None,
        help="Override retrieval top-k for generation context.",
    )
    parser.add_argument(
        "--context-max-chars",
        type=int,
        default=None,
        help="Override max total context characters passed to generation prompt.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Override embedding model for question embeddings.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help="Override embedding batch size.",
    )
    parser.add_argument(
        "--generation-model",
        type=str,
        default=None,
        help="Override generation model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override generation temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override generation max tokens.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    sample_size = (
        args.sample_size if args.sample_size is not None else cfg["generation_eval_sample_size"]
    )
    seeds = parse_seed_list(args.seeds) if args.seeds is not None else cfg["generation_eval_seeds"]
    retrieval_top_k = (
        args.retrieval_top_k
        if args.retrieval_top_k is not None
        else cfg["generation_eval_retrieval_top_k"]
    )
    context_max_chars = (
        args.context_max_chars
        if args.context_max_chars is not None
        else cfg["generation_eval_context_max_chars"]
    )
    index_name = args.index or cfg["generation_eval_index"]
    embedding_model = args.embedding_model or cfg["embedding_model"]
    embedding_batch_size = (
        args.embedding_batch_size
        if args.embedding_batch_size is not None
        else cfg["embedding_batch_size"]
    )
    generation_model = args.generation_model or cfg["generation_model"]
    generation_temperature = (
        args.temperature if args.temperature is not None else cfg["generation_temperature"]
    )
    generation_max_tokens = args.max_tokens if args.max_tokens is not None else cfg["generation_max_tokens"]
    output_dir = args.output_dir or Path(cfg["generation_eval_output_dir"])

    if sample_size <= 0:
        raise ValueError("`sample-size` must be > 0.")
    if retrieval_top_k <= 0:
        raise ValueError("`retrieval-top-k` must be > 0.")
    if context_max_chars <= 0:
        raise ValueError("`context-max-chars` must be > 0.")
    if embedding_batch_size <= 0:
        raise ValueError("`embedding-batch-size` must be > 0.")
    if generation_max_tokens <= 0:
        raise ValueError("`max-tokens` must be > 0.")
    if generation_temperature < 0:
        raise ValueError("`temperature` must be >= 0.")

    if not args.embeddings.exists():
        raise FileNotFoundError(f"Embeddings file not found: {args.embeddings}")

    metadata, embedding_dim = load_embedding_metadata(args.embeddings)

    index_path_map = {
        "flat_l2": args.indexes_dir / cfg["flat_l2_index_filename"],
        "ivf_flat": args.indexes_dir / cfg["ivf_flat_index_filename"],
        "hnsw": args.indexes_dir / cfg["hnsw_index_filename"],
    }
    selected_index_path = index_path_map[index_name]
    if not selected_index_path.exists():
        raise FileNotFoundError(f"Index file not found: {selected_index_path}")

    pool = load_question_pool(args.qa_dir)
    sampled_by_seed = sample_query_ids(pool=pool, sample_size=sample_size, seeds=seeds)

    selected_ids = {query_id for ids in sampled_by_seed.values() for query_id in ids}
    selected_questions = [item for item in pool if item["query_id"] in selected_ids]
    selected_questions.sort(key=lambda item: item["query_id"])
    question_by_id = {item["query_id"]: item for item in selected_questions}

    print(f"QA pool size: {len(pool)}")
    print(f"Unique sampled questions across all seeds: {len(selected_questions)}")
    print(f"Seeds: {seeds}")
    print(f"Sample size per seed: {sample_size}")
    print(f"Retrieval index: {index_name}")
    print(f"Retrieval top-k: {retrieval_top_k}")
    print(f"Generation model: {generation_model}")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The `openai` package is not installed in this Python environment. "
            "Install dependencies from environment/requirements.txt first."
        ) from exc

    client = OpenAI(api_key=api_key)
    query_vectors = embed_questions(
        client=client,
        model=embedding_model,
        questions=selected_questions,
        batch_size=embedding_batch_size,
    )

    sample_vector = next(iter(query_vectors.values()))
    if sample_vector.shape[1] != embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: question={sample_vector.shape[1]}, "
            f"chunks={embedding_dim}"
        )

    index = faiss.read_index(str(selected_index_path))
    set_index_runtime_params(index, cfg["ivf_nprobe"], cfg["hnsw_ef_search"])

    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_queries_path = output_dir / "sampled_queries_by_seed.json"
    query_results_path = output_dir / "query_results.jsonl"
    per_seed_summary_path = output_dir / "per_seed_summary.json"
    per_seed_csv_path = output_dir / "per_seed_summary.csv"
    combined_summary_path = output_dir / "combined_summary.json"
    combined_csv_path = output_dir / "combined_summary.csv"
    report_path = output_dir / "report.md"

    with sampled_queries_path.open("w", encoding="utf-8") as f:
        json.dump(sampled_by_seed, f, ensure_ascii=False, indent=2)

    per_seed_rows = []

    with query_results_path.open("w", encoding="utf-8") as out:
        for seed in seeds:
            print(f"Evaluating seed {seed}...")
            seed_query_ids = sampled_by_seed[seed]

            em_scores = []
            f1_scores = []
            rouge_scores = []
            bleu_scores = []
            generation_latencies = []

            for query_id in seed_query_ids:
                item = question_by_id[query_id]
                question = item["question"]
                reference_answer = item["answer"]
                query_vector = query_vectors[query_id]

                retrieved_ids, retrieved_distances, context = retrieve_context(
                    index=index,
                    query_vector=query_vector,
                    metadata=metadata,
                    top_k=retrieval_top_k,
                    context_max_chars=context_max_chars,
                )

                prediction, generation_latency_ms = generate_answer(
                    client=client,
                    model=generation_model,
                    question=question,
                    context=context,
                    max_tokens=generation_max_tokens,
                    temperature=generation_temperature,
                )

                em = exact_match_score(prediction, reference_answer)
                f1 = f1_score(prediction, reference_answer)
                rouge_l = rouge_l_f1_score(prediction, reference_answer)
                bleu = bleu_score(prediction, reference_answer)

                em_scores.append(em)
                f1_scores.append(f1)
                rouge_scores.append(rouge_l)
                bleu_scores.append(bleu)
                generation_latencies.append(generation_latency_ms)

                result_row = {
                    "seed": seed,
                    "query_id": query_id,
                    "question": question,
                    "reference_answer": reference_answer,
                    "predicted_answer": prediction,
                    "retrieval_index": index_name,
                    "retrieved_ids": retrieved_ids,
                    "retrieved_distances": retrieved_distances,
                    "exact_match": em,
                    "f1": f1,
                    "rouge_l": rouge_l,
                    "bleu": bleu,
                    "generation_latency_ms": generation_latency_ms,
                }
                out.write(json.dumps(result_row, ensure_ascii=False) + "\n")

            latency_stats = summarize_values(generation_latencies)
            per_seed_rows.append(
                {
                    "seed": seed,
                    "index": index_name,
                    "generation_model": generation_model,
                    "exact_match": float(np.mean(em_scores)),
                    "f1": float(np.mean(f1_scores)),
                    "rouge_l": float(np.mean(rouge_scores)),
                    "bleu": float(np.mean(bleu_scores)),
                    "avg_gen_ms": latency_stats["avg"],
                    "p50_gen_ms": latency_stats["p50"],
                    "p95_gen_ms": latency_stats["p95"],
                    "p99_gen_ms": latency_stats["p99"],
                }
            )

    with per_seed_summary_path.open("w", encoding="utf-8") as f:
        json.dump(per_seed_rows, f, ensure_ascii=False, indent=2)

    with per_seed_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "index",
                "generation_model",
                "exact_match",
                "f1",
                "rouge_l",
                "bleu",
                "avg_gen_ms",
                "p50_gen_ms",
                "p95_gen_ms",
                "p99_gen_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(per_seed_rows)

    combined_summary = {
        "index": index_name,
        "generation_model": generation_model,
        "exact_match_mean": float(np.mean([row["exact_match"] for row in per_seed_rows])),
        "exact_match_std": float(np.std([row["exact_match"] for row in per_seed_rows], ddof=0)),
        "f1_mean": float(np.mean([row["f1"] for row in per_seed_rows])),
        "f1_std": float(np.std([row["f1"] for row in per_seed_rows], ddof=0)),
        "rouge_l_mean": float(np.mean([row["rouge_l"] for row in per_seed_rows])),
        "rouge_l_std": float(np.std([row["rouge_l"] for row in per_seed_rows], ddof=0)),
        "bleu_mean": float(np.mean([row["bleu"] for row in per_seed_rows])),
        "bleu_std": float(np.std([row["bleu"] for row in per_seed_rows], ddof=0)),
        "avg_gen_ms_mean": float(np.mean([row["avg_gen_ms"] for row in per_seed_rows])),
        "avg_gen_ms_std": float(np.std([row["avg_gen_ms"] for row in per_seed_rows], ddof=0)),
        "p50_gen_ms_mean": float(np.mean([row["p50_gen_ms"] for row in per_seed_rows])),
        "p95_gen_ms_mean": float(np.mean([row["p95_gen_ms"] for row in per_seed_rows])),
        "p99_gen_ms_mean": float(np.mean([row["p99_gen_ms"] for row in per_seed_rows])),
    }

    with combined_summary_path.open("w", encoding="utf-8") as f:
        json.dump(combined_summary, f, ensure_ascii=False, indent=2)

    with combined_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "generation_model",
                "exact_match_mean",
                "exact_match_std",
                "f1_mean",
                "f1_std",
                "rouge_l_mean",
                "rouge_l_std",
                "bleu_mean",
                "bleu_std",
                "avg_gen_ms_mean",
                "avg_gen_ms_std",
                "p50_gen_ms_mean",
                "p95_gen_ms_mean",
                "p99_gen_ms_mean",
            ],
        )
        writer.writeheader()
        writer.writerow(combined_summary)

    create_markdown_report(
        path=report_path,
        per_seed_rows=per_seed_rows,
        combined_summary=combined_summary,
        sample_size=sample_size,
        seeds=seeds,
        index_name=index_name,
        generation_model=generation_model,
        top_k=retrieval_top_k,
    )

    print(f"Saved sampled query IDs: {sampled_queries_path}")
    print(f"Saved per-query raw results: {query_results_path}")
    print(f"Saved per-seed summary: {per_seed_summary_path}")
    print(f"Saved per-seed CSV: {per_seed_csv_path}")
    print(f"Saved combined summary: {combined_summary_path}")
    print(f"Saved combined CSV: {combined_csv_path}")
    print(f"Saved markdown report: {report_path}")


if __name__ == "__main__":
    main()
