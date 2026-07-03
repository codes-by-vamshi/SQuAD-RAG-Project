import argparse
import csv
import json
import os
import random
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
        raise ValueError("`index_eval_seeds` must be a list[int] or comma-separated string.")

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
        "index_eval_sample_size": int(raw.get("index_eval_sample_size", 100)),
        "index_eval_top_k": int(raw.get("index_eval_top_k", 5)),
        "index_eval_seeds": parse_seed_list(raw.get("index_eval_seeds", [11, 22, 33, 44, 55, 66])),
        "index_eval_output_dir": str(raw.get("index_eval_output_dir", "results/index_eval")),
    }

    if cfg["embedding_batch_size"] <= 0:
        raise ValueError("`embedding_batch_size` must be > 0.")
    if cfg["ivf_nprobe"] <= 0:
        raise ValueError("`ivf_nprobe` must be > 0.")
    if cfg["hnsw_ef_search"] <= 0:
        raise ValueError("`hnsw_ef_search` must be > 0.")
    if cfg["index_eval_sample_size"] <= 0:
        raise ValueError("`index_eval_sample_size` must be > 0.")
    if cfg["index_eval_top_k"] <= 0:
        raise ValueError("`index_eval_top_k` must be > 0.")

    return cfg


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
            if not question:
                continue
            query_id = f"{qa_file.stem}:{qa_index}"
            pool.append(
                {
                    "query_id": query_id,
                    "question": question,
                    "qa_file": qa_file.name,
                    "qa_index": qa_index,
                }
            )

    if not pool:
        raise ValueError(f"No non-empty questions found in QA directory: {qa_dir}")

    return pool


def sample_query_ids(pool: list[dict], sample_size: int, seeds: list[int]) -> dict[int, list[str]]:
    if sample_size > len(pool):
        raise ValueError(
            f"Sample size ({sample_size}) is larger than available questions ({len(pool)})."
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

    for batch_no, batch in enumerate(batched(questions, batch_size), start=1):
        inputs = [item["question"] for item in batch]
        response = client.embeddings.create(model=model, input=inputs)

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


def search_with_latency(index, query_vector: np.ndarray, top_k: int) -> tuple[list[int], list[float], float]:
    started = perf_counter_ns()
    distances, ids = index.search(query_vector, top_k)
    elapsed_ms = (perf_counter_ns() - started) / 1_000_000
    return [int(v) for v in ids[0]], [float(v) for v in distances[0]], elapsed_ms


def recall_against_flat(flat_ids: list[int], approx_ids: list[int]) -> float:
    flat_set = {idx for idx in flat_ids if idx >= 0}
    approx_set = {idx for idx in approx_ids if idx >= 0}
    if not flat_set:
        return 0.0
    return len(flat_set & approx_set) / len(flat_set)


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
    combined_rows: list[dict],
    sample_size: int,
    top_k: int,
    seeds: list[int],
) -> None:
    lines = [
        "# Index Quality Evaluation",
        "",
        f"- Sample size per seed: {sample_size}",
        f"- Top-k: {top_k}",
        f"- Seeds: {', '.join(str(s) for s in seeds)}",
        "",
        "## Per-Seed Results",
        "",
        "| seed | index | recall@k | avg_ms | p50_ms | p95_ms | p99_ms |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]

    for row in per_seed_rows:
        lines.append(
            f"| {row['seed']} | {row['index']} | {row['recall_at_k']:.4f} | "
            f"{row['avg_ms']:.4f} | {row['p50_ms']:.4f} | {row['p95_ms']:.4f} | {row['p99_ms']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Combined Across Seeds",
            "",
            "| index | recall_mean | recall_std | avg_ms_mean | avg_ms_std | p50_ms_mean | p95_ms_mean | p99_ms_mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in combined_rows:
        lines.append(
            f"| {row['index']} | {row['recall_at_k_mean']:.4f} | {row['recall_at_k_std']:.4f} | "
            f"{row['avg_ms_mean']:.4f} | {row['avg_ms_std']:.4f} | {row['p50_ms_mean']:.4f} | "
            f"{row['p95_ms_mean']:.4f} | {row['p99_ms_mean']:.4f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 6: Evaluate FAISS index fidelity (Recall@k vs FlatL2) and latency."
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
        help="Override number of sampled questions per seed.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top-k neighbors for retrieval fidelity.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (example: 11,22,33,44,55,66).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override embedding model used for query embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override embedding batch size for sampled questions.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    sample_size = args.sample_size if args.sample_size is not None else cfg["index_eval_sample_size"]
    top_k = args.top_k if args.top_k is not None else cfg["index_eval_top_k"]
    seeds = parse_seed_list(args.seeds) if args.seeds is not None else cfg["index_eval_seeds"]
    model = args.model or cfg["embedding_model"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["embedding_batch_size"]
    output_dir = args.output_dir or Path(cfg["index_eval_output_dir"])

    if sample_size <= 0:
        raise ValueError("`sample-size` must be > 0.")
    if top_k <= 0:
        raise ValueError("`top-k` must be > 0.")
    if batch_size <= 0:
        raise ValueError("`batch-size` must be > 0.")

    index_paths = {
        "flat_l2": args.indexes_dir / cfg["flat_l2_index_filename"],
        "ivf_flat": args.indexes_dir / cfg["ivf_flat_index_filename"],
        "hnsw": args.indexes_dir / cfg["hnsw_index_filename"],
    }
    for label, path in index_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing index file for {label}: {path}")

    pool = load_question_pool(args.qa_dir)
    sampled_by_seed = sample_query_ids(pool=pool, sample_size=sample_size, seeds=seeds)

    selected_ids = {query_id for query_ids in sampled_by_seed.values() for query_id in query_ids}
    selected_questions = [item for item in pool if item["query_id"] in selected_ids]
    selected_questions.sort(key=lambda item: item["query_id"])
    question_by_id = {item["query_id"]: item["question"] for item in selected_questions}

    print(f"Question pool size: {len(pool)}")
    print(f"Unique sampled questions across all seeds: {len(selected_questions)}")
    print(f"Seeds: {seeds}")
    print(f"Sample size per seed: {sample_size}")
    print(f"Top-k: {top_k}")

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
        model=model,
        questions=selected_questions,
        batch_size=batch_size,
    )

    flat_index = faiss.read_index(str(index_paths["flat_l2"]))
    ivf_index = faiss.read_index(str(index_paths["ivf_flat"]))
    hnsw_index = faiss.read_index(str(index_paths["hnsw"]))
    set_index_runtime_params(flat_index, cfg["ivf_nprobe"], cfg["hnsw_ef_search"])
    set_index_runtime_params(ivf_index, cfg["ivf_nprobe"], cfg["hnsw_ef_search"])
    set_index_runtime_params(hnsw_index, cfg["ivf_nprobe"], cfg["hnsw_ef_search"])

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "sampled_queries_by_seed.json"
    query_results_path = output_dir / "query_results.jsonl"
    per_seed_summary_path = output_dir / "per_seed_summary.json"
    per_seed_csv_path = output_dir / "per_seed_summary.csv"
    combined_summary_path = output_dir / "combined_summary.json"
    combined_csv_path = output_dir / "combined_summary.csv"
    report_path = output_dir / "report.md"

    with samples_path.open("w", encoding="utf-8") as f:
        json.dump(sampled_by_seed, f, ensure_ascii=False, indent=2)

    per_seed_rows = []
    with query_results_path.open("w", encoding="utf-8") as out:
        for seed in seeds:
            print(f"Evaluating seed {seed}...")
            seed_query_ids = sampled_by_seed[seed]

            recalls = {"flat_l2": [], "ivf_flat": [], "hnsw": []}
            latencies = {"flat_l2": [], "ivf_flat": [], "hnsw": []}

            for query_id in seed_query_ids:
                query_vector = query_vectors[query_id]
                question = question_by_id[query_id]

                flat_ids, flat_distances, flat_ms = search_with_latency(flat_index, query_vector, top_k)
                ivf_ids, ivf_distances, ivf_ms = search_with_latency(ivf_index, query_vector, top_k)
                hnsw_ids, hnsw_distances, hnsw_ms = search_with_latency(hnsw_index, query_vector, top_k)

                flat_recall = recall_against_flat(flat_ids, flat_ids)
                ivf_recall = recall_against_flat(flat_ids, ivf_ids)
                hnsw_recall = recall_against_flat(flat_ids, hnsw_ids)

                recalls["flat_l2"].append(flat_recall)
                recalls["ivf_flat"].append(ivf_recall)
                recalls["hnsw"].append(hnsw_recall)
                latencies["flat_l2"].append(flat_ms)
                latencies["ivf_flat"].append(ivf_ms)
                latencies["hnsw"].append(hnsw_ms)

                query_row = {
                    "seed": seed,
                    "query_id": query_id,
                    "question": question,
                    "flat_ids": flat_ids,
                    "flat_distances": flat_distances,
                    "ivf_ids": ivf_ids,
                    "ivf_distances": ivf_distances,
                    "hnsw_ids": hnsw_ids,
                    "hnsw_distances": hnsw_distances,
                    "flat_recall_at_k": flat_recall,
                    "ivf_recall_at_k": ivf_recall,
                    "hnsw_recall_at_k": hnsw_recall,
                    "flat_latency_ms": flat_ms,
                    "ivf_latency_ms": ivf_ms,
                    "hnsw_latency_ms": hnsw_ms,
                }
                out.write(json.dumps(query_row, ensure_ascii=False) + "\n")

            for index_name in ["flat_l2", "ivf_flat", "hnsw"]:
                latency_stats = summarize_values(latencies[index_name])
                row = {
                    "seed": seed,
                    "index": index_name,
                    "recall_at_k": float(np.mean(recalls[index_name])),
                    "avg_ms": latency_stats["avg"],
                    "p50_ms": latency_stats["p50"],
                    "p95_ms": latency_stats["p95"],
                    "p99_ms": latency_stats["p99"],
                }
                per_seed_rows.append(row)

    with per_seed_summary_path.open("w", encoding="utf-8") as f:
        json.dump(per_seed_rows, f, ensure_ascii=False, indent=2)

    with per_seed_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "index", "recall_at_k", "avg_ms", "p50_ms", "p95_ms", "p99_ms"],
        )
        writer.writeheader()
        writer.writerows(per_seed_rows)

    combined_rows = []
    for index_name in ["flat_l2", "ivf_flat", "hnsw"]:
        rows = [row for row in per_seed_rows if row["index"] == index_name]

        combined_rows.append(
            {
                "index": index_name,
                "recall_at_k_mean": float(np.mean([row["recall_at_k"] for row in rows])),
                "recall_at_k_std": float(np.std([row["recall_at_k"] for row in rows], ddof=0)),
                "avg_ms_mean": float(np.mean([row["avg_ms"] for row in rows])),
                "avg_ms_std": float(np.std([row["avg_ms"] for row in rows], ddof=0)),
                "p50_ms_mean": float(np.mean([row["p50_ms"] for row in rows])),
                "p95_ms_mean": float(np.mean([row["p95_ms"] for row in rows])),
                "p99_ms_mean": float(np.mean([row["p99_ms"] for row in rows])),
            }
        )

    with combined_summary_path.open("w", encoding="utf-8") as f:
        json.dump(combined_rows, f, ensure_ascii=False, indent=2)

    with combined_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "recall_at_k_mean",
                "recall_at_k_std",
                "avg_ms_mean",
                "avg_ms_std",
                "p50_ms_mean",
                "p95_ms_mean",
                "p99_ms_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(combined_rows)

    create_markdown_report(
        path=report_path,
        per_seed_rows=per_seed_rows,
        combined_rows=combined_rows,
        sample_size=sample_size,
        top_k=top_k,
        seeds=seeds,
    )

    print(f"Saved sampled query IDs: {samples_path}")
    print(f"Saved per-query raw results: {query_results_path}")
    print(f"Saved per-seed summary: {per_seed_summary_path}")
    print(f"Saved per-seed CSV: {per_seed_csv_path}")
    print(f"Saved combined summary: {combined_summary_path}")
    print(f"Saved combined CSV: {combined_csv_path}")
    print(f"Saved markdown report: {report_path}")


if __name__ == "__main__":
    main()
