import argparse
import json
import os
import random
from pathlib import Path

import faiss
import numpy as np
import yaml
from dotenv import load_dotenv


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = {
        "embedding_model": str(raw.get("embedding_model", "text-embedding-3-small")),
        "ivf_nprobe": int(raw.get("ivf_nprobe", 32)),
        "hnsw_ef_search": int(raw.get("hnsw_ef_search", 32)),
        "flat_l2_index_filename": str(raw.get("flat_l2_index_filename", "flat_l2.index")),
        "ivf_flat_index_filename": str(raw.get("ivf_flat_index_filename", "ivf_flat.index")),
        "hnsw_index_filename": str(raw.get("hnsw_index_filename", "hnsw.index")),
        "retrieval_top_k": int(raw.get("retrieval_top_k", 5)),
        "retrieval_preview_chars": int(raw.get("retrieval_preview_chars", 220)),
        "retrieval_seed": int(raw.get("retrieval_seed", 42)),
    }

    if cfg["ivf_nprobe"] <= 0:
        raise ValueError("`ivf_nprobe` must be > 0.")
    if cfg["hnsw_ef_search"] <= 0:
        raise ValueError("`hnsw_ef_search` must be > 0.")
    if cfg["retrieval_top_k"] <= 0:
        raise ValueError("`retrieval_top_k` must be > 0.")
    if cfg["retrieval_preview_chars"] <= 0:
        raise ValueError("`retrieval_preview_chars` must be > 0.")

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


def load_metadata(path: Path) -> tuple[list[dict], int]:
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
                "doc_name": record.get("doc_name", ""),
                "doc_id": record.get("doc_id"),
                "chunk_id": record.get("chunk_id", ""),
                "chunk_index": record.get("chunk_index"),
                "text": record.get("text", ""),
            }
        )

    if not metadata:
        raise ValueError(f"No embedding records found: {path}")

    return metadata, expected_dim


def sample_question(qa_dir: Path, rng: random.Random) -> tuple[str, str, str]:
    qa_files = sorted(qa_dir.glob("*.json"))
    if not qa_files:
        raise FileNotFoundError(f"No QA files found in directory: {qa_dir}")

    candidates = qa_files[:]
    rng.shuffle(candidates)
    for qa_file in candidates:
        with qa_file.open("r", encoding="utf-8") as f:
            qa_pairs = json.load(f)
        if not isinstance(qa_pairs, list) or not qa_pairs:
            continue

        pair = rng.choice(qa_pairs)
        question = str(pair.get("q", "")).strip()
        answer = str(pair.get("a", "")).strip()
        if question:
            return question, answer, qa_file.name

    raise ValueError(f"No valid QA pairs with non-empty questions found in {qa_dir}")


def embed_question(client, model: str, question: str) -> np.ndarray:
    response = client.embeddings.create(model=model, input=question)
    embedding = response.data[0].embedding
    return np.array([embedding], dtype=np.float32)


def clip_text(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def search_and_print(
    label: str,
    index_path: Path,
    query_vector: np.ndarray,
    top_k: int,
    metadata: list[dict],
    ivf_nprobe: int,
    hnsw_ef_search: int,
    preview_chars: int,
) -> None:
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    index = faiss.read_index(str(index_path))
    if hasattr(index, "nprobe"):
        nlist = getattr(index, "nlist", ivf_nprobe)
        index.nprobe = min(ivf_nprobe, max(1, nlist))
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = hnsw_ef_search

    distances, ids = index.search(query_vector, top_k)
    print(f"\n{label} ({index_path}):")

    for rank, (chunk_idx, distance) in enumerate(zip(ids[0], distances[0]), start=1):
        if chunk_idx < 0 or chunk_idx >= len(metadata):
            print(f"{rank}. id={chunk_idx} dist={distance:.4f} (no match)")
            continue

        record = metadata[chunk_idx]
        snippet = clip_text(str(record.get("text", "")), preview_chars)
        print(
            f"{rank}. id={chunk_idx} dist={distance:.4f} "
            f"doc={record.get('doc_name')} chunk={record.get('chunk_id')}"
        )
        print(f"   {snippet}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: Retrieve top-k chunks using FAISS indexes.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./configs/baseline.yaml"),
        help="Path to YAML config file.",
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
        help="Directory containing FAISS index files from Step 4.",
    )
    parser.add_argument(
        "--qa-dir",
        type=Path,
        default=Path("./data/qa"),
        help="Directory containing QA JSON files (used if --question is not provided).",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Optional question text. If omitted, a random QA sample is used.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top-k retrieval results.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=None,
        help="Override text preview length per retrieved chunk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible random QA sampling.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override embedding model used for query embedding.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    top_k = args.top_k if args.top_k is not None else cfg["retrieval_top_k"]
    preview_chars = (
        args.preview_chars if args.preview_chars is not None else cfg["retrieval_preview_chars"]
    )
    seed = args.seed if args.seed is not None else cfg["retrieval_seed"]
    model = args.model or cfg["embedding_model"]

    if top_k <= 0:
        raise ValueError("`top-k` must be > 0.")
    if preview_chars <= 0:
        raise ValueError("`preview-chars` must be > 0.")

    if not args.embeddings.exists():
        raise FileNotFoundError(f"Embeddings file not found: {args.embeddings}")

    metadata, embedding_dim = load_metadata(args.embeddings)

    rng = random.Random(seed)
    reference_answer = None
    qa_file_name = None

    if args.question and args.question.strip():
        question = args.question.strip()
    else:
        question, reference_answer, qa_file_name = sample_question(args.qa_dir, rng)

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
    query_vector = embed_question(client=client, model=model, question=question)

    if query_vector.shape[1] != embedding_dim:
        raise ValueError(
            f"Query embedding dimension mismatch: query={query_vector.shape[1]}, "
            f"index embeddings={embedding_dim}"
        )

    flat_path = args.indexes_dir / cfg["flat_l2_index_filename"]
    ivf_path = args.indexes_dir / cfg["ivf_flat_index_filename"]
    hnsw_path = args.indexes_dir / cfg["hnsw_index_filename"]

    print(f"Model: {model}")
    print(f"Top-k: {top_k}")
    print(f"Seed: {seed}")
    if qa_file_name:
        print(f"QA sample file: {qa_file_name}")
    print(f"Question: {question}")
    if reference_answer is not None:
        print(f"Reference answer: {reference_answer}")

    search_and_print(
        label="FlatL2",
        index_path=flat_path,
        query_vector=query_vector,
        top_k=top_k,
        metadata=metadata,
        ivf_nprobe=cfg["ivf_nprobe"],
        hnsw_ef_search=cfg["hnsw_ef_search"],
        preview_chars=preview_chars,
    )
    search_and_print(
        label="IVFFlat",
        index_path=ivf_path,
        query_vector=query_vector,
        top_k=top_k,
        metadata=metadata,
        ivf_nprobe=cfg["ivf_nprobe"],
        hnsw_ef_search=cfg["hnsw_ef_search"],
        preview_chars=preview_chars,
    )
    search_and_print(
        label="HNSW",
        index_path=hnsw_path,
        query_vector=query_vector,
        top_k=top_k,
        metadata=metadata,
        ivf_nprobe=cfg["ivf_nprobe"],
        hnsw_ef_search=cfg["hnsw_ef_search"],
        preview_chars=preview_chars,
    )


if __name__ == "__main__":
    main()
