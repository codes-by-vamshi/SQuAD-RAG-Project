import argparse
import json
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv


def load_config(config_path: Path) -> tuple[str, int, int]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    model = str(config.get("embedding_model", "text-embedding-3-small"))
    batch_size = int(config.get("embedding_batch_size", 100))
    flush_every_batches = int(config.get("flush_every_batches", 10))

    if batch_size <= 0:
        raise ValueError("`embedding_batch_size` must be > 0.")
    if flush_every_batches < 0:
        raise ValueError("`flush_every_batches` must be >= 0.")

    return model, batch_size, flush_every_batches


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
            yield record


def batched(items, batch_size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def create_embeddings(client, model: str, texts: list[str]):
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _call():
        return client.embeddings.create(model=model, input=texts)

    return _call()


def write_embeddings(
    client,
    input_path: Path,
    output_path: Path,
    model: str,
    batch_size: int,
    flush_every_batches: int,
    limit: int | None,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    vector_dim = 0
    started = time.time()
    processed_input = 0

    def limited_records():
        nonlocal processed_input
        for record in iter_jsonl(input_path):
            if limit is not None and processed_input >= limit:
                break
            processed_input += 1
            yield record

    with output_path.open("w", encoding="utf-8") as out:
        for batch_index, batch in enumerate(batched(limited_records(), batch_size), start=1):
            texts = [item.get("text", "") for item in batch]
            response = create_embeddings(client, model, texts)
            embeddings = [row.embedding for row in response.data]

            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Embedding API size mismatch in batch {batch_index}: "
                    f"expected {len(batch)}, got {len(embeddings)}"
                )

            for record, embedding in zip(batch, embeddings):
                if not vector_dim:
                    vector_dim = len(embedding)

                output_record = dict(record)
                output_record["embedding_model"] = model
                output_record["embedding"] = embedding
                out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                total_chunks += 1

            if flush_every_batches > 0 and batch_index % flush_every_batches == 0:
                out.flush()

            if batch_index % 10 == 0:
                elapsed = time.time() - started
                print(f"Processed {total_chunks} chunks in {elapsed:.1f}s")

        out.flush()

    return total_chunks, vector_dim


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: Create OpenAI embeddings for chunked text.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./configs/baseline.yaml"),
        help="Path to YAML config (embedding_model, embedding_batch_size).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./data/splits/chunks.jsonl"),
        help="Input JSONL file from Step 2.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./embeddings/embeddings.jsonl"),
        help="Output JSONL file with embeddings.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override embedding model from config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override embedding batch size from config.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of chunks to embed (useful for smoke tests).",
    )
    parser.add_argument(
        "--flush-every-batches",
        type=int,
        default=None,
        help="Flush output writer every N batches (0 disables periodic flushing).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing output file if it already exists.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("`--limit` must be > 0 when provided.")

    model_cfg, batch_cfg, flush_cfg = load_config(args.config)
    model = args.model or model_cfg
    batch_size = args.batch_size or batch_cfg
    flush_every_batches = (
        args.flush_every_batches if args.flush_every_batches is not None else flush_cfg
    )

    if batch_size <= 0:
        raise ValueError("`batch-size` must be > 0.")
    if flush_every_batches < 0:
        raise ValueError("`flush-every-batches` must be >= 0.")

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {args.output}\n"
            "Use --overwrite to replace it."
        )

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
    total_chunks, vector_dim = write_embeddings(
        client=client,
        input_path=args.input,
        output_path=args.output,
        model=model,
        batch_size=batch_size,
        flush_every_batches=flush_every_batches,
        limit=args.limit,
    )

    print(f"Model: {model}")
    print(f"Batch size: {batch_size}")
    print(f"Flush every batches: {flush_every_batches}")
    print(f"Embedded chunks: {total_chunks}")
    print(f"Embedding dimension: {vector_dim}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
