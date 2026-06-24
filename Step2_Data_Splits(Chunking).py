import argparse
import json
from pathlib import Path

import yaml


def load_config(config_path: Path) -> tuple[int, int]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    chunk_size = int(config.get("chunk_size", 1024))
    overlap_size = int(config.get("overlap_size", 128))

    if chunk_size <= 0:
        raise ValueError("`chunk_size` must be > 0.")
    if overlap_size < 0:
        raise ValueError("`overlap_size` must be >= 0.")
    if overlap_size >= chunk_size:
        raise ValueError("`overlap_size` must be smaller than `chunk_size`.")

    return chunk_size, overlap_size


def chunk_text(text: str, chunk_size: int, overlap_size: int) -> list[tuple[str, int, int]]:
    if not text:
        return []

    step = chunk_size - overlap_size
    chunks: list[tuple[str, int, int]] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append((chunk, start, start + len(chunk)))
        if end >= len(text):
            break
        start += step

    return chunks


def parse_doc_id(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def write_chunks(input_dir: Path, output_path: Path, chunk_size: int, overlap_size: int) -> tuple[int, int]:
    txt_files = sorted(input_dir.glob("*.txt"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    total_chunks = 0

    with output_path.open("w", encoding="utf-8") as out:
        for file_path in txt_files:
            text = file_path.read_text(encoding="utf-8")
            doc_chunks = chunk_text(text, chunk_size, overlap_size)
            doc_id = parse_doc_id(file_path)

            for chunk_idx, (chunk, start, end) in enumerate(doc_chunks):
                record = {
                    "text": chunk,
                    "doc_id": doc_id,
                    "doc_name": file_path.name,
                    "chunk_id": f"{file_path.stem}_{chunk_idx}",
                    "chunk_index": chunk_idx,
                    "char_start": start,
                    "char_end": end,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1
            total_docs += 1

    return total_docs, total_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: Split documents into overlapping chunks.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./configs/baseline.yaml"),
        help="Path to YAML config with chunk_size and overlap_size.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./data/documents"),
        help="Directory containing .txt files from Step 1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/splits/chunks.jsonl"),
        help="Output JSONL path for chunks.",
    )
    args = parser.parse_args()

    chunk_size, overlap_size = load_config(args.config)
    total_docs, total_chunks = write_chunks(args.input_dir, args.output, chunk_size, overlap_size)

    print(f"Chunk size: {chunk_size}, overlap: {overlap_size}")
    print(f"Processed documents: {total_docs}")
    print(f"Written chunks: {total_chunks}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
