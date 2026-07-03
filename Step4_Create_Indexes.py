import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = {
        "ivf_nlist_divisor": int(raw.get("ivf_nlist_divisor", 39)),
        "ivf_nlist_min": int(raw.get("ivf_nlist_min", 1)),
        "ivf_nprobe": int(raw.get("ivf_nprobe", 32)),
        "hnsw_m": int(raw.get("hnsw_m", 16)),
        "hnsw_ef_construction": int(raw.get("hnsw_ef_construction", 100)),
        "hnsw_ef_search": int(raw.get("hnsw_ef_search", 32)),
        "flat_l2_index_filename": str(raw.get("flat_l2_index_filename", "flat_l2.index")),
        "ivf_flat_index_filename": str(raw.get("ivf_flat_index_filename", "ivf_flat.index")),
        "hnsw_index_filename": str(raw.get("hnsw_index_filename", "hnsw.index")),
    }

    if cfg["ivf_nlist_divisor"] <= 0:
        raise ValueError("`ivf_nlist_divisor` must be > 0.")
    if cfg["ivf_nlist_min"] <= 0:
        raise ValueError("`ivf_nlist_min` must be > 0.")
    if cfg["ivf_nprobe"] <= 0:
        raise ValueError("`ivf_nprobe` must be > 0.")
    if cfg["hnsw_m"] <= 0:
        raise ValueError("`hnsw_m` must be > 0.")
    if cfg["hnsw_ef_construction"] <= 0:
        raise ValueError("`hnsw_ef_construction` must be > 0.")
    if cfg["hnsw_ef_search"] <= 0:
        raise ValueError("`hnsw_ef_search` must be > 0.")

    return cfg


def load_vectors(path: Path) -> np.ndarray:
    vectors = []
    expected_dim = None

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc

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

            vectors.append(embedding)

    if not vectors:
        raise ValueError(f"No embeddings found in input: {path}")

    return np.array(vectors, dtype=np.float32)


def build_and_write_indexes(vectors: np.ndarray, output_dir: Path, cfg: dict) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    num_vectors, dim = vectors.shape

    flat_l2_path = output_dir / cfg["flat_l2_index_filename"]
    ivf_flat_path = output_dir / cfg["ivf_flat_index_filename"]
    hnsw_path = output_dir / cfg["hnsw_index_filename"]

    flat_l2_index = faiss.IndexFlatL2(dim)
    flat_l2_index.add(vectors)
    faiss.write_index(flat_l2_index, str(flat_l2_path))

    nlist_raw = max(cfg["ivf_nlist_min"], num_vectors // cfg["ivf_nlist_divisor"])
    nlist = min(nlist_raw, num_vectors)
    ivf_index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist)
    ivf_index.train(vectors)
    ivf_index.add(vectors)
    ivf_index.nprobe = min(cfg["ivf_nprobe"], nlist)
    faiss.write_index(ivf_index, str(ivf_flat_path))

    hnsw_index = faiss.IndexHNSWFlat(dim, cfg["hnsw_m"])
    hnsw_index.hnsw.efConstruction = cfg["hnsw_ef_construction"]
    hnsw_index.add(vectors)
    hnsw_index.hnsw.efSearch = cfg["hnsw_ef_search"]
    faiss.write_index(hnsw_index, str(hnsw_path))

    return {
        "flat_l2": flat_l2_path,
        "ivf_flat": ivf_flat_path,
        "hnsw": hnsw_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: Build FAISS indexes from embeddings.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./configs/baseline.yaml"),
        help="Path to YAML config with index parameters.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./embeddings/embeddings.jsonl"),
        help="Input JSONL path produced by Step 3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./indexes"),
        help="Directory to write FAISS index files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing index files.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input embeddings file not found: {args.input}")

    cfg = load_config(args.config)
    flat_l2_path = args.output_dir / cfg["flat_l2_index_filename"]
    ivf_flat_path = args.output_dir / cfg["ivf_flat_index_filename"]
    hnsw_path = args.output_dir / cfg["hnsw_index_filename"]
    output_paths = [flat_l2_path, ivf_flat_path, hnsw_path]

    if not args.overwrite:
        existing = [p for p in output_paths if p.exists()]
        if existing:
            existing_list = "\n".join(str(p) for p in existing)
            raise FileExistsError(
                "Output index file(s) already exist:\n"
                f"{existing_list}\nUse --overwrite to replace them."
            )

    vectors = load_vectors(args.input)
    written = build_and_write_indexes(vectors=vectors, output_dir=args.output_dir, cfg=cfg)

    print(f"Input vectors: {vectors.shape[0]}")
    print(f"Embedding dimension: {vectors.shape[1]}")
    print(f"IVF nlist divisor: {cfg['ivf_nlist_divisor']}")
    print(f"IVF nlist min: {cfg['ivf_nlist_min']}")
    print(f"IVF nprobe: {cfg['ivf_nprobe']}")
    print(f"HNSW M: {cfg['hnsw_m']}")
    print(f"HNSW efConstruction: {cfg['hnsw_ef_construction']}")
    print(f"HNSW efSearch: {cfg['hnsw_ef_search']}")
    print(f"FlatL2 index: {written['flat_l2']}")
    print(f"IVF Flat index: {written['ivf_flat']}")
    print(f"HNSW index: {written['hnsw']}")


if __name__ == "__main__":
    main()
