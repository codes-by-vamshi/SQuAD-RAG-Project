import argparse
import json
from pathlib import Path


def load_squad(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "data" not in data or not isinstance(data["data"], list):
        raise ValueError("Invalid SQuAD format: missing top-level 'data' list.")
    return data


def export_documents_and_qa(raw_data: dict, output_root: Path) -> tuple[int, int]:
    documents_dir = output_root / "documents"
    qa_dir = output_root / "qa"
    documents_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    document_id = 0
    question_count = 0

    for article in raw_data["data"]:
        paragraphs = article.get("paragraphs", [])
        for para in paragraphs:
            context = para.get("context", "")
            qas = para.get("qas", [])

            qa_pairs = []
            for item in qas:
                answers = item.get("answers", [])
                first_answer = answers[0]["text"] if answers else ""
                qa_pairs.append({"q": item.get("question", ""), "a": first_answer})
            question_count += len(qa_pairs)

            with (documents_dir / f"{document_id}.txt").open("w", encoding="utf-8") as f:
                f.write(context)
            with (qa_dir / f"{document_id}.json").open("w", encoding="utf-8") as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

            document_id += 1

    return document_id, question_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: Prepare SQuAD contexts and QA pairs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./data/raw/train-v1.1.json"),
        help="Path to SQuAD JSON (default: ./data/raw/train-v1.1.json).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./data"),
        help="Output root directory (default: ./data).",
    )
    args = parser.parse_args()

    raw_data = load_squad(args.input)
    doc_count, question_count = export_documents_and_qa(raw_data, args.output_root)
    print(f"Wrote {doc_count} documents to: {args.output_root / 'documents'}")
    print(f"Wrote {doc_count} QA files to: {args.output_root / 'qa'}")
    print(f"Total questions exported: {question_count}")


if __name__ == "__main__":
    main()
