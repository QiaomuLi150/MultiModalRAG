from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split=f"validation[:{args.limit}]")
    questions_path = out_dir / f"docvqa_val_{args.limit}.jsonl"
    qrels_path = out_dir / f"docvqa_val_{args.limit}_qrels.jsonl"
    manifest_path = out_dir / f"docvqa_val_{args.limit}_manifest.json"

    unique_images = 0
    seen = set()
    with questions_path.open("w", encoding="utf-8") as qf, qrels_path.open("w", encoding="utf-8") as rf:
        for row in ds:
            question_id = str(row["questionId"])
            doc_id = str(row.get("docId") or "")
            page_no = str(row.get("ucsf_document_page_no") or "1")
            image_name = f"docvqa_doc_{doc_id}_page_{page_no}.png"
            image_path = images_dir / image_name
            if image_name not in seen:
                row["image"].convert("RGB").save(image_path, format="PNG")
                seen.add(image_name)
                unique_images += 1
            answers = row.get("answers") or []
            question_record = {
                "id": f"docvqa_val_{question_id}",
                "question": str(row["question"]).strip(),
                "expected_modality": "visual_page",
                "expected_source": image_name,
                "dataset": "lmms-lab/DocVQA",
                "split": "validation",
                "docvqa_question_id": question_id,
                "expected_answer": answers[0] if answers else "",
                "notes": f"DocVQA validation question for image {image_name}.",
            }
            qf.write(json.dumps(question_record, ensure_ascii=True) + "\n")
            rf.write(
                json.dumps(
                    {
                        "id": question_record["id"],
                        "qrels": [{"corpus_id": image_name, "score": 1}],
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    manifest = {
        "dataset": "lmms-lab/DocVQA",
        "split": "validation",
        "question_count": len(ds),
        "image_count": unique_images,
        "images_dir": str(images_dir),
        "questions_path": str(questions_path),
        "qrels_path": str(qrels_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    print("Prepared DocVQA subset")
    print(f"- question_count: {len(ds)}")
    print(f"- image_count: {unique_images}")
    print(f"- images_dir: {images_dir}")
    print(f"- questions_path: {questions_path}")
    print(f"- qrels_path: {qrels_path}")
    print(f"- manifest_path: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a small DocVQA validation subset for MultiModalRAG experiments.")
    parser.add_argument("--limit", type=int, default=20, help="Number of validation questions to export.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "eval" / "questions" / "public" / "docvqa_small"),
        help="Directory to write DocVQA subset assets into.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
