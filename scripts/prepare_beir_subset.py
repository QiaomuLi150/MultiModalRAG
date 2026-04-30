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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    query_ds = load_dataset(f"BeIR/{args.dataset}", "queries", split="queries")
    qrels_ds = load_dataset(f"BeIR/{args.dataset}-qrels", split=args.qrels_split)
    corpus_ds = load_dataset(f"BeIR/{args.dataset}", "corpus", split="corpus") if args.export_corpus else None

    query_by_id = {str(row["_id"]): row for row in query_ds}
    qrels_by_query: dict[str, list[dict]] = {}
    ordered_query_ids: list[str] = []
    for row in qrels_ds:
        query_id = str(row["query-id"])
        if query_id not in qrels_by_query:
            qrels_by_query[query_id] = []
            ordered_query_ids.append(query_id)
        qrels_by_query[query_id].append(
            {
                "corpus_id": str(row["corpus-id"]),
                "score": int(row["score"]),
            }
        )

    selected_query_ids = ordered_query_ids[: args.limit]
    questions_path = output_dir / f"beir_{args.dataset}_{args.qrels_split}_{args.limit}.jsonl"
    qrels_path = output_dir / f"beir_{args.dataset}_{args.qrels_split}_{args.limit}_qrels.jsonl"
    manifest_path = output_dir / f"beir_{args.dataset}_{args.qrels_split}_{args.limit}_manifest.json"
    corpus_path = output_dir / f"beir_{args.dataset}_{args.qrels_split}_{args.limit}_corpus.csv"

    question_count = 0
    with questions_path.open("w", encoding="utf-8") as qf, qrels_path.open("w", encoding="utf-8") as rf:
        for query_id in selected_query_ids:
            query_row = query_by_id.get(query_id)
            if query_row is None:
                continue
            question_record = {
                "id": f"{args.dataset}_{args.qrels_split}_{query_id}",
                "question": str(query_row.get("text") or "").strip(),
                "expected_modality": "text",
                "expected_source": "",
                "dataset": f"BeIR/{args.dataset}",
                "split": args.qrels_split,
                "beir_query_id": query_id,
                "notes": f"Converted from BeIR/{args.dataset} with qrels split {args.qrels_split}.",
            }
            qf.write(json.dumps(question_record, ensure_ascii=True) + "\n")
            rf.write(
                json.dumps(
                    {
                        "id": question_record["id"],
                        "beir_query_id": query_id,
                        "qrels": sorted(
                            qrels_by_query.get(query_id, []),
                            key=lambda row: row["score"],
                            reverse=True,
                        ),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            question_count += 1

    corpus_doc_count = 0
    if args.export_corpus and corpus_ds is not None:
        needed_doc_ids = {
            doc["corpus_id"]
            for query_id in selected_query_ids
            for doc in qrels_by_query.get(query_id, [])
        }
        with corpus_path.open("w", encoding="utf-8", newline="") as handle:
            import csv

            writer = csv.DictWriter(handle, fieldnames=["doc_id", "title", "text"])
            writer.writeheader()
            for row in corpus_ds:
                doc_id = str(row["_id"])
                if doc_id not in needed_doc_ids:
                    continue
                writer.writerow(
                    {
                        "doc_id": doc_id,
                        "title": str(row.get("title") or ""),
                        "text": str(row.get("text") or ""),
                    }
                )
                corpus_doc_count += 1

    manifest = {
        "dataset": f"BeIR/{args.dataset}",
        "qrels_dataset": f"BeIR/{args.dataset}-qrels",
        "qrels_split": args.qrels_split,
        "question_count": question_count,
        "corpus_doc_count": corpus_doc_count,
        "questions_path": str(questions_path),
        "qrels_path": str(qrels_path),
        "corpus_path": str(corpus_path) if args.export_corpus else None,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    print("Prepared BEIR subset")
    print(f"- dataset: {manifest['dataset']}")
    print(f"- qrels_split: {args.qrels_split}")
    print(f"- question_count: {question_count}")
    if args.export_corpus:
        print(f"- corpus_doc_count: {corpus_doc_count}")
    print(f"- questions_path: {questions_path}")
    print(f"- qrels_path: {qrels_path}")
    if args.export_corpus:
        print(f"- corpus_path: {corpus_path}")
    print(f"- manifest_path: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a small BEIR question subset for MultiModalRAG evaluation.")
    parser.add_argument("--dataset", default="nfcorpus", help="BEIR subset name, e.g. nfcorpus or trec-covid.")
    parser.add_argument("--qrels-split", default="test", help="Qrels split to sample from, e.g. test or validation.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of unique queries to export.")
    parser.add_argument(
        "--export-corpus",
        action="store_true",
        help="Also export the matching corpus rows referenced by the selected qrels into a local CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "eval" / "questions" / "public"),
        help="Directory to write question and qrels files into.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
