from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    jsonl_files = discover_jsonl_files(log_dir, run_labels=args.run_label)
    if not jsonl_files:
        raise SystemExit(f"No eval jsonl files found in {log_dir}")

    rows = []
    for path in jsonl_files:
        records = load_records(path)
        if not records:
            continue
        rows.append(summarize_run(path, records))

    if not rows:
        raise SystemExit("No non-empty eval runs were available to summarize.")

    rows.sort(key=lambda row: row["run_label"])
    print_table(rows)

    if args.output_csv:
        write_csv(Path(args.output_csv), rows)
        print(f"\nWrote summary CSV: {args.output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MultiModalRAG eval JSONL runs into one comparison table.")
    parser.add_argument(
        "--log-dir",
        default=str(PROJECT_DIR / ".artifacts" / "eval_logs"),
        help="Directory containing eval JSONL files.",
    )
    parser.add_argument(
        "--run-label",
        action="append",
        default=[],
        help="Optional specific run label(s) to include. Can be passed multiple times.",
    )
    parser.add_argument("--output-csv", help="Optional output CSV path for the aggregated comparison table.")
    return parser.parse_args()


def discover_jsonl_files(log_dir: Path, *, run_labels: list[str]) -> list[Path]:
    if not log_dir.exists():
        return []
    if run_labels:
        wanted = {slugify(label) for label in run_labels}
        return sorted(path for path in log_dir.glob("*.jsonl") if path.stem in wanted)
    return sorted(log_dir.glob("*.jsonl"))


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def summarize_run(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    first = records[0]
    settings = first.get("settings", {})
    latencies = [record.get("latencies", {}) for record in records]
    retrieval_metrics = [record.get("retrieval_metrics") for record in records if record.get("retrieval_metrics")]
    answer_metrics = [record.get("answer_metrics") for record in records if record.get("answer_metrics")]
    blocked = 0
    no_results = 0
    used_api = 0
    confidence_counts: dict[str, int] = {}

    for record in records:
        answer = record.get("answer", {}) or {}
        answer_text = str(answer.get("answer", "") or "")
        guardrails = record.get("guardrails", {}) or {}
        guardrail_states = {str(value).lower() for value in guardrails.values()}
        if "review" in guardrail_states or "fail" in guardrail_states:
            blocked += 1
        elif not record.get("results") and "No relevant evidence was found." in answer_text:
            no_results += 1
        if record.get("used_api"):
            used_api += 1
        confidence = str(answer.get("confidence", "") or "")
        if confidence:
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1

    row = {
        "run_label": str(first.get("run_label") or path.stem),
        "questions": len(records),
        "mode": settings.get("rag_mode", ""),
        "engine": settings.get("search_backend", ""),
        "reranker": settings.get("reranker_label", ""),
        "guardrail_level": settings.get("guardrail_level", ""),
        "blocked": blocked,
        "no_results": no_results,
        "used_api_count": used_api,
        "avg_retrieval_s": mean([lat.get("retrieval_s", 0.0) for lat in latencies]),
        "avg_answer_s": mean([lat.get("answer_s", 0.0) for lat in latencies]),
        "avg_total_s": mean([lat.get("total_s", 0.0) for lat in latencies]),
        "high_conf": confidence_counts.get("high", 0),
        "medium_conf": confidence_counts.get("medium", 0),
        "low_conf": confidence_counts.get("low", 0),
    }
    if retrieval_metrics:
        row["hit_at_k"] = mean([metric.get("hit_at_k", 0.0) for metric in retrieval_metrics])
        row["recall_at_k"] = mean([metric.get("recall_at_k", 0.0) for metric in retrieval_metrics])
        row["mrr_at_k"] = mean([metric.get("mrr_at_k", 0.0) for metric in retrieval_metrics])
        row["judged"] = len(retrieval_metrics)
    else:
        row["hit_at_k"] = ""
        row["recall_at_k"] = ""
        row["mrr_at_k"] = ""
        row["judged"] = 0
    if answer_metrics:
        row["normalized_exact_match"] = mean(
            [metric.get("normalized_exact_match", 0.0) for metric in answer_metrics]
        )
        row["contains_expected"] = mean(
            [metric.get("contains_expected", 0.0) for metric in answer_metrics]
        )
        row["token_f1"] = mean(
            [metric.get("token_f1", 0.0) for metric in answer_metrics]
        )
        row["anls"] = mean(
            [metric.get("anls", 0.0) for metric in answer_metrics]
        )
        row["answer_judged"] = len(answer_metrics)
    else:
        row["normalized_exact_match"] = ""
        row["contains_expected"] = ""
        row["token_f1"] = ""
        row["anls"] = ""
        row["answer_judged"] = 0
    return row


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "run_label",
        "questions",
        "mode",
        "engine",
        "blocked",
        "no_results",
        "hit_at_k",
        "recall_at_k",
        "mrr_at_k",
        "normalized_exact_match",
        "contains_expected",
        "token_f1",
        "anls",
        "avg_retrieval_s",
        "avg_answer_s",
        "avg_total_s",
    ]
    widths = {header: len(header) for header in headers}
    formatted_rows = []
    for row in rows:
        formatted = {
            "run_label": str(row["run_label"]),
            "questions": str(row["questions"]),
            "mode": str(row["mode"]),
            "engine": str(row["engine"]),
            "blocked": str(row["blocked"]),
            "no_results": str(row["no_results"]),
            "hit_at_k": fmt_metric(row["hit_at_k"]),
            "recall_at_k": fmt_metric(row["recall_at_k"]),
            "mrr_at_k": fmt_metric(row["mrr_at_k"]),
            "normalized_exact_match": fmt_metric(row["normalized_exact_match"]),
            "contains_expected": fmt_metric(row["contains_expected"]),
            "token_f1": fmt_metric(row["token_f1"]),
            "anls": fmt_metric(row["anls"]),
            "avg_retrieval_s": f"{row['avg_retrieval_s']:.3f}",
            "avg_answer_s": f"{row['avg_answer_s']:.3f}",
            "avg_total_s": f"{row['avg_total_s']:.3f}",
        }
        formatted_rows.append(formatted)
        for header, value in formatted.items():
            widths[header] = max(widths[header], len(value))

    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in formatted_rows:
        print(" | ".join(row[header].ljust(widths[header]) for header in headers))


def fmt_metric(value: Any) -> str:
    if value == "":
        return ""
    return f"{float(value):.3f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def slugify(value: str) -> str:
    value = value.strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


if __name__ == "__main__":
    main()
