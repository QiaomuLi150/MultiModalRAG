from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class EvalPreset:
    name: str
    description: str
    settings: dict[str, Any]


EVAL_PRESETS = [
    EvalPreset(
        name="Custom",
        description="Use the current sidebar settings exactly as shown.",
        settings={},
    ),
    EvalPreset(
        name="Text Baseline",
        description="Cheap text-only baseline for retrieval comparisons.",
        settings={
            "selected_mode_label": "Text Hybrid",
            "search_backend": "Hybrid",
            "reranker_label": "Off",
            "top_k": 5,
            "token_budget": 3000,
            "guardrail_level": "Balanced",
            "force_retrieval_only_eval": True,
        },
    ),
    EvalPreset(
        name="Hybrid Default",
        description="Recommended practical baseline for mixed enterprise content.",
        settings={
            "selected_mode_label": "Visual Page + Text Hybrid",
            "search_backend": "Qdrant Cloud hybrid",
            "reranker_label": "MiniLM-L4",
            "rerank_top_n": 8,
            "top_k": 5,
            "token_budget": 3000,
            "page_render_scale": 1.25,
            "visual_embedding_batch_size": 1,
            "light_compression_mode": "similarity_merge",
            "light_target_tokens": 64,
            "guardrail_level": "Balanced",
            "force_retrieval_only_eval": False,
        },
    ),
    EvalPreset(
        name="Visual Stress",
        description="Visual-heavy setting for page retrieval and memory/latency checks.",
        settings={
            "selected_mode_label": "Visual Page",
            "search_backend": "Qdrant Cloud",
            "reranker_label": "Off",
            "top_k": 6,
            "token_budget": 2500,
            "page_render_scale": 1.25,
            "visual_embedding_batch_size": 1,
            "light_compression_mode": "similarity_merge",
            "light_target_tokens": 64,
            "guardrail_level": "Relaxed",
            "force_retrieval_only_eval": True,
        },
    ),
]


def preset_names() -> list[str]:
    return [preset.name for preset in EVAL_PRESETS]


def preset_by_name(name: str) -> EvalPreset:
    for preset in EVAL_PRESETS:
        if preset.name == name:
            return preset
    return EVAL_PRESETS[0]


def apply_preset(name: str, values: dict[str, Any]) -> dict[str, Any]:
    preset = preset_by_name(name)
    merged = dict(values)
    merged.update(preset.settings)
    return merged


def write_eval_run(
    root: Path,
    *,
    run_label: str,
    record: dict[str, Any],
) -> dict[str, str]:
    slug = _slugify(run_label) or "default"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = root / ".artifacts" / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = log_dir / f"{slug}.jsonl"
    summary_csv_path = log_dir / f"{slug}_summary.csv"
    detail_json_path = log_dir / f"{slug}_{timestamp}.json"

    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    with detail_json_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=True, indent=2)

    _append_summary_row(summary_csv_path, record)
    return {
        "jsonl_path": str(jsonl_path),
        "summary_csv_path": str(summary_csv_path),
        "detail_json_path": str(detail_json_path),
    }


def build_eval_record(
    *,
    run_label: str,
    question: str,
    effective_question: str,
    settings: dict[str, Any],
    answer_payload: dict[str, Any],
    results: list[dict[str, Any]],
    latencies: dict[str, float],
    stepback_question: str,
    used_api: bool,
    guardrail_details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "question": question,
        "effective_question": effective_question,
        "stepback_question": stepback_question,
        "used_api": used_api,
        "settings": settings,
        "latencies": latencies,
        "answer": answer_payload,
        "results": results,
        "guardrails": {name: detail.get("status") for name, detail in guardrail_details.items()},
    }


def _append_summary_row(path: Path, record: dict[str, Any]) -> None:
    latencies = record.get("latencies", {})
    answer = record.get("answer", {})
    results = record.get("results", [])
    row = {
        "timestamp_utc": record.get("timestamp_utc", ""),
        "run_label": record.get("run_label", ""),
        "question": record.get("question", ""),
        "effective_question": record.get("effective_question", ""),
        "mode": record.get("settings", {}).get("rag_mode", ""),
        "engine": record.get("settings", {}).get("search_backend", ""),
        "reranker": record.get("settings", {}).get("reranker_label", ""),
        "guardrail_level": record.get("settings", {}).get("guardrail_level", ""),
        "used_api": record.get("used_api", False),
        "result_count": len(results),
        "top_chunk_id": results[0].get("chunk_id", "") if results else "",
        "top_score": results[0].get("score", 0.0) if results else 0.0,
        "confidence": answer.get("confidence", ""),
        "retrieval_s": latencies.get("retrieval_s", 0.0),
        "stepback_s": latencies.get("stepback_s", 0.0),
        "answer_s": latencies.get("answer_s", 0.0),
        "total_s": latencies.get("total_s", 0.0),
    }
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")
