from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

LOG_DIR = PROJECT_DIR / ".artifacts" / "eval_logs"
DEFAULT_MANIFEST = PROJECT_DIR / "eval" / "manifests" / "formal_full_system_v1.json"


def main() -> None:
    st.set_page_config(page_title="Formal Experiment Dashboard", page_icon="E", layout="wide")
    st.title("Formal Full-System Experiment")
    st.caption("Progress view for the formal MultiModalRAG evaluation suite.")

    with st.sidebar:
        st.header("Controls")
        manifest_path = st.text_input("Manifest", value=str(DEFAULT_MANIFEST))
        auto_refresh = st.toggle("Auto refresh", value=True)
        refresh_seconds = st.slider("Refresh every N seconds", min_value=5, max_value=60, value=10, step=5)
        if st.button("Refresh now"):
            st.rerun()

    if auto_refresh:
        st_autorefresh(refresh_seconds)

    manifest = load_manifest(Path(manifest_path))
    runs = manifest.get("runs") or []
    if not runs:
        st.error("No runs found in the manifest.")
        return

    run_labels = [str(run["run_label"]) for run in runs]
    rows = load_all_runs(run_labels)
    progress_df, summary_df, record_df = build_frames(rows, runs)

    runtime = manifest.get("expected_runtime_hours", {})
    costs = manifest.get("expected_openai_cost_usd", {})

    st.subheader("Plan")
    cols = st.columns(4)
    cols[0].metric("Planned runs", len(runs))
    cols[1].metric("Likely runtime", f"{runtime.get('likely', 'n/a')} h")
    cols[2].metric("Likely OpenAI cost", f"${costs.get('likely', 'n/a')}")
    cols[3].metric("Completed runs", int((progress_df["status"] == "completed").sum()))

    with st.expander("Cost and runtime assumptions", expanded=False):
        st.json(
            {
                "expected_runtime_hours": runtime,
                "expected_openai_cost_usd": costs,
                "notes": manifest.get("notes", ""),
            }
        )

    st.subheader("Run Status")
    st.dataframe(progress_df, width="stretch", hide_index=True)

    for row in progress_df.to_dict("records"):
        label = f"{row['run_label']} [{row['status']}]"
        st.write(label)
        st.progress(float(row["pct"]))

    if not summary_df.empty:
        st.subheader("Completed Run Metrics")
        st.dataframe(summary_df, width="stretch", hide_index=True)

        metric_cols = [
            col
            for col in [
                "hit_at_k",
                "recall_at_k",
                "mrr_at_k",
                "normalized_exact_match",
                "contains_expected",
                "token_f1",
                "anls",
            ]
            if col in summary_df.columns
        ]
        if metric_cols:
            st.bar_chart(summary_df.set_index("run_label")[metric_cols])

        latency_cols = [col for col in ["avg_retrieval_s", "avg_answer_s", "avg_total_s"] if col in summary_df.columns]
        if latency_cols:
            st.bar_chart(summary_df.set_index("run_label")[latency_cols])

    if not record_df.empty:
        st.subheader("Per-Question Trace")
        view_cols = [
            "run_label",
            "question_id",
            "question",
            "confidence",
            "result_count",
            "retrieval_s",
            "answer_s",
            "total_s",
            "hit_at_k",
            "recall_at_k",
            "mrr_at_k",
            "normalized_exact_match",
            "token_f1",
            "anls",
        ]
        present_cols = [col for col in view_cols if col in record_df.columns]
        st.dataframe(record_df[present_cols], width="stretch", hide_index=True)


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def st_autorefresh(seconds: int) -> None:
    st.markdown(
        f"""
        <script>
        setTimeout(function() {{
            window.location.reload();
        }}, {seconds * 1000});
        </script>
        """,
        unsafe_allow_html=True,
    )


def load_all_runs(run_labels: list[str]) -> list[dict[str, Any]]:
    if not LOG_DIR.exists():
        return []
    wanted = set(run_labels)
    rows: list[dict[str, Any]] = []
    for path in sorted(LOG_DIR.glob("*.jsonl")):
        if path.stem not in wanted:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["_run_file"] = path.stem
            rows.append(row)
    return rows


def build_frames(rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_order = [str(run["run_label"]) for run in runs]
    per_run: dict[str, list[dict[str, Any]]] = {label: [] for label in run_order}
    per_question: list[dict[str, Any]] = []

    for row in rows:
        run_label = str(row.get("run_label") or row.get("_run_file") or "")
        per_run.setdefault(run_label, []).append(row)
        answer = row.get("answer", {}) or {}
        retrieval_metrics = row.get("retrieval_metrics") or {}
        answer_metrics = row.get("answer_metrics") or {}
        progress = row.get("progress") or {}
        per_question.append(
            {
                "run_label": run_label,
                "question_id": row.get("question_id", ""),
                "question": row.get("question", ""),
                "confidence": answer.get("confidence", ""),
                "result_count": len(row.get("results") or []),
                "retrieval_s": float((row.get("latencies") or {}).get("retrieval_s", 0.0)),
                "answer_s": float((row.get("latencies") or {}).get("answer_s", 0.0)),
                "total_s": float((row.get("latencies") or {}).get("total_s", 0.0)),
                "hit_at_k": retrieval_metrics.get("hit_at_k", ""),
                "recall_at_k": retrieval_metrics.get("recall_at_k", ""),
                "mrr_at_k": retrieval_metrics.get("mrr_at_k", ""),
                "normalized_exact_match": answer_metrics.get("normalized_exact_match", ""),
                "token_f1": answer_metrics.get("token_f1", ""),
                "anls": answer_metrics.get("anls", ""),
                "completed": progress.get("completed", 0),
                "expected": progress.get("expected", 0),
            }
        )

    progress_rows = []
    summary_rows = []
    for run in runs:
        run_label = str(run["run_label"])
        phase = str(run.get("phase") or "")
        records = per_run.get(run_label, [])
        if not records:
            progress_rows.append(
                {
                    "phase": phase,
                    "run_label": run_label,
                    "status": "not_started",
                    "completed": 0,
                    "expected": 0,
                    "pct": 0.0,
                }
            )
            continue

        latest = records[-1]
        progress = latest.get("progress") or {"completed": len(records), "expected": len(records), "pct": 1.0}
        completed = int(progress.get("completed", len(records)))
        expected = int(progress.get("expected", len(records)))
        pct = float(progress.get("pct", 1.0 if expected and completed >= expected else 0.0))
        status = "completed" if expected and completed >= expected else "running"
        progress_rows.append(
            {
                "phase": phase,
                "run_label": run_label,
                "status": status,
                "completed": completed,
                "expected": expected,
                "pct": pct,
            }
        )

        latencies = [record.get("latencies", {}) for record in records]
        retrieval_metrics = [record.get("retrieval_metrics") for record in records if record.get("retrieval_metrics")]
        answer_metrics = [record.get("answer_metrics") for record in records if record.get("answer_metrics")]
        settings = latest.get("settings", {}) or {}
        summary_rows.append(
            {
                "phase": phase,
                "run_label": run_label,
                "mode": settings.get("rag_mode", ""),
                "engine": settings.get("search_backend", ""),
                "avg_retrieval_s": mean([lat.get("retrieval_s", 0.0) for lat in latencies]),
                "avg_answer_s": mean([lat.get("answer_s", 0.0) for lat in latencies]),
                "avg_total_s": mean([lat.get("total_s", 0.0) for lat in latencies]),
                "hit_at_k": mean([metric.get("hit_at_k", 0.0) for metric in retrieval_metrics]) if retrieval_metrics else 0.0,
                "recall_at_k": mean([metric.get("recall_at_k", 0.0) for metric in retrieval_metrics]) if retrieval_metrics else 0.0,
                "mrr_at_k": mean([metric.get("mrr_at_k", 0.0) for metric in retrieval_metrics]) if retrieval_metrics else 0.0,
                "normalized_exact_match": mean([metric.get("normalized_exact_match", 0.0) for metric in answer_metrics]) if answer_metrics else 0.0,
                "contains_expected": mean([metric.get("contains_expected", 0.0) for metric in answer_metrics]) if answer_metrics else 0.0,
                "token_f1": mean([metric.get("token_f1", 0.0) for metric in answer_metrics]) if answer_metrics else 0.0,
                "anls": mean([metric.get("anls", 0.0) for metric in answer_metrics]) if answer_metrics else 0.0,
            }
        )

    progress_df = pd.DataFrame(progress_rows)
    summary_df = pd.DataFrame(summary_rows)
    record_df = pd.DataFrame(per_question)
    return progress_df, summary_df, record_df


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


if __name__ == "__main__":
    main()
