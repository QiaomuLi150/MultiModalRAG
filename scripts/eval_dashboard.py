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


def main() -> None:
    st.set_page_config(page_title="Eval Dashboard", page_icon="E", layout="wide")
    st.title("Experiment Dashboard")
    st.caption("Live view over MultiModalRAG evaluation runs.")

    with st.sidebar:
        st.header("Controls")
        auto_refresh = st.toggle("Auto refresh", value=False)
        refresh_seconds = st.slider("Refresh every N seconds", min_value=5, max_value=60, value=10, step=5)
        selected_runs = st.multiselect("Run labels", options=available_run_labels(), default=[])
        if st.button("Refresh now"):
            st.rerun()

    if auto_refresh:
        st.caption(f"Auto refresh enabled: every {refresh_seconds}s")
        st_autorefresh(refresh_seconds)

    rows = load_all_runs(selected_runs)
    if not rows:
        st.warning("No eval JSONL logs found yet. Run experiments with --log-runs first.")
        return

    summary_df, progress_df, record_df = build_frames(rows)

    st.subheader("Run Overview")
    top_cols = st.columns(4)
    top_cols[0].metric("Runs", len(summary_df))
    top_cols[1].metric("Questions logged", int(summary_df["questions"].sum()))
    top_cols[2].metric("Avg retrieval (s)", f"{summary_df['avg_retrieval_s'].mean():.3f}")
    top_cols[3].metric("Avg total (s)", f"{summary_df['avg_total_s'].mean():.3f}")

    st.dataframe(summary_df, width="stretch", hide_index=True)

    st.subheader("Progress")
    for row in progress_df.to_dict("records"):
        label = f"{row['run_label']} ({row['completed']}/{row['expected']})"
        st.write(label)
        st.progress(float(row["pct"]))

    st.subheader("Metric Comparison")
    metric_view = summary_df[["run_label", "hit_at_k", "recall_at_k", "mrr_at_k"]].copy()
    metric_view = metric_view.set_index("run_label")
    st.bar_chart(metric_view)

    st.subheader("Latency Comparison")
    latency_view = summary_df[["run_label", "avg_retrieval_s", "avg_answer_s", "avg_total_s"]].copy()
    latency_view = latency_view.set_index("run_label")
    st.bar_chart(latency_view)

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
    ]
    present_cols = [col for col in view_cols if col in record_df.columns]
    st.dataframe(record_df[present_cols], width="stretch", hide_index=True)


def available_run_labels() -> list[str]:
    if not LOG_DIR.exists():
        return []
    return sorted(path.stem for path in LOG_DIR.glob("*.jsonl"))


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


def load_all_runs(selected_runs: list[str]) -> list[dict[str, Any]]:
    if not LOG_DIR.exists():
        return []
    files = sorted(LOG_DIR.glob("*.jsonl"))
    if selected_runs:
        wanted = set(selected_runs)
        files = [path for path in files if path.stem in wanted]
    rows: list[dict[str, Any]] = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["_run_file"] = path.stem
            rows.append(row)
    return rows


def build_frames(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_question = []
    per_run: dict[str, list[dict[str, Any]]] = {}
    progress_rows = []

    for row in rows:
        run_label = str(row.get("run_label") or row.get("_run_file") or "")
        per_run.setdefault(run_label, []).append(row)
        answer = row.get("answer", {}) or {}
        metrics = row.get("retrieval_metrics") or {}
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
                "hit_at_k": metrics.get("hit_at_k", ""),
                "recall_at_k": metrics.get("recall_at_k", ""),
                "mrr_at_k": metrics.get("mrr_at_k", ""),
                "completed": progress.get("completed", 0),
                "expected": progress.get("expected", 0),
            }
        )

    summary_rows = []
    for run_label, records in sorted(per_run.items()):
        latest = records[-1]
        latencies = [record.get("latencies", {}) for record in records]
        retrieval_metrics = [record.get("retrieval_metrics") for record in records if record.get("retrieval_metrics")]
        blocked = 0
        no_results = 0
        for record in records:
            answer = record.get("answer", {}) or {}
            answer_text = str(answer.get("answer", "") or "")
            if not record.get("results"):
                if "No relevant evidence was found." in answer_text:
                    no_results += 1
                elif "could not answer" in answer_text.lower() or "safely" in answer_text.lower():
                    blocked += 1

        progress = latest.get("progress") or {"completed": len(records), "expected": len(records), "pct": 1.0}
        progress_rows.append(
            {
                "run_label": run_label,
                "completed": int(progress.get("completed", len(records))),
                "expected": int(progress.get("expected", len(records))),
                "pct": float(progress.get("pct", 0.0)),
            }
        )

        summary_rows.append(
            {
                "run_label": run_label,
                "questions": len(records),
                "mode": ((latest.get("settings") or {}).get("rag_mode", "")),
                "engine": ((latest.get("settings") or {}).get("search_backend", "")),
                "blocked": blocked,
                "no_results": no_results,
                "avg_retrieval_s": mean([lat.get("retrieval_s", 0.0) for lat in latencies]),
                "avg_answer_s": mean([lat.get("answer_s", 0.0) for lat in latencies]),
                "avg_total_s": mean([lat.get("total_s", 0.0) for lat in latencies]),
                "hit_at_k": mean([metric.get("hit_at_k", 0.0) for metric in retrieval_metrics]) if retrieval_metrics else 0.0,
                "recall_at_k": mean([metric.get("recall_at_k", 0.0) for metric in retrieval_metrics]) if retrieval_metrics else 0.0,
                "mrr_at_k": mean([metric.get("mrr_at_k", 0.0) for metric in retrieval_metrics]) if retrieval_metrics else 0.0,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(progress_rows), pd.DataFrame(per_question)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


if __name__ == "__main__":
    main()
