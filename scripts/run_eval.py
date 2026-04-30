from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
from collections import Counter
import re
import sys
import time
import tomllib
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.audio import DEFAULT_ASR_MODEL, transcribe_audio_bytes, transcribe_video_audio_bytes
from src.chunking import DocumentChunk, add_parent_metadata, assign_chunk_ids, text_preview
from src.docling_adapter import chunks_from_pdf_docling_bytes
from src.eval_support import build_eval_record, preset_by_name, write_eval_run
from src.generation import MODEL_NAME, generate_answer, generate_stepback_question, get_api_key
from src.guardrails.config import config_for_level
from src.guardrails.pipeline import GuardrailPipeline
from src.ingest import (
    chunk_from_image_bytes,
    chunk_from_video_frame,
    chunks_from_csv_bytes,
    chunks_from_pdf_bytes,
    chunks_from_text_bytes,
    load_sample_chunks,
)
from src.rag_modes import RagModeSpec, build_mode_index, mode_by_label
from src.render import ingest_image_bytes, ingest_pil_image, render_pdf_bytes
from src.rerankers import reranker_by_label
from src.schemas import PageRecord
from src.vision import sample_video_frames

APP_DIR = PROJECT_DIR
SAMPLE_DIR = APP_DIR / "sample_data"
PAGE_IMAGE_DIR = APP_DIR / ".artifacts" / "page_images"


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    questions = load_questions(Path(args.questions_file), limit=args.limit)
    qrels = load_qrels(args.qrels_file, Path(args.questions_file))
    if not questions:
        raise SystemExit("No questions found in the supplied questions file.")

    print("Configuration check")
    print(f"- preset: {config['eval_preset']}")
    print(f"- rag_mode: {config['rag_mode_label']}")
    print(f"- search_backend: {config['search_backend']}")
    print(f"- retrieval_only: {config['retrieval_only']}")
    print(f"- question_count: {len(questions)}")
    print(f"- qrels_loaded: {'yes' if qrels else 'no'}")

    chunks, page_records = load_corpus(config)
    index = build_index(config, chunks, page_records)
    run_eval(index=index, chunks=chunks, config=config, questions=questions, qrels=qrels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeatable evaluations against the current MultiModalRAG stack.",
    )
    parser.add_argument("--questions-file", required=True, help="JSONL/CSV/TXT file with evaluation questions.")
    parser.add_argument("--eval-preset", default="Custom", help="Evaluation preset name.")
    parser.add_argument("--run-label", default="manual_eval", help="Label used for JSONL/CSV logs.")
    parser.add_argument("--input-dir", help="Optional corpus directory to ingest before evaluation.")
    parser.add_argument("--qrels-file", help="Optional JSONL qrels file aligned to the questions file.")
    parser.add_argument("--use-sample-data", action="store_true", help="Use bundled sample data.")
    parser.add_argument("--search-backend", help="Search backend, e.g. Hybrid or Qdrant Cloud hybrid.")
    parser.add_argument("--rag-mode", dest="rag_mode_label", help="RAG mode label from the Streamlit app.")
    parser.add_argument("--top-k", type=int, help="Top-k retrieval depth.")
    parser.add_argument("--token-budget", type=int, help="Generation context budget.")
    parser.add_argument("--guardrail-level", choices=["Off", "Relaxed", "Balanced", "Strict"], help="Guardrail level.")
    parser.add_argument("--reranker", dest="reranker_label", help="Reranker label, e.g. Off or MiniLM-L4.")
    parser.add_argument("--rerank-top-n", type=int, help="Second-stage rerank candidate count.")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip answer generation.")
    parser.add_argument("--score-answers", action="store_true", help="Compute answer metrics when expected answers are available.")
    parser.add_argument("--log-runs", action="store_true", help="Write JSONL/CSV logs under .artifacts/eval_logs.")
    parser.add_argument("--provider", default=f"OpenAI {MODEL_NAME}", help="Model provider label.")
    parser.add_argument("--openai-api-key", help="Optional API key for generation.")
    parser.add_argument("--qdrant-cloud-url", help="Qdrant Cloud URL.")
    parser.add_argument("--qdrant-cloud-api-key", help="Qdrant Cloud API key.")
    parser.add_argument("--qdrant-collection", help="Qdrant Cloud collection name.")
    parser.add_argument("--page-render-scale", type=float, help="PDF render scale for visual modes.")
    parser.add_argument("--visual-model-name", default="vidore/colqwen2-v1.0-hf", help="Visual retriever model.")
    parser.add_argument("--visual-local-files-only", action="store_true", default=True, help="Use local visual model cache only.")
    parser.add_argument("--enable-openai-visual-descriptions", action="store_true", help="Allow image/video ingestion to call GPT-5-nano for VLM descriptions during eval corpus loading.")
    parser.add_argument("--visual-embedding-batch-size", type=int, help="Visual embedding batch size.")
    parser.add_argument("--light-compression-mode", choices=["none", "mean_pool", "similarity_merge"], help="Visual multivector compression mode.")
    parser.add_argument("--light-target-tokens", type=int, help="Target vectors per page after compression.")
    parser.add_argument("--enable-page-ocr", action="store_true", default=True, help="OCR rendered pages/images.")
    parser.add_argument("--enable-video-frames", action="store_true", help="Sample video frames during ingestion.")
    parser.add_argument("--max-video-segments", type=int, default=5, help="Maximum sampled video frames.")
    parser.add_argument("--transcribe-video-audio", action="store_true", help="Extract and transcribe video audio.")
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL, help="faster-whisper model size.")
    parser.add_argument("--pdf-extractor", default="pypdf", choices=["pypdf", "Docling balanced", "Docling strict", "Docling broad"], help="PDF extractor.")
    parser.add_argument("--use-muvera-proxy", action="store_true", help="Enable MUVERA-style proxy retrieval for visual modes.")
    parser.add_argument("--muvera-candidate-count", type=int, default=12, help="Proxy candidate pool size.")
    parser.add_argument("--limit", type=int, help="Limit number of evaluation questions.")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    preset = preset_by_name(args.eval_preset)
    config: dict[str, Any] = {
        "eval_preset": preset.name,
        "run_label": args.run_label,
        "provider": args.provider,
        "rag_mode_label": "Text Hybrid",
        "search_backend": "Hybrid",
        "top_k": 5,
        "token_budget": 3000,
        "guardrail_level": "Off",
        "reranker_label": "Off",
        "rerank_top_n": 8,
        "page_render_scale": 1.5,
        "visual_embedding_batch_size": 1,
        "light_compression_mode": "none",
        "light_target_tokens": None,
        "retrieval_only": False,
        "provider_api_key": args.openai_api_key or "",
        "input_dir": args.input_dir,
        "use_sample_data": args.use_sample_data,
        "visual_model_name": args.visual_model_name,
        "visual_local_files_only": bool(args.visual_local_files_only),
        "enable_openai_visual_descriptions": bool(args.enable_openai_visual_descriptions),
        "enable_page_ocr": bool(args.enable_page_ocr),
        "enable_video_frames": bool(args.enable_video_frames),
        "max_video_segments": max(1, args.max_video_segments),
        "transcribe_video_audio": bool(args.transcribe_video_audio),
        "asr_model": args.asr_model,
        "pdf_extractor": args.pdf_extractor,
        "use_muvera_proxy": bool(args.use_muvera_proxy),
        "muvera_candidate_count": max(1, args.muvera_candidate_count),
        "log_runs": bool(args.log_runs),
        "qdrant_cloud_url": args.qdrant_cloud_url or os.environ.get("QDRANT_CLOUD_URL", ""),
        "qdrant_cloud_api_key": args.qdrant_cloud_api_key or os.environ.get("QDRANT_CLOUD_API_KEY", ""),
        "qdrant_collection": args.qdrant_collection or os.environ.get("QDRANT_COLLECTION", "multimodal_chunks"),
    }
    config.update(preset.settings)

    explicit_overrides = {
        "rag_mode_label": args.rag_mode_label,
        "search_backend": args.search_backend,
        "top_k": args.top_k,
        "token_budget": args.token_budget,
        "guardrail_level": args.guardrail_level,
        "reranker_label": args.reranker_label,
        "rerank_top_n": args.rerank_top_n,
        "page_render_scale": args.page_render_scale,
        "visual_embedding_batch_size": args.visual_embedding_batch_size,
        "light_compression_mode": args.light_compression_mode,
        "light_target_tokens": args.light_target_tokens,
    }
    for key, value in explicit_overrides.items():
        if value is not None:
            config[key] = value
    if args.retrieval_only:
        config["retrieval_only"] = True
    if preset.settings.get("force_retrieval_only_eval"):
        config["retrieval_only"] = True

    config["rag_mode"] = mode_by_label(str(config["rag_mode_label"]))
    config["reranker_spec"] = reranker_by_label(str(config["reranker_label"]))
    config["qdrant_cloud_config"] = qdrant_cloud_config(
        search_backend=str(config["search_backend"]),
        url=str(config["qdrant_cloud_url"]),
        api_key=str(config["qdrant_cloud_api_key"]),
        collection_name=str(config["qdrant_collection"]),
    )
    config["api_key"] = resolve_openai_api_key(user_api_key=str(config["provider_api_key"]))
    if config["retrieval_only"]:
        config["api_key"] = None
    config["score_answers"] = bool(args.score_answers or (not config["retrieval_only"]))
    return config


def load_questions(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = [{"question": line.strip()} for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    normalized = []
    for index, row in enumerate(rows, start=1):
        question = str(row.get("question") or row.get("query") or row.get("prompt") or "").strip()
        if not question:
            continue
        normalized.append(
            {
                "id": row.get("id") or f"q{index:03d}",
                "question": question,
                "metadata": {key: value for key, value in row.items() if key not in {"id", "question", "query", "prompt"}},
            }
        )
    if limit is not None:
        return normalized[: max(0, limit)]
    return normalized


def load_qrels(qrels_file: str | None, questions_path: Path) -> dict[str, list[dict[str, Any]]]:
    candidate = Path(qrels_file) if qrels_file else questions_path.with_name(f"{questions_path.stem}_qrels.jsonl")
    if not candidate.exists():
        return {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for line in candidate.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row.get("id") or "")] = list(row.get("qrels") or [])
    return rows


def load_corpus(config: dict[str, Any]) -> tuple[list[DocumentChunk], list[PageRecord]]:
    if config["use_sample_data"]:
        print("- loading bundled sample data")
        chunks = add_parent_metadata(assign_chunk_ids(load_sample_chunks(SAMPLE_DIR)))
        return chunks, []

    input_dir = config.get("input_dir")
    if not input_dir:
        return [], []

    root = Path(str(input_dir))
    if not root.exists():
        raise SystemExit(f"Input directory does not exist: {root}")

    chunks: list[DocumentChunk] = []
    page_records: list[PageRecord] = []
    visual_description_api_key = config["api_key"] if config["enable_openai_visual_descriptions"] else None
    for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
        suffix = file_path.suffix.lower()
        data = file_path.read_bytes()
        name = str(file_path.relative_to(root))
        if suffix in {".txt", ".md"}:
            chunks.extend(chunks_from_text_bytes(data, name, modality="text"))
        elif suffix == ".csv":
            if is_eval_corpus_csv(data):
                chunks.extend(chunks_from_eval_corpus_csv(data, name))
            else:
                chunks.extend(chunks_from_csv_bytes(data, name))
        elif suffix == ".pdf":
            chunks.extend(extract_pdf_chunks(data, name, config["pdf_extractor"]))
            if config["rag_mode"].use_visual_backend:
                page_records.extend(
                    render_pdf_bytes(
                        data,
                        name,
                        PAGE_IMAGE_DIR,
                        render_scale=float(config["page_render_scale"]),
                        include_ocr=bool(config["enable_page_ocr"]),
                    )
                )
        elif suffix in {".wav", ".mp3", ".m4a"}:
            audio_chunks, _status = transcribe_audio_bytes(data, name, model_size=str(config["asr_model"]))
            chunks.extend(audio_chunks)
        elif suffix in {".png", ".jpg", ".jpeg"}:
            chunks.append(chunk_from_image_bytes(data, name, description="", openai_api_key=visual_description_api_key))
            if config["rag_mode"].use_visual_backend:
                page_records.append(
                    ingest_image_bytes(
                        data,
                        name,
                        PAGE_IMAGE_DIR,
                        include_ocr=bool(config["enable_page_ocr"]),
                    )
                )
        elif suffix in {".mp4", ".mov"}:
            if config["transcribe_video_audio"]:
                audio_chunks, _status = transcribe_video_audio_bytes(data, name, model_size=str(config["asr_model"]))
                chunks.extend(audio_chunks)
            if config["enable_video_frames"]:
                frames = sample_video_frames(
                    data,
                    max_frames=int(config["max_video_segments"]),
                    source_name=name,
                )
                for frame in frames:
                    chunks.append(
                        chunk_from_video_frame(
                            name,
                            frame.image,
                            f"{frame.segment_start}-{frame.segment_end}",
                            frame.frame,
                            "",
                            openai_api_key=visual_description_api_key,
                        )
                    )
                    if config["rag_mode"].use_visual_backend:
                        page_records.append(
                            ingest_pil_image(
                                frame.image,
                                f"{name}_{frame.frame}",
                                PAGE_IMAGE_DIR,
                                include_ocr=bool(config["enable_page_ocr"]),
                                page_num=int(frame.frame) if str(frame.frame).isdigit() else len(page_records) + 1,
                                source_type="video_frame",
                                extra_metadata={"timestamp": frame.timestamp, "frame_number": frame.frame},
                            )
                        )
    if chunks:
        chunks = add_parent_metadata(assign_chunk_ids(chunks))
        chunks = restore_eval_doc_chunk_ids(chunks)
    return chunks, page_records


def is_eval_corpus_csv(data: bytes) -> bool:
    sample = data.decode("utf-8", errors="replace").splitlines()
    if not sample:
        return False
    header = sample[0].strip().lower()
    return header == "doc_id,title,text" or header.startswith("doc_id,")


def chunks_from_eval_corpus_csv(data: bytes, source_name: str) -> list[DocumentChunk]:
    rows = list(csv.DictReader(data.decode("utf-8", errors="replace").splitlines()))
    chunks: list[DocumentChunk] = []
    for row in rows:
        doc_id = str(row.get("doc_id") or "").strip()
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()
        combined = f"{title}\n\n{text}".strip() if title else text
        if not doc_id or not combined:
            continue
        chunks.append(
            DocumentChunk(
                chunk_id=doc_id,
                text=combined,
                source_name=source_name,
                modality="text",
                metadata={
                    "eval_doc_id": doc_id,
                    "converter": "eval_corpus_csv",
                    "chunker": "single_document_row",
                },
            )
        )
    return chunks


def restore_eval_doc_chunk_ids(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    restored = []
    for chunk in chunks:
        eval_doc_id = chunk.metadata.get("eval_doc_id")
        if eval_doc_id:
            restored.append(replace(chunk, chunk_id=str(eval_doc_id)))
        else:
            restored.append(chunk)
    return restored


def extract_pdf_chunks(data: bytes, name: str, extractor: str) -> list[DocumentChunk]:
    if extractor.startswith("Docling"):
        mode = extractor.split()[-1]
        docling_chunks, _status = chunks_from_pdf_docling_bytes(data, name, mode=mode)
        if docling_chunks:
            return docling_chunks
    return chunks_from_pdf_bytes(data, name)


def build_index(config: dict[str, Any], chunks: list[DocumentChunk], page_records: list[PageRecord]):
    rag_mode: RagModeSpec = config["rag_mode"]
    search_backend = str(config["search_backend"])
    if not chunks and not search_backend.startswith("Qdrant Cloud"):
        raise SystemExit("No local corpus was loaded. Supply --input-dir, --use-sample-data, or use a Qdrant Cloud backend.")
    if rag_mode.use_visual_backend and not page_records and not search_backend.startswith("Qdrant Cloud"):
        raise SystemExit("Visual mode selected, but no rendered page/image records were built from the local corpus.")

    print("- building index")
    return build_mode_index(
        chunks,
        rag_mode,
        search_backend,
        qdrant_cloud_config=config["qdrant_cloud_config"],
        page_records=page_records,
        visual_model_name=str(config["visual_model_name"]),
        visual_local_files_only=bool(config["visual_local_files_only"]),
        visual_embedding_batch_size=int(config["visual_embedding_batch_size"]),
        light_compression_mode=str(config["light_compression_mode"]),
        light_target_tokens=config["light_target_tokens"],
        use_muvera_proxy=bool(config["use_muvera_proxy"]),
        muvera_candidate_count=int(config["muvera_candidate_count"]),
        reranker_spec=config["reranker_spec"],
        rerank_top_n=int(config["rerank_top_n"]),
    )


def run_eval(*, index, chunks: list[DocumentChunk], config: dict[str, Any], questions: list[dict[str, Any]], qrels: dict[str, list[dict[str, Any]]]) -> None:
    guardrails = None
    if str(config["guardrail_level"]) != "Off":
        guardrails = GuardrailPipeline(config_for_level(str(config["guardrail_level"])))

    total_count = 0
    blocked_count = 0
    no_result_count = 0
    answer_count = 0
    total_retrieval_s = 0.0
    total_answer_s = 0.0
    metric_accumulator = {"judged": 0, "hit_at_k": 0.0, "recall_at_k": 0.0, "mrr_at_k": 0.0}
    answer_metric_accumulator = {
        "judged": 0,
        "normalized_exact_match": 0.0,
        "contains_expected": 0.0,
        "token_f1": 0.0,
        "anls": 0.0,
    }

    for row in questions:
        total_count += 1
        question = str(row["question"])
        question_id = str(row["id"])
        guardrail_details: dict[str, Any] = {}
        effective_question = question
        stepback_question = ""
        stepback_seconds = 0.0
        retrieval_seconds = 0.0
        answer_seconds = 0.0
        total_start = time.perf_counter()

        if guardrails is not None:
            pre = guardrails.run_pre_retrieval(question)
            guardrail_details.update(pre.get("results", {}))
            effective_question = pre.get("query", question)
            if pre["action"] != "continue":
                blocked_count += 1
                record = build_eval_record(
                    run_label=str(config["run_label"]),
                    question=question,
                    effective_question=effective_question,
                    settings=eval_settings_snapshot(config),
                    answer_payload=pre["response"],
                    results=[],
                    latencies={"retrieval_s": 0.0, "stepback_s": 0.0, "answer_s": 0.0, "total_s": 0.0},
                    stepback_question="",
                    used_api=False,
                    guardrail_details=guardrail_details,
                )
                record["question_id"] = question_id
                record["question_metadata"] = row.get("metadata", {})
                attach_progress(record, completed=total_count, expected=len(questions))
                maybe_write_eval_record(config, record)
                print(f"[{question_id}] blocked_by_pre_guardrail")
                continue

        retrieval_start = time.perf_counter()
        results = index.search(effective_question, top_k=int(config["top_k"]))
        retrieval_seconds = time.perf_counter() - retrieval_start
        total_retrieval_s += retrieval_seconds
        result_records = result_records_from_search(results)
        retrieval_metrics = metrics_for_query(
            row_id=question_id,
            results=results,
            qrels=qrels,
            top_k=int(config["top_k"]),
        )
        if retrieval_metrics is not None:
            metric_accumulator["judged"] += 1
            metric_accumulator["hit_at_k"] += retrieval_metrics["hit_at_k"]
            metric_accumulator["recall_at_k"] += retrieval_metrics["recall_at_k"]
            metric_accumulator["mrr_at_k"] += retrieval_metrics["mrr_at_k"]

        if guardrails is not None:
            post_retrieval = guardrails.run_post_retrieval(result_records)
            guardrail_details.update(post_retrieval.get("results", {}))
            if post_retrieval["action"] != "continue":
                blocked_count += 1
                record = build_eval_record(
                    run_label=str(config["run_label"]),
                    question=question,
                    effective_question=effective_question,
                    settings=eval_settings_snapshot(config),
                    answer_payload=post_retrieval["response"],
                    results=result_records,
                    latencies={
                        "retrieval_s": retrieval_seconds,
                        "stepback_s": 0.0,
                        "answer_s": 0.0,
                        "total_s": time.perf_counter() - total_start,
                    },
                    stepback_question="",
                    used_api=False,
                    guardrail_details=guardrail_details,
                )
                record["question_id"] = question_id
                record["question_metadata"] = row.get("metadata", {})
                record["retrieval_metrics"] = retrieval_metrics
                attach_progress(record, completed=total_count, expected=len(questions))
                maybe_write_eval_record(config, record)
                print(f"[{question_id}] blocked_by_post_retrieval_guardrail top_results={len(results)}")
                continue

        if config["rag_mode"].use_step_back and config["api_key"]:
            stepback_start = time.perf_counter()
            stepback_question = generate_stepback_question(effective_question, str(config["api_key"]))
            stepback_results = index.search(stepback_question, top_k=int(config["top_k"]))
            stepback_seconds = time.perf_counter() - stepback_start
            results = merge_results(results, stepback_results, int(config["top_k"]))

        if not results:
            no_result_count += 1
            record = build_eval_record(
                run_label=str(config["run_label"]),
                question=question,
                effective_question=effective_question,
                settings=eval_settings_snapshot(config),
                answer_payload={
                    "answer": "No relevant evidence was found.",
                    "sources": [],
                    "confidence": "low",
                    "guardrail_summary": {},
                },
                results=[],
                latencies={
                    "retrieval_s": retrieval_seconds,
                    "stepback_s": stepback_seconds,
                    "answer_s": 0.0,
                    "total_s": time.perf_counter() - total_start,
                },
                stepback_question=stepback_question,
                used_api=False,
                guardrail_details=guardrail_details,
            )
            record["question_id"] = question_id
            record["question_metadata"] = row.get("metadata", {})
            record["retrieval_metrics"] = retrieval_metrics
            attach_progress(record, completed=total_count, expected=len(questions))
            maybe_write_eval_record(config, record)
            print(f"[{question_id}] no_results")
            continue

        chunks_with_scores = [(result.chunk, result.score) for result in results]
        final_answer_text = ""
        if config["api_key"]:
            answer_start = time.perf_counter()
            final_answer_text = generate_answer(
                effective_question,
                chunks_with_scores,
                str(config["api_key"]),
                token_budget=int(config["token_budget"]),
            )
            answer_seconds = time.perf_counter() - answer_start
            total_answer_s += answer_seconds
        if not final_answer_text:
            final_answer_text = evidence_only_answer(results)

        answer_payload = build_answer_payload(final_answer_text, results)
        if guardrails is not None:
            post_generation = guardrails.run_post_generation(
                answer_payload=answer_payload,
                evidence=[result.chunk.text for result in results],
            )
            guardrail_details.update(post_generation.get("results", {}))
            answer_payload = post_generation["response"]

        answer_metrics = score_answer_for_row(
            answer_text=str(answer_payload.get("answer") or ""),
            question_metadata=row.get("metadata", {}),
            enabled=bool(config["score_answers"]),
        )
        if answer_metrics is not None:
            answer_metric_accumulator["judged"] += 1
            answer_metric_accumulator["normalized_exact_match"] += answer_metrics["normalized_exact_match"]
            answer_metric_accumulator["contains_expected"] += answer_metrics["contains_expected"]
            answer_metric_accumulator["token_f1"] += answer_metrics["token_f1"]
            answer_metric_accumulator["anls"] += answer_metrics["anls"]

        answer_count += 1
        record = build_eval_record(
            run_label=str(config["run_label"]),
            question=question,
            effective_question=effective_question,
            settings=eval_settings_snapshot(config),
            answer_payload=answer_payload,
            results=result_records_from_search(results),
            latencies={
                "retrieval_s": retrieval_seconds,
                "stepback_s": stepback_seconds,
                "answer_s": answer_seconds,
                "total_s": time.perf_counter() - total_start,
            },
            stepback_question=stepback_question,
            used_api=bool(config["api_key"]),
            guardrail_details=guardrail_details,
        )
        record["question_id"] = question_id
        record["question_metadata"] = row.get("metadata", {})
        record["retrieval_metrics"] = retrieval_metrics
        record["answer_metrics"] = answer_metrics
        attach_progress(record, completed=total_count, expected=len(questions))
        maybe_write_eval_record(config, record)
        print(
            f"[{question_id}] ok results={len(results)} confidence={answer_payload['confidence']} "
            f"retrieval_s={retrieval_seconds:.2f} answer_s={answer_seconds:.2f}"
        )

    print("Summary")
    print(f"- total_questions: {total_count}")
    print(f"- answered: {answer_count}")
    print(f"- blocked: {blocked_count}")
    print(f"- no_results: {no_result_count}")
    print(f"- avg_retrieval_s: {total_retrieval_s / total_count:.2f}" if total_count else "- avg_retrieval_s: 0.00")
    print(f"- avg_answer_s: {total_answer_s / max(answer_count, 1):.2f}")
    if metric_accumulator["judged"]:
        judged = metric_accumulator["judged"]
        print(f"- judged_queries: {judged}")
        print(f"- hit@{config['top_k']}: {metric_accumulator['hit_at_k'] / judged:.3f}")
        print(f"- recall@{config['top_k']}: {metric_accumulator['recall_at_k'] / judged:.3f}")
        print(f"- mrr@{config['top_k']}: {metric_accumulator['mrr_at_k'] / judged:.3f}")
    if answer_metric_accumulator["judged"]:
        judged = answer_metric_accumulator["judged"]
        print(f"- answer_judged: {judged}")
        print(
            f"- normalized_exact_match: "
            f"{answer_metric_accumulator['normalized_exact_match'] / judged:.3f}"
        )
        print(
            f"- contains_expected: "
            f"{answer_metric_accumulator['contains_expected'] / judged:.3f}"
        )
        print(f"- token_f1: {answer_metric_accumulator['token_f1'] / judged:.3f}")
        print(f"- anls: {answer_metric_accumulator['anls'] / judged:.3f}")
    if config["log_runs"]:
        print(f"- logs_dir: {APP_DIR / '.artifacts' / 'eval_logs'}")


def qdrant_cloud_config(*, search_backend: str, url: str, api_key: str, collection_name: str) -> dict[str, str] | None:
    if not search_backend.startswith("Qdrant Cloud"):
        return None
    if not url.strip() or not api_key.strip():
        raise SystemExit("Qdrant Cloud backend selected, but URL/API key are missing.")
    return {
        "url": url.strip(),
        "api_key": api_key.strip(),
        "collection_name": (collection_name or "multimodal_chunks").strip(),
    }


def maybe_write_eval_record(config: dict[str, Any], record: dict[str, Any]) -> None:
    if not config["log_runs"]:
        return
    write_eval_run(APP_DIR, run_label=str(config["run_label"]), record=record)


def attach_progress(record: dict[str, Any], *, completed: int, expected: int) -> None:
    pct = (completed / expected) if expected > 0 else 0.0
    record["progress"] = {
        "completed": completed,
        "expected": expected,
        "pct": pct,
    }


def result_records_from_search(results) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": result.chunk.chunk_id,
            "source_name": result.chunk.source_name,
            "text": result.chunk.text,
            "score": float(result.score),
        }
        for result in results
    ]


def metrics_for_query(*, row_id: str, results, qrels: dict[str, list[dict[str, Any]]], top_k: int) -> dict[str, float] | None:
    labels = qrels.get(row_id)
    if not labels:
        return None
    relevant = {str(item["corpus_id"]) for item in labels if int(item.get("score", 0)) > 0}
    if not relevant:
        return None
    ranked_ids = [result_identifier(result.chunk) for result in results[:top_k]]
    hits = [doc_id for doc_id in ranked_ids if doc_id in relevant]
    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            reciprocal_rank = 1.0 / rank
            break
    return {
        "hit_at_k": 1.0 if hits else 0.0,
        "recall_at_k": len(set(hits)) / len(relevant),
        "mrr_at_k": reciprocal_rank,
        "relevant_count": float(len(relevant)),
    }


def score_answer_for_row(*, answer_text: str, question_metadata: dict[str, Any], enabled: bool) -> dict[str, Any] | None:
    if not enabled:
        return None
    expected_answer = str(question_metadata.get("expected_answer") or "").strip()
    if not expected_answer:
        return None
    normalized_answer = normalize_answer_text(answer_text)
    normalized_expected = normalize_answer_text(expected_answer)
    return {
        "expected_answer": expected_answer,
        "normalized_answer": normalized_answer,
        "normalized_expected_answer": normalized_expected,
        "normalized_exact_match": 1.0 if normalized_answer == normalized_expected else 0.0,
        "contains_expected": 1.0 if normalized_expected and normalized_expected in normalized_answer else 0.0,
        "token_f1": token_f1(normalized_answer, normalized_expected),
        "anls": answer_nls(normalized_answer, normalized_expected),
    }


def normalize_answer_text(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9a-z\s.$:%/-]", "", text)
    return text.strip()


def token_f1(answer: str, expected: str) -> float:
    answer_tokens = answer.split()
    expected_tokens = expected.split()
    if not answer_tokens or not expected_tokens:
        return 0.0
    answer_counts = Counter(answer_tokens)
    expected_counts = Counter(expected_tokens)
    overlap = sum(min(answer_counts[token], expected_counts[token]) for token in answer_counts.keys() & expected_counts.keys())
    if overlap <= 0:
        return 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(expected_tokens)
    return (2 * precision * recall) / (precision + recall)


def answer_nls(answer: str, expected: str) -> float:
    if not answer or not expected:
        return 0.0
    if answer == expected:
        return 1.0
    distance = levenshtein_distance(answer, expected)
    denom = max(len(answer), len(expected))
    if denom <= 0:
        return 0.0
    similarity = 1.0 - (distance / denom)
    return similarity if similarity >= 0.5 else 0.0


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        curr = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def result_identifier(chunk: DocumentChunk) -> str:
    eval_doc_id = chunk.metadata.get("eval_doc_id")
    if eval_doc_id:
        return str(eval_doc_id)
    if chunk.modality in {"image", "visual_page", "pdf", "slide", "video"}:
        return str(chunk.source_name)
    return str(chunk.chunk_id)


def eval_settings_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    rag_mode: RagModeSpec = config["rag_mode"]
    return {
        "rag_mode": rag_mode.label,
        "rag_family": rag_mode.family,
        "search_backend": config["search_backend"],
        "reranker_label": config["reranker_label"],
        "rerank_top_n": config["rerank_top_n"],
        "top_k": config["top_k"],
        "token_budget": config["token_budget"],
        "guardrail_level": config["guardrail_level"],
        "page_render_scale": config["page_render_scale"],
        "visual_embedding_batch_size": config["visual_embedding_batch_size"],
        "enable_openai_visual_descriptions": config["enable_openai_visual_descriptions"],
        "light_compression_mode": config["light_compression_mode"],
        "light_target_tokens": config["light_target_tokens"],
        "score_answers": config["score_answers"],
    }


def build_answer_payload(answer: str, results) -> dict[str, Any]:
    sources = []
    seen = set()
    for result in results[:4]:
        source_id = result.chunk.chunk_id
        if source_id in seen:
            continue
        seen.add(source_id)
        sources.append(
            {
                "source_id": source_id,
                "doc_id": str(result.chunk.metadata.get("doc_id") or result.chunk.source_name),
                "page_num": result.chunk.metadata.get("page_num") or result.chunk.page_or_frame,
                "chunk_id": result.chunk.chunk_id,
            }
        )
    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence_from_results(results),
        "guardrail_summary": {},
    }


def confidence_from_results(results) -> str:
    if not results:
        return "low"
    top_score = float(results[0].score)
    if top_score >= 0.5:
        return "high"
    if top_score >= 0.15:
        return "medium"
    return "low"


def evidence_only_answer(results) -> str:
    snippets = []
    for result in results[:3]:
        snippets.append(f"- {text_preview(result.chunk.text, limit=160)} [{result.chunk.chunk_id}]")
    if not snippets:
        return "I could not find enough support in the indexed materials to answer reliably."
    return "Top retrieved evidence:\n" + "\n".join(snippets)


def merge_results(primary, secondary, top_k: int):
    merged = {}
    for result in list(primary) + list(secondary):
        key = result.chunk.chunk_id
        if key not in merged or result.score > merged[key].score:
            merged[key] = result
    return sorted(merged.values(), key=lambda result: result.score, reverse=True)[:top_k]


def resolve_openai_api_key(*, user_api_key: str | None = None) -> str | None:
    key = get_api_key(user_api_key=user_api_key)
    if key:
        return key
    secrets = load_local_streamlit_secrets()
    if not secrets:
        return None
    return get_api_key(secrets=secrets, user_api_key=user_api_key)


def load_local_streamlit_secrets() -> dict[str, Any]:
    secrets_path = APP_DIR / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return {}
    try:
        with secrets_path.open("rb") as handle:
            data = tomllib.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


if __name__ == "__main__":
    main()
