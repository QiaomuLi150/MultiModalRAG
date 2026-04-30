from __future__ import annotations

import os
from pathlib import Path
import time

import pandas as pd
import streamlit as st

from src.audio import ASR_MODELS, DEFAULT_ASR_MODEL, transcribe_audio_bytes, transcribe_video_audio_bytes
from src.chunking import DocumentChunk, add_parent_metadata, assign_chunk_ids, text_preview
from src.generation import MODEL_NAME, generate_answer, generate_stepback_question, get_api_key
from src.guardrails.config import config_for_level
from src.guardrails.pipeline import GuardrailPipeline
from src.docling_adapter import DOCLING_MODES, chunks_from_pdf_docling_bytes
from src.eval_support import apply_preset, build_eval_record, preset_by_name, preset_names, write_eval_run
from src.ingest import (
    chunk_from_image_bytes,
    chunk_from_video_frame,
    chunks_from_csv_bytes,
    chunks_from_pdf_bytes,
    chunks_from_text_bytes,
    load_sample_chunks,
)
from src.production_architecture import production_rows
from src.rag_modes import (
    RagModeSpec,
    build_mode_index,
    families as rag_families,
    mode_by_label,
    modes_for_family,
)
from src.render import ingest_image_bytes, ingest_pil_image, render_pdf_bytes
from src.rerankers import reranker_by_label, reranker_choices
from src.schemas import PageRecord
from src.vision import sample_video_frames

APP_DIR = Path(__file__).parent
SAMPLE_DIR = APP_DIR / "sample_data"
PAGE_IMAGE_DIR = APP_DIR / ".artifacts" / "page_images"


def init_state() -> None:
    defaults = {
        "raw_chunks": [],
        "chunks": [],
        "page_records": [],
        "index": None,
        "qdrant_banner": None,
        "latest_eval_artifacts": None,
        "sample_loaded": False,
        "video_frames": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def add_chunks(chunks: list[DocumentChunk]) -> None:
    st.session_state.raw_chunks.extend(chunks)
    st.session_state.chunks = add_parent_metadata(assign_chunk_ids(st.session_state.raw_chunks))
    st.session_state.index = None


def api_status(provider: str, user_api_key: str) -> tuple[str | None, bool]:
    key = get_api_key(st.secrets, user_api_key=user_api_key)
    use_api = provider == f"OpenAI {MODEL_NAME}" and bool(key)
    return key, use_api


def _prefill_secret(secret_key: str, env_key: str, default: str = "") -> str:
    try:
        value = st.secrets.get(secret_key)
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get(env_key, default)


def main() -> None:
    st.set_page_config(page_title="Multimodal RAG Demo", page_icon="M", layout="wide")
    init_state()

    st.title("Multimodal RAG Demo")
    st.write(
        "Turn slides, transcripts, images, and optional video frames into searchable "
        "evidence, then ask grounded questions with citations."
    )
    st.caption(
        "Recommended flow: choose a RAG mode, connect or build the index, inspect evidence, then ask grounded questions."
    )
    default_openai_key = _prefill_secret("OPENAI_API_KEY", "OPENAI_API_KEY")
    default_qdrant_url = _prefill_secret("QDRANT_CLOUD_URL", "QDRANT_CLOUD_URL")
    default_qdrant_api_key = _prefill_secret("QDRANT_CLOUD_API_KEY", "QDRANT_CLOUD_API_KEY")
    default_qdrant_collection = _prefill_secret("QDRANT_COLLECTION", "QDRANT_COLLECTION", "multimodal_chunks")

    with st.sidebar:
        st.header("Settings")
        st.caption("Configure retrieval first, then load files and build the index.")
        provider = st.radio(
            "Model provider",
            [f"OpenAI {MODEL_NAME}", "No API / retrieval only"],
            index=0,
        )
        user_api_key = st.text_input(
            "OpenAI API key",
            type="password",
            value=default_openai_key,
            placeholder="Optional. Used only for this session.",
            help="Paste a key here to enable chat answers without Streamlit secrets. The app does not print or log it.",
        )
        retrieval_family = st.radio(
            "Retrieval family",
            rag_families(),
            index=0,
            help="Choose the overall RAG style first, then the specific mode under that family.",
        )
        family_modes = modes_for_family(retrieval_family)
        selected_mode_label = st.selectbox(
            "RAG mode",
            [mode.label for mode in family_modes],
            index=0,
            help="Modes stay optional so users can choose cheaper text retrieval or stronger page-oriented retrieval.",
        )
        rag_mode = mode_by_label(selected_mode_label)
        search_backend = st.selectbox(
            "Search engine",
            [
                "Hybrid",
                "Semantic",
                "TF-IDF",
                "FAISS",
                "FAISS hybrid",
                "Qdrant",
                "Qdrant hybrid",
                "Qdrant Cloud",
                "Qdrant Cloud hybrid",
                "Parent-child",
            ],
            index=0,
            help="This powers the selected RAG mode. Visual modes now use rendered page images when available.",
        )
        st.caption(rag_mode.description)
        st.info(
            "Mode guide: Text is cheapest, Visual is best for diagrams/slides/scans, Hybrid is the best general-purpose enterprise choice."
        )
        reranker_label = st.selectbox(
            "Text reranker",
            reranker_choices(),
            index=0,
            help="Optional second-stage reranker. Improves final ranking quality more than recall.",
        )
        reranker_spec = reranker_by_label(reranker_label)
        rerank_top_n = st.slider(
            "Rerank top N",
            min_value=4,
            max_value=20,
            value=8,
            step=2,
            help="Only the top retrieved candidates are reranked. Lower values are faster.",
        )
        st.caption(f"Reranker note: {reranker_spec.note}")
        page_render_scale = 1.5
        page_ocr_enabled = rag_mode.use_visual_backend or rag_mode.use_text_hybrid
        visual_model_name = "vidore/colqwen2-v1.0-hf"
        visual_local_files_only = True
        visual_embedding_batch_size = 1
        light_compression_mode = "none"
        light_target_tokens: int | None = None
        use_muvera_proxy = False
        muvera_candidate_count = 12
        if rag_mode.use_visual_backend:
            st.divider()
            st.subheader("Visual Retrieval")
            visual_model_name = st.text_input(
                "Visual model",
                value="vidore/colqwen2-v1.0-hf",
                help="ColQwen2-compatible retrieval model used for true page-image retrieval.",
            )
            visual_local_files_only = st.toggle(
                "Local model files only",
                value=True,
                help="Keep this enabled unless the model weights are already downloadable in your runtime environment.",
            )
            visual_embedding_batch_size = st.slider(
                "Visual embedding batch size",
                min_value=1,
                max_value=8,
                value=1,
                step=1,
                help="Lower this if GPU memory is tight. Batch size 1 is safest for large page images and ColQwen2.",
            )
            light_mode_enabled = st.toggle(
                "Enable Light-ColPali-style compression",
                value=True,
                help="Reduces the number of patch/token vectors stored per page by pooling them before indexing.",
            )
            if light_mode_enabled:
                light_compression_mode = st.selectbox(
                    "Compression mode",
                    ["similarity_merge", "mean_pool"],
                    index=0,
                    help="Similarity-aware merging preserves local structure better than plain pooling. Mean pooling is the simpler fallback.",
                )
                light_target_tokens = st.slider(
                    "Target vectors per page",
                    min_value=16,
                    max_value=256,
                    value=64,
                    step=16,
                    help="Lower values save more memory and storage, but can reduce retrieval quality.",
                )
            use_muvera_proxy = st.toggle(
                "Enable MUVERA-style proxy retrieval",
                value=False,
                help="Optional fast candidate generation for multivector visual retrieval. Not the default.",
            )
            if use_muvera_proxy:
                muvera_candidate_count = st.slider(
                    "MUVERA candidate pool",
                    min_value=8,
                    max_value=40,
                    value=12,
                    step=4,
                    help="The proxy stage retrieves this many candidates before exact local multivector reranking when page embeddings are available in the current session.",
                )
                st.caption(
                    "MUVERA-style proxy retrieval is an optional acceleration layer for visual multivector search. "
                    "It adds a proxy collection and two-stage retrieval, so it is not enabled by default."
                )
            page_render_scale = st.slider(
                "Page render scale",
                min_value=1.0,
                max_value=3.0,
                value=1.5,
                step=0.25,
                help="Higher scale improves page-image fidelity but increases indexing cost and storage.",
            )
            page_ocr_enabled = st.toggle(
                "OCR rendered pages",
                value=True,
                help="Attach OCR text to rendered page records for evidence display and hybrid search.",
            )
            st.caption("Visual modes require ColQwen2-compatible model weights plus rendered page/image assets.")
        pdf_extractor = st.selectbox(
            "PDF/slide extractor",
            ["pypdf", "Docling balanced", "Docling strict", "Docling broad"],
            index=0,
            help="Docling is structure-aware but optional and heavier. pypdf is the lightweight fallback.",
        )
        qdrant_cloud_url = ""
        qdrant_cloud_api_key = ""
        qdrant_collection = "multimodal_chunks"
        connect_qdrant_cloud = False
        if search_backend.startswith("Qdrant Cloud"):
            qdrant_cloud_url = st.text_input(
                "Qdrant Cloud URL",
                value=default_qdrant_url,
                placeholder="https://...cloud.qdrant.io",
            )
            qdrant_cloud_api_key = st.text_input(
                "Qdrant API key",
                type="password",
                value=default_qdrant_api_key,
                placeholder="Paste key for this session",
            )
            qdrant_collection = st.text_input("Qdrant collection", value=default_qdrant_collection)
            connect_qdrant_cloud = st.button(
                "Connect existing Qdrant collection",
                help="Use vectors already saved in Qdrant Cloud without re-uploading files after a restart.",
            )
        top_k = st.slider("Retrieved chunks", 1, 10, 5)
        token_budget = st.slider("Context token budget", 1000, 8000, 3000, step=500)
        enable_video = st.toggle("Enable video frame sampling", value=True)
        max_video_segments = st.slider("Video segments", 1, 10, 5)
        transcribe_video_audio = st.toggle("Transcribe video audio", value=True)
        asr_model = st.selectbox(
            "ASR model",
            ASR_MODELS,
            index=ASR_MODELS.index(DEFAULT_ASR_MODEL),
            help="large-v3 is more accurate but much slower and heavier on CPU.",
        )
        if asr_model == "large-v3":
            st.warning("large-v3 may be slow or run out of memory on Streamlit Community Cloud CPU.")
        st.divider()
        enable_guardrails = st.toggle(
            "Enable guardrails",
            value=True,
            help="Adds lightweight input, retrieval, groundedness, PII, and output checks around the RAG pipeline.",
        )
        guardrail_level = "Off"
        if enable_guardrails:
            guardrail_level = st.selectbox(
                "Guardrail level",
                ["Relaxed", "Balanced", "Strict"],
                index=1,
                help="Relaxed warns more and blocks less. Strict is the most conservative.",
            )
            guardrail_notes = {
                "Relaxed": "Allows more borderline answers through, with lower confidence and softer checks.",
                "Balanced": "Default. Blocks clearly weak or unsupported answers while keeping normal questions usable.",
                "Strict": "Most conservative. Requires stronger evidence and refuses more weak or ambiguous cases.",
            }
            st.caption(f"Guardrail note: {guardrail_notes[guardrail_level]}")
        else:
            st.caption("Guardrails are off. The app will answer based only on retrieval and generation behavior.")
        st.divider()
        st.subheader("Evaluation")
        enable_eval_mode = st.toggle(
            "Enable evaluation mode",
            value=False,
            help="Keep run settings reproducible and write structured outputs for later comparison.",
        )
        eval_preset_name = "Custom"
        eval_run_label = "manual_eval"
        eval_log_runs = False
        force_retrieval_only_eval = False
        if enable_eval_mode:
            eval_preset_name = st.selectbox(
                "Evaluation preset",
                preset_names(),
                index=0,
                help="Presets override run-time settings for reproducible experiments without changing the core architecture.",
            )
            eval_run_label = st.text_input(
                "Evaluation run label",
                value="manual_eval",
                help="Used for JSONL and CSV output file names under .artifacts/eval_logs/.",
            )
            eval_log_runs = st.toggle(
                "Log evaluation runs",
                value=True,
                help="Write per-question JSON and append JSONL/CSV summaries for each run.",
            )
            force_retrieval_only_eval = st.toggle(
                "Force retrieval-only evaluation",
                value=False,
                help="Useful for retrieval benchmarks where you do not want LLM answer generation cost or variance.",
            )
            st.caption(preset_by_name(eval_preset_name).description)
        st.caption("Retrieval is local/open-source. OpenAI is used for VLM descriptions and final chat when a key is provided.")

    if enable_eval_mode:
        resolved = apply_preset(
            eval_preset_name,
            {
                "selected_mode_label": selected_mode_label,
                "search_backend": search_backend,
                "reranker_label": reranker_label,
                "rerank_top_n": rerank_top_n,
                "top_k": top_k,
                "token_budget": token_budget,
                "page_render_scale": page_render_scale,
                "visual_embedding_batch_size": visual_embedding_batch_size,
                "light_compression_mode": light_compression_mode,
                "light_target_tokens": light_target_tokens,
                "guardrail_level": guardrail_level,
                "force_retrieval_only_eval": force_retrieval_only_eval,
            },
        )
        if resolved["selected_mode_label"] != selected_mode_label:
            rag_mode = mode_by_label(resolved["selected_mode_label"])
            selected_mode_label = resolved["selected_mode_label"]
        search_backend = resolved["search_backend"]
        reranker_label = resolved["reranker_label"]
        reranker_spec = reranker_by_label(reranker_label)
        rerank_top_n = resolved.get("rerank_top_n", rerank_top_n)
        top_k = resolved.get("top_k", top_k)
        token_budget = resolved.get("token_budget", token_budget)
        page_render_scale = resolved.get("page_render_scale", page_render_scale)
        visual_embedding_batch_size = resolved.get("visual_embedding_batch_size", visual_embedding_batch_size)
        light_compression_mode = resolved.get("light_compression_mode", light_compression_mode)
        light_target_tokens = resolved.get("light_target_tokens", light_target_tokens)
        guardrail_level = resolved.get("guardrail_level", guardrail_level)
        force_retrieval_only_eval = resolved.get("force_retrieval_only_eval", force_retrieval_only_eval)
        if search_backend.startswith("Qdrant Cloud"):
            qdrant_cloud_url = qdrant_cloud_url or default_qdrant_url
            qdrant_cloud_api_key = qdrant_cloud_api_key or default_qdrant_api_key
            qdrant_collection = qdrant_collection or default_qdrant_collection

    api_key, use_api = api_status(provider, user_api_key)
    if enable_eval_mode and force_retrieval_only_eval:
        use_api = False
    vlm_api_key = api_key if provider == f"OpenAI {MODEL_NAME}" else None
    guardrails = GuardrailPipeline(config_for_level(guardrail_level)) if enable_guardrails else None
    if not api_key:
        st.warning(
            "No OPENAI_API_KEY found. The app will show retrieved evidence instead of "
            "generating a final answer."
        )
    elif provider == "No API / retrieval only":
        st.info("Retrieval-only mode is selected. No OpenAI call will be made.")

    _render_workspace_summary(rag_mode, search_backend, provider)
    _render_qdrant_banner(search_backend)
    if enable_eval_mode:
        st.caption(
            f"Evaluation mode is active. Preset={eval_preset_name}, run_label={eval_run_label}, "
            f"logging={'on' if eval_log_runs else 'off'}, retrieval_only={'on' if force_retrieval_only_eval else 'off'}."
        )
        if st.session_state.get("latest_eval_artifacts"):
            latest = st.session_state["latest_eval_artifacts"]
            st.caption(
                "Latest eval outputs: "
                f"jsonl={latest.get('jsonl_path', '')}, "
                f"summary={latest.get('summary_csv_path', '')}, "
                f"detail={latest.get('detail_json_path', '')}"
            )

    if connect_qdrant_cloud:
        try:
            qdrant_cloud_config = _qdrant_cloud_config(
                search_backend,
                qdrant_cloud_url,
                qdrant_cloud_api_key,
                qdrant_collection,
            )
            cloud_index = build_mode_index(
                [],
                rag_mode,
                search_backend,
                qdrant_cloud_config=qdrant_cloud_config,
                page_records=st.session_state.get("page_records", []),
                visual_model_name=visual_model_name,
                visual_local_files_only=visual_local_files_only,
                visual_embedding_batch_size=visual_embedding_batch_size,
                light_compression_mode=light_compression_mode,
                light_target_tokens=light_target_tokens,
                use_muvera_proxy=use_muvera_proxy,
                muvera_candidate_count=muvera_candidate_count,
                reranker_spec=reranker_spec,
                rerank_top_n=rerank_top_n,
            )
            if "qdrant_cloud" not in getattr(cloud_index, "encoder_name", ""):
                st.error("Could not connect to Qdrant Cloud. Check the URL, API key, and collection name.")
            else:
                status = _index_status(cloud_index)
                point_count = status.get("point_count", 0)
                if point_count:
                    st.session_state.index = cloud_index
                    st.session_state.qdrant_banner = {
                        "kind": "success",
                        "message": (
                            f"Connected to Qdrant Cloud collection "
                            f"'{status.get('collection_name', qdrant_collection)}' with {point_count} points."
                        ),
                    }
                    st.success(
                        f"Connected to Qdrant Cloud collection "
                        f"'{status.get('collection_name', qdrant_collection)}' with {point_count} points."
                    )
                else:
                    st.session_state.qdrant_banner = {
                        "kind": "warning",
                        "message": (
                            f"Connected to Qdrant Cloud collection "
                            f"'{status.get('collection_name', qdrant_collection)}', but it currently has 0 points."
                        ),
                    }
                    st.warning(
                        f"Connected to Qdrant Cloud collection "
                        f"'{status.get('collection_name', qdrant_collection)}', but it contains 0 points."
                    )
                    st.info("Load files and click Build index to upload vectors to this collection.")
        except Exception as exc:
            st.session_state.qdrant_banner = {
                "kind": "error",
                "message": f"Could not connect to Qdrant Cloud: {exc}",
            }
            st.error(f"Could not connect to Qdrant Cloud: {exc}")

    with st.expander("Production target mapping", expanded=False):
        st.dataframe(pd.DataFrame(production_rows()), width="stretch", hide_index=True)
        st.caption(
            "The Streamlit demo keeps heavyweight production services optional. The code path "
            "still follows the demanded stages: storage, modality conversion, chunking, "
            "encoding, retrieval, context packing, and grounded answering."
        )

    st.subheader("1. Add Content")
    st.caption("Upload files or load the bundled sample set. Visual modes benefit from PDFs, images, or video frames.")
    left, right, reset_col = st.columns([1, 1, 1])
    if left.button("Load sample meeting"):
        st.session_state.raw_chunks = load_sample_chunks(SAMPLE_DIR)
        st.session_state.chunks = add_parent_metadata(assign_chunk_ids(st.session_state.raw_chunks))
        st.session_state.page_records = []
        st.session_state.index = None
        st.session_state.sample_loaded = True
        st.success("Loaded sample meeting transcript, slide notes, and image descriptions.")

    uploaded_files = st.file_uploader(
        "Upload text, CSV, PDF, images, audio, or video",
        type=["txt", "md", "csv", "pdf", "png", "jpg", "jpeg", "wav", "mp3", "m4a", "mp4", "mov"],
        accept_multiple_files=True,
        help="Supported files are converted into searchable chunks. Images and video frames can use OCR, manual notes, and optional gpt-5-nano visual descriptions.",
    )

    image_assets: list[tuple[str, bytes, str]] = []
    video_assets: list[tuple[str, object, str, str, str]] = []
    text_files: list[tuple[str, bytes]] = []
    csv_files: list[tuple[str, bytes]] = []
    pdf_files: list[tuple[str, bytes]] = []
    audio_files: list[tuple[str, bytes]] = []
    video_audio_files: list[tuple[str, bytes]] = []

    for uploaded_index, uploaded in enumerate(uploaded_files):
        suffix = Path(uploaded.name).suffix.lower()
        data = uploaded.getvalue()
        if suffix in {".txt", ".md"}:
            text_files.append((uploaded.name, data))
        elif suffix == ".csv":
            csv_files.append((uploaded.name, data))
        elif suffix == ".pdf":
            pdf_files.append((uploaded.name, data))
        elif suffix in {".wav", ".mp3", ".m4a"}:
            audio_files.append((uploaded.name, data))
            st.audio(data)
        elif suffix in {".png", ".jpg", ".jpeg"}:
            st.image(data, caption=uploaded.name, width=260)
            description = st.text_area(
                f"Describe image: {uploaded.name}",
                key=f"image_description_{uploaded_index}_{uploaded.name}",
                placeholder="Visible text, chart trends, diagrams, people, or other searchable details.",
            )
            image_assets.append((uploaded.name, data, description))
        elif suffix in {".mp4", ".mov"}:
            if transcribe_video_audio:
                video_audio_files.append((uploaded.name, data))
            if not enable_video:
                st.info(f"Skipped {uploaded.name}. Enable video frame sampling in the sidebar.")
                continue
            if not vlm_api_key:
                st.warning("No API key found. Video frames will use OCR/manual text without gpt-5-nano VLM descriptions.")
            try:
                cache_key = f"frames_{uploaded_index}_{uploaded.name}"
                if cache_key not in st.session_state.video_frames:
                    st.session_state.video_frames[cache_key] = sample_video_frames(
                        data,
                        max_frames=max_video_segments,
                        source_name=uploaded.name,
                    )
                for frame_info in st.session_state.video_frames[cache_key]:
                    st.image(
                        frame_info.image,
                        caption=(
                            f"{uploaded.name} {frame_info.segment_id} "
                            f"{frame_info.segment_start}-{frame_info.segment_end} "
                            f"(frame {frame_info.frame})"
                        ),
                        width=260,
                    )
                    description = st.text_area(
                        f"Optional correction for {frame_info.segment_id} from {uploaded.name}",
                        key=f"video_description_{uploaded_index}_{uploaded.name}_{frame_info.frame}",
                        placeholder=(
                            "Optional. Add visible slide text, speaker actions, scene changes, "
                            "or meeting relevance if OCR/VLM are unavailable."
                        ),
                    )
                    video_assets.append(
                        (
                            uploaded.name,
                            frame_info.image,
                            f"{frame_info.segment_start}-{frame_info.segment_end}",
                            frame_info.frame,
                            description,
                        )
                    )
            except RuntimeError as exc:
                st.error(f"{exc} Upload a transcript or screenshots instead.")
        else:
            st.info(f"Unsupported file type: {uploaded.name}")

    _render_upload_summary(
        text_files=text_files,
        csv_files=csv_files,
        pdf_files=pdf_files,
        audio_files=audio_files,
        video_audio_files=video_audio_files,
        image_assets=image_assets,
        video_assets=video_assets,
    )

    build_clicked = right.button("Build / sync index", type="primary")
    if reset_col.button("Reset workspace"):
        for key in [
            "raw_chunks",
            "chunks",
            "page_records",
            "index",
            "qdrant_banner",
            "sample_loaded",
            "video_frames",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

    if build_clicked:
        chunks: list[DocumentChunk] = []
        page_records: list[PageRecord] = []
        for name, data in text_files:
            chunks.extend(chunks_from_text_bytes(data, name, modality="text"))
        for name, data in csv_files:
            chunks.extend(chunks_from_csv_bytes(data, name))
        for name, data in pdf_files:
            try:
                if pdf_extractor.startswith("Docling"):
                    docling_mode = pdf_extractor.split()[-1]
                    docling_chunks, status = chunks_from_pdf_docling_bytes(data, name, mode=docling_mode)
                    if docling_chunks:
                        chunks.extend(docling_chunks)
                        st.success(status)
                    else:
                        st.warning(f"{status} Falling back to pypdf.")
                        chunks.extend(chunks_from_pdf_bytes(data, name))
                else:
                    chunks.extend(chunks_from_pdf_bytes(data, name))
            except Exception as exc:
                st.error(f"Could not extract text from {name}: {exc}")
            if rag_mode.use_visual_backend:
                try:
                    page_records.extend(
                        render_pdf_bytes(
                            data,
                            name,
                            PAGE_IMAGE_DIR,
                            render_scale=page_render_scale,
                            include_ocr=page_ocr_enabled,
                        )
                    )
                except Exception as exc:
                    st.error(f"Could not render pages for {name}: {exc}")
        for name, data in audio_files:
            audio_chunks, status = transcribe_audio_bytes(data, name, model_size=asr_model)
            if audio_chunks:
                chunks.extend(audio_chunks)
                if any(chunk.metadata.get("chunker") == "audio_status_fallback" for chunk in audio_chunks):
                    st.warning(status)
                else:
                    st.success(status)
            else:
                st.warning(status)
        for name, data in video_audio_files:
            audio_chunks, status = transcribe_video_audio_bytes(data, name, model_size=asr_model)
            if audio_chunks:
                chunks.extend(audio_chunks)
                if any(chunk.metadata.get("chunker") == "audio_status_fallback" for chunk in audio_chunks):
                    st.warning(status)
                else:
                    st.success(status)
            else:
                st.warning(status)
        for name, data, description in image_assets:
            try:
                chunks.append(chunk_from_image_bytes(data, name, description, openai_api_key=vlm_api_key))
            except Exception as exc:
                st.error(f"Could not encode image {name}: {exc}")
            if rag_mode.use_visual_backend:
                try:
                    page_records.append(
                        ingest_image_bytes(
                            data,
                            name,
                            PAGE_IMAGE_DIR,
                            include_ocr=page_ocr_enabled,
                            title_hint=description.strip() or None,
                        )
                    )
                except Exception as exc:
                    st.error(f"Could not create visual page record for {name}: {exc}")
        for name, frame_image, timestamp, frame_number, description in video_assets:
            try:
                chunks.append(
                    chunk_from_video_frame(
                        name,
                        frame_image,
                        timestamp,
                        frame_number,
                        description,
                        openai_api_key=vlm_api_key,
                    )
                )
            except Exception as exc:
                st.error(f"Could not encode video frame {frame_number} from {name}: {exc}")
            if rag_mode.use_visual_backend:
                try:
                    page_records.append(
                        ingest_pil_image(
                            frame_image,
                            f"{name}_{frame_number}",
                            PAGE_IMAGE_DIR,
                            include_ocr=page_ocr_enabled,
                            page_num=int(frame_number) if str(frame_number).isdigit() else len(page_records) + 1,
                            title_hint=description.strip() or None,
                            source_type="video_frame",
                            extra_metadata={"timestamp": timestamp, "frame_number": frame_number},
                        )
                    )
                except Exception as exc:
                    st.error(f"Could not create visual page record for video frame {frame_number} from {name}: {exc}")
        if chunks:
            add_chunks(chunks)
        st.session_state.page_records = page_records
        if not st.session_state.chunks:
            st.error("No chunks were available to index. Load sample data or upload files first.")
            return
        if rag_mode.use_visual_backend and not st.session_state.page_records:
            st.error("Visual mode was selected, but no rendered page/image records were available to index.")
            return
        qdrant_cloud_config = _qdrant_cloud_config(
            search_backend,
            qdrant_cloud_url,
            qdrant_cloud_api_key,
            qdrant_collection,
        )
        try:
            st.session_state.index = build_mode_index(
                st.session_state.chunks,
                rag_mode,
                search_backend,
                qdrant_cloud_config=qdrant_cloud_config,
                page_records=st.session_state.page_records,
                visual_model_name=visual_model_name,
                visual_local_files_only=visual_local_files_only,
                visual_embedding_batch_size=visual_embedding_batch_size,
                light_compression_mode=light_compression_mode,
                light_target_tokens=light_target_tokens,
                use_muvera_proxy=use_muvera_proxy,
                muvera_candidate_count=muvera_candidate_count,
                reranker_spec=reranker_spec,
                rerank_top_n=rerank_top_n,
            )
        except Exception as exc:
            st.session_state.index = None
            st.error(f"Indexing failed: {exc}")
            if search_backend.startswith("Qdrant Cloud"):
                st.caption(
                    "For large corpora, Qdrant Cloud uploads are batched. "
                    "If this still fails, reduce chunk size or avoid storing very large raw text payloads."
                )
            return
        if (
            search_backend.startswith("Qdrant Cloud")
            and "qdrant_cloud" not in getattr(st.session_state.index, "encoder_name", "")
        ):
            st.error("Qdrant Cloud indexing failed. Check the URL, API key, and collection name.")
            return
        qdrant_status = _index_status(st.session_state.index)
        qdrant_status_text = ""
        if search_backend.startswith("Qdrant Cloud"):
            if qdrant_status.get("point_count", 0) <= 0:
                st.error("Qdrant Cloud indexing produced 0 saved points. Check credentials, collection name, and upload data.")
                return
            st.session_state.qdrant_banner = {
                "kind": "success",
                "message": (
                    f"Qdrant Cloud is active. Collection '{qdrant_status.get('collection_name', qdrant_collection)}' "
                    f"now has {qdrant_status.get('point_count', 0)} points available for retrieval."
                ),
            }
            qdrant_status_text = (
                f" Uploaded {qdrant_status.get('uploaded_count', 0)} chunks to Qdrant Cloud; "
                f"collection now has {qdrant_status.get('point_count', 0)} points."
            )
        st.success(
            f"Built {getattr(st.session_state.index, 'encoder_name', search_backend)} "
            f"index with {len(st.session_state.chunks)} chunks.{qdrant_status_text}"
        )

    st.subheader("2. Inspect Indexed Evidence")
    st.caption("Use this section to verify what the retriever can actually see before you ask a question.")
    if st.session_state.get("page_records"):
        with st.expander("Rendered page records", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "page_id": page.page_id,
                            "doc_id": page.doc_id,
                            "page_num": page.page_num,
                            "source_name": page.source_name,
                            "source_type": page.source_type,
                            "image_path": page.image_path,
                            "ocr_preview": text_preview(page.ocr_text, limit=120),
                        }
                        for page in st.session_state.page_records
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
    if st.session_state.chunks:
        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "modality": chunk.modality,
                "source_name": chunk.source_name,
                "timestamp/frame": chunk.timestamp or chunk.page_or_frame or "",
                "parent_id": chunk.metadata.get("parent_id", ""),
                "converter": chunk.metadata.get("converter", ""),
                "chunker": chunk.metadata.get("chunker", ""),
                "encoder": chunk.metadata.get("encoder", ""),
                "OCR/ASR/VLM": (
                    chunk.metadata.get("ocr_status")
                    or chunk.metadata.get("vlm_status")
                    or chunk.metadata.get("asr_status")
                    or chunk.metadata.get("converter", "")
                ),
                "text preview": text_preview(chunk.text),
            }
            for chunk in st.session_state.chunks
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("Load sample data or upload files, then build an index.")

    st.subheader("3. Ask Questions")
    st.caption("Ask either exact policy questions or visual/layout questions like 'show me the page with the architecture diagram'.")
    with st.form("qa_form", clear_on_submit=False, enter_to_submit=True):
        question = st.text_input(
            "Question",
            key="qa_question",
            placeholder="What are the key product roadmap decisions?",
        )
        ask = st.form_submit_button("Ask", type="primary")

    if ask:
        guardrail_details: dict = {}
        effective_question = question
        if guardrails is not None:
            pre = guardrails.run_pre_retrieval(question)
            guardrail_details.update(pre.get("results", {}))
            effective_question = pre.get("query", question)
            if pre["action"] != "continue":
                _render_guardrail_result(pre["response"], guardrail_details)
                if enable_eval_mode and eval_log_runs:
                    st.session_state["latest_eval_artifacts"] = write_eval_run(
                        APP_DIR,
                        run_label=eval_run_label,
                        record=build_eval_record(
                            run_label=eval_run_label,
                            question=question,
                            effective_question=effective_question,
                            settings=_eval_settings_snapshot(
                                rag_mode=rag_mode,
                                search_backend=search_backend,
                                reranker_label=reranker_label,
                                rerank_top_n=rerank_top_n,
                                top_k=top_k,
                                token_budget=token_budget,
                                guardrail_level=guardrail_level,
                                page_render_scale=page_render_scale,
                                visual_embedding_batch_size=visual_embedding_batch_size,
                                light_compression_mode=light_compression_mode,
                                light_target_tokens=light_target_tokens,
                            ),
                            answer_payload=pre["response"],
                            results=[],
                            latencies={"retrieval_s": 0.0, "stepback_s": 0.0, "answer_s": 0.0, "total_s": 0.0},
                            stepback_question="",
                            used_api=False,
                            guardrail_details=guardrail_details,
                        ),
                    )
        can_use_cloud_index = (
            search_backend.startswith("Qdrant Cloud")
            and st.session_state.index is not None
            and "qdrant_cloud" in getattr(st.session_state.index, "encoder_name", "")
            and _index_status(st.session_state.index).get("point_count", 0) > 0
        )
        if guardrails is not None and pre["action"] != "continue":
            return
        if not st.session_state.chunks and not can_use_cloud_index:
            st.error("Build an index before asking a question.")
            return
        total_start = time.perf_counter()
        if st.session_state.index is None:
            qdrant_cloud_config = _qdrant_cloud_config(
                search_backend,
                qdrant_cloud_url,
                qdrant_cloud_api_key,
                qdrant_collection,
            )
            st.session_state.index = build_mode_index(
                st.session_state.chunks,
                rag_mode,
                search_backend,
                qdrant_cloud_config=qdrant_cloud_config,
                page_records=st.session_state.get("page_records", []),
                visual_model_name=visual_model_name,
                visual_local_files_only=visual_local_files_only,
                visual_embedding_batch_size=visual_embedding_batch_size,
                light_compression_mode=light_compression_mode,
                light_target_tokens=light_target_tokens,
                use_muvera_proxy=use_muvera_proxy,
                muvera_candidate_count=muvera_candidate_count,
                reranker_spec=reranker_spec,
                rerank_top_n=rerank_top_n,
            )
        retrieval_start = time.perf_counter()
        results = st.session_state.index.search(effective_question, top_k=top_k)
        retrieval_seconds = time.perf_counter() - retrieval_start
        result_records = _result_records(results)
        if guardrails is not None:
            post_retrieval = guardrails.run_post_retrieval(result_records)
            guardrail_details.update(post_retrieval.get("results", {}))
            if post_retrieval["action"] != "continue":
                _render_guardrail_result(post_retrieval["response"], guardrail_details)
                metric_cols = st.columns(4)
                metric_cols[0].metric("Retrieval", f"{retrieval_seconds:.2f}s")
                metric_cols[1].metric("Step-back", "0.00s")
                metric_cols[2].metric("Answer", "0.00s")
                metric_cols[3].metric("Total", f"{time.perf_counter() - total_start:.2f}s")
                if enable_eval_mode and eval_log_runs:
                    st.session_state["latest_eval_artifacts"] = write_eval_run(
                        APP_DIR,
                        run_label=eval_run_label,
                        record=build_eval_record(
                            run_label=eval_run_label,
                            question=question,
                            effective_question=effective_question,
                            settings=_eval_settings_snapshot(
                                rag_mode=rag_mode,
                                search_backend=search_backend,
                                reranker_label=reranker_label,
                                rerank_top_n=rerank_top_n,
                                top_k=top_k,
                                token_budget=token_budget,
                                guardrail_level=guardrail_level,
                                page_render_scale=page_render_scale,
                                visual_embedding_batch_size=visual_embedding_batch_size,
                                light_compression_mode=light_compression_mode,
                                light_target_tokens=light_target_tokens,
                            ),
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
                        ),
                    )
                return
        stepback_question = ""
        stepback_seconds = 0.0
        if rag_mode.use_step_back and use_api and api_key:
            try:
                stepback_start = time.perf_counter()
                stepback_question = generate_stepback_question(effective_question, api_key)
                stepback_results = st.session_state.index.search(stepback_question, top_k=top_k)
                stepback_seconds = time.perf_counter() - stepback_start
                results = _merge_results(results, stepback_results, top_k)
                st.caption(f"Step-back retrieval query: {stepback_question}")
            except Exception as exc:
                st.warning(f"Step-back query generation failed, using original query only: {exc}")
        if not results:
            total_seconds = time.perf_counter() - total_start
            st.warning("No relevant evidence was found.")
            if search_backend.startswith("Qdrant Cloud"):
                status = _index_status(st.session_state.index)
                st.info(
                    "Qdrant Cloud status: "
                    f"collection={status.get('collection_name', qdrant_collection)}, "
                    f"points={status.get('point_count', 'unknown')}, "
                    f"encoder={status.get('encoder_name', getattr(st.session_state.index, 'encoder_name', 'unknown'))}."
                )
                st.caption(
                    "If points=0, the earlier indexing run did not save vectors to this collection. "
                    "If points>0, confirm you are using the same collection and the same local embedding model."
                )
            if enable_eval_mode and eval_log_runs:
                st.session_state["latest_eval_artifacts"] = write_eval_run(
                    APP_DIR,
                    run_label=eval_run_label,
                    record=build_eval_record(
                        run_label=eval_run_label,
                        question=question,
                        effective_question=effective_question,
                        settings=_eval_settings_snapshot(
                            rag_mode=rag_mode,
                            search_backend=search_backend,
                            reranker_label=reranker_label,
                            rerank_top_n=rerank_top_n,
                            top_k=top_k,
                            token_budget=token_budget,
                            guardrail_level=guardrail_level,
                            page_render_scale=page_render_scale,
                            visual_embedding_batch_size=visual_embedding_batch_size,
                            light_compression_mode=light_compression_mode,
                            light_target_tokens=light_target_tokens,
                        ),
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
                            "total_s": total_seconds,
                        },
                        stepback_question=stepback_question,
                        used_api=False,
                        guardrail_details=guardrail_details,
                    ),
                )
            metric_cols = st.columns(4)
            metric_cols[0].metric("Retrieval", f"{retrieval_seconds:.2f}s")
            metric_cols[1].metric("Step-back", f"{stepback_seconds:.2f}s")
            metric_cols[2].metric("Answer", "0.00s")
            metric_cols[3].metric("Total", f"{total_seconds:.2f}s")
            return

        chunks_with_scores = [(result.chunk, result.score) for result in results]
        answer_seconds = 0.0
        final_answer_text = ""
        if use_api and api_key:
            with st.spinner(f"Generating answer with {MODEL_NAME}..."):
                try:
                    answer_start = time.perf_counter()
                    final_answer_text = generate_answer(effective_question, chunks_with_scores, api_key, token_budget=token_budget)
                    answer_seconds = time.perf_counter() - answer_start
                except Exception as exc:
                    st.error(f"Answer generation failed: {exc}")
                    st.info("Showing retrieved evidence instead.")
        else:
            st.info(
                "No API key found or retrieval-only mode selected, so I returned the "
                "most relevant evidence instead of generating an answer."
            )
        if not final_answer_text:
            final_answer_text = _evidence_only_answer(results)

        answer_payload = _build_answer_payload(final_answer_text, results)
        if guardrails is not None:
            post_generation = guardrails.run_post_generation(
                answer_payload=answer_payload,
                evidence=[result.chunk.text for result in results],
            )
            guardrail_details.update(post_generation.get("results", {}))
            if post_generation["action"] != "continue":
                answer_payload = post_generation["response"]
            else:
                answer_payload = post_generation["response"]

        st.markdown("### Answer")
        st.write(answer_payload["answer"])
        st.caption(f"Confidence: {answer_payload['confidence']}")
        if answer_payload.get("sources"):
            st.caption(
                "Sources: "
                + ", ".join(
                    f"{source['source_id']}" for source in answer_payload["sources"]
                )
            )
        if guardrail_details:
            with st.expander("Guardrail summary", expanded=False):
                st.json({name: result["status"] for name, result in guardrail_details.items()})

        total_seconds = time.perf_counter() - total_start
        if enable_eval_mode and eval_log_runs:
            st.session_state["latest_eval_artifacts"] = write_eval_run(
                APP_DIR,
                run_label=eval_run_label,
                record=build_eval_record(
                    run_label=eval_run_label,
                    question=question,
                    effective_question=effective_question,
                    settings=_eval_settings_snapshot(
                        rag_mode=rag_mode,
                        search_backend=search_backend,
                        reranker_label=reranker_label,
                        rerank_top_n=rerank_top_n,
                        top_k=top_k,
                        token_budget=token_budget,
                        guardrail_level=guardrail_level,
                        page_render_scale=page_render_scale,
                        visual_embedding_batch_size=visual_embedding_batch_size,
                        light_compression_mode=light_compression_mode,
                        light_target_tokens=light_target_tokens,
                    ),
                    answer_payload=answer_payload,
                    results=result_records,
                    latencies={
                        "retrieval_s": retrieval_seconds,
                        "stepback_s": stepback_seconds,
                        "answer_s": answer_seconds,
                        "total_s": total_seconds,
                    },
                    stepback_question=stepback_question,
                    used_api=bool(use_api and api_key),
                    guardrail_details=guardrail_details,
                ),
            )

        metric_cols = st.columns(4)
        metric_cols[0].metric("Retrieval", f"{retrieval_seconds:.2f}s")
        metric_cols[1].metric("Step-back", f"{stepback_seconds:.2f}s")
        metric_cols[2].metric("Answer", f"{answer_seconds:.2f}s")
        metric_cols[3].metric("Total", f"{total_seconds:.2f}s")

        st.markdown("### Evidence")
        for result in results:
            with st.expander(f"[{result.chunk.chunk_id}] {result.chunk.source_name} ({result.score:.3f})"):
                image_path = result.chunk.metadata.get("image_path")
                if image_path and Path(str(image_path)).exists():
                    st.image(str(image_path), caption=f"Rendered page preview for {result.chunk.chunk_id}", width=520)
                st.write(result.chunk.text)
                st.caption(
                    f"modality={result.chunk.modality} "
                    f"timestamp={result.chunk.timestamp or 'None'} "
                    f"frame={result.chunk.page_or_frame or 'None'}"
                )


def _result_records(results) -> list[dict]:
    return [
        {
            "chunk_id": result.chunk.chunk_id,
            "source_name": result.chunk.source_name,
            "text": result.chunk.text,
            "score": float(result.score),
        }
        for result in results
    ]


def _eval_settings_snapshot(
    *,
    rag_mode: RagModeSpec,
    search_backend: str,
    reranker_label: str,
    rerank_top_n: int,
    top_k: int,
    token_budget: int,
    guardrail_level: str,
    page_render_scale: float,
    visual_embedding_batch_size: int,
    light_compression_mode: str,
    light_target_tokens: int | None,
) -> dict:
    return {
        "rag_mode": rag_mode.label,
        "rag_family": rag_mode.family,
        "search_backend": search_backend,
        "reranker_label": reranker_label,
        "rerank_top_n": rerank_top_n,
        "top_k": top_k,
        "token_budget": token_budget,
        "guardrail_level": guardrail_level,
        "page_render_scale": page_render_scale,
        "visual_embedding_batch_size": visual_embedding_batch_size,
        "light_compression_mode": light_compression_mode,
        "light_target_tokens": light_target_tokens,
    }


def _build_answer_payload(answer: str, results) -> dict:
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
        "confidence": _confidence_from_results(results),
        "guardrail_summary": {},
    }


def _confidence_from_results(results) -> str:
    if not results:
        return "low"
    top_score = float(results[0].score)
    if top_score >= 0.5:
        return "high"
    if top_score >= 0.15:
        return "medium"
    return "low"


def _evidence_only_answer(results) -> str:
    snippets = []
    for result in results[:3]:
        snippets.append(f"- {text_preview(result.chunk.text, limit=160)} [{result.chunk.chunk_id}]")
    if not snippets:
        return "I could not find enough support in the indexed materials to answer reliably."
    return "Top retrieved evidence:\n" + "\n".join(snippets)


def _render_guardrail_result(response: dict, guardrail_details: dict) -> None:
    st.markdown("### Answer")
    st.write(response.get("answer", "I could not answer that safely from the indexed materials."))
    st.caption(f"Confidence: {response.get('confidence', 'low')}")
    if guardrail_details:
        with st.expander("Guardrail summary", expanded=False):
            st.json({name: result["status"] for name, result in guardrail_details.items()})


def _merge_results(primary, secondary, top_k: int):
    merged = {}
    for result in list(primary) + list(secondary):
        key = result.chunk.chunk_id
        if key not in merged or result.score > merged[key].score:
            merged[key] = result
    return sorted(merged.values(), key=lambda result: result.score, reverse=True)[:top_k]


def _render_workspace_summary(rag_mode: RagModeSpec, search_backend: str, provider: str) -> None:
    chunks = st.session_state.get("chunks", [])
    sources = {chunk.source_name for chunk in chunks}
    modalities = sorted({chunk.modality for chunk in chunks})
    index = st.session_state.get("index")
    index_label = "Ready" if index is not None else "Not built"

    cols = st.columns(4)
    cols[0].metric("Chunks", len(chunks))
    cols[1].metric("Sources", len(sources))
    cols[2].metric("Index", index_label)
    cols[3].metric("Retriever", rag_mode.label)
    if modalities:
        st.caption(
            "Indexed modalities: "
            + ", ".join(modalities)
            + f" | family={rag_mode.family} | engine={search_backend}"
        )
    else:
        st.caption(f"Answer mode: {provider}. Load content, then build an index.")


def _render_qdrant_banner(search_backend: str) -> None:
    banner = st.session_state.get("qdrant_banner")
    if not search_backend.startswith("Qdrant Cloud"):
        return
    if not banner:
        st.info(
            "Qdrant VectorDB mode is selected. When connected, this app can reuse vectors already stored in your cloud collection across restarts."
        )
        return

    message = (
        f"{banner['message']} "
        "Available through Qdrant VectorDB: persisted vectors, payload metadata, and cloud-backed retrieval reuse."
    )
    kind = banner.get("kind", "info")
    if kind == "success":
        st.success(message)
    elif kind == "warning":
        st.warning(message)
    elif kind == "error":
        st.error(message)
    else:
        st.info(message)


def _render_upload_summary(
    *,
    text_files: list[tuple[str, bytes]],
    csv_files: list[tuple[str, bytes]],
    pdf_files: list[tuple[str, bytes]],
    audio_files: list[tuple[str, bytes]],
    video_audio_files: list[tuple[str, bytes]],
    image_assets: list[tuple[str, bytes, str]],
    video_assets: list[tuple[str, object, str, str, str]],
) -> None:
    rows = [
        ("Text/Markdown", len(text_files)),
        ("CSV", len(csv_files)),
        ("PDF", len(pdf_files)),
        ("Audio", len(audio_files)),
        ("Video audio tracks", len(video_audio_files)),
        ("Images", len(image_assets)),
        ("Video frame segments", len(video_assets)),
    ]
    rows = [{"Input type": label, "Queued": count} for label, count in rows if count]
    if rows:
        with st.expander("Queued inputs", expanded=True):
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _qdrant_cloud_config(backend: str, url: str, api_key: str, collection_name: str) -> dict | None:
    if not backend.startswith("Qdrant Cloud"):
        return None
    return {
        "url": url,
        "api_key": api_key,
        "collection_name": collection_name,
    }


def _index_status(index) -> dict:
    if hasattr(index, "status"):
        try:
            return index.status()
        except Exception as exc:
            return {"status_error": str(exc), "encoder_name": getattr(index, "encoder_name", "")}
    return {"encoder_name": getattr(index, "encoder_name", "")}


if __name__ == "__main__":
    main()
