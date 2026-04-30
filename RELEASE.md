# Release Notes

## MultiModalRAG Formal Evaluation Release

This repository now includes a completed formal evaluation suite, a publishable README summary, and a full benchmark archive.

### What the project does
- Multimodal ingestion for text, PDF, image, audio, and video assets.
- Optional text, visual, and hybrid retrieval.
- Optional reranking, guardrails, Qdrant Cloud persistence, and vision-first answers.
- Grounded generation with `gpt-5-nano` when an OpenAI key is available.

### Formal results
- `BEIR nfcorpus`: `Text Hybrid + MiniLM-L4`, `hit@5 = 0.610`, `mrr@5 = 0.570`
- `BEIR trec-covid`: `Text Hybrid + MiniLM-L4`, `hit@5 = 0.960`, `mrr@5 = 0.890`
- `DocVQA small answer quality`: `Visual Page + Text Hybrid + MiniLM-L4`, `normalized_exact_match = 0.350`, `contains_expected = 0.750`, `token_f1 = 0.568`, `anls = 0.729`

### Full benchmark archive
- BEIR text retrieval runs: `beir_nfcorpus_*`, `beir_trec_covid_*`
- DocVQA retrieval runs: `docvqa_small_text_off_v1`, `docvqa_small_visual_hybrid_off_v1`, `docvqa_small_visual_hybrid_l4_off_v1`
- DocVQA answer-quality runs: `docvqa_small_text_answer_full_v1`, `docvqa_small_visual_hybrid_l4_answer_full_v1` through `v4`
- Formal runs: `formal_*`

### Important artifacts
- `eval/results/formal_full_system_report.csv`
- `eval/results/formal_full_system_comparison.csv`
- `eval/results/all_runs_comparison.csv`

### Reproduce
```bash
python scripts/run_formal_full_system_experiment.py
```

### Monitor
```bash
streamlit run scripts/formal_experiment_dashboard.py --server.port 9015
```
