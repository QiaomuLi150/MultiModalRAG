## Evaluation Scaffold

This folder holds the minimal assets needed to run repeatable experiments with
`scripts/run_eval.py` without changing the app codepath.

### Layout

- `questions/`
  - question files in `jsonl`, `csv`, or `txt`
- `manifests/`
  - storage-aware benchmark plans and subset definitions

### Recommended first runs

Text baseline on bundled sample data:

```bash
python3 scripts/run_eval.py \
  --questions-file eval/questions/sample_mixed_questions.jsonl \
  --use-sample-data \
  --eval-preset "Text Baseline" \
  --run-label sample_text_baseline \
  --log-runs
```

Hybrid baseline on a local corpus:

```bash
python3 scripts/run_eval.py \
  --questions-file eval/questions/sample_mixed_questions.jsonl \
  --input-dir /path/to/corpus \
  --eval-preset "Hybrid Default" \
  --run-label local_hybrid_default \
  --log-runs
```

Visual stress run on a local PDF/image corpus:

```bash
python3 scripts/run_eval.py \
  --questions-file eval/questions/visual_page_questions.jsonl \
  --input-dir /path/to/visual_corpus \
  --eval-preset "Visual Stress" \
  --run-label visual_stress_v1 \
  --log-runs
```

Qdrant Cloud reuse:

```bash
python3 scripts/run_eval.py \
  --questions-file eval/questions/sample_mixed_questions.jsonl \
  --eval-preset "Hybrid Default" \
  --search-backend "Qdrant Cloud hybrid" \
  --qdrant-cloud-url "$QDRANT_CLOUD_URL" \
  --qdrant-cloud-api-key "$QDRANT_CLOUD_API_KEY" \
  --qdrant-collection multimodal_eval_v1 \
  --run-label qdrant_hybrid_eval \
  --log-runs
```

### Prepared public subset

A real public subset is now prepared here:

- `eval/questions/public/beir_nfcorpus_test_100.jsonl`
- `eval/questions/public/beir_nfcorpus_test_100_qrels.jsonl`
- `eval/questions/public/beir_nfcorpus_test_100_corpus.csv`
- `eval/questions/public/beir_nfcorpus_test_100_manifest.json`

These files come from `BeIR/nfcorpus` test qrels and include the matching local
corpus subset needed for a first retrieval benchmark run.

Example:

```bash
python3 scripts/run_eval.py \
  --questions-file eval/questions/public/beir_nfcorpus_test_100.jsonl \
  --input-dir eval/questions/public \
  --eval-preset "Text Baseline" \
  --run-label beir_nfcorpus_text_v1 \
  --log-runs
```

### Question file schema

Preferred `jsonl` format:

```json
{"id":"q001","question":"What are the key roadmap decisions?","expected_modality":"text"}
```

Recommended fields:

- `id`
- `question`
- `expected_modality`
- `expected_source`
- `notes`

The runner only requires `question`. Everything else is preserved in the log.

### Storage-aware benchmark plan

Given the current Qdrant free-tier limits, start with small curated subsets:

- `BEIR`: 100 to 300 queries, text-only mode
- `DocVQA`: 100 to 200 queries, visual/hybrid mode
- `InfoVQA`: 50 to 100 queries, visual/hybrid mode
- `ChartQA`: 50 to 100 queries, visual/hybrid mode
- custom video/audio: 20 to 50 queries

Do not try to index full public visual corpora into one free-tier cloud
collection. Use local indexing first, and only promote compact validated subsets
to Qdrant Cloud.

### Output

When `--log-runs` is enabled, results are written to:

- `.artifacts/eval_logs/<run_label>.jsonl`
- `.artifacts/eval_logs/<run_label>_summary.csv`
- `.artifacts/eval_logs/<run_label>_<timestamp>.json`

To compare multiple logged runs:

```bash
python3 scripts/summarize_eval_runs.py
```

To restrict the comparison to specific runs:

```bash
python3 scripts/summarize_eval_runs.py \
  --run-label beir_nfcorpus_text_v1 \
  --run-label beir_trec_covid_text_v1
```

To visualize live experiment progress in Streamlit:

```bash
python3 -m streamlit run scripts/eval_dashboard.py --server.headless true --server.port 9003
```

### Current scope

This scaffold defines:

- runnable sample question sets
- benchmark subset manifests
- a consistent place to store public-benchmark-derived question files later

It does not download public datasets automatically.
