# Results Summary

This repository includes a completed formal evaluation campaign plus the intermediate development runs used to stabilize the system.

The goal of the benchmark suite is to show both retrieval quality and answer quality across text and document-image tasks, not just one-off best-case numbers.

## Formal Results

The formal report is saved in `eval/results/formal_full_system_report.csv`.

| Benchmark | Best setting | Result |
| --- | --- | --- |
| BEIR `nfcorpus` | `Text Hybrid + MiniLM-L4` | `hit@5 = 0.610`, `mrr@5 = 0.570` |
| BEIR `trec-covid` | `Text Hybrid + MiniLM-L4` | `hit@5 = 0.960`, `mrr@5 = 0.890` |
| DocVQA small retrieval | `Visual Page + Text Hybrid + MiniLM-L4` | `mrr@5 = 0.893` |
| DocVQA small answer quality | `Visual Page + Text Hybrid + MiniLM-L4` | `normalized_exact_match = 0.350`, `contains_expected = 0.750`, `token_f1 = 0.568`, `anls = 0.729` |

## Full Formal Run List

- `formal_beir_nfcorpus_text_l4_off_v1`
- `formal_beir_trec_covid_text_l4_off_v1`
- `formal_docvqa_small_text_off_v1`
- `formal_docvqa_small_text_answer_v1`
- `formal_docvqa_small_visual_hybrid_l4_off_v1`
- `formal_docvqa_small_visual_hybrid_l4_answer_v1`

## Main Conclusion

The system is stable for formal retrieval evaluation and ready for formal multimodal QA evaluation.
The visual-hybrid answer path is competitive with the text-only answer path on the DocVQA slice used in the formal suite, while the retrieval stack remains strong on both BEIR text benchmarks.

## Reproducibility

Run the formal suite:

```bash
python scripts/run_formal_full_system_experiment.py
```

Watch progress:

```bash
streamlit run scripts/formal_experiment_dashboard.py --server.port 9015
```

The full run archive is listed in the `Complete Benchmark Archive` section of `README.md` and is also preserved in `eval/results/all_runs_comparison.csv`.
