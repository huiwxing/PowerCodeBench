# code/ — method code and analysis scripts

This directory holds the paper's method code together with the analysis
and aggregation scripts that produced the frozen artifacts. The method
modules are repository copies of the core modules from the private
experiment repository the runs were executed in, adapted to this
repository's layout. Each file's header states its source path and
whether it was copied unmodified or received import/data-path
adaptations. All files compile with `python -m py_compile`.

Self-contained CPU modules run against the archived corpora, compact
item-level evidence, and aggregates in this repository. Serving-side
modules need a GPU stack (vLLM/torch) or provider API keys; they are
archived to document the frozen runs.

Three files are recorded by path and SHA-256 inside frozen artifacts under
`results/supplementary_evidence/`, as the inputs that produced them:
`task_demand/task_demand_model.py`, `task_demand/e4_conditional_eval.py`,
and `aggregate/export_s4_ce_reconstruction.py`. They stay byte-for-byte as
the exporters ran them, comments included, so that record keeps holding.

To use the method code as a package, run from the repository root with
`PYTHONPATH=code`, e.g.:

```bash
PYTHONPATH=code python3 -c "from probing.probe_framework import ProbeGenerator"
```

## Module map (paper component → files)

| Paper component | Directory / files | Runs here? |
|---|---|---|
| Benchmark generator (§ benchmark construction; SM S1): network registry (40 cases), task templates, modification operators, NL query templating, and the construction-time admission/feasibility filtering (convergence + finite-scalar checks, D3/D4 semantic-target validation, exact-match reference-code re-execution) | `benchmark_generator/benchmark_config.py`, `benchmark_generator/benchmark_engine.py`, `benchmark_generator/expanded_nl_templates.json` | Yes (CPU, needs `pandapower` from the pinned environment) |
| Knowledge-boundary probing (L0–L3; § probing): probe generation from the standardized API-spec JSON, evaluation, knowledge-profile construction, and injection-snippet generation | `probing/probe_framework.py` with `dataset/pandapower_docs.json` (spec), `dataset/all_probes.json` (frozen probe set) | Yes (CPU) |
| Raw probe-record reconstruction (S2/S3): 29,120 response/evaluator records, byte-exact apart from the audited diagnostic-path normalization (`audit/README.md`); source and release provenance are validated, and historical profile aggregation is compared field by field for 14 models | `aggregate/recompute_probe_transcripts.py`; `results/raw/probe_transcripts/manifest.json`; `audit/probe_diagnostic_path_normalization.json` | Yes (standard library; no inference or L3 re-execution) |
| Task-demand model (§ demand modeling; E4): query-only API-primitive demand predictor trained on `dataset/augmented_dataset.json` | `task_demand/task_demand_model.py`; frozen demand-suite export consumed by the runner under `results/demand_suites/` | Yes (CPU, needs scikit-learn) |
| S4 demand evidence: exact unfiltered item rankings for three TF-IDF estimators, pinned-snapshot Zero-shot SBERT and Zero-shot Cross-Encoder, plus the complete Hybrid before/after tradeoff. The Hybrid Cross-Encoder entry is a separately labelled deterministic rerun on the pinned stack, described in `results/supplementary_evidence/` | `aggregate/export_s4_item_evidence.py`, `aggregate/extend_s4_sbert_evidence.py`, `aggregate/export_s4_ce_reconstruction.py`, `aggregate/extend_s4_cross_encoder_evidence.py`, `aggregate/aggregate_s4_item_evidence.py`; `results/supplementary_evidence/s4_item_evidence.json` | Public reaggregation is standard-library only; fresh neural recovery uses the pinned model/software revisions and CPU (SBERT) or GPU (Cross-Encoder) |
| E4 conditional-adaptation evaluation. The five-seed item rankings are deterministic reruns on the pinned stack, described in `results/supplementary_evidence/` | `task_demand/e4_conditional_eval.py`, `task_demand/e4_per_role_v2.py`, `task_demand/summarize_e4_evals.py`; five-seed item-ranking reconstruction in `aggregate/export_e4_multiseed_predictions.py`; public item evidence under `results/supplementary_evidence/` | Numerical tables and five-seed sensitivity rebuild here; fresh training needs the pinned scikit-learn stack |
| Intervention: proactive injection (conditions B/C/X), reactive fix-round injection (FX/FD/FDR/FS/FDRS), semantic trace-grounding, RAG baselines (C2/E5), and the derived library-knowledge chain (`dataset/pandapower_docs.json` → `dataset/pandapower_library_knowledge.json`) | `intervention/` (see `intervention/README.md`) | Yes (CPU) |
| Backend + orchestrator: model serving (vLLM/HF/API), the probe + benchmark runner (all conditions), execution sandbox, output parsing, prompt construction (the verbatim templates in `docs/PROMPTS.md` live in `backend/utils.py`) | `backend/llm_backend.py`, `backend/probe_runner.py`, `backend/utils.py` | `utils.py` yes (CPU); runner/backend require GPU serving stack or API keys |
| E2 transfer mini-bench construction with explicit admission filters (n_bus ≥ 3, converged/optimal + finite ground truth), C-recal paired aggregation, and D-6 pre/post import scan | `build/build_e2_opendss_bench.py`, `build/build_e2_pypsa_bench.py`, `aggregate/aggregate_e2_crecal.py`, `audit/recompute_e2_import_scans.py` | Builders need `opendssdirect` / `pypsa`; C-recal and the 3,240-generation import audit are standard-library CPU-only |
| C3 cross-style holdout, difficulty splits, and naturalistic holdout construction. The published C3 item rankings are deterministic reruns on the pinned stack, described in `results/supplementary_evidence/` | `aggregate/export_c3_deterministic_rerun.py`, `aggregate/aggregate_c3_deterministic_rerun.py`, `aggregate/aggregate_historical_demand_run_logs.py`, `build/split_benchmark_by_difficulty.py`, `build/build_naturalistic_holdout.py` | Public C3 item/paired-bootstrap reaggregation and historical-log verification are standard-library only; a fresh rerun uses the pinned scikit-learn stack |
| Composition figure (README / benchmark stats): counts cross-checked against `benchmark/composition_stats.json`, network sizes measured through the release's own registry and loader | `build/plot_benchmark_stats.py` | Yes (CPU, needs `pandapower` + `matplotlib`) |
| SM robustness figure (4 panels: risk-weight grid, seed variance, scenario adaptation, budget-matched dense retrieval), rebuilt from the frozen aggregates with per-panel cross-checks | `build/plot_robustness_panel.py` | Yes (CPU, needs `matplotlib`) |
| SM token-economy figure (pooled-exact T6 diamonds, per-model cells, proactive/reactive saving decompositions) | `build/plot_token_economy.py` | Yes (CPU, needs `matplotlib`; writes `assets/fig_token_economy.pdf`) |
| Knowledge-profile heatmap and six-estimator demand recall@k curves (210 displayed data points, cross-checked against raw compact evidence) | `build/plot_profile_demand_evidence.py` | Yes (CPU, needs `matplotlib`; writes two PDFs under `assets/`) |
| Cross-vendor contribution bars, per-task A/FDRS heatmap, and model-scale endpoint scatter | `build/plot_result_breakdowns.py` | Yes (CPU, needs `matplotlib`; checks every plotted cell against `results/aggregates/primary_results_from_compact.json`) |
| Reactive-router event shares and proactive prompt-component accounting | `aggregate/aggregate_reactive_router_shares.py`, `aggregate/aggregate_proactive_prompt_components.py` | Yes (standard library; rebuilds 68,107 item-round events and 20,000 aligned prompt-accounting records) |
| E3 serving-table builders and full raw-measurement verifier | `build/update_e3_serving_table_next.py`, `build/update_e3_serving_table_480b_final.py`, `aggregate/recompute_serving_tables.py` | Yes (standard library against 249 archived Protocol-A/B input files) |
| Engineering-validity audit (E1/E1-P/T4), matcher-FN scope and census audit (`audit/matcher_fn/`), Failure Anatomy (`audit/failure_taxonomy/`), and API-parity outcome audit | `audit/`, `audit/recompute_failure_taxonomy_revision.py`, `aggregate/aggregate_p02_parity.py` | Recomputes by default from the archived ledgers and compact outcomes; Failure Anatomy also replays its 100-case draw from the released sampling frame; simulator re-execution stays environment-dependent |
| Release diagnostic-path normalization: private environment prefixes are replaced in diagnostic string fields, leaving outcomes, tokens, and every other non-text field as recorded (`audit/README.md`) | `audit/normalize_primary_diagnostic_paths.py` | Yes (standard library; idempotent and independently audited) |
| Aggregators behind the frozen result JSONs (primary T3--T6, probes/demand/router, transfer, robustness, bootstrap CIs, contamination, …) | `aggregate/` | CPU rebuilds by default; `reproduce.py` runs the aggregators behind the reported numbers |

## Licence

Code in this directory (and `reproduce.py` at the repository root) is
licensed under the Apache License 2.0 (`code/LICENSE`), separately from
the CC BY 4.0 data/documentation licence at the repository root; see the
"Licences" section of the root `README.md`.
