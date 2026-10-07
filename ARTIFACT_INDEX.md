# PowerCodeBench — Artifact Index

This repository is the official **PowerCodeBench** distribution and the code-and-data
artifact for the paper *"Knowledge Boundary Probing and Demand-Guided Intervention
for LLM-Based Power System Code Generation"* (Advanced Engineering Informatics,
77, 105328, 2027; [DOI: 10.1016/j.aei.2026.105328](https://doi.org/10.1016/j.aei.2026.105328)).
This index maps the paper's quantitative claims to the frozen files that carry
them: main-text Tables T2–T12, Supplementary Tables S1–S19, the published figure
evidence, and the headline prose numbers. Each row gives the artifact and the
script that generated it. Additional repository plots are labelled separately.

Verification entry point (CPU-only, Python 3 standard library):

```
python3 reproduce.py
```

recomputes the paper's headline number groups from the frozen artifacts and prints
a PASS/FAIL table against the expected values recorded in this index. See the
README for the repository tour.

Provenance: the artifacts are the frozen outputs of the paper's experimental
pipeline. Frozen source outputs are byte-exact copies, apart from the documented
absolute-path normalization. Compact migration artifacts and recomputed summaries
name their source commits, hashes, and canonical generators. Files under `code/`
have path adaptations to this repository's layout, noted in each file header.

Text normalization note: in the public `benchmark.json`, 105 of the 2,000
`natural_language_query` strings have U+2019 (right single quotation mark)
normalized to the ASCII apostrophe. The frozen pipeline inputs, for example
`benchmark/c3_splits/`, retain the original character, and all reported numbers
were computed on those frozen versions. CI checks that this is the only difference
and that it covers exactly those 105 items
(`.github/scripts/check_text_normalization.py`).

## Main table — claim → artifact → generating script

The mappings below follow the published article and supplementary material.
The frozen evidence and its generating scripts retain their recorded provenance.

### Main-text tables (T2--T11)

| Claim / Table | Artifact | Generating script |
|---|---|---|
| T3 cross-vendor headline on the full 2,000-item benchmark (C+FDRS endpoints; hero cell Llama-405B 68.7% vs Claude-Haiku-4.5 63.75%, paired bootstrap delta +4.95pp CI [2.95, 6.95]) | `results/aggregates/hero_bootstrap_ci.json` | `code/aggregate/bootstrap_hero_ci.py` |
| &nbsp;&nbsp;the two C+FDRS endpoints entering the hero comparison (Llama-405B 0.687, Claude-Haiku-4.5 0.6375) | `results/aggregates/api_comparison_2000/reactive_comparison.json` | evaluation-harness freeze; independently rebuilt from the compact item outcomes by `code/aggregate/aggregate_primary_results_compact.py` |
| &nbsp;&nbsp;matching A/B/C/X endpoints of the same 15-model panel | `results/aggregates/api_comparison_2000/proactive_comparison.json` | 〃 |
| &nbsp;&nbsp;shared 2,000-item axis of the API arms (merged from the 505- and 1,495-item subset manifests) | `results/aggregates/api_comparison_2000/subset_items.json` | evaluation-harness freeze |
| Cross-vendor no-injection baseline aggregate over the same 15-model 2,000-item panel, with per-task / per-difficulty / per-network / error-class breakdowns. Its per-model accuracies are condition A for the four APIs and A_FX (baseline plus fix rounds) for the eleven open-weight runs; the C+FDRS endpoints and the T3 hero number come from the reactive file above | `results/aggregates/api_comparison_2000/benchmark_comparison.json` | evaluation-harness freeze |
| T3 70B/120B nearest-API paired bootstrap + TOST (e.g., GPT-OSS-120B vs GPT-5.4-mini -1.90pp, CI [-3.90, +0.15], equivalent within +/-5pp). reproduce.py check 12 | `results/aggregates/p02_parity_tost.json` | `code/aggregate/aggregate_p02_parity.py` |
| &nbsp;&nbsp;five aligned 2,000-item match vectors + source provenance | `audit/p02_parity_item_outcomes.json` | compact export verified/rebuilt by `code/aggregate/aggregate_p02_parity.py` |
| T4 proactive main effects, open-weight panel (conditions A/B/C/X) | `results/aggregates/comparison/proactive_comparison.json` | evaluation-harness freeze; independently rebuilt by `code/aggregate/aggregate_primary_results_compact.py` |
| &nbsp;&nbsp;matching no-injection baseline aggregate for the same eleven-model panel (A_FX accuracies plus per-task / per-difficulty / per-network / error-class breakdowns) | `results/aggregates/comparison/benchmark_comparison.json` | 〃 |
| &nbsp;&nbsp;companion L0--L3 probe-profile x benchmark cross-table for the same panel | `results/aggregates/comparison/comparison_table.json` | 〃 |
| T4 prose headline: layer-wise C delivers +30.95pp panel-mean gain over baseline A, 95% paired-bootstrap CI [+29.75, +32.13]pp (B=2,000 item-level resamples, seed 22, 10-model panel; "C - A" row of the frozen file). reproduce.py check 10 | `results/aggregates/mw2_panel_mean_paired_bootstrap_ci.json` | `code/aggregate/aggregate_mw2_paired_bootstrap.py` |
| T5 reactive master (C_FX / C_FD / C_FDR chains + round snapshots; C_FDRS in the API file) | `results/aggregates/comparison/reactive_comparison.json` | evaluation-harness freeze; independently rebuilt by `code/aggregate/aggregate_primary_results_compact.py` |
| &nbsp;&nbsp;〃 | `results/aggregates/api_comparison_2000/reactive_comparison.json` | 〃 |
| T6 unified token economy; headline: full proactive C reaches parity at 41% of X's prompt tokens (Abstract; Highlights). reproduce.py checks 2 and 13 | `results/aggregates/token_economy.json` | `code/aggregate/aggregate_token_economy.py`, cross-rebuilt from compact token totals by `code/aggregate/aggregate_primary_results_compact.py` |
| T3--T6 complete public rebuild: 270 compact runs, shared 2,000/80-item axes, match/execute vectors, tokens, round snapshots, hero/MW2 bootstraps, R/Rsem/RsemB controls, and pooled-exact T6 display | `results/raw/main_experiment/primary_outcomes_compact.json` | deterministic minimum-sufficient export; diagnostic-only private paths normalized and audited by `code/audit/normalize_primary_diagnostic_paths.py`; rebuilt by `code/aggregate/aggregate_primary_results_compact.py` |
| &nbsp;&nbsp;〃 | `results/aggregates/primary_results_from_compact.json` | `code/aggregate/aggregate_primary_results_compact.py` (`reproduce.py --group 13`) |
| &nbsp;&nbsp;diagnostic-path normalization: 215 dictionary entries / 1,662 item occurrences, with all non-text evidence exact (see `audit/README.md`) | `audit/primary_diagnostic_path_normalization.json` | `code/audit/normalize_primary_diagnostic_paths.py` (`reproduce.py --group 25`) |
| T6 prose prompt decomposition: 20,000 aligned C items, pooled means 182.173 base + 767.793 documentation + 500.939 framing residual = 1450.906 total | `results/supplementary_evidence/proactive_prompt_components.json` | source export and release aggregation by `code/aggregate/aggregate_proactive_prompt_components.py` (`reproduce.py --group 25`) |
| &nbsp;&nbsp;〃 | `results/aggregates/proactive_prompt_components.json` | `code/aggregate/aggregate_proactive_prompt_components.py` |
| T7 / S5 / S6 external-query conditional adaptation, E4 v2 four sets (layer1_holdout_n80, layer2a_n84, layer2a_ext_n10, layer2b_n20) incl. exit-a margin criterion. reproduce.py check 6 | `external_queries/conditional_selector/metrics_by_set_v2.json` | `code/task_demand/e4_conditional_eval.py`, `code/task_demand/e4_per_role_v2.py` |
| &nbsp;&nbsp;〃 | `external_queries/conditional_selector/metrics_by_set.json` | 〃 |
| &nbsp;&nbsp;〃 | `external_queries/conditional_selector/per_role_v2.json` | 〃 |
| T7 seed-sensitivity companion (train seeds 42-46). reproduce.py check 6 | `external_queries/conditional_selector/seed_sensitivity/metrics_by_set_seed42.json` | `code/task_demand/e4_conditional_eval.py --train-seed` |
| &nbsp;&nbsp;〃 | `external_queries/conditional_selector/seed_sensitivity/metrics_by_set_seed43.json` | 〃 |
| &nbsp;&nbsp;〃 | `external_queries/conditional_selector/seed_sensitivity/metrics_by_set_seed44.json` | 〃 |
| &nbsp;&nbsp;〃 | `external_queries/conditional_selector/seed_sensitivity/metrics_by_set_seed45.json` | 〃 |
| &nbsp;&nbsp;〃 | `external_queries/conditional_selector/seed_sensitivity/metrics_by_set_seed46.json` | 〃 |
| &nbsp;&nbsp;five-seed item-level reconstruction: 5 seeds x 194 queries x 2 independent top-20 arms; 5,948 public checks against every archived metric, alpha, and role weight. reproduce.py check 22 | `results/supplementary_evidence/e4_multiseed_predictions.json` | deterministic reconstruction by `code/aggregate/export_e4_multiseed_predictions.py`; release recomputation by `code/aggregate/aggregate_e4_multiseed.py` |
| &nbsp;&nbsp;recomputed five-seed summaries and validation report | `results/aggregates/e4_multiseed_recomputed.json` | `code/aggregate/aggregate_e4_multiseed.py` |
| T8 reasoning-tier comparison (200-item subset; e.g. DeepSeek-Reasoner 56.00, +44.00) | `results/aggregates/round5_exp_c_reasoning_summary.json` | `code/aggregate/aggregate_round5_exp_c_reasoning.py` |
| &nbsp;&nbsp;〃 | `results/aggregates/api_comparison_reasoning_200/subset_items.json` | 〃 |
| T9 / S18 / S19 backend-transfer minibench: E2 24-cell matrix (2 backends x 3 models x A/Rsem/RsemB/C) + OpenDSS C-recal pairing. reproduce.py check 4 | `results/aggregates/e2_bench_v2_aggregate.json` | `code/aggregate/aggregate_e2_bench.py` |
| &nbsp;&nbsp;〃 | `results/aggregates/e2_opendss_bench_v2/proactive_comparison.json` | 〃 |
| &nbsp;&nbsp;〃 | `results/aggregates/e2_opendss_bench_v2/benchmark_comparison.json` | 〃 |
| &nbsp;&nbsp;〃 | `results/aggregates/e2_pypsa_bench_v2/proactive_comparison.json` | 〃 |
| &nbsp;&nbsp;〃 | `results/aggregates/e2_pypsa_bench_v2/benchmark_comparison.json` | 〃 |
| &nbsp;&nbsp;〃 | `results/aggregates/e2_opendss_bench_v2_crecal/proactive_comparison.json` | 〃 |
| &nbsp;&nbsp;C-recal paired CIs, exact McNemar tests, 27--29% total-prompt ratio, and control-device mechanism counts from nine per-item runs | `results/aggregates/e2_crecal_paired.json` | `code/aggregate/aggregate_e2_crecal.py` |
| &nbsp;&nbsp;full raw-run migration and arm provenance (24 cold-start + 9 C-recal cells; source commits and per-file SHA-256). reproduce.py check 15 | `results/raw/e2_transfer/manifest.json` | manifest verified by `code/aggregate/aggregate_e2_transfer.py` |
| &nbsp;&nbsp;24-cell + C-recal full recomputation | `results/aggregates/e2_transfer_full_recomputed.json` | `code/aggregate/aggregate_e2_transfer.py` |
| T11 / S16 / S17 serving metrics; 480B anchor composite convention (169,068 tok / 149.316 s = 1,132.3 tok/s; energy = frozen probe-window values); Qwen3-Coder-Next row (46.7% @ 5,568 tok/s, 238 J/item, 0.51 kJ/success; C+FDRS 57.7% @ 2,301 J/item, E ratio 9.7x). reproduce.py checks 7 and 9 | `serving/measurement_json/e3_serving_table.json` | `code/build/update_e3_serving_table_480b_final.py`, `code/build/update_e3_serving_table_next.py` |
| &nbsp;&nbsp;complete public rebuild of all 20 Protocol-A cells / 56 repetitions and all 15 Protocol-B cells; 7,306 raw/frozen/manuscript leaf checks, zero differences. reproduce.py check 24 | `results/aggregates/e3_serving_tables_recomputed.json` | `code/aggregate/recompute_serving_tables.py` from 249 public raw input files |
| R (BM25 RAG) baseline rows of T4/T6 | `results/aggregates/comparison_bm25rag/proactive_comparison.json` | `code/aggregate/aggregate_c2_bm25rag.py` |
| T10 matcher mechanism diagnostic, a fresh readjudication of the same restricted A/C-family n=100 sample: 8 confirmed parser-extraction misses, 3 task-specification ambiguities, and 89 genuine errors (nominal Wilson 4.1--15.0%). Two practising power-systems engineers ran two independent primary passes plus an isolated blind re-check, followed by an adversarial challenge pass and joint reconciliation. The rates describe this sample frame. See `audit/matcher_fn/`. reproduce.py check 11 | `audit/matcher_fn/revision_readjudication/adjudication_v2.json` | `code/audit/recompute_matcher_fn_readjudication.py` |
| &nbsp;&nbsp;100 complete source records with source commit/blob/hash provenance; their labels come from the readjudication ledger above | `audit/matcher_fn/sample_records.json` | deterministic source export in `code/audit/recompute_matcher_fn_audit.py` |
| &nbsp;&nbsp;historical aggregate validation plus de-duplicated 196-cell / 392,000-output census (66,810 executed-but-unmatched) | `audit/matcher_fn/recomputed_summary.json` | `code/audit/recompute_matcher_fn_audit.py` |
| &nbsp;&nbsp;historical adjudication aggregate (8/5/87; the recomputed unique census above replaces its duplicated pool census) | `audit/matcher_fn/P01_false_negative_audit.json` | frozen aggregate; scope/arithmetic validated by `code/audit/recompute_matcher_fn_audit.py` |
| T10 / S13 gate operating characteristics of the five-check automatic layer (pooled TP=129 FP=170 FN=14 TN=259) + tolerance-tightening panel. reproduce.py check 8 | `audit/t4_final_labels_v2.json` | `code/audit/run_t4.py` (checkers: `code/audit/checkers.py`) |
| &nbsp;&nbsp;〃 | `audit/t4_auto_results.json` | 〃 |
| &nbsp;&nbsp;〃 | `audit/t4_tolerance_sensitivity.json` | 〃 |
| &nbsp;&nbsp;〃 | `audit/t4_far_weighted.json` | 〃 |
| T10 / S13--S14 complete 610-case rebuild: point/Wilson estimates, 41 frozen bootstrap/contrast fields, gate/tolerance panels, exact 11-model tier mapping, and the 2,106-item common-support standardisation. reproduce.py check 18 | `audit/validity_tables_recomputed.json` | `code/audit/recompute_validity_tables.py` |
| T10 / S14 design-weighted FAR + stratified bootstrap + common support (main_P 22.4% unweighted -> 18.51% weighted [15.13, 21.95]; tightened 11.96% is in S13(b)). reproduce.py check 8 | `audit/t4_far_weighted.json` | `code/audit/run_t4.py` |
| &nbsp;&nbsp;〃 | `audit/e1_sample_manifest.json` | 〃 |
| &nbsp;&nbsp;〃 | `audit/t4_adjudication.json` | 〃 |
| &nbsp;&nbsp;〃 | `audit/t4_census2.json` | 〃 |
| &nbsp;&nbsp;〃 | `audit/t4_refute.json` | 〃 |
| T10 / S15 E1-P paired McNemar, 170 pairs (A 34.12% vs C 12.35% contaminated; delta +21.76pp CI [14.71, 28.82]; exact p = 9.33e-09) + tier breakdown. reproduce.py check 3 | `audit/e1p_mcnemar.json` | `code/audit/pair_mcnemar.py` (sampler: `code/audit/pair_sampler.py`) |
| &nbsp;&nbsp;〃 | `audit/e1p_pair_manifest.json` | 〃 |
| &nbsp;&nbsp;〃 | `audit/e1p_auto_results.json` | 〃 |
| &nbsp;&nbsp;〃 | `audit/e1p_adjudication.json` | 〃 |
| &nbsp;&nbsp;〃 | `audit/e1p_census.json` | 〃 |
| &nbsp;&nbsp;〃 | `audit/e1p_refute.json` | 〃 |
| &nbsp;&nbsp;〃 | `audit/e1p_ext_manifest.json` | 〃 |
| &nbsp;&nbsp;〃 | `audit/e1p_ext_auto_results.json` | 〃 |
| &nbsp;&nbsp;〃 | `audit/e1p_ext_adjudication.json` | 〃 |
| &nbsp;&nbsp;〃 | `audit/e1p_ext_census.json` | 〃 |
| &nbsp;&nbsp;〃 | `audit/e1p_ext_refute.json` | 〃 |
| S7 item-defect errata: the 38 audit records ruled `item-defect`, resolved through the sampling manifest to the 31 distinct benchmark items behind them, plus the exclusion sensitivity of every archived accuracy cell (226 cells, denominator 1,969; largest shift -0.531pp at GPT-OSS-120B C+FD) and the Table 3(b) TOST decisions under both denominators (all unchanged) | `audit/item_defect_errata.json` | `code/audit/item_defect_errata.py` |
| T2 / S1 benchmark protocol, composition and difficulty tiers (2,000 items) | `benchmark/composition_stats.json` | benchmark builder (`code/benchmark_generator/`); distribution copy = `benchmark.json` at repo root |
| E2 minibench query sets (OpenDSS / PyPSA, 90 items each) | `benchmark/e2_opendss/e2_opendss_bench.json` | benchmark builder (`code/benchmark_generator/`) |
| &nbsp;&nbsp;〃 | `benchmark/e2_pypsa/e2_pypsa_bench.json` | 〃 |
| &nbsp;&nbsp;T9 instrument freeze for those two minibenches: the 2026-07-25 B2 freeze plus the 2026-07-26 B1prime class-1 re-freeze. It records per-artifact SHA-256 and byte counts for each backend's documentation spec, probe set, injection snippets, minibench + ground truth, demand export and three per-model knowledge profiles, together with the frozen backend-library/matcher environment. Its scope is the two minibenches | `benchmark/e2_freeze_manifest.json` | hand-maintained E2 freeze ledger (protocol documented in `transfer/DEFECT_LEDGER.md` and `docs/MIGRATION.md`) |
| T9 transfer defect ledger + import scans (360/360 pre-fix; 0/1170, 629/630, 540/540 post-fix). reproduce.py check 23 | `transfer/import_scan_prefix_B2.json` | public rescan: `code/audit/recompute_e2_import_scans.py`; ledger: `transfer/DEFECT_LEDGER.md` |
| &nbsp;&nbsp;〃 | `transfer/import_scan_v2_postfix.json` | `code/audit/recompute_e2_import_scans.py` |
| &nbsp;&nbsp;all 1,080 complete pre-fix generated programs (the post-fix 2,160 are the full public E2 raw runs) | `audit/e2_import_scan/pre_fix_generations.json` | source export and public field-by-field rescan by `code/audit/recompute_e2_import_scans.py` |
| &nbsp;&nbsp;3,240-generation pre/post cross-check and exact validation against the frozen report | `results/aggregates/e2_import_scans/crosscheck.json` | `code/audit/recompute_e2_import_scans.py` |
### Supplementary tables (S2--S19)

| Claim / Table | Artifact | Generating script |
|---|---|---|
| Tables S2–S3 complete L0--L3 evidence (14 models x 275 functions) and all 56 displayed profile means. reproduce.py check 14 | `results/supplementary_evidence/probe_profiles.json` | compact per-function export; `code/aggregate/aggregate_probe_profiles.py` |
| &nbsp;&nbsp;29,120 probe response/evaluator records (14 models x 2,080), copied unchanged apart from the documented diagnostic-path normalization. The manifest pins source Git blobs plus source/release SHA-256 and byte counts, and the rebuild checks all 14 x 275 L0--L3 profiles field-for-field. reproduce.py check 28 | `results/raw/probe_transcripts/manifest.json` | `code/aggregate/recompute_probe_transcripts.py` |
| &nbsp;&nbsp;path-normalization audit: 4,900 occurrences in 2,938 diagnostic strings across 2,435 records and 14 files. All 29,120 raw responses and scores, 21,378 `correct` fields, and 528,168 other scalar leaves remain exact; the normalization is idempotent and the release carries only normalized paths. reproduce.py check 28 | `audit/probe_diagnostic_path_normalization.json` | `code/aggregate/recompute_probe_transcripts.py` |
| &nbsp;&nbsp;recomputed profile and provenance cross-check | `results/aggregates/probe_transcripts_recomputed.json` | 〃 |
| S4 demand estimators/recall curves and the deployed Hybrid reweighting comparison (0.5222 to 0.4708 Aug-test; 0.5115 to 0.7184 Bench-ref). reproduce.py check 14 | `results/aggregates/supplementary_demand_evidence.json` | `code/aggregate/aggregate_demand_evidence.py` from the frozen suite metrics and candidate exports |
| &nbsp;&nbsp;S4 six-estimator frozen metric source, including the historical Hybrid alpha curves (Cross-Encoder selected alpha 0.3) | `results/supplementary_evidence/demand_estimators.json` | `code/aggregate/aggregate_demand_evidence.py --export-estimators` |
| &nbsp;&nbsp;S4 item evidence: exact unfiltered top-20 reconstructions for all three TF-IDF estimators, Zero-shot SBERT, and Zero-shot Cross-Encoder on Aug-test/Aug-orig/Bench-ref, plus the Hybrid before/after tradeoff (-5.14/-16.81/+20.69pp). Both zero-shot neural reconstructions use pinned model revisions and match all seven frozen slices at every reported k. The Hybrid Cross-Encoder rankings are kept separately, as a deterministic rerun on the pinned stack that leaves the conclusions unchanged (R@10 drift +2.83/+4.50/+1.17pp); the job-log ledger shows the historical Hybrid model and hash state were not saved, so see `results/supplementary_evidence/README.md`. Six historical role-filtered deployment top-10 exports are listed on their own. reproduce.py check 26 | `results/supplementary_evidence/s4_item_evidence.json` | locked-stack/pinned-snapshot recovery by `code/aggregate/export_s4_item_evidence.py`, `code/aggregate/extend_s4_sbert_evidence.py`, `code/aggregate/export_s4_ce_reconstruction.py`, and `code/aggregate/extend_s4_cross_encoder_evidence.py`; public recomputation by `code/aggregate/aggregate_s4_item_evidence.py` |
| &nbsp;&nbsp;S4 item-evidence recomputation, which keeps the exact reconstructions and the rerun apart, validates the frozen metrics and alphas independently, and ledgers the remaining exact coverage for Hybrid Cross-Encoder | `results/aggregates/s4_item_evidence_recomputed.json` | `code/aggregate/aggregate_s4_item_evidence.py` |
| S11 reactive-router destination shares: 68,107 real item-round events over 10 models x C+FDR/C+FDRS; all 20 source route summaries rebuild exactly. reproduce.py checks 14 and 25 | `results/supplementary_evidence/reactive_router_counts.json` | source export and event aggregation by `code/aggregate/aggregate_reactive_router_shares.py` |
| &nbsp;&nbsp;〃 | `results/aggregates/supplementary_reactive_router_shares.json` | `code/aggregate/aggregate_reactive_router_shares.py` |
| S19 OpenDSS per-probe L0--L3 profiles and control-device token/error mechanism. reproduce.py check 19 | `results/aggregates/e2_opendss_profiles_recomputed.json` | `code/aggregate/aggregate_e2_opendss_profiles.py` from 18 public profile/probe files plus transfer item runs |
| S8 reactive pairwise deltas on base C (panel-mean paired bootstrap; C+FDR - C+FD CI [-0.44, +0.32]pp) | `results/aggregates/mw2_panel_mean_paired_bootstrap_ci.json` | `code/aggregate/aggregate_mw2_paired_bootstrap.py` |
| S9 per-task / per-difficulty compact matrices | `results/full_matrices/pertask_C_R0.csv` | pipeline freeze export from the frozen run trees |
| &nbsp;&nbsp;〃 | `results/full_matrices/perdiff_C_R0.csv` | 〃 |
| &nbsp;&nbsp;〃 | `results/full_matrices/perdiff_R3.csv` | 〃 |
| S10 naturalistic-holdout (n=80) holdout-vs-PCB per-model comparison | `results/aggregates/naturalistic_holdout/reactive_comparison.json` | evaluation-harness freeze; item-level holdout outcomes rebuilt by `code/aggregate/aggregate_primary_results_compact.py` |
| &nbsp;&nbsp;〃 | `benchmark/naturalistic_holdout/naturalistic_holdout_seed22_n80.json` | 〃 |
| &nbsp;&nbsp;〃 | `benchmark/naturalistic_holdout/naturalistic_holdout_seed22_n80.manifest.json` | 〃 |
| &nbsp;&nbsp;〃 | `benchmark/naturalistic_holdout/source_answer_key_seed22_n80.json` | 〃 |
| S16 serving protocol A raw measurement cells (per model/condition/rep: aggregate, run_meta, per-node energy) | `serving/measurement_json/protocolA/Qwen_Qwen3-Coder-480B-A35B-Instruct/C/rep1/aggregate.json` | serving probe harness; full trees under `serving/measurement_json/protocol{A,A_anchor_retest,B}/` |
| &nbsp;&nbsp;〃 | `serving/measurement_json/protocolA_anchor_retest/Qwen_Qwen3-Coder-480B-A35B-Instruct/C/rep1/aggregate.json` | 〃 |
| &nbsp;&nbsp;〃 (Qwen3-Coder-Next cells, jobs 5876965/5876967/5879735; reproduce.py check 9) | `serving/measurement_json/protocolA/Qwen_Qwen3-Coder-Next/C/rep1/aggregate.json` | 〃 |
| &nbsp;&nbsp;〃 | `serving/measurement_json/protocolB/Qwen_Qwen3-Coder-Next/cell_summary.json` | 〃 |
| S12 model identifiers / frozen software environment | `environment/environment.yaml` | pip/conda freeze of the experiment venv |
| &nbsp;&nbsp;〃 | `environment/requirements.lock.txt` | 〃 |
### Figures (published figure mappings and additional repository plots)

The published supplementary material contains Fig. S1 (the benchmark pipeline)
and Fig. S2 (four robustness panels). The risk-weight, seed, unadapted-demand,
and dense-retrieval comparisons below map to Fig. S2(a–d), respectively.
The standalone recall curves and full token-economy rendering are additional
repository plots, not numbered figures in the published supplementary material.

| Claim / Table | Artifact | Generating script |
|---|---|---|
| Cross-vendor A + proactive + reactive contribution rendering (5 open-weight deployment anchors + 4 APIs) | `assets/fig_cross_vendor_decomposition.pdf` | `code/build/plot_result_breakdowns.py` (hard-validates the complete T3 panel and all plotted endpoints) |
| Per-task A-R0 vs C+FDRS-R3 heatmap (15 families x 13 models) | `assets/fig_task_heatmap.pdf` | `code/build/plot_result_breakdowns.py` (reconstructs every cell from counts and validates row/column totals) |
| Model-scale scatter (10 open-weight models x A/C/C+FDRS endpoints) | `assets/fig_model_scale_scatter.pdf` | `code/build/plot_result_breakdowns.py` |
| Additional repository plot: pooled-exact token-economy Pareto/saving rendering | `assets/fig_token_economy.pdf` | `code/build/plot_token_economy.py` (hard-validates all ten model cells and pooled T6 totals) |
| Knowledge-profile heatmap (150 displayed cells, including layer diagnostics) | `assets/fig_knowledge_profile_heatmap.pdf` | `code/build/plot_profile_demand_evidence.py` |
| Additional repository plot: six-estimator recall@k curves (60 displayed points) | `assets/fig_demand_recall_k.pdf` | `code/build/plot_profile_demand_evidence.py` |
| &nbsp;&nbsp;unadapted deployed Hybrid suite (role reweight=false; selected alpha 0.7) | `results/demand_suites/suite_unadapted/hybrid_tfidf/metrics.json` | frozen demand-suite output, hash-checked by `code/aggregate/aggregate_demand_evidence.py` |
| Fig. S2(a) risk-weight sensitivity; weight-grid max abs(Delta) per model (0.85 / 0.85 / 0.90 / 0.95 / 1.30 pp). reproduce.py check 5 | `results/aggregates/a53_weight_grid_summary.json` | `code/aggregate/aggregate_a53_weight_grid.py`, `code/aggregate/aggregate_c4_riskweights.py` |
| &nbsp;&nbsp;〃 | `results/aggregates/comparison_riskw_uniform/proactive_comparison.json` | 〃 |
| &nbsp;&nbsp;〃 | `results/aggregates/comparison_riskw_l0heavy/proactive_comparison.json` | 〃 |
| &nbsp;&nbsp;〃 | `results/aggregates/comparison_riskw_l3heavy/proactive_comparison.json` | 〃 |
| Fig. S2(c) C-unadapted (cross-corpus) ablation | `results/aggregates/c_unadapted_summary.json` | `code/aggregate/aggregate_c_unadapted.py` |
| &nbsp;&nbsp;〃 | `results/aggregates/comparison_c_unadapted/proactive_comparison.json` | 〃 |
| &nbsp;&nbsp;〃 | `results/aggregates/comparison_c_unadapted/reactive_comparison.json` | 〃 |
| Fig. S2(b) generation-seed variance (10 seeds x 400 items; per-cell sd +/-0.00 to +/-0.85pp, panel median +/-0.20pp) | `results/aggregates/action7_multiseed_summary.json` | `code/aggregate/aggregate_action7_multiseed.py` |
| Fig. S2(a/b/d), E6 and reasoning raw reconstruction (149 compact model/arm/seed cells). reproduce.py check 16 | `results/raw/robustness_compact/manifest.json` | manifest consumed and hash-checked by the public robustness aggregators |
| Fig. S2(c) C-unadapted ablation: 40 runs x 2,000 aligned outcomes; +27.79pp over A and +3.16pp adapted increment. reproduce.py check 21 | `results/raw/c_unadapted/c_unadapted_outcomes_compact.json` | minimum-sufficient export; `code/aggregate/aggregate_c_unadapted.py` |
| &nbsp;&nbsp;〃 manuscript claim cross-check | `results/aggregates/c_unadapted_manuscript_crosscheck.json` | `code/aggregate/aggregate_c_unadapted.py` |
| Fig. S2(d) semantic-RAG (E5) baseline comparison | `results/aggregates/comparison_e5_semantic_rag/proactive_comparison.json` | `code/aggregate/aggregate_e5_semantic_rag.py` |
| Compact token-economy rendering, panels (a)-(b) of the pooled-exact figure plus the T5 reactive-ladder panel (X-C lead flips negative across fix rounds) | `assets/fig_token_economy_compact.pdf` | `code/build/plot_token_economy.py --compact` (hard-validates all ten T6 model cells, pooled totals, and the fifteen frozen T5 ladder panel means against per-model run summaries) |
| Accuracy--energy serving Pareto scatter (11 Protocol-A configurations; front 1.5B-7B-8B-14B-Next-480B) | `assets/fig_serving_pareto.pdf` | `code/build/plot_serving_pareto.py` (hard-validates all eleven frozen T11 rows field-for-field and the recomputed Pareto front) |
| Failure-taxonomy consensus stacked bars (100 double-reviewed items by tier and by condition; three unadjudicated disagreements drawn as their own hatched segment) | `assets/fig_failure_taxonomy.pdf` | `code/build/plot_failure_taxonomy.py` (hard-validates the 4x5x5 design, both reviewer marginals, kappa, and both consensus cross-tabs against the frozen human-review ledger) |
| Cross-vendor contribution rendering v2 (adds the sixth open-weight anchor Qwen3-Next, the panel's largest A -> C+FDRS lift +55.55pp; replaces the v1 rendering in the manuscript, v1 kept above) | `assets/fig_cross_vendor_decomposition_v2.pdf` | `code/build/plot_result_breakdowns.py --figure cross-vendor-v2` (same EXPECTED_T3 hard validation as v1) |
| Per-task heatmap v2, line-width 1:1 re-rendering (same 15 families x 13 models x 2 conditions; 150 mm source canvas placed at \linewidth so the model-name labels keep >= 6.5 pt effective size; replaces the v1 rendering in the manuscript, v1 kept above) | `assets/fig_task_heatmap_v2.pdf` | `code/build/plot_result_breakdowns.py --figure task-heatmap-v2` (same per-cell count reconstruction and row/column-total validation as v1) |
| Knowledge-profile heatmap v2, line-width 1:1 re-rendering (same 150 hard-validated displayed cells; 150 mm source canvas placed at \linewidth so the model-name labels keep >= 6.5 pt effective size; replaces the v1 rendering in the manuscript, v1 kept above) | `assets/fig_knowledge_profile_heatmap_v2.pdf` | `code/build/plot_profile_demand_evidence.py --figure profile-heatmap-v2` (same 150-cell manuscript-value hard validation as v1) |
### Headline prose, provenance layers, and documentation pointers

| Claim / Table | Artifact | Generating script |
|---|---|---|
| Headline: proactive + reactive documentation injection raises every evaluated >=7B open-weight model and every API by +32 to +56 accuracy points over the no-injection baseline A (Abstract; Sec. 1; Discussion). reproduce.py checks 1 and 13 | `results/aggregates/api_comparison_2000/proactive_comparison.json` | evaluation-harness freeze; independently rebuilt from the compact item outcomes by `code/aggregate/aggregate_primary_results_compact.py` |
| &nbsp;&nbsp;〃 | `results/aggregates/api_comparison_2000/reactive_comparison.json` | 〃 |
| Failure Anatomy labelling (primary): two practising power-systems engineers independently labelled the same frozen 100-record sample under the shared rubric established in the preliminary expert pass. They agreed on 97/100 records (Cohen's kappa 0.955); the 3 disagreements are reported rather than adjudicated, so the counts are ranges over the 8 possible resolutions. Across all 8 resolutions: C1 = 47--48 and is the largest single category, C3 = 28, C1+C2 = 59--60 of 98--99 classifiable failures (60.2--60.6%), and tier-4 C3 = 11 exceeds tier-4 C1 = 8. Agreement with the preliminary pass is 86.0% (kappa 0.790) for A and 85.0% (kappa 0.774) for B. It also has per-reviewer marginal / tier / condition tables. No reproduce.py check covers this layer: its inputs are the two engineers' completed booklets rather than a frozen public record | `audit/failure_taxonomy/human_review/agreement.json` | `code/audit/build_human_taxonomy_review.py` |
| &nbsp;&nbsp;reviewer A's 100 record-level labels in the record-review schema (`sample_id`, `label`, `confidence`, `rationale`, `primary_evidence`) plus the shared six-category rubric; marginals C1/C2/C3/C4/C5/item-defect = 47/13/28/2/9/1 | `audit/failure_taxonomy/human_review/reviewer_A.json` | 〃 |
| &nbsp;&nbsp;reviewer B's 100 record-level labels in the same schema and rubric; marginals C1/C2/C3/C4/C5/item-defect = 48/11/28/1/10/2 | `audit/failure_taxonomy/human_review/reviewer_B.json` | 〃 |
| Failure Anatomy preliminary expert pass over the same 100 records, kept as provenance; the formal expert review above supplies the reported labels. It covers the public 89,420-key sampling frame, the exactly replayed balanced 100-case draw, two independent reviews by practising power-systems engineers (85% agreement), and joint resolution of all disagreements. Its ledger (C1/C2/C3/C4/C5/item-defect = 52/11/23/1/10/3; C1+C2 = 63/97 classifiable; matched A-FDRS to C-FDRS C1 13 to 3) is provenance, and it supplies each record's tier/condition metadata plus the draw-replay checks behind the formal review. See `audit/failure_taxonomy/`. reproduce.py check 17 | `audit/failure_taxonomy/revision_sample/adjudication_v2.json` | `code/audit/recompute_failure_taxonomy_revision.py` |
| &nbsp;&nbsp;minimum-sufficient ordered frame for all 89,420 match-false records across 55 hash-identified source runs; supports replaying the draw from the released frame keys (see `audit/failure_taxonomy/`) | `audit/failure_taxonomy/revision_sample/sampling_frame_keys.json` | `code/audit/export_failure_taxonomy_revision_sample.py` |
| &nbsp;&nbsp;100 complete stable-ID sample records with query, reference, generated code, execution evidence, and source provenance | `audit/failure_taxonomy/revision_sample/sample_records.json` | 〃 |
| &nbsp;&nbsp;historical Failure Anatomy ledger (53/21/19/1/6), kept and reaggregated as provenance; the reported estimate comes from the expert review | `audit/failure_taxonomy/taxonomy_results.json` | `code/audit/recompute_failure_taxonomy.py` |
| &nbsp;&nbsp;conservative forensic linkage of the historical ledger: 32 unique links and 68 candidate sets left unresolved | `audit/failure_taxonomy/item_linkage_report.json` | `code/audit/link_failure_taxonomy_items.py` |
| E4 layer-2 query sets + construction/screening ledgers (84 naturally-occurring + 10 cross-platform + 20 engineer-written). Engineer 1 performed the initial screen and controlled rewrites. Engineer 2 blind-reviewed the candidate pool independently and verified semantic preservation. Both resolved hard disagreements jointly under the frozen rules | `benchmark/e4_layer2a/e4_layer2a_set.json` | `benchmark/e4_layer2a/` build+screening ledgers (archived alongside); `code/task_demand/summarize_e4_evals.py` |
| &nbsp;&nbsp;〃 | `benchmark/e4_layer2a/e4_layer2a_ext.json` | 〃 |
| &nbsp;&nbsp;〃 | `benchmark/e4_layer2b/e4_layer2b_set.json` | 〃 |
| Corpus-separation diagnostic: function Jaccard 0.25, token Jaccard 0.192, maximum query overlap 0.409, and zero query near-clones above 0.5 | `results/aggregates/contamination_analysis.json` | `code/aggregate/aggregate_contamination_analysis.py` |
| E6 risk-gate budget scan + probe AUROC (main-text gate diagnostics) | `results/aggregates/e6_budget/e6_budget_scan_summary.json` | `code/aggregate/aggregate_e6_budget_scan.py` |
| &nbsp;&nbsp;〃 | `results/aggregates/e6_probe_auroc.json` | 〃 |
| C3 historical split-holdout aggregates (D1D2 vs D3D4), retained as provenance | `results/aggregates/c3_holdout/R1_baseline_eval_D12/summary.json` | `code/aggregate/aggregate_c3_holdout.py`; full R1-R6 aggregate trees under `results/aggregates/c3_holdout/` |
| &nbsp;&nbsp;〃 | `benchmark/c3_splits/benchmark_D1D2.json` | 〃 |
| &nbsp;&nbsp;〃 | `benchmark/c3_splits/benchmark_D3D4.json` | 〃 |
| C3 deterministic rerun on the pinned stack: six complete per-query top-20 runs; cross-source vs source-matched recall@10 gaps +0.07pp on D1D2 (CI [-0.12,+0.27]) and -0.29pp on D3D4 (CI [-0.62,0.00]). reproduce.py check 27 | `results/supplementary_evidence/c3_deterministic_rerun_items.json` | locked-stack export by `code/aggregate/export_c3_deterministic_rerun.py`; public recomputation by `code/aggregate/aggregate_c3_deterministic_rerun.py` |
| &nbsp;&nbsp;C3 per-query metric and paired-bootstrap recomputation | `results/aggregates/c3_deterministic_rerun_recomputed.json` | `code/aggregate/aggregate_c3_deterministic_rerun.py` |
| C3/E4 contemporaneous execution provenance: six historical C3 logs, the E4 seed-42--46 log, and the original hash-seed regression; 48 private source-prefix occurrences are normalized and all logged endpoints match the frozen aggregates. reproduce.py check 29 | `results/supplementary_evidence/historical_demand_runs/manifest.json` | source export and public verification by `code/aggregate/aggregate_historical_demand_run_logs.py` |
| &nbsp;&nbsp;historical C3/E4 log recomputation, recording that these logs hold aggregates and endpoints only | `results/aggregates/historical_demand_run_logs_recomputed.json` | `code/aggregate/aggregate_historical_demand_run_logs.py` |
| Instrument migration ledger (main-text pointer: docs/MIGRATION.md) | `docs/MIGRATION.md` | hand-maintained ledger (pointer preamble localized to this repository) |
| Verbatim prompt templates for every condition (SM Sec. S3 and S6 pointer: docs/PROMPTS.md): baseline system/user turns, the B/C/X/R/Rsem proactive injection blocks, the FX/FD/FDR/FDRS fix-round templates, and the BM25 tokeniser | `docs/PROMPTS.md` | quoted verbatim with file:line from `code/backend/utils.py`, `code/intervention/{proactive,reactive,semantic,library_spec,rag_baseline}.py`; rendered examples produced by running those modules against the released artifacts (recipe in Sec. 5 of the file) |
| Probe-profile vs bare-R0 Spearman diagnostic (10-model L0--L3 rho = 0.78/0.93/0.78/0.78; five-model L1 rho = 0.90). reproduce.py check 20 | `results/aggregates/probe_r0_spearman.json` | `code/aggregate/aggregate_probe_r0_spearman.py` from unrounded 275-function profiles and 2,000-item A outcomes |
| Earlier probe-failure correlation diagnostic, retained as provenance | `results/aggregates/exp_d_probe_failure_correlation.json` | probe analysis (`code/probing/probe_framework.py`) |
### Method inputs (dataset/)

These corpora are the frozen inputs the method code consumes.

| Method input | Artifact | Consumed by |
|---|---|---|
| Augmented supervision corpus (task-demand model training/eval; also the corpus-separation diagnostic input) | `dataset/augmented_dataset.json` | `code/task_demand/task_demand_model.py`; SHA-pinned by `code/aggregate/export_e4_multiseed_predictions.py`; also read by `code/aggregate/export_c3_deterministic_rerun.py`, `code/aggregate/export_s4_item_evidence.py`, `code/aggregate/aggregate_contamination_analysis.py` |
| pandapower API-spec corpus (proactive/reactive injection source and probe substrate) | `dataset/pandapower_docs.json` | `code/probing/probe_framework.py`; `code/intervention/proactive.py`, `code/intervention/reactive.py`, `code/intervention/rag_baseline.py`; `code/backend/probe_runner.py` |
| L0--L3 probe set (2,080 probes over 275 functions) | `dataset/all_probes.json` | `code/backend/probe_runner.py` |
| Derived pandapower library-knowledge pack | `dataset/pandapower_library_knowledge.json` | derived by `code/intervention/library_knowledge.py` (see `code/README.md`) |
| OpenDSS documentation spec (E2 backend transfer; 76 entries) | `dataset/opendss_docs.json` | `code/build/build_e2_opendss_bench.py`; frozen per-artifact SHA-256 recorded in `benchmark/e2_freeze_manifest.json` (see `docs/MIGRATION.md`) |
| PyPSA documentation spec (E2 backend transfer; 67 entries) | `dataset/pypsa_docs.json` | frozen per-artifact SHA-256 recorded in `benchmark/e2_freeze_manifest.json`; consumed by the frozen E2 PyPSA runs (`results/demand_suites/e2_pypsa/zero_shot_tfidf/run_config.json`) |
## Coverage and limitations

[`docs/PROVENANCE_BOUNDARIES.md`](docs/PROVENANCE_BOUNDARIES.md) walks through each
evidence family above: what this repository publishes, what can be recomputed from
it, and what the original runs did not retain. It also names the GB-scale
production logs that stay unpublished.
