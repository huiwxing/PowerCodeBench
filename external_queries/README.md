# external_queries/ — external-query evaluation archive

This directory collects the external-query work in one place: how the three
external query sets were screened and rewritten, and how models scored on
them. Look here to trace a single external item from its source posting to
its evaluation row. All files are read-only copies of the frozen
construction and evaluation artifacts, whose sources are
`benchmark/e4_layer2a/`, `benchmark/e4_layer2b/`, and
`task_demand/results/`. Cited in the supplementary material as the SM S5
external-query subsection (`tab:sm_external_protocols` and
`tab:sm_external_queries`).

## Construction ledgers (per-item screening, adjudication, rewrite logs)

Sets 1 and 2 were screened independently by two practising power-systems
engineers under the frozen inclusion/exclusion rules. Engineer 1 produced the
controlled rewrites and Engineer 2 independently verified semantic
preservation; hard disagreements were resolved jointly against the frozen
rules and the original source text. For Set 3, the same two-engineer protocol
covers the approved query text, reference annotation, and ambiguity review.

| Dir | Set | Contents |
|-----|-----|----------|
| `benchmark/e4_layer2a/` | Set 1, community naturally occurring (n=84) | `collection_manifest.json` (29 frozen search routes; titles/excerpts/URLs/dates only), `candidates_pool.json` (448 -> 362 dedup), `screening_decisions.json` (Engineer 1 first pass: 80/137/145), `disagreements.json` (Engineer 2 blind re-screen of the 217 pool; 26 joint disagreement decisions), `e4_layer2a_set.json` (frozen set and engineer rewrite log fields), `source_lut.json`, `manifest.json` |
| `benchmark/e4_layer2a/` | Set 2, cross-platform supplement (n=10) | `all_external_candidates.json` (595-item multi-source pool triage), `round2_screening.json` (independent engineer screen of the 69-item shortlist), `round2_disagreements.json` (six joint decisions), `e4_layer2a_ext.json` (frozen set and engineer rewrite log fields), `source_lut_ext.json`, `manifest_ext.json` (8 direct + 2/6 admitted on adjudication) |
| `benchmark/e4_layer2b/` | Set 3, engineer-written (n=20) | `e4_layer2b_set.json` (two-engineer approved edition, byte-identical query text, independently verified reference annotations), `ambiguities.json` (joint engineer ambiguity record), `ieee69_network.py` (Baran-Wu 69-bus feeder implementation frozen with the set), `manifest.json` |

Where the files live: Sets 1 and 2 are canonically under
`benchmark/e4_layer2a/`, with `all_external_candidates.json` in its
`external_supplement/` subdirectory. The earlier
`set1_naturally_occurring_n84/` and `set2_cross_platform_n10/` directories
held byte-identical copies and were removed. Set 3's JSON artifacts are
canonically under `benchmark/e4_layer2b/`, while `ieee69_network.py`
remains in `set3_engineer_written_n20/`.

## Frozen evaluation outputs (both arms)

`eval_set1/`, `eval_set2/`, `eval_set3/`, `eval_internal_holdout/` each
hold `unadapted/`, `adapted/` (initial run, kept for provenance) and
`unadapted_v2/`, `adapted_v2/`. The v2 pair is the frozen evaluation quoted
in SM `tab:sm_external_queries`(a)/(b): the `benchmark_reference_eval_only`
split of `metrics.json` reports the same recall@10 / hit@10 and per-role
recall values, for example Set 1 unadapted 0.5507/0.8333.

## Conditional-selector re-evaluation table

`conditional_selector/` is the exploratory query-conditional selector
record. It holds `metrics_by_set_v2.json` (frozen S1 rule, routed counts,
conditional-arm recall), `per_role_v2.json`, and `seed_sensitivity/`, the
five predictor-retraining seeds behind the mean [min, max] column. Its
re-evaluated fixed arms are the full re-evaluation table the SM points to;
they differ from the frozen unadapted/adapted columns by at most 1.0pp,
which is retraining variance.
