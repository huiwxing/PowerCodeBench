# Compact supplementary evidence

This directory holds the smallest source records that rebuild the
supplementary probe, demand-estimation, external-query, and reactive-router
results without the private source checkout. Every JSON carries the frozen
source commit and repository-relative source-file SHA-256 provenance, plus a
Git blob where the source record was tracked.

## Artifacts

- `probe_profiles.json` (2,341,354 bytes): all 14 manuscript models x 275 API
  functions x L0--L3. The complete layer objects are retained, including
  `n_probes`, `by_probe_type`, and diagnostics. This supports the S2 protocol
  counts and mean-score table, the heatmap, and the uniform/mild-L3
  function-risk features used by the AUROC analysis. The 29,120 underlying
  response records (10 open-weight + 4 API models, 2,080 responses each) are
  archived byte-exact, apart from the diagnostic-path normalization of private
  workstation prefixes in four diagnostic error/output fields (see
  `audit/README.md`). The manifest pins every source Git blob, source/release
  SHA-256, byte count, layer, and record count. The independent invariance and
  idempotence checks are recorded in
  `audit/probe_diagnostic_path_normalization.json`.
- `demand_estimators.json` (93,883 bytes): the complete reported metrics
  objects and alpha-selection curves for the six `suite_4428033` estimators.
  This supports every point on the S4 recall/hit-at-k curves and the panel-(a)
  table, including the frozen Hybrid Cross-Encoder alpha target.
- `s4_item_evidence.json` (10,423,644 bytes): exact unfiltered top-20 rankings
  for Zero-shot TF-IDF, Pairwise TF-IDF+LogReg, Hybrid TF-IDF, Zero-shot SBERT,
  and Zero-shot Cross-Encoder on Aug-test, Aug-orig, and Bench-ref, plus both
  arms of the Hybrid role-reweighting comparison. Hybrid Cross-Encoder sits in
  a separate `revision_reruns` namespace with its full top-20 rankings, 0/7
  frozen-slice agreement ledger, alpha diagnostic, and historical
  no-saved-model/hash-state record. The six historical role-filtered Bench-ref
  deployment top-10 exports are kept separate from the unfiltered Fig. S2
  rankings.
- `e4_item_predictions.json` (316,716 bytes): 194 final seed-22 E4 queries,
  with query text, gold functions and roles, selector decision, and top-20
  predictions for the unadapted, adapted, and conditional arms. This is
  sufficient to independently recompute recall/hit at k={1,3,5,10,20}, per-role
  recall and counts, and the conditional-selector table. The recomputation runs
  from these predictions, and the archived aggregates serve as its cross-check
  targets.
- `e4_multiseed_predictions.json` (142,017 bytes): deterministic item-ranking
  reconstruction for seeds 42--46. All 900 archived metric cells, selected
  alphas, and role weights match. See "Coverage and limitations" for what the
  historical seed-42--46 runs retained.
- `c3_deterministic_rerun_items.json` (2,904,344 bytes): six complete top-20
  runs from the deterministic C3 rerun on the locked stack, including both
  cross-style directions and their historical-aggregate provenance.
- `historical_demand_runs/`: six contemporaneous C3 execution logs, the E4
  seed-42--46 execution log, and the original hash-seed determinism regression.
  Their manifest preserves source/release hashes, and the verifier matches
  every logged endpoint to its frozen aggregate. These logs establish execution
  provenance for the runs described under "Coverage and limitations".
- `reactive_router_counts.json` (2,013,072 bytes): all 68,107 item-round route
  events from the 20 C+FDR/C+FDRS run logs used by the S11 per-model
  route-share table, with source paths and hashes for every cell.

## PowerCodeBench-only rebuild

```bash
python3 code/aggregate/aggregate_probe_profiles.py
python3 code/aggregate/recompute_probe_transcripts.py
python3 code/aggregate/aggregate_demand_evidence.py
python3 code/aggregate/aggregate_s4_item_evidence.py
python3 code/aggregate/aggregate_c3_deterministic_rerun.py
python3 code/aggregate/aggregate_historical_demand_run_logs.py
python3 code/aggregate/aggregate_reactive_router_shares.py
```

The outputs land under `results/aggregates/`. Running `python3 reproduce.py`
invokes every rebuild and compares the result field by field with the frozen
release artifacts. Maintainers who hold the frozen source repository can repeat
the one-time source exports wherever the corresponding exporter exposes
`--source-root`.

## Coverage and limitations

**Reconstructed retrieval baselines.** Zero-shot SBERT is a deterministic CPU
reconstruction from the immutable `sentence-transformers/all-MiniLM-L6-v2`
snapshot at revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, whose
90,868,376-byte `model.safetensors` file has SHA-256 `53aa5117...d9db`. It
matches all seven frozen evaluation slices at every reported k, and unfiltered
top-20 rankings are retained for Aug-test, Aug-orig, and Bench-ref. Zero-shot
Cross-Encoder is reconstructed the same way, from the immutable
`cross-encoder/ms-marco-MiniLM-L-12-v2` revision
`7b0235231ca2674cb8ca8f022859a6eba2b1c968`, and it likewise matches all seven
frozen slices.

**Hybrid Cross-Encoder alpha.** The historical Hybrid Cross-Encoder run saved
no model and no hash state, so its own item rankings are gone. What is
published instead is a deterministic rerun on the locked stack, held in the
`revision_reruns` namespace of `s4_item_evidence.json` rather than merged into
the frozen curve. Role weights match the frozen run. The alpha optimum moves
from 0.3 to 0.4, a narrow separation between neighbouring grid points, and the
agreement ledger records 0/7 frozen slices reproduced. Every conclusion the
paper draws from this estimator is unchanged, so the shift is quantitative
drift within a stable result.

**C3 and E4 per-query rankings.** The seed-42--46 E4 runs and the historical C3
runs kept their aggregates and their execution logs. Their writers serialized
no per-query rankings, and the C3 logs record null prediction and model
exports. The published item records for those runs,
`e4_multiseed_predictions.json` and `c3_deterministic_rerun_items.json`, are
therefore deterministic reruns on the pinned stack, and they are labelled as
such. The contemporaneous execution logs in `historical_demand_runs/` match
the frozen aggregate endpoints.
