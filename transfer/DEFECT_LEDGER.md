# E2 transfer pilot — per-defect ledger

Five defects surfaced during the OpenDSS/PyPSA bring-up and were fixed:
two at the construction level, three in the instrument itself.
Each entry below gives the symptom, the root cause, the fix, and the frozen
artifact that records it, and every number can be recomputed from those
artifacts. Cited in the supplementary material as SM S10, together with the
pre-fix evidence under `transfer/`.

## Construction-level defects (class-1)

A class-1 defect changes which items or conditions enter the measurement,
so the fix requires re-freezing the affected artifacts and re-running every
cell.

### D-A. Self-built networks unreachable from the prompt (B1' fix)

- **Symptom**: v1 accuracy floor of 1.1-5.6% on both backends under all
  conditions.
- **Root cause**: the initial freeze embedded each self-built network only
  in the *reference* code. The prompt carried an NL query referencing
  entities ('ldb'/'cap1') of a network the model never saw, so the ground
  truth was unreachable in principle. 176/180 and 179/180 generations
  fabricated substitute networks, and lenient re-parsing rescued 0/202
  once instrument-layer attribution was excluded.
- **Fix (B1', self-contained prompt protocol)**: the network setup code is
  embedded in the prompt as given context at the prompt-render layer
  (`utils.render_benchmark_task_text`). Demand-model, retrieval, and
  layer-need selector inputs stay as they were in the B2 freeze, and ground
  truth, reference code, matcher, and artifacts stay unchanged
  (`natural_language_query` / `reference_code` / `ground_truth` verified
  180/180 byte-identical against the B2-frozen result files).
- **Record**: class-1 re-freeze note `b1prime_refreeze` in
  `benchmark/e2_freeze_manifest.json`, with new bench sha256 prefixes
  OpenDSS `fd5e978fde35a5ec` and PyPSA `49d561b5f47742f3`; all cells were
  re-run (v2).

### D-B. Bare-condition system prompt named the wrong library

- **Symptom / pre-fix evidence**: condition A's system prompt hard-coded
  "...code generator for the pandapower library" while the injected
  conditions named the backend under test. In the B2 round, 360/360
  condition-A generations imported pandapower and 0/360 imported the target
  library (per-file scan: `transfer/import_scan_prefix_B2.json`). The C-A
  execution-direction signal was therefore confounded by the library the
  prompt pointed at, and it was downgraded pending v2.
- **Fix**: `utils.SYSTEM_PROMPT_BENCHMARK` / `SYSTEM_PROMPT_FIX` gained a
  `{library}` slot filled from the `BENCHMARK_LIBRARY` environment
  variable. Leaving the variable unset selects pandapower and leaves the
  main-line prompts unchanged, regression-verified over all 2,000 main-line
  items; the change is reversible by policy.
- **Post-fix verification**: a full-corpus scan of all 1,170 v2 generations
  (13 files at scan time) found 0/1,170 pandapower imports and 629/630
  (OpenDSS) / 540/540 (PyPSA) target-library imports. The single OpenDSS
  miss is an empty DeepSeek generation (`e2opendss_b3b64a09a5d9`) that
  failed code extraction, not a generation using the wrong library. See
  `transfer/import_scan_v2_postfix.json`, which also records the 24-file
  superset scan including the later DeepSeek and R_semB cells.

## Backend-neutral instrument defects (fix-no-redraw)

These three sit in measurement code rather than in item selection, so each
fix was applied without re-drawing the frozen sample. All three were
regression-tested to leave pandapower behaviour unchanged.

### D-C. Aggregation clobber in per-model SLURM runs

- `probe_runner.py` `_print_proactive_comparison` built its model list
  only from the calling run's in-memory summaries. Per-model jobs each
  rewrote `proactive_comparison.json`, so the last-finishing job dropped
  the other model's rows. It now scans
  `output_dir/*/benchmark_results_cond*.json` on disk, which is idempotent;
  result files were unaffected. Record: the frozen experimental pipeline's
  `probe_eval_results/e2_bench_diagnosis.json` (section 1).

### D-D. L3 scorer blind to `Module_Method` spec names

- `_check_api_validity` reduced dotted call paths to their last segment
  (`Circuit.TotalPower` -> `TotalPower`). That segment cannot match the
  underscore-joined `Module_Method` spec-name convention the OpenDSS
  artifact requires, since the probe generator's fake-name pool needs
  underscore-tokenisable names. PyPSA instance-method calls were also
  invisible to target-API extraction. Two minimal fixes in
  `probe_framework.py`: dotted calls gain an underscore-joined candidate,
  accepted only if it is a documented corpus name, and target-function
  detection gains a `Class_method` instance-call fallback. A three-library
  smoke test left pandapower behaviour unchanged, moved OpenDSS validity
  from 0 to 1.0 with a target hit, and produced a PyPSA target hit.

### D-E. L3 scoring executed in-process (native segfault risk)

- `execute_code_safely` ran probe L3 code via in-process `exec`, and
  OpenDSS's native C++ engine can segfault, taking the runner with it. The
  fix is `utils.execute_code_safely_subprocess`: same result schema, an
  isolated subprocess, negative return codes recorded as `NativeCrash`, a
  single-threaded environment, and `MPLBACKEND=Agg`.
  `probe_framework.evaluate_L3` switched to it. In regression, normal
  execution, a deliberate ctypes segfault, and an ordinary exception were
  all classified correctly with the parent process surviving. The benchmark
  path already had watchdog process isolation and was unaffected.

## Environment prerequisite

The target library must be installed in the pinned execution environment:
`opendssdirect` 0.9.4 and `pypsa` 1.2.3, with the frozen pandapower stack
held at its pinned versions.
