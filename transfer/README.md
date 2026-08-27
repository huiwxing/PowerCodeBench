# transfer/ — OpenDSS/PyPSA cross-backend transfer pilot

This directory records the cross-backend transfer pilot that ported the
pandapower instrument to OpenDSS and PyPSA. It holds the archived
schema-level pilot the work grew out of, the ledger of defects found and
fixed during bring-up, and the import scans that back the ledger's counts.
Cited in the supplementary material as SM Sec. S9 ("Cross-backend transfer
at library cold start"), whose results are Tables S18--S19.

| File | Contents | Verifies |
|------|----------|----------|
| `pypsa_schema_pilot.md` | The earlier schema-level PyPSA pilot, archived; the end-to-end pilot reported in the supplement replaced it | Superseded, not cited in the current SM; background to SM Sec. S9 |
| `DEFECT_LEDGER.md` | Per-defect write-ups: two construction-level defects that forced a re-freeze (B1' self-contained prompt; bare-arm library naming) and three backend-neutral instrument defects | SM Sec. S9.1 "Benchmarks and self-contained prompts"; migration components in `docs/MIGRATION.md` |
| `import_scan_prefix_B2.json` | Pre-fix evidence: per-file scan of the frozen B2 round, with condition A at 360/360 pandapower imports and 0/360 target-library imports | The pre-fix baseline behind SM Sec. S9.1 "a scan of all 24 result files finds zero pandapower imports across all 2,160 generations" |
| `import_scan_v2_postfix.json` | Post-fix verification: frozen 13-file scope (0/1,170 pandapower; 629/630 OpenDSS, 540/540 PyPSA target imports) plus the 24-file superset scan | SM Sec. S9.1, same sentence (the 24-file / 2,160-generation figure; the 13-file / 1,170-generation frozen scope is recorded alongside) |

The pilot's primary frozen artifacts stay in their measurement locations
rather than being copied here. The mini-benchmarks and freeze manifest are
`benchmark/e2_opendss/`,
`benchmark/e2_pypsa/`, and `benchmark/e2_freeze_manifest.json`. The curated
backend documentation specs are `dataset/opendss_docs.json` (76 entries) and
`dataset/pypsa_docs.json` (67 entries). Probes, snippets, per-model
profiles, and all bench cells sit in the frozen experimental pipeline's
`probe_eval_results/e2_*` run trees; the frozen aggregates archived here are
`results/aggregates/e2_*`. The six-component migration ledger is
`docs/MIGRATION.md`.
