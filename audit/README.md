# audit/ — engineering-validity audit chain

This directory holds every per-case record behind the paper's
false-acceptance analysis: the sampling manifests, the automatic screening
results, both engineers' reviews, the refutation and census passes, and the
final label sets. Anyone checking a reported audit rate can trace it back to
the cases that produced it. The design, the rulebook, and the rates
themselves are in the supplementary material as SM S7.

The audit asks one question of each output the scalar criterion accepted: did
it get there through a physically faithful workflow? A `contaminated` label
means it did not.

## Three rounds

| Round | Scope | What it answers |
|---|---|---|
| **pilot20** | 20 cases | Calibration before freezing the rubric. Established that the five automatic checks were high-recall and that two independent engineer reviewers could apply the equivalence exception consistently. |
| **t4** | 610 accepted outputs | The main stratified round. Produces the design-weighted false-acceptance rate (18.51% at the deployed open-weight endpoint) and the gate operating characteristics. |
| **e1p** | 170 matched pairs | The paired arm: same item, same model, accepted under both C+FDRS and A+FX. Produces the headline 34.12% → 12.35% reduction and the exact McNemar test. `e1p_ext_*` is the forward-frozen 50-pair extension of the 405B+ stratum. |

## Reading one round

Each round runs the same pipeline, and its files appear in that order:

1. **`*_manifest.json` / `e1_sample_manifest.json`** — the sampling design:
   population counts, stratum allocations, and the item pointers drawn. Frozen
   before any case was inspected.
2. **`*_auto_results.json`** — re-execution in the instrumented sandbox with
   the five automatic checks (network/modification state, solver identity and
   convergence, power balance and range, result-table extraction, printed-scalar
   provenance). This is a *screening* layer, not a verdict.
3. **`*_judge1.json`, `*_judge2.json`** — independent reviews by two
   practising power-systems engineers against the same frozen equivalence rubric
   (κ = 0.88 on the main round, measuring inter-engineer agreement).
4. **`*_disagreements.json`** — cases where the two engineer reviewers split.
5. **`*_refute.json`** — an expert refutation pass over every *contamination*
   label, testing the strongest legitimate-equivalence interpretation. Labels
   that survive this pass are the ones counted; disputed physical facts are
   supported by deterministic re-execution evidence.
6. **`*_census.json`** — engineer review of the cases that passed all five
   automatic checks, so an automatic pass also has a human ruling behind it.
7. **`*_adjudication.json`** — the two engineers' reconciled rulings on cases
   still in dispute, using the frozen rubric and deterministic oracle evidence.

## Which file is authoritative

- `t4_final_labels_v2.json` is the final label set. It differs from
  `t4_final_labels.json` on 27 of 610 cases: v2 records the census and
  refutation outcomes, which moved several `clean-autopass` entries to
  `contaminated`. `reproduce.py` check 8 reads v2, and v1 is kept as the
  pre-census snapshot.
- `e1p_mcnemar.json` is the frozen paired result. `code/audit/pair_mcnemar.py`
  recomputes it end-to-end from the manifest and label chain; run it and compare.
- `t4_far_weighted.json` contains the sampling weights, the per-cell design
  diagnostics, and the Hájek-weighted rates by frame and by tier.
- `t4_refute.json` carries a refutation verdict for every contamination
  ruling: the 138 flagged-layer rulings of 2026-07-25, plus a census
  addendum recorded 2026-08-28 for the 14 contaminations the all-pass
  census found (category `census`, 14/14 upheld). Its 143 `UPHELD` keys are
  exactly the 143 `contaminated` keys of `t4_final_labels_v2.json`.

## Supporting files

| File | Contents |
|---|---|
| `t4_census2.json` | Second-pass census with per-item notes, including why three heavy `calc_sc`/parallel items on `case6470rte`/`case9241pegase` had to be escalated to a compute node |
| `t4_allpass_check.json` | Blind-spot rescan plus a random sample of all-pass cases, testing whether a whole class of contamination slips past the screening layer |
| `t4_tolerance_sensitivity.json` | Per-case replay under the tightened matcher (rel. 1e-3, abs. 1e-4); drives the "rejects 40.48% of false acceptances for 1.72% of legitimate matches" result |
| `t4_audit_provenance.json` | Engineer-review provenance, role assignment, agreement statistics, and pointers to the frozen decision protocol |
| `pilot20_regression_check.json` | Re-run of the pilot cases under the final instrument, confirming the rubric freeze did not move earlier judgements |

## Label vocabulary

Counts are over the 610 cases of `t4_final_labels_v2.json`:

| Label | n | Meaning |
|---|---:|---|
| `clean-autopass` | 251 | Passed all five automatic checks and the engineer census |
| `equivalent` | 169 | A check fired, but the deviation fell under the frozen equivalence exception (idiom, mathematically identical computation, or a method the query does not constrain) |
| `contaminated` | 143 | Upheld false acceptance that survived the refutation pass |
| `item-defect` | 38 | The benchmark item itself is at fault; excluded from every rate denominator and from every model's rate |
| `clean` | 9 | Ruled faithful at adjudication |

The false-acceptance denominator is therefore 610 − 38 = **572**, and
143/572 = 25.0% is the pooled unweighted prevalence in SM Table S19(a). The
18.51% headline is this quantity restricted to the deployed open-weight frame
and reweighted by the sampling design.

`item_defect_errata.json` makes those 38 rulings checkable. It lists each
audit record with its frame and resolves it to the canonical benchmark item
behind it — the `bench_index` inside a record key numbers positions within the
sampling frame, not within the benchmark, so the join runs through
`e1_sample_manifest.json` — which leaves 31 distinct items, each given with the
first 80 characters of its query. The file then recomputes every archived
accuracy cell on the 2,000-item suite with those 31 items dropped (denominator
1,969) and re-runs the Table 3(b) equivalence contrasts under both
denominators. Across all 226 cells the largest movement is -0.531 pp
(GPT-OSS-120B, C+FD), and every TOST decision is unchanged. The accuracy
tables are reported on the full 2,000 items and are not restated; the errata
exists so a reader can confirm that the defect rulings do not carry them.

## Recompute

```bash
PYTHONPATH=code python3 code/audit/pair_mcnemar.py   # writes e1p_mcnemar.recomputed.json
python3 code/audit/recompute_validity_tables.py      # all 610-case tables and bootstraps
python3 code/audit/item_defect_errata.py             # rebuilds item_defect_errata.json
python3 reproduce.py                                 # audit groups 3, 8, 11, 17, 18
```

## Related diagnostic artifacts

- [`audit/matcher_fn/`](matcher_fn/) holds the earlier executed-but-unmatched
  mechanism audit: 100 complete source records, a de-duplicated result census,
  and a fresh item-level ledger, since the original item-level labels were not
  retained (`reproduce.py --group 11`).
- `p02_parity_item_outcomes.json` contains the five aligned 2,000-item match
  vectors behind the 70B/120B nearest-API bootstrap and TOST comparisons
  (`reproduce.py --group 12`).
- `validity_tables_recomputed.json` is rebuilt from the 610 final labels,
  sampling manifests, design cells, and public compact outcome vectors.  It
  covers the point estimates, Wilson intervals, gate/tolerance panels, all
  frozen stratified bootstraps, and the 2,106-item common-support
  standardisation (`reproduce.py --group 18`).
- [`audit/failure_taxonomy/`](failure_taxonomy/) contains the complete 100-case
  C1--C5 classification ledger, exemplars, and an independent summary rebuild
  for the 89,420-event sampling frame (`reproduce.py --group 17`).
- [`audit/e2_import_scan/`](e2_import_scan/) contains all 1,080 pre-fix
  generated programs, copies of the two frozen scan reports, and pointers to
  the 2,160 public post-fix programs.  The audit rescans all 3,240 programs and
  rebuilds both historical reports byte for byte (`reproduce.py --group 23`).

## Coverage and limitations

Two release-time normalizations replaced private Python-environment prefixes
in diagnostic string fields. They are narrowly scoped and idempotent, and all
outcome and score evidence is unchanged.

- `primary_diagnostic_path_normalization.json` records the replacement in
  diagnostic error strings. It verifies that all outcome, token, task,
  injection, and routing evidence remains exact (`reproduce.py --group 25`).
- `probe_diagnostic_path_normalization.json` records the corresponding
  release-only normalization for the 29,120 raw probe records. Changes are
  confined to four diagnostic traceback/output/error fields. Raw responses,
  scores, `correct` labels, and all other scalar evidence are verified exact,
  with idempotence and zero residual private prefixes checked by
  `reproduce.py --group 28`.
