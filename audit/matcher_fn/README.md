# Matcher false-negative mechanism audit

This directory restores the frozen P0-1 matcher false-negative diagnostic,
which was omitted when the public repository was first assembled: a 100-record
sample of executed-but-unmatched outputs, reviewed to tell genuine model
errors apart from misses by the matcher and the answer parser.  It keeps
three levels of evidence separate: the complete source records, the
historical aggregate, and an item-level ledger built over those same records.

## Files

- `sample_records.json` contains the 100 complete source records reconstructed
  from the frozen experimental pipeline.  Every record is joined by the
  unique `(model, condition, natural_language_query)` key and includes the
  original item identifier, source-relative path, execution/match flags, full
  generated and reference code, execution output, source-record hash, source
  Git blob, and the source commit that contains those exact blobs.
  Adjudication labels are held separately, in
  `revision_readjudication/adjudication_v2.json`.  Two records carry six
  occurrences of a source-machine experiment-environment prefix, published as
  `<EXPERIMENT_VENV>`.  The embedded normalization manifest stores the rule,
  the counts, and both the original and release canonical-record SHA-256
  values; the recomputation script checks the released records, the redaction
  count, and manifest consistency.
- `recomputed_summary.json` is the frozen output of the independent aggregate
  consistency and unique-census reconstruction.  `reproduce.py --group 11`
  regenerates this file in a temporary directory and compares every field.
- `P01_false_negative_audit.json` is a byte-exact copy of the historical
  aggregate (SHA-256
  `cc13ba40bd1fcb8cf3a803c9778fee7f1dbaf331e94896640d9dfb3d28126967`).
  Reviewer M1 was a practising power-systems engineer.  The file records 8
  confirmed false negatives, 5 ambiguous cases, and 87 genuine errors,
  together with aggregate breakdowns.
- `p01_fn_samples_legacy_truncated.json` is the historical sample export,
  copied unchanged (SHA-256
  `1fba186f5a2273a9557ea547fdfc032515cbbc6bdce255e3cdb43683061d5d5f`).
  It is kept for provenance: 21 `generated_code` fields end at 1,500
  characters and 10 `reference_code` fields end at 800 characters.
- `revision_readjudication/adjudication_v2.json` is a versioned item-level
  ledger over the same 100 complete source records.  It combines three
  record-level review passes conducted by two practising power-systems
  engineers: two independent primary passes and an isolated blind re-check by
  one of those engineers.  The second engineer performed the other primary
  pass; one of the same two engineers then performed the adversarial pass
  over candidate false negatives, and both engineers jointly reconciled every
  disagreement or challenge.  Deterministic frozen record evidence supports
  these decisions, with matcher and parser outputs entering as evidence for
  the engineers to weigh.  The ledger records 8 false negatives, 3
  task-specification ambiguities, and 89 genuine errors; all eight false
  negatives are parser-extraction misses.  Their tier, model, and task
  marginals match the historical aggregate exactly, in a check made only
  after the item labels were frozen.

## What the historical files contain

The historical files hold aggregate breakdowns.  They have no per-sample
final label, no judge or refutation record, and no inclusion probability.
The historical 8/5/87 judgments can therefore be checked for aggregate
arithmetic consistency, but they cannot be recomputed case by case from those
files alone.  `revision_readjudication/` supplies the item-level layer by
re-adjudicating the complete records: it is a fresh set of judgments over the
same 100 records, made from the record evidence rather than inferred from the
published historical totals, and it stands on its own rather than recovering
the lost historical mapping.

## De-duplicated census

The historical aggregate's 744,000-item census counted the 11-model local
panel twice.  The current public aggregates provide a unique-key census:

- `results/aggregates/api_comparison_2000/proactive_comparison.json`
- `results/aggregates/api_comparison_2000/reactive_comparison.json`

Iterating each `(condition, model)` cell once yields 196 cells, 392,000 graded
outputs, 188,297 executed outputs, 121,487 matches, and 66,810
executed-but-unmatched outputs.  The eight-condition diagnostic frame that
this audit sampled contains 104 cells and 35,446 executed-but-unmatched
outputs.

## Recompute

Run the independent consistency and census reconstruction with:

```bash
python3 code/audit/recompute_matcher_fn_audit.py
python3 code/audit/recompute_matcher_fn_readjudication.py
python3 reproduce.py --group 11
```

## Coverage and limitations

The sample is a mechanism diagnostic drawn from exactly eight A/C-family
conditions: `A`, `A_FD`, `A_FDR`, `A_FDRS`, `C`, `C_FD`, `C_FDR`, and
`C_FDRS`.  Inside that frame it spans 15 models, five capability tiers, and
15 task families.  The remaining prompting conditions sit outside it, and the
allocation is not self-weighting, so the simple sample proportion is a
description of the drawn cases rather than a design-unbiased estimate of any
population, including the eight-condition frame itself.  Read the historical
8% sample proportion and its nominal Wilson interval on those terms: they
characterise the mechanism in the cases actually inspected, and they are not a
full-panel correction, a ceiling on how often the mechanism occurs, evidence
that it is uniform across models or conditions, or evidence about bias in
cross-model rankings.  The de-duplicated census above sizes the frame this
audit sampled; it does not carry the 8% proportion beyond that frame.
