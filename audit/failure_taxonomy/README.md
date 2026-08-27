# Failure Anatomy evidence

This directory holds the Failure Anatomy evidence: the 100-record sample of
failed generations, the expert category labels over it, the frame the records
were drawn from, and the earlier preliminary pass kept as provenance.  The
reported category counts come from the formal expert review under
`human_review/`.

## Expert review

Two practising power-systems engineers independently labelled the same frozen
100-record sample using the rubric established in the preliminary expert pass.
Their labels replace the preliminary pass for every reported category count;
the preliminary pass is retained for provenance and reproducibility.

The human layer contains:

- `human_review/reviewer_A.json` and `human_review/reviewer_B.json`: 100
  record-level labels each, in the same review schema as the preliminary pass
  (`sample_id`, `label`, `confidence`, `rationale`, `primary_evidence`), with
  the shared six-category rubric embedded in both files; and
- `human_review/agreement.json`: 97/100 direct agreement, Cohen's kappa 0.955,
  the three disagreements listed item by item, per-reviewer marginal, tier, and
  condition tables, agreement against the preliminary expert pass, and a
  sensitivity block.

The three disagreements are reported item by item rather than merged into a
single label set, so there is no single human point estimate.  The reported
numbers come from the sensitivity block, which recomputes each quantity under
all eight resolutions of those disagreements.  Across all eight: C1 = 47-48
and is the largest single category in every resolution; C3 = 28; C1+C2 = 59-60
of 98-99 classifiable failures (60.2-60.6%); and tier-4 C3 = 11 exceeds tier-4
C1 = 8 in every resolution.  The remaining categories move only within
C2 = 11-13, C4 = 1-2, C5 = 9-10, and item-defect = 1-2, and every resolution
sums to exactly 100 records.  The per-reviewer marginals
(C1/C2/C3/C4/C5/item-defect) are 47/13/28/2/9/1 for A and 48/11/28/1/10/2 for
B.  In the matched slices the C1 count falls from 10 in `condA_FDRS` (both
reviewers) to 3 (reviewer A) or 4 (reviewer B) in `condC_FDRS`.

The two engineers agree with the preliminary expert pass on 86 records
(kappa 0.790) and 85 records (kappa 0.774) respectively.  Two differences are
worth naming: C1 is 47-48 against the preliminary pass's 52 while C3 is 28
against 23; and within tier 4 the formal labels place C3 (11) above C1 (8),
reversing the preliminary ordering (C1 11, C3 7).  The paper's mechanism claim
holds under all eight resolutions: C1 largest overall, C1+C2 the majority of
classifiable failures, and C1 collapsing between the matched `condA_FDRS` and
`condC_FDRS` slices.

## How the labels were collected

Both halves of the round are in `code/audit/`.  `build_taxonomy_booklet.py`
renders `revision_sample/sample_records.json` into the booklets the engineers
worked from.  Each entry has the operator query, the expected answer, the
execution status and error, the automatic diagnostics, the line-numbered
generated program, and a collapsed reference solution, followed by an empty
judgement block.  The booklets are blind: each engineer labels from that
evidence alone.  `build_human_taxonomy_review.py` reads the completed booklets
back and writes the `human_review/` files above:

```bash
python3 code/audit/build_taxonomy_booklet.py --out-dir booklets --batches 4
python3 code/audit/build_human_taxonomy_review.py --booklets /path/to/booklets
```

The pair is reusable: point it at any sample in the same schema, with
`--pattern` for a different booklet naming, to run the equivalent labelling
elsewhere.

## Sampling frame

The 100-record sample that both layers label was drawn from the fully public
89,420-key frame (`revision_sample/sampling_frame_keys.json`) by a fixed
stratified design with seed `20260809`.  Stratification cells: four frozen
model groups (tier1: sub-7B; tier2: 7B--32B; tier3: 70B--120B; tier4: 405B+
augmented by Qwen3-Next) x five prompt conditions (`condA_FX`, `condA_FD`,
`condA_FDR`, `condA_FDRS`, `condC_FDRS`), at 5 samples per (tier, condition)
cell, covering all 15 task families present in the failure pool.
`reproduce.py --group 17` replays the draw exactly from the released frame
keys.

The frame is what this repository ships in place of the 2.98 GB of source
result logs behind it.  Everything the audit rests on is here: the 89,420
ordered keys, the 100 complete sampled records, both engineers' label sets,
and every decision ledger.  Maintainers who hold that original source-result
directory can regenerate the public frame and sample with:

```bash
python3 code/audit/export_failure_taxonomy_revision_sample.py \
  --source-root /path/to/frozen/source-root \
  --output audit/failure_taxonomy/revision_sample
```

## Preliminary expert pass, retained for provenance

`revision_sample/` is the earlier fully public expert review pass.  It supplies
the sampling substrate that the current round still stands on: the expert
review labels exactly the records it selected, and it remains the source of
each record's tier and condition metadata.  Its label layer is the part that
has been replaced: `judge_a.json`, `judge_b.json`,
`disagreement_adjudication.json`, and the `final_label` field and cross-tabs of
`adjudication_v2.json` no longer supply the reported estimate.

The public chain contains:

- `sampling_frame_keys.json`: all 89,420 ordered `(bench_index, item_id)`
  keys, grouped under the 55 hash-identified source runs;
- `sample_records.json`: the 100 complete records selected by the fixed
  4-tier x 5-condition x 5-item design and seed `20260809`;
- `judge_a.json` and `judge_b.json`: preliminary, independent,
  rubric-constrained record-level reviews by two practising power-systems
  engineers (85/100 direct agreement);
- `disagreement_adjudication.json`: the two engineers' rubric-constrained
  consensus decisions for all 15 and only the 15 disagreements; and
- `adjudication_v2.json`: the deterministic merged ledger and all category,
  tier, condition, and task cross-tabs.

The preliminary pass records the two engineers' roles, shared rubric,
item-level rationales, confidence, primary evidence, and consensus resolution
of disagreements in the JSON ledgers.

Run the public rebuild of that pass with:

```bash
python3 reproduce.py --group 17
```

This command replays the exact 100-item draw from the released frame keys and
rebuilds the preliminary expert ledger field-for-field:
C1/C2/C3/C4/C5/item-defect = 52/11/23/1/10/3, C1+C2 = 63/97 classifiable
failures, and a C1 count of 13 against 3 between the matched `condA_FDRS` and
`condC_FDRS` slices.  Those figures document the preliminary pass rather than
the reported result; the draw replay they verify also supports the expert
review, which labels the same 100 records.

## Coverage and limitations

- No `reproduce.py` check covers the expert review layer: its inputs are the
  two engineers' booklets rather than a frozen public record.
- `taxonomy_results.json` and `recomputed_summary.json` retain the older
  53/21/19/1/6 audit as historical provenance.  Its original records lacked
  stable item IDs.  `item_linkage_candidates.json` records a conservative
  forensic recovery: 32 unique links, with 68 candidate sets left unresolved.
  That linkage stands as provenance only; the expert review and the current
  draw do not use it.
