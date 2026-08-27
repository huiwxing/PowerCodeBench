# Coverage notes

These notes accompany the claim map in the repository-root
[`ARTIFACT_INDEX.md`](../ARTIFACT_INDEX.md).  Each bullet takes one evidence
family and states what this repository publishes, what can be recomputed from
it, and what the original runs kept no record of.

- **Qualitative rows.**  **T1** (related-work positioning), **T12** (deployment
  guidance), and **S12**'s identifier list are qualitative or synthesised from
  the rows of the index.  They have no independent numeric artifact, so nothing
  is recomputed for them.

- **Supplementary rebuilds.**  **S2/S3**, **S4**, and **S11** rebuild from the
  compact per-function, per-query, frozen-suite, and route-event evidence under
  `results/supplementary_evidence/`, `results/demand_suites/`, and
  `results/raw/probe_transcripts/`, through `reproduce.py --group 14`,
  `--group 25`, `--group 26`, and `--group 28`.  The default full run invokes
  all four.

- **Source-level run trees.**  Every experiment behind a reported number ships
  its source-level tree: `results/raw/main_experiment/` (270 compact runs),
  `results/raw/probe_transcripts/` (29,120 probe response/evaluator records),
  `results/raw/e2_transfer/` (24 cold-start + 9 C-recal cells),
  `results/raw/robustness_compact/` (149 model/arm/seed cells),
  `results/raw/c_unadapted/` (40 runs), the complete
  `serving/measurement_json/protocol{A,A_anchor_retest,B}/` measurement trees,
  and the `audit/` adjudication chain.  What stays unpublished is the
  production pipeline's full per-item generation and execution logs, which run
  to GB scale.  The read-only pipeline copies under `code/` that consume those
  logs record this in their file headers and do not run from this repository;
  the frozen outputs they produced are listed in the index and re-verified by
  `reproduce.py` from the public evidence alone.

- **Failure Anatomy.**  The release carries a complete 89,420-key ordered
  sampling frame plus the 100 drawn records, in place of the 2.98 GB of source
  logs behind that frame.  From the frame keys, check 17 replays the balanced
  draw and rebuilds the result using only this repository; both independent
  reviews and the full disagreement adjudication are published with it.  The
  historical export lacked benchmark item IDs, so a conservative forensic pass
  recovered 32 unique links and left 68 candidate sets unresolved, and the
  legacy 53/21/19/1/6 ledger is kept as separately validated provenance.  Full
  detail is in `audit/failure_taxonomy/README.md`.

- **Matcher false negatives.**  The 100 complete source records of this older
  audit survive; their original item-level label mapping does not.  A fresh
  adjudication over those same records supplies a public item ledger of 8 false
  negatives, 3 specification-ambiguous items, and 89 genuine errors, replacing
  the aggregate-only 5/87 split; the eight false-negative IDs reproduce every
  historical FN tier/model/task marginal in a post-hoc check.  Check 11
  rebuilds the historical provenance and the new ledger together.  Full detail
  is in `audit/matcher_fn/README.md`.

- **C3 and E4 seed-42--46 runs.**  Both kept their aggregates and their
  contemporaneous execution logs, but no per-query top-20 rankings, and the
  historical C3 runs predate deterministic split traversal, so the published
  item rankings are deterministic reruns on the pinned stack.  Check
  29 matches every logged endpoint, C3 and E4 alike, to its frozen aggregate,
  and the six C3 logs record null prediction and model exports.  Check 27 uses
  the separately labelled C3 rerun with complete per-query rankings.  The E4
  rankings are reconstructed from the frozen code, data, seeds, and pinned
  training stack, and they reproduce all 900 archived metric cells, all 10
  selected alphas, and all adapted role weights (`reproduce.py --group 22`).
  Full detail is in `results/supplementary_evidence/README.md`.

- **S4 estimators.**  Exact unfiltered item rankings are published for the three
  TF-IDF estimators, Zero-shot SBERT, Zero-shot Cross-Encoder, and the complete
  Hybrid before/after comparison; the two zero-shot neural estimators were
  reconstructed from pinned model revisions and accepted only after all seven
  frozen slices matched at every reported k.  Hybrid Cross-Encoder keeps its
  frozen aggregate curve and its historical role-filtered Bench-ref top-10,
  while its unfiltered rankings come from a deterministic rerun on the locked
  stack: the historical first-stage model and hash state were not retained, and
  in the rerun the narrowly separated alpha optimum moves from 0.3 to 0.4 with
  the conclusions unchanged.  Check 26 reaggregates both levels and keeps the
  frozen curve as the reported one.  Full detail is in
  `results/supplementary_evidence/README.md`.

- **Audit chain.**  The full E1/E1-P/T4 adjudication chain, covering judge
  transcripts, disagreements, census, refutation rounds, and the pilot
  regression, is archived under `audit/` beyond the rows of the index, so every
  audit number in the supplementary material can be traced end to end.  Private
  environment prefixes were replaced in diagnostic string fields only, with all
  outcome and score evidence unchanged; see `audit/README.md`.
