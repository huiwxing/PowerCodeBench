# E2 D-6 import-scan evidence

In the E2 transfer pilot, the pre-fix bare-condition prompt named
`pandapower` instead of the backend under test.  This directory holds the
generations and the frozen scan reports for that defect, D-6, and for the
fix.

## Public evidence

- `pre_fix_generations.json` contains all 1,080 pre-fix generations from
  12 frozen result cells: two backends, two models, and three conditions
  (`A`, `C`, and `Rsem`), with 90 items per cell. It retains each item's
  identity and complete `generated_code`, but omits prompts, references,
  execution output, and other fields irrelevant to the import scan. Stored
  flags, matching import lines, code byte length, and code SHA-256 are
  cross-checks; the public audit reruns the regex over the complete code.
- `frozen/import_scan_prefix_B2.json` and
  `frozen/import_scan_v2_postfix.json` are byte-exact copies of the two
  historical frozen scan reports.
- The 2,160 post-fix generations are already distributed as the 24 complete
  raw JSONs under `results/raw/e2_transfer/{opendss,pypsa}/base/`. Their
  file hashes and source provenance are frozen in
  `results/raw/e2_transfer/manifest.json`.

## Public recomputation

From the repository root, run:

```bash
python3 code/audit/recompute_e2_import_scans.py
```

The standard-library-only script rescans all 3,240 generations and writes
`results/aggregates/e2_import_scans/`. It fails unless the pre-fix and
post-fix reports it reconstructs are byte-identical to the historical
reports. The expected checks are:

- pre-fix condition A: 360/360 pandapower imports and 0/360 target-library
  imports;
- post-fix frozen 13-file scope: 0/1,170 pandapower imports, 629/630 OpenDSS
  and 540/540 PyPSA target-library imports;
- post-fix full 24-file scope: 0/2,160 pandapower imports and 2,158/2,160
  target-library imports.

Maintainers who possess the frozen source repository can independently
re-export the compact pre-fix records before running the same audit:

```bash
python3 code/audit/recompute_e2_import_scans.py \
  --export-pre-source /path/to/igpt
```

The historical reports recorded `id: null` for non-target items because the
original scan requested an absent `id` field. The compact evidence retains
the actual `item_id`, and the reconstruction keeps the original `null` so the
rebuilt reports match the historical ones.
