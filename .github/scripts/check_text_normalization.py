#!/usr/bin/env python3
"""Assert the exact text-normalization boundary of the public benchmark.

The public ``benchmark.json`` normalizes U+2019 (right single quotation
mark) to the ASCII apostrophe in ``natural_language_query``; the frozen
pipeline inputs (``benchmark/c3_splits/``) retain the original character.
This check pins that boundary: every one of the 2,000 queries is either
byte-identical between the two distributions or differs *only* by that
single normalization, and the number of normalized items is exactly 105.
All reported numbers were computed on the frozen versions.
"""

import json
import sys
from pathlib import Path

EXPECTED_ITEMS = 2000
EXPECTED_NORMALIZED = 105

ROOT = Path(__file__).resolve().parents[2]


def main():
    bench = {t["id"]: t["natural_language_query"]
             for t in json.load(open(ROOT / "benchmark.json", encoding="utf-8"))}
    frozen = {}
    for rel in ("benchmark/c3_splits/benchmark_D1D2.json",
                "benchmark/c3_splits/benchmark_D3D4.json"):
        for t in json.load(open(ROOT / rel, encoding="utf-8")):
            frozen[t["id"]] = t["natural_language_query"]

    errors = []
    if len(bench) != EXPECTED_ITEMS:
        errors.append(f"benchmark.json has {len(bench)} ids, "
                      f"expected {EXPECTED_ITEMS}")
    if set(bench) != set(frozen):
        errors.append("id sets differ between benchmark.json and c3_splits")

    normalized = []
    for item_id in sorted(set(bench) & set(frozen)):
        b, f = bench[item_id], frozen[item_id]
        if f == b:
            continue
        if f.replace("’", "'") == b:
            normalized.append(item_id)
        else:
            errors.append(f"{item_id}: query differs beyond the documented "
                          "U+2019 -> ASCII apostrophe normalization")

    if len(normalized) != EXPECTED_NORMALIZED:
        errors.append(f"{len(normalized)} normalized queries, "
                      f"expected exactly {EXPECTED_NORMALIZED}")

    for e in errors:
        print("BAD:", e)
    print(f"text-normalization boundary: {len(normalized)}/{len(bench)} "
          f"queries differ, all by U+2019 -> ASCII apostrophe only "
          f"(expected exactly {EXPECTED_NORMALIZED})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
