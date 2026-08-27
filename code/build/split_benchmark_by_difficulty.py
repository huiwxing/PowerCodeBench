#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Repository copy of scripts/build/split_benchmark_by_difficulty.py, with
# imports and data-path constants adapted to this repository's layout. Runs
# CPU-only against the archived corpus/spec files in this repository. See
# code/README.md for the module map.
# --------------------------------------------------------------------------
"""Split PowerCodeBench frozen release by difficulty group.

Produces two new benchmark files for the demand-model cross-style
held-out experiment (SM):
  benchmark_D1D2.json  - 1200 items with explicit-parameter difficulty levels
  benchmark_D3D4.json  -  800 items with semantic-grounded difficulty levels

Both keep the original schema unchanged; nothing else is touched.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


D1D2_LABELS = {"D1_basic", "D2_multi_step"}
D3D4_LABELS = {"D3_semantic", "D4_compound"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "benchmark.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "benchmark/c3_splits",
    )
    args = ap.parse_args()

    items = json.loads(args.input.read_text())
    if not isinstance(items, list):
        raise SystemExit(f"Expected list at top level of {args.input}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    d12 = [it for it in items if it.get("difficulty_level") in D1D2_LABELS]
    d34 = [it for it in items if it.get("difficulty_level") in D3D4_LABELS]

    if len(d12) + len(d34) != len(items):
        skipped = len(items) - len(d12) - len(d34)
        raise SystemExit(
            f"Difficulty labels do not cover all items "
            f"({skipped} skipped of {len(items)})"
        )

    out12 = args.out_dir / "benchmark_D1D2.json"
    out34 = args.out_dir / "benchmark_D3D4.json"
    out12.write_text(json.dumps(d12, indent=2, ensure_ascii=False))
    out34.write_text(json.dumps(d34, indent=2, ensure_ascii=False))

    def _summary(label: str, group):
        diff_counts = Counter(it.get("difficulty_level") for it in group)
        task_counts = Counter(
            (it.get("scenario") or {}).get("task") for it in group
        )
        print(f"[{label}] n={len(group)}")
        print(f"  difficulty: {dict(diff_counts)}")
        print(f"  task families: {len(task_counts)}")
        for t, n in sorted(task_counts.items()):
            print(f"    {t}: {n}")

    print(f"Source: {args.input}  ({len(items)} items)")
    _summary("D1+D2", d12)
    _summary("D3+D4", d34)
    print(f"\nWrote:\n  {out12}\n  {out34}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
