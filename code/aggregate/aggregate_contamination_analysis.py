#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Repository copy of scripts/aggregate/aggregate_contamination_analysis.py,
# adapted only to the public repository layout. All inputs are archived here,
# so this analysis runs from a standalone checkout without the private
# experimental tree.
# --------------------------------------------------------------------------
"""Quantify overlap between augmented_dataset and PowerCodeBench.

Computes:
  - Function-label set Jaccard
  - Bench-side coverage by augmented training functions
  - Bench-only functions (training never sees)
  - Token-level vocabulary Jaccard between query corpora
  - Per-query best-match Jaccard distribution (n=500 bench × 5000 aug)

Writes the public aggregate (or ``--out`` path) for the paper's corpus-
separation diagnostic.  The implementation is standard-library-only so the
result can be regenerated in a clean artifact checkout.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TOK = re.compile(r"[a-z][a-z0-9_]+")
PP_CALL_RE = re.compile(r"\bpp\.([A-Za-z_]\w*)\s*\(")
PN_CALL_RE = re.compile(r"\bpn\.([A-Za-z_]\w*)\s*\(")
SC_CALL_RE = re.compile(r"\b(?:sc|shortcircuit)\.([A-Za-z_]\w*)\s*\(")
IMPORT_FROM_RE = re.compile(
    r"\bfrom\s+pandapower(?:\.[\w.]+)?\s+import\s+([^\n]+)")


def tokenise(text: str) -> set[str]:
    return set(TOK.findall(text.lower()))


def function_names() -> set[str]:
    raw = json.loads(
        (ROOT / "dataset/pandapower_docs.json").read_text(encoding="utf-8"))
    return {
        str(row.get("name", "")).strip()
        for row in raw.get("functions", [])
        if str(row.get("name", "")).strip()
    }


def augmented_examples(fnames: set[str]) -> list[tuple[str, set[str]]]:
    raw = json.loads(
        (ROOT / "dataset/augmented_dataset.json").read_text(encoding="utf-8"))
    examples = []
    for row in raw.get("samples", []):
        question = str(row.get("question", "")).strip()
        functions = {
            str(name).strip() for name in (row.get("functions") or [])
            if str(name).strip() in fnames
        }
        if question and functions:
            examples.append((question, functions))
    return examples


def reference_functions(code: str, fnames: set[str]) -> set[str]:
    functions = set(PP_CALL_RE.findall(code))
    functions.update(PN_CALL_RE.findall(code))
    functions.update(SC_CALL_RE.findall(code))
    for imports in IMPORT_FROM_RE.findall(code):
        for part in imports.split(","):
            name = part.strip().split(" as ", 1)[0].strip()
            if name in fnames:
                functions.add(name)
    return functions & fnames


def benchmark_examples(fnames: set[str]) -> list[tuple[str, set[str]]]:
    raw = json.loads((ROOT / "benchmark.json").read_text(encoding="utf-8"))
    examples = []
    for row in raw:
        functions = reference_functions(row.get("reference_code", ""), fnames)
        if functions:
            examples.append((
                str(row.get("natural_language_query", "")).strip(), functions))
    return examples


def main(out_path: Path) -> int:
    fnames = function_names()
    print(f"Function corpus (pandapower_docs): {len(fnames)} canonical entries")

    aug = augmented_examples(fnames)
    bench = benchmark_examples(fnames)

    aug_funcs = set().union(*(functions for _, functions in aug))
    bench_funcs = set().union(*(functions for _, functions in bench))

    inter = aug_funcs & bench_funcs
    union = aug_funcs | bench_funcs

    print(f"Aug:   n={len(aug)}, unique funcs={len(aug_funcs)}")
    print(f"Bench: n={len(bench)}, unique funcs={len(bench_funcs)}")
    print(f"Function intersection: {len(inter)}, union: {len(union)}, "
          f"Jaccard: {len(inter)/len(union):.3f}")
    print(f"Bench coverage by aug: {len(inter)/len(bench_funcs)*100:.1f}%")

    bench_only = bench_funcs - aug_funcs
    print(f"Bench-only functions ({len(bench_only)}): {sorted(bench_only)}")

    aug_qs = [question for question, _ in aug]
    bench_qs = [question for question, _ in bench]
    aug_v = set().union(*[tokenise(q) for q in aug_qs])
    bench_v = set().union(*[tokenise(q) for q in bench_qs])
    print(f"Token Jaccard: {len(aug_v & bench_v)/len(aug_v | bench_v):.3f}")

    sample = bench_qs[:500]
    aug_token_sets = [tokenise(q) for q in aug_qs[:5000]]
    mean_max_jac = []
    for bq in sample:
        bq_set = tokenise(bq)
        if not bq_set:
            continue
        best = max(
            (len(bq_set & a) / max(1, len(bq_set | a))
             for a in aug_token_sets if a),
            default=0,
        )
        mean_max_jac.append(best)
    print(f"Per-query best-match Jaccard mean={st.mean(mean_max_jac):.3f} "
          f"median={st.median(mean_max_jac):.3f} max={max(mean_max_jac):.3f}")
    print(f"Near-clones (>0.7): {sum(1 for j in mean_max_jac if j > 0.7)}")
    print(f"High-sim (>0.5):    {sum(1 for j in mean_max_jac if j > 0.5)}")

    out = {
        "aug_n_samples": len(aug),
        "aug_unique_functions": len(aug_funcs),
        "bench_n_items": len(bench),
        "bench_unique_functions": len(bench_funcs),
        "function_intersection": len(inter),
        "function_union": len(union),
        "function_jaccard": len(inter) / len(union),
        "bench_function_coverage_by_aug": len(inter) / len(bench_funcs),
        "aug_function_coverage_by_bench": len(inter) / len(aug_funcs),
        "bench_only_function_count": len(bench_only),
        "bench_only_functions": sorted(bench_only),
        "aug_only_function_count": len(aug_funcs - bench_funcs),
        "token_jaccard": len(aug_v & bench_v) / len(aug_v | bench_v),
        "bench_token_coverage_by_aug": len(aug_v & bench_v) / len(bench_v),
        "per_query_best_match_jaccard_mean": st.mean(mean_max_jac),
        "per_query_best_match_jaccard_median": st.median(mean_max_jac),
        "per_query_best_match_jaccard_max": max(mean_max_jac),
        "per_query_near_clone_count_0.7":
            sum(1 for j in mean_max_jac if j > 0.7),
        "per_query_high_sim_count_0.5":
            sum(1 for j in mean_max_jac if j > 0.5),
        "per_query_sample_size": len(mean_max_jac),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "results/aggregates/contamination_analysis.json",
        help="output JSON (default: archived aggregate in this repository)")
    args = parser.parse_args()
    raise SystemExit(main(args.out))
