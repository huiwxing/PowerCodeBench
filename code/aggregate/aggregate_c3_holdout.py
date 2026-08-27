#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Repository copy of scripts/aggregate/aggregate_c3_holdout.py, with path
# constants adapted to this repository's layout. It runs here against the
# archived artifacts; see ARTIFACT_INDEX.md.
# --------------------------------------------------------------------------
"""Aggregate C3 held-out cross-evaluation results into a publication-ready table.

Reads task_demand/results/c3_holdout/R*/metrics.json and emits the recall@10 /
hit@10 / recall@20 numbers along with the held-out vs in-distribution lifts.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "results/aggregates/c3_holdout"


def _short_name(p: str) -> str:
    if not p or p == "(none)":
        return "baseline"
    return p.split("/")[-1].replace("benchmark_", "").replace(".json", "")


def main() -> int:
    if not ROOT.exists():
        print(f"No C3 results at {ROOT}; run scripts/submit/run_c3_holdout.sh first.")
        return 1
    runs = sorted(d.name for d in ROOT.iterdir() if d.is_dir() and d.name.startswith("R"))

    print(f"{'Run':<32} {'reweight':<10} {'eval':<6} {'recall@10':>10} {'hit@10':>8} {'recall@20':>10}")
    print("-" * 85)
    results: dict[str, dict] = {}
    for run in runs:
        metrics_path = ROOT / run / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        bench = m.get("metrics", {}).get("benchmark_reference_eval_only", {}).get("top_k", {})
        if "10" not in bench:
            continue
        cfg = m.get("config", {})
        rw = cfg.get("reweight_benchmark_path") or "(none)"
        ev = cfg.get("benchmark_path", "")
        rw_short = _short_name(rw)
        ev_short = _short_name(ev)
        r10 = bench["10"]["recall"]
        h10 = bench["10"]["hit_rate"]
        r20 = bench["20"]["recall"]
        results[run] = {
            "recall@10": r10, "hit@10": h10, "recall@20": r20,
            "reweight": rw_short, "eval": ev_short,
        }
        print(f"{run:<32} {rw_short:<10} {ev_short:<6} {r10:>10.4f} {h10:>8.4f} {r20:>10.4f}")

    print()
    print("=== Held-out reweighting lifts ===")
    print()
    pairs = [
        ("R2_baseline_eval_D34", "R4_reweight_D12_eval_D34",
         "D34 eval, source=D12 (HELD-OUT)"),
        ("R1_baseline_eval_D12", "R6_reweight_D34_eval_D12",
         "D12 eval, source=D34 (HELD-OUT, reverse)"),
    ]
    for base, rew, label in pairs:
        if base in results and rew in results:
            b = results[base]["recall@10"]
            r = results[rew]["recall@10"]
            print(f"  {label:<55} {b:.4f} -> {r:.4f}  Δ = {(r-b)*100:+.2f}pp")

    print()
    print("=== In-distribution controls (same split for source and eval) ===")
    print()
    in_dist = [
        ("R1_baseline_eval_D12", "R3_reweight_D12_eval_D12",
         "D12 eval, source=D12 (in-dist)"),
        ("R2_baseline_eval_D34", "R5_reweight_D34_eval_D34",
         "D34 eval, source=D34 (in-dist)"),
    ]
    for base, rew, label in in_dist:
        if base in results and rew in results:
            b = results[base]["recall@10"]
            r = results[rew]["recall@10"]
            print(f"  {label:<55} {b:.4f} -> {r:.4f}  Δ = {(r-b)*100:+.2f}pp")

    print()
    print("=== Interpretation ===")
    print("If held-out Δ is comparable to in-distribution Δ, the role-frequency")
    print("reweighting is genuinely transferring across query styles rather than")
    print("merely overfitting the source distribution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
