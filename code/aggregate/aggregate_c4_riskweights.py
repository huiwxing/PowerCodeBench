#!/usr/bin/env python3
"""Aggregate C4 risk-weighting sensitivity results.

Compares base C R0 accuracy across three weight configurations:
  default  (existing comparison/, weights = 0.20/0.25/0.20/0.35)
  uniform  (comparison_riskw_uniform/, weights = 0.25/0.25/0.25/0.25)
  l3heavy  (comparison_riskw_l3heavy/, weights = 0.0/0.20/0.20/0.60)

Reports the maximum R0 accuracy swing across the L0--L3 weight sweep, per
model and overall.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAIN_COMPACT = ROOT / "results/raw/main_experiment/primary_outcomes_compact.json"
DEFAULT_ROBUSTNESS_COMPACT = ROOT / "results/raw/robustness_compact"

MODELS = [
    "Qwen_Qwen2.5-Coder-32B-Instruct",
    "meta-llama_Llama-3.1-70B-Instruct",
    "Qwen_Qwen3-Coder-480B-A35B-Instruct",
]


def _load_acc(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception:
        return None
    summary = data.get("summary") or data
    for key in ("accuracy", "acc", "matched_rate"):
        v = summary.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    matched = summary.get("matched") or summary.get("n_matched")
    n = summary.get("n_items") or summary.get("total")
    if matched is not None and n:
        return matched / n
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-compact", type=Path, default=DEFAULT_MAIN_COMPACT,
        help="Path to primary_outcomes_compact.json",
    )
    parser.add_argument("--robustness-compact", type=Path,
                        default=DEFAULT_ROBUSTNESS_COMPACT)
    args = parser.parse_args()
    main_compact = json.loads(args.main_compact.read_text())
    uniform_dir = args.robustness_compact / "comparison_riskw_uniform"
    l3heavy_dir = args.robustness_compact / "comparison_riskw_l3heavy"

    print(f"{'Model':<40} {'default':>9} {'uniform':>9} {'l3heavy':>9} {'swing':>8}")
    print("-" * 80)
    swings = []
    for m in MODELS:
        bits = main_compact["runs"][f"comparison|{m}|C"]["match_bits"]
        d = bits.count("1") / len(bits)
        u = _load_acc(uniform_dir / m / "benchmark_results_condC.json")
        l = _load_acc(l3heavy_dir / m / "benchmark_results_condC.json")
        d_s = f"{d*100:.2f}" if d is not None else "—"
        u_s = f"{u*100:.2f}" if u is not None else "—"
        l_s = f"{l*100:.2f}" if l is not None else "—"
        vals = [v for v in (d, u, l) if v is not None]
        if len(vals) >= 2:
            swing = (max(vals) - min(vals)) * 100
            swings.append(swing)
            sw_s = f"±{swing/2:.2f}pp"
        else:
            sw_s = "—"
        print(f"{m:<40} {d_s:>9} {u_s:>9} {l_s:>9} {sw_s:>8}")

    print()
    print("=== Interpretation ===")
    if swings:
        max_swing = max(swings)
        print(f"Max R0 swing across weight configs: {max_swing:.2f}pp")
        if max_swing < 2.0:
            print("→ Method is INSENSITIVE to L0-L3 weighting. Paper claim of")
            print("  'simple uniform weighting suffices' is empirically supported.")
        elif max_swing < 5.0:
            print("→ Method is moderately sensitive to weighting. Worth a brief")
            print("  appendix paragraph noting that production weights were chosen")
            print("  as a small refinement over uniform.")
        else:
            print("→ Method is sensitive to weighting. Recommend formal lever in")
            print("  the paper, possibly fitted weights vs the current heuristic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
