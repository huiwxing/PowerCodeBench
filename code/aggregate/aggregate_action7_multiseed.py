#!/usr/bin/env python3
"""Full-panel multi-seed aggregator (feeds Table 14).

Reads:
  results/raw/robustness_compact/seed_sweep_400/seed<seed>/<model>/
      benchmark_results_condC_FDRS.json

for all 10 open-weight panel models x 10 generation seeds. Reports per-model
mean, std, min, max, range of the C+FDRS Round-3 accuracy on the same fixed
400-item stratified subset; reports panel-mean and panel-mean std across models.
"""
from __future__ import annotations

import json
import statistics as st
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "results/raw/robustness_compact/seed_sweep_400"
DEFAULT_OUTPUT = REPO / "results/aggregates/action7_multiseed_summary.json"

# Full open-weight panel (Round 5 Table 4 / Exp-B), small → large.
PANEL = [
    ("Qwen2.5-Coder-1.5B",  "Qwen_Qwen2.5-Coder-1.5B-Instruct"),
    ("Qwen2.5-Coder-7B",    "Qwen_Qwen2.5-Coder-7B-Instruct"),
    ("Llama-3.1-8B",        "meta-llama_Llama-3.1-8B-Instruct"),
    ("Qwen2.5-Coder-14B",   "Qwen_Qwen2.5-Coder-14B-Instruct"),
    ("Qwen2.5-Coder-32B",   "Qwen_Qwen2.5-Coder-32B-Instruct"),
    ("Llama-3.1-70B",       "meta-llama_Llama-3.1-70B-Instruct"),
    ("GPT-OSS-120B",        "openai_gpt-oss-120b"),
    ("Qwen3-Coder-Next",    "Qwen_Qwen3-Coder-Next"),
    ("Llama-3.1-405B",      "meta-llama_Llama-3.1-405B-Instruct"),
    ("Qwen3-Coder-480B",    "Qwen_Qwen3-Coder-480B-A35B-Instruct"),
]
SEEDS = [22, 33, 44, 55, 66, 77, 88, 99, 111, 222]


def read_acc(p: Path) -> float | None:
    """Read C+FDRS Round-3 endpoint. Prefer summary; fall back to last round_snapshot."""
    if not p.exists():
        return None
    with p.open() as f:
        d = json.load(f)
    s = d.get("summary") or {}
    n = s.get("n_matched"); t = s.get("total")
    if n is not None and t:
        items = d.get("item_results") or []
        if items and (len(items) != t or sum(bool(x.get("match")) for x in items) != n):
            raise ValueError(f"compact item outcomes disagree with summary: {p}")
        return n / t * 100
    snaps = d.get("round_snapshots") or []
    if not snaps:
        return None
    last = snaps[-1]
    if last.get("round") != 3:
        return None
    n = last.get("n_matched"); t = last.get("total")
    if n is None or not t:
        return None
    return n / t * 100


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print("Multi-seed variance (C+FDRS Round-3, 400-item stratified subset)")
    print("=" * 100)
    print(f"{'Model':<22} {'n_seeds':>8} {'mean':>9} {'sample_std':>11} {'range':>16} {'spread':>9}")
    print("-" * 100)

    out = {"per_model": {}, "panel_summary": {}}
    panel_means = []
    panel_stds = []
    spreads = []
    for label, mdir in PANEL:
        accs = []
        per_seed = {}
        for seed in SEEDS:
            p = args.input_root / f"seed{seed}" / mdir / "benchmark_results_condC_FDRS.json"
            acc = read_acc(p)
            if acc is not None:
                accs.append(acc)
                per_seed[seed] = acc
        if not accs:
            print(f"{label:<22} {0:>8}        —            —                —         —")
            continue
        mean = st.mean(accs)
        sstd = st.stdev(accs) if len(accs) > 1 else 0.0
        mn, mx = min(accs), max(accs)
        spread = mx - mn
        print(f"{label:<22} {len(accs):>8} {mean:>8.2f}% {sstd:>10.3f}pp "
              f"[{mn:>5.2f},{mx:>5.2f}] {spread:>7.2f}pp")
        out["per_model"][label] = {
            "model_dir": mdir,
            "n_seeds_complete": len(accs),
            "per_seed_acc_pct": per_seed,
            "mean_pct": mean,
            "sample_std_pp": sstd,
            "min_pct": mn,
            "max_pct": mx,
            "range_pp": spread,
        }
        if len(accs) == 10:  # only count fully-completed cells in panel summary
            panel_means.append(mean)
            panel_stds.append(sstd)
            spreads.append(spread)

    print("-" * 100)
    if panel_means:
        # Panel-mean std across models is the typical "noise band" reference for
        # FDRS-FDR / FDR-FD / C-R panel-mean deltas (within-cell seed variance).
        pmean_across_models = st.mean(panel_means)
        std_avg = st.mean(panel_stds)
        std_max = max(panel_stds)
        std_med = st.median(panel_stds)
        spread_avg = st.mean(spreads)
        spread_max = max(spreads)
        print(f"Panel summary (across {len(panel_means)} fully-completed models):")
        print(f"  Panel-mean C+FDRS accuracy:    {pmean_across_models:.2f}%")
        print(f"  Mean per-cell sample std:      {std_avg:.3f}pp")
        print(f"  Max per-cell sample std:       {std_max:.3f}pp")
        print(f"  Median per-cell sample std:    {std_med:.3f}pp")
        print(f"  Mean per-cell spread (max-min): {spread_avg:.2f}pp")
        print(f"  Max per-cell spread:           {spread_max:.2f}pp")
        out["panel_summary"] = {
            "n_models_complete": len(panel_means),
            "panel_mean_acc_pct": pmean_across_models,
            "mean_per_cell_sample_std_pp": std_avg,
            "max_per_cell_sample_std_pp": std_max,
            "median_per_cell_sample_std_pp": std_med,
            "mean_per_cell_spread_pp": spread_avg,
            "max_per_cell_spread_pp": spread_max,
        }
    save_path = args.output
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
