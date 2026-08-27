#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Repository copy of scripts/aggregate/bootstrap_hero_ci.py from the frozen
# experimental pipeline. Its inputs are raw per-item run trees too large to
# ship here, so it does not run from this checkout; see ARTIFACT_INDEX.md for
# the outputs it produced.
# --------------------------------------------------------------------------
"""Item-level paired bootstrap 95% CI for the hero +4.95pp gap
(Llama-3.1-405B C+FDRS vs Claude-Haiku-4-5 C+FDRS).

Reads per-item match arrays from the production benchmark JSONs and
resamples paired by item index.  B=1000 default.
"""
from __future__ import annotations

import json
import random
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = 1000
SEED = 22


def load_match_array(path: Path) -> list[bool]:
    with path.open() as f:
        d = json.load(f)
    items = d["item_results"]
    items_sorted = sorted(
        items,
        key=lambda x: x.get("item_id", x.get("benchmark_original_index", 0)),
    )
    return [bool(it.get("match", False)) for it in items_sorted]


def bootstrap_acc(matches: list[bool], B: int = B, seed: int = SEED):
    rng = random.Random(seed)
    accs = []
    n = len(matches)
    for _ in range(B):
        idxs = [rng.randrange(n) for _ in range(n)]
        accs.append(sum(matches[i] for i in idxs) / n * 100)
    accs.sort()
    return st.mean(accs), accs[int(0.025 * B)], accs[int(0.975 * B)]


def main() -> int:
    llama = load_match_array(
        ROOT / "probe_eval_results/comparison/meta-llama_Llama-3.1-405B-Instruct/"
             "benchmark_results_condC_FDRS.json"
    )
    claude = load_match_array(
        ROOT / "probe_eval_results/api_comparison_2000/claude-haiku-4-5/"
             "benchmark_results_condC_FDRS.json"
    )
    print(f"Llama-3.1-405B C+FDRS:    n={len(llama)}, "
          f"matched={sum(llama)}, acc={sum(llama)/len(llama)*100:.2f}%")
    print(f"Claude-Haiku-4-5 C+FDRS:  n={len(claude)}, "
          f"matched={sum(claude)}, acc={sum(claude)/len(claude)*100:.2f}%")

    # Per-model 95% CI
    m, lo, hi = bootstrap_acc(llama)
    print(f"\nLlama-405B  bootstrap mean={m:.2f}%, 95% CI [{lo:.2f}, {hi:.2f}]")
    m, lo, hi = bootstrap_acc(claude)
    print(f"Claude-Haiku bootstrap mean={m:.2f}%, 95% CI [{lo:.2f}, {hi:.2f}]")

    # Paired bootstrap on the gap
    rng = random.Random(SEED)
    n = min(len(llama), len(claude))
    deltas = []
    for _ in range(B):
        idxs = [rng.randrange(n) for _ in range(n)]
        l_acc = sum(llama[i] for i in idxs) / n
        c_acc = sum(claude[i] for i in idxs) / n
        deltas.append((l_acc - c_acc) * 100)
    deltas.sort()
    mean_d = st.mean(deltas)
    ci_lo = deltas[int(0.025 * B)]
    ci_hi = deltas[int(0.975 * B)]
    print(f"\nPaired bootstrap (n={n}, B={B}):")
    print(f"  Δ(Llama-405B − Claude-Haiku) = {mean_d:+.2f}pp  "
          f"95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]")
    print(f"  Original point estimate: "
          f"{(sum(llama)/len(llama) - sum(claude)/len(claude))*100:+.2f}pp")

    out = {
        "method": "item-level paired bootstrap",
        "B": B,
        "seed": SEED,
        "n_items": n,
        "llama_405b_fdrs_acc_pct": sum(llama) / len(llama) * 100,
        "claude_haiku_fdrs_acc_pct": sum(claude) / len(claude) * 100,
        "paired_delta_mean_pct": mean_d,
        "paired_delta_ci_95_low_pct": ci_lo,
        "paired_delta_ci_95_high_pct": ci_hi,
    }
    out_path = ROOT / "probe_eval_results/hero_bootstrap_ci.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
