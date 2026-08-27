#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Repository copy of scripts/aggregate/aggregate_mw2_paired_bootstrap.py from
# the frozen experimental pipeline. Its inputs are raw per-item run trees too
# large to ship here, so it does not run from this checkout; see
# ARTIFACT_INDEX.md for the outputs it produced.
# --------------------------------------------------------------------------
"""Paired-bootstrap CI on panel-mean reactive-loop deltas (Table 14 CI column).

Item-level bootstrap. For each of B resamples we draw the 2000 benchmark
items with replacement (paired across all models and conditions), recompute
each model's per-condition accuracy on the resample, then take the
panel-mean delta. The 95% percentile interval gives the CI.

Used to certify per-condition reactive-loop lifts (FDRS-FDR, FDR-FD, C-R,
...) as strictly positive against item-level noise.
"""
from __future__ import annotations

import json
import random
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "probe_eval_results" / "comparison"

PANEL = [
    "Qwen_Qwen2.5-Coder-1.5B-Instruct",
    "Qwen_Qwen2.5-Coder-7B-Instruct",
    "meta-llama_Llama-3.1-8B-Instruct",
    "Qwen_Qwen2.5-Coder-14B-Instruct",
    "Qwen_Qwen2.5-Coder-32B-Instruct",
    "meta-llama_Llama-3.1-70B-Instruct",
    "openai_gpt-oss-120b",
    "Qwen_Qwen3-Coder-Next",
    "meta-llama_Llama-3.1-405B-Instruct",
    "Qwen_Qwen3-Coder-480B-A35B-Instruct",
]

# Deltas we want CIs for. (a, b) means a − b, by panel-mean accuracy.
DELTAS = [
    ("C+FDRS", "C+FDR", "benchmark_results_condC_FDRS.json", "benchmark_results_condC_FDR.json"),
    ("C+FDR",  "C+FD",  "benchmark_results_condC_FDR.json",  "benchmark_results_condC_FD.json"),
    ("C+FX",   "A+FX",  "benchmark_results_condC_FX.json",   "benchmark_results_condA_FX.json"),
    ("C",      "A",     "benchmark_results_condC.json",      "benchmark_results_condA.json"),
    ("C+FDRS", "C+FX",  "benchmark_results_condC_FDRS.json", "benchmark_results_condC_FX.json"),
]

B_BOOTSTRAP = 2000
SEED = 22


def load_match(p: Path) -> dict[str, bool] | None:
    if not p.exists():
        return None
    with p.open() as f:
        d = json.load(f)
    items = d.get("item_results") or []
    return {x["item_id"]: bool(x.get("match")) for x in items if x.get("item_id") is not None}


def main() -> int:
    rng = random.Random(SEED)

    # Load all (model, file) match arrays we need.
    needed_files = set()
    for _, _, fa, fb in DELTAS:
        needed_files.add(fa); needed_files.add(fb)
    data: dict[tuple[str, str], dict[str, bool]] = {}
    for m in PANEL:
        for fn in needed_files:
            p = ROOT / m / fn
            d = load_match(p)
            if d is not None:
                data[(m, fn)] = d
    if not data:
        print("No data loaded — aborting.")
        return 1

    # Determine common item_ids: intersect across (model × needed_file) sets.
    common_ids = None
    for k, d in data.items():
        ids = set(d.keys())
        common_ids = ids if common_ids is None else (common_ids & ids)
    if not common_ids:
        print("No common items across panel — aborting.")
        return 1
    ids = sorted(common_ids)
    n_items = len(ids)
    print(f"Loaded panel: {len(PANEL)} models × {len(needed_files)} files; common items = {n_items}")
    print()

    def panel_delta_mean(file_a: str, file_b: str, item_ids: list[str]) -> float | None:
        """Mean across PANEL of (acc_a − acc_b) over the given items."""
        deltas = []
        for m in PANEL:
            da = data.get((m, file_a)); db = data.get((m, file_b))
            if da is None or db is None: continue
            try:
                acc_a = sum(da[i] for i in item_ids) / len(item_ids) * 100
                acc_b = sum(db[i] for i in item_ids) / len(item_ids) * 100
            except KeyError:
                continue
            deltas.append(acc_a - acc_b)
        if not deltas:
            return None
        return st.mean(deltas)

    # Point estimate + bootstrap distribution.
    rows = []
    for label_a, label_b, fa, fb in DELTAS:
        point = panel_delta_mean(fa, fb, ids)
        # Bootstrap
        bs = []
        for _ in range(B_BOOTSTRAP):
            resample = [ids[rng.randrange(n_items)] for _ in range(n_items)]
            v = panel_delta_mean(fa, fb, resample)
            if v is not None:
                bs.append(v)
        bs_sorted = sorted(bs)
        lo = bs_sorted[int(0.025 * len(bs_sorted))]
        hi = bs_sorted[int(0.975 * len(bs_sorted)) - 1]
        median = bs_sorted[len(bs_sorted) // 2]
        rows.append({
            "delta_label": f"{label_a} − {label_b}",
            "file_a": fa, "file_b": fb,
            "point_pp": point,
            "boot_median_pp": median,
            "ci_low_pp": lo,
            "ci_high_pp": hi,
            "strict_positive": lo > 0,
            "strict_negative": hi < 0,
            "n_bootstrap": len(bs_sorted),
            "n_items": n_items,
        })

    print(f"Paired-bootstrap CI (B={B_BOOTSTRAP}, seed={SEED}, n={n_items} items, "
          f"resampling item-IDs with replacement; panel-mean across {len(PANEL)} open-weight models)")
    print("=" * 90)
    print(f"{'Δ':<18} {'point':>9} {'boot med':>9} {'95% CI':>22} {'strict?':>10}")
    print("-" * 90)
    for r in rows:
        ci = f"[{r['ci_low_pp']:+6.2f}, {r['ci_high_pp']:+6.2f}]"
        verdict = "pos" if r["strict_positive"] else ("neg" if r["strict_negative"] else "0 spans")
        print(f"{r['delta_label']:<18} {r['point_pp']:>+8.2f}pp {r['boot_median_pp']:>+8.2f}pp "
              f"{ci:>22} {verdict:>10}")

    save_path = ROOT.parent / "mw2_panel_mean_paired_bootstrap_ci.json"
    save_path.write_text(json.dumps({"rows": rows, "n_items": n_items,
                                      "n_panel": len(PANEL),
                                      "n_bootstrap": B_BOOTSTRAP,
                                      "seed": SEED}, indent=2))
    print(f"\nSaved: {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
