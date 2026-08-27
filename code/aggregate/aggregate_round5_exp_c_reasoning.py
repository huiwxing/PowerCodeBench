#!/usr/bin/env python3
"""Round 5 Exp-C aggregation: reasoning-on (gpt-5.4-mini reasoning_high,
deepseek-reasoner R1-style) vs vanilla baselines on a paired 200-item subset.

Reads:
  reusable main compact/api_comparison_2000/...                                 (vanilla)
  results/raw/robustness_compact/api_comparison_reasoning_200/...               (reasoning)
  results/aggregates/api_comparison_reasoning_200/subset_items.json             (200 ids)

Vanilla full-2000 results are subset to the same 200 item_ids for paired comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAIN_COMPACT = ROOT / "results/raw/main_experiment/primary_outcomes_compact.json"
DEFAULT_RSN = ROOT / "results/raw/robustness_compact/api_comparison_reasoning_200"
DEFAULT_SUBSET = ROOT / "results/aggregates/api_comparison_reasoning_200/subset_items.json"
DEFAULT_OUTPUT = ROOT / "results/aggregates/round5_exp_c_reasoning_summary.json"

CONDITIONS = ["A", "C", "C_FX", "C_FDR", "C_FDRS"]
PAIRS = [
    ("gpt-5.4-mini",     "gpt-5.4-mini"),       # vanilla dir, reasoning dir
    ("deepseek-v4-flash", "deepseek-reasoner"),
]


def load_subset_ids(path: Path) -> set[str]:
    with path.open() as f:
        return set(json.load(f)["item_ids"])


def acc_subset(p: Path, ids: set[str]) -> tuple[float | None, int | None]:
    """Subset full-2000 item_results to given ids; return (acc%, n_matched)."""
    if not p.exists():
        return None, None
    with p.open() as f:
        d = json.load(f)
    items = d.get("item_results") or []
    sub = [x for x in items if x.get("item_id") in ids]
    if len(sub) != 200:
        return None, None
    m = sum(1 for x in sub if x.get("match") is True)
    return m / 200 * 100, m


def acc_full(p: Path) -> tuple[float | None, int | None]:
    """Read summary.n_matched / summary.total."""
    if not p.exists():
        return None, None
    with p.open() as f:
        d = json.load(f)
    s = d.get("summary") or {}
    n = s.get("n_matched"); t = s.get("total")
    if n is None or t is None:
        return None, None
    return n / t * 100, n


def compact_acc(main: dict, model: str, condition: str,
                subset_ids: set[str] | None = None) -> tuple[float, int]:
    axis = main["datasets"]["powercodebench_2000"]["item_ids"]
    bits = main["runs"][f"api|{model}|{condition}"]["match_bits"]
    if len(axis) != len(bits):
        raise ValueError(f"invalid API compact vector for {model}/{condition}")
    selected = [bit for iid, bit in zip(axis, bits)
                if subset_ids is None or iid in subset_ids]
    expected = len(axis) if subset_ids is None else len(subset_ids)
    if len(selected) != expected:
        raise ValueError(f"API compact subset mismatch for {model}/{condition}")
    matched = selected.count("1")
    return matched / len(selected) * 100, matched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-compact", type=Path, default=DEFAULT_MAIN_COMPACT,
        help="Path to primary_outcomes_compact.json",
    )
    parser.add_argument("--reasoning-root", type=Path, default=DEFAULT_RSN)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main_compact = json.loads(args.main_compact.read_text())
    rsn = args.reasoning_root
    ids = load_subset_ids(args.subset)
    print(f"Subset: {len(ids)} item_ids")
    print()
    print("Exp-C Reasoning-on vs Vanilla (paired on identical 200 items)")
    print("=" * 100)
    header = (f"{'Vendor':<22} {'Cond':<7} {'Vanilla(full)':>14} {'Vanilla(200)':>13} "
              f"{'Reasoning(200)':>15} {'Δ vs vanilla':>14}")
    print(header)
    print("-" * 100)

    rows = []
    for van_dir, rsn_dir in PAIRS:
        for cond in CONDITIONS:
            fn = f"benchmark_results_cond{cond}.json"
            v_full, _ = compact_acc(main_compact, van_dir, cond)
            v_sub, v_sub_n = compact_acc(main_compact, van_dir, cond, ids)
            r_200, r_200_n = acc_full(rsn / rsn_dir / fn)
            delta = (r_200 - v_sub) if (r_200 is not None and v_sub is not None) else None

            f_pct = lambda x, w=14: f"{x:>{w-1}.2f}%" if x is not None else " " * (w-2) + "— "
            d_str = f"{delta:>+13.2f}pp" if delta is not None else "             —"
            print(f"{van_dir:<22} {cond:<7} {f_pct(v_full)} {f_pct(v_sub,13)} "
                  f"{f_pct(r_200,15)} {d_str}")
            rows.append({
                "vanilla_dir": van_dir, "reasoning_dir": rsn_dir, "condition": cond,
                "vanilla_full_2000_acc": v_full,
                "vanilla_subset_200_acc": v_sub, "vanilla_subset_200_n_matched": v_sub_n,
                "reasoning_200_acc": r_200, "reasoning_200_n_matched": r_200_n,
                "delta_reasoning_minus_vanilla_pp": delta,
            })
        print("-" * 100)

    # Within-vendor lifts for branch (a)/(b) verdict.
    print()
    print("Intervention lift (R0 → C, R0 → C_FDRS) under each regime")
    print("=" * 100)
    print(f"{'Vendor':<22} {'Reg':<10} {'R0(A)':>8} {'C':>8} {'C_FDRS':>9} "
          f"{'Δ C-A':>8} {'Δ FDRS-A':>11}")
    print("-" * 100)
    lifts = {}
    for van_dir, rsn_dir in PAIRS:
        for regime in ("vanilla", "reasoning"):
            d = van_dir if regime == "vanilla" else rsn_dir
            if regime == "vanilla":
                getter = lambda condition: compact_acc(main_compact, d, condition, ids)
            else:
                getter = lambda condition: acc_full(
                    rsn / d / f"benchmark_results_cond{condition}.json")
            a, _ = getter("A")
            c, _ = getter("C")
            fd, _ = getter("C_FDRS")
            d_ca = (c - a) if (c is not None and a is not None) else None
            d_fa = (fd - a) if (fd is not None and a is not None) else None
            f = lambda x, w=8, suf="%": f"{x:>{w-1}.2f}{suf}" if x is not None else " " * (w-2) + "— "
            f2 = lambda x: f"{x:>+7.2f}pp" if x is not None else "       —"
            f3 = lambda x: f"{x:>+10.2f}pp" if x is not None else "          —"
            print(f"{d:<22} {regime:<10} {f(a)} {f(c)} {f(fd,9)} {f2(d_ca)} {f3(d_fa)}")
            lifts[(van_dir, regime)] = {"r0": a, "c": c, "c_fdrs": fd, "d_ca": d_ca, "d_fdrs_a": d_fa}
        print("-" * 100)

    out = {
        "rows": rows,
        "intervention_lift": {f"{k[0]}_{k[1]}": v for k, v in lifts.items()},
    }
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
