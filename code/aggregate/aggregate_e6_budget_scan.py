#!/usr/bin/env python3
"""E6.1 budget-scan aggregation (S5): Delta(gate-on - gate-off) + paired-bootstrap CI.

Protocol authority: E6_risk_gate.md (2.1 / 3 / 8).
Grid = 4 budgets {600,800,1000,2000} x 2 arms {gate-on, gate-off} x 3 models
       {14B, 70B, 480B} = 24 cells, all on the SAME stratified 400-item subset
       (seed=22, strata=difficulty,task; sha256(str(selected_indices))[:16]
       == ceba3d55f3115085 -- verified here as the pairing precondition).

Bootstrap mechanism reused verbatim from scripts/aggregate/aggregate_mw2_paired_bootstrap.py:
item-level resampling with replacement, paired across every model x budget x arm
(one resample drives all cells), B=2000, seed=22, 95% percentile interval.

Injection-token curve: per-item injected tokens = prompt_tokens(cell condC)
                       - prompt_tokens(condA baseline, same item_id) from the
                       frozen comparison/ tree (read-only).

Outputs:
  results/aggregates/e6_budget/e6_budget_scan_summary.json
  markdown table on stdout (for appending to E6 doc 8)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCAN = ROOT / "results/raw/robustness_compact/e6_budget"
DEFAULT_MAIN_COMPACT = ROOT / "results/raw/main_experiment/primary_outcomes_compact.json"
DEFAULT_OUTPUT = ROOT / "results/aggregates/e6_budget/e6_budget_scan_summary.json"

MODELS = [
    "Qwen_Qwen2.5-Coder-14B-Instruct",
    "meta-llama_Llama-3.1-70B-Instruct",
    "Qwen_Qwen3-Coder-480B-A35B-Instruct",
]
SHORT = {
    "Qwen_Qwen2.5-Coder-14B-Instruct": "14B",
    "meta-llama_Llama-3.1-70B-Instruct": "70B",
    "Qwen_Qwen3-Coder-480B-A35B-Instruct": "480B",
}
BUDGETS = [600, 800, 1000, 2000]
ARMS = ["on", "off"]

EXPECTED_SHA = "ceba3d55f3115085"
B_BOOTSTRAP = 2000
SEED = 22


def cell_path(scan: Path, budget: int, arm: str, model: str) -> Path:
    return scan / f"{budget}_gate-{arm}" / model / "benchmark_results_condC.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-root", type=Path, default=DEFAULT_SCAN)
    parser.add_argument(
        "--main-compact", type=Path, default=DEFAULT_MAIN_COMPACT,
        help="Path to primary_outcomes_compact.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main_compact = json.loads(args.main_compact.read_text())

    # ---------------- load 24 cells + pairing verification ----------------
    cells: dict[tuple[int, str, str], dict] = {}
    pairing = []
    for b in BUDGETS:
        for a in ARMS:
            for m in MODELS:
                p = cell_path(args.scan_root, b, a, m)
                if not p.exists():
                    print(f"MISSING CELL: {p}")
                    return 1
                d = json.load(p.open())
                bs = d["config"]["benchmark_sample"]
                sha = hashlib.sha256(str(bs["selected_indices"]).encode()).hexdigest()[:16]
                items = d["item_results"]
                cells[(b, a, m)] = {
                    "match": {x["item_id"]: bool(x.get("match")) for x in items},
                    "ptok": {x["item_id"]: x.get("prompt_tokens") for x in items},
                    "summary": d["summary"],
                    "sha": sha,
                    "sample": {k: v for k, v in bs.items() if k not in ("selected_indices", "selected_ids")},
                }
                pairing.append({
                    "budget": b, "arm": a, "model": m,
                    "sha16": sha, "sha_ok": sha == EXPECTED_SHA,
                    "n_items": len(items),
                    "summary_result_accuracy": d["summary"]["result_accuracy"],
                    "sample_mode": bs.get("mode"), "sample_seed": bs.get("seed"),
                    "sample_strata": bs.get("strata"), "max_items": bs.get("max_items"),
                })

    sha_all_ok = all(r["sha_ok"] for r in pairing)

    # item_id set identity across all 24 cells
    id_sets = [set(v["match"].keys()) for v in cells.values()]
    common = set.intersection(*id_sets)
    union = set.union(*id_sets)
    ids_identical = (common == union) and len(common) == 400
    ids = sorted(common)
    n_items = len(ids)

    # recomputed vs reported accuracy cross-check
    acc_mismatch = []
    for key, v in cells.items():
        recomputed = sum(v["match"][i] for i in ids) / n_items
        reported = v["summary"]["result_accuracy"]
        if abs(recomputed - reported) > 1e-6:
            acc_mismatch.append({"cell": list(key), "recomputed": recomputed, "reported": reported})

    # ---------------- point estimates ----------------
    def acc(b: int, a: str, m: str, item_ids: list[str]) -> float:
        d = cells[(b, a, m)]["match"]
        return sum(d[i] for i in item_ids) / len(item_ids) * 100.0

    def all_stats(item_ids: list[str]) -> tuple[dict, dict]:
        """per-(model,budget) delta and per-budget panel-mean delta, in pp."""
        per = {}
        panel = {}
        for b in BUDGETS:
            ds = []
            for m in MODELS:
                d = acc(b, "on", m, item_ids) - acc(b, "off", m, item_ids)
                per[(b, m)] = d
                ds.append(d)
            panel[b] = st.mean(ds)
        return per, panel

    per_point, panel_point = all_stats(ids)

    # ---------------- paired bootstrap (mechanism = aggregate_mw2_paired_bootstrap.py) ----------------
    rng = random.Random(SEED)
    per_bs: dict[tuple[int, str], list[float]] = {k: [] for k in per_point}
    panel_bs: dict[int, list[float]] = {b: [] for b in BUDGETS}
    abs_bs: dict[tuple[int, str, str], list[float]] = {k: [] for k in cells}

    for _ in range(B_BOOTSTRAP):
        resample = [ids[rng.randrange(n_items)] for _ in range(n_items)]
        p, pan = all_stats(resample)
        for k, v in p.items():
            per_bs[k].append(v)
        for k, v in pan.items():
            panel_bs[k].append(v)
        for (b, a, m) in cells:
            abs_bs[(b, a, m)].append(acc(b, a, m, resample))

    def ci(vals: list[float]) -> tuple[float, float, float]:
        s = sorted(vals)
        lo = s[int(0.025 * len(s))]
        hi = s[int(0.975 * len(s)) - 1]
        return lo, hi, s[len(s) // 2]

    # ---------------- injected-token curve ----------------
    base_ptok: dict[str, dict[str, int]] = {}
    for m in MODELS:
        axis = main_compact["datasets"]["powercodebench_2000"]["item_ids"]
        run = main_compact["runs"][f"comparison|{m}|A"]
        if len(axis) != len(run["prompt_tokens"]):
            raise ValueError(f"invalid condition-A token vector for {m}")
        base_ptok[m] = dict(zip(axis, run["prompt_tokens"]))

    token_rows = []
    for b in BUDGETS:
        for a in ARMS:
            for m in MODELS:
                v = cells[(b, a, m)]
                pt = [v["ptok"][i] for i in ids if v["ptok"][i] is not None]
                inj = None
                inj_max = None
                if m in base_ptok:
                    diffs = [v["ptok"][i] - base_ptok[m][i]
                             for i in ids
                             if v["ptok"][i] is not None and base_ptok[m].get(i) is not None]
                    if diffs:
                        inj = st.mean(diffs)
                        inj_max = max(diffs)
                token_rows.append({
                    "budget": b, "arm": a, "model": m, "model_short": SHORT[m],
                    "prompt_tokens_mean": st.mean(pt) if pt else None,
                    "prompt_tokens_min": min(pt) if pt else None,
                    "prompt_tokens_max": max(pt) if pt else None,
                    "injected_tokens_mean_vs_condA": inj,
                    "injected_tokens_max_vs_condA": inj_max,
                    "injected_exact_over_heuristic_budget": (inj / b) if inj else None,
                    "summary_prompt_token_stats": v["summary"].get("prompt_token_stats"),
                })

    # ---------------- manipulation check: did the two arms actually diverge? ----------------
    # prompt_tokens is the only prompt-side field persisted per item, so a token
    # difference is a LOWER BOUND on "gate-on and gate-off selected different docs".
    manip = []
    for b in BUDGETS:
        for m in MODELS:
            on = cells[(b, "on", m)]["ptok"]
            off = cells[(b, "off", m)]["ptok"]
            diff = [i for i in ids if on[i] != off[i]]
            manip.append({
                "budget": b, "model": m, "model_short": SHORT[m],
                "n_items_prompt_tokens_differ": len(diff),
                "frac_items_prompt_tokens_differ": len(diff) / n_items,
                "mean_abs_token_diff": st.mean([abs(on[i] - off[i]) for i in ids]),
            })
    # cross-model content identity probe: 14B and 480B share the Qwen tokenizer, so
    # identical per-item prompt_tokens across those two models implies identical
    # injected content (expected for gate-off = model-agnostic constant risk).
    qwen_pair = []
    for b in BUDGETS:
        for a in ARMS:
            p1 = cells[(b, a, "Qwen_Qwen2.5-Coder-14B-Instruct")]["ptok"]
            p2 = cells[(b, a, "Qwen_Qwen3-Coder-480B-A35B-Instruct")]["ptok"]
            same = sum(1 for i in ids if p1[i] == p2[i])
            qwen_pair.append({"budget": b, "arm": a,
                              "frac_items_14B_eq_480B_prompt_tokens": same / n_items})

    # ---------------- assemble ----------------
    per_rows = []
    for b in BUDGETS:
        for m in MODELS:
            lo, hi, med = ci(per_bs[(b, m)])
            a_on = acc(b, "on", m, ids)
            a_off = acc(b, "off", m, ids)
            lo_on, hi_on, _ = ci(abs_bs[(b, "on", m)])
            lo_off, hi_off, _ = ci(abs_bs[(b, "off", m)])
            per_rows.append({
                "budget": b, "model": m, "model_short": SHORT[m],
                "acc_gate_on_pct": a_on, "acc_gate_off_pct": a_off,
                "acc_gate_on_ci": [lo_on, hi_on], "acc_gate_off_ci": [lo_off, hi_off],
                "delta_pp": per_point[(b, m)],
                "boot_median_pp": med, "ci_low_pp": lo, "ci_high_pp": hi,
                "strict_positive": lo > 0, "strict_negative": hi < 0,
            })

    panel_rows = []
    for b in BUDGETS:
        lo, hi, med = ci(panel_bs[b])
        panel_rows.append({
            "budget": b,
            "panel_mean_acc_gate_on_pct": st.mean([acc(b, "on", m, ids) for m in MODELS]),
            "panel_mean_acc_gate_off_pct": st.mean([acc(b, "off", m, ids) for m in MODELS]),
            "panel_mean_delta_pp": panel_point[b],
            "boot_median_pp": med, "ci_low_pp": lo, "ci_high_pp": hi,
            "strict_positive": lo > 0, "strict_negative": hi < 0,
        })

    out = {
        "experiment": "E6.1 budget scan (S5 aggregation)",
        "protocol": "E6_risk_gate.md 2.1/3/8",
        "generated_by": "scripts/aggregate/aggregate_e6_budget_scan.py",
        "grid": {"budgets": BUDGETS, "arms": ["gate-on (C)", "gate-off (C, PROACTIVE_RISK_UNIFORM=1)"],
                 "models": MODELS, "n_cells": len(cells)},
        "pairing_verification": {
            "expected_sha16": EXPECTED_SHA,
            "all_cells_sha_match": sha_all_ok,
            "item_id_sets_identical_across_24_cells": ids_identical,
            "n_common_items": n_items,
            "recomputed_vs_reported_accuracy_mismatches": acc_mismatch,
            "per_cell": pairing,
        },
        "bootstrap": {"B": B_BOOTSTRAP, "seed": SEED, "unit": "item-id, with replacement, "
                      "paired across all 24 cells", "interval": "95% percentile",
                      "mechanism_source": "scripts/aggregate/aggregate_mw2_paired_bootstrap.py"},
        "per_model_budget": per_rows,
        "panel_mean_by_budget": panel_rows,
        "token_curve": token_rows,
        "token_curve_note": (
            "injected_tokens_mean_vs_condA = exact backend-tokenizer prompt_tokens of the "
            "cell minus condA (no-injection) prompt_tokens for the same item_id, from the "
            "frozen comparison/ tree. NOTE the injector's own budget accounting uses the "
            "chars//4 heuristic (knowledge_injection/library_spec.py:29), so exact injected "
            "tokens exceed the nominal budget B by ~1.2-1.5x in the binding regime."),
        "manipulation_check": manip,
        "qwen_pair_content_identity": qwen_pair,
    }

    save = args.output
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps(out, indent=2))

    # ---------------- markdown ----------------
    L = []
    L.append("| budget | 14B Δ [95% CI] | 70B Δ [95% CI] | 480B Δ [95% CI] | panel-mean Δ [95% CI] | acc gate-on (14B/70B/480B) | acc gate-off (14B/70B/480B) |")
    L.append("|---|---|---|---|---|---|---|")
    pr = {(r["budget"], r["model_short"]): r for r in per_rows}
    for b in BUDGETS:
        cellsmd = []
        for s in ("14B", "70B", "480B"):
            r = pr[(b, s)]
            cellsmd.append(f"{r['delta_pp']:+.2f} [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]")
        pm = next(r for r in panel_rows if r["budget"] == b)
        on = " / ".join(f"{pr[(b, s)]['acc_gate_on_pct']:.2f}" for s in ("14B", "70B", "480B"))
        off = " / ".join(f"{pr[(b, s)]['acc_gate_off_pct']:.2f}" for s in ("14B", "70B", "480B"))
        L.append(f"| **{b}** | {cellsmd[0]} | {cellsmd[1]} | {cellsmd[2]} | "
                 f"**{pm['panel_mean_delta_pp']:+.2f} [{pm['ci_low_pp']:+.2f}, {pm['ci_high_pp']:+.2f}]** | {on} | {off} |")
    L.append("")
    L.append("| budget | arm | 14B inj-tok mean | 70B inj-tok mean | 480B inj-tok mean | panel mean |")
    L.append("|---|---|---|---|---|---|")
    tr = {(r["budget"], r["arm"], r["model_short"]): r for r in token_rows}
    for b in BUDGETS:
        for a in ARMS:
            vals = [tr[(b, a, s)]["injected_tokens_mean_vs_condA"] for s in ("14B", "70B", "480B")]
            L.append(f"| {b} | gate-{a} | " + " | ".join(f"{v:.1f}" for v in vals) +
                     f" | {st.mean(vals):.1f} |")
    L.append("")
    L.append("| budget | 14B arms differ | 70B arms differ | 480B arms differ |")
    L.append("|---|---|---|---|")
    mp = {(r["budget"], r["model_short"]): r for r in manip}
    for b in BUDGETS:
        L.append(f"| {b} | " + " | ".join(
            f"{mp[(b, s)]['frac_items_prompt_tokens_differ'] * 100:.1f}%" for s in ("14B", "70B", "480B")) + " |")
    md = "\n".join(L)
    print(md)
    print()
    print(f"Saved: {save}")
    print(f"pairing: sha_all_ok={sha_all_ok} ids_identical={ids_identical} n_items={n_items} "
          f"acc_mismatches={len(acc_mismatch)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
