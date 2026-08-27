#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Repository copy of benchmark/e4_layer2b/summarize_e4_evals.py from the
# frozen experimental pipeline. Its inputs are raw per-item run trees too
# large to ship here, so it does not run from this checkout; see
# ARTIFACT_INDEX.md for the outputs it produced.
# --------------------------------------------------------------------------
"""Side-by-side E4 demand-recall table over the layer-1 / 2a / 2a-ext / 2b' sets.

Reads the frozen eval outputs under ``task_demand/results/`` and prints the
unadapted vs adapted recall@10 / hit@10 (+ per-role recall@10) for every E4 set
that has been unsealed.  Read-only: it never re-runs the ranker.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "task_demand" / "results"
SPLIT = "benchmark_reference_eval_only"

SETS = [
    ("layer 1  80-item holdout (PCB-scenario-reused)", "e4_layer1_holdout"),
    ("layer 2a n=84 naturally occurring (round 1)", "e4_layer2a_eval"),
    ("layer 2a n=10 platform-diversity (round 2)", "e4_layer2a_ext_eval"),
    ("layer 2b' n=20 engineer-written", "e4_layer2b_eval"),
]


def read(dirname: str, arm: str):
    path = RESULTS / dirname / arm / "metrics.json"
    if not path.exists():
        return None
    blob = json.load(open(path))
    m = blob["metrics"][SPLIT]
    qt = blob["config"].get("reweight_benchmark_path")
    return {
        "n": m["n_examples"],
        "recall@10": m["top_k"]["10"]["recall"],
        "hit@10": m["top_k"]["10"]["hit_rate"],
        "recall@5": m["top_k"]["5"]["recall"],
        "recall@20": m["top_k"]["20"]["recall"],
        "per_role": {k: v["10"] for k, v in (m.get("per_role_recall") or {}).items()},
        "q_t": qt,
        "alpha": blob.get("alpha_tuning", {}).get("selected_alpha"),
    }


def main():
    print("%-46s %5s | %-17s | %-17s | %s"
          % ("set", "n", "unadapted r@10/h@10", "adapted r@10/h@10", "lift r@10"))
    print("-" * 118)
    rows = []
    for label, d in SETS:
        u, a = read(d, "unadapted"), read(d, "adapted")
        if not u or not a:
            print("%-46s %5s | %-17s" % (label, "-", "NOT YET UNSEALED"))
            continue
        assert a["q_t"] and "benchmark/benchmark.json" in str(a["q_t"]), \
            "adapted arm must pin Q_T to the frozen PCB queries: %s" % a["q_t"]
        print("%-46s %5d | %6.4f / %6.4f  | %6.4f / %6.4f  | %+.4f"
              % (label, u["n"], u["recall@10"], u["hit@10"],
                 a["recall@10"], a["hit@10"], a["recall@10"] - u["recall@10"]))
        rows.append((label, u, a))
    print()
    for label, u, a in rows:
        roles = sorted(set(u["per_role"]) | set(a["per_role"]))
        print(label)
        for r in roles:
            print("    %-34s unadapted %.4f -> adapted %.4f"
                  % (r, u["per_role"].get(r, float("nan")),
                     a["per_role"].get(r, float("nan"))))
        print()


if __name__ == "__main__":
    main()
