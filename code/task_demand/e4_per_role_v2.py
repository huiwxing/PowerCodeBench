#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Repository copy of task_demand/e4_per_role_v2.py from the frozen
# experimental pipeline. Its inputs are raw per-item run trees too large to
# ship here, so it does not run from this checkout; see ARTIFACT_INDEX.md for
# the outputs it produced.
# --------------------------------------------------------------------------
"""E4 per-role recall@10 breakdown, v2 deterministic pipeline (one-shot analysis).

Regenerates the numbers behind SM Table `tab:sm_external_per_role` (per-role
recall@10, unadapted -> adapted, four evaluation sets) under the v2
deterministic pipeline: ``task_demand_model.py`` with the sorted-iteration
determinism fix (regression-tested PYTHONHASHSEED-independent in
``tests/test_demand_determinism.py``), seed 22, same configuration as the
frozen ``*_v2`` artifacts.  Both arms are reproduced through the exact code
path of ``e4_conditional_eval.py`` (``make_args``/``train_arm_and_score``);
per-role hits/totals at k=10 are recomputed at full precision and
cross-checked bit-for-bit (at the archived 4-dp rounding) against the frozen
v2 artifacts ``results/<set>/{unadapted,adapted}_v2/metrics.json`` that back
the SM aggregate table.

Output: ``results/e4_conditional/per_role_v2.json``.
No parameter is tuned or changed anywhere in this script.
"""

import json
import sys
from fractions import Fraction
from pathlib import Path

TASK_DEMAND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DEMAND_DIR))

import e4_conditional_eval as ce  # noqa: E402  (frozen-arm reproduction path)
import task_demand_model as tdm  # noqa: E402

# Eval-set key -> (frozen v2 results dir, SM table row label)
SET_INFO = {
    "layer1_holdout_n80": ("e4_layer1_holdout", "Internal holdout (n=80)"),
    "layer2a_n84": ("e4_layer2a_eval", "Naturally occurring (n=84)"),
    "layer2a_ext_n10": ("e4_layer2a_ext_eval", "Cross-platform (n=10)"),
    "layer2b_n20": ("e4_layer2b_eval", "Engineer-written (n=20)"),
}

# Internal role key -> SM table column (plotting is excluded from the table;
# the caption reports it separately for the n=84 set).
ROLE_COLUMNS = {
    "analysis_executor": "Analysis executor",
    "network_loader": "Network loader",
    "network_or_element_construction": "Construction",
    "controller_or_timeseries": "Controller/time-series",
    "other": "Other",
}

K = 10


def per_role_at_k(ranker, examples, cards):
    """Full-precision per-role recall@K plus aggregate recall/hit@K.

    Identical definition to ``tdm.evaluate_ranker``: for every (query, gold
    function) occurrence, the role is ``infer_function_role``; role recall is
    pooled hits/total over occurrences.  Kept exact (Fractions) so 1-dp table
    percentages are not distorted by the archived 4-dp rounding.
    """
    role_counts = {}
    agg_recall = Fraction(0)
    agg_hit = 0
    n = 0
    for ex in examples:
        truth = ex.functions & set(cards)
        if not truth:
            continue
        n += 1
        pred = set(tdm.rank_names(ranker.score(ex.question), K))
        overlap = pred & truth
        agg_recall += Fraction(len(overlap), len(truth))
        agg_hit += 1 if overlap else 0
        for fn in truth:
            role = tdm.infer_function_role(fn, cards.get(fn))
            hits, total = role_counts.get(role, (0, 0))
            role_counts[role] = (hits + (1 if fn in pred else 0), total + 1)
    return {
        "n_eval": n,
        "aggregate": {"recall": agg_recall / n, "hit_rate": Fraction(agg_hit, n)},
        "roles": {r: Fraction(h, t) for r, (h, t) in role_counts.items()},
        "role_counts": {r: {"hits": h, "total": t} for r, (h, t) in role_counts.items()},
    }


def pct(x):
    """Exact fraction -> table percentage string, 1 decimal, half-up.

    Exact half-up on the Fraction (e.g. 5/16 -> 31.3, 11/16 -> 68.8) matches
    the SM table's rounding convention; float %.1f would give half-even 31.2.
    """
    tenths = (Fraction(x) * 1000 + Fraction(1, 2)).__floor__()
    return f"{tenths / 10:.1f}"


def main() -> int:
    cards = tdm.load_function_cards(tdm.DEFAULT_DOCS_PATH)
    function_names = set(cards)

    set_examples = {}
    all_questions = []
    for name, rel in ce.EVAL_SETS.items():
        exs = tdm.load_benchmark_reference_examples(ce.PROJECT_ROOT / rel, function_names)
        set_examples[name] = exs
        all_questions.extend(ex.question for ex in exs)

    ua_alpha, _, ua_scores = ce.train_arm_and_score("unadapted", all_questions, seed=22)
    ad_alpha, ad_role_weights, ad_scores = ce.train_arm_and_score("adapted", all_questions, seed=22)
    print(f"Reproduced arms: unadapted alpha={ua_alpha}, adapted alpha={ad_alpha}")

    arms = {
        "unadapted": ce.PrecomputedRanker(ua_scores),
        "adapted": ce.PrecomputedRanker(ad_scores),
    }

    out = {
        "provenance": {
            "pipeline": "v2 deterministic (task_demand_model.py sorted-iteration fix; "
                        "PYTHONHASHSEED-independent per tests/test_demand_determinism.py)",
            "seed": 22,
            "arm_reproduction": "e4_conditional_eval.make_args/train_arm_and_score "
                                "(frozen run_config replicas; adapted arm pins "
                                "reweight to frozen PCB benchmark/benchmark.json)",
            "cross_check": "results/<set>/{unadapted,adapted}_v2/metrics.json "
                           "benchmark_reference_eval_only (per-role @10 and aggregates, 4-dp)",
            "alphas": {"unadapted": ua_alpha, "adapted": ad_alpha},
            "role_weights_adapted": ad_role_weights,
            "role_column_map": ROLE_COLUMNS,
            "definition": "per-role recall@10 = pooled hits/total over (query, gold "
                          "function) occurrences of that role (evaluate_ranker semantics)",
        },
        "sets": {},
        "prose_ranges_pp": {},
        "all_checks_pass": True,
    }

    deltas = {r: {} for r in ROLE_COLUMNS}
    for name, (v2_dir, label) in SET_INFO.items():
        exs = set_examples[name]
        recomputed = {arm: per_role_at_k(r, exs, cards) for arm, r in arms.items()}

        set_out = {"label": label, "n_eval": recomputed["unadapted"]["n_eval"],
                   "arms": {}, "table_row": {}, "checks": {}}
        for arm in ("unadapted", "adapted"):
            rec = recomputed[arm]
            frozen = json.load(open(
                TASK_DEMAND_DIR / "results" / v2_dir / f"{arm}_v2" / "metrics.json"
            ))["metrics"]["benchmark_reference_eval_only"]

            ok = (round(float(rec["aggregate"]["recall"]), 4) == frozen["top_k"]["10"]["recall"]
                  and round(float(rec["aggregate"]["hit_rate"]), 4) == frozen["top_k"]["10"]["hit_rate"])
            frozen_roles = {r: v["10"] for r, v in frozen["per_role_recall"].items()}
            ok = ok and set(frozen_roles) == set(rec["roles"]) and all(
                round(float(rec["roles"][r]), 4) == frozen_roles[r] for r in frozen_roles)
            set_out["checks"][arm] = {
                "matches_frozen_v2": ok,
                "frozen_aggregate_recall@10": frozen["top_k"]["10"]["recall"],
                "frozen_per_role@10": frozen_roles,
            }
            out["all_checks_pass"] = out["all_checks_pass"] and ok

            set_out["arms"][arm] = {
                "aggregate_recall@10": round(float(rec["aggregate"]["recall"]), 4),
                "aggregate_hit@10": round(float(rec["aggregate"]["hit_rate"]), 4),
                "per_role@10": {r: round(float(v), 4) for r, v in rec["roles"].items()},
                "role_counts": rec["role_counts"],
            }

        for role, col in ROLE_COLUMNS.items():
            ua = recomputed["unadapted"]["roles"].get(role)
            ad = recomputed["adapted"]["roles"].get(role)
            if ua is None and ad is None:
                set_out["table_row"][col] = "---"
                continue
            delta = float(ad - ua) * 100
            set_out["table_row"][col] = f"{pct(ua)} -> {pct(ad)}"
            set_out["table_row"][col + " (delta_pp)"] = round(delta, 1)
            deltas[role][name] = delta
        plotting = {arm: recomputed[arm]["roles"].get("plotting")
                    for arm in ("unadapted", "adapted")}
        if any(v is not None for v in plotting.values()):
            set_out["plotting_excluded_from_table"] = {
                arm: (None if v is None else round(float(v), 4))
                for arm, v in plotting.items()}
        out["sets"][name] = set_out

    def rng(role):
        vals = deltas[role]
        return {"min": round(min(vals.values()), 1), "max": round(max(vals.values()), 1),
                "per_set": {k: round(v, 1) for k, v in vals.items()}}

    out["prose_ranges_pp"] = {
        "analysis_executor_lift": rng("analysis_executor"),
        "network_loader_lift": rng("network_loader"),
        "construction_penalty": rng("network_or_element_construction"),
    }

    out_path = TASK_DEMAND_DIR / "results" / "e4_conditional" / "per_role_v2.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")

    print(f"\nall_checks_pass = {out['all_checks_pass']}")
    for name, s in out["sets"].items():
        print(f"\n{s['label']}:")
        for col, cell in s["table_row"].items():
            print(f"  {col}: {cell}")
    print("\nprose ranges (pp):", json.dumps(out["prose_ranges_pp"], indent=1))
    return 0 if out["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
