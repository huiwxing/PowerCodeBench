#!/usr/bin/env python3
# --------------------------------------------------------------------------
# PowerCodeBench repository copy of the pipeline module task_demand/e4_conditional_eval.py, archived read-only to
# document how the frozen artifacts in this repository were generated.
# Its inputs are raw per-item run trees of the experimental pipeline that
# exceed this repository's size policy and are not archived here, so it
# is not runnable from this repository.  The frozen outputs it generated
# are archived here (see ARTIFACT_INDEX.md); headline-number verification
# is done by reproduce.py from the frozen artifacts alone.
# --------------------------------------------------------------------------
"""E4 conditional-adaptation evaluation (prereg 2026-08-02, selector S1 frozen).

Per-query intent selector (construction vs analysis) routes each query to the
unadapted or adapted demand profile; metrics reuse the frozen
``benchmark_reference_eval_only`` pathway (``load_benchmark_reference_examples``
+ ``evaluate_ranker``) unchanged. Both arms are reproduced bit-for-bit from
their frozen run configs (seed 22, hybrid_tfidf, auto-hybrid-alpha; adapted arm
pins role reweighting to the frozen PCB ``benchmark/benchmark.json``,
max_role_weight 3.0). Single evaluation; the selector rule is frozen in
``revision_aei_r1/E4_external_queries.md`` §8 and must not be edited here.
"""

import argparse
import gc
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

TASK_DEMAND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DEMAND_DIR.parent
sys.path.insert(0, str(TASK_DEMAND_DIR))

import task_demand_model as tdm  # noqa: E402

EVAL_SETS = {
    "layer1_holdout_n80": "benchmark/naturalistic_holdout/naturalistic_holdout_seed22_n80.json",
    "layer2a_n84": "benchmark/e4_layer2a/e4_layer2a_set.json",
    "layer2a_ext_n10": "benchmark/e4_layer2a/e4_layer2a_ext.json",
    "layer2b_n20": "benchmark/e4_layer2b/e4_layer2b_set.json",
}

# Documented frozen-arm aggregates (recall@10) used only as a reproduction check.
DOCUMENTED_R10 = {
    "layer1_holdout_n80": {"unadapted": 0.5515, "adapted": 0.6967},
    "layer2a_n84": {"unadapted": 0.5506, "adapted": 0.5300},
    "layer2a_ext_n10": {"unadapted": 0.6472, "adapted": 0.4900},
    "layer2b_n20": {"unadapted": 0.4869, "adapted": 0.6452},
}

# ── Selector S1 (frozen 2026-08-02; see E4 doc §8 — do not edit) ─────────────
NAMED_NET_PATTERNS = [
    r"\b(?:ieee|cigre|pegase|rte|wscc|pjm|kerber|baran|oberrhein|simbench|polish|illinois|gb|rts)\b",
    r"\bnew england\b",
    r"\bcase[_ ]?\d+",
    r"\b\d+[\s-]*bus\s+(?:\w+\s+){0,2}(?:case|system|grid|feeder)\b",
    r"\bthe \d+[\s-]*bus\b",
    r"\bexample net(?:work)?\b|\bopen ring\b|\b4[\s-]*loads\b|\bfour loads\b|loads-with-branches|voltage[\s-]*control lv|european (?:low-voltage|lv) test feeder",
]
NAMED_NET_RE = [re.compile(p) for p in NAMED_NET_PATTERNS]


def selector_intent(query: str) -> str:
    """Return 'analysis' (adapted arm) or 'construction' (unadapted arm)."""
    text = query.lower()
    text = text.replace("–", "-").replace("—", "-")
    for pat in NAMED_NET_RE:
        if pat.search(text):
            return "analysis"
    return "construction"


# ── Frozen arm reproduction ──────────────────────────────────────────────────

def make_args(role_reweight: bool, seed: int = 22) -> argparse.Namespace:
    """Replicate the frozen run_config fields consumed by the training path."""
    return argparse.Namespace(
        augmented_path=tdm.DEFAULT_AUGMENTED_PATH,
        docs_path=tdm.DEFAULT_DOCS_PATH,
        model="hybrid_tfidf",
        include_variants=True,
        seed=seed,
        negative_per_positive=8,
        max_train_pairs=None,
        max_positive_per_function=None,
        hybrid_alpha=0.5,
        auto_hybrid_alpha=True,
        hybrid_alpha_target_k=10,
        sbert_model="sentence-transformers/all-MiniLM-L6-v2",
        sbert_batch_size=64,
        sentence_transformers_device=None,
        role_reweight=role_reweight,
        max_role_weight=3.0,
    )


def train_arm_and_score(arm: str, questions, seed: int = 22):
    """Train one frozen arm exactly as its CLI run did, score all questions.

    Returns (alpha, role_weights, {question_text: score_dict}).
    """
    args = make_args(role_reweight=(arm == "adapted"), seed=seed)
    # Each frozen arm ran as a fresh process: replicate main()'s seeding.
    random.seed(args.seed)
    np.random.seed(args.seed)

    cards = tdm.load_function_cards(args.docs_path)
    function_names = set(cards)
    examples = tdm.load_augmented_examples(
        args.augmented_path, function_names, include_variants=args.include_variants)
    train, dev, _test = tdm.split_by_group(examples, seed=args.seed)

    ranker = tdm.build_ranker(args)

    role_weights = None
    if args.role_reweight:
        bench_for_weights = tdm.load_benchmark_reference_examples(
            PROJECT_ROOT / "benchmark" / "benchmark.json", function_names, max_items=None)
        role_weights = tdm.compute_role_weights(
            train, bench_for_weights, cards, max_weight=args.max_role_weight)

    tdm.train_or_fit_ranker(ranker, args, train, cards, role_weights=role_weights)
    tdm.tune_hybrid_alpha(
        ranker, dev, cards,
        candidate_alphas=[round(x * 0.1, 1) for x in range(0, 11)],
        target_k=args.hybrid_alpha_target_k)

    score_map = {q: ranker.score(q) for q in questions}
    alpha = ranker.alpha
    del ranker
    gc.collect()
    return alpha, role_weights, score_map


class PrecomputedRanker:
    """Serves stored per-question score dicts through the ranker interface."""

    def __init__(self, score_map):
        self.score_map = score_map

    def score(self, question: str):
        return self.score_map[question]


class ConditionalRanker:
    """Frozen selector S1 routes each query to one arm's stored scores."""

    def __init__(self, unadapted_map, adapted_map):
        self.maps = {"construction": unadapted_map, "analysis": adapted_map}

    def score(self, question: str):
        return self.maps[selector_intent(question)][question]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="E4 conditional-adaptation evaluation (selector S1 frozen).")
    parser.add_argument("--train-seed", type=int, default=22,
                        help="Pipeline seed for both arms (split + negative "
                             "sampling + LogReg). Frozen runs use 22; other "
                             "values are for seed-sensitivity analysis only.")
    parser.add_argument("--output-name", default="metrics_by_set.json",
                        help="Output filename (relative to results/e4_conditional/). "
                             "Use a distinct name to avoid overwriting archived runs.")
    cli = parser.parse_args()

    cards = tdm.load_function_cards(tdm.DEFAULT_DOCS_PATH)
    function_names = set(cards)

    set_examples = {}
    all_questions = []
    for name, rel in EVAL_SETS.items():
        exs = tdm.load_benchmark_reference_examples(PROJECT_ROOT / rel, function_names)
        set_examples[name] = exs
        all_questions.extend(ex.question for ex in exs)

    ua_alpha, _, ua_scores = train_arm_and_score("unadapted", all_questions, seed=cli.train_seed)
    ad_alpha, ad_role_weights, ad_scores = train_arm_and_score("adapted", all_questions, seed=cli.train_seed)
    print(f"Reproduced arms: unadapted alpha={ua_alpha}, adapted alpha={ad_alpha}")

    rankers = {
        "unadapted": PrecomputedRanker(ua_scores),
        "adapted": PrecomputedRanker(ad_scores),
        "conditional": ConditionalRanker(ua_scores, ad_scores),
    }

    top_ks = [1, 3, 5, 10, 20]
    out = {
        "prereg": "E4_external_queries.md §8 conditional adaptation prereg + selector S1 freeze (2026-08-02)",
        "train_seed": cli.train_seed,
        "arms": {
            "unadapted": {"alpha": ua_alpha, "role_reweight": False},
            "adapted": {"alpha": ad_alpha, "role_reweight": True,
                        "reweight_source": "benchmark/benchmark.json (frozen PCB)",
                        "role_weights": ad_role_weights},
        },
        "selector": {"rule": "S1 (frozen)", "patterns": NAMED_NET_PATTERNS},
        "sets": {},
        "comparison_recall_at_10": {},
        "exit_reading": {},
    }

    for name, exs in set_examples.items():
        decisions = {ex.sample_id: selector_intent(ex.question) for ex in exs}
        dist = {"construction": sum(1 for v in decisions.values() if v == "construction"),
                "analysis": sum(1 for v in decisions.values() if v == "analysis")}
        metrics = {
            arm: tdm.evaluate_ranker(r, exs, cards=cards, top_ks=top_ks)
            for arm, r in rankers.items()
        }
        out["sets"][name] = {
            "n": len(exs),
            "selector_distribution": dist,
            "selector_decisions": decisions,
            "metrics": {arm: m["top_k"] for arm, m in metrics.items()},
        }
        row = {arm: metrics[arm]["top_k"]["10"]["recall"] for arm in rankers}
        row["hit@10_conditional"] = metrics["conditional"]["top_k"]["10"]["hit_rate"]
        row["hit@10_unadapted"] = metrics["unadapted"]["top_k"]["10"]["hit_rate"]
        row["hit@10_adapted"] = metrics["adapted"]["top_k"]["10"]["hit_rate"]
        row["max_arm"] = max(row["unadapted"], row["adapted"])
        row["conditional_minus_max_pp"] = round((row["conditional"] - row["max_arm"]) * 100, 2)
        row["meets_exit_a_margin"] = row["conditional"] >= row["max_arm"] - 0.01
        if cli.train_seed == 22:
            # Only meaningful against the documented frozen-arm numbers (seed 22).
            docd = DOCUMENTED_R10[name]
            row["reproduction_check"] = {
                "documented": docd,
                "recomputed": {"unadapted": row["unadapted"], "adapted": row["adapted"]},
                "match": (abs(row["unadapted"] - docd["unadapted"]) < 5e-5
                          and abs(row["adapted"] - docd["adapted"]) < 5e-5),
            }
        out["comparison_recall_at_10"][name] = row

    n_pass = sum(1 for r in out["comparison_recall_at_10"].values() if r["meets_exit_a_margin"])
    exit_code = "a" if n_pass == 4 else ("b" if n_pass > 0 else "c")
    out["exit_reading"] = {
        "sets_meeting_max_minus_1pp": n_pass,
        "exit": exit_code,
        "criterion": "exit a: all four sets conditional recall@10 >= max(adapted, unadapted) - 1pp",
    }

    out_dir = TASK_DEMAND_DIR / "results" / "e4_conditional"
    out_path = out_dir / cli.output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")

    print("\nrecall@10 comparison (conditional / adapted / unadapted / max):")
    for name, row in out["comparison_recall_at_10"].items():
        repro = row.get("reproduction_check", {}).get("match", "n/a")
        print(f"  {name}: cond={row['conditional']:.4f} adapt={row['adapted']:.4f} "
              f"unadapt={row['unadapted']:.4f} max={row['max_arm']:.4f} "
              f"delta={row['conditional_minus_max_pp']:+.2f}pp "
              f"repro_ok={repro}")
    print(f"\nExit reading: {exit_code} ({n_pass}/4 sets within max-1pp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
