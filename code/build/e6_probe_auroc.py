#!/usr/bin/env python3
"""E6.2: probe-derived risk as a function-level failure predictor (AUROC / calibration).

Rebuilds Exp-D's item-level probe->failure correlation at the (model, function)
level and scores whether rho_M(f) predicts function-level failure:

  * analysis unit  = (model, function), function in used_functions of >=1 item
  * label (frozen, E6 doc S2.2): among condition-A items where f in used_functions,
      attributed failure(M, f, item) <=> item match == false AND either
        (i)  the error_msg offending trace names f (identifier match), or
        (ii) f is the item's task analysis function and correct_task_fn == false.
      Failures whose trace names some corpus function are attributed ONLY to the
      named function (no spreading across co-occurring used_functions); failures
      naming no corpus function fall in the unattributable bucket (reported, not
      labelled).  binary label = 1 iff >=1 attributed failure, given min-support
      K items using f (K = 3).
  * features = per-layer risk s^(l) (weight-independent, primary) and the
      aggregate rho_M(f) under two weight sets: uniform 0.25x4 (headline, paper
      Eq.1 canonical) and deployed mild-L3 (0.20,0.25,0.20,0.35) (robustness).

The aggregate rho implementation is self-contained here and mirrors the deployed
injector (including its L3 diagnostic special case).  Item outcomes, diagnostic
text, and function-use metadata come from the reusable main-panel compact tree;
complete L0--L3 profiles come from the supplementary profile evidence file.

Output: results/aggregates/e6_probe_auroc.json  (+ stdout summary table)
"""
from __future__ import annotations

import json
import re
import statistics as st
import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAIN_COMPACT = ROOT / "results/raw/main_experiment/primary_outcomes_compact.json"
DEFAULT_PROFILES = ROOT / "results/supplementary_evidence/probe_profiles.json"
DEFAULT_OUTPUT = ROOT / "results/aggregates/e6_probe_auroc.json"
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

# rho weight variants (E6 doc S2.2, user ruling 2026-07-23: uniform = headline).
WEIGHTS_UNIFORM = {"L0": 0.25, "L1": 0.25, "L2": 0.25, "L3": 0.25}
WEIGHTS_MILD_L3 = {"L0": 0.20, "L1": 0.25, "L2": 0.20, "L3": 0.35}
LAYERS = ("L0", "L1", "L2", "L3")
MIN_SUPPORT = 3

# task -> canonical analysis function (bare name), mirrors
# benchmark/benchmark_config.py TASK_FUNCTION_PATTERNS (label condition ii).
TASK_FN = {
    "power_flow": "runpp",
    "dc_power_flow": "rundcpp",
    "opf": "runopp",
    "short_circuit_3ph": "calc_sc",
    "short_circuit_2ph": "calc_sc",
    "contingency": "run_contingency",
    "time_series": "run_timeseries",
    "state_estimation": "estimate",
}


def rho_per_function(profile: dict, weights: dict) -> dict:
    """Replicate the deployed injector's aggregate function-risk calculation."""
    out = {}
    for fn, layers in profile.items():
        if not isinstance(layers, dict):
            continue
        total_w = 0.0
        mastery = 0.0
        for layer, weight in weights.items():
            data = layers.get(layer)
            if data is None:
                continue
            if isinstance(data, dict):
                if layer == "L3" and "diagnostics" in data:
                    diag = data["diagnostics"] or {}
                    es = diag.get("execution_success")
                    tu = diag.get("target_function_used")
                    if isinstance(es, (int, float)) and isinstance(tu, (int, float)):
                        score = float(es) * float(tu)
                    else:
                        score = data.get("score")
                else:
                    score = data.get("score")
            else:
                score = data
            if not isinstance(score, (int, float)):
                continue
            mastery += weight * max(0.0, min(1.0, float(score)))
            total_w += weight
        if total_w > 0:
            out[fn] = 1.0 - mastery / total_w
    return out


def layer_risk_per_function(profile: dict) -> dict:
    """Per-layer risk 1 - s^(l) for each function (weight-independent feature).

    Mirrors exp_d's L3 special case: score = execution_success * target_function_used
    when L3 diagnostics are present.
    """
    out: dict = {}
    for fn, layers in profile.items():
        if not isinstance(layers, dict):
            continue
        per = {}
        for layer in LAYERS:
            data = layers.get(layer)
            if data is None:
                continue
            if isinstance(data, dict):
                if layer == "L3" and "diagnostics" in data:
                    diag = data["diagnostics"] or {}
                    es = diag.get("execution_success")
                    tu = diag.get("target_function_used")
                    if isinstance(es, (int, float)) and isinstance(tu, (int, float)):
                        score = float(es) * float(tu)
                    else:
                        score = data.get("score")
                else:
                    score = data.get("score")
            else:
                score = data
            if isinstance(score, (int, float)):
                per[layer] = 1.0 - max(0.0, min(1.0, float(score)))
        out[fn] = per
    return out


def named_corpus_functions(msg: str, corpus: set) -> set:
    """Corpus functions named as identifiers in an error_msg trace."""
    if not msg:
        return set()
    return {f for f in corpus
            if re.search(r"(?<![A-Za-z0-9_])" + re.escape(f) + r"(?![A-Za-z0-9_])", msg)}


def auroc(scores: list, labels: list) -> float | None:
    """AUROC via the rank-sum (Mann-Whitney U) statistic with tie averaging."""
    n = len(scores)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum_pos = sum(ranks[i] for i in range(n) if labels[i] == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def calibration(preds: list, labels: list, n_bins: int = 10) -> tuple[list, float]:
    """Equal-width reliability bins over [0,1] and expected calibration error."""
    bins = []
    ece = 0.0
    n = len(labels)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, p in enumerate(preds)
               if (p >= lo and p < hi) or (b == n_bins - 1 and p == hi)]
        if not idx:
            continue
        mean_pred = sum(preds[i] for i in idx) / len(idx)
        emp = sum(labels[i] for i in idx) / len(idx)
        bins.append({
            "bin": [round(lo, 2), round(hi, 2)],
            "n": len(idx),
            "mean_pred_rho": round(mean_pred, 4),
            "empirical_fail_rate": round(emp, 4),
        })
        ece += (len(idx) / n) * abs(mean_pred - emp)
    return bins, ece


def build_labels(items: dict, audit: dict, corpus: set):
    """Return (support, positives, n_failures, trace_unattr, fully_unattr).

    support[f]     = # condA items using f
    positives[f]   = # items with an attributed failure to f
    trace_unattr   = # failures whose trace names no corpus function
    fully_unattr   = # failures attributed by neither (i) nor (ii)
    """
    support: dict = defaultdict(int)
    positives: dict = defaultdict(int)
    n_failures = trace_unattr = fully_unattr = 0
    for iid, it in items.items():
        aud = audit.get(iid, {})
        used = set(aud.get("used_functions") or [])
        task_fn = TASK_FN.get(aud.get("task"))
        fail = not it.get("match")
        named = set()
        cond_ii = False
        if fail:
            n_failures += 1
            named = named_corpus_functions(it.get("error_msg") or "", corpus)
            if not named:
                trace_unattr += 1
            ctf = (it.get("diagnostics") or {}).get("correct_task_fn", True)
            cond_ii = (task_fn is not None and task_fn in used and not ctf)
            if not named and not cond_ii:
                fully_unattr += 1
        for f in used:
            support[f] += 1
            if fail and (f in named or (f == task_fn and cond_ii)):
                positives[f] += 1
    return support, positives, n_failures, trace_unattr, fully_unattr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-compact", type=Path, default=DEFAULT_MAIN_COMPACT,
        help="Path to primary_outcomes_compact.json",
    )
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES,
                        help="Path to the complete model probe-profile evidence")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main_compact = json.loads(args.main_compact.read_text())
    profile_evidence = json.loads(args.profiles.read_text())
    profiles = {
        row["source_model"]: row["profile"]
        for row in profile_evidence["models"].values()
    }
    axis = main_compact["datasets"]["powercodebench_2000"]
    audit_view = main_compact["bench_function_audit_compact"]
    audit = {
        item_id: {"used_functions": functions, "task": task}
        for item_id, functions, task in zip(
            axis["item_ids"], audit_view["used_functions"], axis["task"]
        )
    }

    corpus: set = set()
    per_model: dict = {}
    # pooled feature/label vectors
    pool_rho = {"uniform": [], "mild_l3": []}
    pool_layer = {l: {"scores": [], "labels": []} for l in LAYERS}
    pool_labels: list = []

    print("=" * 96)
    print("E6.2  (model, function) probe-risk -> failure  (headline weights = uniform 0.25x4)")
    print("=" * 96)
    print(f"{'Model':<40}{'n(M,f)':>7}{'pos':>5}{'AUROC_u':>9}{'AUROC_L3':>10}"
          f"{'trace_un%':>11}{'full_un%':>10}")
    print("-" * 96)

    for model in PANEL:
        profile = profiles.get(model)
        run = main_compact["runs"].get(f"comparison|{model}|A")
        if profile is None or run is None:
            continue
        if not corpus:
            corpus = set(profile.keys())
        rho_u = rho_per_function(profile, WEIGHTS_UNIFORM)
        rho_l = rho_per_function(profile, WEIGHTS_MILD_L3)
        lrisk = layer_risk_per_function(profile)

        diagnostic = run["condition_a_diagnostics"]
        dictionary = diagnostic["error_msg_dictionary"]
        items = {
            item_id: {
                "match": match == "1",
                "error_msg": dictionary[message_index],
                "diagnostics": ({} if correct_task_fn is None else
                                {"correct_task_fn": correct_task_fn}),
            }
            for item_id, match, message_index, correct_task_fn in zip(
                axis["item_ids"], run["match_bits"],
                diagnostic["error_msg_index"], diagnostic["correct_task_fn"]
            )
        }
        support, positives, n_fail, trace_un, full_un = build_labels(items, audit, corpus)
        del items

        funcs = [f for f in support if support[f] >= MIN_SUPPORT and f in rho_u]
        labels = [1 if positives[f] > 0 else 0 for f in funcs]
        s_u = [rho_u[f] for f in funcs]
        s_l = [rho_l[f] for f in funcs]
        au_u = auroc(s_u, labels)
        au_l = auroc(s_l, labels)

        layer_au = {}
        for layer in LAYERS:
            lf = [f for f in funcs if layer in lrisk.get(f, {})]
            if lf:
                ll = [1 if positives[f] > 0 else 0 for f in lf]
                layer_au[layer] = auroc([lrisk[f][layer] for f in lf], ll)
                for f in lf:
                    pool_layer[layer]["scores"].append(lrisk[f][layer])
                    pool_layer[layer]["labels"].append(1 if positives[f] > 0 else 0)

        pool_rho["uniform"].extend(s_u)
        pool_rho["mild_l3"].extend(s_l)
        pool_labels.extend(labels)

        n_pos = sum(labels)
        per_model[model] = {
            "n_Mf": len(funcs),
            "n_pos": n_pos,
            "n_neg": len(funcs) - n_pos,
            "within_auroc_uniform": None if au_u is None else round(au_u, 4),
            "within_auroc_mild_l3": None if au_l is None else round(au_l, 4),
            "per_layer_auroc": {l: (None if layer_au.get(l) is None else round(layer_au[l], 4))
                                for l in LAYERS},
            "n_failures": n_fail,
            "trace_unattributable_pct": round(100.0 * trace_un / n_fail, 2) if n_fail else None,
            "fully_unattributable_pct": round(100.0 * full_un / n_fail, 2) if n_fail else None,
        }
        print(f"{model:<40}{len(funcs):>7}{n_pos:>5}"
              f"{(au_u if au_u is not None else float('nan')):>9.3f}"
              f"{(au_l if au_l is not None else float('nan')):>10.3f}"
              f"{per_model[model]['trace_unattributable_pct']:>11.1f}"
              f"{per_model[model]['fully_unattributable_pct']:>10.1f}")

    # ---- pooled + calibration ----
    pooled_au_u = auroc(pool_rho["uniform"], pool_labels)
    pooled_au_l = auroc(pool_rho["mild_l3"], pool_labels)
    cal_bins, ece = calibration(pool_rho["uniform"], pool_labels)
    pooled_layer_au = {l: auroc(pool_layer[l]["scores"], pool_layer[l]["labels"]) for l in LAYERS}

    within_u = [v["within_auroc_uniform"] for v in per_model.values()
                if v["within_auroc_uniform"] is not None]
    within_l = [v["within_auroc_mild_l3"] for v in per_model.values()
                if v["within_auroc_mild_l3"] is not None]

    summary = {
        "within_model_auroc_uniform": {
            "mean": round(st.mean(within_u), 4), "min": round(min(within_u), 4),
            "max": round(max(within_u), 4), "n_models": len(within_u),
        },
        "within_model_auroc_mild_l3": {
            "mean": round(st.mean(within_l), 4), "min": round(min(within_l), 4),
            "max": round(max(within_l), 4), "n_models": len(within_l),
        },
        "pooled_auroc_uniform": round(pooled_au_u, 4),
        "pooled_auroc_mild_l3": round(pooled_au_l, 4),
        "pooled_n": len(pool_labels),
        "pooled_n_pos": sum(pool_labels),
        "ece_uniform": round(ece, 4),
        "pooled_per_layer_auroc": {l: (None if pooled_layer_au[l] is None else round(pooled_layer_au[l], 4))
                                   for l in LAYERS},
    }

    out = {
        "meta": {
            "unit": "(model, function)",
            "headline_weights": "uniform_0.25x4",
            "robustness_weights": "mild_l3_0.20_0.25_0.20_0.35",
            "min_support_K": MIN_SUPPORT,
            "label": "condA function-level attributed failure (E6 doc S2.2)",
            "feature_note": "rho = 1 - weighted mastery; per-layer AUROC uses per-layer risk 1 - s^(l)",
            "unattributable_note": ("trace_unattributable = failure names no corpus function "
                                    "(S2.2 bucket); fully_unattributable = neither cond (i) nor (ii)"),
        },
        "summary": summary,
        "per_model": per_model,
        "pooled": {
            "auroc_uniform": round(pooled_au_u, 4),
            "auroc_mild_l3": round(pooled_au_l, 4),
            "n": len(pool_labels),
            "n_pos": sum(pool_labels),
            "ece_uniform": round(ece, 4),
            "calibration_bins_uniform": cal_bins,
            "per_layer_auroc": summary["pooled_per_layer_auroc"],
        },
    }
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("-" * 96)
    print(f"within-model AUROC (uniform): mean={summary['within_model_auroc_uniform']['mean']:.3f} "
          f"[{summary['within_model_auroc_uniform']['min']:.3f}, "
          f"{summary['within_model_auroc_uniform']['max']:.3f}]  (n={len(within_u)})")
    print(f"pooled AUROC (uniform)={pooled_au_u:.3f}  (N={len(pool_labels)}, pos={sum(pool_labels)})  "
          f"ECE={ece:.3f}")
    print(f"pooled per-layer AUROC: "
          + "  ".join(f"{l}={pooled_layer_au[l]:.3f}" for l in LAYERS if pooled_layer_au[l] is not None))
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
