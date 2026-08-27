"""A5-3 risk-weight sensitivity grid: 4 weightings x 5 models @ n=2000, item-paired.

Cells: deployed = frozen comparison/ condC; uniform / l3heavy (C4 + fills);
l0heavy (new inverted direction).  Pre-registered reading (plan.md s9 A5-3,
frozen 2026-08-01 before ignition): per model, max pairwise |delta acc| across
the four weightings with item-paired bootstrap CIs (10k, seed 20260801) and
exact McNemar per pair; insensitivity margin = point estimate <= 2pp.

Reads the deployed condition-C arm from the reusable main-panel compact tree and
the three alternate arms from ``results/raw/robustness_compact``.  Emits the
frozen aggregate JSON plus a stdout table.
"""

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAIN_COMPACT = ROOT / "results/raw/main_experiment/primary_outcomes_compact.json"
DEFAULT_ROBUSTNESS_COMPACT = ROOT / "results/raw/robustness_compact"
DEFAULT_OUTPUT = ROOT / "results/aggregates/a53_weight_grid_summary.json"
SEED = 20260801
N_BOOT = 10_000
MARGIN_PP = 2.0

MODELS = [
    "Qwen_Qwen2.5-Coder-14B-Instruct",
    "Qwen_Qwen2.5-Coder-32B-Instruct",
    "meta-llama_Llama-3.1-70B-Instruct",
    "openai_gpt-oss-120b",
    "Qwen_Qwen3-Coder-480B-A35B-Instruct",
]
def match_vector(base, model):
    data = json.loads((base / model / "benchmark_results_condC.json").read_text())
    vec = {r["bench_index"]: bool(r["match"]) for r in data["item_results"]}
    if len(vec) != data["summary"]["total"]:
        raise SystemExit(f"duplicate bench_index in {base}/{model}")
    item_ids = {r["bench_index"]: r["item_id"] for r in data["item_results"]}
    return vec, item_ids


def deployed_vector(main: dict, model: str, item_ids: dict[int, str]) -> dict[int, bool]:
    axis = main["datasets"]["powercodebench_2000"]["item_ids"]
    bits = main["runs"][f"comparison|{model}|C"]["match_bits"]
    if len(axis) != len(bits):
        raise SystemExit(f"invalid main compact vector for {model}")
    by_id = dict(zip(axis, map(lambda bit: bit == "1", bits)))
    if set(by_id) != set(item_ids.values()):
        raise SystemExit(f"item sets differ between deployed and risk-grid arms for {model}")
    return {index: by_id[item_id] for index, item_id in item_ids.items()}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, j) for j in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-compact", type=Path, default=DEFAULT_MAIN_COMPACT,
        help="Path to primary_outcomes_compact.json",
    )
    parser.add_argument("--robustness-compact", type=Path,
                        default=DEFAULT_ROBUSTNESS_COMPACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main_compact = json.loads(args.main_compact.read_text())
    weightings = {
        "uniform": args.robustness_compact / "comparison_riskw_uniform",
        "l3heavy": args.robustness_compact / "comparison_riskw_l3heavy",
        "l0heavy": args.robustness_compact / "comparison_riskw_l0heavy",
    }

    out = {"meta": {"seed": SEED, "n_boot": N_BOOT, "margin_pp": MARGIN_PP,
                    "design": "item-paired over the frozen 2000-item benchmark"},
           "models": {}}
    grid_ok = True
    for model in MODELS:
        loaded = {w: match_vector(base, model) for w, base in weightings.items()}
        vecs = {w: row[0] for w, row in loaded.items()}
        item_maps = {w: row[1] for w, row in loaded.items()}
        reference_ids = item_maps["uniform"]
        if any(mapping != reference_ids for mapping in item_maps.values()):
            raise SystemExit(f"bench_index/item_id map differs across weightings for {model}")
        vecs = {"deployed": deployed_vector(main_compact, model, reference_ids), **vecs}
        keys = set(next(iter(vecs.values())))
        if any(set(v) != keys for v in vecs.values()):
            raise SystemExit(f"item sets differ across weightings for {model}")
        idx = sorted(keys)
        n = len(idx)
        acc = {w: sum(v.values()) / n * 100 for w, v in vecs.items()}

        pairs, max_abs = {}, 0.0
        names = list(vecs)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                w1, w2 = names[i], names[j]
                v1, v2 = vecs[w1], vecs[w2]
                delta = (acc[w2] - acc[w1])
                b = sum(1 for k in idx if v1[k] and not v2[k])
                c = sum(1 for k in idx if not v1[k] and v2[k])
                rng = random.Random(SEED)
                diffs = [v2[k] - v1[k] for k in idx]
                boot = []
                for _ in range(N_BOOT):
                    boot.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n * 100)
                boot.sort()
                pairs[f"{w1}->{w2}"] = {
                    "delta_pp": round(delta, 2),
                    "ci95_pp": [round(boot[int(0.025 * N_BOOT)], 2),
                                round(boot[int(0.975 * N_BOOT)], 2)],
                    "discordant": [b, c],
                    "mcnemar_exact_p": mcnemar_exact(b, c),
                }
                max_abs = max(max_abs, abs(delta))
        within = max_abs <= MARGIN_PP
        grid_ok &= within
        out["models"][model] = {
            "acc_pct": {w: round(a, 2) for w, a in acc.items()},
            "pairwise": pairs,
            "max_pairwise_abs_pp": round(max_abs, 2),
            "within_margin": within,
        }

    out["verdict"] = {
        "all_models_within_margin": grid_ok,
        "grid_max_pairwise_abs_pp": round(
            max(m["max_pairwise_abs_pp"] for m in out["models"].values()), 2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))

    print(f"{'model':44s} {'deploy':>7s} {'unif':>7s} {'l3hvy':>7s} {'l0hvy':>7s} {'max|d|':>7s} {'<=2pp':>6s}")
    for m, r in out["models"].items():
        a = r["acc_pct"]
        print(f"{m:44s} {a['deployed']:7.2f} {a['uniform']:7.2f} {a['l3heavy']:7.2f} "
              f"{a['l0heavy']:7.2f} {r['max_pairwise_abs_pp']:7.2f} "
              f"{'PASS' if r['within_margin'] else 'FAIL':>6s}")
    v = out["verdict"]
    print(f"\ngrid max pairwise |delta| = {v['grid_max_pairwise_abs_pp']}pp; "
          f"all within {MARGIN_PP}pp margin: {v['all_models_within_margin']}")


if __name__ == "__main__":
    main()
