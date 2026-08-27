#!/usr/bin/env python3
"""Recompute C3 revision metrics from public per-query top-20 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
DEFAULT_INPUT = (
    ROOT / "results/supplementary_evidence/c3_deterministic_rerun_items.json"
)
DEFAULT_OUTPUT = ROOT / "results/aggregates/c3_deterministic_rerun_recomputed.json"
KS = (1, 3, 5, 10, 20)

LIFT_PAIRS = {
    "D12_source_in_distribution": (
        "R1_baseline_eval_D12", "R3_reweight_D12_eval_D12"
    ),
    "D12_source_held_out_D34": (
        "R2_baseline_eval_D34", "R4_reweight_D12_eval_D34"
    ),
    "D34_source_in_distribution": (
        "R2_baseline_eval_D34", "R5_reweight_D34_eval_D34"
    ),
    "D34_source_held_out_D12": (
        "R1_baseline_eval_D12", "R6_reweight_D34_eval_D12"
    ),
}

ENDPOINT_PAIRS = {
    "D12_eval_cross_D34_vs_in_D12": {
        "evaluation_split": "D12",
        "in_source_run": "R3_reweight_D12_eval_D12",
        "cross_source_run": "R6_reweight_D34_eval_D12",
    },
    "D34_eval_cross_D12_vs_in_D34": {
        "evaluation_split": "D34",
        "in_source_run": "R5_reweight_D34_eval_D34",
        "cross_source_run": "R4_reweight_D12_eval_D34",
    },
}
BOOTSTRAP_SEED = 22
BOOTSTRAP_REPLICATES = 10_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(rows: list[dict]) -> dict:
    output = {"n_eval": len(rows), "top_k": {}}
    for k in KS:
        recall_sum = precision_sum = 0.0
        hit_count = 0
        for row in rows:
            truth = set(row["gold_functions"])
            pred = set(row["top20"][:k])
            overlap = truth & pred
            recall_sum += len(overlap) / len(truth)
            precision_sum += len(overlap) / k
            hit_count += bool(overlap)
        output["top_k"][str(k)] = {
            "recall": round(recall_sum / len(rows), 4),
            "precision": round(precision_sum / len(rows), 4),
            "hit_rate": round(hit_count / len(rows), 4),
        }
    return output


def percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap_mean_difference_pp(
    differences: list[float], *, seed: int, replicates: int
) -> dict:
    rng = random.Random(seed)
    n_items = len(differences)
    values = []
    for _ in range(replicates):
        total = 0.0
        for _ in range(n_items):
            total += differences[rng.randrange(n_items)]
        values.append(total / n_items * 100.0)
    values.sort()
    return {
        "seed": seed,
        "replicates": replicates,
        "unit": "paired evaluation query",
        "statistic": "mean(cross-source item recall@10 - in-source item recall@10), pp",
        "ci95_pp": [
            round(percentile(values, 0.025), 2),
            round(percentile(values, 0.975), 2),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    artifact = json.loads(args.input.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported C3 evidence schema")

    provenance_checks = {}
    for descriptor in artifact["provenance"]["files"]:
        path = ROOT / descriptor["path"]
        provenance_checks[descriptor["path"]] = {
            "exists": path.is_file(),
            "sha256_matches": path.is_file() and sha256_file(path) == descriptor["sha256"],
            "bytes_match": path.is_file() and path.stat().st_size == descriptor["bytes"],
        }
    if not all(all(check.values()) for check in provenance_checks.values()):
        raise AssertionError("one or more C3 provenance files changed")

    gold_by_split = {}
    expected_counts = {"D12": 1200, "D34": 800}
    for split, data in artifact["evaluation_sets"].items():
        rows = data["items"]
        if len(rows) != expected_counts[split]:
            raise AssertionError(f"{split}: expected {expected_counts[split]} items")
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"{split}: duplicate sample_id")
        if any(not row["gold_functions"] for row in rows):
            raise AssertionError(f"{split}: empty gold functions")
        gold_by_split[split] = {row["sample_id"]: row["gold_functions"] for row in rows}

    recomputed_runs = {}
    metric_checks = {}
    item_recall_at_10 = {}
    for run_name, run in artifact["runs"].items():
        split = run["evaluation_split"]
        gold = gold_by_split[split]
        predictions = run["predictions"]
        if len(predictions) != len(gold):
            raise AssertionError(f"{run_name}: prediction count mismatch")
        prediction_ids = [row["sample_id"] for row in predictions]
        if set(prediction_ids) != set(gold):
            raise AssertionError(f"{run_name}: prediction ids do not match gold set")
        rows = []
        recall_by_id = {}
        for prediction in predictions:
            top20 = prediction["top20"]
            if len(top20) != 20 or len(set(top20)) != 20:
                raise AssertionError(f"{run_name}/{prediction['sample_id']}: malformed top20")
            rows.append({
                "gold_functions": gold[prediction["sample_id"]],
                "top20": top20,
            })
            truth = set(gold[prediction["sample_id"]])
            recall_by_id[prediction["sample_id"]] = (
                len(truth & set(top20[:10])) / len(truth)
            )
        result = metrics(rows)
        recomputed_runs[run_name] = result
        item_recall_at_10[run_name] = recall_by_id
        metric_checks[run_name] = result == run["metrics"]

    if not all(metric_checks.values()):
        raise AssertionError("stored C3 metrics differ from item reaggregation")

    recall = {
        run_name: result["top_k"]["10"]["recall"]
        for run_name, result in recomputed_runs.items()
    }
    lifts = {
        label: round((recall[after] - recall[before]) * 100, 2)
        for label, (before, after) in LIFT_PAIRS.items()
    }
    lift_checks = lifts == artifact["recall_at_10_lifts_pp"]
    if not lift_checks:
        raise AssertionError("stored C3 lifts differ from item reaggregation")

    endpoint_transfer = {}
    bootstrap_determinism = {}
    for label, pair in ENDPOINT_PAIRS.items():
        in_run = pair["in_source_run"]
        cross_run = pair["cross_source_run"]
        in_values = item_recall_at_10[in_run]
        cross_values = item_recall_at_10[cross_run]
        if set(in_values) != set(cross_values):
            raise AssertionError(f"{label}: endpoint runs do not share item support")
        sample_ids = sorted(in_values)
        differences = [cross_values[sid] - in_values[sid] for sid in sample_ids]
        point_gap = sum(differences) / len(differences) * 100.0
        bootstrap = paired_bootstrap_mean_difference_pp(
            differences, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES
        )
        repeated = paired_bootstrap_mean_difference_pp(
            differences, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES
        )
        deterministic = bootstrap == repeated
        if not deterministic:
            raise AssertionError(f"{label}: paired bootstrap is not deterministic")
        bootstrap_determinism[label] = deterministic
        endpoint_transfer[label] = {
            **pair,
            "n_paired_queries": len(sample_ids),
            "in_source_recall_at_10": recomputed_runs[in_run]["top_k"]["10"]["recall"],
            "cross_source_recall_at_10": recomputed_runs[cross_run]["top_k"]["10"]["recall"],
            "cross_minus_in_source_pp": round(point_gap, 2),
            "paired_bootstrap": bootstrap,
            "interpretation": "cross-source reaches nearly the same recall endpoint",
        }

    output = {
        "source": args.input.relative_to(ROOT).as_posix(),
        "source_sha256": sha256_file(args.input),
        "status": artifact["status"],
        "runs": recomputed_runs,
        "recall_at_10_lifts_pp": lifts,
        "cross_source_endpoint_comparison": endpoint_transfer,
        "historical_recall_at_10_lifts_pp": artifact[
            "historical_aggregate_comparison"
        ]["recall_at_10_lifts_pp"],
        "integrity_checks": {
            "provenance": provenance_checks,
            "run_metrics_match": metric_checks,
            "lifts_match": lift_checks,
            "paired_bootstrap_deterministic": bootstrap_determinism,
        },
        "all_integrity_checks_pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "n_runs": len(recomputed_runs),
        "lifts_pp": lifts,
        "historical_lifts_pp": output["historical_recall_at_10_lifts_pp"],
        "all_integrity_checks_pass": True,
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
