#!/usr/bin/env python3
"""Export the deterministic revision rerun of the C3 cross-style experiment.

The May 2026 C3 aggregates predate the deterministic split-order fix.  Their
item predictions were not retained, so this script does *not* claim to replay
those historical items.  It reruns the same six frozen configurations with the
fixed split traversal, shares identical fitted arms where the configuration is
identical, and exports every evaluation query, gold label, and top-20 ranking.

This maintainer-side exporter needs NumPy and scikit-learn.  The corresponding
``aggregate_c3_deterministic_rerun.py`` verifier is standard-library only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
TASK_DEMAND_DIR = ROOT / "code/task_demand"
sys.path.insert(0, str(TASK_DEMAND_DIR))

import task_demand_model as tdm  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT / "results/supplementary_evidence/c3_deterministic_rerun_items.json"
)
DEFAULT_PARTS_DIR = ROOT / "results/supplementary_evidence/c3_revision_rerun_parts"
KS = (1, 3, 5, 10, 20)
SEED = 22

RUNS = {
    "R1_baseline_eval_D12": ("baseline", "D12"),
    "R2_baseline_eval_D34": ("baseline", "D34"),
    "R3_reweight_D12_eval_D12": ("reweight_D12", "D12"),
    "R4_reweight_D12_eval_D34": ("reweight_D12", "D34"),
    "R5_reweight_D34_eval_D34": ("reweight_D34", "D34"),
    "R6_reweight_D34_eval_D12": ("reweight_D34", "D12"),
}

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    """Write a compact JSON artifact through a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def describe(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": path.relative_to(ROOT.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def metric_from_rows(rows: list[dict]) -> dict:
    result = {"n_eval": len(rows), "top_k": {}}
    for k in KS:
        recall_sum = 0.0
        precision_sum = 0.0
        hits = 0
        for row in rows:
            truth = set(row["gold_functions"])
            pred = set(row["top20"][:k])
            overlap = truth & pred
            recall_sum += len(overlap) / len(truth)
            precision_sum += len(overlap) / k
            hits += bool(overlap)
        result["top_k"][str(k)] = {
            "recall": round(recall_sum / len(rows), 4),
            "precision": round(precision_sum / len(rows), 4),
            "hit_rate": round(hits / len(rows), 4),
        }
    return result


def fit_arm(
    train: list[tdm.DemandExample],
    dev: list[tdm.DemandExample],
    cards: dict[str, tdm.FunctionCard],
    role_weights: dict[str, float] | None,
) -> tuple[tdm.HybridTfidfDemandRanker, dict]:
    ranker = tdm.HybridTfidfDemandRanker(
        alpha=0.5,
        negative_per_positive=8,
        max_train_pairs=None,
        max_positive_per_function=None,
        seed=SEED,
    )
    ranker.fit(train, cards, role_weights=role_weights)
    tuning = tdm.tune_hybrid_alpha(
        ranker,
        dev,
        cards,
        candidate_alphas=[round(x * 0.1, 1) for x in range(11)],
        target_k=10,
    )
    return ranker, tuning


def ranked_rows(
    ranker: tdm.HybridTfidfDemandRanker,
    examples: list[tdm.DemandExample],
    cards: dict[str, tdm.FunctionCard],
) -> list[dict]:
    names = set(cards)
    rows = []
    for index, example in enumerate(examples, start=1):
        truth = sorted(example.functions & names)
        if not truth:
            continue
        rows.append({
            "ordinal": index,
            "sample_id": example.sample_id,
            "gold_functions": truth,
            "top20": tdm.rank_names(ranker.score(example.question), 20),
        })
    return rows


def historical_comparison() -> tuple[dict, list[dict]]:
    root = ROOT / "results/aggregates/c3_holdout"
    runs = {}
    files = []
    for run_name in RUNS:
        path = root / run_name / "metrics.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        files.append(describe(path))
        runs[run_name] = {
            "selected_alpha": data["alpha_tuning"]["selected"]["alpha"],
            "role_weights": data.get("role_weights"),
            "benchmark_metrics": data["metrics"]["benchmark_reference_eval_only"],
        }
    recall = {
        name: row["benchmark_metrics"]["top_k"]["10"]["recall"]
        for name, row in runs.items()
    }
    lifts = {
        label: round((recall[after] - recall[before]) * 100, 2)
        for label, (before, after) in LIFT_PAIRS.items()
    }
    return {"runs": runs, "recall_at_10_lifts_pp": lifts}, files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--only-arm",
        choices=("baseline", "reweight_D12", "reweight_D34"),
        help="Fit and score one arm into --parts-dir, then exit (lower peak memory).",
    )
    parser.add_argument(
        "--combine-parts",
        action="store_true",
        help="Combine the three previously exported arm parts without refitting.",
    )
    parser.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS_DIR)
    args = parser.parse_args()
    if args.only_arm and args.combine_parts:
        parser.error("--only-arm and --combine-parts are mutually exclusive")

    docs_path = ROOT / "dataset/pandapower_docs.json"
    augmented_path = ROOT / "dataset/augmented_dataset.json"
    split_paths = {
        "D12": ROOT / "benchmark/c3_splits/benchmark_D1D2.json",
        "D34": ROOT / "benchmark/c3_splits/benchmark_D3D4.json",
    }

    cards = tdm.load_function_cards(docs_path)
    function_names = set(cards)
    augmented = tdm.load_augmented_examples(
        augmented_path, function_names, include_variants=True
    )
    train, dev, test = tdm.split_by_group(augmented, seed=SEED)
    evaluation_examples = {
        label: tdm.load_benchmark_reference_examples(path, function_names)
        for label, path in split_paths.items()
    }

    role_weights = {
        "baseline": None,
        "reweight_D12": tdm.compute_role_weights(
            train, evaluation_examples["D12"], cards, max_weight=3.0
        ),
        "reweight_D34": tdm.compute_role_weights(
            train, evaluation_examples["D34"], cards, max_weight=3.0
        ),
    }

    eval_sets = {}
    for split_name, examples in evaluation_examples.items():
        eval_sets[split_name] = {
            "source_path": split_paths[split_name].relative_to(ROOT).as_posix(),
            "items": [
                {
                    "ordinal": index,
                    "sample_id": example.sample_id,
                    "query": example.question,
                    "gold_functions": sorted(example.functions & function_names),
                }
                for index, example in enumerate(examples, start=1)
                if example.functions & function_names
            ],
        }

    arms = {}
    runs = {}
    if args.combine_parts:
        for arm_name in ("baseline", "reweight_D12", "reweight_D34"):
            part_path = args.parts_dir / f"{arm_name}.json"
            part = json.loads(part_path.read_text(encoding="utf-8"))
            if part.get("arm_name") != arm_name or set(part.get("arms", {})) != {arm_name}:
                raise AssertionError(f"malformed C3 arm part: {part_path}")
            arms.update(part["arms"])
            overlap = set(runs) & set(part["runs"])
            if overlap:
                raise AssertionError(f"duplicate runs across C3 parts: {sorted(overlap)}")
            runs.update(part["runs"])
    else:
        # One-arm mode is the recommended maintainer path: scikit-learn may not
        # return the first arm's large sparse allocations to the OS before the
        # next fit, so separate processes keep peak memory bounded.
        arm_names = (
            (args.only_arm,)
            if args.only_arm
            else ("baseline", "reweight_D12", "reweight_D34")
        )
        for arm_name in arm_names:
            print(f"Fitting deterministic C3 arm: {arm_name}", flush=True)
            ranker, tuning = fit_arm(train, dev, cards, role_weights[arm_name])
            arms[arm_name] = {
                "role_weights": role_weights[arm_name],
                "alpha_tuning": tuning,
            }
            for run_name, (run_arm, split_name) in RUNS.items():
                if run_arm != arm_name:
                    continue
                print(f"Scoring {run_name}", flush=True)
                rows = ranked_rows(ranker, evaluation_examples[split_name], cards)
                runs[run_name] = {
                    "arm": arm_name,
                    "evaluation_split": split_name,
                    "predictions": [
                        {
                            "ordinal": row["ordinal"],
                            "sample_id": row["sample_id"],
                            "top20": row["top20"],
                        }
                        for row in rows
                    ],
                    "metrics": metric_from_rows(rows),
                }
            del ranker
            gc.collect()

        if args.only_arm:
            part_path = args.parts_dir / f"{args.only_arm}.json"
            atomic_write_json(part_path, {
                "schema_version": 1,
                "status": "C3 deterministic revision rerun arm part",
                "arm_name": args.only_arm,
                "arms": arms,
                "runs": runs,
            })
            print(f"Wrote arm part {part_path} ({part_path.stat().st_size:,} bytes)")
            return 0

    if set(runs) != set(RUNS):
        raise AssertionError(f"incomplete C3 runs: {sorted(runs)}")

    recalls = {
        run_name: run["metrics"]["top_k"]["10"]["recall"]
        for run_name, run in runs.items()
    }
    revision_lifts = {
        label: round((recalls[after] - recalls[before]) * 100, 2)
        for label, (before, after) in LIFT_PAIRS.items()
    }
    historical, historical_files = historical_comparison()

    import numpy
    import sklearn

    source_files = [
        describe(HERE),
        describe(TASK_DEMAND_DIR / "task_demand_model.py"),
        describe(docs_path),
        describe(augmented_path),
        *(describe(path) for path in split_paths.values()),
        *historical_files,
    ]
    output = {
        "schema_version": 1,
        "evidence_unit": "evaluation query x gold functions x top-20 prediction",
        "status": "revision deterministic rerun; not an item-level replay of the pre-fix historical runs",
        "determinism_note": (
            "The historical May 2026 runs iterated split-id sets in hash order and "
            "retained no prediction exports. This rerun uses the fixed sorted split "
            "traversal in code/task_demand/task_demand_model.py."
        ),
        "configuration": {
            "model": "hybrid_tfidf",
            "include_variants": True,
            "seed": SEED,
            "negative_per_positive": 8,
            "hybrid_alpha_initial": 0.5,
            "auto_hybrid_alpha": True,
            "hybrid_alpha_target_k": 10,
            "candidate_alphas": [round(x * 0.1, 1) for x in range(11)],
            "top_k": list(KS),
            "role_reweight_max": 3.0,
            "shared_fit_policy": (
                "Runs with identical training/reweight configuration share one "
                "fit; only their evaluation split differs."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "thread_limits": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "provenance": {
            "repository_commit": git_head(),
            "files": source_files,
        },
        "augmented_split_counts": {
            "all": len(augmented),
            "train": len(train),
            "dev": len(dev),
            "test": len(test),
        },
        "arms": arms,
        "evaluation_sets": eval_sets,
        "runs": runs,
        "recall_at_10_lifts_pp": revision_lifts,
        "historical_aggregate_comparison": historical,
    }
    atomic_write_json(args.output, output)
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)")
    print("revision lifts:", json.dumps(revision_lifts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
