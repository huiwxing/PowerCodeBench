#!/usr/bin/env python3
"""Pure-stdlib recomputation of the E4 seed-42--46 selector evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve()
PCB_ROOT = HERE.parents[2]
DEFAULT_PREDICTIONS = PCB_ROOT / "results/supplementary_evidence/e4_multiseed_predictions.json"
DEFAULT_BASE = PCB_ROOT / "results/supplementary_evidence/e4_item_predictions.json"
DEFAULT_SUMMARY_DIR = PCB_ROOT / "external_queries/conditional_selector/seed_sensitivity"
DEFAULT_OUTPUT = PCB_ROOT / "results/aggregates/e4_multiseed_recomputed.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        stream.write("\n")


def metrics(items: list[dict], arm: str, ks: list[int]) -> dict:
    result = {}
    for k in ks:
        recall_sum = 0.0
        precision_sum = 0.0
        hit_sum = 0.0
        for item in items:
            truth = set(item["gold_functions"])
            predicted = set(item["top20"][arm][:k])
            overlap = truth & predicted
            recall_sum += len(overlap) / len(truth)
            precision_sum += len(overlap) / k
            hit_sum += float(bool(overlap))
        n = len(items)
        result[str(k)] = {
            "recall": round(recall_sum / n, 4),
            "precision": round(precision_sum / n, 4),
            "hit_rate": round(hit_sum / n, 4),
        }
    return result


def record_check(checks: list[dict], name: str, recomputed, archived) -> None:
    checks.append({
        "name": name,
        "pass": recomputed == archived,
        "recomputed": recomputed,
        "archived": archived,
    })


def aggregate(predictions_path: Path, base_path: Path, summary_dir: Path) -> dict:
    compact = read_json(predictions_path)
    base = read_json(base_path)
    seeds = [int(seed) for seed in compact["seeds"]]
    ks = [int(k) for k in compact["top_k_support"]]
    set_order = compact["set_order"]
    axis = [tuple(key) for key in compact["item_axis"]]
    dictionary = compact["function_dictionary"]

    base_items = {
        (set_name, item["sample_id"]): item
        for set_name, set_data in base["sets"].items()
        for item in set_data["items"]
    }
    checks = []
    record_check(checks, "item_axis_unique", len(set(axis)), len(axis))
    record_check(
        checks,
        "item_axis_matches_seed22_evidence",
        sorted(axis),
        sorted(base_items),
    )
    record_check(checks, "item_count", len(axis), 194)

    per_seed = {}
    conditional_r10 = {set_name: [] for set_name in set_order}
    for seed in seeds:
        archived = read_json(summary_dir / f"metrics_by_set_seed{seed}.json")
        encoded = compact["top20_ids_by_seed"][str(seed)]
        decoded = {}
        for arm in ("unadapted", "adapted"):
            rankings = encoded[arm]
            record_check(checks, f"seed{seed}.{arm}.item_count", len(rankings), len(axis))
            decoded[arm] = []
            for index, ranking in enumerate(rankings):
                valid_ids = all(isinstance(value, int) and 0 <= value < len(dictionary) for value in ranking)
                record_check(checks, f"seed{seed}.{arm}.item{index}.valid_ids", valid_ids, True)
                names = [dictionary[value] for value in ranking] if valid_ids else []
                record_check(checks, f"seed{seed}.{arm}.item{index}.top20_unique", len(set(names)), 20)
                record_check(checks, f"seed{seed}.{arm}.item{index}.top20_length", len(names), 20)
                decoded[arm].append(names)

        pipeline = compact["pipeline_by_seed"][str(seed)]
        record_check(
            checks,
            f"seed{seed}.unadapted_alpha",
            pipeline["unadapted_alpha"],
            archived["arms"]["unadapted"]["alpha"],
        )
        record_check(
            checks,
            f"seed{seed}.adapted_alpha",
            pipeline["adapted_alpha"],
            archived["arms"]["adapted"]["alpha"],
        )
        record_check(
            checks,
            f"seed{seed}.adapted_role_weights",
            pipeline["adapted_role_weights"],
            archived["arms"]["adapted"]["role_weights"],
        )

        seed_result = {"pipeline": pipeline, "sets": {}}
        by_set = {set_name: [] for set_name in set_order}
        for index, key in enumerate(axis):
            base_item = base_items.get(key)
            if base_item is None:
                continue
            selected_arm = "adapted" if base_item["selector_intent"] == "analysis" else "unadapted"
            item = {
                "sample_id": key[1],
                "gold_functions": base_item["gold_functions"],
                "selector_intent": base_item["selector_intent"],
                "top20": {
                    "unadapted": decoded["unadapted"][index],
                    "adapted": decoded["adapted"][index],
                    "conditional": decoded[selected_arm][index],
                },
            }
            by_set[key[0]].append(item)

        for set_name in set_order:
            rows = by_set[set_name]
            archived_set = archived["sets"][set_name]
            record_check(checks, f"seed{seed}.{set_name}.n", len(rows), archived_set["n"])
            selector_decisions = {item["sample_id"]: item["selector_intent"] for item in rows}
            record_check(
                checks,
                f"seed{seed}.{set_name}.selector_decisions",
                selector_decisions,
                archived_set["selector_decisions"],
            )
            recomputed_metrics = {}
            for arm in ("unadapted", "adapted", "conditional"):
                recomputed_metrics[arm] = metrics(rows, arm, ks)
                record_check(
                    checks,
                    f"seed{seed}.{set_name}.{arm}.top_k",
                    recomputed_metrics[arm],
                    archived_set["metrics"][arm],
                )
            seed_result["sets"][set_name] = {
                "n": len(rows),
                "metrics": recomputed_metrics,
            }
            conditional_r10[set_name].append(recomputed_metrics["conditional"]["10"]["recall"])
        per_seed[str(seed)] = seed_result

    labels = {
        "layer1_holdout_n80": "Internal holdout",
        "layer2a_n84": "Naturally occurring",
        "layer2a_ext_n10": "Cross-platform",
        "layer2b_n20": "Engineer-designed",
    }
    manuscript_summary = {}
    for set_name in set_order:
        values = conditional_r10[set_name]
        manuscript_summary[set_name] = {
            "label": labels[set_name],
            "seed_values": {str(seed): value for seed, value in zip(seeds, values)},
            "conditional_recall_at_10_percent": {
                "mean": round(sum(values) / len(values) * 100, 2),
                "min": round(min(values) * 100, 2),
                "max": round(max(values) * 100, 2),
            },
        }

    mismatches = [check for check in checks if not check["pass"]]
    return {
        "schema_version": 1,
        "method": (
            "Decode the two independent per-query top-20 arms; derive the frozen "
            "conditional selector arm from the seed-22 item metadata; recompute "
            "query-macro recall, precision and hit-rate at k={1,3,5,10,20}."
        ),
        "inputs": {
            "multiseed_predictions": {
                "path": predictions_path.relative_to(PCB_ROOT).as_posix(),
                "sha256": sha256_file(predictions_path),
            },
            "seed22_item_metadata": {
                "path": base_path.relative_to(PCB_ROOT).as_posix(),
                "sha256": sha256_file(base_path),
            },
            "archived_seed_summaries": {
                str(seed): {
                    "path": (summary_dir / f"metrics_by_set_seed{seed}.json")
                    .relative_to(PCB_ROOT)
                    .as_posix(),
                    "sha256": sha256_file(summary_dir / f"metrics_by_set_seed{seed}.json"),
                }
                for seed in seeds
            },
        },
        "per_seed": per_seed,
        "manuscript_summary": manuscript_summary,
        "validation": {
            "status": "pass" if not mismatches else "fail",
            "n_checks": len(checks),
            "n_mismatches": len(mismatches),
            "mismatches": mismatches,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--base-items", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = aggregate(args.predictions, args.base_items, args.summary_dir)
    write_json(args.output, result)
    validation = result["validation"]
    print(
        f"{validation['status']}: {validation['n_checks']} checks, "
        f"{validation['n_mismatches']} mismatches -> {args.output}"
    )
    for set_name, row in result["manuscript_summary"].items():
        stats = row["conditional_recall_at_10_percent"]
        print(f"{set_name}: {stats['mean']:.2f} [{stats['min']:.2f}, {stats['max']:.2f}]")
    return 0 if validation["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
