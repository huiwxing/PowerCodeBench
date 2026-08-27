#!/usr/bin/env python3
"""Reaggregate all available S4 item evidence using only the stdlib."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
DEFAULT_INPUT = ROOT / "results/supplementary_evidence/s4_item_evidence.json"
DEFAULT_DEMAND_INPUT = ROOT / "results/supplementary_evidence/demand_estimators.json"
DEFAULT_OUTPUT = ROOT / "results/aggregates/s4_item_evidence_recomputed.json"
KS = (1, 3, 5, 10, 20)
EXPECTED_FROZEN_SLICES = {
    "augmented_dev",
    "augmented_dev_original_only",
    "augmented_dev_variant_only",
    "augmented_test",
    "augmented_test_original_only",
    "augmented_test_variant_only",
    "benchmark_reference_eval_only",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(gold: dict[str, list[str]], predictions: list[dict], field: str) -> dict:
    max_k = 20 if field == "top20" else 10
    output = {"n_eval": len(predictions), "top_k": {}}
    for k in (value for value in KS if value <= max_k):
        recall_sum = precision_sum = 0.0
        hits = 0
        for row in predictions:
            truth = set(gold[row["sample_id"]])
            predicted = set(row[field][:k])
            overlap = truth & predicted
            recall_sum += len(overlap) / len(truth)
            precision_sum += len(overlap) / k
            hits += bool(overlap)
        output["top_k"][str(k)] = {
            "recall": round(recall_sum / len(predictions), 4),
            "precision": round(precision_sum / len(predictions), 4),
            "hit_rate": round(hits / len(predictions), 4),
        }
    return output


def frozen_metric_check(recomputed: dict, archived: dict) -> bool:
    return (
        recomputed["n_eval"] == archived["n_examples"]
        and all(
            recomputed["top_k"][str(k)] == archived["top_k"][str(k)]
            for k in KS
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--demand-input", type=Path, default=DEFAULT_DEMAND_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = json.loads(args.input.read_text(encoding="utf-8"))
    demand = json.loads(args.demand_input.read_text(encoding="utf-8"))
    if artifact.get("schema_version") not in (1, 2, 3, 4):
        raise ValueError("unsupported S4 evidence schema")

    gold = {}
    for split, data in artifact["evaluation_sets"].items():
        rows = data["items"]
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"{split}: duplicate sample IDs")
        if any(not row["gold_functions"] for row in rows):
            raise AssertionError(f"{split}: empty gold functions")
        gold[split] = {row["sample_id"]: row["gold_functions"] for row in rows}
    gold["Aug-orig"] = {
        row["sample_id"]: row["gold_functions"]
        for row in artifact["evaluation_sets"]["Aug-test"]["items"]
        if not row["is_variant"]
    }

    raw_checks = {}
    raw_metrics = {}
    for label, data in artifact["historical_benchmark_top10"].items():
        result = metrics(gold["Bench-ref"], data["predictions"], "top10")
        raw_metrics[label] = result
        raw_checks[label] = (
            result == data["filtered_export_metrics"] and data["raw_structure_check"]
        )
    if not all(raw_checks.values()):
        raise AssertionError("historical S4 raw top-10 reaggregation failed")

    reconstruction_checks = {}
    reconstruction_metrics = {}
    frozen_slice_checks = {}
    for label, method in artifact["reconstructions"].items():
        reconstruction_checks[label] = {}
        reconstruction_metrics[label] = {}
        for split, data in method["splits"].items():
            predictions = data["predictions"]
            if set(row["sample_id"] for row in predictions) != set(gold[split]):
                raise AssertionError(f"{label}/{split}: prediction IDs differ from gold")
            if any(len(row["top20"]) != 20 or len(set(row["top20"])) != 20
                   for row in predictions):
                raise AssertionError(f"{label}/{split}: malformed top-20")
            result = metrics(gold[split], predictions, "top20")
            reconstruction_metrics[label][split] = result
            reconstruction_checks[label][split] = result == data["metrics"]
        if "all_seven_frozen_slice_checks" in method:
            if (
                "frozen_metrics" in method
                and method["frozen_metrics"] != demand["runs"][label]["metrics"]
            ):
                raise AssertionError(f"{label}: frozen metrics differ from public demand evidence")
            frozen_slice_checks[label] = method["all_seven_frozen_slice_checks"]
            if (
                set(frozen_slice_checks[label]) != EXPECTED_FROZEN_SLICES
                or not all(frozen_slice_checks[label].values())
            ):
                raise AssertionError(f"{label}: incomplete frozen-slice validation")
    if not all(
        value for method in reconstruction_checks.values() for value in method.values()
    ):
        raise AssertionError("S4 exact reconstruction reaggregation failed")

    revision_checks = {}
    revision_metrics = {}
    revision_frozen_slice_checks = {}
    revision_recall_deltas_pp = {}
    revision_first_stage_diagnostics = {}
    for label, method in artifact.get("revision_reruns", {}).items():
        if method.get("exact_historical_metric_reconstruction") is not False:
            raise AssertionError(f"{label}: revision rerun lacks an explicit non-exact label")
        revision_checks[label] = {}
        revision_metrics[label] = {}
        for split, data in method["splits"].items():
            predictions = data["predictions"]
            if set(row["sample_id"] for row in predictions) != set(gold[split]):
                raise AssertionError(f"{label}/{split}: revision IDs differ from gold")
            if any(len(row["top20"]) != 20 or len(set(row["top20"])) != 20
                   for row in predictions):
                raise AssertionError(f"{label}/{split}: malformed revision top-20")
            result = metrics(gold[split], predictions, "top20")
            revision_metrics[label][split] = result
            revision_checks[label][split] = result == data["metrics"]

        rerun_all = method["all_seven_frozen_slice_metrics"]
        frozen_all = method["frozen_metrics"]
        public_frozen = demand["runs"][label]
        if frozen_all != public_frozen["metrics"]:
            raise AssertionError(f"{label}: embedded frozen metrics differ from public evidence")
        recorded_checks = method["all_seven_frozen_slice_checks"]
        if (
            set(rerun_all) != EXPECTED_FROZEN_SLICES
            or set(frozen_all) != EXPECTED_FROZEN_SLICES
            or set(recorded_checks) != EXPECTED_FROZEN_SLICES
        ):
            raise AssertionError(f"{label}: incomplete revision/frozen slice ledger")
        computed_checks = {
            split: frozen_metric_check(rerun_all[split], frozen_all[split])
            for split in EXPECTED_FROZEN_SLICES
        }
        if computed_checks != recorded_checks or all(computed_checks.values()):
            raise AssertionError(f"{label}: revision/frozen labels are inconsistent")
        revision_frozen_slice_checks[label] = computed_checks

        first_stage = method["hybrid_first_stage_checks"]
        frozen_alpha = public_frozen["hybrid_alpha_tuning"]["selected"]["alpha"]
        revision_alpha = first_stage["alpha_tuning"]["selected"]["alpha"]
        if (
            first_stage["role_weights"] != public_frozen["role_weights"]
            or not first_stage["role_weights_match"]
            or first_stage["alpha_curve_and_selected_alpha_match"]
            or frozen_alpha == revision_alpha
        ):
            raise AssertionError(f"{label}: first-stage revision diagnostic is inconsistent")
        revision_first_stage_diagnostics[label] = {
            "frozen_selected_alpha": frozen_alpha,
            "revision_selected_alpha": revision_alpha,
            "role_weights_match": True,
            "alpha_curve_and_selected_alpha_match": False,
        }

        split_map = {
            "Aug-test": "augmented_test",
            "Aug-orig": "augmented_test_original_only",
            "Bench-ref": "benchmark_reference_eval_only",
        }
        revision_recall_deltas_pp[label] = {
            display: {
                str(k): round(
                    (
                        rerun_all[archived]["top_k"][str(k)]["recall"]
                        - frozen_all[archived]["top_k"][str(k)]["recall"]
                    ) * 100,
                    2,
                )
                for k in KS
            }
            for display, archived in split_map.items()
        }
    if not all(
        value for method in revision_checks.values() for value in method.values()
    ):
        raise AssertionError("S4 revision-rerun reaggregation failed")

    s4b_metrics = {}
    s4b_checks = {}
    if "s4b_before_after_by_split" in artifact:
        s4b = artifact["s4b_before_after_by_split"]
        for split, data in s4b["splits"].items():
            before_metrics = metrics(gold[split], data["before"]["predictions"], "top20")
            after_metrics = reconstruction_metrics["Hybrid TF-IDF"][split]
            delta = round(
                (after_metrics["top_k"]["10"]["recall"]
                 - before_metrics["top_k"]["10"]["recall"]) * 100,
                2,
            )
            check = (
                before_metrics == data["before"]["metrics"]
                and after_metrics == data["after"]["metrics"]
                and delta == data["recall_at_10_delta_pp"]
                and data["before_all_frozen_k_metrics_match"]
                and data["after_all_frozen_k_metrics_match"]
            )
            s4b_checks[split] = check
            s4b_metrics[split] = {
                "before": before_metrics,
                "after": after_metrics,
                "recall_at_10_delta_pp": delta,
            }
    else:
        s4b = artifact["s4b_aug_test_before_after"]
        before_metrics = metrics(gold["Aug-test"], s4b["before"]["predictions"], "top20")
        after_metrics = reconstruction_metrics["Hybrid TF-IDF"]["Aug-test"]
        delta = round(
            (after_metrics["top_k"]["10"]["recall"]
             - before_metrics["top_k"]["10"]["recall"]) * 100,
            2,
        )
        s4b_checks["Aug-test"] = (
            before_metrics == s4b["before"]["metrics"]
            and after_metrics == s4b["after"]["metrics"]
            and delta == s4b["recall_at_10_delta_pp"]
        )
        s4b_metrics["Aug-test"] = {
            "before": before_metrics,
            "after": after_metrics,
            "recall_at_10_delta_pp": delta,
        }
    if not all(s4b_checks.values()):
        raise AssertionError(f"S4(b) before/after reaggregation failed: {s4b_checks}")

    output = {
        "source": args.input.relative_to(ROOT).as_posix(),
        "source_sha256": sha256_file(args.input),
        "frozen_demand_source": args.demand_input.relative_to(ROOT).as_posix(),
        "frozen_demand_source_sha256": sha256_file(args.demand_input),
        "coverage_note": artifact["coverage_note"],
        "historical_benchmark_top10_metrics": raw_metrics,
        "exact_top20_reconstruction_metrics": reconstruction_metrics,
        "revision_rerun_top20_metrics": revision_metrics,
        "revision_rerun_recall_deltas_pp_vs_frozen": revision_recall_deltas_pp,
        "revision_rerun_first_stage_diagnostics": revision_first_stage_diagnostics,
        "s4b_before_after_by_split": s4b_metrics,
        "unavailable_exact_item_coverage": artifact["unavailable_exact_item_coverage"],
        "integrity_checks": {
            "historical_raw_top10": raw_checks,
            "exact_reconstructions": reconstruction_checks,
            "source_export_frozen_slices": frozen_slice_checks,
            "revision_reruns": revision_checks,
            "revision_rerun_frozen_slice_matches": revision_frozen_slice_checks,
            "s4b_before_after": s4b_checks,
        },
        "all_integrity_checks_pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "raw_benchmark_top10_estimators": len(raw_metrics),
        "full_top20_estimators": len(reconstruction_metrics),
        "revision_rerun_estimators": len(revision_metrics),
        "s4b_recall_at_10_delta_pp": {
            split: data["recall_at_10_delta_pp"] for split, data in s4b_metrics.items()
        },
        "unavailable_full_top20": sorted(artifact["unavailable_exact_item_coverage"]),
        "all_integrity_checks_pass": True,
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
