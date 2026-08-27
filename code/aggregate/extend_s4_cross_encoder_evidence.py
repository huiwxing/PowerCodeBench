#!/usr/bin/env python3
"""Inject validated Cross-Encoder reconstructions into public S4 evidence.

The Zero-shot arm is admitted to the exact-reconstruction section only when
all seven frozen slices match at every reported k.  The Hybrid arm is kept in
a separate rerun section: its released item rankings can be reaggregated in
full, and they are labelled as a rerun rather than as the historical raw
predictions or an exact reconstruction of the frozen curve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
DEFAULT_EVIDENCE = ROOT / "results/supplementary_evidence/s4_item_evidence.json"
MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-12-v2"
MODEL_REVISION = "7b0235231ca2674cb8ca8f022859a6eba2b1c968"
SOURCE_REVISION = "7916953d4f289bb472224eae958935d1afb24ce5"
EXPECTED_MODEL_FILES = {
    "config.json": "69a745055b0307584d2903ea5cf4e1899254e511834f358be80d049918b144d8",
    "model.safetensors": "1ed84b90cdf3518f76ec9bb93a16f97887eea7c4e7ee5dfb03cc297e394fbbc3",
    "special_tokens_map.json": "3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "tokenizer_config.json": "a5c2e5a7b1a29a0702cd28c08a399b5ecc110c263009d17f7e3b415f25905fd8",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
EXPECTED_SOURCE_FILES = {
    "task_demand/task_demand_model.py": "b7cc2cd1446a9b03a064bac62e6db82aa40fb96b365262276333464ca0338bf3",
    "task_demand/run_task_demand.sh": "36de7a8e20a8a5cb13c7b0cc0e5b4955332c273aeea71384630555ab8156ea18",
    "dataset/augmented_dataset.json": "19ddccf566f0cf1938abd94ebbc7025e07d4d2fc62e55fe52235ffbc1b080d73",
    "dataset/pandapower_docs.json": "453599cc83e09e110c6be9c9b82bf4a1f6583d04ccdb2938e8deafb68f03701e",
    "benchmark/benchmark6.json": "fc4df22306d3803a25e985e6bebdb9fc91f27d00a96fad297aa9166fb5b70519",
}
EXPECTED_FROZEN_METRICS = {
    "Zero-shot Cross-Encoder": "53a83e7b7180ff3e0f721489822f362b87685a8dd66d01505edf6b2e64ef371d",
    "Hybrid Cross-Encoder": "c1ab34f9b4daed1c809ce0d81e33ab9fcee505c0c2e0845c1d8dc1d5cde1f241",
}
HISTORICAL_LOG_SHA256 = "f3f08b0e394ba5ae3cb8ffd6a294225df144ead1c40462a2400b47e827ea78aa"
HISTORICAL_LOG_BYTES = 124291
EXPECTED_SLICES = {
    "augmented_dev",
    "augmented_dev_original_only",
    "augmented_dev_variant_only",
    "augmented_test",
    "augmented_test_original_only",
    "augmented_test_variant_only",
    "benchmark_reference_eval_only",
}
KS = (1, 3, 5, 10, 20)


_HISTORICAL_ROOT_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:[^/\s()'\"<>]+/)+myprojects/igpt"
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def metrics(gold: dict[str, list[str]], predictions: list[dict]) -> dict:
    output = {"n_eval": len(predictions), "top_k": {}}
    for k in KS:
        recall_sum = precision_sum = 0.0
        hits = 0
        for row in predictions:
            truth = set(gold[row["sample_id"]])
            predicted = set(row["top20"][:k])
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


def public_gold(artifact: dict) -> dict[str, dict[str, list[str]]]:
    gold = {
        split: {row["sample_id"]: row["gold_functions"] for row in data["items"]}
        for split, data in artifact["evaluation_sets"].items()
    }
    gold["Aug-orig"] = {
        row["sample_id"]: row["gold_functions"]
        for row in artifact["evaluation_sets"]["Aug-test"]["items"]
        if not row["is_variant"]
    }
    return gold


def validate_snapshot(recovery: dict) -> None:
    snapshot = recovery["model_snapshot"]
    if snapshot["model_id"] != MODEL_ID or snapshot["revision"] != MODEL_REVISION:
        raise AssertionError("unexpected Cross-Encoder model snapshot")
    found = {row["path"]: row["sha256"] for row in snapshot["files"]}
    if found != EXPECTED_MODEL_FILES:
        raise AssertionError("Cross-Encoder snapshot files differ from the pinned revision")


def validate_source(recovery: dict, label: str, artifact: dict) -> None:
    if recovery["source_revision"] != SOURCE_REVISION:
        raise AssertionError(f"{label}: unexpected historical source revision")
    blobs = {
        row["path"]: row["sha256"]
        for row in recovery["provenance"]["historical_source_blobs"]
    }
    if blobs != EXPECTED_SOURCE_FILES:
        raise AssertionError(f"{label}: historical source blobs differ")
    descriptors = recovery["provenance"]["frozen_metrics"]
    if len(descriptors) != 1 or descriptors[0]["sha256"] != EXPECTED_FROZEN_METRICS[label]:
        raise AssertionError(f"{label}: wrong frozen metric source")
    public_descriptor = next(
        row for row in artifact["provenance"]["files"]
        if row["sha256"] == EXPECTED_FROZEN_METRICS[label]
    )
    if public_descriptor["bytes"] != descriptors[0]["bytes"]:
        raise AssertionError(f"{label}: public frozen metric descriptor differs")


def validate_evaluation_sets(recovery: dict, artifact: dict, label: str) -> None:
    for split in ("Aug-test", "Bench-ref"):
        public_rows = {
            row["sample_id"]: row
            for row in artifact["evaluation_sets"][split]["items"]
        }
        recovery_rows = {
            row["sample_id"]: row
            for row in recovery["evaluation_sets"][split]["items"]
        }
        if public_rows != recovery_rows:
            raise AssertionError(f"{label}/{split}: public evaluation records differ")


def validate_method(recovery: dict, artifact: dict, label: str) -> dict:
    validate_snapshot(recovery)
    validate_source(recovery, label, artifact)
    validate_evaluation_sets(recovery, artifact, label)
    if not recovery["all_integrity_checks_pass"]:
        raise AssertionError(f"{label}: standalone integrity checks failed")
    method = recovery["reconstructions"][label]
    checks = method["all_seven_frozen_slice_checks"]
    if set(checks) != EXPECTED_SLICES:
        raise AssertionError(f"{label}: incomplete seven-slice ledger")

    gold = public_gold(artifact)
    for split, data in method["splits"].items():
        rows = data["predictions"]
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(gold[split]):
            raise AssertionError(f"{label}/{split}: item IDs differ")
        if any(len(row["top20"]) != 20 or len(set(row["top20"])) != 20 for row in rows):
            raise AssertionError(f"{label}/{split}: malformed top-20")
        if metrics(gold[split], rows) != data["metrics"]:
            raise AssertionError(f"{label}/{split}: metrics do not reaggregate")
    return method


def input_descriptor(path: Path, recovery: dict) -> dict:
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "status": recovery["status"],
        "evidence_type": recovery["evidence_type"],
    }


def code_descriptor(path: Path) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-reconstruction", type=Path, required=True)
    parser.add_argument("--hybrid-rerun", type=Path, required=True)
    parser.add_argument("--historical-log", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args()

    artifact = json.loads(args.input.read_text(encoding="utf-8"))
    zero = json.loads(args.zero_reconstruction.read_text(encoding="utf-8"))
    hybrid = json.loads(args.hybrid_rerun.read_text(encoding="utf-8"))
    historical_log = args.historical_log.read_text(encoding="utf-8")
    if (
        args.historical_log.stat().st_size != HISTORICAL_LOG_BYTES
        or sha256_file(args.historical_log) != HISTORICAL_LOG_SHA256
    ):
        raise AssertionError("unexpected historical S4 execution log")
    zero_method = validate_method(zero, artifact, "Zero-shot Cross-Encoder")
    hybrid_method = validate_method(hybrid, artifact, "Hybrid Cross-Encoder")

    if not zero["exact_historical_metric_reconstruction"] or not all(
        zero_method["all_seven_frozen_slice_checks"].values()
    ):
        raise AssertionError("Zero-shot Cross-Encoder is not an exact reconstruction")
    if hybrid["exact_historical_metric_reconstruction"] or all(
        hybrid_method["all_seven_frozen_slice_checks"].values()
    ):
        raise AssertionError("Hybrid Cross-Encoder must remain a revision rerun")
    first_stage = hybrid["hybrid_first_stage_checks"]
    if not first_stage["role_weights_match"] or first_stage[
        "alpha_curve_and_selected_alpha_match"
    ]:
        raise AssertionError("unexpected Hybrid first-stage diagnostic")
    if zero["model_snapshot"] != hybrid["model_snapshot"]:
        raise AssertionError("Cross-Encoder runs used different model snapshots")

    lines = historical_log.splitlines()
    hybrid_command = next(
        line for line in lines if "--run-name hybrid_cross_encoder" in line
    )
    command_index = lines.index(hybrid_command)
    next_command = next(
        (index for index in range(command_index + 1, len(lines))
         if lines[index].startswith(">>> ")),
        len(lines),
    )
    hybrid_segment = "\n".join(lines[command_index:next_command])
    if (
        "--save-model" in hybrid_command
        or '"saved_model": null' not in hybrid_segment
        or "PYTHONHASHSEED" in historical_log
    ):
        raise AssertionError("historical Hybrid retention evidence differs")

    preserved = {
        "evaluation_sets": canonical_sha256(artifact["evaluation_sets"]),
        "historical_benchmark_top10": canonical_sha256(
            artifact["historical_benchmark_top10"]
        ),
        "s4b_before_after_by_split": canonical_sha256(
            artifact["s4b_before_after_by_split"]
        ),
        "preexisting_reconstructions": {
            label: canonical_sha256(data)
            for label, data in artifact["reconstructions"].items()
            if label != "Zero-shot Cross-Encoder"
        },
    }

    artifact["schema_version"] = 4
    artifact["status"] = (
        "mixed historical raw evidence, exact pinned reconstructions, and a "
        "separately labelled deterministic revision rerun"
    )
    artifact["coverage_note"] = (
        "Full unfiltered top-20 curves are retained for the three TF-IDF "
        "estimators, Zero-shot SBERT, the exact Zero-shot Cross-Encoder "
        "reconstruction, and the S4(b) unadapted Hybrid arm. Hybrid "
        "Cross-Encoder has a separately labelled locked-stack revision rerun; "
        "its frozen aggregate curve and historical role-filtered deployment "
        "top-10 remain distinct from those revision rankings."
    )
    artifact["reconstructions"]["Zero-shot Cross-Encoder"] = zero_method
    hybrid_method["evidence_type"] = hybrid["evidence_type"]
    hybrid_method["exact_historical_metric_reconstruction"] = False
    hybrid_method["comparison_label"] = (
        "conclusion-stable quantitative drift; non-exact historical reconstruction"
    )
    hybrid_method["hybrid_first_stage_checks"] = first_stage
    artifact["revision_reruns"] = {"Hybrid Cross-Encoder": hybrid_method}
    artifact["unavailable_exact_item_coverage"].pop("Zero-shot Cross-Encoder", None)
    artifact["unavailable_exact_item_coverage"]["Hybrid Cross-Encoder"] = {
        "available": (
            "frozen aggregate curves, historical role-filtered Bench-ref top-10, "
            "and a separately labelled deterministic unfiltered top-20 revision rerun"
        ),
        "missing": "historical unfiltered item predictions used by the frozen S4 curve",
        "reason": (
            "The historical Hybrid first-stage estimator and process hash state were "
            "not retained. The locked-stack rerun reproduces the role weights but not "
            "the alpha-selection curve or frozen metrics, so it remains revision evidence."
        ),
    }
    artifact["environment"]["cross_encoder_recovery"] = {
        "zero_exact": zero["environment"],
        "hybrid_revision_rerun": hybrid["environment"],
    }
    artifact["provenance"]["cross_encoder_recovery"] = {
        "source_revision": SOURCE_REVISION,
        "historical_source_blobs": zero["provenance"]["historical_source_blobs"],
        "model_snapshot": zero["model_snapshot"],
        "recovery_code": zero["provenance"]["recovery_code"],
        "extension_code": code_descriptor(HERE),
        "standalone_inputs": {
            "Zero-shot Cross-Encoder": input_descriptor(args.zero_reconstruction, zero),
            "Hybrid Cross-Encoder": input_descriptor(args.hybrid_rerun, hybrid),
        },
        "historical_execution": zero["historical_execution"],
        "historical_job_log": {
            "bytes": HISTORICAL_LOG_BYTES,
            "sha256": HISTORICAL_LOG_SHA256,
        },
        "hybrid_historical_retention": {
            "normalized_command": _HISTORICAL_ROOT_RE.sub(
                "<HISTORICAL_IGPT_ROOT>", hybrid_command
            ),
            "save_model_argument_present": False,
            "saved_model": None,
            "pythonhashseed_logged": False,
            "implication": (
                "The historical first-stage weights and process hash state cannot be "
                "recovered from the retained execution record."
            ),
        },
        "zero_exact_historical_metric_reconstruction": True,
        "hybrid_revision_rerun": {
            "exact_historical_metric_reconstruction": False,
            "hybrid_first_stage_checks": first_stage,
        },
    }
    artifact["preserved_section_sha256_during_cross_encoder_extension"] = preserved

    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": sha256_file(args.output),
        "zero_exact_seven_slices": all(
            zero_method["all_seven_frozen_slice_checks"].values()
        ),
        "hybrid_frozen_slice_matches": sum(
            hybrid_method["all_seven_frozen_slice_checks"].values()
        ),
        "preserved_sections": preserved,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
