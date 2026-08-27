#!/usr/bin/env python3
"""Inject a validated pinned-snapshot SBERT reconstruction into S4 evidence.

This is a bounded recovery path: it preserves the existing TF-IDF, S4(b), and
historical-export sections, verifies the standalone SBERT recovery against the
public gold labels and retained historical metrics descriptor, then adds only
the SBERT reconstruction and provenance fields.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
DEFAULT_EVIDENCE = ROOT / "results/supplementary_evidence/s4_item_evidence.json"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
WEIGHT_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
METRICS_PATH = "task_demand/results/suites/suite_4428033/zero_shot_sbert/metrics.json"
KS = (1, 3, 5, 10, 20)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    blob = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def metrics(rows: list[dict]) -> dict:
    output = {"n_eval": len(rows), "top_k": {}}
    for k in KS:
        recall_sum = precision_sum = 0.0
        hits = 0
        for row in rows:
            truth = set(row["gold_functions"])
            predicted = set(row["ranking"][:k])
            overlap = truth & predicted
            recall_sum += len(overlap) / len(truth)
            precision_sum += len(overlap) / k
            hits += bool(overlap)
        output["top_k"][str(k)] = {
            "recall": round(recall_sum / len(rows), 4),
            "precision": round(precision_sum / len(rows), 4),
            "hit_rate": round(hits / len(rows), 4),
        }
    return output


def descriptor(path: Path) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args()

    artifact = json.loads(args.input.read_text(encoding="utf-8"))
    recovery = json.loads(args.reconstruction.read_text(encoding="utf-8"))
    if recovery["snapshot_revision"] != REVISION:
        raise AssertionError("unexpected SBERT snapshot revision")
    weights = next(
        (item for item in recovery["snapshot_files"] if item["path"] == "model.safetensors"),
        None,
    )
    if weights is None or weights["sha256"] != WEIGHT_SHA256:
        raise AssertionError("unexpected SBERT model.safetensors hash")
    expected_slices = {
        "augmented_dev",
        "augmented_dev_original_only",
        "augmented_dev_variant_only",
        "augmented_test",
        "augmented_test_original_only",
        "augmented_test_variant_only",
        "benchmark_reference_eval_only",
    }
    if set(recovery["integrity_checks"]) != expected_slices or not all(
        recovery["integrity_checks"].values()
    ):
        raise AssertionError("standalone recovery did not validate every frozen slice")

    historical_metrics = next(
        item for item in artifact["provenance"]["files"] if item["path"] == METRICS_PATH
    )
    if historical_metrics["sha256"] != recovery["source_metrics_sha256"]:
        raise AssertionError("SBERT recovery used a different historical metrics file")

    gold = {
        split: {row["sample_id"]: row["gold_functions"] for row in data["items"]}
        for split, data in artifact["evaluation_sets"].items()
    }
    gold["Aug-orig"] = {
        row["sample_id"]: row["gold_functions"]
        for row in artifact["evaluation_sets"]["Aug-test"]["items"]
        if not row["is_variant"]
    }
    expected_counts = {"Aug-test": 633, "Aug-orig": 23, "Bench-ref": 2000}
    rebuilt = {}
    for split, rows in recovery["predictions"].items():
        if len(rows) != expected_counts[split]:
            raise AssertionError(f"{split}: unexpected row count")
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(gold[split]):
            raise AssertionError(f"{split}: IDs differ from public gold records")
        for row in rows:
            if row["gold_functions"] != gold[split][row["sample_id"]]:
                raise AssertionError(f"{split}/{row['sample_id']}: gold labels differ")
            if len(row["ranking"]) != 20 or len(set(row["ranking"])) != 20:
                raise AssertionError(f"{split}/{row['sample_id']}: malformed top-20")
        result = metrics(rows)
        archived_result = recovery["metrics"][
            {
                "Aug-test": "augmented_test",
                "Aug-orig": "augmented_test_original_only",
                "Bench-ref": "benchmark_reference_eval_only",
            }[split]
        ]
        if result["top_k"] != archived_result["top_k"]:
            raise AssertionError(f"{split}: reconstructed metrics differ")
        rebuilt[split] = {
            "predictions": [
                {"sample_id": row["sample_id"], "top20": row["ranking"]} for row in rows
            ],
            "metrics": result,
        }

    preserved = {
        "evaluation_sets": canonical_sha256(artifact["evaluation_sets"]),
        "historical_benchmark_top10": canonical_sha256(
            artifact["historical_benchmark_top10"]
        ),
        "s4b_before_after_by_split": canonical_sha256(
            artifact["s4b_before_after_by_split"]
        ),
        "existing_reconstructions": {
            label: canonical_sha256(data)
            for label, data in artifact["reconstructions"].items()
            if label != "Zero-shot SBERT"
        },
    }
    artifact["schema_version"] = 3
    artifact["status"] = (
        "mixed historical raw evidence and strictly validated exact or "
        "pinned-snapshot deterministic reconstructions"
    )
    artifact["coverage_note"] = (
        "Full top-20 curves are recovered for the three TF-IDF estimators, "
        "Zero-shot SBERT from its immutable model snapshot, and the S4(b) "
        "unadapted Hybrid arm on all three evaluation splits. The two "
        "cross-encoder estimators retain only historical role-filtered "
        "Bench-ref deployment top-10 evidence, which is not the unfiltered "
        "S4 curve ranking."
    )
    artifact["reconstructions"]["Zero-shot SBERT"] = {
        "origin": f"pinned-snapshot deterministic CPU reconstruction from {MODEL_ID}@{REVISION}",
        "coverage": "Aug-test, Aug-orig, and Bench-ref top-20",
        "splits": rebuilt,
        "integrity_checks": {
            split: {
                "all_frozen_k_metrics_match": True,
                "historical_raw_top10_prefix_matches": (
                    "not_applicable_role_filtered_export" if split == "Bench-ref" else None
                ),
            }
            for split in rebuilt
        },
        "all_seven_frozen_slice_checks": recovery["integrity_checks"],
    }
    artifact["unavailable_exact_item_coverage"].pop("Zero-shot SBERT", None)
    artifact["environment"]["sbert_recovery"] = {
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "sentence_transformers": importlib.metadata.version("sentence-transformers"),
        "device": "cpu",
        "batch_size_for_card_embeddings": 16,
        "query_scoring": "one query at a time",
    }
    artifact["provenance"]["sbert_model_snapshot"] = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "files": recovery["snapshot_files"],
    }
    artifact["provenance"]["historical_base_recovery_code"] = artifact[
        "provenance"
    ].get("historical_base_recovery_code", artifact["provenance"]["recovery_code"])
    artifact["provenance"]["recovery_code"] = descriptor(
        ROOT / "code/aggregate/export_s4_item_evidence.py"
    )
    artifact["provenance"]["sbert_recovery_code"] = descriptor(HERE)
    artifact["provenance"]["sbert_source_validation"] = {
        "historical_metrics": historical_metrics,
        "benchmark_blob": recovery["benchmark_blob"],
        "benchmark_sha256": recovery["benchmark_sha256"],
        "all_seven_frozen_slice_checks": recovery["integrity_checks"],
    }
    artifact["preserved_section_sha256_during_sbert_extension"] = preserved

    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": sha256_file(args.output),
        "sbert_rows": {split: len(data["predictions"]) for split, data in rebuilt.items()},
        "all_seven_frozen_slice_checks": recovery["integrity_checks"],
        "preserved_sections": preserved,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
