#!/usr/bin/env python3
"""Rebuild the frozen OpenDSS E2 knowledge profiles from per-probe records.

The release keeps both the per-probe L0--L3 evaluations and the historical
``knowledge_profile.json`` files used by the transfer pilot.  This script
validates release hashes, independently rebuilds every per-function layer
score, checks it against the frozen profile, and reconstructs the profile /
control-device rows reported in Supplementary Table S19.

Only Python's standard library and repository-local files are required.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ROOT = REPO_ROOT / "results/raw/e2_transfer/opendss/profiles"
PROFILE_MANIFEST = PROFILE_ROOT / "manifest.json"
TRANSFER_MANIFEST = REPO_ROOT / "results/raw/e2_transfer/manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "results/aggregates/e2_opendss_profiles_recomputed.json"

RAW_FILES = {
    "L0": "results_L0_recognition.json",
    "L1": "results_L1_recall.json",
    "L2": "results_L2_comprehension.json",
    "L3": "results_L3_application.json",
}
EXPECTED_COUNTS = {"L0": 152, "L1": 76, "L2": 199, "L3": 76}
TABLE_MODELS = {
    "Qwen/Qwen2.5-Coder-32B-Instruct": {
        "profile_3dp": [0.618, 0.737, 0.969, 0.408],
        "injected_tokens": {"C": 393, "Rsem": 3969},
        "non_executable": {"A": 15, "C": 17, "Rsem": 6},
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "profile_3dp": [0.447, 0.171, 0.978, 0.079],
        "injected_tokens": {"C": 526, "Rsem": 3854},
        "non_executable": {"A": 13, "C": 2, "Rsem": 1},
    },
}


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def mean4(values: list[float]) -> float:
    return round(mean(values), 4)


def validate_release_file(path: Path, metadata: list, label: str) -> None:
    if len(metadata) != 5:
        raise ValueError(f"{label}: malformed manifest tuple")
    release_bytes, release_hash = metadata[3], metadata[4]
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != release_bytes:
        raise ValueError(
            f"{label}: expected {release_bytes} bytes, got {path.stat().st_size}"
        )
    got_hash = sha256(path)
    if got_hash != release_hash:
        raise ValueError(
            f"{label}: expected SHA-256 {release_hash}, got {got_hash}"
        )


def validate_transfer_file(path: Path, metadata: dict, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != metadata["bytes"]:
        raise ValueError(f"{label}: byte-count mismatch")
    if sha256(path) != metadata["sha256"]:
        raise ValueError(f"{label}: SHA-256 mismatch")


def collect_raw(records_by_layer: dict[str, list[dict]]) -> dict:
    """Collect the raw dimensions used by the frozen profile builder."""
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for layer, records in records_by_layer.items():
        seen_ids = set()
        for record in records:
            probe_id = record["probe_id"]
            if probe_id in seen_ids:
                raise ValueError(f"{layer}: duplicate probe ID {probe_id}")
            seen_ids.add(probe_id)
            details = record["details"]
            scores = record["scores"]
            function = details["function_name"]
            if layer == "L0":
                raw[function][layer][details["question_type"]].append(
                    scores.get("accuracy", 0.0)
                )
            elif layer == "L1":
                for metric in (
                    "path_correct",
                    "required_params_jaccard",
                    "required_params_precision",
                    "required_params_recall",
                ):
                    raw[function][layer][metric].append(scores.get(metric, 0.0))
            elif layer == "L2":
                raw[function][layer][details["sub_type"]].append(
                    scores.get("accuracy", 0.0)
                )
            elif layer == "L3":
                for metric in (
                    "execution_success",
                    "api_validity_rate",
                    "target_function_used",
                ):
                    raw[function][layer][metric].append(scores.get(metric, 0.0))
        expected = EXPECTED_COUNTS[layer]
        if len(records) != expected:
            raise ValueError(f"{layer}: expected {expected} records, got {len(records)}")
    return raw


def rebuild_profile(records_by_layer: dict[str, list[dict]]) -> dict[str, dict]:
    raw = collect_raw(records_by_layer)
    profile = {}
    for function, layers in raw.items():
        if set(layers) != set(RAW_FILES):
            raise ValueError(f"{function}: incomplete layer set {sorted(layers)}")

        l0_groups = layers["L0"]
        l0_by_type = {
            name: {"score": mean4(values), "n_probes": len(values)}
            for name, values in l0_groups.items()
        }
        l0_all = [value for values in l0_groups.values() for value in values]

        l1 = layers["L1"]
        path = mean4(l1["path_correct"])
        jaccard = mean4(l1["required_params_jaccard"])
        precision = mean4(l1["required_params_precision"])
        recall = mean4(l1["required_params_recall"])

        l2_groups = layers["L2"]
        l2_by_type = {
            name: {"score": mean4(values), "n_probes": len(values)}
            for name, values in l2_groups.items()
        }
        l2_all = [value for values in l2_groups.values() for value in values]

        l3 = layers["L3"]
        execution = mean4(l3["execution_success"])
        api_validity = mean4(l3["api_validity_rate"])
        target_used = mean4(l3["target_function_used"])
        application = round(execution * target_used, 4)

        profile[function] = {
            "L0": {
                "score": mean4(l0_all),
                "n_probes": len(l0_all),
                "by_probe_type": l0_by_type,
            },
            "L1": {
                "score": round(0.5 * path + 0.5 * jaccard, 4),
                "n_probes": len(l1["path_correct"]),
                "diagnostics": {
                    "path_correct": path,
                    "required_params_jaccard": jaccard,
                    "required_params_precision": precision,
                    "required_params_recall": recall,
                },
            },
            "L2": {
                "score": mean4(l2_all),
                "n_probes": len(l2_all),
                "by_probe_type": l2_by_type,
            },
            "L3": {
                "score": application,
                "n_probes": len(l3["execution_success"]),
                "diagnostics": {
                    "execution_success": execution,
                    "api_validity_rate": api_validity,
                    "target_function_used": target_used,
                    "application_score": application,
                },
            },
        }
    if len(profile) != 76:
        raise ValueError(f"expected 76 functions, got {len(profile)}")
    return profile


def profile_means(profile: dict[str, dict]) -> dict[str, float]:
    return {
        layer: mean([entry[layer]["score"] for entry in profile.values()])
        for layer in RAW_FILES
    }


def model_profile_report(model: str, metadata: dict) -> dict:
    directory = PROFILE_ROOT / metadata["directory"]
    for name, file_metadata in metadata["files"].items():
        validate_release_file(directory / name, file_metadata, f"{model}/{name}")

    records = {
        layer: read_json(directory / filename)
        for layer, filename in RAW_FILES.items()
    }
    rebuilt = rebuild_profile(records)
    frozen = read_json(directory / "knowledge_profile.json")
    if rebuilt != frozen:
        mismatched = [name for name in rebuilt if rebuilt[name] != frozen.get(name)]
        raise ValueError(
            f"{model}: rebuilt profile differs for {len(mismatched)} functions; "
            f"first={mismatched[:3]}"
        )

    summary = read_json(directory / "summary.json")
    raw_metric_checks = {
        "L0_accuracy": mean(
            [record["scores"].get("accuracy", 0.0) for record in records["L0"]]
        ),
        "L1_overall": mean(
            [record["scores"].get("overall", 0.0) for record in records["L1"]]
        ),
        "L2_accuracy": mean(
            [record["scores"].get("accuracy", 0.0) for record in records["L2"]]
        ),
        "L3_execution_success": mean(
            [
                record["scores"].get("execution_success", 0.0)
                for record in records["L3"]
            ]
        ),
    }
    frozen_summary_metrics = {
        "L0_accuracy": summary["layers"]["L0_recognition"]["scores"]["accuracy"]["mean"],
        "L1_overall": summary["layers"]["L1_recall"]["scores"]["overall"]["mean"],
        "L2_accuracy": summary["layers"]["L2_comprehension"]["scores"]["accuracy"]["mean"],
        "L3_execution_success": summary["layers"]["L3_application"]["scores"]["execution_success"]["mean"],
    }
    for key, value in raw_metric_checks.items():
        if round(value, 4) != frozen_summary_metrics[key]:
            raise ValueError(f"{model}: summary mismatch for {key}")

    means = profile_means(rebuilt)
    return {
        "directory": metadata["directory"],
        "n_functions": len(rebuilt),
        "n_probe_records": {layer: len(records[layer]) for layer in RAW_FILES},
        "profile_means": means,
        "profile_means_3dp": {key: round(value, 3) for key, value in means.items()},
        "frozen_profile_exact_match": True,
        "frozen_summary_metrics_match": True,
    }


def control_device_report(profile_manifest: dict, transfer_manifest: dict) -> dict:
    cold = transfer_manifest["cold_start_cells"]
    pattern = cold["release_path_pattern"]
    reports = {}
    model_by_directory = {
        values["directory"]: model
        for model, values in profile_manifest["models"].items()
    }
    for model_dir in (
        "Qwen_Qwen2.5-Coder-32B-Instruct",
        "meta-llama_Llama-3.1-70B-Instruct",
    ):
        model = model_by_directory[model_dir]
        results = {}
        for condition in ("A", "C", "Rsem"):
            relative = pattern.format(
                backend="opendss", model=model_dir, condition=condition
            )
            path = REPO_ROOT / relative
            file_metadata = cold["files"]["opendss"][model_dir][condition]
            validate_transfer_file(path, file_metadata, relative)
            result = read_json(path)
            control = [item for item in result["item_results"] if item["task"] == "control_device"]
            if len(control) != 18:
                raise ValueError(f"{model}/{condition}: expected 18 control items")
            results[condition] = {
                "prompt_tokens_total": sum(item["prompt_tokens"] for item in result["item_results"]),
                "prompt_tokens_mean": mean(
                    [item["prompt_tokens"] for item in result["item_results"]]
                ),
                "control_items": len(control),
                "non_executable": sum(not item["executed"] for item in control),
                "attribute_error": sum(
                    item.get("error_type") == "AttributeError" for item in control
                ),
            }
        base_tokens = results["A"]["prompt_tokens_mean"]
        injected = {
            condition: results[condition]["prompt_tokens_mean"] - base_tokens
            for condition in ("C", "Rsem")
        }
        reports[model] = {
            "conditions": results,
            "injected_tokens_mean_exact": injected,
            "injected_tokens_mean_rounded": {
                key: round(value) for key, value in injected.items()
            },
        }
    return reports


def validate_manuscript_rows(profile_reports: dict, control_reports: dict) -> dict:
    checks = {}
    layer_order = ("L0", "L1", "L2", "L3")
    for model, expected in TABLE_MODELS.items():
        profile_values = [
            profile_reports[model]["profile_means_3dp"][layer]
            for layer in layer_order
        ]
        injected = control_reports[model]["injected_tokens_mean_rounded"]
        non_executable = {
            condition: values["non_executable"]
            for condition, values in control_reports[model]["conditions"].items()
        }
        checks[model] = {
            "profile_rows_match": profile_values == expected["profile_3dp"],
            "injected_token_rows_match": injected == expected["injected_tokens"],
            "non_executable_rows_match": non_executable == expected["non_executable"],
        }
        if not all(checks[model].values()):
            raise ValueError(f"{model}: Supplementary Table S19 mismatch")
    qwen = control_reports["Qwen/Qwen2.5-Coder-32B-Instruct"]["conditions"]
    checks["Qwen name-level caption counts"] = {
        "A_attribute_error": qwen["A"]["attribute_error"],
        "C_attribute_error": qwen["C"]["attribute_error"],
        "Rsem_attribute_error": qwen["Rsem"]["attribute_error"],
        "matches_caption": (
            qwen["A"]["attribute_error"] == 14
            and qwen["C"]["attribute_error"] == 17
            and qwen["Rsem"]["attribute_error"] == 6
        ),
    }
    if not checks["Qwen name-level caption counts"]["matches_caption"]:
        raise ValueError("Qwen AttributeError caption counts do not match")
    return checks


def aggregate(profile_manifest_path: Path, transfer_manifest_path: Path) -> dict:
    manifest = read_json(profile_manifest_path)
    profile_reports = {
        model: model_profile_report(model, metadata)
        for model, metadata in manifest["models"].items()
    }
    control_reports = control_device_report(
        manifest, read_json(transfer_manifest_path)
    )
    checks = validate_manuscript_rows(profile_reports, control_reports)
    return {
        "schema_version": "1.0",
        "source_profile_manifest": str(profile_manifest_path.relative_to(REPO_ROOT)),
        "source_transfer_manifest": str(transfer_manifest_path.relative_to(REPO_ROOT)),
        "profile_definition": {
            "L0": "mean per-function recognition accuracy",
            "L1": "mean per-function (0.5 * path_correct + 0.5 * required_params_jaccard)",
            "L2": "mean per-function comprehension accuracy",
            "L3": "mean per-function (execution_success * target_function_used)",
        },
        "profiles": profile_reports,
        "control_device_benchmark": control_reports,
        "supplementary_table_s19_checks": checks,
        "all_checks_passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-manifest", type=Path, default=PROFILE_MANIFEST)
    parser.add_argument("--transfer-manifest", type=Path, default=TRANSFER_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = aggregate(args.profile_manifest.resolve(), args.transfer_manifest.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
