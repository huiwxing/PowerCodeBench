#!/usr/bin/env python3
"""Rebuild the manuscript knowledge profiles from the raw probe records.

The release records are byte-exact copies of the frozen inference outputs,
apart from a normalization of workstation prefixes in four diagnostic fields
(``audit/README.md``).  A record contains the model's ``details.raw_response``
together with the frozen per-probe evaluator fields.  This script replays the
historical ``build_knowledge_profile`` aggregation over those evaluator fields
and compares every resulting profile field with ``probe_profiles.json``.  The
aggregation is the only step it runs: the archived inference and L3 results
are used as they stand.

Maintainers with the frozen source checkout can recreate the raw archive and
its provenance manifest with ``--export --source-root <IGPT_ROOT>``.  Release
users need only the default command; it is CPU-only and uses the Python
standard library.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
DEFAULT_SOURCE = ROOT.parent / "igpt"
DEFAULT_ARCHIVE = ROOT / "results/raw/probe_transcripts"
DEFAULT_PROFILES = ROOT / "results/supplementary_evidence/probe_profiles.json"
DEFAULT_OUTPUT = ROOT / "results/aggregates/probe_transcripts_recomputed.json"
DEFAULT_AUDIT = ROOT / "audit/probe_diagnostic_path_normalization.json"

LOCAL_MODELS = {
    "Qwen2.5-1.5B": "Qwen_Qwen2.5-Coder-1.5B-Instruct",
    "Llama-8B": "meta-llama_Llama-3.1-8B-Instruct",
    "Qwen2.5-7B": "Qwen_Qwen2.5-Coder-7B-Instruct",
    "Qwen2.5-14B": "Qwen_Qwen2.5-Coder-14B-Instruct",
    "Qwen2.5-32B": "Qwen_Qwen2.5-Coder-32B-Instruct",
    "Qwen3-Next": "Qwen_Qwen3-Coder-Next",
    "Llama-70B": "meta-llama_Llama-3.1-70B-Instruct",
    "GPT-OSS-120B": "openai_gpt-oss-120b",
    "Llama-405B": "meta-llama_Llama-3.1-405B-Instruct",
    "Qwen3-480B": "Qwen_Qwen3-Coder-480B-A35B-Instruct",
}
API_MODELS = {
    "Gemini-2.5-Flash": "gemini-2.5-flash",
    "DeepSeek-V4-Flash": "deepseek-v4-flash",
    "Claude-Haiku-4-5": "claude-haiku-4-5",
    "GPT-5.4-mini": "gpt-5.4-mini",
}
LOCAL_FILES = {
    "L0": "results_L0_recognition.json",
    "L1": "results_L1_recall.json",
    "L2": "results_L2_comprehension.json",
    "L3": "results_L3_application.json",
}
EXPECTED_LAYER_COUNTS = {"L0": 550, "L1": 275, "L2": 980, "L3": 275}

# Strong credential shapes only.  Words such as ``password`` and ``token`` are
# legitimate pandapower API parameter names and occur in generated examples.
CREDENTIAL_PATTERNS = {
    "openai": re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "google": re.compile(rb"AIza[A-Za-z0-9_-]{30,}"),
    "aws_access_key": re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "github": re.compile(rb"gh[oprsu]_[A-Za-z0-9]{30,}"),
    "huggingface": re.compile(rb"hf_[A-Za-z0-9]{24,}"),
    "private_key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
# Generic release-hygiene patterns deliberately avoid preserving the removed
# maintainer identity in public code or metadata.
ABSOLUTE_SITE_PACKAGES = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:(?:\.\./)+|/)(?:[^/\s()'\"<>]+/)+site-packages"
)
ABSOLUTE_PYTHON_RUNTIME = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:[^/\s()'\"<>]+/)+"
    r"(?:miniforge3|miniconda3|anaconda3)(?=/lib/python)"
)
ABSOLUTE_SOURCE_REPO = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:[^/\s()'\"<>]+/)+myprojects/igpt"
)
RUNTIME_TMP_ROOT = "/".join(("", "local", "user"))
ABSOLUTE_RUNTIME_TMP = re.compile(
    rf"(?<![A-Za-z0-9_.-]){re.escape(RUNTIME_TMP_ROOT)}/[0-9]+"
)
PRIVATE_PATH_PATTERN = re.compile(
    rb"/(?:home/[^/\s\"']+|projects/[^/\s\"']+|local/user/[0-9]+)"
)
DIAGNOSTIC_PATHS = (
    ("details", "exec_result", "traceback"),
    ("details", "exec_result", "output"),
    ("details", "error_msg"),
    ("details", "exec_result", "error"),
)


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        stream.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip() or None


def git_blob(root: Path, path: Path) -> str | None:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    row = git_output(root, "ls-files", "-s", "--", relative)
    return row.split()[1] if row else None


def scan_release_bytes(payload: bytes) -> tuple[dict[str, int], int]:
    credentials = {
        name: len(pattern.findall(payload))
        for name, pattern in CREDENTIAL_PATTERNS.items()
    }
    return credentials, len(PRIVATE_PATH_PATTERN.findall(payload))


def raw_response_count(records: list[dict]) -> int:
    return sum(
        isinstance(row.get("details"), dict)
        and isinstance(row["details"].get("raw_response"), str)
        for row in records
    )


def normalize_diagnostic_text(value: str) -> tuple[str, int]:
    """Replace only machine-specific prefixes; preserve diagnostic suffixes."""
    value, env_count = ABSOLUTE_SITE_PACKAGES.subn(
        "<PYTHON_ENV>/site-packages", value)
    value, runtime_count = ABSOLUTE_PYTHON_RUNTIME.subn("<PYTHON_ENV>", value)
    value, source_count = ABSOLUTE_SOURCE_REPO.subn("<SOURCE_REPO>", value)
    value, temp_count = ABSOLUTE_RUNTIME_TMP.subn("<PYTHON_ENV>", value)
    return value, env_count + runtime_count + source_count + temp_count


def path_value(record: dict, path: tuple[str, ...]):
    value = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None, False
        value = value[key]
    return value, True


def set_path_value(record: dict, path: tuple[str, ...], value) -> None:
    parent = record
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value


def diagnostic_projection(records: list[dict]) -> list[dict]:
    projected = copy.deepcopy(records)
    for record in projected:
        for path in DIAGNOSTIC_PATHS:
            _, present = path_value(record, path)
            if present:
                set_path_value(record, path, "<DIAGNOSTIC_FIELD>")
    return projected


def named_field_projection(value, name: str, path=()):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + (key,)
            if key == name:
                found.append((child_path, child))
            found.extend(named_field_projection(child, name, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(named_field_projection(child, name, path + (index,)))
    return found


def nondiagnostic_leaf_count(value, path=()) -> int:
    if any(path == allowed for allowed in DIAGNOSTIC_PATHS):
        return 0
    if isinstance(value, dict):
        return sum(nondiagnostic_leaf_count(child, path + (key,))
                   for key, child in value.items())
    if isinstance(value, list):
        return sum(nondiagnostic_leaf_count(child, path + (index,))
                   for index, child in enumerate(value))
    return 1


def normalize_records(records: list[dict]) -> tuple[list[dict], dict]:
    normalized = copy.deepcopy(records)
    changed_entries = 0
    changed_occurrences = 0
    changed_records = 0
    field_counts = defaultdict(lambda: {"entries": 0, "occurrences": 0})
    for source_record, release_record in zip(records, normalized):
        record_changed = False
        for path in DIAGNOSTIC_PATHS:
            value, present = path_value(source_record, path)
            if not present or not isinstance(value, str):
                continue
            portable, occurrences = normalize_diagnostic_text(value)
            if portable != value:
                set_path_value(release_record, path, portable)
                changed_entries += 1
                changed_occurrences += occurrences
                record_changed = True
                label = ".".join(path)
                field_counts[label]["entries"] += 1
                field_counts[label]["occurrences"] += occurrences
        changed_records += int(record_changed)

    # A single projection equality proves every field outside the four
    # explicitly allowed diagnostic leaves is unchanged.
    nondiagnostic_exact = (
        diagnostic_projection(records) == diagnostic_projection(normalized))
    raw_exact = sum(
        source.get("details", {}).get("raw_response")
        == release.get("details", {}).get("raw_response")
        for source, release in zip(records, normalized)
    )
    scores_exact = sum(
        source.get("scores") == release.get("scores")
        for source, release in zip(records, normalized)
    )
    source_correct = named_field_projection(records, "correct")
    release_correct = named_field_projection(normalized, "correct")
    if not nondiagnostic_exact or raw_exact != len(records) or scores_exact != len(records):
        raise ValueError("diagnostic path normalization changed analytical evidence")
    if source_correct != release_correct:
        raise ValueError("diagnostic path normalization changed a correct field")
    return normalized, {
        "changed_entries": changed_entries,
        "changed_occurrences": changed_occurrences,
        "changed_records": changed_records,
        "by_field": dict(sorted(field_counts.items())),
        "nondiagnostic_fields_exact": nondiagnostic_exact,
        "nondiagnostic_scalar_leaves_compared": sum(
            nondiagnostic_leaf_count(record) for record in records),
        "raw_response_records_exact": raw_exact,
        "score_records_exact": scores_exact,
        "correct_fields_compared_exact": len(source_correct),
    }


def normalize_payload(payload: bytes) -> tuple[bytes, int]:
    text = payload.decode("utf-8")
    normalized, occurrences = normalize_diagnostic_text(text)
    return normalized.encode("utf-8"), occurrences


def export(source_root: Path, archive: Path, audit_path: Path) -> None:
    source_root = source_root.resolve()
    archive = archive.resolve()
    entries = []
    credential_totals = {name: 0 for name in CREDENTIAL_PATTERNS}
    source_private_occurrences = 0
    source_private_files = 0
    released_private_occurrences = 0
    released_private_files = 0
    audit_files = []

    def copy_one(panel: str, label: str, source_model: str, layer: str,
                 source: Path, released: Path) -> None:
        nonlocal source_private_occurrences, source_private_files
        nonlocal released_private_occurrences, released_private_files
        records = read_json(source)
        if not isinstance(records, list):
            raise ValueError(f"{source}: expected a JSON list")
        payload = source.read_bytes()
        credential_hits, source_path_hits = scan_release_bytes(payload)
        for key, value in credential_hits.items():
            credential_totals[key] += value
        source_private_occurrences += source_path_hits
        source_private_files += int(source_path_hits > 0)
        normalized_records, normalization = normalize_records(records)
        released_payload, payload_occurrences = normalize_payload(payload)
        released_records = json.loads(released_payload)
        if released_records != normalized_records:
            raise RuntimeError(f"byte normalization escaped allowed fields: {source}")
        if payload_occurrences != normalization["changed_occurrences"]:
            raise RuntimeError(f"path occurrence count mismatch: {source}")
        _, release_path_hits = scan_release_bytes(released_payload)
        released_private_occurrences += release_path_hits
        released_private_files += int(release_path_hits > 0)
        released.parent.mkdir(parents=True, exist_ok=True)
        released.write_bytes(released_payload)
        idempotent_records, idempotent = normalize_records(released_records)
        if idempotent_records != released_records:
            raise RuntimeError(f"normalization is not idempotent: {released}")
        if idempotent["changed_entries"] or idempotent["changed_occurrences"]:
            raise RuntimeError(f"normalization still finds private paths: {released}")
        layer_counts = {}
        for row in records:
            row_layer = row.get("layer")
            layer_counts[row_layer] = layer_counts.get(row_layer, 0) + 1
        entry = {
            "panel": panel,
            "model": label,
            "source_model": source_model,
            "layer": layer,
            "count": len(records),
            "per_layer_counts": dict(sorted(layer_counts.items())),
            "raw_response_count": raw_response_count(released_records),
            "source_path": source.relative_to(source_root).as_posix(),
            "released_path": released.relative_to(ROOT).as_posix(),
            "source_bytes": len(payload),
            "release_bytes": len(released_payload),
            "source_sha256": hashlib.sha256(payload).hexdigest(),
            "release_sha256": hashlib.sha256(released_payload).hexdigest(),
            "source_git_blob": git_blob(source_root, source),
            "normalization": {
                "changed_entries": normalization["changed_entries"],
                "changed_occurrences": normalization["changed_occurrences"],
                "changed_records": normalization["changed_records"],
                "by_field": normalization["by_field"],
                "source_private_path_occurrences": source_path_hits,
                "release_private_path_occurrences": release_path_hits,
            },
        }
        entries.append(entry)
        audit_files.append({
            "source_path": entry["source_path"],
            "released_path": entry["released_path"],
            "source_sha256": entry["source_sha256"],
            "release_sha256": entry["release_sha256"],
            "source_git_blob": entry["source_git_blob"],
            **normalization,
            "release_normalization_idempotent": True,
            "release_private_path_occurrences": release_path_hits,
        })

    for label, source_model in LOCAL_MODELS.items():
        for layer, filename in LOCAL_FILES.items():
            source = (source_root / "probe_eval_results/comparison" /
                      source_model / filename)
            released = archive / "open_weight" / label / filename
            copy_one("open_weight", label, source_model, layer, source, released)
    for label, source_model in API_MODELS.items():
        source = (source_root / "probe_eval_results/api_comparison_2000" /
                  source_model / "probe_results.json")
        released = archive / "api" / label / "probe_results.json"
        copy_one("api", label, source_model, "all", source, released)

    if any(credential_totals.values()):
        raise RuntimeError(
            "credential-shaped value found; archive not safe to release: "
            f"{credential_totals}"
        )
    if released_private_occurrences:
        raise RuntimeError(
            f"private paths remain after normalization: {released_private_occurrences}")
    audit = {
        "schema_version": 1,
        "description": (
            "Audit of release-only workstation-path normalization in raw probe "
            "diagnostic fields"
        ),
        "source_repository": "frozen experimental pipeline",
        "source_commit": git_output(source_root, "rev-parse", "HEAD"),
        "allowed_diagnostic_fields": [".".join(path) for path in DIAGNOSTIC_PATHS],
        "replacement_classes": {
            "source_checkout": "<SOURCE_REPO>",
            "python_environment_and_runtime_temp": "<PYTHON_ENV>",
        },
        "totals": {
            "files": len(entries),
            "files_changed": sum(
                row["normalization"]["changed_entries"] > 0 for row in entries),
            "records": sum(row["count"] for row in entries),
            "raw_responses": sum(row["raw_response_count"] for row in entries),
            "changed_entries": sum(
                row["normalization"]["changed_entries"] for row in entries),
            "changed_occurrences": sum(
                row["normalization"]["changed_occurrences"] for row in entries),
            "changed_records": sum(
                row["normalization"]["changed_records"] for row in entries),
            "source_private_path_occurrences": source_private_occurrences,
            "release_private_path_occurrences": released_private_occurrences,
            "raw_response_records_exact": sum(
                row["raw_response_records_exact"] for row in audit_files),
            "score_records_exact": sum(
                row["score_records_exact"] for row in audit_files),
            "correct_fields_compared_exact": sum(
                row["correct_fields_compared_exact"] for row in audit_files),
            "nondiagnostic_scalar_leaves_compared": sum(
                row["nondiagnostic_scalar_leaves_compared"] for row in audit_files),
        },
        "invariance": {
            "all_changes_confined_to_allowed_diagnostic_fields": all(
                row["nondiagnostic_fields_exact"] for row in audit_files),
            "all_raw_responses_exact": all(
                row["raw_response_records_exact"]
                == next(entry["count"] for entry in entries
                        if entry["released_path"] == row["released_path"])
                for row in audit_files),
            "all_scores_exact": all(
                row["score_records_exact"]
                == next(entry["count"] for entry in entries
                        if entry["released_path"] == row["released_path"])
                for row in audit_files),
            "all_correct_fields_exact": True,
            "release_normalization_idempotent": all(
                row["release_normalization_idempotent"] for row in audit_files),
        },
        "files": sorted(audit_files, key=lambda row: row["released_path"]),
    }
    write_json(audit_path, audit)
    manifest = {
        "schema_version": 2,
        "evidence_unit": "one model response and its frozen per-probe evaluator record",
        "source_repository": "frozen experimental pipeline",
        "source_commit": git_output(source_root, "rev-parse", "HEAD"),
        "normalization_audit": {
            "path": audit_path.relative_to(ROOT).as_posix(),
            "bytes": audit_path.stat().st_size,
            "sha256": sha256_file(audit_path),
        },
        "release_policy": {
            "copy_mode": "byte-exact except documented diagnostic path normalization",
            "credential_shape_scan": credential_totals,
            "credential_shape_matches": 0,
            "source_private_absolute_path_files": source_private_files,
            "source_private_absolute_path_occurrences": source_private_occurrences,
            "release_private_absolute_path_files": released_private_files,
            "release_private_absolute_path_occurrences": released_private_occurrences,
            "changed_diagnostic_entries": audit["totals"]["changed_entries"],
            "changed_path_occurrences": audit["totals"]["changed_occurrences"],
            "changed_records": audit["totals"]["changed_records"],
            "private_path_note": (
                "Historical workstation prefixes were replaced only in the four "
                "allowed traceback/output/error fields. Raw responses, scores, "
                "correct labels, and every other field remain exact."
            ),
            "context_term_note": (
                "Generated responses mention password/token as pandapower API parameter "
                "names or placeholders; the strong credential-shape scan found none."
            ),
        },
        "coverage": {
            "models": len(LOCAL_MODELS) + len(API_MODELS),
            "open_weight_models": len(LOCAL_MODELS),
            "api_models": len(API_MODELS),
            "files": len(entries),
            "records": sum(row["count"] for row in entries),
            "raw_responses": sum(row["raw_response_count"] for row in entries),
        },
        "files": sorted(entries, key=lambda row: row["released_path"]),
    }
    write_json(archive / "manifest.json", manifest)
    print(
        f"exported {manifest['coverage']['files']} raw files, "
        f"{manifest['coverage']['records']} records; normalized "
        f"{audit['totals']['changed_occurrences']} diagnostic paths -> {archive}"
    )


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def build_knowledge_profile(all_results: dict[str, list[dict]]) -> dict:
    """Historical ``probe_framework.build_knowledge_profile`` logic."""
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for layer_key, results in all_results.items():
        for row in results:
            details = row.get("details", {})
            scores = row.get("scores", {})
            function = details.get("function_name")
            if not function:
                continue
            if "L0" in layer_key:
                raw[function]["L0"][details.get("question_type", "unknown")].append(
                    scores.get("accuracy", 0.0))
            elif "L1" in layer_key:
                for metric in (
                    "path_correct", "required_params_jaccard",
                    "required_params_precision", "required_params_recall",
                ):
                    raw[function]["L1"][metric].append(scores.get(metric, 0.0))
            elif "L2" in layer_key:
                raw[function]["L2"][details.get("sub_type", "unknown")].append(
                    scores.get("accuracy", 0.0))
            elif "L3" in layer_key:
                for metric in (
                    "execution_success", "api_validity_rate", "target_function_used",
                ):
                    raw[function]["L3"][metric].append(scores.get(metric, 0.0))

    profile = {}
    for function, layers in raw.items():
        profile[function] = {}
        if "L0" in layers:
            groups = layers["L0"]
            by_type = {
                probe_type: {"score": mean(values), "n_probes": len(values)}
                for probe_type, values in groups.items()
            }
            values = [value for group in groups.values() for value in group]
            profile[function]["L0"] = {
                "score": mean(values), "n_probes": len(values),
                "by_probe_type": by_type,
            }
        else:
            profile[function]["L0"] = None

        if "L1" in layers:
            values = layers["L1"]
            path = mean(values.get("path_correct", []))
            jaccard = mean(values.get("required_params_jaccard", []))
            precision = mean(values.get("required_params_precision", []))
            recall = mean(values.get("required_params_recall", []))
            profile[function]["L1"] = {
                "score": round(0.5 * path + 0.5 * jaccard, 4),
                "n_probes": len(values.get("path_correct", [])),
                "diagnostics": {
                    "path_correct": path,
                    "required_params_jaccard": jaccard,
                    "required_params_precision": precision,
                    "required_params_recall": recall,
                },
            }
        else:
            profile[function]["L1"] = None

        if "L2" in layers:
            groups = layers["L2"]
            by_type = {
                probe_type: {"score": mean(values), "n_probes": len(values)}
                for probe_type, values in groups.items()
            }
            values = [value for group in groups.values() for value in group]
            profile[function]["L2"] = {
                "score": mean(values), "n_probes": len(values),
                "by_probe_type": by_type,
            }
        else:
            profile[function]["L2"] = None

        if "L3" in layers:
            values = layers["L3"]
            execution = mean(values.get("execution_success", []))
            validity = mean(values.get("api_validity_rate", []))
            target = mean(values.get("target_function_used", []))
            application = round(execution * target, 4)
            profile[function]["L3"] = {
                "score": application,
                "n_probes": len(values.get("execution_success", [])),
                "diagnostics": {
                    "execution_success": execution,
                    "api_validity_rate": validity,
                    "target_function_used": target,
                    "application_score": application,
                },
            }
        else:
            profile[function]["L3"] = None
    return profile


def leaf_count(value) -> int:
    if isinstance(value, dict):
        return sum(leaf_count(child) for child in value.values())
    if isinstance(value, list):
        return sum(leaf_count(child) for child in value)
    return 1


def aggregate(archive: Path, profiles_path: Path, output: Path) -> None:
    manifest = read_json(archive / "manifest.json")
    expected = read_json(profiles_path)
    audit_ref = manifest["normalization_audit"]
    audit_path = ROOT / audit_ref["path"]
    if (audit_path.stat().st_size != audit_ref["bytes"]
            or sha256_file(audit_path) != audit_ref["sha256"]):
        raise ValueError("probe path-normalization audit integrity mismatch")
    audit = read_json(audit_path)
    entries_by_model = defaultdict(list)
    total_records = 0
    total_raw = 0
    idempotent_files = 0
    released_private_paths = 0
    for entry in manifest["files"]:
        path = ROOT / entry["released_path"]
        if path.stat().st_size != entry["release_bytes"]:
            raise ValueError(f"byte-count mismatch: {path}")
        if sha256_file(path) != entry["release_sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        records = read_json(path)
        if len(records) != entry["count"]:
            raise ValueError(f"record-count mismatch: {path}")
        count_raw = raw_response_count(records)
        if count_raw != entry["raw_response_count"]:
            raise ValueError(f"raw-response-count mismatch: {path}")
        renormalized, idempotent = normalize_records(records)
        if renormalized != records:
            raise ValueError(f"release path normalization is not idempotent: {path}")
        if idempotent["changed_entries"] or idempotent["changed_occurrences"]:
            raise ValueError(f"normalized diagnostic path remains: {path}")
        _, path_hits = scan_release_bytes(path.read_bytes())
        released_private_paths += path_hits
        idempotent_files += 1
        entries_by_model[entry["model"]].append((entry, records))
        total_records += len(records)
        total_raw += count_raw

    if set(entries_by_model) != set(expected["models"]):
        raise ValueError("raw model set does not equal probe_profiles.json model set")

    model_reports = {}
    all_profile_leaves = 0
    for label in sorted(entries_by_model):
        rows = entries_by_model[label]
        panel = rows[0][0]["panel"]
        grouped = {}
        per_layer = defaultdict(int)
        if panel == "open_weight":
            if len(rows) != 4:
                raise ValueError(f"{label}: expected four layer files")
            for entry, records in rows:
                layer = entry["layer"]
                grouped[f"{layer}_release"] = records
                per_layer[layer] += len(records)
        elif panel == "api":
            if len(rows) != 1:
                raise ValueError(f"{label}: expected one API transcript file")
            for record in rows[0][1]:
                layer = record.get("layer")
                per_layer[layer] += 1
                grouped.setdefault(
                    f"{layer}_{record.get('probe_type', 'unknown')}", []).append(record)
        else:
            raise ValueError(f"{label}: unknown panel {panel}")
        if dict(per_layer) != EXPECTED_LAYER_COUNTS:
            raise ValueError(f"{label}: unexpected layer counts {dict(per_layer)}")
        rebuilt = build_knowledge_profile(grouped)
        frozen = expected["models"][label]["profile"]
        if rebuilt != frozen:
            raise ValueError(f"{label}: reconstructed profile differs from public profile")
        leaves = leaf_count(rebuilt)
        all_profile_leaves += leaves
        model_reports[label] = {
            "panel": panel,
            "source_model": expected["models"][label]["source_model"],
            "files": len(rows),
            "records": sum(len(records) for _, records in rows),
            "raw_responses": sum(raw_response_count(records) for _, records in rows),
            "per_layer_counts": dict(sorted(per_layer.items())),
            "functions": len(rebuilt),
            "profile_scalar_leaves": leaves,
            "profile_matches_public_field_for_field": True,
        }

    report = {
        "schema_version": 1,
        "status": "PASS",
        "source_manifest": (archive / "manifest.json").relative_to(ROOT).as_posix(),
        "target_profiles": profiles_path.relative_to(ROOT).as_posix(),
        "method": (
            "historical build_knowledge_profile aggregation over the raw transcript "
            "records' frozen per-probe evaluator fields"
        ),
        "scope_note": (
            "Raw model responses are archived for independent inspection; this CPU "
            "rebuild does not rerun inference or re-execute L3 generated code. "
            "Only documented diagnostic path prefixes differ from the frozen source."
        ),
        "coverage": {
            "models": len(model_reports),
            "open_weight_models": sum(
                row["panel"] == "open_weight" for row in model_reports.values()),
            "api_models": sum(row["panel"] == "api" for row in model_reports.values()),
            "files": len(manifest["files"]),
            "records": total_records,
            "raw_responses": total_raw,
            "functions_per_model": 275,
            "profile_scalar_leaves_compared": all_profile_leaves,
            "profile_mismatches": 0,
        },
        "diagnostic_path_normalization": {
            "audit_path": audit_ref["path"],
            "copy_mode": manifest["release_policy"]["copy_mode"],
            "files_checked_idempotent": idempotent_files,
            "release_private_path_occurrences": released_private_paths,
            "source_to_release": audit["totals"],
            "invariance": audit["invariance"],
        },
        "models": model_reports,
    }
    if report["coverage"]["records"] != manifest["coverage"]["records"]:
        raise ValueError("manifest total-record count differs from raw archive")
    if report["coverage"]["raw_responses"] != manifest["coverage"]["raw_responses"]:
        raise ValueError("manifest raw-response count differs from raw archive")
    write_json(output, report)
    print(
        f"PASS: {len(model_reports)} models, {total_raw} raw responses, "
        f"{all_profile_leaves} profile fields -> {output}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    if args.export:
        export(args.source_root, args.archive, args.audit.resolve())
    aggregate(args.archive.resolve(), args.profiles.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
