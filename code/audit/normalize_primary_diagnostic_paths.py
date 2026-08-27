#!/usr/bin/env python3
"""Normalize private Python-environment prefixes in the primary compact.

This is a narrowly scoped, idempotent release sanitizer.  It edits only
``condition_a_diagnostics.error_msg_dictionary`` and the hash/normalization
metadata attached to that dictionary.  All outcome, token, task, injection,
and ``correct_task_fn`` fields are asserted unchanged.

The removed prefix is not written to the public artifact.  A maintainer can
recover the originals from each run's content-pinned source file, and the
pre-normalization diagnostic hash is retained to verify the recovery.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "code/aggregate"))

from aggregate_primary_results_compact import (  # noqa: E402
    DIAGNOSTIC_ENV_PLACEHOLDER,
    diagnostics_hash,
    normalize_diagnostic_message,
)


DEFAULT_ARCHIVE = ROOT / "results/raw/main_experiment/primary_outcomes_compact.json"
DEFAULT_REPORT = ROOT / "audit/primary_diagnostic_path_normalization.json"


def canonical(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def nondiagnostic_projection(data: dict) -> dict:
    """Projection whose equality proves that no non-text evidence changed."""
    projected = copy.deepcopy(data)
    for row in projected.get("runs", {}).values():
        diagnostic = row.get("condition_a_diagnostics")
        if not diagnostic:
            continue
        diagnostic["error_msg_dictionary"] = "<EXCLUDED FROM PROJECTION>"
        diagnostic["diagnostics_sha256"] = "<EXCLUDED FROM PROJECTION>"
        diagnostic.pop("path_normalization", None)
    return projected


def normalize_archive(data: dict) -> tuple[dict, dict]:
    before_projection = digest(nondiagnostic_projection(data))
    runs_changed = 0
    dictionary_entries_changed = 0
    item_occurrences_changed = 0
    prefix_substitutions = 0
    pre_hashes = {}
    post_hashes = {}

    for run_key, row in data.get("runs", {}).items():
        diagnostic = row.get("condition_a_diagnostics")
        if not diagnostic:
            continue
        ids = data["datasets"][row["dataset"]]["item_ids"]
        old_dictionary = diagnostic["error_msg_dictionary"]
        existing_normalization = diagnostic.get("path_normalization") or {}
        old_indices = diagnostic["error_msg_index"]
        old_messages = [old_dictionary[index] for index in old_indices]
        correct = diagnostic["correct_task_fn"]
        old_hash = diagnostics_hash(ids, old_messages, correct)
        if old_hash != diagnostic["diagnostics_sha256"]:
            raise ValueError(f"pre-normalization diagnostic hash mismatch: {run_key}")

        normalized_dictionary = []
        changed_in_dictionary = 0
        substitutions_in_dictionary = 0
        for message in old_dictionary:
            portable, count = normalize_diagnostic_message(message)
            normalized_dictionary.append(portable)
            changed_in_dictionary += portable != message
            substitutions_in_dictionary += count

        # Normalizing only a prefix should not merge distinct diagnostic
        # messages in this archive.  Keeping indices byte-identical is a useful
        # guard that the release sanitizer has not altered item attribution.
        if len(set(map(json.dumps, normalized_dictionary))) != len(normalized_dictionary):
            raise ValueError(f"normalization would merge dictionary entries: {run_key}")
        normalized_messages = [normalized_dictionary[index] for index in old_indices]
        changed_items = sum(a != b for a, b in zip(old_messages, normalized_messages))
        substitutions_in_items = sum(
            normalize_diagnostic_message(message)[1] for message in old_messages
        )

        if changed_in_dictionary:
            runs_changed += 1
            dictionary_entries_changed += changed_in_dictionary
            item_occurrences_changed += changed_items
            prefix_substitutions += substitutions_in_dictionary
        new_hash = diagnostics_hash(ids, normalized_messages, correct)
        diagnostic["error_msg_dictionary"] = normalized_dictionary
        diagnostic["diagnostics_sha256"] = new_hash
        if changed_in_dictionary:
            prior_dictionary = int(existing_normalization.get(
                "changed_dictionary_entries", 0
            ))
            prior_items = int(existing_normalization.get(
                "changed_item_occurrences", 0
            ))
            prior_substitutions = int(existing_normalization.get(
                "prefix_substitutions_in_item_messages",
                existing_normalization.get("prefix_substitutions", 0),
            ))
            # v1 already changed the absolute occurrence in a small set of
            # messages which also carried a ../../ traceback spelling.  Those
            # are the same dictionary/item entries, so extend the path count
            # without double-counting entry coverage.
            overlapping_dictionary = sum(
                portable != message
                and isinstance(message, str)
                and DIAGNOSTIC_ENV_PLACEHOLDER in message
                for message, portable in zip(old_dictionary, normalized_dictionary)
            )
            overlapping_items = sum(
                portable != message
                and isinstance(message, str)
                and DIAGNOSTIC_ENV_PLACEHOLDER in message
                for message, portable in zip(old_messages, normalized_messages)
            )
            diagnostic["path_normalization"] = {
                "scheme": "python-env-site-packages-prefix-v2",
                "placeholder": DIAGNOSTIC_ENV_PLACEHOLDER,
                "changed_dictionary_entries": (
                    prior_dictionary + changed_in_dictionary - overlapping_dictionary
                ),
                "changed_item_occurrences": (
                    prior_items + changed_items - overlapping_items
                ),
                "prefix_substitutions_in_item_messages": (
                    prior_substitutions + substitutions_in_items
                ),
                "pre_normalization_diagnostics_sha256": existing_normalization.get(
                    "pre_normalization_diagnostics_sha256", old_hash
                ),
            }
        elif existing_normalization:
            if (existing_normalization.get("scheme") not in {
                    "absolute-site-packages-prefix-v1",
                    "python-env-site-packages-prefix-v2",
                }
                    or existing_normalization.get("placeholder")
                    != DIAGNOSTIC_ENV_PLACEHOLDER):
                raise ValueError(f"unknown existing normalization: {run_key}")
            diagnostic["path_normalization"] = {
                **existing_normalization,
                "scheme": "python-env-site-packages-prefix-v2",
                "prefix_substitutions_in_item_messages": int(
                    existing_normalization.get(
                        "prefix_substitutions_in_item_messages",
                        existing_normalization.get("prefix_substitutions", 0),
                    )
                ),
            }
            diagnostic["path_normalization"].pop("prefix_substitutions", None)
        else:
            diagnostic["path_normalization"] = {
                "scheme": "python-env-site-packages-prefix-v2",
                "placeholder": DIAGNOSTIC_ENV_PLACEHOLDER,
                "changed_dictionary_entries": 0,
                "changed_item_occurrences": 0,
                "prefix_substitutions_in_item_messages": 0,
                "pre_normalization_diagnostics_sha256": old_hash,
            }
        pre_hashes[run_key] = diagnostic["path_normalization"][
            "pre_normalization_diagnostics_sha256"
        ]
        post_hashes[run_key] = new_hash

    after_projection = digest(nondiagnostic_projection(data))
    if before_projection != after_projection:
        raise AssertionError("a non-diagnostic field changed during normalization")

    remaining = []
    for run_key, row in data.get("runs", {}).items():
        diagnostic = row.get("condition_a_diagnostics")
        if not diagnostic:
            continue
        for index, message in enumerate(diagnostic["error_msg_dictionary"]):
            if normalize_diagnostic_message(message)[1]:
                remaining.append([run_key, index])
    if remaining:
        raise AssertionError(f"absolute environment paths remain: {remaining[:3]}")

    report = {
        "schema_version": 1,
        "target": "results/raw/main_experiment/primary_outcomes_compact.json",
        "scope": (
            "Only condition_a_diagnostics.error_msg_dictionary, its integrity "
            "hash, and attached normalization metadata."
        ),
        "scheme": {
            "id": "python-env-site-packages-prefix-v2",
            "replacement": DIAGNOSTIC_ENV_PLACEHOLDER,
            "preserved": "the complete package-relative path after site-packages/",
            "recovery": (
                "Original messages are recoverable by maintainers from each "
                "content-pinned source run; pre-normalization hashes verify the "
                "recovery. The removed private prefix is not published."
            ),
        },
        "counts_this_run": {
            "runs_changed": runs_changed,
            "dictionary_entries_changed": dictionary_entries_changed,
            "item_occurrences_changed": item_occurrences_changed,
            "dictionary_prefix_substitutions": prefix_substitutions,
        },
        "archive_recorded_totals": {
            "runs_with_normalization": sum(
                bool((row.get("condition_a_diagnostics") or {})
                     .get("path_normalization", {})
                     .get("changed_dictionary_entries"))
                for row in data.get("runs", {}).values()
            ),
            "dictionary_entries_changed": sum(
                int((row.get("condition_a_diagnostics") or {})
                    .get("path_normalization", {})
                    .get("changed_dictionary_entries", 0))
                for row in data.get("runs", {}).values()
            ),
            "item_occurrences_changed": sum(
                int((row.get("condition_a_diagnostics") or {})
                    .get("path_normalization", {})
                    .get("changed_item_occurrences", 0))
                for row in data.get("runs", {}).values()
            ),
        },
        "invariance": {
            "nondiagnostic_projection_sha256_before": before_projection,
            "nondiagnostic_projection_sha256_after": after_projection,
            "all_nontext_evidence_exact": before_projection == after_projection,
            "error_msg_index_unchanged": True,
            "correct_task_fn_unchanged": True,
            "remaining_absolute_site_packages_prefixes": len(remaining),
        },
        "pre_normalization_diagnostics_sha256_by_run": pre_hashes,
        "normalized_diagnostics_sha256_by_run": post_hashes,
    }
    return data, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    archive = args.archive.resolve()
    report_path = args.report.resolve()
    data = json.loads(archive.read_text(encoding="utf-8"))
    data, report = normalize_archive(data)
    archive.write_text(
        json.dumps(data, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    totals = report["archive_recorded_totals"]
    print(
        "diagnostic normalization verified: "
        f"{totals['dictionary_entries_changed']} dictionary entries / "
        f"{totals['item_occurrences_changed']} item occurrences; "
        "all non-text evidence exact"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
