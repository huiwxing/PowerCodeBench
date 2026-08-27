#!/usr/bin/env python3
"""Validate the archived matcher-FN diagnostic and de-duplicate its census.

The script deliberately separates three operations:

1. validate the provenance and scope of the 100 complete source records;
2. recompute arithmetic that is identifiable from the frozen historical
   aggregate, including its nominal unweighted Wilson interval;
3. recompute a unique `(condition, model)` census from the public proactive
   and reactive aggregates.

The historical archive has no item-level adjudication ledger, so the script
reports the 8/5/87 figures as what they are, the historical aggregate.  Full
scope in ``audit/matcher_fn/README.md``.
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import NormalDist


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "audit" / "matcher_fn"
HISTORICAL_AUDIT = AUDIT_DIR / "P01_false_negative_audit.json"
SAMPLE_RECORDS = AUDIT_DIR / "sample_records.json"
PROACTIVE = (ROOT / "results" / "aggregates" / "api_comparison_2000"
             / "proactive_comparison.json")
REACTIVE = (ROOT / "results" / "aggregates" / "api_comparison_2000"
            / "reactive_comparison.json")

SAMPLED_CONDITIONS = (
    "A", "A_FD", "A_FDR", "A_FDRS",
    "C", "C_FD", "C_FDR", "C_FDRS",
)

PUBLIC_VENV_TOKEN = "<EXPERIMENT_VENV>"
PRIVATE_VENV_PATTERN = re.compile(
    r"/projects/[^/\s)]+/[^/\s)]+/igpt_venv"
)
PRIVATE_VENV_RULE = (
    "replace an absolute /projects/<account>/<user>/igpt_venv prefix "
    "with <EXPERIMENT_VENV>"
)

EXPECTED_CONDITIONS = {
    # condition: cells, total, executed, matched, unmatched, parse failures
    "A": (15, 30000, 4090, 2152, 1938, 169),
    "B": (11, 22000, 2852, 1794, 1058, 73),
    "C": (15, 30000, 17503, 10711, 6792, 1583),
    "X": (11, 22000, 11909, 7637, 4272, 1075),
    "A_FD": (11, 22000, 5118, 2462, 2656, 489),
    "A_FDR": (11, 22000, 5762, 2856, 2906, 576),
    "A_FDRS": (11, 22000, 5721, 3084, 2637, 196),
    "A_FX": (11, 22000, 4653, 2334, 2319, 496),
    "C_FD": (11, 22000, 14170, 9424, 4746, 1501),
    "C_FDR": (15, 30000, 20571, 13789, 6782, 2104),
    "C_FDRS": (15, 30000, 21510, 14521, 6989, 1206),
    "C_FX": (15, 30000, 19979, 13424, 6555, 2106),
    "X_FD": (11, 22000, 13571, 9228, 4343, 1283),
    "X_FDR": (11, 22000, 13675, 9386, 4289, 1382),
    "X_FDRS": (11, 22000, 14167, 9662, 4505, 940),
    "X_FX": (11, 22000, 13046, 9023, 4023, 1372),
}


def load(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record_sha256(record):
    payload = json.dumps(
        record, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def replace_in_json(value, old, new):
    """Recursively replace a literal string and return (value, count)."""
    if isinstance(value, str):
        return value.replace(old, new), value.count(old)
    if isinstance(value, list):
        out = []
        count = 0
        for item in value:
            replaced, n = replace_in_json(item, old, new)
            out.append(replaced)
            count += n
        return out, count
    if isinstance(value, dict):
        out = {}
        count = 0
        for key, item in value.items():
            replaced, n = replace_in_json(item, old, new)
            out[key] = replaced
            count += n
        return out, count
    return value, 0


def normalize_private_venv(value):
    """Recursively redact machine-specific experiment-environment prefixes."""
    if isinstance(value, str):
        return PRIVATE_VENV_PATTERN.subn(PUBLIC_VENV_TOKEN, value)
    if isinstance(value, list):
        out = []
        count = 0
        for item in value:
            replaced, n = normalize_private_venv(item)
            out.append(replaced)
            count += n
        return out, count
    if isinstance(value, dict):
        out = {}
        count = 0
        for key, item in value.items():
            replaced, n = normalize_private_venv(item)
            out[key] = replaced
            count += n
        return out, count
    return value, 0


def git_output(repo, *args):
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    require(proc.returncode == 0,
            "git provenance check failed: "
            + proc.stderr.decode(errors="replace").strip())
    return proc.stdout.decode().strip()


def wilson_interval(successes, total, confidence=0.95):
    require(0 <= successes <= total and total > 0, "invalid Wilson inputs")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (z * math.sqrt(p * (1 - p) / total
                          + z * z / (4 * total * total)) / denominator)
    return center - half, center + half


def _normalise_condition(condition):
    require(condition.startswith("cond"),
            f"historical condition lacks cond prefix: {condition}")
    return condition[4:]


def export_source_records(source_root, legacy_sample, output):
    """One-time deterministic export from the original full result files."""
    legacy = load(legacy_sample)
    require(isinstance(legacy, list) and len(legacy) == 100,
            "legacy sample must contain 100 records")
    result_root = source_root / "probe_eval_results"
    source_commit = git_output(source_root, "rev-parse", "HEAD")
    require(len(source_commit) == 40,
            f"unexpected source commit identifier: {source_commit}")
    source_cache = {}
    source_hashes = {}
    source_blobs = {}
    exported = []
    replacements = 0
    records_modified = 0
    original_records = []
    release_records = []

    for sampled in legacy:
        model = sampled["model"]
        condition = sampled["condition"]
        filename = f"benchmark_results_{condition}.json"
        candidates = [
            result_root / "comparison" / model / filename,
            result_root / "api_comparison_2000" / model / filename,
        ]
        candidates = [path for path in candidates if path.exists()]
        # The historical api_comparison_2000 tree symlinks its local-model
        # directories back to comparison/.  Treat those aliases as one source
        # record and prefer the canonical comparison/ spelling above.
        distinct_sources = {}
        for candidate in candidates:
            distinct_sources.setdefault(candidate.resolve(), candidate)
        require(len(distinct_sources) == 1,
                f"expected one source file for {sampled['sample_id']}: {candidates}")
        path = next(iter(distinct_sources.values()))
        relative = path.relative_to(source_root).as_posix()
        if path not in source_cache:
            source_blob = git_output(
                source_root, "rev-parse", f"{source_commit}:{relative}")
            worktree_blob = git_output(source_root, "hash-object", str(path))
            require(source_blob == worktree_blob,
                    f"source file differs from commit {source_commit}: {relative}")
            source_cache[path] = load(path)["item_results"]
            source_hashes[path] = file_sha256(path)
            source_blobs[path] = source_blob
        matches = [
            item for item in source_cache[path]
            if item.get("natural_language_query")
            == sampled["natural_language_query"]
        ]
        require(len(matches) == 1,
                f"source join is not unique for {sampled['sample_id']}")
        source_record = matches[0]
        require(source_record.get("executed") is True
                and source_record.get("match") is False,
                f"source is not executed-but-unmatched: {sampled['sample_id']}")
        require(source_record.get("task") == sampled["task"],
                f"task mismatch for {sampled['sample_id']}")
        require(source_record.get("difficulty_level")
                == sampled["difficulty_level"],
                f"difficulty mismatch for {sampled['sample_id']}")

        original_records.append(source_record)
        release_record, n_replaced = normalize_private_venv(source_record)
        replacements += n_replaced
        records_modified += int(n_replaced > 0)
        release_records.append(release_record)

        exported.append({
            "sample_id": sampled["sample_id"],
            "model": model,
            "tier": sampled["tier"],
            "condition": condition,
            "source_relative_path": relative,
            "source_git_blob": source_blobs[path],
            "source_file_sha256": source_hashes[path],
            "item_id": source_record.get("item_id"),
            "bench_index": source_record.get("bench_index"),
            "benchmark_original_index": source_record.get(
                "benchmark_original_index"),
            "executed": source_record.get("executed"),
            "match": source_record.get("match"),
            "source_record_original_sha256": record_sha256(source_record),
            "source_record_release_sha256": record_sha256(release_record),
            "source_record": release_record,
        })

    artifact = {
        "schema_version": 1,
        "provenance": {
            "source_repository": "frozen experimental pipeline",
            "source_commit": source_commit,
            "legacy_sample_sha256": file_sha256(legacy_sample),
            "join_key": ["model", "condition", "natural_language_query"],
            "record_policy": "complete source item_result; no inferred labels",
            "adjudication_ledger_present": False,
            "normalization": {
                "scope": "all string values in source_record",
                "rule": PRIVATE_VENV_RULE,
                "replacement": PUBLIC_VENV_TOKEN,
                "replacements": replacements,
                "records_modified": records_modified,
                "original_records_sha256": record_sha256(original_records),
                "release_records_sha256": record_sha256(release_records),
            },
        },
        "records": exported,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    return output


def validate_samples(records_artifact, historical):
    records = records_artifact["records"]
    source_commit = records_artifact["provenance"].get("source_commit", "")
    require(len(source_commit) == 40
            and all(ch in "0123456789abcdef" for ch in source_commit),
            "sample artifact lacks a full source commit")
    require(len(records) == 100, "sample_records.json must contain 100 rows")
    ids = [row["sample_id"] for row in records]
    require(len(set(ids)) == 100, "sample IDs are not unique")
    require(ids == [f"fn{i:03d}" for i in range(1, 101)],
            "sample IDs are not the frozen fn001..fn100 sequence")

    release_records = []
    replacement_count = 0
    records_modified = 0
    for row in records:
        source = row["source_record"]
        require(row["executed"] is True and row["match"] is False,
                f"outer flags invalid for {row['sample_id']}")
        require(source.get("executed") is True and source.get("match") is False,
                f"source flags invalid for {row['sample_id']}")
        require(row["item_id"] == source.get("item_id"),
                f"item_id mismatch for {row['sample_id']}")
        require(row["source_record_release_sha256"] == record_sha256(source),
                f"release source-record hash mismatch for {row['sample_id']}")
        rendered_source = json.dumps(source, ensure_ascii=False)
        require(PRIVATE_VENV_PATTERN.search(rendered_source) is None,
                f"private experiment-environment path remains in {row['sample_id']}")
        n_replaced = rendered_source.count(PUBLIC_VENV_TOKEN)
        original_hash = row["source_record_original_sha256"]
        require(len(original_hash) == 64
                and all(ch in "0123456789abcdef" for ch in original_hash),
                f"invalid original source-record hash for {row['sample_id']}")
        if n_replaced == 0:
            require(original_hash == row["source_record_release_sha256"],
                    f"unmodified hashes differ for {row['sample_id']}")
        release_records.append(source)
        replacement_count += n_replaced
        records_modified += int(n_replaced > 0)
        require(len(row.get("source_git_blob", "")) == 40,
                f"source git blob missing for {row['sample_id']}")
        require(Path(row["source_relative_path"]).name
                == f"benchmark_results_{row['condition']}.json",
                f"condition/path mismatch for {row['sample_id']}")

    condition_counts = Counter(
        _normalise_condition(row["condition"]) for row in records)
    tier_counts = Counter(row["tier"] for row in records)
    model_counts = Counter(row["model"] for row in records)
    task_counts = Counter(row["source_record"]["task"] for row in records)
    difficulty_counts = Counter(
        row["source_record"]["difficulty_level"] for row in records)

    require(set(condition_counts) == set(SAMPLED_CONDITIONS),
            f"unexpected sample conditions: {sorted(condition_counts)}")
    require(len(tier_counts) == 5 and len(model_counts) == 15
            and len(task_counts) == 15 and len(difficulty_counts) == 4,
            "sample scope does not match the frozen 5/15/15/4 design")
    require(dict(tier_counts) == historical["sample"]["tier_breakdown_sampled"],
            "sample tier counts disagree with historical aggregate")

    normalization = records_artifact["provenance"].get("normalization", {})
    original_collection_hash = normalization.get("original_records_sha256", "")
    require(len(original_collection_hash) == 64
            and all(ch in "0123456789abcdef"
                    for ch in original_collection_hash),
            "normalization manifest lacks original-record collection SHA-256")
    expected_normalization = {
        "scope": "all string values in source_record",
        "rule": PRIVATE_VENV_RULE,
        "replacement": PUBLIC_VENV_TOKEN,
        "replacements": replacement_count,
        "records_modified": records_modified,
        "original_records_sha256": original_collection_hash,
        "release_records_sha256": record_sha256(release_records),
    }
    require(normalization == expected_normalization,
            "sample normalization manifest does not reproduce")

    for key, counts in (("by_tier", tier_counts),
                        ("by_task", task_counts),
                        ("by_model", model_counts)):
        archived = historical["breakdowns"][key]
        require(set(archived) == set(counts), f"{key} strata disagree")
        require(all(archived[stratum]["n"] == count
                    for stratum, count in counts.items()),
                f"{key} sample counts disagree")

    return {
        "n": len(records),
        "conditions": dict(sorted(condition_counts.items())),
        "tiers": dict(sorted(tier_counts.items())),
        "n_models": len(model_counts),
        "n_task_families": len(task_counts),
        "n_difficulty_levels": len(difficulty_counts),
        "all_executed_but_unmatched": True,
        "all_source_record_hashes_verified": True,
        "normalization_verified": True,
        "normalization_replacements": replacement_count,
        "normalization_records_modified": records_modified,
        "source_commit": source_commit,
        "item_level_adjudication_present": False,
    }


def validate_historical_aggregate(historical):
    sample = historical["sample"]
    upheld = sample["n_false_negative_upheld"]
    ambiguous = sample["n_ambiguous"]
    genuine = sample["n_genuine_error"]
    total = sample["n_judged"]
    require((upheld, ambiguous, genuine, total) == (8, 5, 87, 100),
            "historical 8/5/87 aggregate changed")
    require(upheld + ambiguous + genuine == total,
            "historical labels do not sum to n_judged")

    for key in ("by_tier", "by_task", "by_model"):
        rows = historical["breakdowns"][key].values()
        require(sum(row["n"] for row in rows) == total,
                f"historical {key} n does not sum to 100")
        require(sum(row["fn"] for row in rows) == upheld,
                f"historical {key} FN does not sum to 8")
        require(sum(row["ambiguous"] for row in rows) == ambiguous,
                f"historical {key} ambiguous does not sum to 5")

    subtype = historical["breakdowns"]["by_fn_subtype_upheld_fn_only"]
    require(subtype == {"parser_extraction": 8},
            "historical upheld subtype aggregate changed")
    low, high = wilson_interval(upheld, total)
    archived = historical["false_negative_rate"]["wilson95_ci"]
    require(math.isclose(low, archived[0], abs_tol=5e-5)
            and math.isclose(high, archived[1], abs_tol=5e-5),
            "historical Wilson interval does not independently recompute")

    return {
        "reported_counts": {
            "confirmed_false_negative": upheld,
            "ambiguous": ambiguous,
            "genuine_error": genuine,
        },
        "unweighted_rate": upheld / total,
        "nominal_wilson95": [low, high],
        "upheld_subtype_aggregate": subtype,
        "breakdowns_arithmetically_consistent": True,
        "reproducibility_status": (
            "aggregate-only: no item-level adjudication ledger"
        ),
        "population_transport_performed": False,
    }


def unique_census():
    seen = set()
    per_condition = {}
    for path in (PROACTIVE, REACTIVE):
        for condition, models in load(path).items():
            row = per_condition.setdefault(condition, {
                "cells": 0, "total": 0, "executed": 0, "matched": 0,
                "executed_but_unmatched": 0, "parse_failures": 0,
            })
            for model, cell in models.items():
                key = (condition, model)
                require(key not in seen, f"duplicate public aggregate cell: {key}")
                seen.add(key)
                total = cell["n_total"]
                executed = cell["n_executed"]
                matched = cell["n_matched"]
                require(total == 2000, f"unexpected cell size for {key}: {total}")
                require(0 <= matched <= executed <= total,
                        f"invalid cell counts for {key}")
                row["cells"] += 1
                row["total"] += total
                row["executed"] += executed
                row["matched"] += matched
                row["executed_but_unmatched"] += executed - matched
                row["parse_failures"] += cell.get("n_parse_failures", 0)

    require(set(per_condition) == set(EXPECTED_CONDITIONS),
            "public aggregate condition set changed")
    for condition, expected in EXPECTED_CONDITIONS.items():
        row = per_condition[condition]
        got = (row["cells"], row["total"], row["executed"], row["matched"],
               row["executed_but_unmatched"], row["parse_failures"])
        require(got == expected,
                f"condition census changed for {condition}: {got} != {expected}")

    fields = ("cells", "total", "executed", "matched",
              "executed_but_unmatched", "parse_failures")
    totals = {field: sum(row[field] for row in per_condition.values())
              for field in fields}
    expected_totals = {
        "cells": 196,
        "total": 392000,
        "executed": 188297,
        "matched": 121487,
        "executed_but_unmatched": 66810,
        "parse_failures": 16551,
    }
    require(totals == expected_totals,
            f"unique full-panel census changed: {totals}")

    sampled_frame = {
        field: sum(per_condition[condition][field]
                   for condition in SAMPLED_CONDITIONS)
        for field in fields
    }
    expected_sampled = {
        "cells": 104,
        "total": 208000,
        "executed": 94445,
        "matched": 58999,
        "executed_but_unmatched": 35446,
        "parse_failures": 7824,
    }
    require(sampled_frame == expected_sampled,
            f"eight-condition frame census changed: {sampled_frame}")

    return {
        "deduplication_key": ["condition", "model"],
        "input_artifacts": [
            PROACTIVE.relative_to(ROOT).as_posix(),
            REACTIVE.relative_to(ROOT).as_posix(),
        ],
        "full_unique_panel": totals,
        "eight_condition_diagnostic_frame": sampled_frame,
        "per_condition": dict(sorted(per_condition.items())),
        "audit_rate_transport_performed": False,
    }


def recompute():
    historical = load(HISTORICAL_AUDIT)
    records = load(SAMPLE_RECORDS)
    return {
        "status": "ok",
        "sample_provenance": validate_samples(records, historical),
        "historical_aggregate": validate_historical_aggregate(historical),
        "unique_census": unique_census(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-from-source", type=Path, metavar="IGPT_ROOT",
        help="one-time export of complete records from the original checkout")
    parser.add_argument(
        "--legacy-sample", type=Path,
        help="legacy p01_fn_samples.json used with --export-from-source")
    parser.add_argument(
        "--output", type=Path, default=SAMPLE_RECORDS,
        help="export destination (default: audit/matcher_fn/sample_records.json)")
    parser.add_argument(
        "--out", type=Path,
        help="write the recomputed validation report to this path")
    args = parser.parse_args(argv)

    if args.export_from_source is not None:
        if args.legacy_sample is None:
            parser.error("--legacy-sample is required with --export-from-source")
        exported = export_source_records(
            args.export_from_source.resolve(), args.legacy_sample.resolve(),
            args.output.resolve())
        print(f"exported {exported}")
        print(f"sha256 {file_sha256(exported)}")
        return 0

    try:
        result = recompute()
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"matcher-FN audit validation failed: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
