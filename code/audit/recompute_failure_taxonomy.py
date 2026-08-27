#!/usr/bin/env python3
"""Recompute the 100-item failure-taxonomy diagnostic from public evidence.

The item-level classification ledger rebuilds every category, tier, condition,
and task cross-tabulation.  The 89,420-element sampling frame is rebuilt
independently from the compact primary-outcome archive.  This script does not
re-adjudicate qualitative labels or replay the historical sampler because the
legacy ledger did not retain original benchmark item IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "audit/failure_taxonomy/taxonomy_results.json"
DEFAULT_MANIFEST = ROOT / "audit/failure_taxonomy/manifest.json"
DEFAULT_COMPACT = ROOT / "results/raw/main_experiment/primary_outcomes_compact.json"
DEFAULT_OUTPUT = ROOT / "audit/failure_taxonomy/recomputed_summary.json"

CATEGORIES = [
    "C1_api_contract",
    "C2_param_misuse",
    "C3_workflow_logic",
    "C4_numerical_extraction",
    "C5_format_env",
]
TIERS = ["tier1", "tier2", "tier3", "tier4"]
CONDITIONS = ["condA_FD", "condA_FDR", "condA_FDRS", "condA_FX", "condC_FDRS"]
POOL_CONDITIONS = ["A_FD", "A_FDR", "A_FDRS", "A_FX", "C_FDRS"]
MODEL_TIERS = {
    "Qwen2.5-Coder-0.5B": "tier1",
    "Qwen2.5-Coder-1.5B": "tier1",
    "Qwen2.5-Coder-7B": "tier2",
    "Llama-3.1-8B": "tier2",
    "Qwen2.5-Coder-14B": "tier2",
    "Qwen2.5-Coder-32B": "tier2",
    "Llama-3.1-70B": "tier3",
    "gpt-oss-120b": "tier3",
    "Llama-3.1-405B": "tier4",
    "Qwen3-Coder-480B": "tier4",
    "Qwen3-Coder-Next": "tier4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zero_row() -> dict[str, int]:
    return {category: 0 for category in CATEGORIES}


def cross_tab(rows: list[dict], field: str) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = defaultdict(zero_row)
    for row in rows:
        table[str(row[field])][str(row["category_id"])] += 1
    return {key: table[key] for key in sorted(table)}


def pool_count(compact: dict) -> tuple[int, int, int]:
    runs = compact["runs"]
    selected = []
    for key, run in runs.items():
        if run.get("group") == "comparison" and run.get("condition") in POOL_CONDITIONS:
            selected.append((key, run))
    if len(selected) != 55:
        raise ValueError(f"expected 55 model-condition runs, found {len(selected)}")
    models = {run["model"] for _, run in selected}
    if len(models) != 11:
        raise ValueError(f"expected 11 models in sampling frame, found {len(models)}")
    n_total = sum(len(run["match_bits"]) for _, run in selected)
    n_failed = sum(run["match_bits"].count("0") for _, run in selected)
    return n_failed, n_total, len(models)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    rows = ledger["classifications"]

    ids = [str(row["sample_id"]) for row in rows]
    expected_ids = {f"s{number:03d}" for number in range(1, 101)}
    if len(rows) != 100 or len(set(ids)) != 100 or set(ids) != expected_ids:
        raise ValueError("classification ledger is not the complete s001--s100 set")
    if {row["category_id"] for row in rows} - set(CATEGORIES):
        raise ValueError("unknown category in classification ledger")
    if {row["model_tier"] for row in rows} != set(TIERS):
        raise ValueError("tier coverage mismatch")
    if {row["condition"] for row in rows} != set(CONDITIONS):
        raise ValueError("condition coverage mismatch")

    cell_counts = Counter((row["model_tier"], row["condition"]) for row in rows)
    expected_cells = {(tier, condition) for tier in TIERS for condition in CONDITIONS}
    if set(cell_counts) != expected_cells or set(cell_counts.values()) != {5}:
        raise ValueError("ledger is not a balanced 4 x 5 x 5 design")

    summary_counts = Counter(row["category_id"] for row in rows)
    summary = {category: summary_counts[category] for category in CATEGORIES}
    tier_table = cross_tab(rows, "model_tier")
    condition_table = cross_tab(rows, "condition")
    task_table = cross_tab(rows, "task")

    archived = ledger["tally"]
    checks = {
        "summary_matches_archived": summary == archived["summary"],
        "total_matches_archived": len(rows) == archived["total"],
        "tier_table_matches_archived": tier_table == archived["tier_x_category"],
        "condition_table_matches_archived": condition_table == archived["condition_x_category"],
        "task_table_matches_archived": task_table == archived["task_x_category"],
        "model_tier_mapping_matches_frozen_design": (
            {row["model"] for row in rows} == set(MODEL_TIERS)
            and all(MODEL_TIERS.get(row["model"]) == row["model_tier"]
                    for row in rows)
        ),
    }

    expected_hashes = {entry["path"]: entry for entry in manifest["files"]}
    ledger_rel = str(args.ledger.resolve().relative_to(ROOT))
    checks["ledger_release_hash_matches_manifest"] = (
        sha256(args.ledger) == expected_hashes[ledger_rel]["release_sha256"]
        and args.ledger.stat().st_size == expected_hashes[ledger_rel]["release_size_bytes"]
    )
    checks["compact_hash_matches_export"] = (
        sha256(args.compact)
        == manifest["pool_reconstruction"]["sha256_after_path_normalization"]
    )

    n_failed, n_pool_total, n_pool_models = pool_count(compact)
    checks["pool_count_matches_historical_report"] = (
        n_failed == manifest["pool_reconstruction"]["expected_count"] == 89420
    )

    tier_totals = {tier: sum(tier_table[tier].values()) for tier in TIERS}
    historical_reported_claims = {
        "n_classified": len(rows),
        "sampling_frame_match_false": n_failed,
        "sampling_frame_total_outcomes": n_pool_total,
        "sampling_frame_models": n_pool_models,
        "n_tiers": len({row["model_tier"] for row in rows}),
        "n_conditions": len({row["condition"] for row in rows}),
        "n_task_families": len({row["task"] for row in rows}),
        "C1_plus_C2": summary["C1_api_contract"] + summary["C2_param_misuse"],
        "tier1_C1_count": tier_table["tier1"]["C1_api_contract"],
        "tier1_total": tier_totals["tier1"],
        "tier1_C1_pct": round(100 * tier_table["tier1"]["C1_api_contract"] / tier_totals["tier1"], 1),
        "tier3_C2_plus_C3": (
            tier_table["tier3"]["C2_param_misuse"]
            + tier_table["tier3"]["C3_workflow_logic"]
        ),
        "tier3_total": tier_totals["tier3"],
        "tier3_C2_plus_C3_pct": round(
            100 * (tier_table["tier3"]["C2_param_misuse"]
                   + tier_table["tier3"]["C3_workflow_logic"])
            / tier_totals["tier3"], 1
        ),
        "condA_FDRS_C1": condition_table["condA_FDRS"]["C1_api_contract"],
        "condC_FDRS_C1": condition_table["condC_FDRS"]["C1_api_contract"],
        "condC_FDRS_C3": condition_table["condC_FDRS"]["C3_workflow_logic"],
        "C4_C5_nonzero_tiers": [
            tier for tier in TIERS
            if tier_table[tier]["C4_numerical_extraction"]
            or tier_table[tier]["C5_format_env"]
        ],
    }
    checks["historical_reported_claims_match"] = historical_reported_claims == {
        "n_classified": 100,
        "sampling_frame_match_false": 89420,
        "sampling_frame_total_outcomes": 110000,
        "sampling_frame_models": 11,
        "n_tiers": 4,
        "n_conditions": 5,
        "n_task_families": 12,
        "C1_plus_C2": 74,
        "tier1_C1_count": 21,
        "tier1_total": 25,
        "tier1_C1_pct": 84.0,
        "tier3_C2_plus_C3": 16,
        "tier3_total": 25,
        "tier3_C2_plus_C3_pct": 64.0,
        "condA_FDRS_C1": 12,
        "condC_FDRS_C1": 3,
        "condC_FDRS_C3": 10,
        "C4_C5_nonzero_tiers": ["tier4"],
    }

    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("failure-taxonomy validation failed: " + ", ".join(failed))

    report = {
        "schema_version": 1,
        "status": "ok",
        "scope": (
            "Historical provenance only: rebuilds the match-false frame count "
            "and every archived classification cross-tab. It does not replay "
            "sampling or re-adjudicate labels and is superseded for current "
            "claims by the formal expert ledgers under human_review/."
        ),
        "checks": checks,
        "sampling_design": {
            "cell_counts": {
                f"{tier}|{condition}": cell_counts[(tier, condition)]
                for tier in TIERS for condition in CONDITIONS
            }
        },
        "category_tally": summary,
        "tier_x_category": tier_table,
        "condition_x_category": condition_table,
        "task_x_category": task_table,
        "historical_reported_claims": historical_reported_claims,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] rebuilt failure taxonomy from public evidence -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
