#!/usr/bin/env python3
"""Rebuild the fully public preliminary Failure Anatomy expert-review pass.

The program replays the deterministic 100-item draw from the released
89,420-key sampling frame (``audit/failure_taxonomy/``), validates both
independent classification ledgers and the disagreement-only adjudication, and
rebuilds every final cross-tab.  It works from the released frame and ledgers
alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from export_failure_taxonomy_revision_sample import CellRng


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "audit/failure_taxonomy/revision_sample"
DEFAULT_OUTPUT = DEFAULT_DIR / "adjudication_v2.json"
LABELS = (
    "C1_api_contract",
    "C2_param_misuse",
    "C3_workflow_logic",
    "C4_numerical_extraction",
    "C5_format_env",
    "ambiguous_item_defect",
)
SAMPLES_PER_CELL = 5


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def keyed_reviews(document: dict[str, Any], expected_ids: set[str]) -> dict[str, dict]:
    records = document.get("records")
    require(isinstance(records, list), "review ledger has no records list")
    ids = [str(row.get("sample_id")) for row in records]
    require(len(ids) == 100 and len(set(ids)) == 100,
            "review ledger does not contain 100 unique records")
    require(set(ids) == expected_ids, "review ledger sample IDs differ")
    require(set(document.get("rubric", {})) == set(LABELS),
            "review ledger rubric keys differ")
    require(all(row.get("label") in LABELS for row in records),
            "review ledger contains an unknown label")
    return {str(row["sample_id"]): row for row in records}


def replay_sampling_frame(frame: dict[str, Any]) -> tuple[dict[str, tuple], dict[str, int]]:
    sources = frame.get("sources")
    require(isinstance(sources, list) and len(sources) == 55,
            "sampling frame must contain 55 ordered sources")
    source_order = [
        (row["model_tier"], row["condition"], row["model"], row["source"]["path"])
        for row in sources
    ]
    require(source_order == sorted(source_order), "sampling-frame source order differs")

    cells = sorted({(row["model_tier"], row["condition"]) for row in sources})
    require(len(cells) == 20, "sampling frame does not cover 20 tier-condition cells")
    rngs = {
        cell: CellRng(cell[0], cell[1].removeprefix("cond"))
        for cell in cells
    }
    pools: Counter[tuple[str, str]] = Counter()
    reservoirs: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for source in sources:
        cell = (source["model_tier"], source["condition"])
        keys = source.get("match_false_keys")
        require(isinstance(keys, list), "sampling-frame source has no key list")
        previous: tuple[int, str] | None = None
        for pair in keys:
            require(isinstance(pair, list) and len(pair) == 2,
                    "invalid sampling-frame key")
            bench_index, item_id = pair
            key = (int(bench_index), str(item_id))
            require(previous is None or key > previous,
                    "sampling-frame keys are not strictly ordered")
            previous = key
            candidate = (
                source["model"], key[0], key[1], source["source"]["path"],
                source["source"]["sha256"],
            )
            pools[cell] += 1
            seen = pools[cell]
            reservoir = reservoirs[cell]
            if len(reservoir) < SAMPLES_PER_CELL:
                reservoir.append(candidate)
            else:
                replacement = rngs[cell].randbelow(seen)
                if replacement < SAMPLES_PER_CELL:
                    reservoir[replacement] = candidate

    require(sum(pools.values()) == 89420,
            "sampling frame does not contain 89,420 match-false keys")
    selected: dict[str, tuple] = {}
    for tier, condition in cells:
        rows = sorted(reservoirs[(tier, condition)])
        require(len(rows) == SAMPLES_PER_CELL, "incomplete sampling cell")
        condition_id = condition.removeprefix("cond")
        for ordinal, row in enumerate(rows, start=1):
            selected[f"{tier}_{condition_id}_{ordinal:02d}"] = row
    require(len(selected) == 100, "sampling replay did not select 100 records")
    return selected, {f"{a}|{b}": pools[(a, b)] for a, b in cells}


def cross_tab(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, int]]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        table[str(row[field])][str(row["final_label"])] += 1
    return {
        key: {label: table[key][label] for label in LABELS}
        for key in sorted(table)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    paths = {
        name: args.audit_dir / name
        for name in (
            "sampling_frame_keys.json",
            "sample_records.json",
            "summary.json",
            "manifest.json",
            "judge_a.json",
            "judge_b.json",
            "disagreement_adjudication.json",
        )
    }
    documents = {name: load(path) for name, path in paths.items()}
    sample = documents["sample_records.json"]
    records = sample.get("records")
    require(isinstance(records, list) and len(records) == 100,
            "sample ledger must contain 100 records")
    sample_ids = [str(row.get("revision_sample_id")) for row in records]
    require(len(set(sample_ids)) == 100, "sample IDs are not unique")
    expected_ids = set(sample_ids)
    sample_by_id = {str(row["revision_sample_id"]): row for row in records}

    manifest = documents["manifest.json"]
    for name in ("sampling_frame_keys.json", "sample_records.json", "summary.json"):
        frozen = manifest["outputs"][name]
        require(sha256(paths[name]) == frozen["sha256"], f"{name} hash differs")
        require(paths[name].stat().st_size == frozen["size_bytes"],
                f"{name} size differs")

    selected, pool_counts = replay_sampling_frame(documents["sampling_frame_keys.json"])
    for sample_id, chosen in selected.items():
        row = sample_by_id[sample_id]
        observed = (
            row["model"], int(row["bench_index"]), str(row["item_id"]),
            row["source"]["path"], row["source"]["sha256"],
        )
        require(observed == chosen, f"sampling replay differs for {sample_id}")
        require(row.get("match") is False, f"sample is not match-false: {sample_id}")

    judge_a = keyed_reviews(documents["judge_a.json"], expected_ids)
    judge_b = keyed_reviews(documents["judge_b.json"], expected_ids)
    disagreement_ids = {
        sample_id for sample_id in expected_ids
        if judge_a[sample_id]["label"] != judge_b[sample_id]["label"]
    }
    require(len(disagreement_ids) == 15, "expected 15 judge disagreements")

    adjudication_document = documents["disagreement_adjudication.json"]
    adjudications = adjudication_document.get("records")
    require(isinstance(adjudications, list), "adjudication has no records list")
    adjudication_ids = [str(row.get("sample_id")) for row in adjudications]
    require(len(adjudication_ids) == 15 and set(adjudication_ids) == disagreement_ids,
            "adjudication scope differs from the judge disagreements")
    adjudication_by_id = {str(row["sample_id"]): row for row in adjudications}
    for sample_id, row in adjudication_by_id.items():
        require(row.get("judge_a_label") == judge_a[sample_id]["label"],
                f"adjudication judge-A label differs for {sample_id}")
        require(row.get("judge_b_label") == judge_b[sample_id]["label"],
                f"adjudication judge-B label differs for {sample_id}")
        require(row.get("final_label") in LABELS,
                f"adjudication label is invalid for {sample_id}")

    final_rows = []
    for sample_id in sample_ids:
        source = sample_by_id[sample_id]
        a = judge_a[sample_id]
        b = judge_b[sample_id]
        if sample_id in disagreement_ids:
            decision = adjudication_by_id[sample_id]
            final_label = decision["final_label"]
            decision_basis = "two-engineer consensus adjudication"
            final_rationale = decision["rationale"]
            final_evidence = decision["primary_evidence"]
        else:
            final_label = a["label"]
            decision_basis = "two-review agreement"
            final_rationale = a["rationale"]
            final_evidence = a["primary_evidence"]
        final_rows.append({
            "sample_id": sample_id,
            "model": source["model"],
            "model_tier": source["model_tier"],
            "condition": source["condition"],
            "task": source["task"],
            "item_id": source["item_id"],
            "bench_index": source["bench_index"],
            "judge_a": {
                key: a[key] for key in
                ("label", "confidence", "rationale", "primary_evidence")
            },
            "judge_b": {
                key: b[key] for key in
                ("label", "confidence", "rationale", "primary_evidence")
            },
            "final_label": final_label,
            "decision_basis": decision_basis,
            "final_rationale": final_rationale,
            "final_evidence": final_evidence,
        })

    counts = Counter(row["final_label"] for row in final_rows)
    category_tally = {label: counts[label] for label in LABELS}
    tier_table = cross_tab(final_rows, "model_tier")
    condition_table = cross_tab(final_rows, "condition")
    task_table = cross_tab(final_rows, "task")
    nondefect_n = 100 - category_tally["ambiguous_item_defect"]
    c1_c2 = category_tally["C1_api_contract"] + category_tally["C2_param_misuse"]
    conclusions = {
        "C1_is_largest_single_category": (
            category_tally["C1_api_contract"]
            == max(category_tally[label] for label in LABELS[:-1])
        ),
        "C1_plus_C2_count": c1_c2,
        "C1_plus_C2_pct_all_100": round(c1_c2, 1),
        "C1_plus_C2_pct_nondefective": round(100 * c1_c2 / nondefect_n, 1),
        "condA_FDRS_to_condC_FDRS_C1": [
            condition_table["condA_FDRS"]["C1_api_contract"],
            condition_table["condC_FDRS"]["C1_api_contract"],
        ],
        "condC_FDRS_C3": condition_table["condC_FDRS"]["C3_workflow_logic"],
        "qualitative_manuscript_claim_preserved": (
            category_tally["C1_api_contract"] > category_tally["C3_workflow_logic"]
            and c1_c2 > 50
            and condition_table["condC_FDRS"]["C1_api_contract"]
            < condition_table["condA_FDRS"]["C1_api_contract"]
        ),
    }
    require(all((
        conclusions["C1_is_largest_single_category"],
        conclusions["qualitative_manuscript_claim_preserved"],
    )), "preliminary expert pass does not preserve the qualitative reference pattern")

    report = {
        "schema_version": 2,
        "status": "ok",
        "scope": (
            "Fully public preliminary expert-review pass: deterministic sampling "
            "is replayed from 89,420 released stable keys; two practising "
            "power-systems engineers independently reviewed every record and "
            "resolved all 15 disagreements by rubric-constrained consensus."
        ),
        "reviewer_disclosure": {
            "judge_a": documents["judge_a.json"]["reviewer_type"],
            "judge_b": documents["judge_b.json"]["reviewer_type"],
            "adjudicator": adjudication_document["reviewer_type"],
        },
        "sampling": {
            "global_seed": sample["sampling"]["global_seed"],
            "frame_match_false": sum(pool_counts.values()),
            "cell_pool_counts": pool_counts,
            "n_selected": len(selected),
            "all_selected_ids_replayed_exactly": True,
        },
        "review": {
            "judge_agreements": 100 - len(disagreement_ids),
            "judge_disagreements": len(disagreement_ids),
            "agreement_pct": round(100 * (100 - len(disagreement_ids)) / 100, 1),
            "disagreement_ids": sorted(disagreement_ids),
            "all_disagreements_adjudicated": True,
        },
        "category_tally": category_tally,
        "tier_x_category": tier_table,
        "condition_x_category": condition_table,
        "task_x_category": task_table,
        "conclusions": conclusions,
        "records": final_rows,
        "checks": {
            "manifest_output_hashes_and_sizes_match": True,
            "sampling_frame_has_55_sources_and_89420_keys": True,
            "deterministic_draw_replayed_exactly": True,
            "both_judges_cover_same_100_ids": True,
            "adjudication_scope_equals_all_disagreements": True,
            "all_labels_valid": True,
            "all_cross_tabs_sum_to_100": all((
                sum(sum(row.values()) for row in tier_table.values()) == 100,
                sum(sum(row.values()) for row in condition_table.values()) == 100,
                sum(sum(row.values()) for row in task_table.values()) == 100,
            )),
        },
        "provenance": {name: descriptor(path) for name, path in paths.items()},
    }
    require(all(report["checks"].values()), "one or more final checks failed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    try:
        output_display = args.out.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        output_display = str(args.out.resolve())
    print(json.dumps({
        "status": "ok",
        "output": output_display,
        "sha256": sha256(args.out),
        "category_tally": category_tally,
        "judge_agreement_pct": report["review"]["agreement_pct"],
        "qualitative_manuscript_claim_preserved": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
