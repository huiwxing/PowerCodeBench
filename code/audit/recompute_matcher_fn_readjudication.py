#!/usr/bin/env python3
"""Build the versioned matcher-FN readjudication from expert-review ledgers.

The final labels are derived only from the frozen 100 source records and three
record-level review passes conducted by two practising power-systems engineers:
two independent primary passes, an isolated blind re-check, an adversarial pass
over candidate false negatives, and joint expert reconciliation of every
disagreement/challenge.  The historical 8/5/87 aggregate is loaded only after
labels are frozen, and only for a post-hoc marginal consistency check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "audit/matcher_fn"
READJUDICATION = AUDIT / "revision_readjudication"
DEFAULT_OUTPUT = READJUDICATION / "adjudication_v2.json"
SOURCE = AUDIT / "sample_records.json"
HISTORICAL = AUDIT / "P01_false_negative_audit.json"
JUDGES = {
    "judge_a": READJUDICATION / "judge_a.json",
    "judge_b": READJUDICATION / "judge_b.json",
    "judge_blind": READJUDICATION / "judge_blind.json",
}
REFUTATION = READJUDICATION / "adversarial_refutation.json"
DECISIONS = READJUDICATION / "final_decisions.json"
LABELS = {"false_negative", "ambiguous", "genuine_error"}
REFUTATION_LABELS = {
    "uphold_false_negative": "false_negative",
    "ambiguous": "ambiguous",
    "genuine_error": "genuine_error",
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def indexed_records(path: Path, key: str, allowed: set[str]) -> dict[str, dict]:
    payload = read(path)
    rows = payload["records"]
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate sample IDs")
    bad = sorted({row[key] for row in rows} - allowed)
    if bad:
        raise ValueError(f"{path}: unsupported labels {bad}")
    return {row["sample_id"]: row for row in rows}


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(center - half, 4), round(center + half, 4)]


def count_by(rows: list[dict], source: dict[str, dict], field: str) -> dict:
    counts: dict[str, Counter] = {}
    for row in rows:
        record = source[row["sample_id"]]
        value = (
            record["source_record"][field]
            if field in {"task", "difficulty_level"}
            else record[field]
        )
        counts.setdefault(str(value), Counter())[row["final_label"]] += 1
    return {
        value: {
            "n": sum(counter.values()),
            **{label: counter[label] for label in sorted(LABELS)},
        }
        for value, counter in sorted(counts.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_payload = read(SOURCE)
    source_rows = source_payload["records"]
    source = {row["sample_id"]: row for row in source_rows}
    expected_ids = {f"fn{number:03d}" for number in range(1, 101)}
    if len(source_rows) != 100 or set(source) != expected_ids:
        raise ValueError("source record set is not exactly fn001--fn100")

    judges = {
        name: indexed_records(path, "label", LABELS)
        for name, path in JUDGES.items()
    }
    for name, rows in judges.items():
        if set(rows) != expected_ids:
            raise ValueError(f"{name}: record IDs are incomplete")

    refutation = indexed_records(
        REFUTATION, "verdict", set(REFUTATION_LABELS)
    )
    decisions_payload = read(DECISIONS)
    decisions = {row["sample_id"]: row for row in decisions_payload["decisions"]}
    if len(decisions) != len(decisions_payload["decisions"]):
        raise ValueError("duplicate final-decision IDs")
    if set(decisions) - expected_ids:
        raise ValueError("final decision names an unknown sample")

    final_rows = []
    unresolved = []
    pairwise_agreement = Counter()
    for sample_id in sorted(expected_ids):
        review_labels = {name: rows[sample_id]["label"] for name, rows in judges.items()}
        labels = list(review_labels.values())
        unanimous = len(set(labels)) == 1
        for left, right in (("judge_a", "judge_b"), ("judge_a", "judge_blind"),
                            ("judge_b", "judge_blind")):
            pairwise_agreement[f"{left}|{right}"] += review_labels[left] == review_labels[right]
        adversarial = refutation.get(sample_id)
        adversarial_label = (
            REFUTATION_LABELS[adversarial["verdict"]] if adversarial else None
        )
        challenged = adversarial_label is not None and (
            not unanimous or adversarial_label != labels[0]
        )
        decision = decisions.get(sample_id)
        if decision:
            final_label = decision["final_label"]
            if final_label not in LABELS:
                raise ValueError(f"{sample_id}: invalid final decision")
            basis = "explicit resolution of reviewer disagreement/adversarial challenge"
            rationale = decision["rationale"]
        elif unanimous and not challenged:
            final_label = labels[0]
            basis = "unanimous across three expert-review passes"
            rationale = judges["judge_blind"][sample_id]["rationale"]
        else:
            unresolved.append({
                "sample_id": sample_id,
                "review_labels": review_labels,
                "adversarial_label": adversarial_label,
            })
            continue

        record = source[sample_id]
        parser_error = record["source_record"].get("match_detail", {}).get("error")
        final_rows.append({
            "sample_id": sample_id,
            "item_id": record["item_id"],
            "source_record_release_sha256": record["source_record_release_sha256"],
            "final_label": final_label,
            "final_basis": basis,
            "final_rationale": rationale,
            "parser_error": parser_error,
            "reviews": [
                {
                    "reviewer": name,
                    "label": rows[sample_id]["label"],
                    "confidence": rows[sample_id]["confidence"],
                    "rationale": rows[sample_id]["rationale"],
                    "parser_evidence": rows[sample_id]["parser_evidence"],
                }
                for name, rows in judges.items()
            ],
            "adversarial_refutation": adversarial,
        })

    if unresolved:
        print(json.dumps({"unresolved": unresolved}, indent=2))
        raise ValueError(
            f"{len(unresolved)} items need an explicit entry in {DECISIONS}"
        )
    if len(final_rows) != 100:
        raise ValueError("final adjudication is incomplete")

    tally = Counter(row["final_label"] for row in final_rows)
    breakdowns = {
        field: count_by(final_rows, source, field)
        for field in ("tier", "model", "condition", "task", "difficulty_level")
    }
    fn_rows = [row for row in final_rows if row["final_label"] == "false_negative"]
    parser_only = all(
        str(row["parser_error"]).startswith("cannot_parse_") for row in fn_rows
    )

    # This comparison happens only after all item labels have been fixed.
    historical = read(HISTORICAL)
    historical_breakdowns = historical["breakdowns"]
    posthoc = {
        "fn_total_matches": tally["false_negative"]
        == historical["sample"]["n_false_negative_upheld"],
        "fn_by_tier_matches": all(
            row["false_negative"] == historical_breakdowns["by_tier"][key]["fn"]
            for key, row in breakdowns["tier"].items()
        ),
        "fn_by_model_matches": all(
            row["false_negative"] == historical_breakdowns["by_model"][key]["fn"]
            for key, row in breakdowns["model"].items()
        ),
        "fn_by_task_matches": all(
            row["false_negative"] == historical_breakdowns["by_task"][key]["fn"]
            for key, row in breakdowns["task"].items()
        ),
    }

    output = {
        "schema_version": 1,
        "audit_version": "matcher-fn-revision-readjudication-v2",
        "scope": (
            "Same frozen 100 A/C-family source records; two practising "
            "power-systems engineers conducted three record-level review "
            "passes (two independent primary passes and one isolated blind "
            "re-check), followed by an adversarial challenge pass and joint "
            "expert reconciliation. Deterministic frozen record evidence "
            "supported the decisions; matcher and parser outputs were not "
            "verdicts. No generated code was re-executed."
        ),
        "source": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": sha256(SOURCE),
            "records": len(source_rows),
        },
        "review_files": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
            }
            for name, path in JUDGES.items()
        } | {
            "adversarial_refutation": {
                "path": REFUTATION.relative_to(ROOT).as_posix(),
                "sha256": sha256(REFUTATION),
            },
            "final_decisions": {
                "path": DECISIONS.relative_to(ROOT).as_posix(),
                "sha256": sha256(DECISIONS),
            },
        },
        "tally": {label: tally[label] for label in sorted(LABELS)},
        "false_negative_rate": round(tally["false_negative"] / 100, 4),
        "false_negative_wilson95": wilson(tally["false_negative"], 100),
        "false_negative_ids": [row["sample_id"] for row in fn_rows],
        "all_upheld_false_negatives_are_parser_extraction_misses": parser_only,
        "review_agreement": {
            "three_way_unanimous": sum(
                len({judges[name][sample_id]["label"] for name in judges}) == 1
                for sample_id in expected_ids
            ),
            "pairwise_agreement_counts_out_of_100": dict(pairwise_agreement),
        },
        "breakdowns": breakdowns,
        "posthoc_historical_fn_marginal_check": {
            "note": "Historical aggregate loaded only after final item labels were frozen.",
            **posthoc,
            "all_match": all(posthoc.values()),
        },
        "records": final_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "tally": output["tally"],
        "review_agreement": output["review_agreement"],
        "posthoc_historical_fn_marginal_check": output[
            "posthoc_historical_fn_marginal_check"
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
