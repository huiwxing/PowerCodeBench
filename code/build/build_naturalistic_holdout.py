#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Repository copy of scripts/build/build_naturalistic_holdout.py, with imports
# and data-path constants adapted to this repository's layout. Runs CPU-only
# against the archived corpus/spec files in this repository. See
# code/README.md for the module map.
# --------------------------------------------------------------------------
"""Build a naturalistic-query holdout benchmark JSON from a reviewed CSV."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[2]
FINAL_STATUSES = {"accepted", "final", "reviewed"}


def _read_rows(path: Path) -> List[Dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_answer_key(path: Path) -> Dict[str, Dict]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return {item["id"]: item for item in payload.get("items", [])}


def _validate_references(items: List[Dict], timeout: int) -> Dict:
    """Optionally execute reference code and compare against stored GT."""
    import sys

    code_root = ROOT / "code"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from backend.utils import execute_code_safely, match_ground_truth  # noqa: WPS433

    failures = []
    for item in items:
        result = execute_code_safely(item["reference_code"], timeout=timeout)
        if not result.get("success"):
            failures.append({
                "id": item["id"],
                "type": "execution",
                "error_type": result.get("error_type"),
                "error": result.get("error"),
            })
            continue
        match = match_ground_truth(
            result.get("output", ""),
            item.get("ground_truth"),
            item.get("ground_truth_type", "float"),
        )
        if not match.get("match"):
            failures.append({
                "id": item["id"],
                "type": "match",
                "detail": match,
            })
    return {"ok": not failures, "failures": failures}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--authoring-csv", required=True)
    ap.add_argument("--answer-key", required=True)
    ap.add_argument("--output", default="benchmark/naturalistic_holdout/naturalistic_holdout.json")
    ap.add_argument(
        "--include-draft",
        action="store_true",
        help="Include rows whose status is not accepted/final/reviewed. Use only for dry runs.",
    )
    ap.add_argument(
        "--validate-reference",
        action="store_true",
        help="Execute each reference program and compare against stored ground truth.",
    )
    ap.add_argument("--validation-timeout", type=int, default=60)
    args = ap.parse_args()

    rows = _read_rows((ROOT / args.authoring_csv).resolve())
    source_by_id = _load_answer_key((ROOT / args.answer_key).resolve())

    items = []
    skipped = []
    for row in rows:
        status = (row.get("status") or "").strip().lower()
        if not args.include_draft and status not in FINAL_STATUSES:
            skipped.append((row.get("holdout_id"), "status"))
            continue
        query = (row.get("naturalistic_query") or "").strip()
        if not query:
            skipped.append((row.get("holdout_id"), "empty_query"))
            continue
        source_id = (row.get("source_id") or "").strip()
        source = source_by_id.get(source_id)
        if source is None:
            skipped.append((row.get("holdout_id"), "missing_source"))
            continue

        item = copy.deepcopy(source)
        item["id"] = (row.get("holdout_id") or source_id).strip()
        item["natural_language_query"] = query
        item["holdout_metadata"] = {
            "source_benchmark_id": source_id,
            "query_origin": "human-authored naturalistic query",
            "author": row.get("author", ""),
            "reviewer": row.get("reviewer", ""),
            "review_status": row.get("status", ""),
            "notes": row.get("notes", ""),
        }
        items.append(item)

    if not items:
        raise SystemExit(
            "No holdout items were built. Fill naturalistic_query and set "
            "status to accepted/final/reviewed, or pass --include-draft for a dry run."
        )

    validation = None
    if args.validate_reference:
        validation = _validate_references(items, timeout=args.validation_timeout)
        if not validation["ok"]:
            print(json.dumps(validation, indent=2))
            raise SystemExit("Reference validation failed; not writing holdout JSON.")

    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    manifest = output.with_suffix(".manifest.json")
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output": str(output),
                "n_items": len(items),
                "skipped": skipped,
                "validation": validation,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Wrote holdout benchmark: {output}")
    print(f"Wrote manifest: {manifest}")
    if skipped:
        print(f"Skipped {len(skipped)} row(s); see manifest for reasons.")


if __name__ == "__main__":
    main()
