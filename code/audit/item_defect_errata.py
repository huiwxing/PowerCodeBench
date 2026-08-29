#!/usr/bin/env python3
"""Build the checkable errata record for the 38 item-defect rulings.

The engineering-validity audit ruled 38 of its 610 inspected cases
``item-defect``: the benchmark item itself, not the model output, was at
fault.  Those 38 audit records resolve to 31 distinct benchmark items.  The
audit already drops them from every false-acceptance denominator, but the
frozen 2,000-item accuracy tables still contain them, so this script states
exactly which items they are and how much the archived numbers would move if
they were removed.

Two joins matter and are both done explicitly:

* The ``model|condition|bench_index`` key of ``t4_final_labels_v2.json`` is
  frame-local -- ``bench_index`` numbers positions inside the sampling frame,
  not inside the benchmark.  The canonical ``item_id`` is therefore recovered
  through ``e1_sample_manifest.json`` on ``(frame, model, condition,
  bench_index)``.
* Character *i* of every ``match_bits`` string in the compact primary archive
  is the outcome for ``item_ids[i]``, lexicographic by ``item_id``.  The
  exclusion mask is built on that ordering.

The sensitivity section recomputes every archived full-suite accuracy cell
(226 cells over 19 conditions and 15 models) on the reduced 1,969-item
denominator, and re-runs the Table 3(b) paired bootstrap and TOST contrasts
under both denominators using the settings frozen in
``code/aggregate/aggregate_p02_parity.py`` (10,000 paired draws, seed
20260617, seed reset independently per contrast, equivalence margin +/-5pp
decided by the two-sided 90% percentile interval).

Deterministic and standard-library only.  Run from the repository root::

    python3 code/audit/item_defect_errata.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "audit/item_defect_errata.json"

CREATED = "2026-08-29"
DEFECT_LABEL = "item-defect"
SUITE = "powercodebench_2000"
HOLDOUT = "naturalistic_holdout_80"
QUERY_PREFIX_CHARS = 80

LABELS = "audit/t4_final_labels_v2.json"
MANIFEST = "audit/e1_sample_manifest.json"
BENCHMARK = "benchmark.json"
PRIMARY = "results/raw/main_experiment/primary_outcomes_compact.json"
PARITY = "audit/p02_parity_item_outcomes.json"
SOURCES = (LABELS, MANIFEST, BENCHMARK, PRIMARY, PARITY)

# Frozen bootstrap contract, mirrored from code/aggregate/aggregate_p02_parity.py.
N_BOOTSTRAP = 10000
SEED = 20260617
TOST_MARGIN_PP = 5.0
CONTRASTS = (
    ("llama70_vs_nearest_deepseek", "Llama-3.1-70B", "DeepSeek-V4-Flash", "nearest API"),
    ("llama70_vs_strongest_claude", "Llama-3.1-70B", "Claude-Haiku-4-5", "strongest API"),
    ("gptoss120_vs_nearest_gpt54mini", "GPT-OSS-120B", "GPT-5.4-mini", "nearest API"),
    ("gptoss120_vs_strongest_claude", "GPT-OSS-120B", "Claude-Haiku-4-5", "strongest API"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def item_id_index(manifest: dict) -> dict:
    """Map (frame, model, condition, bench_index) -> canonical item_id."""
    index = {}
    for frame, rows in manifest["samples"].items():
        for row in rows:
            key = (frame, row["model"], row["condition"], row["bench_index"])
            if key in index and index[key] != row["item_id"]:
                raise ValueError(f"manifest pointer is ambiguous: {key}")
            index[key] = row["item_id"]
    return index


def defect_records(labels: dict, index: dict) -> list[dict]:
    records = []
    for record_key, row in sorted(labels.items()):
        if row["label"] != DEFECT_LABEL:
            continue
        model, condition, bench_index = record_key.split("|")
        pointer = (row["frame"], model, condition, int(bench_index))
        if pointer not in index:
            raise ValueError(f"no manifest pointer for audit record {record_key}")
        records.append({
            "bench_index": int(bench_index),
            "condition": condition,
            "frame": row["frame"],
            "item_id": index[pointer],
            "model": model,
            "record_key": record_key,
        })
    return records


def percentile(sorted_values: list[float], probability: float) -> float:
    """NumPy-compatible linear percentile on an already sorted sequence."""
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_bootstrap(treatment_bits: str, reference_bits: str) -> dict:
    deltas = [int(a) - int(b) for a, b in zip(treatment_bits, reference_bits)]
    n_pairs = len(deltas)
    rng = random.Random(SEED)
    draws = sorted(
        100.0 * sum(rng.choices(deltas, k=n_pairs)) / n_pairs
        for _ in range(N_BOOTSTRAP)
    )
    ci90 = [percentile(draws, 0.05), percentile(draws, 0.95)]
    ci95 = [percentile(draws, 0.025), percentile(draws, 0.975)]
    return {
        "ci90_pp": [round(value, 2) for value in ci90],
        "ci95_includes_zero": ci95[0] <= 0.0 <= ci95[1],
        "ci95_pp": [round(value, 2) for value in ci95],
        "n_pairs": n_pairs,
        "point_pp": round(100.0 * sum(deltas) / n_pairs, 2),
        "tost_equivalent_alpha_0_05": ci90[0] > -TOST_MARGIN_PP and ci90[1] < TOST_MARGIN_PP,
    }


def cell_sensitivity(primary: dict, defect_items: set) -> tuple[list[dict], dict]:
    item_ids = primary["datasets"][SUITE]["item_ids"]
    keep = [i for i, item_id in enumerate(item_ids) if item_id not in defect_items]
    n_full, n_excl = len(item_ids), len(keep)
    cells = []
    for row in primary["runs"].values():
        if row["dataset"] != SUITE:
            continue
        bits = row["match_bits"]
        if len(bits) != n_full:
            raise ValueError(f"match_bits length != {n_full} on a full-suite run")
        acc_full = 100.0 * bits.count("1") / n_full
        acc_excl = 100.0 * sum(1 for i in keep if bits[i] == "1") / n_excl
        cells.append({
            "acc_excl": round(acc_excl, 3),
            "acc_full": round(acc_full, 3),
            "condition": row["condition"],
            "delta_pp": round(acc_excl - acc_full, 3),
            "model": row["model"],
        })
    cells.sort(key=lambda cell: (cell["condition"], cell["model"]))
    worst = max(cells, key=lambda cell: (abs(cell["delta_pp"]), cell["condition"], cell["model"]))
    summary = {
        "cell": {"condition": worst["condition"], "model": worst["model"]},
        "acc_excl": worst["acc_excl"],
        "acc_full": worst["acc_full"],
        "delta_pp": worst["delta_pp"],
        "max_abs_delta_pp": abs(worst["delta_pp"]),
    }
    return cells, summary


def parity_sensitivity(parity: dict, defect_items: set) -> dict:
    item_ids = parity["item_ids"]
    keep = [i for i, item_id in enumerate(item_ids) if item_id not in defect_items]
    vectors = {
        model: row["match_bits"] for model, row in parity["models"].items()
    }
    reduced = {
        model: "".join(bits[i] for i in keep) for model, bits in vectors.items()
    }
    contrasts = {}
    unchanged = True
    for key, treatment, reference, role in CONTRASTS:
        full = paired_bootstrap(vectors[treatment], vectors[reference])
        excl = paired_bootstrap(reduced[treatment], reduced[reference])
        same = (full["tost_equivalent_alpha_0_05"]
                == excl["tost_equivalent_alpha_0_05"])
        unchanged = unchanged and same
        contrasts[key] = {
            "decision_unchanged": same,
            "excluding_item_defects": excl,
            "full_suite": full,
            "reference": reference,
            "reference_role": role,
            "treatment": treatment,
        }
    return {
        "all_tost_decisions_unchanged": unchanged,
        "contrasts": contrasts,
        "method": {
            "condition": "condC_FDRS",
            "item_order": "lexicographic item_id",
            "n_bootstrap": N_BOOTSTRAP,
            "pairing_key": "item_id",
            "percentile_interpolation": "linear at (B-1)*p",
            "resampling": "paired items with replacement",
            "rng": "Python random.Random; seed reset independently per contrast",
            "seed": SEED,
            "settings_from": "code/aggregate/aggregate_p02_parity.py",
            "tost": (
                "alpha=0.05, equivalence margin +/-5pp; equivalent iff the "
                "two-sided 90% percentile interval lies strictly inside the margin"
            ),
            "tost_margin_pp": TOST_MARGIN_PP,
        },
    }


def build() -> dict:
    labels = load(LABELS)
    manifest = load(MANIFEST)
    benchmark = load(BENCHMARK)
    primary = load(PRIMARY)
    parity = load(PARITY)

    records = defect_records(labels, item_id_index(manifest))
    defect_items = {record["item_id"] for record in records}

    queries = {item["id"]: item["natural_language_query"] for item in benchmark}
    missing = sorted(defect_items - set(queries))
    if missing:
        raise ValueError(f"item_ids absent from {BENCHMARK}: {missing}")

    suite_ids = set(primary["datasets"][SUITE]["item_ids"])
    if not defect_items <= suite_ids:
        raise ValueError("an item-defect item is outside the frozen 2,000-item suite")
    holdout = set(primary["datasets"][HOLDOUT]["item_ids"])
    if defect_items & holdout:
        raise ValueError("an item-defect item is on the naturalistic holdout frame")
    if parity["item_ids"] != primary["datasets"][SUITE]["item_ids"]:
        raise ValueError("the parity and primary archives disagree on item order")

    cells, worst = cell_sensitivity(primary, defect_items)

    return {
        "created": CREATED,
        "description": (
            "Errata record for the 38 audit cases ruled item-defect in "
            "audit/t4_final_labels_v2.json. Lists the affected audit records, "
            "resolves them to the 31 distinct benchmark items behind them, and "
            "recomputes every archived full-suite accuracy cell and the Table "
            "3(b) equivalence contrasts with those 31 items excluded. The "
            "rulings are already excluded from every false-acceptance "
            "denominator; the frozen accuracy tables are unchanged and this "
            "file quantifies what excluding the items would do to them."
        ),
        "generated_by": "code/audit/item_defect_errata.py",
        "generated_from": {
            path: sha256_file(ROOT / path) for path in sorted(SOURCES)
        },
        "join": {
            "audit_record_to_item_id": (
                "e1_sample_manifest.json samples[frame][*] on "
                "(frame, model, condition, bench_index); the bench_index in an "
                "audit record key is frame-local, not a benchmark position"
            ),
            "item_id_to_match_bits": (
                "character i of match_bits is the outcome for "
                "datasets.powercodebench_2000.item_ids[i] (lexicographic item_id)"
            ),
        },
        "n_records": len(records),
        "n_unique_items": len(defect_items),
        "records": records,
        "sensitivity": {
            "accuracy_cells": {
                "cells": cells,
                "denominator_excl": 2000 - len(defect_items),
                "denominator_full": 2000,
                "largest_absolute_shift": worst,
                "n_cells": len(cells),
                "n_conditions": len({cell["condition"] for cell in cells}),
                "n_models": len({cell["model"] for cell in cells}),
                "note": (
                    "Every archived accuracy cell on the 2,000-item suite. The "
                    "44 naturalistic-holdout runs are on an 80-item frame and "
                    "carry no item-defect items, so they are not listed."
                ),
                "scope": "results/raw/main_experiment/primary_outcomes_compact.json runs "
                         "on dataset powercodebench_2000",
                "units": "percentage points",
            },
            "table3b_equivalence": parity_sensitivity(parity, defect_items),
        },
        "unique_items": [
            {
                "item_id": item_id,
                "natural_language_query_prefix": queries[item_id][:QUERY_PREFIX_CHARS],
            }
            for item_id in sorted(defect_items)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
