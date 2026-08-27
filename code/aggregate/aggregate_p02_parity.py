#!/usr/bin/env python3
"""Recompute the P02 C+FDRS parity contrasts from compact item outcomes.

The default path is CPU-only, standard-library-only, and reads the frozen
compact artifact in ``audit/p02_parity_item_outcomes.json``.  The optional
``--export-source-root`` mode rebuilds that compact artifact from the five
original Git-LFS result files in the pre-release ``igpt`` repository.

Bootstrap determinism is part of the artifact contract: item IDs are sorted
lexicographically, Python's ``random.Random`` is used, and the documented seed
is reset independently for each contrast.  TOST equivalence at alpha=0.05 is
implemented by the equivalent two-sided 90% confidence-interval criterion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPACT = ROOT / "audit/p02_parity_item_outcomes.json"
DEFAULT_OUTPUT = ROOT / "results/aggregates/p02_parity_tost.json"

SOURCE_COMMIT = "0ca5647c6a6ef3376763e907f12b245316613d97"
REFERENCE_SUMMARY = {
    # The pre-submission working record this aggregate reproduces. It is not
    # publicly distributed, so it is identified by content hash rather than by
    # a repository/commit pair no reader could resolve.
    "repository": "authors' pre-submission working records (not distributed)",
    "path": "P02_parity_tost.json",
    "sha256": "cf33c7feaa15d95f540e8c4e949868e3fd50f67a096365e6b5a06d8e86d2d959",
}

MODEL_SOURCES = {
    "Llama-3.1-70B": {
        "display_name": "Llama-3.1-70B",
        "path": (
            "probe_eval_results/comparison/"
            "meta-llama_Llama-3.1-70B-Instruct/"
            "benchmark_results_condC_FDRS.json"
        ),
        "raw_sha256": "0e72dc3eed203529a00126c917801102fb4cf439549b8d94a2ad15c9b182c52e",
        "raw_size_bytes": 20924817,
    },
    "GPT-OSS-120B": {
        "display_name": "GPT-OSS-120B",
        "path": (
            "probe_eval_results/comparison/openai_gpt-oss-120b/"
            "benchmark_results_condC_FDRS.json"
        ),
        "raw_sha256": "c2dd175ae081ff998abdb48cde26abf1c383221963fa67241e31141681731005",
        "raw_size_bytes": 22592081,
    },
    "DeepSeek-V4-Flash": {
        "display_name": "DeepSeek-V4-Flash",
        "path": (
            "probe_eval_results/api_comparison_2000/deepseek-v4-flash/"
            "benchmark_results_condC_FDRS.json"
        ),
        "raw_sha256": "19f0821b050383c106b8982149ac04dd0b907e6574fdbeb20cce6ce928c51a4f",
        "raw_size_bytes": 9781629,
    },
    "GPT-5.4-mini": {
        "display_name": "GPT-5.4-mini",
        "path": (
            "probe_eval_results/api_comparison_2000/gpt-5.4-mini/"
            "benchmark_results_condC_FDRS.json"
        ),
        "raw_sha256": "f8394e1933a20e2b20082c0c0fc808228fe8d9bac7807d6425b2c1b90743ebce",
        "raw_size_bytes": 8752164,
    },
    "Claude-Haiku-4-5": {
        "display_name": "Claude-Haiku-4-5",
        "path": (
            "probe_eval_results/api_comparison_2000/claude-haiku-4-5/"
            "benchmark_results_condC_FDRS.json"
        ),
        "raw_sha256": "45f6e9bc5a2a0c6f9696843c8b9b04ac626b6e54cc7e47624db6342133928a2d",
        "raw_size_bytes": 9927456,
    },
}

CONTRASTS = (
    ("llama70_vs_nearest_deepseek", "Llama-3.1-70B", "DeepSeek-V4-Flash", "nearest API"),
    ("llama70_vs_strongest_claude", "Llama-3.1-70B", "Claude-Haiku-4-5", "strongest API"),
    ("gptoss120_vs_nearest_gpt54mini", "GPT-OSS-120B", "GPT-5.4-mini", "nearest API"),
    ("gptoss120_vs_strongest_claude", "GPT-OSS-120B", "Claude-Haiku-4-5", "strongest API"),
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector_hash(item_ids: list[str], bits: str) -> str:
    payload = "".join(
        f"{item_id}\t{bit}\n" for item_id, bit in zip(item_ids, bits)
    ).encode("utf-8")
    return sha256_bytes(payload)


def export_compact(source_root: Path) -> dict:
    """Extract only item IDs and match outcomes from the five source runs."""
    raw_rows = {}
    source_metadata = {}
    for model, spec in MODEL_SOURCES.items():
        path = source_root / spec["path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing source result: {path}")
        got_hash = sha256_file(path)
        got_size = path.stat().st_size
        if got_hash != spec["raw_sha256"] or got_size != spec["raw_size_bytes"]:
            raise ValueError(
                f"source provenance mismatch for {model}: "
                f"sha256={got_hash}, size={got_size}"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("item_results")
        if not isinstance(rows, list) or len(rows) != 2000:
            raise ValueError(f"{model}: expected 2,000 item_results")
        by_id = {}
        for row in rows:
            item_id = row.get("item_id")
            match = row.get("match")
            if not isinstance(item_id, str) or not isinstance(match, bool):
                raise ValueError(f"{model}: invalid item_id/match row")
            if item_id in by_id:
                raise ValueError(f"{model}: duplicate item_id {item_id}")
            by_id[item_id] = match
        raw_rows[model] = by_id
        source_metadata[model] = {
            "repository": "frozen experimental pipeline",
            "commit": SOURCE_COMMIT,
            "path": spec["path"],
            "git_lfs_oid_sha256": spec["raw_sha256"],
            "raw_size_bytes": spec["raw_size_bytes"],
        }

    item_sets = [set(rows) for rows in raw_rows.values()]
    if any(items != item_sets[0] for items in item_sets[1:]):
        raise ValueError("the five source runs do not have identical item_id sets")
    item_ids = sorted(item_sets[0])
    item_ids_sha256 = sha256_bytes(
        "".join(f"{item_id}\n" for item_id in item_ids).encode("utf-8")
    )

    models = {}
    for model in MODEL_SOURCES:
        bits = "".join("1" if raw_rows[model][item_id] else "0" for item_id in item_ids)
        models[model] = {
            "display_name": MODEL_SOURCES[model]["display_name"],
            "match_bits": bits,
            "n_match": bits.count("1"),
            "item_outcome_sha256": vector_hash(item_ids, bits),
            "source": source_metadata[model],
        }

    return {
        "schema_version": 1,
        "description": (
            "Compact item-level C+FDRS outcomes for the P02 paired-bootstrap "
            "and equivalence contrasts. Character i of each match_bits string "
            "is the 0/1 match outcome for item_ids[i]."
        ),
        "condition": "condC_FDRS",
        "n_items": len(item_ids),
        "item_order": "lexicographic item_id",
        "item_ids_sha256": item_ids_sha256,
        "item_ids": item_ids,
        "models": models,
        "provenance": {
            "source_repository_commit": SOURCE_COMMIT,
            "source_files_are_git_lfs_objects": True,
            "legacy_p02_summary": REFERENCE_SUMMARY,
        },
    }


def validate_compact(data: dict) -> tuple[list[str], dict[str, str]]:
    if data.get("schema_version") != 1:
        raise ValueError("unsupported compact-artifact schema")
    if data.get("condition") != "condC_FDRS":
        raise ValueError("compact artifact is not the condC_FDRS endpoint")
    item_ids = data.get("item_ids")
    if not isinstance(item_ids, list) or item_ids != sorted(item_ids):
        raise ValueError("item_ids must be a lexicographically sorted list")
    if len(item_ids) != 2000 or len(set(item_ids)) != 2000:
        raise ValueError("compact artifact must contain 2,000 unique item IDs")
    if data.get("n_items") != len(item_ids):
        raise ValueError("n_items disagrees with item_ids")
    got_ids_hash = sha256_bytes(
        "".join(f"{item_id}\n" for item_id in item_ids).encode("utf-8")
    )
    if got_ids_hash != data.get("item_ids_sha256"):
        raise ValueError("item_ids_sha256 mismatch")

    expected_models = set(MODEL_SOURCES)
    if set(data.get("models", {})) != expected_models:
        raise ValueError("compact artifact does not contain the five P02 models")
    vectors = {}
    for model, row in data["models"].items():
        bits = row.get("match_bits")
        if not isinstance(bits, str) or len(bits) != len(item_ids):
            raise ValueError(f"{model}: invalid match_bits length")
        if set(bits) - {"0", "1"}:
            raise ValueError(f"{model}: match_bits is not binary")
        if bits.count("1") != row.get("n_match"):
            raise ValueError(f"{model}: n_match disagrees with match_bits")
        if vector_hash(item_ids, bits) != row.get("item_outcome_sha256"):
            raise ValueError(f"{model}: item_outcome_sha256 mismatch")
        source = row.get("source", {})
        spec = MODEL_SOURCES[model]
        if (
            source.get("commit") != SOURCE_COMMIT
            or source.get("path") != spec["path"]
            or source.get("git_lfs_oid_sha256") != spec["raw_sha256"]
            or source.get("raw_size_bytes") != spec["raw_size_bytes"]
        ):
            raise ValueError(f"{model}: source provenance mismatch")
        vectors[model] = bits
    return item_ids, vectors


def percentile(sorted_values: list[float], probability: float) -> float:
    """NumPy-compatible linear percentile on an already sorted sequence."""
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_bootstrap(
    treatment_bits: str,
    reference_bits: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict:
    deltas = [int(a) - int(b) for a, b in zip(treatment_bits, reference_bits)]
    n_pairs = len(deltas)
    rng = random.Random(seed)
    bootstrap_pp = sorted(
        100.0 * sum(rng.choices(deltas, k=n_pairs)) / n_pairs
        for _ in range(n_bootstrap)
    )
    ci90 = [percentile(bootstrap_pp, 0.05), percentile(bootstrap_pp, 0.95)]
    ci95 = [percentile(bootstrap_pp, 0.025), percentile(bootstrap_pp, 0.975)]
    point_pp = 100.0 * sum(deltas) / n_pairs
    margin_pp = 5.0
    return {
        "n_pairs": n_pairs,
        "point_pp": round(point_pp, 2),
        "discordant_treatment_only": deltas.count(1),
        "discordant_reference_only": deltas.count(-1),
        "ci90_pp": [round(value, 2) for value in ci90],
        "ci95_pp": [round(value, 2) for value in ci95],
        "ci95_includes_zero": ci95[0] <= 0.0 <= ci95[1],
        "tost_margin_pp": margin_pp,
        "tost_equivalent_alpha_0_05": ci90[0] > -margin_pp and ci90[1] < margin_pp,
    }


def aggregate(data: dict, compact_path: Path) -> dict:
    item_ids, vectors = validate_compact(data)
    n_bootstrap = 10000
    seed = 20260617
    comparisons = {}
    for key, treatment, reference, reference_role in CONTRASTS:
        row = paired_bootstrap(
            vectors[treatment], vectors[reference],
            n_bootstrap=n_bootstrap, seed=seed,
        )
        comparisons[key] = {
            "treatment": treatment,
            "reference": reference,
            "reference_role": reference_role,
            **row,
        }

    models = {
        model: {
            "n": len(item_ids),
            "n_match": bits.count("1"),
            "accuracy_pct": round(100.0 * bits.count("1") / len(item_ids), 2),
        }
        for model, bits in vectors.items()
    }
    return {
        "schema_version": 1,
        "method": {
            "condition": "condC_FDRS",
            "pairing_key": "item_id",
            "n_common_items": len(item_ids),
            "item_order": "lexicographic item_id",
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "rng": "Python random.Random; seed reset independently per contrast",
            "resampling": "paired items with replacement",
            "percentile_interpolation": "linear at (B-1)*p",
            "ci95": "two-sided percentile interval",
            "tost": (
                "alpha=0.05, equivalence margin +/-5pp; equivalent iff the "
                "two-sided 90% percentile interval lies strictly inside the margin"
            ),
        },
        "source": {
            "compact_artifact": str(compact_path.relative_to(ROOT)),
            "compact_artifact_sha256": sha256_file(compact_path),
            "source_repository_commit": SOURCE_COMMIT,
            "legacy_p02_summary": REFERENCE_SUMMARY,
        },
        "models": models,
        "comparisons": comparisons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--export-source-root", type=Path,
        help="rebuild --compact from the original igpt repository before aggregating",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compact_path = args.compact.resolve()
    if args.export_source_root is not None:
        data = export_compact(args.export_source_root.resolve())
        compact_path.parent.mkdir(parents=True, exist_ok=True)
        compact_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    else:
        data = json.loads(compact_path.read_text(encoding="utf-8"))
    result = aggregate(data, compact_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
