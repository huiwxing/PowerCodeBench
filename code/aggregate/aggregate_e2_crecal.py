#!/usr/bin/env python3
"""Recompute the OpenDSS C-recal evidence from frozen per-item results.

This is a CPU-only, Python-standard-library aggregation.  It checks each raw
file against the hash recorded in its migration manifest, aligns all
conditions by item ID, and recomputes:

* C-recal endpoints;
* paired C-recal--C and C-recal--Rsem accuracy contrasts;
* fixed-seed paired-bootstrap confidence intervals and exact McNemar p-values;
* total prompt-token ratios; and
* control-device non-execution/error counts used by the mechanism analysis.

The bootstrap pins the RNG implementation and the order statistics.  Each
contrast resets ``random.Random(20260727)`` and resamples the C-recal item
order 10,000 times, which keeps the archived result stable across runs
without NumPy or SciPy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "results/raw/e2_transfer/manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "results/aggregates/e2_crecal_paired.json"
N_BOOT = 10_000
SEED = 20_260_727


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifacts(repo_root: Path, manifest: dict) -> None:
    for artifact in manifest["artifacts"]:
        path = repo_root / artifact["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != artifact["bytes"]:
            raise ValueError(
                f"byte-size mismatch for {artifact['path']}: "
                f"expected {artifact['bytes']}, got {path.stat().st_size}"
            )
        got_hash = sha256(path)
        if got_hash != artifact["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {artifact['path']}: "
                f"expected {artifact['sha256']}, got {got_hash}"
            )


def validate_result(result: dict, label: str) -> None:
    items = result["item_results"]
    summary = result["summary"]
    ids = [item["item_id"] for item in items]
    if len(items) != 90 or summary["total"] != 90:
        raise ValueError(f"{label}: expected 90 items")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label}: duplicate item IDs")
    n_match = sum(bool(item.get("match")) for item in items)
    n_exec = sum(bool(item.get("executed")) for item in items)
    prompt_total = sum(int(item["prompt_tokens"]) for item in items)
    if n_match != summary["n_matched"]:
        raise ValueError(f"{label}: item/summary match-count mismatch")
    if n_exec != summary["n_executed"]:
        raise ValueError(f"{label}: item/summary execution-count mismatch")
    if prompt_total != summary["prompt_token_stats"]["total"]:
        raise ValueError(f"{label}: item/summary prompt-token mismatch")


def item_map(result: dict) -> dict[str, dict]:
    return {item["item_id"]: item for item in result["item_results"]}


def paired_bootstrap(treatment: dict, reference: dict) -> dict:
    """Paired percentile bootstrap for accuracy(treatment)-accuracy(reference)."""
    treatment_items = treatment["item_results"]
    reference_by_id = item_map(reference)
    ids = [item["item_id"] for item in treatment_items]
    if set(ids) != set(reference_by_id):
        raise ValueError("paired conditions do not contain the same item IDs")
    diffs = [
        int(bool(item.get("match")))
        - int(bool(reference_by_id[item["item_id"]].get("match")))
        for item in treatment_items
    ]
    n_pairs = len(diffs)
    rng = random.Random(SEED)
    bootstrap = sorted(
        sum(diffs[rng.randrange(n_pairs)] for _ in range(n_pairs)) / n_pairs
        for _ in range(N_BOOT)
    )
    n_positive = sum(diff > 0 for diff in diffs)
    n_negative = sum(diff < 0 for diff in diffs)
    discordant = n_positive + n_negative
    if discordant:
        tail = sum(
            math.comb(discordant, i)
            for i in range(min(n_positive, n_negative) + 1)
        ) / (2 ** discordant)
        mcnemar_p = min(1.0, 2.0 * tail)
    else:
        mcnemar_p = 1.0
    return {
        "n_pairs": n_pairs,
        "delta": round(sum(diffs) / n_pairs, 4),
        "delta_pp": round(100 * sum(diffs) / n_pairs, 2),
        "ci95": [
            round(bootstrap[int(0.025 * N_BOOT)], 4),
            round(bootstrap[int(0.975 * N_BOOT) - 1], 4),
        ],
        "ci95_pp": [
            round(100 * bootstrap[int(0.025 * N_BOOT)], 2),
            round(100 * bootstrap[int(0.975 * N_BOOT) - 1], 2),
        ],
        "discordant_treatment_only": n_positive,
        "discordant_reference_only": n_negative,
        "mcnemar_exact_p": mcnemar_p,
    }


def prompt_tokens(result: dict) -> dict:
    total = sum(int(item["prompt_tokens"]) for item in result["item_results"])
    return {
        "total": total,
        "mean": round(total / len(result["item_results"]), 4),
    }


def control_device_counts(result: dict) -> dict:
    items = [
        item for item in result["item_results"]
        if item.get("task") == "control_device"
    ]
    non_executable = [item for item in items if not item.get("executed")]
    return {
        "n": len(items),
        "executed": sum(bool(item.get("executed")) for item in items),
        "non_executable": len(non_executable),
        "matched": sum(bool(item.get("match")) for item in items),
        "non_executable_error_types": dict(sorted(Counter(
            item.get("error_type") or "unspecified" for item in non_executable
        ).items())),
    }


def aggregate(repo_root: Path, manifest_path: Path) -> dict:
    manifest = read_json(manifest_path)
    validate_artifacts(repo_root, manifest)
    artifacts = {
        (row["model"], row["arm"]): repo_root / row["path"]
        for row in manifest["artifacts"]
    }
    report = {
        "schema_version": "1.0",
        "source_manifest": str(manifest_path.relative_to(repo_root)),
        "method": {
            "n_boot": N_BOOT,
            "seed": SEED,
            "rng": "Python random.Random (MT19937), reset for each contrast",
            "pair_order": "C-recal item_results order",
            "ci": "percentile order statistics at sorted indices 250 and 9749",
            "mcnemar": "two-sided exact binomial test on discordant pairs",
            "token_ratio": "C-recal total prompt tokens / Rsem total prompt tokens",
        },
        "recal_name_layer_floor": manifest["recalibration_arm"][
            "recal_name_layer_floor"
        ],
        "models": {},
    }
    for model in manifest["recalibration_arm"]["models"]:
        arms = {
            "C": read_json(artifacts[(model, "C")]),
            "Rsem": read_json(artifacts[(model, "Rsem")]),
            "C-recal": read_json(artifacts[(model, "C-recal")]),
        }
        for arm, result in arms.items():
            validate_result(result, f"{model}/{arm}")
        id_sets = {arm: set(item_map(result)) for arm, result in arms.items()}
        if len({frozenset(ids) for ids in id_sets.values()}) != 1:
            raise ValueError(f"{model}: arm item-ID sets differ")

        token_stats = {arm: prompt_tokens(result) for arm, result in arms.items()}
        recal_total = token_stats["C-recal"]["total"]
        rsem_total = token_stats["Rsem"]["total"]
        report["models"][model] = {
            "n": 90,
            "accuracy": {
                arm: round(
                    sum(bool(item.get("match")) for item in result["item_results"])
                    / len(result["item_results"]),
                    4,
                )
                for arm, result in arms.items()
            },
            "paired": {
                "C-recal-C": paired_bootstrap(arms["C-recal"], arms["C"]),
                "C-recal-Rsem": paired_bootstrap(
                    arms["C-recal"], arms["Rsem"]
                ),
            },
            "prompt_tokens": token_stats,
            "crecal_over_rsem_total_prompt_ratio": round(
                recal_total / rsem_total, 12
            ),
            "control_device": {
                arm: control_device_counts(result) for arm, result in arms.items()
            },
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest.resolve()
    output = args.out.resolve()
    report = aggregate(repo_root, manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[saved] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
