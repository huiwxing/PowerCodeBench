#!/usr/bin/env python3
"""Rebuild the complete E2 transfer evidence from repository-local raw data.

Inputs are the 24 frozen cold-start cells (two backends x three models x
A/C/Rsem/RsemB), the three OpenDSS C-recal cells, and their migration manifest
under ``results/raw/e2_transfer``.  The script uses only Python's standard
library and paths inside this release repository.  It validates every raw cell
against the source SHA-256 recorded at migration, then rebuilds cell summaries,
paired item-level bootstrap contrasts, prompt-token accounting, and the full
C-recal report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aggregate_e2_bench import CONDITIONS, item_vectors, paired_bootstrap, summarize
from aggregate_e2_crecal import aggregate as aggregate_crecal
from aggregate_e2_crecal import read_json, sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "results/raw/e2_transfer/manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "results/aggregates/e2_transfer_full_recomputed.json"
N_BOOT = 10_000
SEED = 20_260_727


def validate_file(path: Path, metadata: dict, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != metadata["bytes"]:
        raise ValueError(
            f"{label}: expected {metadata['bytes']} bytes, "
            f"got {path.stat().st_size}"
        )
    got_hash = sha256(path)
    if got_hash != metadata["sha256"]:
        raise ValueError(
            f"{label}: expected SHA-256 {metadata['sha256']}, got {got_hash}"
        )


def validate_result(result: dict, label: str) -> None:
    items = result["item_results"]
    summary = result["summary"]
    ids = [item["item_id"] for item in items]
    if len(items) != 90 or summary["total"] != 90:
        raise ValueError(f"{label}: expected 90 items")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}: duplicate item IDs")
    if sum(bool(item.get("match")) for item in items) != summary["n_matched"]:
        raise ValueError(f"{label}: item/summary match-count mismatch")
    if sum(bool(item.get("executed")) for item in items) != summary["n_executed"]:
        raise ValueError(f"{label}: item/summary execution-count mismatch")
    if sum(int(item["prompt_tokens"]) for item in items) != (
        summary["prompt_token_stats"]["total"]
    ):
        raise ValueError(f"{label}: item/summary prompt-token mismatch")


def load_cold_start(repo_root: Path, manifest: dict) -> dict:
    section = manifest["cold_start_cells"]
    cells = {}
    n_cells = 0
    for backend, models in section["files"].items():
        cells[backend] = {}
        for model, conditions in models.items():
            if set(conditions) != set(CONDITIONS):
                raise ValueError(f"{backend}/{model}: incomplete condition set")
            cells[backend][model] = {}
            for condition, metadata in conditions.items():
                relative = section["release_path_pattern"].format(
                    backend=backend, model=model, condition=condition
                )
                path = repo_root / relative
                validate_file(path, metadata, relative)
                result = read_json(path)
                validate_result(result, f"{backend}/{model}/{condition}")
                cells[backend][model][condition] = result
                n_cells += 1
            id_sets = {
                condition: frozenset(item_vectors(result))
                for condition, result in cells[backend][model].items()
            }
            if len(set(id_sets.values())) != 1:
                raise ValueError(f"{backend}/{model}: condition item-ID sets differ")
    if n_cells != 24:
        raise ValueError(f"expected 24 cold-start cells, found {n_cells}")
    return cells


def validate_released_logs(repo_root: Path, manifest: dict) -> None:
    for metadata in manifest["recalibration_arm"]["source_job_logs"]:
        path = repo_root / metadata["released_sanitized_path"]
        validate_file(
            path,
            {
                "bytes": metadata["released_bytes"],
                "sha256": metadata["released_sha256"],
            },
            metadata["released_sanitized_path"],
        )


def cold_start_report(cells: dict) -> dict:
    report = {
        "n_cells": 24,
        "n_items_per_cell": 90,
        "conditions": list(CONDITIONS),
        "n_boot": N_BOOT,
        "seed": SEED,
        "backends": {},
    }
    for backend in sorted(cells):
        backend_out = {
            "run_dir": f"results/raw/e2_transfer/{backend}/base",
            "models": {},
        }
        for model in sorted(cells[backend]):
            conditions = cells[backend][model]
            model_out = {
                "conditions": {
                    condition: summarize(result)
                    for condition, result in conditions.items()
                },
                "paired": {},
            }
            prompt_averages = {
                condition: result["summary"]["prompt_token_stats"]["avg"]
                for condition, result in conditions.items()
            }
            model_out["prompt_tokens_avg"] = prompt_averages
            bare_tokens = prompt_averages["A"]
            model_out["injected_tokens_avg"] = {
                condition: value - bare_tokens
                for condition, value in prompt_averages.items()
            }

            vectors = {
                condition: item_vectors(result)
                for condition, result in conditions.items()
            }
            for reference in ("A", "Rsem", "RsemB"):
                ids = [item_id for item_id in vectors["C"]
                       if item_id in vectors[reference]]
                if len(ids) != 90:
                    raise ValueError(
                        f"{backend}/{model}/C-{reference}: expected 90 pairs"
                    )
                for key, label in (("match", "accuracy"), ("executed", "exec")):
                    model_out["paired"][f"C-{reference}::{label}"] = paired_bootstrap(
                        ids, vectors["C"], vectors[reference], key, N_BOOT, SEED
                    )
                for difficulty in ("D1_basic", "D3_semantic"):
                    subset = [
                        item_id for item_id in ids
                        if vectors["C"][item_id]["difficulty"] == difficulty
                    ]
                    model_out["paired"][
                        f"C-{reference}::accuracy@{difficulty}"
                    ] = paired_bootstrap(
                        subset, vectors["C"], vectors[reference],
                        "match", N_BOOT, SEED
                    )
            backend_out["models"][model] = model_out
        report["backends"][backend] = backend_out
    return report


def aggregate(repo_root: Path, manifest_path: Path) -> dict:
    manifest = read_json(manifest_path)
    cells = load_cold_start(repo_root, manifest)
    validate_released_logs(repo_root, manifest)
    crecal = aggregate_crecal(repo_root, manifest_path)
    return {
        "schema_version": "1.0",
        "source_manifest": str(manifest_path.relative_to(repo_root)),
        "cold_start": cold_start_report(cells),
        "crecal": crecal,
    }


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
