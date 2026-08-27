#!/usr/bin/env python3
"""Export and verify the surviving historical C3/E4 execution logs.

The logs establish that the archived aggregate runs were executed and record
their seeds, fitted settings, and displayed endpoints.  They stop at the
aggregate level: the historical C3 manifests say ``prediction_export: null``,
and the E4 writer retained score maps only in memory before serialising
aggregate metrics.  ``results/supplementary_evidence/README.md`` sets out what
survived from those runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
EVIDENCE_DIR = ROOT / "results/supplementary_evidence/historical_demand_runs"
DEFAULT_OUTPUT = ROOT / "results/aggregates/historical_demand_run_logs_recomputed.json"

C3_RUNS = (
    "R1_baseline_eval_D12",
    "R2_baseline_eval_D34",
    "R3_reweight_D12_eval_D12",
    "R4_reweight_D12_eval_D34",
    "R5_reweight_D34_eval_D34",
    "R6_reweight_D34_eval_D12",
)
C3_CODE_COMMIT = "3841e89"
E4_CODE_COMMIT = "eb5f346"
E4_SEEDS = (42, 43, 44, 45, 46)
E4_SETS = (
    "layer1_holdout_n80",
    "layer2a_n84",
    "layer2a_ext_n10",
    "layer2b_n20",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def git_output(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def normalise_source_text(text: str, source_root: Path) -> tuple[str, int]:
    prefix = source_root.resolve().as_posix()
    count = text.count(prefix)
    return text.replace(prefix, "<SOURCE_REPOSITORY>"), count


def export_one(source_root: Path, relative: str, target_relative: str) -> dict:
    source = source_root / relative
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    released_text, replacements = normalise_source_text(text, source_root)
    released = released_text.encode("utf-8")
    target = EVIDENCE_DIR / target_relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(released)
    tracked = git_output(source_root, "ls-files", "-s", "--", relative)
    return {
        "release_path": target.relative_to(ROOT).as_posix(),
        "source_path": relative,
        "source_bytes": len(raw),
        "source_sha256": sha256_bytes(raw),
        "source_git_blob": tracked.split()[1] if tracked else None,
        "release_bytes": len(released),
        "release_sha256": sha256_bytes(released),
        "private_prefix_replacements": replacements,
    }


def export(source_root: Path) -> None:
    files = []
    for run in C3_RUNS:
        files.append(export_one(
            source_root,
            f"task_demand/results/c3_holdout/{run}/run.log",
            f"c3/{run}.log",
        ))
    files.append(export_one(
        source_root,
        "task_demand/logs/e4_p0_refix_recompute_20260802.log",
        "e4/e4_p0_refix_recompute_20260802.log",
    ))
    files.append(export_one(
        source_root,
        "tests/test_demand_determinism.py",
        "test_demand_determinism.py",
    ))
    manifest = {
        "schema_version": 1,
        "evidence_role": (
            "contemporaneous execution provenance for historical aggregate runs; "
            "not item-level prediction evidence"
        ),
        "historical_code_commits": {
            "c3_submit_and_writer": C3_CODE_COMMIT,
            "e4_writer_and_determinism_regression": E4_CODE_COMMIT,
        },
        "normalization": (
            "source-worktree absolute prefix replaced with <SOURCE_REPOSITORY>; "
            "all other bytes preserved"
        ),
        "files": files,
    }
    write_json(EVIDENCE_DIR / "manifest.json", manifest)


def parse_c3_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if '"prediction_export": null' not in text or '"saved_model": null' not in text:
        raise AssertionError(f"{path}: missing historical null-export markers")
    marker = "Metrics summary:\n"
    if marker not in text:
        raise AssertionError(f"{path}: missing metrics summary")
    metrics = json.loads(text.rsplit(marker, 1)[1])
    return {
        "prediction_export_null": True,
        "saved_model_null": True,
        "benchmark_recall_at_10": metrics["benchmark_reference_eval_only"]["10"]["recall"],
    }


def parse_e4_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    seed_header = re.compile(r"^=== \[[^]]+\] seed sensitivity seed=(\d+)$", re.MULTILINE)
    matches = list(seed_header.finditer(text))
    if tuple(int(match.group(1)) for match in matches) != E4_SEEDS:
        raise AssertionError("E4 execution log does not contain the five expected seeds")
    parsed = {}
    for index, match in enumerate(matches):
        seed = int(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end():end]
        alpha = re.search(
            r"Reproduced arms: unadapted alpha=([0-9.]+), adapted alpha=([0-9.]+)",
            block,
        )
        if not alpha:
            raise AssertionError(f"E4 seed {seed}: missing alpha line")
        rows = {}
        for set_name in E4_SETS:
            row = re.search(
                rf"^  {re.escape(set_name)}: cond=([0-9.]+) adapt=([0-9.]+) "
                rf"unadapt=([0-9.]+) max=([0-9.]+) delta=([+-][0-9.]+)pp",
                block,
                re.MULTILINE,
            )
            if not row:
                raise AssertionError(f"E4 seed {seed}/{set_name}: missing endpoint line")
            rows[set_name] = {
                "conditional": float(row.group(1)),
                "adapted": float(row.group(2)),
                "unadapted": float(row.group(3)),
                "max_arm": float(row.group(4)),
                "conditional_minus_max_pp": float(row.group(5)),
            }
        parsed[str(seed)] = {
            "unadapted_alpha": float(alpha.group(1)),
            "adapted_alpha": float(alpha.group(2)),
            "sets": rows,
        }
    return parsed


def verify(output: Path) -> None:
    manifest_path = EVIDENCE_DIR / "manifest.json"
    manifest = read_json(manifest_path)
    file_checks = {}
    for descriptor in manifest["files"]:
        path = ROOT / descriptor["release_path"]
        file_checks[descriptor["release_path"]] = {
            "exists": path.is_file(),
            "bytes_match": path.is_file() and path.stat().st_size == descriptor["release_bytes"],
            "sha256_matches": path.is_file() and sha256_file(path) == descriptor["release_sha256"],
        }
    if not all(all(check.values()) for check in file_checks.values()):
        raise AssertionError("historical demand-run provenance file changed")

    private_pattern = re.compile(r"/home/[^\s\"']+|/projects/[^\s\"']+|/local/[^\s\"']+")
    private_hits = []
    for descriptor in manifest["files"]:
        path = ROOT / descriptor["release_path"]
        if private_pattern.search(path.read_text(encoding="utf-8")):
            private_hits.append(descriptor["release_path"])
    if private_hits:
        raise AssertionError(f"private path remains in released provenance: {private_hits}")

    c3 = {}
    for run in C3_RUNS:
        parsed = parse_c3_log(EVIDENCE_DIR / f"c3/{run}.log")
        frozen = read_json(ROOT / f"results/aggregates/c3_holdout/{run}/metrics.json")
        expected = frozen["metrics"]["benchmark_reference_eval_only"]["top_k"]["10"]["recall"]
        parsed["matches_frozen_aggregate"] = parsed["benchmark_recall_at_10"] == expected
        if not parsed["matches_frozen_aggregate"]:
            raise AssertionError(f"{run}: execution-log endpoint differs from frozen aggregate")
        c3[run] = parsed

    e4 = parse_e4_log(EVIDENCE_DIR / "e4/e4_p0_refix_recompute_20260802.log")
    e4_checks = {}
    for seed in E4_SEEDS:
        frozen = read_json(
            ROOT / f"external_queries/conditional_selector/seed_sensitivity/metrics_by_set_seed{seed}.json"
        )
        row = e4[str(seed)]
        checks = {
            "alphas_match": (
                row["unadapted_alpha"] == frozen["arms"]["unadapted"]["alpha"]
                and row["adapted_alpha"] == frozen["arms"]["adapted"]["alpha"]
            ),
            "displayed_endpoints_match": True,
        }
        for set_name in E4_SETS:
            expected = frozen["comparison_recall_at_10"][set_name]
            observed = row["sets"][set_name]
            for key in ("conditional", "adapted", "unadapted", "max_arm"):
                if observed[key] != round(expected[key], 4):
                    checks["displayed_endpoints_match"] = False
            if observed["conditional_minus_max_pp"] != expected["conditional_minus_max_pp"]:
                checks["displayed_endpoints_match"] = False
        if not all(checks.values()):
            raise AssertionError(f"E4 seed {seed}: execution log differs from frozen artifact")
        e4_checks[str(seed)] = checks

    determinism_text = (EVIDENCE_DIR / "test_demand_determinism.py").read_text(encoding="utf-8")
    determinism_checks = {
        "pythonhashseed_matrix_present": "PYTHONHASHSEED" in determinism_text,
        "subprocess_regression_present": "subprocess" in determinism_text,
    }
    if not all(determinism_checks.values()):
        raise AssertionError("released determinism regression is incomplete")

    result = {
        "schema_version": 1,
        "manifest_sha256": sha256_file(manifest_path),
        "file_checks": file_checks,
        "private_path_hits": private_hits,
        "c3": c3,
        "e4": {
            "seeds": list(E4_SEEDS),
            "checks": e4_checks,
        },
        "determinism_regression": determinism_checks,
        "boundary": (
            "logs prove contemporaneous execution and aggregate endpoints; historical "
            "per-query rankings were not serialised"
        ),
    }
    write_json(output, result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-source-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.export_source_root:
        export(args.export_source_root.resolve())
    verify(args.output)
    print(f"verified historical C3/E4 run provenance -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
