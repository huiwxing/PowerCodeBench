#!/usr/bin/env python3
"""Rebuild the C-unadapted ablation from public item-level evidence.

Default operation is release-local: it reads the compact C-unadapted runs and
the public primary-experiment archive, then rebuilds the legacy Fig. S4(c)
summary plus a machine-readable manuscript-claim cross-check.  Maintainers can
use ``--export-source-root`` to recreate the compact archive from the frozen
raw run tree, which is that export's only input.

The implementation uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = ROOT / "results/raw/c_unadapted/c_unadapted_outcomes_compact.json"
DEFAULT_PRIMARY = ROOT / "results/raw/main_experiment/primary_outcomes_compact.json"
DEFAULT_OUTPUT = ROOT / "results/aggregates/c_unadapted_summary.json"
DEFAULT_CLAIMS_OUTPUT = ROOT / "results/aggregates/c_unadapted_manuscript_crosscheck.json"

PANEL = (
    ("Qwen2.5-Coder-1.5B", "Qwen_Qwen2.5-Coder-1.5B-Instruct"),
    ("Qwen2.5-Coder-7B", "Qwen_Qwen2.5-Coder-7B-Instruct"),
    ("Llama-3.1-8B", "meta-llama_Llama-3.1-8B-Instruct"),
    ("Qwen2.5-Coder-14B", "Qwen_Qwen2.5-Coder-14B-Instruct"),
    ("Qwen2.5-Coder-32B", "Qwen_Qwen2.5-Coder-32B-Instruct"),
    ("Llama-3.1-70B", "meta-llama_Llama-3.1-70B-Instruct"),
    ("GPT-OSS-120B", "openai_gpt-oss-120b"),
    ("Qwen3-Coder-Next", "Qwen_Qwen3-Coder-Next"),
    ("Llama-3.1-405B", "meta-llama_Llama-3.1-405B-Instruct"),
    ("Qwen3-Coder-480B", "Qwen_Qwen3-Coder-480B-A35B-Instruct"),
)
UNADAPTED_CONDITIONS = ("C", "C_FX", "C_FDR", "C_FDRS")
API_MODELS = (
    "gpt-5.4-mini",
    "claude-haiku-4-5",
    "deepseek-v4-flash",
    "gemini-2.5-flash",
)


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_compact_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        stream.write("\n")


def write_legacy_summary(path: Path, value) -> None:
    """Preserve the byte format of the previously frozen summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_path(path: Path) -> str:
    """Use a release-relative path, with a stable label for staging copies."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def git_output(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def run_key(model: str, condition: str) -> str:
    return f"{model}|{condition}"


def primary_key(group: str, model: str, condition: str) -> str:
    return f"{group}|{model}|{condition}"


def source_specs(source_root: Path) -> list[tuple[str, str, Path]]:
    base = source_root / "probe_eval_results/comparison_c_unadapted"
    return [
        (model, condition, base / model / f"benchmark_results_cond{condition}.json")
        for _, model in PANEL
        for condition in UNADAPTED_CONDITIONS
    ]


def vector_hash(
    item_ids: list[str], executed: str, matched: str,
    prompt_tokens: list, prompt_tokens_approx: list,
) -> str:
    rows = (
        f"{item_id}\t{exe}\t{match}\t{tokens}\t{approx}\n"
        for item_id, exe, match, tokens, approx in zip(
            item_ids, executed, matched, prompt_tokens, prompt_tokens_approx
        )
    )
    return sha256_bytes("".join(rows).encode("utf-8"))


def compact_token_summary(summary: dict) -> dict:
    fix = summary.get("fix_prompt_token_stats") or {}
    fix_keep = {
        key: fix.get(key)
        for key in (
            "attempts", "prompt", "docs", "docs_total", "docs_nonempty",
            "docs_nonempty_rate", "successful_fixes",
            "prompt_tokens_per_successful_fix", "per_round",
        )
        if key in fix
    }
    return {
        "round0_prompt_token_stats": summary.get("round0_prompt_token_stats"),
        "fix_prompt_token_stats": fix_keep,
        "total_prompt_token_stats": summary.get("total_prompt_token_stats"),
    }


def export_archive(source_root: Path, primary_path: Path) -> dict:
    specs = source_specs(source_root)
    missing = [str(path) for _, _, path in specs if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing C-unadapted source runs:\n" + "\n".join(missing))

    source_commit = git_output(source_root, "rev-parse", "HEAD")
    if source_commit is None:
        raise ValueError("source repository has no resolvable Git commit")
    relative_paths = [str(path.relative_to(source_root)) for _, _, path in specs]
    for cached in (False, True):
        command = ["diff", "--quiet"]
        if cached:
            command.append("--cached")
        command.extend(["--", *relative_paths])
        result = subprocess.run(["git", "-C", str(source_root), *command])
        if result.returncode != 0:
            raise ValueError("selected raw C-unadapted source files are not clean")

    primary = read_json(primary_path)
    primary_axis = primary["datasets"]["powercodebench_2000"]
    item_ids = primary_axis["item_ids"]
    if len(item_ids) != 2000 or item_ids != sorted(item_ids):
        raise ValueError("public primary archive has an invalid 2,000-item axis")

    runs = {}
    for number, (model, condition, path) in enumerate(specs, start=1):
        payload = path.read_bytes()
        data = json.loads(payload)
        items = data.get("item_results")
        if not isinstance(items, list) or len(items) != len(item_ids):
            raise ValueError(f"{path}: expected 2,000 item_results")
        by_id = {}
        for item in items:
            item_id = item.get("item_id")
            if not isinstance(item_id, str) or item_id in by_id:
                raise ValueError(f"{path}: invalid or duplicate item_id")
            if not isinstance(item.get("executed"), bool) or not isinstance(item.get("match"), bool):
                raise ValueError(f"{path}: executed/match must be boolean")
            by_id[item_id] = item
        if sorted(by_id) != item_ids:
            raise ValueError(f"{path}: item axis differs from the public primary archive")
        if any(
            by_id[item_id].get("task") != task
            or by_id[item_id].get("difficulty_level") != difficulty
            for item_id, task, difficulty in zip(
                item_ids, primary_axis["task"], primary_axis["difficulty"]
            )
        ):
            raise ValueError(f"{path}: task/difficulty metadata differ from the frozen axis")

        executed = "".join("1" if by_id[item_id]["executed"] else "0" for item_id in item_ids)
        matched = "".join("1" if by_id[item_id]["match"] else "0" for item_id in item_ids)
        prompt_tokens = [by_id[item_id].get("prompt_tokens") for item_id in item_ids]
        prompt_tokens_approx = [by_id[item_id].get("prompt_tokens_approx") for item_id in item_ids]
        if any(
            value is not None and (not isinstance(value, int) or isinstance(value, bool))
            for value in prompt_tokens + prompt_tokens_approx
        ):
            raise ValueError(f"{path}: prompt-token vectors must contain integers/nulls")
        summary = data.get("summary") or {}
        if summary.get("total") != len(item_ids):
            raise ValueError(f"{path}: source total does not match item_results")
        if summary.get("n_executed") != executed.count("1"):
            raise ValueError(f"{path}: source n_executed does not match item_results")
        if summary.get("n_matched") != matched.count("1"):
            raise ValueError(f"{path}: source n_matched does not match item_results")
        snapshots = [
            {
                "round": row.get("round"),
                "total": row.get("total"),
                "n_executed": row.get("n_executed"),
                "n_matched": row.get("n_matched"),
            }
            for row in (data.get("round_snapshots") or [])
        ]
        if not snapshots or snapshots[-1]["n_matched"] != matched.count("1"):
            raise ValueError(f"{path}: final round snapshot does not match item_results")

        relative = str(path.relative_to(source_root))
        tracked = git_output(source_root, "ls-files", "-s", "--", relative)
        runs[run_key(model, condition)] = {
            "model": model,
            "condition": condition,
            "executed_bits": executed,
            "match_bits": matched,
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_approx": prompt_tokens_approx,
            "round_snapshots": snapshots,
            "token_summary": compact_token_summary(summary),
            "vector_sha256": vector_hash(
                item_ids, executed, matched, prompt_tokens, prompt_tokens_approx
            ),
            "source": {
                "path": relative,
                "commit": source_commit,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
                "git_blob": tracked.split()[1] if tracked else None,
                "git_lfs_object": True,
            },
        }
        if number % 10 == 0 or number == len(specs):
            print(f"exported {number}/{len(specs)} C-unadapted runs", flush=True)

    return {
        "schema_version": 1,
        "description": (
            "Minimum sufficient item-level archive for the ten-model C-unadapted "
            "Round-0 ablation, its reactive endpoints, and prompt-token accounting."
        ),
        "provenance": {
            "source_repository": "frozen experimental pipeline",
            "source_commit": source_commit,
            "selected_paths_clean_at_export": True,
            "source_values": "raw item_results, round_snapshots, and raw-run token summaries",
            "not_used_as_export_inputs": ["paper", "results/aggregates"],
            "round_snapshot_granularity": (
                "aggregate cumulative R0--R3 counts; raw runs do not preserve "
                "item-level intermediate-round match vectors"
            ),
        },
        "dataset": {
            "name": "powercodebench_2000",
            "n_items": len(item_ids),
            "item_order": "lexicographic item_id",
            "item_ids": item_ids,
            "task": primary_axis["task"],
            "difficulty": primary_axis["difficulty"],
            "metadata_sha256": primary_axis["metadata_sha256"],
        },
        "runs": runs,
    }


def validate_archive(archive: dict, primary: dict) -> None:
    if archive.get("schema_version") != 1:
        raise ValueError("unsupported C-unadapted compact schema")
    axis = archive.get("dataset") or {}
    item_ids = axis.get("item_ids")
    if (
        not isinstance(item_ids, list)
        or not all(isinstance(item_id, str) for item_id in item_ids)
        or item_ids != sorted(item_ids)
    ):
        raise ValueError("invalid C-unadapted compact item axis")
    if len(item_ids) != 2000 or len(set(item_ids)) != 2000:
        raise ValueError("C-unadapted compact must contain exactly 2,000 unique items")
    primary_axis = primary["datasets"]["powercodebench_2000"]
    if (
        item_ids != primary_axis["item_ids"]
        or axis.get("task") != primary_axis["task"]
        or axis.get("difficulty") != primary_axis["difficulty"]
        or axis.get("metadata_sha256") != primary_axis["metadata_sha256"]
    ):
        raise ValueError("C-unadapted and primary compact axes disagree")

    expected = {
        run_key(model, condition)
        for _, model in PANEL
        for condition in UNADAPTED_CONDITIONS
    }
    runs = archive.get("runs") or {}
    if set(runs) != expected:
        raise ValueError("C-unadapted compact run set is incomplete or unexpected")
    for key, row in runs.items():
        executed = row.get("executed_bits")
        matched = row.get("match_bits")
        tokens = row.get("prompt_tokens")
        approximate = row.get("prompt_tokens_approx")
        if (
            not isinstance(executed, str)
            or not isinstance(matched, str)
            or len(executed) != len(item_ids)
            or len(matched) != len(item_ids)
            or set(executed) - {"0", "1"}
            or set(matched) - {"0", "1"}
            or not isinstance(tokens, list)
            or not isinstance(approximate, list)
            or len(tokens) != len(item_ids)
            or len(approximate) != len(item_ids)
            or any(
                value is not None and (not isinstance(value, int) or isinstance(value, bool))
                for value in tokens + approximate
            )
        ):
            raise ValueError(f"{key}: invalid compact vectors")
        if vector_hash(item_ids, executed, matched, tokens, approximate) != row.get("vector_sha256"):
            raise ValueError(f"{key}: vector hash mismatch")
        snapshots = row.get("round_snapshots") or []
        if (
            not snapshots
            or snapshots[-1].get("n_executed") != executed.count("1")
            or snapshots[-1].get("n_matched") != matched.count("1")
        ):
            raise ValueError(f"{key}: final snapshot does not match item outcomes")

    required_primary = {
        primary_key("comparison", model, condition)
        for _, model in PANEL
        for condition in ("A", "C")
    } | {
        primary_key("bm25", model, "R") for _, model in PANEL
    } | {
        primary_key("api", model, "C_FDRS") for model in API_MODELS
    }
    primary_runs = primary.get("runs") or {}
    missing_primary = required_primary - set(primary_runs)
    if missing_primary:
        raise ValueError(f"primary compact lacks required runs: {sorted(missing_primary)}")
    for key in required_primary:
        row = primary_runs[key]
        executed = row.get("executed_bits")
        matched = row.get("match_bits")
        tokens = row.get("prompt_tokens")
        approximate = row.get("prompt_tokens_approx")
        if (
            not isinstance(executed, str)
            or not isinstance(matched, str)
            or len(executed) != len(item_ids)
            or len(matched) != len(item_ids)
            or set(executed) - {"0", "1"}
            or set(matched) - {"0", "1"}
            or not isinstance(tokens, list)
            or not isinstance(approximate, list)
            or len(tokens) != len(item_ids)
            or len(approximate) != len(item_ids)
        ):
            raise ValueError(f"{key}: invalid primary compact vectors")
        if vector_hash(item_ids, executed, matched, tokens, approximate) != row.get("vector_sha256"):
            raise ValueError(f"{key}: primary compact vector hash mismatch")


def accuracy(row: dict) -> float:
    bits = row["match_bits"]
    return bits.count("1") / len(bits)


def get_primary(primary: dict, group: str, model: str, condition: str) -> dict:
    try:
        return primary["runs"][primary_key(group, model, condition)]
    except KeyError as exc:
        raise ValueError(f"primary compact lacks {group}/{model}/{condition}") from exc


def get_unadapted(archive: dict, model: str, condition: str) -> dict:
    try:
        return archive["runs"][run_key(model, condition)]
    except KeyError as exc:
        raise ValueError(f"C-unadapted compact lacks {model}/{condition}") from exc


def build_legacy_summary(archive: dict, primary: dict) -> dict:
    rows = []
    a_means, r_means, adapted_means, unadapted_means = [], [], [], []
    adapted_deltas, unadapted_minus_a, unadapted_minus_r = [], [], []
    for _, model in PANEL:
        a_acc = accuracy(get_primary(primary, "comparison", model, "A"))
        r_acc = accuracy(get_primary(primary, "bm25", model, "R"))
        adapted_acc = accuracy(get_primary(primary, "comparison", model, "C"))
        unadapted_acc = accuracy(get_unadapted(archive, model, "C"))
        delta = (adapted_acc - unadapted_acc) * 100
        delta_a = (unadapted_acc - a_acc) * 100
        delta_r = (unadapted_acc - r_acc) * 100
        a_means.append(a_acc * 100)
        r_means.append(r_acc * 100)
        adapted_means.append(adapted_acc * 100)
        unadapted_means.append(unadapted_acc * 100)
        adapted_deltas.append(delta)
        unadapted_minus_a.append(delta_a)
        unadapted_minus_r.append(delta_r)
        rows.append({
            "model": model,
            "A_acc": a_acc,
            "R_acc": r_acc,
            "C_adapted": adapted_acc,
            "C_unadapted": unadapted_acc,
            "delta_adapted_minus_unadapted": delta,
            "delta_unadapted_minus_a": delta_a,
            "delta_unadapted_minus_r": delta_r,
        })
    return {
        "panel_n_models": len(rows),
        "panel_mean_A_acc": st.mean(a_means) / 100,
        "panel_mean_R_acc": st.mean(r_means) / 100,
        "panel_mean_C_adapted_acc": st.mean(adapted_means) / 100,
        "panel_mean_C_unadapted_acc": st.mean(unadapted_means) / 100,
        "panel_mean_delta_adapted_minus_unadapted_pp": st.mean(adapted_deltas),
        "panel_mean_delta_unadapted_minus_a_pp": st.mean(unadapted_minus_a),
        "panel_mean_delta_unadapted_minus_r_pp": st.mean(unadapted_minus_r),
        "rows": rows,
    }


def manuscript_crosscheck(
    archive: dict, primary: dict, summary: dict,
    archive_path: Path, primary_path: Path,
) -> dict:
    rows = summary["rows"]
    positive_a = sum(row["delta_unadapted_minus_a"] > 0 for row in rows)
    positive_r = sum(row["delta_unadapted_minus_r"] > 0 for row in rows)
    full_adapted_lift_pp = (
        summary["panel_mean_C_adapted_acc"] - summary["panel_mean_A_acc"]
    ) * 100
    adaptation_share = (
        summary["panel_mean_delta_adapted_minus_unadapted_pp"]
        / full_adapted_lift_pp
    )

    unadapted_fdrs = {
        model: accuracy(get_unadapted(archive, model, "C_FDRS"))
        for _, model in PANEL
    }
    api_fdrs = {
        model: accuracy(get_primary(primary, "api", model, "C_FDRS"))
        for model in API_MODELS
    }
    best_open_model = max(unadapted_fdrs, key=unadapted_fdrs.get)
    best_api_model = max(api_fdrs, key=api_fdrs.get)
    best_open = unadapted_fdrs[best_open_model]
    best_api = api_fdrs[best_api_model]

    claims = {
        "unadapted_C_minus_A_panel_pp": {
            "observed": summary["panel_mean_delta_unadapted_minus_a_pp"],
            "manuscript_display": f"{summary['panel_mean_delta_unadapted_minus_a_pp']:.2f}",
            "expected_display": "27.79",
        },
        "unadapted_C_minus_R_panel_pp": {
            "observed": summary["panel_mean_delta_unadapted_minus_r_pp"],
            "manuscript_display": f"{summary['panel_mean_delta_unadapted_minus_r_pp']:.2f}",
            "expected_display": "22.59",
        },
        "adapted_minus_unadapted_C_panel_pp": {
            "observed": summary["panel_mean_delta_adapted_minus_unadapted_pp"],
            "manuscript_display": f"{summary['panel_mean_delta_adapted_minus_unadapted_pp']:.2f}",
            "expected_display": "3.16",
        },
        "positive_on_every_panel_model": {
            "unadapted_C_minus_A": f"{positive_a}/{len(rows)}",
            "unadapted_C_minus_R": f"{positive_r}/{len(rows)}",
            "expected": "10/10",
        },
        "adaptation_share_of_full_adapted_lift": {
            "full_adapted_C_minus_A_panel_pp": full_adapted_lift_pp,
            "share": adaptation_share,
            "display_percent": f"{100 * adaptation_share:.2f}",
            "manuscript_description": "about 10%",
        },
        "upper_tier_cross_vendor_C_FDRS": {
            "best_open_unadapted_model": best_open_model,
            "best_open_unadapted_accuracy": best_open,
            "best_open_manuscript_percent": f"{100 * best_open:.2f}",
            "best_api_adapted_model": best_api_model,
            "best_api_adapted_accuracy": best_api,
            "best_api_manuscript_percent": f"{100 * best_api:.2f}",
            "open_minus_api_pp": (best_open - best_api) * 100,
            "all_open_unadapted_C_FDRS": unadapted_fdrs,
            "all_api_adapted_C_FDRS": api_fdrs,
            "expected_display": "65.65 versus 63.75",
        },
    }
    claims["unadapted_C_minus_A_panel_pp"]["pass"] = (
        claims["unadapted_C_minus_A_panel_pp"]["manuscript_display"] == "27.79"
    )
    claims["unadapted_C_minus_R_panel_pp"]["pass"] = (
        claims["unadapted_C_minus_R_panel_pp"]["manuscript_display"] == "22.59"
    )
    claims["adapted_minus_unadapted_C_panel_pp"]["pass"] = (
        claims["adapted_minus_unadapted_C_panel_pp"]["manuscript_display"] == "3.16"
    )
    claims["positive_on_every_panel_model"]["pass"] = positive_a == positive_r == len(rows)
    claims["adaptation_share_of_full_adapted_lift"]["pass"] = math.isclose(
        adaptation_share, 0.10210016155088852, rel_tol=0.0, abs_tol=1e-15
    )
    claims["upper_tier_cross_vendor_C_FDRS"]["pass"] = (
        f"{100 * best_open:.2f}" == "65.65"
        and f"{100 * best_api:.2f}" == "63.75"
        and best_open > best_api
    )
    all_pass = all(claim["pass"] for claim in claims.values())
    if not all_pass:
        raise ValueError(f"C-unadapted evidence disagrees with manuscript: {claims}")
    return {
        "schema_version": 1,
        "source_artifacts": {
            "c_unadapted_compact": {
                "path": logical_path(archive_path),
                "sha256": sha256_file(archive_path),
            },
            "primary_compact": {
                "path": logical_path(primary_path),
                "sha256": sha256_file(primary_path),
            },
        },
        "definitions": {
            "accuracy": "exact matched item count divided by all 2,000 frozen items",
            "panel_mean": "unweighted arithmetic mean of ten unrounded per-model accuracies",
            "C_unadapted": "condition C using the unadapted hybrid TF-IDF demand predictor",
            "cross_vendor_comparator": "maximum C+FDRS endpoint among the four adapted mid-tier API models",
        },
        "all_pass": all_pass,
        "claims": claims,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-source-root", type=Path)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--primary-archive", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--claims-output", type=Path, default=DEFAULT_CLAIMS_OUTPUT)
    args = parser.parse_args()

    archive_path = args.archive.resolve()
    primary_path = args.primary_archive.resolve()
    if args.export_source_root is not None:
        archive = export_archive(args.export_source_root.resolve(), primary_path)
        write_compact_json(archive_path, archive)
        print(f"exported C-unadapted compact evidence -> {archive_path}")

    archive = read_json(archive_path)
    primary = read_json(primary_path)
    validate_archive(archive, primary)
    summary = build_legacy_summary(archive, primary)
    claims = manuscript_crosscheck(archive, primary, summary, archive_path, primary_path)
    write_legacy_summary(args.output.resolve(), summary)
    write_compact_json(args.claims_output.resolve(), claims)
    print(f"rebuilt Fig. S4(c) C-unadapted summary -> {args.output.resolve()}")
    print(f"verified C-unadapted manuscript claims -> {args.claims_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
