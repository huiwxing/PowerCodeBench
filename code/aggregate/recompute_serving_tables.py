#!/usr/bin/env python3
"""Rebuild the manuscript serving tables from public raw measurements.

This is a read-only, CPU-only verifier for main-text Table T11 and
supplementary Tables S16--S17.  It reads the per-repetition Protocol-A
``aggregate.json``/``run_meta.json``/``energy_*.json`` records and the
per-concurrency Protocol-B records (including the original ``bench_serve``
latency output).  The frozen ``e3_serving_table.json`` is used only as an
expected cross-check; none of its measurements is an input to the rebuild.

The 480B anchor follows the ratified composite convention encoded by the
release builder: generation-window throughput, but uncorrected probe-window
energy and all other fields.  The 405B anchor is an ordinary single-repetition
probe-window projection.

Only the Python standard library is required.  Output is deterministic and
contains source hashes, coverage, field lineage, complete rebuilt blocks,
manuscript-table projections, and every detected difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEASUREMENTS = ROOT / "serving" / "measurement_json"
DEFAULT_EXPECTED = DEFAULT_MEASUREMENTS / "e3_serving_table.json"
DEFAULT_OUTPUT = ROOT / "results" / "aggregates" / "e3_serving_tables_recomputed.json"

# Order is the order used by the manuscript tables.  ``n_reps`` and the
# operating points are study-design facts, not values read from the frozen
# table being checked.
PROTOCOL_A_SPECS = (
    ("0.5B", "Qwen2.5-Coder", "Qwen_Qwen2.5-Coder-0.5B-Instruct", 3, True),
    ("1.5B", "Qwen2.5-Coder", "Qwen_Qwen2.5-Coder-1.5B-Instruct", 3, True),
    ("7B", "Qwen2.5-Coder", "Qwen_Qwen2.5-Coder-7B-Instruct", 3, True),
    ("8B", "Llama-3.1", "meta-llama_Llama-3.1-8B-Instruct", 3, True),
    ("14B", "Qwen2.5-Coder", "Qwen_Qwen2.5-Coder-14B-Instruct", 3, True),
    ("32B", "Qwen2.5-Coder", "Qwen_Qwen2.5-Coder-32B-Instruct", 3, True),
    ("70B", "Llama-3.1", "meta-llama_Llama-3.1-70B-Instruct", 3, True),
    ("Next", "Qwen3-Coder-Next (MoE)", "Qwen_Qwen3-Coder-Next", 3, True),
    ("120B", "gpt-oss (MoE)", "openai_gpt-oss-120b", 3, True),
    ("405B", "Llama-3.1", "meta-llama_Llama-3.1-405B-Instruct", 1, False),
    (
        "480B",
        "Qwen3-Coder (MoE)",
        "Qwen_Qwen3-Coder-480B-A35B-Instruct",
        1,
        False,
    ),
)

PROTOCOL_B_TIERS = ("7B", "14B", "32B", "70B", "Next")
CONCURRENCY_LEVELS = (1, 4, 16)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def summary(values: list[float | int]) -> dict[str, Any]:
    low = min(values)
    high = max(values)
    return {
        "max": high,
        "median": statistics.median(values),
        "min": low,
        "n": len(values),
        "spread_ratio": high / low if low != 0 else None,
        "values": values,
    }


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_values(
    actual: Any,
    expected: Any,
    path: str,
    differences: list[dict[str, Any]],
    *,
    rel_tol: float = 1e-12,
    abs_tol: float = 1e-12,
) -> int:
    """Recursively compare values and return the number of leaf checks."""
    checked = 0
    if isinstance(actual, dict) and isinstance(expected, dict):
        actual_keys = set(actual)
        expected_keys = set(expected)
        for key in sorted(actual_keys - expected_keys):
            differences.append(
                {"path": f"{path}.{key}", "actual": actual[key], "expected": "<missing>"}
            )
        for key in sorted(expected_keys - actual_keys):
            differences.append(
                {"path": f"{path}.{key}", "actual": "<missing>", "expected": expected[key]}
            )
        for key in sorted(actual_keys & expected_keys):
            checked += compare_values(
                actual[key], expected[key], f"{path}.{key}", differences,
                rel_tol=rel_tol, abs_tol=abs_tol,
            )
        return checked
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            differences.append(
                {"path": f"{path}.length", "actual": len(actual), "expected": len(expected)}
            )
        for index, (left, right) in enumerate(zip(actual, expected)):
            checked += compare_values(
                left, right, f"{path}[{index}]", differences,
                rel_tol=rel_tol, abs_tol=abs_tol,
            )
        return checked

    checked += 1
    if is_number(actual) and is_number(expected):
        equal = math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)
    else:
        equal = actual == expected
    if not equal:
        differences.append({"path": path, "actual": actual, "expected": expected})
    return checked


class RawChecks:
    """Accumulate raw-record consistency checks without hiding failures."""

    def __init__(self) -> None:
        self.n_checked = 0
        self.differences: list[dict[str, Any]] = []

    def equal(self, actual: Any, expected: Any, path: str) -> None:
        self.n_checked += compare_values(
            actual, expected, path, self.differences,
            rel_tol=1e-12, abs_tol=1e-9,
        )


def read_energy_records(rep_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    paths = sorted(rep_dir.glob("energy_*.json"))
    if not paths:
        raise FileNotFoundError(f"{rep_dir}: no energy_*.json records")
    return [load_json(path) for path in paths], paths


def validate_aggregate(
    aggregate: dict[str, Any],
    meta: dict[str, Any],
    energy_records: list[dict[str, Any]],
    label: str,
    checks: RawChecks,
) -> None:
    """Recheck all aggregate quantities used by the manuscript projection."""
    devices = [device for record in energy_records for device in record["devices"]]
    total_energy = sum(device["energy_j"] for device in devices)
    representative_window = statistics.median(
        record["duration_s"] for record in energy_records
    )
    checks.equal(total_energy, aggregate["total_energy_j"], f"{label}.total_energy_j")
    checks.equal(
        representative_window,
        aggregate["representative_window_s"],
        f"{label}.representative_window_s",
    )
    # Each node has its own measured window; summing node energy/window
    # preserves that timing instead of dividing cluster energy by the median
    # node duration (which is wrong when a node starts or stops early).
    cluster_avg_power = sum(
        sum(device["energy_j"] for device in record["devices"])
        / record["duration_s"]
        for record in energy_records
    )
    checks.equal(
        cluster_avg_power,
        aggregate["cluster_avg_power_w"],
        f"{label}.cluster_avg_power_w",
    )
    checks.equal(len(devices), aggregate["n_gpus"], f"{label}.n_gpus")
    checks.equal(
        len(energy_records), aggregate["n_nodes"], f"{label}.n_nodes"
    )
    checks.equal(
        sum(record["peak_memory_used_bytes_sum"] for record in energy_records),
        aggregate["peak_memory_used_bytes_sum"],
        f"{label}.peak_memory_used_bytes_sum",
    )
    checks.equal(
        max(device["peak_memory_used_bytes"] for device in devices),
        aggregate["peak_memory_used_bytes_max_device"],
        f"{label}.peak_memory_used_bytes_max_device",
    )
    checks.equal(
        aggregate["energy_methods"],
        ["nvml_total_energy_counter"],
        f"{label}.aggregate_energy_method",
    )
    for index, device in enumerate(devices):
        checks.equal(
            device["energy_method"],
            "nvml_total_energy_counter",
            f"{label}.device[{index}].energy_method",
        )
        checks.equal(
            device["energy_counter_error"],
            None,
            f"{label}.device[{index}].energy_counter_error",
        )
        checks.equal(
            device["sample_errors"],
            0,
            f"{label}.device[{index}].sample_errors",
        )
        checks.equal(
            device["energy_j"] > 0,
            True,
            f"{label}.device[{index}].positive_energy",
        )
    checks.equal(
        total_energy / aggregate["requests"],
        aggregate["j_per_request"],
        f"{label}.j_per_request",
    )
    checks.equal(
        total_energy / aggregate["successful_requests"],
        aggregate["j_per_successful_request"],
        f"{label}.j_per_successful_request",
    )
    checks.equal(
        aggregate["requests"] / representative_window,
        aggregate["requests_per_s"],
        f"{label}.requests_per_s",
    )
    checks.equal(
        aggregate["total_tokens"] / representative_window,
        aggregate["total_tokens_per_s"],
        f"{label}.total_tokens_per_s",
    )
    checks.equal(
        total_energy / aggregate["completion_tokens"],
        aggregate["j_per_completion_token"],
        f"{label}.j_per_completion_token",
    )
    checks.equal(
        total_energy / aggregate["total_tokens"],
        aggregate["j_per_total_token"],
        f"{label}.j_per_total_token",
    )
    # Protocol A records these counters directly in run_meta.  Protocol B
    # records them in the original bench_serve output and is cross-checked in
    # ``rebuild_protocol_b_cell`` below.
    if "requests" in meta:
        checks.equal(
            meta["requests"], aggregate["requests"], f"{label}.meta_requests"
        )
        checks.equal(
            meta["successful_requests"],
            aggregate["successful_requests"],
            f"{label}.meta_successful_requests",
        )
        checks.equal(
            meta["tokens"]["prompt_tokens"],
            aggregate["prompt_tokens"],
            f"{label}.meta_prompt_tokens",
        )
        checks.equal(
            meta["tokens"]["completion_tokens"],
            aggregate["completion_tokens"],
            f"{label}.meta_completion_tokens",
        )


def rebuild_protocol_a_cell(
    cell_dir: Path,
    n_reps: int,
    tier: str,
    condition: str,
    checks: RawChecks,
    source_paths: set[Path],
) -> dict[str, Any]:
    reps: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    expected_rep_dirs = [cell_dir / f"rep{rep}" for rep in range(1, n_reps + 1)]
    actual_rep_dirs = sorted(path for path in cell_dir.glob("rep*") if path.is_dir())
    if actual_rep_dirs != expected_rep_dirs:
        raise ValueError(
            f"{cell_dir}: rep directories differ; expected "
            f"{[p.name for p in expected_rep_dirs]}, got {[p.name for p in actual_rep_dirs]}"
        )

    for rep_index, rep_dir in enumerate(expected_rep_dirs, start=1):
        aggregate_path = rep_dir / "aggregate.json"
        meta_path = rep_dir / "run_meta.json"
        aggregate = load_json(aggregate_path)
        meta = load_json(meta_path)
        energy_records, energy_paths = read_energy_records(rep_dir)
        source_paths.update((aggregate_path, meta_path, *energy_paths))
        validate_aggregate(
            aggregate, meta, energy_records,
            f"protocol_a.{tier}.{condition}.rep{rep_index}", checks,
        )
        reps.append((aggregate, meta, energy_records))

    aggregates = [row[0] for row in reps]
    metas = [row[1] for row in reps]
    energies = [row[2] for row in reps]

    def agg_values(key: str) -> list[float | int]:
        return [aggregate[key] for aggregate in aggregates]

    def token_values(key: str) -> list[float | int]:
        return [meta["tokens"][key] for meta in metas]

    avg_gpu_util = [
        statistics.fmean(
            device["avg_gpu_utilization_pct"]
            for record in records
            for device in record["devices"]
        )
        for records in energies
    ]
    avg_power_per_gpu = [
        statistics.fmean(
            device["avg_sampled_power_w"]
            for record in records
            for device in record["devices"]
        )
        for records in energies
    ]
    peak_power_per_gpu = [
        max(
            device["peak_power_w"]
            for record in records
            for device in record["devices"]
        )
        for records in energies
    ]
    generate_windows = [meta["window_wall_s"] for meta in metas]
    items_per_generate_window = [
        aggregate["requests"] / meta["window_wall_s"]
        for aggregate, meta, _ in reps
    ]
    tokens_per_generate_window = [
        aggregate["total_tokens"] / meta["window_wall_s"]
        for aggregate, meta, _ in reps
    ]
    probe_ratios = [
        aggregate["representative_window_s"] / meta["window_wall_s"]
        for aggregate, meta, _ in reps
    ]

    first_aggregate, first_meta, _ = reps[0]
    block = {
        "avg_gpu_util_pct": summary(avg_gpu_util),
        "avg_power_w_cluster": summary(agg_values("cluster_avg_power_w")),
        "avg_power_w_per_gpu": summary(avg_power_per_gpu),
        "completion_tokens": summary(token_values("completion_tokens")),
        "duration_s_by_node": [aggregate["duration_s_by_node"] for aggregate in aggregates],
        "energy_methods": first_aggregate["energy_methods"],
        "generate_window_s": summary(generate_windows),
        "items_per_s": summary(agg_values("requests_per_s")),
        "items_per_s_generate_window": summary(items_per_generate_window),
        "j_per_completion_token": summary(agg_values("j_per_completion_token")),
        "j_per_item": summary(agg_values("j_per_request")),
        "j_per_succ_task": summary(agg_values("j_per_successful_request")),
        "j_per_total_token": summary(agg_values("j_per_total_token")),
        "measured_gpu_indices": first_meta["measured_gpu_indices"],
        "model_load_s": first_meta["model_load_s"],
        "n_gpus": first_aggregate["n_gpus"],
        "n_nodes": first_aggregate["n_nodes"],
        "n_reps": len(reps),
        "peak_power_w_per_gpu": summary(peak_power_per_gpu),
        "peak_vram_node_sum_gb": max(
            aggregate["peak_memory_used_bytes_sum"] for aggregate in aggregates
        ) / 1e9,
        "peak_vram_per_gpu_gb": max(
            device["peak_memory_used_bytes"]
            for records in energies
            for record in records
            for device in record["devices"]
        ) / 1e9,
        "pp_size": first_meta["pp_size"],
        "probe_mode": first_meta["probe_mode"],
        "probe_over_generate_window_ratio": summary(probe_ratios),
        "prompt_tokens": summary(token_values("prompt_tokens")),
        "requests": first_aggregate["requests"],
        "start_skew_s": summary(agg_values("start_skew_s")),
        "stop_skew_s": summary(agg_values("stop_skew_s")),
        "successful_requests_by_rep": agg_values("successful_requests"),
        "tokens_per_s": summary(agg_values("total_tokens_per_s")),
        "tokens_per_s_generate_window": summary(tokens_per_generate_window),
        "total_energy_j": summary(agg_values("total_energy_j")),
        "tp_size": first_meta["tp_size"],
        "wall_span_s": summary(agg_values("wall_span_s")),
        "warmup_s": first_meta["warmup_s"],
        "window_s": summary(agg_values("representative_window_s")),
        "window_scope": first_meta["window_scope"],
    }

    if tier == "480B" and condition == "C":
        # Final, estimate-free anchor convention.  Only the displayed throughput
        # changes denominator; probe-window energy remains deliberately untouched.
        block["tokens_per_s"] = summary(tokens_per_generate_window)
    return block


def rebuild_protocol_b_cell(
    model_dir: Path,
    tier: str,
    checks: RawChecks,
    source_paths: set[Path],
) -> dict[str, Any]:
    rendered: dict[str, Any] = {
        "client": "vllm bench serve",
        "concurrency_levels": {},
        "request_pool_n": None,
        "server_startup_s": None,
        "tp_size": None,
        "warmup_s": None,
    }
    for concurrency in CONCURRENCY_LEVELS:
        level_dir = model_dir / f"conc{concurrency}"
        aggregate_path = level_dir / "aggregate.json"
        meta_path = level_dir / "run_meta.json"
        bench_path = level_dir / "bench_serve.json"
        aggregate = load_json(aggregate_path)
        meta = load_json(meta_path)
        bench = load_json(bench_path)
        energy_records, energy_paths = read_energy_records(level_dir)
        source_paths.update((aggregate_path, meta_path, bench_path, *energy_paths))
        label = f"protocol_b.{tier}.conc{concurrency}"
        validate_aggregate(aggregate, meta, energy_records, label, checks)

        latency_keys = (
            "completed", "duration", "mean_e2el_ms", "median_e2el_ms",
            "p95_e2el_ms", "mean_itl_ms", "median_itl_ms", "p95_itl_ms",
            "mean_tpot_ms", "median_tpot_ms", "p95_tpot_ms", "mean_ttft_ms",
            "median_ttft_ms", "p95_ttft_ms", "output_throughput",
            "request_throughput", "total_input_tokens", "total_output_tokens",
            "total_token_throughput",
        )
        for key in latency_keys:
            checks.equal(bench[key], meta["latency"][key], f"{label}.bench_vs_meta.{key}")
        checks.equal(bench["completed"], aggregate["successful_requests"], f"{label}.completed")
        checks.equal(bench["num_prompts"], aggregate["requests"], f"{label}.num_prompts")
        checks.equal(bench["failed"], 0, f"{label}.failed")
        checks.equal(meta["concurrency"], concurrency, f"{label}.concurrency")

        devices = [device for record in energy_records for device in record["devices"]]
        completed = bench["completed"]
        duration = bench["duration"]
        window = aggregate["representative_window_s"]
        overhead = window - duration
        if rendered["request_pool_n"] is None:
            rendered["request_pool_n"] = meta["request_pool"]["pool_size"]
            rendered["server_startup_s"] = meta["server_startup_s"]
            rendered["tp_size"] = meta["tp_size"]
            rendered["warmup_s"] = meta["warmup_s"]

        rendered["concurrency_levels"][str(concurrency)] = {
            "bench_duration_s": duration,
            "completed": completed,
            "e2e_latency_s": {
                "mean": bench["mean_e2el_ms"] / 1000,
                "median": bench["median_e2el_ms"] / 1000,
                "p95": bench["p95_e2el_ms"] / 1000,
            },
            "energy": {
                "avg_gpu_util_pct": statistics.fmean(
                    device["avg_gpu_utilization_pct"] for device in devices
                ),
                "avg_power_w_per_gpu": statistics.fmean(
                    device["avg_sampled_power_w"] for device in devices
                ),
                "cluster_avg_power_w": aggregate["cluster_avg_power_w"],
                "energy_methods": aggregate["energy_methods"],
                "j_per_completion_token": aggregate["j_per_completion_token"],
                "j_per_request_measured_window": aggregate["j_per_request"],
                "j_per_request_serving_lower_bound": (
                    aggregate["total_energy_j"] * duration / window / completed
                ),
                "n_gpus": aggregate["n_gpus"],
                "peak_power_w_per_gpu": max(
                    device["peak_power_w"] for device in devices
                ),
                "peak_vram_per_gpu_gb": max(
                    device["peak_memory_used_bytes"] for device in devices
                ) / 1e9,
                "total_energy_j": aggregate["total_energy_j"],
            },
            "energy_window_s": window,
            "failed": bench["failed"],
            "num_prompts": bench["num_prompts"],
            "output_throughput_tok_s": bench["output_throughput"],
            "request_throughput_req_s": bench["request_throughput"],
            "tbt_itl_ms": {
                "mean": bench["mean_itl_ms"],
                "median": bench["median_itl_ms"],
                "p95": bench["p95_itl_ms"],
            },
            "total_input_tokens": bench["total_input_tokens"],
            "total_output_tokens": bench["total_output_tokens"],
            "total_token_throughput_tok_s": bench["total_token_throughput"],
            "tpot_ms": {
                "mean": bench["mean_tpot_ms"],
                "median": bench["median_tpot_ms"],
                "p95": bench["p95_tpot_ms"],
            },
            "ttft_ms": {
                "mean": bench["mean_ttft_ms"],
                "median": bench["median_ttft_ms"],
                "p95": bench["p95_ttft_ms"],
            },
            "window_overhead_frac": overhead / window,
            "window_overhead_s": overhead,
        }
    return rendered


def validate_480b_anchor_evidence(
    measurements: Path,
    frozen_block: dict[str, Any],
    checks: RawChecks,
    source_paths: set[Path],
) -> dict[str, Any]:
    """Audit the frozen 480B anchor plus every archived retest attempt.

    The retests establish throughput repeatability and the window-instrument
    ruling.  They never replace the frozen anchor's energy fields.
    """
    model_dir = "Qwen_Qwen3-Coder-480B-A35B-Instruct"
    run_specs = (
        (
            "frozen_anchor",
            measurements / "protocolA" / model_dir / "C" / "rep1",
            "5786611",
            "table source: generation-window throughput; conservative probe-window energy",
        ),
        (
            "r2_incomplete_single_node",
            measurements / "protocolA_anchor_retest" / model_dir / "C"
            / "rep1_r2_incomplete_single_node",
            "5865397",
            "excluded: only one of four nodes captured",
        ),
        (
            "r3_probe_started_late",
            measurements / "protocolA_anchor_retest" / model_dir / "C"
            / "rep1_r3_probe_started_late",
            "5867046",
            "throughput replication only; excluded energy because probe window misses generation start",
        ),
        (
            "r4_probe_started_early",
            measurements / "protocolA_anchor_retest" / model_dir / "C" / "rep1",
            "5868153",
            "throughput replication only; excluded energy because early probe start inflates window",
        ),
    )
    rows: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for name, rep_dir, expected_job, use in run_specs:
        aggregate_path = rep_dir / "aggregate.json"
        meta_path = rep_dir / "run_meta.json"
        aggregate = load_json(aggregate_path)
        meta = load_json(meta_path)
        energy_records, energy_paths = read_energy_records(rep_dir)
        source_paths.update((aggregate_path, meta_path, *energy_paths))
        validate_aggregate(
            aggregate, meta, energy_records, f"anchor_480b.{name}", checks
        )
        checks.equal(meta["slurm_job_id"], expected_job, f"anchor_480b.{name}.job")
        row = {
            "name": name,
            "job_id": meta["slurm_job_id"],
            "use": use,
            "captured_nodes": aggregate["n_nodes"],
            "captured_gpus": aggregate["n_gpus"],
            "engine_gpus": meta["n_gpus_engine"],
            "matched": aggregate["successful_requests"],
            "prompt_tokens": aggregate["prompt_tokens"],
            "total_tokens": aggregate["total_tokens"],
            "generation_window_s": meta["window_wall_s"],
            "probe_window_s": aggregate["representative_window_s"],
            "probe_over_generation_ratio": (
                aggregate["representative_window_s"] / meta["window_wall_s"]
            ),
            "generation_window_tokens_per_s": (
                aggregate["total_tokens"] / meta["window_wall_s"]
            ),
            "probe_window_energy_j": aggregate["total_energy_j"],
            "node_energy_records": len(energy_records),
        }
        rows.append(row)
        by_name[name] = row

    frozen = by_name["frozen_anchor"]
    r2 = by_name["r2_incomplete_single_node"]
    r3 = by_name["r3_probe_started_late"]
    r4 = by_name["r4_probe_started_early"]
    checks.equal(frozen["captured_gpus"], frozen["engine_gpus"], "anchor_480b.frozen.complete_capture")
    checks.equal(r2["captured_nodes"], 1, "anchor_480b.r2.single_captured_node")
    checks.equal(r2["captured_gpus"] < r2["engine_gpus"], True, "anchor_480b.r2.incomplete_capture")
    checks.equal(r3["captured_gpus"], r3["engine_gpus"], "anchor_480b.r3.complete_capture")
    checks.equal(r4["captured_gpus"], r4["engine_gpus"], "anchor_480b.r4.complete_capture")
    checks.equal(frozen["probe_over_generation_ratio"] >= 1, True, "anchor_480b.frozen.window_covers_generation")
    checks.equal(r3["probe_over_generation_ratio"] < 1, True, "anchor_480b.r3.window_misses_generation_start")
    checks.equal(r4["probe_over_generation_ratio"] > 1, True, "anchor_480b.r4.window_superset")
    checks.equal(
        frozen_block["tokens_per_s"]["median"],
        frozen["generation_window_tokens_per_s"],
        "anchor_480b.final_table_generation_throughput",
    )
    checks.equal(
        frozen_block["j_per_item"]["median"],
        frozen["probe_window_energy_j"] / 100,
        "anchor_480b.final_table_uncorrected_probe_energy",
    )
    replicated = [frozen, r3, r4]
    throughputs = [row["generation_window_tokens_per_s"] for row in replicated]
    throughput_mean = statistics.fmean(throughputs)
    deviations = [100 * (value / throughput_mean - 1) for value in throughputs]
    checks.equal(
        max(abs(value) for value in deviations) <= 1.51,
        True,
        "anchor_480b.replication_within_rounded_1.5pct_of_mean",
    )
    checks.equal([row["matched"] for row in replicated], [55, 55, 54], "anchor_480b.replication_matched")
    checks.equal(
        len({row["prompt_tokens"] for row in replicated}),
        1,
        "anchor_480b.replication_prompt_tokens_identical",
    )
    token_spread_pct = 100 * (max(row["total_tokens"] for row in replicated) / min(row["total_tokens"] for row in replicated) - 1)
    checks.equal(
        round(token_spread_pct, 1) == 0.2,
        True,
        "anchor_480b.replication_total_token_spread_rounds_to_0.2pct",
    )
    return {
        "final_convention": {
            "throughput_source": "frozen anchor total_tokens / generation window",
            "energy_source": "frozen anchor uncorrected probe-window total",
            "retests_used_for": "throughput and instrumentation validation only",
        },
        "runs": rows,
        "throughput_replication": {
            "included_runs": ["frozen_anchor", "r3_probe_started_late", "r4_probe_started_early"],
            "mean_tokens_per_s": throughput_mean,
            "deviation_from_mean_pct": deviations,
            "maximum_absolute_deviation_pct": max(abs(value) for value in deviations),
            "total_token_spread_pct": token_spread_pct,
        },
    }


def accuracy_pct(block: dict[str, Any]) -> float:
    return 100 * statistics.median(block["successful_requests_by_rep"]) / block["requests"]


def protocol_a_projection(
    tiers: dict[str, dict[str, dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main_rows: list[dict[str, Any]] = []
    sm_rows: list[dict[str, Any]] = []
    labels = {tier: label for tier, label, _, _, _ in PROTOCOL_A_SPECS}
    for tier, _, model_dir, _, has_fdrs in PROTOCOL_A_SPECS:
        c = tiers[tier]["protocol_a_C"]
        main_rows.append({
            "tier": "80B" if tier == "Next" else tier,
            "model": labels[tier],
            "model_dir": model_dir,
            "gpus": c["n_gpus"],
            "tp": c["tp_size"],
            "accuracy_pct": accuracy_pct(c),
            "throughput_tokens_per_s": c["tokens_per_s"]["median"],
            "energy_kj_per_successful_task": c["j_per_succ_task"]["median"] / 1000,
            "peak_vram_cluster_gb": c["peak_vram_node_sum_gb"],
        })
        row: dict[str, Any] = {
            "tier": "80B" if tier == "Next" else tier,
            "model": labels[tier],
            "model_dir": model_dir,
            "condition_C": {
                "accuracy_pct": accuracy_pct(c),
                "energy_j_per_item": c["j_per_item"]["median"],
                "energy_j_per_item_range": [c["j_per_item"]["min"], c["j_per_item"]["max"]],
                "energy_j_per_successful_task": c["j_per_succ_task"]["median"],
                "items_per_s": c["items_per_s"]["median"],
                "throughput_tokens_per_s": c["tokens_per_s"]["median"],
            },
            "gpus": c["n_gpus"],
            "nodes": c["n_nodes"],
            "tp": c["tp_size"],
            "peak_vram_per_gpu_gb": c["peak_vram_per_gpu_gb"],
            "node_power_w": c["avg_power_w_cluster"]["median"],
            "gpu_utilization_pct": c["avg_gpu_util_pct"]["median"],
        }
        if has_fdrs:
            fdrs = tiers[tier]["protocol_a_C_FDRS"]
            row["condition_C_FDRS"] = {
                "accuracy_pct": accuracy_pct(fdrs),
                "energy_j_per_item": fdrs["j_per_item"]["median"],
                "energy_j_per_successful_task": fdrs["j_per_succ_task"]["median"],
            }
            row["energy_ratio_C_FDRS_over_C"] = (
                fdrs["j_per_item"]["median"] / c["j_per_item"]["median"]
            )
        else:
            row["condition_C_FDRS"] = None
            row["energy_ratio_C_FDRS_over_C"] = None
        sm_rows.append(row)
    return main_rows, sm_rows


def protocol_b_projection(
    tiers: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    labels = {tier: label for tier, label, _, _, _ in PROTOCOL_A_SPECS}
    rows: list[dict[str, Any]] = []
    for tier in PROTOCOL_B_TIERS:
        block = tiers[tier]
        for concurrency in CONCURRENCY_LEVELS:
            level = block["concurrency_levels"][str(concurrency)]
            rows.append({
                "tier": tier,
                "model": labels[tier],
                "tp": block["tp_size"],
                "concurrency": concurrency,
                "ttft_ms": {
                    "median": level["ttft_ms"]["median"],
                    "p95": level["ttft_ms"]["p95"],
                },
                # The SM column header abbreviates "TPOT/ITL", but its two
                # displayed statistics are consistently ITL median and p95.
                "itl_ms": {
                    "median": level["tbt_itl_ms"]["median"],
                    "p95": level["tbt_itl_ms"]["p95"],
                },
                "e2e_latency_s": {
                    "median": level["e2e_latency_s"]["median"],
                    "p95": level["e2e_latency_s"]["p95"],
                },
                "request_throughput_req_s": level["request_throughput_req_s"],
                "output_throughput_tok_s": level["output_throughput_tok_s"],
                "energy_j_per_request_measured_window": level["energy"]["j_per_request_measured_window"],
                "energy_j_per_request_serving_lower_bound": level["energy"]["j_per_request_serving_lower_bound"],
                "cluster_power_w": level["energy"]["cluster_avg_power_w"],
                "gpu_utilization_pct": level["energy"]["avg_gpu_util_pct"],
                "completed": level["completed"],
                "failed": level["failed"],
            })
    return rows


FIELD_LINEAGE = {
    "protocol_a": {
        "accuracy": "median(aggregate.successful_requests) / aggregate.requests",
        "throughput_probe_window": "median(aggregate.total_tokens_per_s)",
        "throughput_480b_anchor": "aggregate.total_tokens / run_meta.window_wall_s (sole composite-field exception)",
        "items_per_s": "median(aggregate.requests_per_s)",
        "energy_j_per_item": "median(sum(energy.devices[].energy_j) / aggregate.requests)",
        "energy_j_per_successful_task": "median(sum(energy.devices[].energy_j) / aggregate.successful_requests)",
        "energy_ratio": "median(C_FDRS.j_per_item) / median(C.j_per_item)",
        "cluster_power": (
            "median(aggregate.cluster_avg_power_w), with each repetition "
            "cross-checked as sum_node(sum(device.energy_j) / node.duration_s)"
        ),
        "gpu_utilization": "median(mean(energy.devices[].avg_gpu_utilization_pct))",
        "peak_vram_per_gpu": "max(energy.devices[].peak_memory_used_bytes) / 1e9",
        "peak_vram_cluster": "max(aggregate.peak_memory_used_bytes_sum) / 1e9",
        "hardware": "aggregate.n_gpus/n_nodes plus run_meta.tp_size",
    },
    "protocol_b": {
        "latency_and_throughput": "bench_serve.json (cross-checked against run_meta.latency)",
        "measured_window_energy": "sum(energy.devices[].energy_j) / bench_serve.completed",
        "serving_lower_bound_energy": "total_energy_j * bench_duration / energy_window / completed",
        "cluster_power": (
            "aggregate.cluster_avg_power_w, cross-checked as "
            "sum_node(sum(device.energy_j) / node.duration_s)"
        ),
        "gpu_utilization": "mean(energy.devices[].avg_gpu_utilization_pct)",
    },
}


def rebuild(measurements: Path, expected_path: Path) -> tuple[dict[str, Any], bool]:
    source_paths: set[Path] = set()
    checks = RawChecks()
    rebuilt_a: dict[str, dict[str, dict[str, Any]]] = {}
    model_dirs: dict[str, str] = {}
    rep_count = 0
    node_energy_count_a = 0

    for tier, _, model_dir, n_reps, has_fdrs in PROTOCOL_A_SPECS:
        model_dirs[tier] = model_dir
        rebuilt_a[tier] = {}
        conditions = ("C", "C_FDRS") if has_fdrs else ("C",)
        for condition in conditions:
            cell_dir = measurements / "protocolA" / model_dir / condition
            rebuilt_a[tier][f"protocol_a_{condition}"] = rebuild_protocol_a_cell(
                cell_dir, n_reps, tier, condition, checks, source_paths
            )
            rep_count += n_reps
            node_energy_count_a += sum(
                len(list((cell_dir / f"rep{rep}").glob("energy_*.json")))
                for rep in range(1, n_reps + 1)
            )

    anchor_480b_evidence = validate_480b_anchor_evidence(
        measurements,
        rebuilt_a["480B"]["protocol_a_C"],
        checks,
        source_paths,
    )
    anchor_retest_energy_count = sum(
        row["node_energy_records"]
        for row in anchor_480b_evidence["runs"]
        if row["name"] != "frozen_anchor"
    )

    def device_count(paths: list[Path]) -> int:
        return sum(len(load_json(path)["devices"]) for path in paths)

    formal_a_energy_paths = sorted(
        (measurements / "protocolA").glob("**/energy_*.json")
    )
    retest_energy_paths = sorted(
        (measurements / "protocolA_anchor_retest").glob("**/energy_*.json")
    )

    rebuilt_b: dict[str, dict[str, Any]] = {}
    node_energy_count_b = 0
    for tier in PROTOCOL_B_TIERS:
        model_root = measurements / "protocolB" / model_dirs[tier]
        rebuilt_b[tier] = rebuild_protocol_b_cell(
            model_root, tier, checks, source_paths
        )
        node_energy_count_b += sum(
            len(list((model_root / f"conc{level}").glob("energy_*.json")))
            for level in CONCURRENCY_LEVELS
        )
    formal_b_energy_paths = sorted(
        (measurements / "protocolB").glob("**/energy_*.json")
    )

    expected = load_json(expected_path)
    expected_hash = sha256_file(expected_path)
    block_differences: list[dict[str, Any]] = []
    n_block_leaves = 0
    for tier, conditions in rebuilt_a.items():
        for key, block in conditions.items():
            n_block_leaves += compare_values(
                block,
                expected["tiers"][tier][key],
                f"frozen_table.tiers.{tier}.{key}",
                block_differences,
            )
    for tier, block in rebuilt_b.items():
        n_block_leaves += compare_values(
            block,
            expected["tiers"][tier]["protocol_b"],
            f"frozen_table.tiers.{tier}.protocol_b",
            block_differences,
        )

    main_rows, sm_a_rows = protocol_a_projection(rebuilt_a)
    expected_a = {
        tier: {
            key: expected["tiers"][tier][key]
            for key in conditions
        }
        for tier, conditions in (
            (tier, tuple(rebuilt_a[tier])) for tier in rebuilt_a
        )
    }
    expected_main_rows, expected_sm_a_rows = protocol_a_projection(expected_a)
    expected_b = {tier: expected["tiers"][tier]["protocol_b"] for tier in PROTOCOL_B_TIERS}
    sm_b_rows = protocol_b_projection(rebuilt_b)
    expected_sm_b_rows = protocol_b_projection(expected_b)
    projection_differences: list[dict[str, Any]] = []
    n_projection_leaves = 0
    n_projection_leaves += compare_values(
        main_rows, expected_main_rows, "main_T11", projection_differences
    )
    n_projection_leaves += compare_values(
        sm_a_rows, expected_sm_a_rows, "supplement_S16", projection_differences
    )
    n_projection_leaves += compare_values(
        sm_b_rows, expected_sm_b_rows, "supplement_S17", projection_differences
    )

    source_manifest = [
        {
            "path": release_path(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(source_paths, key=lambda path: release_path(path))
    ]
    all_differences = checks.differences + block_differences + projection_differences
    passed = not all_differences
    artifact = {
        "schema_version": 1,
        "description": "Raw-measurement rebuild of manuscript serving Tables T11, S16, and S17",
        "status": "PASS" if passed else "FAIL",
        "coverage": {
            "protocol_a": {
                "tiers": len(PROTOCOL_A_SPECS),
                "operating_point_cells": sum(2 if spec[4] else 1 for spec in PROTOCOL_A_SPECS),
                "repetition_directories": rep_count,
                "node_energy_records": node_energy_count_a,
                "gpu_device_records": device_count(formal_a_energy_paths),
                "three_rep_dual_condition_tiers": sum(1 for spec in PROTOCOL_A_SPECS if spec[3] == 3),
                "single_rep_anchor_tiers": ["405B", "480B"],
            },
            "protocol_a_480b_retests_not_used_as_table_energy": {
                "attempts": 3,
                "node_energy_records": anchor_retest_energy_count,
                "gpu_device_records": device_count(retest_energy_paths),
                "note": "r2/r3/r4 remain separate from the 62 formal Protocol-A table-input probe files",
            },
            "protocol_b": {
                "tiers": len(PROTOCOL_B_TIERS),
                "concurrency_levels": list(CONCURRENCY_LEVELS),
                "cells": len(PROTOCOL_B_TIERS) * len(CONCURRENCY_LEVELS),
                "node_energy_records": node_energy_count_b,
                "gpu_device_records": device_count(formal_b_energy_paths),
                "completed_requests_per_cell": 200,
            },
            "total_hashed_raw_files": len(source_manifest),
        },
        "field_lineage": FIELD_LINEAGE,
        "raw_record_consistency": {
            "checked_leaves": checks.n_checked,
            "differences": checks.differences,
            "passed": not checks.differences,
        },
        "frozen_table_crosscheck": {
            "expected_path": release_path(expected_path),
            "expected_sha256": expected_hash,
            "checked_leaves": n_block_leaves,
            "differences": block_differences,
            "passed": not block_differences,
        },
        "manuscript_projection_crosscheck": {
            "checked_leaves": n_projection_leaves,
            "differences": projection_differences,
            "passed": not projection_differences,
        },
        "manuscript_tables": {
            "main_T11_protocol_a_condensed": main_rows,
            "supplement_S16_protocol_a_full": sm_a_rows,
            "supplement_S17_protocol_b_full": sm_b_rows,
        },
        "recomputed_blocks": {
            "protocol_a": rebuilt_a,
            "protocol_b": rebuilt_b,
        },
        "anchor_480b_evidence": anchor_480b_evidence,
        "raw_inputs": source_manifest,
        "all_differences": all_differences,
    }
    return artifact, passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurements",
        type=Path,
        default=DEFAULT_MEASUREMENTS,
        help=f"public measurement root (default: {DEFAULT_MEASUREMENTS})",
    )
    parser.add_argument(
        "--expected-table",
        type=Path,
        default=DEFAULT_EXPECTED,
        help=f"frozen table used only for cross-checking (default: {DEFAULT_EXPECTED})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"deterministic JSON output (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact, passed = rebuild(args.measurements.resolve(), args.expected_table.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    coverage = artifact["coverage"]
    print(
        f"Protocol A: {coverage['protocol_a']['operating_point_cells']} cells / "
        f"{coverage['protocol_a']['repetition_directories']} repetitions; "
        f"Protocol B: {coverage['protocol_b']['cells']} cells"
    )
    print(
        f"raw consistency={artifact['raw_record_consistency']['passed']}; "
        f"frozen blocks={artifact['frozen_table_crosscheck']['passed']}; "
        f"T11/S16/S17 projections={artifact['manuscript_projection_crosscheck']['passed']}"
    )
    print(f"wrote {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
