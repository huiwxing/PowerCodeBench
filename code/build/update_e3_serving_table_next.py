# --------------------------------------------------------------------------
# Repository copy of scripts/build/update_e3_serving_table_next.py, with path
# constants adapted to this repository's layout. It runs here against the
# archived artifacts; see ARTIFACT_INDEX.md.
# --------------------------------------------------------------------------
"""Add the preregistered Qwen3-Coder-Next serving cells to the E3 table.

The script rebuilds the Protocol-A projection from the raw per-repetition
artifacts.  If the triggered Protocol-B cell has landed, it also renders the
same concurrency projection used by the existing tiers.  ``--validate-32b``
rebuilds the 32B cells and checks the renderer against the frozen table before
any file is written.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TABLE_PATH = ROOT / "serving" / "measurement_json" / "e3_serving_table.json"
PROTOCOL_A_ROOT = ROOT / "serving" / "measurement_json" / "protocolA"
PROTOCOL_B_ROOT = ROOT / "serving" / "measurement_json" / "protocolB"

NEXT_DIR = "Qwen_Qwen3-Coder-Next"
NEXT_AMENDMENT_PREFIX = "Qwen3-Coder-Next preregistered extension"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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


def energy_record(rep_dir: Path) -> dict[str, Any]:
    paths = sorted(rep_dir.glob("energy_*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one energy record in {rep_dir}, found {len(paths)}")
    return load_json(paths[0])


def render_protocol_a(model_dir: str, operating_point: str) -> dict[str, Any]:
    cell_dir = PROTOCOL_A_ROOT / model_dir / operating_point
    reps: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for rep in range(1, 4):
        rep_dir = cell_dir / f"rep{rep}"
        reps.append(
            (
                load_json(rep_dir / "aggregate.json"),
                load_json(rep_dir / "run_meta.json"),
                energy_record(rep_dir),
            )
        )

    aggregates = [entry[0] for entry in reps]
    metas = [entry[1] for entry in reps]
    energies = [entry[2] for entry in reps]

    def agg_values(key: str) -> list[float | int]:
        return [aggregate[key] for aggregate in aggregates]

    def token_values(key: str) -> list[float | int]:
        return [meta["tokens"][key] for meta in metas]

    avg_gpu_util = [
        statistics.fmean(device["avg_gpu_utilization_pct"] for device in energy["devices"])
        for energy in energies
    ]
    avg_power_per_gpu = [
        statistics.fmean(device["avg_sampled_power_w"] for device in energy["devices"])
        for energy in energies
    ]
    peak_power_per_gpu = [
        max(device["peak_power_w"] for device in energy["devices"])
        for energy in energies
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
    return {
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
        )
        / 1e9,
        "peak_vram_per_gpu_gb": max(
            device["peak_memory_used_bytes"]
            for energy in energies
            for device in energy["devices"]
        )
        / 1e9,
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


def render_protocol_b(model_dir: str) -> dict[str, Any] | None:
    cell_dir = PROTOCOL_B_ROOT / model_dir
    summary_path = cell_dir / "cell_summary.json"
    if not summary_path.exists():
        return None

    raw = load_json(summary_path)
    rendered: dict[str, Any] = {
        "client": "vllm bench serve",
        "concurrency_levels": {},
        "request_pool_n": None,
        "server_startup_s": None,
        "tp_size": raw["tp_size"],
        "warmup_s": None,
    }
    for level in raw["levels"]:
        concurrency = level["concurrency"]
        level_dir = cell_dir / f"conc{concurrency}"
        meta = load_json(level_dir / "run_meta.json")
        aggregate = load_json(level_dir / "aggregate.json")
        energy = energy_record(level_dir)
        latency = level["latency"]
        completed = latency["completed"]
        duration = latency["duration"]
        window = aggregate["representative_window_s"]
        overhead = window - duration

        if rendered["request_pool_n"] is None:
            rendered["request_pool_n"] = meta["request_pool"]["pool_size"]
            rendered["server_startup_s"] = meta["server_startup_s"]
            rendered["warmup_s"] = meta["warmup_s"]

        rendered["concurrency_levels"][str(concurrency)] = {
            "bench_duration_s": duration,
            "completed": completed,
            "e2e_latency_s": {
                "mean": latency["mean_e2el_ms"] / 1000,
                "median": latency["median_e2el_ms"] / 1000,
                "p95": latency["p95_e2el_ms"] / 1000,
            },
            "energy": {
                "avg_gpu_util_pct": statistics.fmean(
                    device["avg_gpu_utilization_pct"] for device in energy["devices"]
                ),
                "avg_power_w_per_gpu": statistics.fmean(
                    device["avg_sampled_power_w"] for device in energy["devices"]
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
                    device["peak_power_w"] for device in energy["devices"]
                ),
                "peak_vram_per_gpu_gb": max(
                    device["peak_memory_used_bytes"] for device in energy["devices"]
                )
                / 1e9,
                "total_energy_j": aggregate["total_energy_j"],
            },
            "energy_window_s": window,
            "failed": meta["config"]["num_prompts"] - completed,
            "num_prompts": meta["config"]["num_prompts"],
            "output_throughput_tok_s": latency["output_throughput"],
            "request_throughput_req_s": latency["request_throughput"],
            "tbt_itl_ms": {
                "mean": latency["mean_itl_ms"],
                "median": latency["median_itl_ms"],
                "p95": latency["p95_itl_ms"],
            },
            "total_input_tokens": latency["total_input_tokens"],
            "total_output_tokens": latency["total_output_tokens"],
            "total_token_throughput_tok_s": latency["total_token_throughput"],
            "tpot_ms": {
                "mean": latency["mean_tpot_ms"],
                "median": latency["median_tpot_ms"],
                "p95": latency["p95_tpot_ms"],
            },
            "ttft_ms": {
                "mean": latency["mean_ttft_ms"],
                "median": latency["median_ttft_ms"],
                "p95": latency["p95_ttft_ms"],
            },
            "window_overhead_frac": overhead / window,
            "window_overhead_s": overhead,
        }
    return rendered


def close_enough(left: Any, right: Any, path: str = "root") -> list[str]:
    errors: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            errors.append(f"{path}: keys differ: {left.keys()} != {right.keys()}")
            return errors
        for key in left:
            errors.extend(close_enough(left[key], right[key], f"{path}.{key}"))
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path}: lengths differ: {len(left)} != {len(right)}"]
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            errors.extend(close_enough(left_item, right_item, f"{path}[{index}]"))
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
            errors.append(f"{path}: {left} != {right}")
    elif left != right:
        errors.append(f"{path}: {left!r} != {right!r}")
    return errors


def validate_32b(table: dict[str, Any]) -> None:
    frozen = table["tiers"]["32B"]
    checks = {
        "protocol_a_C": render_protocol_a("Qwen_Qwen2.5-Coder-32B-Instruct", "C"),
        "protocol_a_C_FDRS": render_protocol_a(
            "Qwen_Qwen2.5-Coder-32B-Instruct", "C_FDRS"
        ),
        "protocol_b": render_protocol_b("Qwen_Qwen2.5-Coder-32B-Instruct"),
    }
    errors: list[str] = []
    for key, rebuilt in checks.items():
        errors.extend(close_enough(rebuilt, frozen[key], f"32B.{key}"))
    if errors:
        raise RuntimeError("32B renderer validation failed:\n" + "\n".join(errors[:30]))
    print("32B renderer validation: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-32b", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    table = load_json(TABLE_PATH)
    if args.validate_32b:
        validate_32b(table)

    next_b = render_protocol_b(NEXT_DIR)
    table["tiers"]["Next"] = {
        "is_anchor": False,
        "kv_cache_gib": 8.28,
        "kv_cache_tokens": 180608,
        "model": "Qwen3-Coder-Next",
        "model_dir": NEXT_DIR,
        "protocol_a_C": render_protocol_a(NEXT_DIR, "C"),
        "protocol_a_C_FDRS": render_protocol_a(NEXT_DIR, "C_FDRS"),
        "protocol_b": next_b,
    }
    amendments = table.setdefault("meta", {}).setdefault("amendments", [])
    amendments[:] = [
        amendment
        for amendment in amendments
        if not amendment.startswith(NEXT_AMENDMENT_PREFIX)
    ]
    amendments.append(
        f"{NEXT_AMENDMENT_PREFIX} (jobs 5876965/5876967): Protocol A uses "
        "the frozen n=300 sample, seed 22, three repetitions, TP=2; the "
        f"triggered Protocol B projection is {'included' if next_b else 'pending'}."
    )

    if args.write:
        with TABLE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(table, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"updated {TABLE_PATH}")
    else:
        c = table["tiers"]["Next"]["protocol_a_C"]
        fdrs = table["tiers"]["Next"]["protocol_a_C_FDRS"]
        print(
            "Next preview: "
            f"C={c['successful_requests_by_rep']} matched, "
            f"{c['tokens_per_s']['median']:.1f} tok/s, "
            f"{c['j_per_succ_task']['median']:.1f} J/success; "
            f"C_FDRS={fdrs['successful_requests_by_rep']} matched, "
            f"Protocol B={'ready' if next_b else 'pending'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
