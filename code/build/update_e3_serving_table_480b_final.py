# --------------------------------------------------------------------------
# Repository copy of scripts/build/update_e3_serving_table_480b_final.py, with
# path constants adapted to this repository's layout. It runs here against the
# archived artifacts; see ARTIFACT_INDEX.md.
# --------------------------------------------------------------------------
"""One-off: set the FINAL 480B protocol_a_C anchor block of e3_serving_table.json.

Context (E3_serving_metrics.md section 8, 2026-08-02 "r4 出数 + 锚点终裁"): after
three anchor re-runs, the ratified anchor is a composite, estimate-free convention:

- Throughput = exact token count over the *generation* window of the frozen anchor
  (job 5786611): 169,068 tok / 149.316 s = 1,132.3 tok/s. This replaces the frozen
  probe-window figure (938.8 tok/s, denominator polluted by the 26.04 s stop skew,
  anomaly A-1) and the retracted r3 provisional figure (1,254.1 tok/s, probe window
  covered only 92.8% of the generation window).
- Energy and every other field = the frozen anchor's uncorrected probe-window
  values (J/req 7,125, J/succ 12,955, total 712,539 J, VRAM
  100.5/1,605.8 GB, ...). The four probes all cover the 149.316 s generation
  call; their median duration is 180.09 s (+20.6%) because of a 26.04 s stop
  skew. The duration excess is not interpreted as an energy percentage.

Replication evidence (generation-window throughput; each run is within +/-1.5%
of the three-run mean):
frozen 5786611 = 1,132.3 | r3 5867046 = 1,164.2 | r4 5868153 = 1,144.5 tok/s;
matched 55/55/54 of 100; prompt tokens identical; total tokens within 0.2%.
r2 (5865397) incomplete single-node capture, quarantined. r3 quarantined
(rep1_r3_probe_started_late: late probe start -> energy under-read, throughput
over-read, favourable direction, not usable). r4 kept in place as throughput
replication + instrument validation (start-side handshake; stop-side scancel-TERM
already validated in r3): its early probe start (~53 s) inflates energy, so its
energy figures are not used either.

The original table builder was a one-off not preserved in-repo; this script mirrors
its per-field formulas exactly (verified bit-identical against the frozen block
before the r3 replacement) and rebuilds only the 480B protocol_a_C block. No other
cell is touched. Replaces update_e3_serving_table_480b_r3.py (deleted).

Usage: python3 update_e3_serving_table_480b_final.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
TABLE = os.path.join(REPO_ROOT, "serving", "measurement_json", "e3_serving_table.json")
MODEL = "Qwen_Qwen3-Coder-480B-A35B-Instruct"
FROZEN = os.path.join(REPO_ROOT, "serving", "measurement_json", "protocolA", MODEL, "C", "rep1")
RETEST = os.path.join(REPO_ROOT, "serving", "measurement_json", "protocolA_anchor_retest", MODEL, "C")
R3 = os.path.join(RETEST, "rep1_r3_probe_started_late")
R4 = os.path.join(RETEST, "rep1")

AMENDMENT = (
    "480B protocol_a_C FINAL anchor (E3 sec.8 2026-08-02 ruling): composite, "
    "estimate-free convention. All fields are the frozen anchor's (job 5786611) "
    "probe-window values -- energy is therefore an uncorrected conservative "
    "measurement. All four probes cover the 149.316 s generation call; their "
    "median duration is 180.09 s (+20.6%) because of a 26.04 s stop skew "
    "(anomaly A-1). The duration excess is not treated as an energy correction. "
    "The sole exception is "
    "tokens_per_s, which reports the exact token count over the generation window "
    "(169,068 tok / 149.316 s = 1,132.3 tok/s), a window-alignment-proof convention. "
    "Each run is within +/-1.5% of the three-run mean generation-window "
    "throughput: frozen "
    "5786611 = 1,132.3, r3 5867046 = 1,164.2, r4 5868153 = 1,144.5 tok/s (matched "
    "55/55/54, prompt tokens identical, total tokens within 0.2%). Run verdicts: "
    "r2 5865397 incomplete single-node capture, quarantined; r3 quarantined "
    "(probe window covered 92.8% of the generation window -> energy under-read, "
    "throughput over-read; provisional 1,254 tok/s / 10,864 J-per-succ retracted); "
    "r4 = clean window superset validating the start-side handshake (stop-side "
    "scancel-TERM validated in r3), kept as throughput replication only (its ~53 s "
    "early probe start inflates energy)."
)


def stat(value):
    """Single-rep stat block in the table's schema."""
    return {
        "max": value,
        "median": value,
        "min": value,
        "n": 1,
        "spread_ratio": 1.0,
        "values": [value],
    }


def load_rep(rep_dir):
    agg = json.load(open(os.path.join(rep_dir, "aggregate.json")))
    run_meta = json.load(open(os.path.join(rep_dir, "run_meta.json")))
    return agg, run_meta


def main() -> int:
    agg, run_meta = load_rep(FROZEN)
    devices = []
    for path in sorted(glob.glob(os.path.join(FROZEN, "energy_*.json"))):
        record = json.load(open(path))
        devices.extend(record["devices"])
    assert len(devices) == agg["n_gpus"] == 16, len(devices)
    assert run_meta["slurm_job_id"] == "5786611", run_meta["slurm_job_id"]

    generate_window_s = run_meta["window_wall_s"]
    total_tokens = agg["total_tokens"]
    requests = agg["requests"]
    tokens_per_s_gen = total_tokens / generate_window_s
    # Every node-level probe must cover the full generation duration, making
    # the uncorrected probe-window energy conservative.
    assert agg["representative_window_s"] >= generate_window_s

    # Replication check: r3/r4 generation-window throughput within 1.5% of frozen.
    for rep_dir, job in ((R3, "5867046"), (R4, "5868153")):
        r_agg, r_meta = load_rep(rep_dir)
        assert r_meta["slurm_job_id"] == job, r_meta["slurm_job_id"]
        r_tps = r_agg["total_tokens"] / r_meta["window_wall_s"]
        dev = abs(r_tps / tokens_per_s_gen - 1.0)
        assert dev <= 0.031, (job, r_tps)  # pairwise band; +/-1.5% around the mean
        print(f"replication {job}: {r_tps:.1f} tok/s (dev vs frozen {dev * 100:.2f}%)")

    block = {
        "avg_gpu_util_pct": stat(
            statistics.mean(d["avg_gpu_utilization_pct"] for d in devices)
        ),
        "avg_power_w_cluster": stat(agg["cluster_avg_power_w"]),
        "avg_power_w_per_gpu": stat(
            statistics.mean(d["avg_sampled_power_w"] for d in devices)
        ),
        "completion_tokens": stat(agg["completion_tokens"]),
        "duration_s_by_node": [agg["duration_s_by_node"]],
        "energy_methods": agg["energy_methods"],
        "generate_window_s": stat(generate_window_s),
        "items_per_s": stat(agg["requests_per_s"]),
        "items_per_s_generate_window": stat(requests / generate_window_s),
        "j_per_completion_token": stat(agg["j_per_completion_token"]),
        "j_per_item": stat(agg["j_per_request"]),
        "j_per_succ_task": stat(agg["j_per_successful_request"]),
        "j_per_total_token": stat(agg["j_per_total_token"]),
        "measured_gpu_indices": run_meta["measured_gpu_indices"],
        "model_load_s": run_meta["model_load_s"],
        "n_gpus": agg["n_gpus"],
        "n_nodes": agg["n_nodes"],
        "n_reps": 1,
        "peak_power_w_per_gpu": stat(max(d["peak_power_w"] for d in devices)),
        "peak_vram_node_sum_gb": agg["peak_memory_used_bytes_sum"] / 1e9,
        "peak_vram_per_gpu_gb": agg["peak_memory_used_bytes_max_device"] / 1e9,
        "pp_size": run_meta["pp_size"],
        "probe_mode": run_meta["probe_mode"],
        "probe_over_generate_window_ratio": stat(
            agg["representative_window_s"] / generate_window_s
        ),
        "prompt_tokens": stat(agg["prompt_tokens"]),
        "requests": requests,
        "start_skew_s": stat(agg["start_skew_s"]),
        "stop_skew_s": stat(agg["stop_skew_s"]),
        "successful_requests_by_rep": [agg["successful_requests"]],
        # FINAL convention: anchor throughput = generation-window figure.
        "tokens_per_s": stat(tokens_per_s_gen),
        "tokens_per_s_generate_window": stat(tokens_per_s_gen),
        "total_energy_j": stat(agg["total_energy_j"]),
        "tp_size": run_meta["tp_size"],
        "wall_span_s": stat(agg["wall_span_s"]),
        "warmup_s": run_meta["warmup_s"],
        "window_s": stat(agg["representative_window_s"]),
        "window_scope": run_meta["window_scope"],
    }

    table = json.load(open(TABLE))
    old = table["tiers"]["480B"]["protocol_a_C"]
    assert sorted(block) == sorted(old), (
        set(block) ^ set(old)
    )  # schema must be identical to the block being replaced
    table["tiers"]["480B"]["protocol_a_C"] = block
    meta = table.setdefault("meta", {})
    other_amendments = [
        note for note in meta.get("amendments", [])
        if not note.startswith("480B protocol_a_C FINAL anchor")
    ]
    meta["amendments"] = [AMENDMENT, *other_amendments]
    table["generated_from"].update({
        "ledger": "logs/e3_jobs.tsv",
        "protocol_a": "serving/measurement_json/protocolA",
        "protocol_a_480B_anchor": os.path.relpath(FROZEN, REPO_ROOT).replace(
            os.sep, "/"
        ),
        "protocol_b": "serving/measurement_json/protocolB",
    })

    with open(TABLE, "w", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("480B protocol_a_C set to FINAL composite anchor:")
    for key in (
        "tokens_per_s",
        "j_per_item",
        "j_per_succ_task",
        "total_energy_j",
        "avg_power_w_cluster",
        "avg_gpu_util_pct",
        "peak_vram_per_gpu_gb",
        "peak_vram_node_sum_gb",
    ):
        old_v = old[key]["median"] if isinstance(old[key], dict) else old[key]
        new_v = block[key]["median"] if isinstance(block[key], dict) else block[key]
        print(f"  {key}: {old_v} -> {new_v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
