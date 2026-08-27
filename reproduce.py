#!/usr/bin/env python3
"""Recompute the paper's headline number groups from the frozen artifacts.

CPU-only, Python 3 standard library, no network.  Each check recomputes a
number group from the archived artifacts in this repository and diffs it
against the expected values recorded in ARTIFACT_INDEX.md (which mirror the
manuscript).  Exit code 0 iff every check passes.

Usage:  python3 reproduce.py            # run all number groups
        python3 reproduce.py --group 3  # run only number group 3
        python3 reproduce.py --help     # list the groups without executing
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

if sys.version_info < (3, 9):  # the analysis scripts use 3.9 syntax/APIs
    sys.exit("reproduce.py requires Python >= 3.9 (found %d.%d)"
             % sys.version_info[:2])

ROOT = Path(__file__).resolve().parent
RESULTS = []  # (check, item, expected, got, ok)


def load(rel):
    return json.loads((ROOT / rel).read_text())


def record(check, item, expected, got, ok=None):
    if ok is None:
        ok = expected == got
    RESULTS.append((check, item, expected, got, bool(ok)))
    return ok


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def rebuild_json(check, script_rel, frozen_rel, output_flag="--output",
                 extra_args=()):
    """Run one public analysis script and compare its complete JSON result."""
    frozen = load(frozen_rel)
    with tempfile.TemporaryDirectory() as td:
        output = Path(td, Path(frozen_rel).name)
        command = [sys.executable, str(ROOT / script_rel),
                   *map(str, extra_args), output_flag, str(output)]
        proc = subprocess.run(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        record(check, f"{script_rel} exits successfully", 0, proc.returncode)
        if proc.returncode != 0 or not output.exists():
            detail = proc.stderr.decode(errors="replace")[-2000:]
            record(check, f"{frozen_rel} output exists", True,
                   (output.exists(), detail), False)
            return None
        rebuilt = json.loads(output.read_text())
    record(check, f"{frozen_rel} rebuilt field-for-field", True,
           rebuilt == frozen)
    return rebuilt


# ---------------------------------------------------------------------------
# Check 1 - headline range: proactive+reactive lifts every evaluated >=7B
# open-weight model and every API by +32 to +56 accuracy points over A.
# Lift = C+FDRS endpoint accuracy - baseline A accuracy (2,000 items).
# ---------------------------------------------------------------------------
def check1():
    name = "1 headline +32..+56pp"
    pro = load("results/aggregates/api_comparison_2000/proactive_comparison.json")
    rea = load("results/aggregates/api_comparison_2000/reactive_comparison.json")
    floor_models = {"Qwen_Qwen2.5-Coder-0.5B-Instruct", "Qwen_Qwen2.5-Coder-1.5B-Instruct"}
    expected_lifts = {
        "Qwen_Qwen2.5-Coder-7B-Instruct": 32.40,
        "meta-llama_Llama-3.1-8B-Instruct": 35.80,
        "Qwen_Qwen2.5-Coder-14B-Instruct": 47.00,
        "Qwen_Qwen2.5-Coder-32B-Instruct": 48.95,
        "meta-llama_Llama-3.1-70B-Instruct": 52.45,
        "openai_gpt-oss-120b": 46.20,
        "Qwen_Qwen3-Coder-Next": 55.55,
        "meta-llama_Llama-3.1-405B-Instruct": 55.25,
        "Qwen_Qwen3-Coder-480B-A35B-Instruct": 50.85,
        "claude-haiku-4-5": 53.40,
        "deepseek-v4-flash": 44.85,
        "gemini-2.5-flash": 47.95,
        "gpt-5.4-mini": 37.90,
    }
    lifts = {}
    for model, arow in pro["A"].items():
        if model in floor_models:
            continue
        fdrs = rea["C_FDRS"][model]["result_accuracy"]
        lifts[model] = round((fdrs - arow["result_accuracy"]) * 100, 2)
    ok_models = set(lifts) == set(expected_lifts) and all(
        close(lifts[m], expected_lifts[m], 1e-6) for m in expected_lifts)
    record(name, "per-model C_FDRS-A lifts (13 models)",
           "as ARTIFACT_INDEX", "match" if ok_models else lifts, ok_models)
    lo, hi = min(lifts.values()), max(lifts.values())
    record(name, "range rounds to [+32, +56]", (32, 56),
           (math.floor(lo), math.ceil(hi)))


# ---------------------------------------------------------------------------
# Check 2 - token economy: full proactive C reaches parity at 41% of X's
# prompt-token cost (panel mean of per-model C/X prompt-token ratios).
# ---------------------------------------------------------------------------
def check2():
    name = "2 token economy 41%"
    te = load("results/aggregates/token_economy.json")
    ratios = [r["c_over_x"] for r in te["rows"]]
    mean = sum(ratios) / len(ratios)
    record(name, "panel mean C/X recomputed == stored",
           round(te["panel_mean_c_over_x"], 12), round(mean, 12),
           close(mean, te["panel_mean_c_over_x"], 1e-9))
    record(name, "panel mean rounds to 41%", 41, round(mean * 100))
    per_row = all(close(r["c_over_x"],
                        r["C_avg_prompt_tokens"] / r["X_avg_prompt_tokens"], 1e-6)
                  for r in te["rows"])
    record(name, "per-row ratio == C/X prompt tokens", True, per_row)


# ---------------------------------------------------------------------------
# Check 3 - E1-P 170-pair McNemar suite: full recomputation from the frozen
# adjudication chain via code/audit/pair_mcnemar.py, diffed field-for-field
# against the frozen audit/e1p_mcnemar.json.
# ---------------------------------------------------------------------------
def check3():
    name = "3 McNemar 170 pairs"
    frozen = load("audit/e1p_mcnemar.json")
    with tempfile.TemporaryDirectory() as td:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "code/audit/pair_mcnemar.py")],
            cwd=td, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            record(name, "pair_mcnemar.py run", "exit 0", proc.returncode, False)
            return
        recomputed = json.loads(Path(td, "e1p_mcnemar.recomputed.json").read_text())
    record(name, "recomputed suite == frozen artifact (all fields)",
           True, recomputed == frozen)
    p = frozen["primary_sid_included_as_clean"]
    record(name, "primary n_pairs", 170, p["n_pairs"])
    record(name, "primary 2x2 (both,C,A,neither)", (17, 4, 41, 108),
           (p["table"]["both"], p["table"]["C_only"],
            p["table"]["A_only"], p["table"]["neither"]))
    record(name, "delta A-C pp / CI95", (21.76, [14.71, 28.82]),
           (p["delta_A_minus_C_pp"], p["delta_ci95_pp"]))
    record(name, "exact McNemar p ~ 9.33e-09", True,
           close(p["mcnemar_exact_p"], 9.334883088740753e-09, 1e-20))


# ---------------------------------------------------------------------------
# Check 4 - E2 backend-transfer minibench: the 24-cell accuracy matrix
# (2 backends x 3 models x A/Rsem/RsemB/C), plus a full recomputation of the
# OpenDSS C-recal endpoints, paired contrasts, exact McNemar tests, total-prompt
# ratios, and accessor-surface mechanism counts from nine frozen per-item runs.
# ---------------------------------------------------------------------------
def check4():
    name = "4 E2 24 cells + C-recal"
    agg = load("results/aggregates/e2_bench_v2_aggregate.json")
    expected_cells = {
        ("opendss", "Qwen_Qwen2.5-Coder-32B-Instruct"): {"A": 0.1778, "Rsem": 0.6667, "RsemB": 0.4111, "C": 0.1556},
        ("opendss", "deepseek-v4-flash"): {"A": 0.0667, "Rsem": 0.7556, "RsemB": 0.6, "C": 0.3778},
        ("opendss", "meta-llama_Llama-3.1-70B-Instruct"): {"A": 0.0333, "Rsem": 0.6333, "RsemB": 0.3889, "C": 0.3667},
        ("pypsa", "Qwen_Qwen2.5-Coder-32B-Instruct"): {"A": 0.4, "Rsem": 0.5444, "RsemB": 0.5333, "C": 0.5778},
        ("pypsa", "deepseek-v4-flash"): {"A": 0.6889, "Rsem": 0.9444, "RsemB": 0.8333, "C": 0.8111},
        ("pypsa", "meta-llama_Llama-3.1-70B-Instruct"): {"A": 0.0556, "Rsem": 0.6, "RsemB": 0.5111, "C": 0.5},
    }
    n_cells, ok_vals, ok_counts = 0, True, True
    for backend, bd in agg["backends"].items():
        for model, md in bd["models"].items():
            for cond, cd in md["conditions"].items():
                o = cd["overall"]
                n_cells += 1
                exp = expected_cells[(backend, model)][cond]
                if not close(o["accuracy"], exp, 1e-9):
                    ok_vals = False
                if not close(o["accuracy"], round(o["n_match"] / o["n"], 4), 1e-9):
                    ok_counts = False
    record(name, "cell count", 24, n_cells)
    record(name, "24 accuracies == ARTIFACT_INDEX values", True, ok_vals)
    record(name, "accuracy == n_match/n in every cell", True, ok_counts)
    ok_runs = True
    for backend, run in (("opendss", "e2_opendss_bench_v2"), ("pypsa", "e2_pypsa_bench_v2")):
        pc = load(f"results/aggregates/{run}/proactive_comparison.json")
        for row in pc["models"]:
            for cond in ("A", "Rsem", "RsemB", "C"):
                a = agg["backends"][backend]["models"][row["model"]]["conditions"][cond]["overall"]["accuracy"]
                if not close(row[cond], a, 1e-9):
                    ok_runs = False
    record(name, "run-level comparisons agree with aggregate", True, ok_runs)
    frozen_crecal = load("results/aggregates/e2_crecal_paired.json")
    with tempfile.TemporaryDirectory() as td:
        recomputed_path = Path(td, "e2_crecal_paired.recomputed.json")
        proc = subprocess.run(
            [sys.executable,
             str(ROOT / "code/aggregate/aggregate_e2_crecal.py"),
             "--out", str(recomputed_path)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            record(name, "aggregate_e2_crecal.py run", "exit 0",
                   (proc.returncode, proc.stderr.decode(errors="replace")), False)
            return
        recomputed_crecal = json.loads(recomputed_path.read_text())
    record(name, "C-recal raw recomputation == frozen aggregate (all fields)",
           True, recomputed_crecal == frozen_crecal)

    models = recomputed_crecal["models"]
    expected_endpoints = {
        "Qwen_Qwen2.5-Coder-32B-Instruct": 0.4111,
        "meta-llama_Llama-3.1-70B-Instruct": 0.4444,
        "deepseek-v4-flash": 0.6889,
    }
    got_endpoints = {
        model: row["accuracy"]["C-recal"] for model, row in models.items()
    }
    record(name, "OpenDSS C-recal endpoints from 90 raw matches/cell",
           expected_endpoints, got_endpoints)

    expected_pairs = {
        "Qwen_Qwen2.5-Coder-32B-Instruct": {
            "C-recal-C": (25.56, [14.44, 36.67], 29, 6,
                           0.0001168418675661087),
            "C-recal-Rsem": (-25.56, [-38.89, -12.22], 9, 32,
                              0.000430857042374555),
        },
        "meta-llama_Llama-3.1-70B-Instruct": {
            "C-recal-C": (7.78, [-1.11, 16.67], 12, 5,
                           0.143463134765625),
            "C-recal-Rsem": (-18.89, [-31.11, -6.67], 10, 27,
                              0.007632078602910042),
        },
        "deepseek-v4-flash": {
            "C-recal-C": (31.11, [17.78, 44.44], 36, 8,
                           2.5448598648836196e-05),
            "C-recal-Rsem": (-6.67, [-15.56, 2.22], 5, 11,
                              0.210113525390625),
        },
    }
    got_pairs = {}
    for model, comparisons in expected_pairs.items():
        got_pairs[model] = {}
        for comparison in comparisons:
            row = models[model]["paired"][comparison]
            got_pairs[model][comparison] = (
                row["delta_pp"], row["ci95_pp"],
                row["discordant_treatment_only"],
                row["discordant_reference_only"],
                row["mcnemar_exact_p"],
            )
    record(name, "six paired deltas/CIs/discordances/exact McNemar p-values",
           expected_pairs, got_pairs)
    record(name, "all paired contrasts align the same 90 item IDs", True,
           all(comp["n_pairs"] == 90 for row in models.values()
               for comp in row["paired"].values()))

    expected_token_ratios = {
        "Qwen_Qwen2.5-Coder-32B-Instruct": 0.270060186593,
        "meta-llama_Llama-3.1-70B-Instruct": 0.271841296396,
        "deepseek-v4-flash": 0.287169087235,
    }
    got_token_ratios = {
        model: row["crecal_over_rsem_total_prompt_ratio"]
        for model, row in models.items()
    }
    record(name, "C-recal/Rsem total-prompt ratios (27--29%)",
           expected_token_ratios, got_token_ratios)
    raw_token_arithmetic = all(close(
        row["crecal_over_rsem_total_prompt_ratio"],
        row["prompt_tokens"]["C-recal"]["total"]
        / row["prompt_tokens"]["Rsem"]["total"], 1e-12)
        for row in models.values())
    record(name, "total-prompt ratios recompute from per-item token sums",
           True, raw_token_arithmetic)

    qwen_control = models[
        "Qwen_Qwen2.5-Coder-32B-Instruct"
    ]["control_device"]
    expected_qwen_mechanism = {
        "C": (17, {"AttributeError": 17}, 0),
        "Rsem": (6, {"AttributeError": 6}, 7),
        "C-recal": (0, {}, 9),
    }
    got_qwen_mechanism = {
        arm: (row["non_executable"], row["non_executable_error_types"],
              row["matched"])
        for arm, row in qwen_control.items()
    }
    record(name, "Qwen32B control-device mechanism counts",
           expected_qwen_mechanism, got_qwen_mechanism)

    legacy_crecal = load(
        "results/aggregates/e2_opendss_bench_v2_crecal/proactive_comparison.json")
    legacy_endpoints = {row["model"]: row["C"]
                        for row in legacy_crecal["models"]}
    record(name, "legacy C-recal endpoint summary agrees with raw data",
           got_endpoints, legacy_endpoints)


# ---------------------------------------------------------------------------
# Check 5 - A5-3 risk-weight grid: pairwise deltas recomputed from the
# per-arm accuracies; per-model max|delta|; deployed/riskw arm cross-check.
# ---------------------------------------------------------------------------
def check5():
    name = "5 weight grid max|d|"
    grid = load("results/aggregates/a53_weight_grid_summary.json")
    expected_max = {
        "Qwen_Qwen2.5-Coder-14B-Instruct": 0.85,
        "Qwen_Qwen2.5-Coder-32B-Instruct": 0.85,
        "meta-llama_Llama-3.1-70B-Instruct": 0.9,
        "openai_gpt-oss-120b": 0.95,
        "Qwen_Qwen3-Coder-480B-A35B-Instruct": 1.3,
    }
    ok_delta, ok_max, got_max = True, True, {}
    for model, md in grid["models"].items():
        acc = md["acc_pct"]
        mx = 0.0
        for pair, pd in md["pairwise"].items():
            a, b = pair.split("->")
            d = round(acc[b] - acc[a], 10)
            if not close(d, pd["delta_pp"], 1e-9):
                ok_delta = False
            mx = max(mx, abs(d))
        got_max[model] = round(mx, 10)
        if not close(mx, md["max_pairwise_abs_pp"], 1e-9):
            ok_max = False
    record(name, "pairwise deltas == acc differences", True, ok_delta)
    record(name, "recomputed max|d| == stored", True, ok_max)
    record(name, "per-model max|d| (pp)", expected_max, got_max,
           set(got_max) == set(expected_max)
           and all(close(got_max[m], expected_max[m], 1e-9) for m in expected_max))
    arms = {"uniform": "comparison_riskw_uniform",
            "l0heavy": "comparison_riskw_l0heavy",
            "l3heavy": "comparison_riskw_l3heavy"}
    ok_arms = True
    for arm, d in arms.items():
        pc = load(f"results/aggregates/{d}/proactive_comparison.json")
        for row in pc["models"]:
            if row["model"] in grid["models"]:
                if not close(row["C"] * 100, grid["models"][row["model"]]["acc_pct"][arm], 1e-6):
                    ok_arms = False
    record(name, "riskw arm files agree with grid accuracies", True, ok_arms)


# ---------------------------------------------------------------------------
# Check 6 - E4 v2 conditional adaptation on the four external sets: exit-a
# margin criterion, frozen @10 metrics, seed sensitivity, weight provenance.
# ---------------------------------------------------------------------------
def check6():
    name = "6 E4 v2 four sets"
    v2 = load("external_queries/conditional_selector/metrics_by_set_v2.json")
    expected = {
        "layer1_holdout_n80": {"unadapted": 0.5546, "adapted": 0.6894, "conditional": 0.6894},
        "layer2a_n84": {"unadapted": 0.5507, "adapted": 0.5343, "conditional": 0.6206},
        "layer2a_ext_n10": {"unadapted": 0.6472, "adapted": 0.5, "conditional": 0.6472},
        "layer2b_n20": {"unadapted": 0.4869, "adapted": 0.6452, "conditional": 0.6452},
    }
    got = {s: {a: v2["sets"][s]["metrics"][a]["10"]["recall"] for a in expected[s]}
           for s in expected}
    record(name, "recall@10, four sets x three arms", expected, got)
    exit_a = all(
        got[s]["conditional"] >= max(got[s]["adapted"], got[s]["unadapted"]) - 0.01
        for s in expected)
    record(name, "exit-a: conditional >= max(arm) - 1pp on all four sets",
           True, exit_a)
    mx = 0.0
    for seed in range(42, 47):
        ss = load("external_queries/conditional_selector/seed_sensitivity/"
                  f"metrics_by_set_seed{seed}.json")
        for s in expected:
            mx = max(mx, abs(ss["sets"][s]["metrics"]["conditional"]["10"]["recall"]
                             - got[s]["conditional"]))
    record(name, "seed 42-46 max |dev| of conditional recall@10",
           0.0658, round(mx, 4))
    pr = load("external_queries/conditional_selector/per_role_v2.json")
    record(name, "adapted role weights identical across artifacts",
           True, pr["provenance"]["role_weights_adapted"]
           == v2["arms"]["adapted"]["role_weights"])


# ---------------------------------------------------------------------------
# Check 7 - E3 480B anchor composite convention: throughput = exact tokens
# over the generation window of the frozen anchor; energy = frozen
# probe-window values; r4 retest replicates throughput.
# ---------------------------------------------------------------------------
def check7():
    name = "7 480B serving anchor"
    table = load("serving/measurement_json/e3_serving_table.json")
    row = table["tiers"]["480B"]["protocol_a_C"]
    base = "serving/measurement_json/protocolA/Qwen_Qwen3-Coder-480B-A35B-Instruct/C/rep1"
    agg = load(f"{base}/aggregate.json")
    meta = load(f"{base}/run_meta.json")
    tps = agg["total_tokens"] / meta["window_wall_s"]
    row_tps = row["tokens_per_s"]["median"]  # single-rep stat block
    record(name, "job id of frozen anchor", "5786611", meta["slurm_job_id"])
    record(name, "tokens_per_s == 169,068 tok / 149.316 s",
           round(row_tps, 6), round(tps, 6), close(tps, row_tps, 1e-6))
    record(name, "tokens_per_s ~ 1,132.3", 1132.3, round(tps, 1))
    record(name, "energy fields == frozen probe-window values",
           (round(agg["j_per_request"], 3), round(agg["j_per_successful_request"], 3)),
           (round(row["j_per_item"]["median"], 3),
            round(row["j_per_succ_task"]["median"], 3)))
    durations = agg["duration_s_by_node"].values()
    duration_excess = (
        agg["representative_window_s"] / meta["window_wall_s"] - 1.0
    ) * 100
    record(name, "all probes cover generation; median excess / stop skew",
           (True, 20.6, 26.04),
           (all(d >= meta["window_wall_s"] for d in durations),
            round(duration_excess, 1), round(agg["stop_skew_s"], 2)))
    r4 = "serving/measurement_json/protocolA_anchor_retest/Qwen_Qwen3-Coder-480B-A35B-Instruct/C/rep1"
    r4_agg, r4_meta = load(f"{r4}/aggregate.json"), load(f"{r4}/run_meta.json")
    r4_tps = r4_agg["total_tokens"] / r4_meta["window_wall_s"]
    record(name, "r4 retest replicates throughput within +/-3.1%",
           True, abs(r4_tps / tps - 1.0) <= 0.031)


# ---------------------------------------------------------------------------
# Check 8 - FAR suite: unweighted and design-weighted (Hajek) FAR recomputed
# from the per-item rulings + sampling cells; pooled gate operating
# characteristics; tolerance-tightening counts.
# ---------------------------------------------------------------------------
def check8():
    name = "8 FAR suite"
    labels = load("audit/t4_final_labels_v2.json")
    far = load("audit/t4_far_weighted.json")
    manifest = load("audit/e1_sample_manifest.json")

    counts = Counter((v["frame"], v["label"]) for v in labels.values())
    n_main = sum(c for (f, _), c in counts.items() if f == "main_P")
    defect = counts[("main_P", "item-defect")]
    contam = counts[("main_P", "contaminated")]
    unweighted = 100 * contam / (n_main - defect)
    record(name, "main_P unweighted FAR (84/375 = 22.4%)",
           far["far_by_frame"]["main_P"]["far_unweighted_pct"],
           round(unweighted, 2),
           close(unweighted, far["far_by_frame"]["main_P"]["far_unweighted_pct"], 0.005))

    def hajek(frame, cell_key):
        num = den = 0.0
        for item in manifest["samples"][frame]:
            key = f"{item['model']}|{item['condition']}|{item['bench_index']}"
            lab = labels.get(key)
            if lab is None or lab["label"] == "item-defect":
                continue
            w = far["cells"][cell_key(item)]["weight"]
            den += w
            if lab["label"] == "contaminated":
                num += w
        return 100 * num / den

    w_main = hajek("main_P", lambda it: f"main_P|{it['tier']}|{it['task']}")
    record(name, "main_P design-weighted FAR (Hajek) = 18.51%",
           far["far_by_frame"]["main_P"]["far_weighted_pct"], round(w_main, 2),
           close(w_main, far["far_by_frame"]["main_P"]["far_weighted_pct"], 0.005))
    w_api = hajek("api_T5", lambda it: f"api_T5|{it['model']}|{it['task']}")
    record(name, "api_T5 design-weighted FAR = 12.65%",
           far["far_by_frame"]["api_T5"]["far_weighted_pct"], round(w_api, 2),
           close(w_api, far["far_by_frame"]["api_T5"]["far_weighted_pct"], 0.005))

    gate = far["gate_operating_characteristics"]["defect_excluded"]
    pooled = tuple(sum(gate[f][k] for f in ("main_P", "api_T5", "contrast_S"))
                   for k in ("tp", "fp", "fn", "tn"))
    record(name, "pooled gate confusion (TP,FP,FN,TN)", (129, 170, 14, 259), pooled)
    record(name, "pooled == stored ALL_610 frame",
           (gate["ALL_610"]["tp"], gate["ALL_610"]["fp"],
            gate["ALL_610"]["fn"], gate["ALL_610"]["tn"]), pooled)
    tp, fp, fn, tn = pooled
    record(name, "pooled sensitivity 90.21% / PPV 43.14%",
           (gate["ALL_610"]["sensitivity_pct"], gate["ALL_610"]["precision_ppv_pct"]),
           (round(100 * tp / (tp + fn), 2), round(100 * tp / (tp + fp), 2)))

    tol = load("audit/t4_tolerance_sensitivity.json")
    rej = sum(1 for e in tol["contaminated"]
              if e["frame"] == "main_P" and e["would_fail_tight"])
    tt = far["tolerance_tightened_far"]["by_frame"]["main_P"]
    record(name, "main_P contaminated rejected by tight tolerance",
           tt["contaminated_before"] - tt["contaminated_after"], rej)
    record(name, "tightened main_P weighted FAR = 11.96%", 11.96,
           tt["far_weighted_pct"],
           close(tt["far_weighted_pct"], 11.96, 0.005))


# ---------------------------------------------------------------------------
# Check 9 - Qwen3-Coder-Next serving row (SM protocol-A record S22 / main-text
# serving table T10): accuracy, throughput and energy recomputed from the
# frozen per-repetition measurement cells and diffed against the frozen
# serving table digit-for-digit at the manuscript's rounding.
# ---------------------------------------------------------------------------
def check9():
    name = "9 Next serving row"
    table = load("serving/measurement_json/e3_serving_table.json")
    tier = table["tiers"]["Next"]
    record(name, "tier model / directory",
           ("Qwen3-Coder-Next", "Qwen_Qwen3-Coder-Next"),
           (tier["model"], tier["model_dir"]))
    c = tier["protocol_a_C"]
    fd = tier["protocol_a_C_FDRS"]
    base = "serving/measurement_json/protocolA/Qwen_Qwen3-Coder-Next"
    succ_c = [load(f"{base}/C/rep{i}/aggregate.json")["successful_requests"]
              for i in (1, 2, 3)]
    succ_fd = sorted(load(f"{base}/C_FDRS/rep{i}/aggregate.json")
                     ["successful_requests"] for i in (1, 2, 3))
    record(name, "C successful_requests, 3 reps (140/300)",
           [140, 140, 140], succ_c)
    record(name, "acc @C rounds to 46.7%", 46.7, round(succ_c[0] / 300 * 100, 1))
    record(name, "acc @C_FDRS rep median rounds to 57.7%",
           57.7, round(succ_fd[1] / 300 * 100, 1))
    tps = sorted(load(f"{base}/C/rep{i}/aggregate.json")["total_tokens_per_s"]
                 for i in (1, 2, 3))
    record(name, "table tok/s median == rep median (exact)",
           c["tokens_per_s"]["median"], tps[1])
    record(name, "throughput @C -> 5,568 tok/s (3.17 items/s)",
           (5568, 3.17),
           (round(c["tokens_per_s"]["median"]),
            round(c["items_per_s"]["median"], 2)))
    record(name, "J/item @C median [min-max] -> 238 [236-266]",
           (238, 236, 266),
           (round(c["j_per_item"]["median"]), round(c["j_per_item"]["min"]),
            round(c["j_per_item"]["max"])))
    record(name, "J/succ.task @C -> 510 (0.51 kJ/success)",
           (510, 0.51),
           (round(c["j_per_succ_task"]["median"]),
            round(c["j_per_succ_task"]["median"] / 1000, 2)))
    record(name, "C+FDRS J/item / J/succ.task -> 2,301 / 4,134",
           (2301, 4134),
           (round(fd["j_per_item"]["median"]),
            round(fd["j_per_succ_task"]["median"])))
    record(name, "E ratio E(C+FDRS)/E(C) rounds to 9.7x", 9.7,
           round(fd["j_per_item"]["median"] / c["j_per_item"]["median"], 1))
    record(name, "node power / util @C -> 754 W / 95.1%",
           (754, 95.1),
           (round(c["avg_power_w_cluster"]["median"]),
            round(c["avg_gpu_util_pct"]["median"], 1)))
    record(name, "peak VRAM per-GPU / node sum -> 101.8 / 203.6 GB",
           (101.8, 203.6),
           (round(c["peak_vram_per_gpu_gb"], 1),
            round(c["peak_vram_node_sum_gb"], 1)))
    cs = load(f"{base}/C/cell_summary.json")["reps"]
    record(name, "table medians == cell_summary medians (exact)", True,
           c["j_per_item"]["median"] == cs["j_per_request"]["median"]
           and c["j_per_succ_task"]["median"]
           == cs["j_per_successful_request"]["median"]
           and c["avg_power_w_cluster"]["median"]
           == cs["cluster_avg_power_w"]["median"])


# ---------------------------------------------------------------------------
# Check 10 - main-text Delta(C-A) hero CI: layer-wise condition C delivers a
# +30.95pp panel-mean accuracy gain over baseline A, 95% paired-bootstrap CI
# [+29.75, +32.13]pp (B=2,000 item-level resamples, seed 22, 10-model panel;
# frozen aggregation source mw2_panel_mean_paired_bootstrap_ci.json).
# ---------------------------------------------------------------------------
def check10():
    name = "10 hero C-A bootstrap CI"
    ci = load("results/aggregates/mw2_panel_mean_paired_bootstrap_ci.json")
    record(name, "protocol (B, seed, n_items, n_panel)",
           (2000, 22, 2000, 10),
           (ci["n_bootstrap"], ci["seed"], ci["n_items"], ci["n_panel"]))
    rows = [r for r in ci["rows"]
            if r["file_a"] == "benchmark_results_condC.json"
            and r["file_b"] == "benchmark_results_condA.json"]
    record(name, "exactly one frozen C-A row", 1, len(rows))
    if len(rows) != 1:
        return
    row = rows[0]
    record(name, "point estimate renders +30.95pp",
           "+30.95", f"{row['point_pp']:+.2f}")
    record(name, "95% CI renders [+29.75, +32.13]pp",
           ("+29.75", "+32.13"),
           (f"{row['ci_low_pp']:+.2f}", f"{row['ci_high_pp']:+.2f}"))
    record(name, "CI strictly positive", True,
           bool(row["strict_positive"]) and row["ci_low_pp"] > 0)
    panel = [
        "Qwen_Qwen2.5-Coder-1.5B-Instruct", "Qwen_Qwen2.5-Coder-7B-Instruct",
        "meta-llama_Llama-3.1-8B-Instruct", "Qwen_Qwen2.5-Coder-14B-Instruct",
        "Qwen_Qwen2.5-Coder-32B-Instruct", "meta-llama_Llama-3.1-70B-Instruct",
        "openai_gpt-oss-120b", "Qwen_Qwen3-Coder-Next",
        "meta-llama_Llama-3.1-405B-Instruct",
        "Qwen_Qwen3-Coder-480B-A35B-Instruct",
    ]
    pro = load("results/aggregates/comparison/proactive_comparison.json")
    acc = {r["model"]: r for r in pro["models"]}
    point = sum((acc[m]["C"] - acc[m]["A"]) * 100 for m in panel) / len(panel)
    record(name, "point == T4 panel-mean C-A (independent artifact)",
           round(row["point_pp"], 9), round(point, 9),
           close(point, row["point_pp"], 1e-9))


# ---------------------------------------------------------------------------
# Check 11 - matcher false-negative mechanism diagnostic.  The complete
# source records and historical aggregate are provenance-checked, while the
# public proactive/reactive aggregates are traversed once per unique
# (condition, model) cell.  The historical archive has no item-level ledger,
# so a separately versioned readjudication supplies one; its scope is set out
# in audit/matcher_fn/README.md.
# ---------------------------------------------------------------------------
def check11():
    name = "11 matcher-FN diagnostic + unique census"
    frozen = load("audit/matcher_fn/recomputed_summary.json")
    with tempfile.TemporaryDirectory() as td:
        recomputed_path = Path(td, "matcher_fn_summary.recomputed.json")
        proc = subprocess.run(
            [sys.executable,
             str(ROOT / "code/audit/recompute_matcher_fn_audit.py"),
             "--out", str(recomputed_path)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            record(name, "recompute_matcher_fn_audit.py run", "exit 0",
                   (proc.returncode,
                    proc.stderr.decode(errors="replace")), False)
            return
        report = json.loads(recomputed_path.read_text())
    record(name, "recomputed report == frozen report (all fields)",
           True, report == frozen)

    record(name, "independent validator status", "ok", report["status"])
    sample = report["sample_provenance"]
    record(name, "100 source records / 15 models / 15 task families",
           (100, 15, 15),
           (sample["n"], sample["n_models"], sample["n_task_families"]))
    record(name, "sample conditions (eight A/C-family arms)",
           ["A", "A_FD", "A_FDR", "A_FDRS",
            "C", "C_FD", "C_FDR", "C_FDRS"],
           sorted(sample["conditions"],
                  key=lambda c: (c[0], ("", "_FD", "_FDR", "_FDRS").index(
                      c[1:] if len(c) > 1 else ""))))
    record(name, "all samples source-hash verified and executed-unmatched",
           (True, True),
           (sample["all_source_record_hashes_verified"],
            sample["all_executed_but_unmatched"]))

    historical = report["historical_aggregate"]
    record(name, "historical aggregate counts (not inferred labels)",
           {"confirmed_false_negative": 8, "ambiguous": 5,
            "genuine_error": 87},
           historical["reported_counts"])
    record(name, "nominal unweighted Wilson 95% interval",
           (0.04109346148438063, 0.14998107700948732),
           tuple(historical["nominal_wilson95"]))
    record(name, "no item-level ledger and no population transport",
           (False, False),
           (sample["item_level_adjudication_present"],
            historical["population_transport_performed"]))

    census = report["unique_census"]
    record(name, "unique full-panel census",
           {"cells": 196, "total": 392000, "executed": 188297,
            "matched": 121487, "executed_but_unmatched": 66810,
            "parse_failures": 16551},
           census["full_unique_panel"])
    record(name, "eight-condition diagnostic-frame census",
           {"cells": 104, "total": 208000, "executed": 94445,
            "matched": 58999, "executed_but_unmatched": 35446,
            "parse_failures": 7824},
           census["eight_condition_diagnostic_frame"])
    record(name, "census does not transport the audit rate", False,
           census["audit_rate_transport_performed"])

    readjudication = rebuild_json(
        name,
        "code/audit/recompute_matcher_fn_readjudication.py",
        "audit/matcher_fn/revision_readjudication/adjudication_v2.json")
    if readjudication is None:
        return
    record(name, "revision readjudication FN / ambiguous / genuine-error",
           (8, 3, 89),
           (readjudication["tally"]["false_negative"],
            readjudication["tally"]["ambiguous"],
            readjudication["tally"]["genuine_error"]))
    record(name, "revision FN Wilson interval / parser-only mechanism",
           ([0.0411, 0.15], True),
           (readjudication["false_negative_wilson95"],
            readjudication[
                "all_upheld_false_negatives_are_parser_extraction_misses"]))
    agreement = readjudication["review_agreement"]
    record(name, "three-review unanimity / post-hoc historical FN marginals",
           (96, True),
           (agreement["three_way_unanimous"],
            readjudication["posthoc_historical_fn_marginal_check"]["all_match"]))


# ---------------------------------------------------------------------------
# Check 12 - Table 3 mid-tier parity: full paired-bootstrap and TOST
# recomputation from the five compact 2,000-item C_FDRS match vectors.  The
# aggregation script validates the item/outcome hashes and source provenance
# before applying the frozen item ordering, RNG, seed, and CI convention.
# ---------------------------------------------------------------------------
def check12():
    name = "12 P02 parity bootstrap + TOST"
    frozen = load("results/aggregates/p02_parity_tost.json")
    with tempfile.TemporaryDirectory() as td:
        recomputed_path = Path(td, "p02_parity_tost.recomputed.json")
        proc = subprocess.run(
            [sys.executable,
             str(ROOT / "code/aggregate/aggregate_p02_parity.py"),
             "--out", str(recomputed_path)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            record(name, "aggregate_p02_parity.py run", "exit 0",
                   (proc.returncode,
                    proc.stderr.decode(errors="replace")), False)
            return
        recomputed = json.loads(recomputed_path.read_text())
    record(name, "item-level recomputation == frozen aggregate (all fields)",
           True, recomputed == frozen)

    method = recomputed["method"]
    record(name, "protocol (condition, pairing, n, B, seed)",
           ("condC_FDRS", "item_id", 2000, 10000, 20260617),
           (method["condition"], method["pairing_key"],
            method["n_common_items"], method["n_bootstrap"], method["seed"]))
    expected_models = {
        "Llama-3.1-70B": (1149, 57.45),
        "GPT-OSS-120B": (1205, 60.25),
        "DeepSeek-V4-Flash": (1118, 55.90),
        "GPT-5.4-mini": (1243, 62.15),
        "Claude-Haiku-4-5": (1275, 63.75),
    }
    got_models = {
        model: (row["n_match"], row["accuracy_pct"])
        for model, row in recomputed["models"].items()
    }
    record(name, "five C_FDRS endpoints from 2,000 match outcomes each",
           expected_models, got_models)

    expected_contrasts = {
        "llama70_vs_nearest_deepseek":
            (1.55, 268, 237, [-0.30, 3.40], [-0.60, 3.75], True, True),
        "llama70_vs_strongest_claude":
            (-6.30, 185, 311, [-8.10, -4.50], [-8.45, -4.15], False, False),
        "gptoss120_vs_nearest_gpt54mini":
            (-1.90, 198, 236, [-3.60, -0.20], [-3.90, 0.15], True, True),
        "gptoss120_vs_strongest_claude":
            (-3.50, 183, 253, [-5.20, -1.80], [-5.55, -1.45], False, False),
    }
    got_contrasts = {}
    for key, row in recomputed["comparisons"].items():
        got_contrasts[key] = (
            row["point_pp"], row["discordant_treatment_only"],
            row["discordant_reference_only"], row["ci90_pp"],
            row["ci95_pp"], row["ci95_includes_zero"],
            row["tost_equivalent_alpha_0_05"],
        )
    record(name, "four paired deltas, discordances, CIs, and TOST decisions",
           expected_contrasts, got_contrasts)
    record(name, "nearest APIs span zero and are equivalent within +/-5pp",
           True,
           all(recomputed["comparisons"][key]["ci95_includes_zero"]
               and recomputed["comparisons"][key]
                   ["tost_equivalent_alpha_0_05"]
               for key in ("llama70_vs_nearest_deepseek",
                           "gptoss120_vs_nearest_gpt54mini")))
    record(name, "strongest-API contrasts are not TOST-equivalent",
           True,
           all(not recomputed["comparisons"][key]
                       ["tost_equivalent_alpha_0_05"]
               for key in ("llama70_vs_strongest_claude",
                           "gptoss120_vs_strongest_claude")))


# ---------------------------------------------------------------------------
# Check 13 - primary experiment tables rebuilt from the compact item-level
# archive (270 runs on shared 2,000/80-item axes).
# ---------------------------------------------------------------------------
def check13():
    name = "13 primary item-level rebuild"
    rebuilt = rebuild_json(
        name, "code/aggregate/aggregate_primary_results_compact.py",
        "results/aggregates/primary_results_from_compact.json",
        output_flag="--out")
    if rebuilt is None:
        return
    record(name, "raw-run coverage (total/comparison/API/RAG/naturalistic)",
           (270, 176, 20, 30, 44),
           (rebuilt["coverage"]["n_runs"],
            rebuilt["coverage"]["groups"]["comparison"],
            rebuilt["coverage"]["groups"]["api"],
            rebuilt["coverage"]["groups"]["bm25"]
            + rebuilt["coverage"]["groups"]["semantic"],
            rebuilt["coverage"]["groups"]["naturalistic"]))
    archive = load("results/raw/main_experiment/primary_outcomes_compact.json")
    record(name, "shared item axes", (2000, 80),
           (archive["datasets"]["powercodebench_2000"]["n_items"],
            archive["datasets"]["naturalistic_holdout_80"]["n_items"]))
    hero = rebuilt["bootstrap"]["hero"]
    record(name, "hero paired delta and CI from item outcomes",
           (4.9466, [2.95, 6.95]),
           (hero["paired_delta_bootstrap_mean_pp"],
            hero["paired_delta_ci95_pp"]))
    record(name, "C/X prompt-token panel ratio", 0.41289449696000796,
           rebuilt["T6_token_economy"]
                  ["panel_mean_of_per_model_C_over_X"])
    pooled = rebuilt["T6_token_economy"]["pooled_exact_display"]
    proactive_rows = {
        condition: (
            round(row["docs_avg"]), round(row["prompt_avg"]),
            round(row["prompt_tokens_per_success"]),
            round(100 * row["vs_x_prompt_ratio"]),
        ) for condition, row in pooled["proactive"].items()
    }
    reactive_rows = {
        condition: (
            round(row["docs_avg"]), round(row["prompt_avg"]),
            round(row["prompt_tokens_per_success"]),
            round(100 * row["vs_x_prompt_ratio"]),
        ) for condition, row in pooled["reactive"].items()
    }
    record(name, "Table 6 proactive pooled exact display",
           {"A": (0, 182, 3149, 5), "B": (65, 248, 2760, 7),
            "C": (1269, 1451, 3950, 41),
            "X": (3331, 3513, 9214, 100),
            "R": (4007, 4189, 38133, 119)}, proactive_rows)
    record(name, "Table 6 reactive pooled exact display",
           {"C_FX": (0, 484, 9881, 14),
            "C_FD": (1375, 1864, 30918, 53),
            "C_FDR": (262, 748, 12427, 21),
            "C_FDRS": (518, 1003, 13551, 29)}, reactive_rows)


# ---------------------------------------------------------------------------
# Check 14 - supplementary profile, demand, E4 and router tables rebuilt from
# public per-function/per-query/per-run compact evidence.
# ---------------------------------------------------------------------------
def check14():
    name = "14 supplementary evidence rebuild"
    probe = rebuild_json(
        name, "code/aggregate/aggregate_probe_profiles.py",
        "results/aggregates/supplementary_probe_profiles.json")
    demand = rebuild_json(
        name, "code/aggregate/aggregate_demand_evidence.py",
        "results/aggregates/supplementary_demand_evidence.json")
    router = rebuild_json(
        name, "code/aggregate/aggregate_reactive_router_shares.py",
        "results/aggregates/supplementary_reactive_router_shares.json")
    if probe is not None:
        record(name, "probe table cells (14 models x 4 layers)", 56,
               sum(len(row) for row in probe["mean_probe_scores"].values()))
    if demand is not None:
        record(name, "all archived E4 item-level cross-checks", True,
               demand["all_e4_archived_checks_pass"])
        s4b = demand["s4_role_reweighting_deployed_hybrid"]
        record(name, "S4(b) frozen hybrid integrity / selected alphas",
               (True, 0.7, 0.4),
               (s4b["all_integrity_checks_pass"],
                s4b["arms"]["before"]["selected_alpha"],
                s4b["arms"]["after"]["selected_alpha"]))
        record(name, "S4(b) Aug-test / Bench-ref recall shifts (pp)",
               (-5.14, 20.69),
               (s4b["recall_at_10"]["Aug-test"]
                   ["delta_percentage_points"],
                s4b["recall_at_10"]["Bench-ref"]
                   ["delta_percentage_points"]))
    if router is not None:
        record(name, "reactive-router displayed share cells", 70,
               len(router["rows"]) * 7)


# ---------------------------------------------------------------------------
# Check 15 - full transfer reconstruction: 24 cold-start cells plus C-recal.
# ---------------------------------------------------------------------------
def check15():
    name = "15 full E2 transfer rebuild"
    rebuilt = rebuild_json(
        name, "code/aggregate/aggregate_e2_transfer.py",
        "results/aggregates/e2_transfer_full_recomputed.json",
        output_flag="--out")
    if rebuilt is None:
        return
    cells = sum(
        len(model["conditions"])
        for backend in rebuilt["cold_start"]["backends"].values()
        for model in backend["models"].values())
    record(name, "cold-start cells", 24, cells)
    record(name, "C-recal models", 3,
           len(rebuilt["crecal"]["models"]))


# ---------------------------------------------------------------------------
# Check 16 - robustness and reasoning outputs rebuilt from 149 compact cells.
# The A5-3 paired bootstrap runs its full 10,000 resamples, which makes this
# the slow group in an otherwise quick verifier.
# ---------------------------------------------------------------------------
def check16():
    name = "16 robustness + reasoning rebuild"
    rebuilt = []
    rebuilt.append(rebuild_json(
        name, "code/aggregate/aggregate_action7_multiseed.py",
        "results/aggregates/action7_multiseed_summary.json"))
    rebuilt.append(rebuild_json(
        name, "code/aggregate/aggregate_a53_weight_grid.py",
        "results/aggregates/a53_weight_grid_summary.json"))
    rebuilt.append(rebuild_json(
        name, "code/aggregate/aggregate_e6_budget_scan.py",
        "results/aggregates/e6_budget/e6_budget_scan_summary.json"))
    rebuilt.append(rebuild_json(
        name, "code/aggregate/aggregate_round5_exp_c_reasoning.py",
        "results/aggregates/round5_exp_c_reasoning_summary.json"))
    rebuilt.append(rebuild_json(
        name, "code/build/e6_probe_auroc.py",
        "results/aggregates/e6_probe_auroc.json"))
    if any(row is None for row in rebuilt):
        return
    seed, risk, _budget, reasoning, auroc = rebuilt
    record(name, "generation-seed mean cell SD (pp)", 0.294,
           round(seed["panel_summary"]["mean_per_cell_sample_std_pp"], 3))
    record(name, "risk grid max swing <= 2 pp", (1.3, True),
           (risk["verdict"]["grid_max_pairwise_abs_pp"],
            risk["verdict"]["all_models_within_margin"]))
    record(name, "reasoning comparison cells", 10,
           len(reasoning["rows"]))
    record(name, "pooled probe-risk AUROC / L3 AUROC", (0.5744, 0.6132),
           (auroc["pooled"]["auroc_uniform"],
            auroc["pooled"]["per_layer_auroc"]["L3"]))


# ---------------------------------------------------------------------------
# Check 17 - Failure Anatomy.  The legacy ledger remains frozen provenance;
# the preliminary expert pass additionally replays the deterministic draw from
# the released 89,420-key frame and rebuilds two reviews + adjudication.  The
# expert labels and sensitivity analysis behind the reported numbers are
# released under audit/failure_taxonomy/human_review/.
# ---------------------------------------------------------------------------
def check17():
    name = "17 failure-taxonomy rebuild"
    historical = rebuild_json(
        name, "code/audit/recompute_failure_taxonomy.py",
        "audit/failure_taxonomy/recomputed_summary.json",
        output_flag="--out")
    revision = rebuild_json(
        name, "code/audit/recompute_failure_taxonomy_revision.py",
        "audit/failure_taxonomy/revision_sample/adjudication_v2.json",
        output_flag="--out")
    if historical is None or revision is None:
        return
    claims = historical["historical_reported_claims"]
    record(name, "historical frame / ledger retained", (89420, 100),
           (claims["sampling_frame_match_false"], claims["n_classified"]))
    record(name, "historical C1--C5 counts",
           {"C1_api_contract": 53, "C2_param_misuse": 21,
            "C3_workflow_logic": 19, "C4_numerical_extraction": 1,
            "C5_format_env": 6}, historical["category_tally"])
    record(name, "public frame / replayed sample / reviewer agreement",
           (89420, 100, 85.0),
           (revision["sampling"]["frame_match_false"],
            revision["sampling"]["n_selected"],
            revision["review"]["agreement_pct"]))
    record(name, "revision C1--C5 + item-defect counts",
           {"C1_api_contract": 52, "C2_param_misuse": 11,
            "C3_workflow_logic": 23, "C4_numerical_extraction": 1,
            "C5_format_env": 10, "ambiguous_item_defect": 3},
           revision["category_tally"])
    conclusions = revision["conclusions"]
    record(name, "revision mechanism / intervention-shift claims",
           (True, 63, [13, 3], 9, True),
           (conclusions["C1_is_largest_single_category"],
            conclusions["C1_plus_C2_count"],
            conclusions["condA_FDRS_to_condC_FDRS_C1"],
            conclusions["condC_FDRS_C3"],
            conclusions["qualitative_manuscript_claim_preserved"]))
    record(name, "public draw + review checks", True,
           all(revision["checks"].values()))


# ---------------------------------------------------------------------------
# Check 18 - all 610 engineering-validity point/Wilson/gate/tolerance fields,
# plus the independently regenerated corpus-separation diagnostic.
# ---------------------------------------------------------------------------
def check18():
    name = "18 validity + separation rebuild"
    validity = rebuild_json(
        name, "code/audit/recompute_validity_tables.py",
        "audit/validity_tables_recomputed.json", output_flag="--out")
    separation = rebuild_json(
        name, "code/aggregate/aggregate_contamination_analysis.py",
        "results/aggregates/contamination_analysis.json",
        output_flag="--out")
    if validity is not None:
        record(name, "validity input rows / validation", (610, "pass"),
               (validity["n_rows"], validity["validation"]["status"]))
        record(name, "validity model-to-tier mapping (11 open models)", True,
               validity["validation"]
                       ["model_tier_mapping_matches_frozen_design"])
        common = validity["common_support"]
        standard = common["tier_standardised"]
        record(name, "common-support population / T2--T4 counts",
               (2106, {"T1": 0, "T2": 288, "T3": 696, "T4": 1122}),
               (common["population"]["n_pairs_matched_both_conditions"],
                common["population"]["by_tier"]))
        record(name, "standardised unweighted / weighted deltas and CIs",
               ((9.28, [-2.69, 20.61]), (13.28, [0.26, 25.37])),
               ((standard["T2_T3_T4|unweighted"]["delta_S_minus_P_pp"],
                 standard["T2_T3_T4|unweighted"]["boot95_delta_pp"]),
                (standard["T2_T3_T4|weighted"]["delta_S_minus_P_pp"],
                 standard["T2_T3_T4|weighted"]["boot95_delta_pp"])))
    if separation is not None:
        record(name, "near-clones above 0.5", 0,
               separation["per_query_high_sim_count_0.5"])


# ---------------------------------------------------------------------------
# Check 19 - OpenDSS transfer-mechanism evidence.  Rebuild the three frozen
# L0--L3 profiles from per-probe records and the control/device error/token
# rows from the public E2 item-level runs (Supplementary Table S19).
# ---------------------------------------------------------------------------
def check19():
    name = "19 OpenDSS profile-mechanism rebuild"
    rebuilt = rebuild_json(
        name, "code/aggregate/aggregate_e2_opendss_profiles.py",
        "results/aggregates/e2_opendss_profiles_recomputed.json",
        output_flag="--out")
    if rebuilt is None:
        return
    record(name, "all archived profile/table checks", True,
           rebuilt["all_checks_passed"])
    record(name, "profile models / functions per model", (3, [76, 76, 76]),
           (len(rebuilt["profiles"]),
            sorted(row["n_functions"] for row in rebuilt["profiles"].values())))
    record(name, "Qwen/Llama C vs Rsem injected-token means",
           {"Qwen/Qwen2.5-Coder-32B-Instruct": {"C": 393, "Rsem": 3969},
            "meta-llama/Llama-3.1-70B-Instruct": {"C": 526, "Rsem": 3854}},
           {model: row["injected_tokens_mean_rounded"]
            for model, row in rebuilt["control_device_benchmark"].items()})


# ---------------------------------------------------------------------------
# Check 20 - relation between the independently measured L0--L3 probe
# profiles and bare condition-A benchmark accuracy.  The public script rebuilds
# both unrounded inputs before applying average-rank Spearman correlation.
# ---------------------------------------------------------------------------
def check20():
    name = "20 probe--R0 Spearman rebuild"
    rebuilt = rebuild_json(
        name, "code/aggregate/aggregate_probe_r0_spearman.py",
        "results/aggregates/probe_r0_spearman.json")
    if rebuilt is None:
        return
    full = rebuilt["analyses"]["full_panel_10"]
    larger = rebuilt["analyses"]["larger_32B_to_405B_5"]
    record(name, "full-panel n / L0--L3 displayed rho",
           (10, {"L0": "0.78", "L1": "0.93",
                 "L2": "0.78", "L3": "0.78"}),
           (full["n_models"],
            {layer: row["manuscript_display_2dp"]
             for layer, row in full["correlations"].items()}))
    record(name, "32B--405B panel n / L1 rho", (5, "0.90"),
           (larger["n_models"],
            larger["correlations"]["L1"]["manuscript_display_2dp"]))


# ---------------------------------------------------------------------------
# Check 21 - end-to-end role-reweighting ablation from 40 compact runs
# (10 models x C/C_FX/C_FDR/C_FDRS; 80,000 aligned item outcomes).
# ---------------------------------------------------------------------------
def check21():
    name = "21 C-unadapted item-level rebuild"
    frozen_summary = load("results/aggregates/c_unadapted_summary.json")
    frozen_claims = load(
        "results/aggregates/c_unadapted_manuscript_crosscheck.json")
    with tempfile.TemporaryDirectory() as td:
        summary_path = Path(td, "c_unadapted_summary.json")
        claims_path = Path(td, "c_unadapted_claims.json")
        proc = subprocess.run(
            [sys.executable,
             str(ROOT / "code/aggregate/aggregate_c_unadapted.py"),
             "--output", str(summary_path),
             "--claims-output", str(claims_path)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        record(name, "aggregate_c_unadapted.py exits successfully",
               0, proc.returncode)
        if (proc.returncode != 0 or not summary_path.exists()
                or not claims_path.exists()):
            detail = proc.stderr.decode(errors="replace")[-2000:]
            record(name, "both rebuilt outputs exist", True,
                   (summary_path.exists(), claims_path.exists(), detail), False)
            return
        rebuilt_summary = json.loads(summary_path.read_text())
        rebuilt_claims = json.loads(claims_path.read_text())
    record(name, "legacy summary rebuilt field-for-field", True,
           rebuilt_summary == frozen_summary)
    record(name, "manuscript cross-check rebuilt field-for-field", True,
           rebuilt_claims == frozen_claims)
    compact = load(
        "results/raw/c_unadapted/c_unadapted_outcomes_compact.json")
    record(name, "models / compact item outcomes", (10, 80000),
           (rebuilt_summary["panel_n_models"],
            len(compact["runs"]) * compact["dataset"]["n_items"]))
    record(name, "adapted-minus-unadapted / unadapted-minus-A (pp)",
           (3.16, 27.79),
           (round(rebuilt_summary[
               "panel_mean_delta_adapted_minus_unadapted_pp"], 2),
            round(rebuilt_summary[
                "panel_mean_delta_unadapted_minus_a_pp"], 2)))
    record(name, "all item-level manuscript checks", True,
           rebuilt_claims["all_pass"])


# ---------------------------------------------------------------------------
# Check 22 - E4 seed-42--46 item-level reconstruction.  Historical runs kept
# the five aggregate summaries but not their item rankings; the public
# evidence deterministically reconstructs both independent top-20 arms and
# verifies every archived metric, alpha, and role-weight field.
# ---------------------------------------------------------------------------
def check22():
    name = "22 E4 five-seed item-level rebuild"
    frozen = load("results/aggregates/e4_multiseed_recomputed.json")
    evidence = load(
        "results/supplementary_evidence/e4_multiseed_predictions.json")
    with tempfile.TemporaryDirectory() as td:
        rebuilt_path = Path(td, "e4_multiseed_recomputed.json")
        proc = subprocess.run(
            [sys.executable,
             str(ROOT / "code/aggregate/aggregate_e4_multiseed.py"),
             "--output", str(rebuilt_path)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        record(name, "aggregate_e4_multiseed.py exits successfully",
               0, proc.returncode)
        if proc.returncode != 0 or not rebuilt_path.exists():
            detail = proc.stderr.decode(errors="replace")[-2000:]
            record(name, "rebuilt output exists", True,
                   (rebuilt_path.exists(), detail), False)
            return
        rebuilt = json.loads(rebuilt_path.read_text())
    record(name, "recomputed artifact rebuilt field-for-field", True,
           rebuilt == frozen)
    record(name, "seed / query / arm / rank evidence dimensions",
           (5, 194, 2, 20),
           (len(evidence["seeds"]), len(evidence["item_axis"]),
            len(evidence["top20_ids_by_seed"]["42"]),
            len(evidence["top20_ids_by_seed"]["42"]["adapted"][0])))
    record(name, "archived-field validation checks / mismatches",
           (5948, 0),
           (rebuilt["validation"]["n_checks"],
            rebuilt["validation"]["n_mismatches"]))
    observed = []
    for key in ("layer1_holdout_n80", "layer2a_n84",
                "layer2a_ext_n10", "layer2b_n20"):
        cell = rebuilt["manuscript_summary"][key][
            "conditional_recall_at_10_percent"]
        observed.append((cell["mean"], cell["min"], cell["max"]))
    record(name, "five-seed conditional recall@10 summaries",
           [(71.74, 70.92, 72.17), (62.39, 57.07, 65.07),
            (61.51, 58.14, 64.72), (65.45, 64.52, 67.26)],
           observed)


# ---------------------------------------------------------------------------
# Check 23 - D-6 import-scan audit.  The pre-fix compact retains all 1,080
# generated programs; the post-fix side reads the 24 public full raw runs.
# Both historical scan reports must rebuild byte-exactly.
# ---------------------------------------------------------------------------
def check23():
    name = "23 E2 D-6 import scans"
    frozen = load("results/aggregates/e2_import_scans/crosscheck.json")
    pre_evidence = load("audit/e2_import_scan/pre_fix_generations.json")
    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td, "e2_import_scans")
        proc = subprocess.run(
            [sys.executable,
             str(ROOT / "code/audit/recompute_e2_import_scans.py"),
             "--output-dir", str(output_dir)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        record(name, "recompute_e2_import_scans.py exits successfully",
               0, proc.returncode)
        rebuilt_path = output_dir / "crosscheck.json"
        if proc.returncode != 0 or not rebuilt_path.exists():
            detail = proc.stderr.decode(errors="replace")[-2000:]
            record(name, "rebuilt crosscheck exists", True,
                   (rebuilt_path.exists(), detail), False)
            return
        rebuilt = json.loads(rebuilt_path.read_text())
        pre_bytes_exact = (
            (output_dir / "import_scan_prefix_B2.json").read_bytes()
            == (ROOT / "audit/e2_import_scan/frozen/"
                "import_scan_prefix_B2.json").read_bytes())
        post_bytes_exact = (
            (output_dir / "import_scan_v2_postfix.json").read_bytes()
            == (ROOT / "audit/e2_import_scan/frozen/"
                "import_scan_v2_postfix.json").read_bytes())
    record(name, "crosscheck rebuilt field-for-field", True,
           rebuilt == frozen)
    record(name, "pre/post historical scans byte-exact",
           (True, True), (pre_bytes_exact, post_bytes_exact))
    records = pre_evidence["records"]
    record(name, "pre-fix complete generated-code records",
           (1080, 1080),
           (len(records), sum("generated_code" in row for row in records)))
    pre = rebuilt["pre_fix"]["headline"]
    record(name, "pre-fix A: generations / pp / target imports",
           (360, 360, 0),
           (pre["condA_generations"], pre["condA_pandapower_imports"],
            pre["condA_target_lib_imports"]))
    frozen_scope = rebuilt["post_fix"]["frozen_scope_13_files"]
    record(name, "post-fix frozen scope: pp / OpenDSS / PyPSA imports",
           (0, 629, 540),
           (frozen_scope["total"]["pandapower_imports"],
            frozen_scope["opendss"]["target_lib_imports"],
            frozen_scope["pypsa"]["target_lib_imports"]))
    full_scope = rebuilt["post_fix"]["full_corpus_24_files"]["total"]
    record(name, "post-fix full scope: generations / pp / target imports",
           (2160, 0, 2158),
           (full_scope["n_generations"], full_scope["pandapower_imports"],
            full_scope["target_lib_imports"]))


# ---------------------------------------------------------------------------
# Check 24 - complete serving-table projection from every public Protocol-A/B
# measurement record, including the two single-repetition anchors and the
# separately audited 480B throughput retests.
# ---------------------------------------------------------------------------
def check24():
    name = "24 complete serving-table rebuild"
    frozen = load("results/aggregates/e3_serving_tables_recomputed.json")
    with tempfile.TemporaryDirectory() as td:
        rebuilt_path = Path(td, "e3_serving_tables_recomputed.json")
        proc = subprocess.run(
            [sys.executable,
             str(ROOT / "code/aggregate/recompute_serving_tables.py"),
             "--output", str(rebuilt_path)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        record(name, "recompute_serving_tables.py exits successfully",
               0, proc.returncode)
        if proc.returncode != 0 or not rebuilt_path.exists():
            detail = proc.stderr.decode(errors="replace")[-2000:]
            record(name, "rebuilt output exists", True,
                   (rebuilt_path.exists(), detail), False)
            return
        rebuilt = json.loads(rebuilt_path.read_text())
    record(name, "serving artifact rebuilt field-for-field", True,
           rebuilt == frozen)
    coverage = rebuilt["coverage"]
    a = coverage["protocol_a"]
    b = coverage["protocol_b"]
    record(name, "Protocol-A cells / reps / node / GPU records",
           (20, 56, 62, 128),
           (a["operating_point_cells"], a["repetition_directories"],
            a["node_energy_records"], a["gpu_device_records"]))
    record(name, "Protocol-B cells / requests / node / GPU records",
           (15, 200, 15, 27),
           (b["cells"], b["completed_requests_per_cell"],
            b["node_energy_records"], b["gpu_device_records"]))
    record(name, "raw / frozen / manuscript checked leaves",
           (2483, 4238, 585),
           (rebuilt["raw_record_consistency"]["checked_leaves"],
            rebuilt["frozen_table_crosscheck"]["checked_leaves"],
            rebuilt["manuscript_projection_crosscheck"]["checked_leaves"]))
    record(name, "all serving projections pass with zero differences",
           ("PASS", 0),
           (rebuilt["status"], len(rebuilt["all_differences"])))


# ---------------------------------------------------------------------------
# Check 25 - event-level reactive routing, item-level proactive prompt
# accounting, and the idempotent release normalization of diagnostic paths.
# ---------------------------------------------------------------------------
def check25():
    name = "25 routing events + prompt accounting"
    router_frozen = load(
        "results/aggregates/supplementary_reactive_router_shares.json")
    router_evidence = load(
        "results/supplementary_evidence/reactive_router_counts.json")
    prompt_frozen = load(
        "results/aggregates/proactive_prompt_components.json")
    primary_path = ROOT / (
        "results/raw/main_experiment/primary_outcomes_compact.json")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        router_path = td_path / "router.json"
        prompt_path = td_path / "prompt.json"
        normalized_path = td_path / "primary.json"
        normalization_report = td_path / "normalization.json"
        normalized_path.write_bytes(primary_path.read_bytes())
        commands = [
            ([sys.executable,
              str(ROOT / "code/aggregate/aggregate_reactive_router_shares.py"),
              "--output", str(router_path)], "router"),
            ([sys.executable,
              str(ROOT / "code/aggregate/aggregate_proactive_prompt_components.py"),
              "--output", str(prompt_path)], "prompt"),
            ([sys.executable,
              str(ROOT / "code/audit/normalize_primary_diagnostic_paths.py"),
              "--archive", str(normalized_path),
              "--report", str(normalization_report)], "normalization"),
        ]
        for command, label in commands:
            proc = subprocess.run(
                command, cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            record(name, f"{label} verifier exits successfully",
                   0, proc.returncode)
            if proc.returncode != 0:
                detail = proc.stderr.decode(errors="replace")[-2000:]
                record(name, f"{label} verifier stderr", "", detail, False)
                return
        router_rebuilt = json.loads(router_path.read_text())
        prompt_rebuilt = json.loads(prompt_path.read_text())
        norm = json.loads(normalization_report.read_text())
        normalization_idempotent = (
            normalized_path.read_bytes() == primary_path.read_bytes())
    record(name, "router shares rebuilt field-for-field", True,
           router_rebuilt == router_frozen)
    event_counts = [
        len(cell["events"])
        for model in router_evidence["models"].values()
        for cell in model["conditions"].values()
    ]
    record(name, "router cells / item-round events",
           (20, 68107), (len(event_counts), sum(event_counts)))
    record(name, "prompt components rebuilt field-for-field", True,
           prompt_rebuilt == prompt_frozen)
    prompt = prompt_rebuilt["panel_pooled_exact_from_item_vectors"]
    record(name, "prompt item count / accounting identity",
           (20000, True),
           (prompt["totals"]["n_items"],
            prompt["accounting_identity_holds"]))
    averages = prompt["averages"]
    record(name, "base / docs / framing / C prompt means",
           (182.17345, 767.79295, 500.9394, 1450.9058),
           (averages["base"], averages["documentation"],
            averages["framing_residual"], averages["total"]))
    invariance = norm["invariance"]
    recorded = norm["archive_recorded_totals"]
    record(name, "path-normalization entries / occurrences / private paths",
           (215, 1662, 0),
           (recorded["dictionary_entries_changed"],
            recorded["item_occurrences_changed"],
            invariance["remaining_absolute_site_packages_prefixes"]))
    record(name, "normalization idempotent / non-text evidence exact",
           (True, True),
           (normalization_idempotent,
            invariance["all_nontext_evidence_exact"]))


# ---------------------------------------------------------------------------
# Check 26 - strongest recoverable item-level S4 evidence.  Three TF-IDF
# estimators, pinned-snapshot Zero-shot SBERT, and pinned-snapshot Zero-shot
# Cross-Encoder have exact unfiltered top-20 reconstructions.  Hybrid
# Cross-Encoder is kept separately as a conclusion-stable deterministic rerun
# on the pinned stack; results/supplementary_evidence/README.md gives its
# scope.
# ---------------------------------------------------------------------------
def check26():
    name = "26 S4 item-evidence rebuild"
    rebuilt = rebuild_json(
        name, "code/aggregate/aggregate_s4_item_evidence.py",
        "results/aggregates/s4_item_evidence_recomputed.json")
    if rebuilt is None:
        return
    evidence = load("results/supplementary_evidence/s4_item_evidence.json")
    record(name, "schema / pinned sklearn / reconstructed methods",
           (4, "1.6.0", 5),
           (evidence["schema_version"],
            evidence["environment"]["scikit_learn_used_for_recovery"],
            len(evidence["reconstructions"])))
    record(name, "historical top-10 / unavailable exact cross-encoder arms",
           (6, 1),
           (len(evidence["historical_benchmark_top10"]),
            len(rebuilt["unavailable_exact_item_coverage"])))
    exact = rebuilt["exact_top20_reconstruction_metrics"]
    record(name, "exact methods / split sizes",
           (5, [23, 633, 2000]),
           (len(exact), sorted(
               row["n_eval"]
               for row in exact["Hybrid TF-IDF"].values())))
    sbert_checks = rebuilt["integrity_checks"]["source_export_frozen_slices"]
    record(name, "SBERT snapshot revision / seven frozen slices",
           ("c9745ed1d9f207416be6d2e6f8de32d1f16199bf", 7, True),
           (evidence["provenance"]["sbert_model_snapshot"]["revision"],
            len(sbert_checks["Zero-shot SBERT"]),
            all(sbert_checks["Zero-shot SBERT"].values())))
    ce_provenance = evidence["provenance"]["cross_encoder_recovery"]
    record(name, "Zero-shot CE snapshot / seven exact frozen slices",
           ("7b0235231ca2674cb8ca8f022859a6eba2b1c968", 7, True),
           (ce_provenance["model_snapshot"]["revision"],
            len(sbert_checks["Zero-shot Cross-Encoder"]),
            all(sbert_checks["Zero-shot Cross-Encoder"].values())))
    retention = ce_provenance["hybrid_historical_retention"]
    record(name, "Hybrid CE historical model/hash-state retention",
           (False, None, False,
            "f3f08b0e394ba5ae3cb8ffd6a294225df144ead1c40462a2400b47e827ea78aa"),
           (retention["save_model_argument_present"], retention["saved_model"],
            retention["pythonhashseed_logged"],
            ce_provenance["historical_job_log"]["sha256"]))
    revision = evidence["revision_reruns"]["Hybrid Cross-Encoder"]
    revision_checks = rebuilt["integrity_checks"][
        "revision_rerun_frozen_slice_matches"
    ]["Hybrid Cross-Encoder"]
    record(name, "Hybrid CE revision label / frozen slice matches",
           ("conclusion-stable quantitative drift; non-exact historical reconstruction",
            0),
           (revision["comparison_label"], sum(revision_checks.values())))
    first_stage = rebuilt["revision_rerun_first_stage_diagnostics"][
        "Hybrid Cross-Encoder"
    ]
    record(name, "Hybrid CE frozen/revision alpha and role-weight diagnostic",
           (0.3, 0.4, True, False),
           (first_stage["frozen_selected_alpha"],
            first_stage["revision_selected_alpha"],
            first_stage["role_weights_match"],
            first_stage["alpha_curve_and_selected_alpha_match"]))
    deltas = rebuilt["revision_rerun_recall_deltas_pp_vs_frozen"][
        "Hybrid Cross-Encoder"
    ]
    record(name, "Hybrid CE revision R@10 drift: Aug-test/orig/Bench-ref (pp)",
           (2.83, 4.5, 1.17),
           (deltas["Aug-test"]["10"], deltas["Aug-orig"]["10"],
            deltas["Bench-ref"]["10"]))
    tradeoff = rebuilt["s4b_before_after_by_split"]
    record(name, "S4(b) Aug-test / Aug-orig / Bench-ref deltas (pp)",
           (-5.14, -16.81, 20.69),
           (tradeoff["Aug-test"]["recall_at_10_delta_pp"],
            tradeoff["Aug-orig"]["recall_at_10_delta_pp"],
            tradeoff["Bench-ref"]["recall_at_10_delta_pp"]))
    record(name, "all archived metric and item checks pass", True,
           rebuilt["all_integrity_checks_pass"])


# ---------------------------------------------------------------------------
# Check 27 - deterministic rerun of the C3 cross-style holdout on the pinned
# stack.  The historical runs kept aggregate values but no per-query rankings,
# so this check reaggregates the published rerun top-20 rankings and the
# paired endpoint bootstrap.
# ---------------------------------------------------------------------------
def check27():
    name = "27 C3 cross-style item rerun"
    rebuilt = rebuild_json(
        name, "code/aggregate/aggregate_c3_deterministic_rerun.py",
        "results/aggregates/c3_deterministic_rerun_recomputed.json")
    if rebuilt is None:
        return
    evidence = load(
        "results/supplementary_evidence/c3_deterministic_rerun_items.json")
    record(name, "runs / D12 items / D34 items / pinned sklearn",
           (6, 1200, 800, "1.6.0"),
           (len(evidence["runs"]),
            len(evidence["evaluation_sets"]["D12"]["items"]),
            len(evidence["evaluation_sets"]["D34"]["items"]),
            evidence["environment"]["scikit_learn"]))
    endpoints = rebuilt["cross_source_endpoint_comparison"]
    d12 = endpoints["D12_eval_cross_D34_vs_in_D12"]
    d34 = endpoints["D34_eval_cross_D12_vs_in_D34"]
    record(name, "cross-minus-in endpoint gaps (pp)", (0.07, -0.29),
           (d12["cross_minus_in_source_pp"],
            d34["cross_minus_in_source_pp"]))
    record(name, "paired endpoint bootstrap CIs",
           ([-0.12, 0.27], [-0.62, 0.0]),
           (d12["paired_bootstrap"]["ci95_pp"],
            d34["paired_bootstrap"]["ci95_pp"]))
    record(name, "paired queries / seed / replicates",
           (1200, 800, 22, 10000),
           (d12["n_paired_queries"], d34["n_paired_queries"],
            d12["paired_bootstrap"]["seed"],
            d12["paired_bootstrap"]["replicates"]))
    record(name, "all item, provenance and determinism checks pass", True,
           rebuilt["all_integrity_checks_pass"])


# ---------------------------------------------------------------------------
# Check 28 - all 29,120 raw L0--L3 probe response/evaluator records and the
# historical profile aggregation for 10 open-weight and four API models.
# Groups 26--27 cover the S4 and C3 reconstructions.
# ---------------------------------------------------------------------------
def check28():
    name = "28 raw probe transcripts"
    rebuilt = rebuild_json(
        name, "code/aggregate/recompute_probe_transcripts.py",
        "results/aggregates/probe_transcripts_recomputed.json")
    if rebuilt is None:
        return
    coverage = rebuilt["coverage"]
    record(name, "models / files / raw responses",
           (14, 44, 29120),
           (coverage["models"], coverage["files"],
            coverage["raw_responses"]))
    record(name, "open-weight / API models / functions each",
           (10, 4, 275),
           (coverage["open_weight_models"], coverage["api_models"],
            coverage["functions_per_model"]))
    record(name, "profile scalar leaves / mismatches",
           (95004, 0),
           (coverage["profile_scalar_leaves_compared"],
            coverage["profile_mismatches"]))
    expected_layers = {"L0": 550, "L1": 275, "L2": 980, "L3": 275}
    record(name, "all 14 profiles match field-for-field with complete layers",
           True, all(
               row["profile_matches_public_field_for_field"]
               and row["per_layer_counts"] == expected_layers
               and row["records"] == row["raw_responses"] == 2080
               for row in rebuilt["models"].values()))
    manifest = load("results/raw/probe_transcripts/manifest.json")
    policy = manifest["release_policy"]
    record(name, "documented release mode / source blobs / credential shapes",
           ("byte-exact except documented diagnostic path normalization", True, 0),
           (policy["copy_mode"],
            all(row["source_git_blob"] for row in manifest["files"]),
            policy["credential_shape_matches"]))
    normalization = rebuilt["diagnostic_path_normalization"]
    totals = normalization["source_to_release"]
    record(name, "diagnostic path changes / released private paths",
           (14, 2938, 4900, 2435, 0),
           (totals["files_changed"], totals["changed_entries"],
            totals["changed_occurrences"], totals["changed_records"],
            normalization["release_private_path_occurrences"]))
    record(name, "raw responses / scores / correct fields invariant",
           (29120, 29120, 21378),
           (totals["raw_response_records_exact"],
            totals["score_records_exact"],
            totals["correct_fields_compared_exact"]))
    invariance = normalization["invariance"]
    record(name, "non-diagnostic evidence exact / release idempotent",
           (528168, True, True, 44),
           (totals["nondiagnostic_scalar_leaves_compared"],
            invariance["all_changes_confined_to_allowed_diagnostic_fields"],
            invariance["release_normalization_idempotent"],
            normalization["files_checked_idempotent"]))


# ---------------------------------------------------------------------------
# Check 29 - contemporaneous C3/E4 execution logs.  These logs independently
# corroborate the historical aggregate runs and mark where the record stops:
# the old writers serialised aggregate metrics, not per-query rankings.
# ---------------------------------------------------------------------------
def check29():
    name = "29 historical C3/E4 run logs"
    rebuilt = rebuild_json(
        name, "code/aggregate/aggregate_historical_demand_run_logs.py",
        "results/aggregates/historical_demand_run_logs_recomputed.json")
    if rebuilt is None:
        return
    manifest = load(
        "results/supplementary_evidence/historical_demand_runs/manifest.json")
    record(name, "released files / normalized source-prefix occurrences",
           (8, 48),
           (len(manifest["files"]),
            sum(row["private_prefix_replacements"]
                for row in manifest["files"])))
    c3 = rebuilt["c3"]
    record(name, "six historical C3 recall@10 endpoints",
           [0.5618, 0.4798, 0.6842, 0.6594, 0.6646, 0.6846],
           [c3[run]["benchmark_recall_at_10"] for run in (
               "R1_baseline_eval_D12", "R2_baseline_eval_D34",
               "R3_reweight_D12_eval_D12", "R4_reweight_D12_eval_D34",
               "R5_reweight_D34_eval_D34", "R6_reweight_D34_eval_D12")])
    record(name, "C3 log endpoints match frozen aggregates / exports null",
           (True, True),
           (all(row["matches_frozen_aggregate"] for row in c3.values()),
            all(row["prediction_export_null"] and row["saved_model_null"]
                for row in c3.values())))
    e4 = rebuilt["e4"]
    record(name, "E4 seed sequence / aggregate endpoint checks",
           ([42, 43, 44, 45, 46], True),
           (e4["seeds"], all(
               all(checks.values()) for checks in e4["checks"].values())))
    record(name, "determinism regression / released private paths",
           (True, True, 0),
           (rebuilt["determinism_regression"]["pythonhashseed_matrix_present"],
            rebuilt["determinism_regression"]["subprocess_regression_present"],
            len(rebuilt["private_path_hits"])))


GROUP_DESCRIPTIONS = {
    1: "proactive+reactive lift over baseline A (+32 to +56 pp)",
    2: "proactive C prompt-token cost vs full-context X (41%)",
    3: "170-pair McNemar engineering-validity suite",
    4: "E2 backend-transfer 24-cell matrix + OpenDSS C-recal pairing",
    5: "risk-weight grid max|delta| per model",
    6: "E4 external-query conditional adaptation + seed sensitivity",
    7: "480B serving anchor composite convention",
    8: "false-acceptance-rate suite",
    9: "Qwen3-Coder-Next serving row",
    10: "Delta(C-A) hero paired-bootstrap CI",
    11: "matcher-FN mechanism diagnostic + de-duplicated census",
    12: "P02 C+FDRS mid-tier parity paired bootstrap + TOST",
    13: "primary Tables 3--6 from 270 compact item-level runs",
    14: "supplementary probe, demand, E4 and router evidence",
    15: "full E2 transfer reconstruction (24 cells + C-recal)",
    16: "robustness, risk-grid, E6 and reasoning reconstruction",
    17: "Failure Anatomy public 89,420-key draw + 100-case adjudication",
    18: "610-case validity tables + corpus-separation diagnostic",
    19: "OpenDSS L0--L3 profiles + control/device mechanism",
    20: "probe-profile vs bare-R0 Spearman correlations",
    21: "C-unadapted 40-run / 80,000-outcome reconstruction",
    22: "E4 seed-42--46 item-level ranking reconstruction",
    23: "E2 D-6 pre/post import scans from 3,240 generations",
    24: "complete Protocol-A/B serving-table reconstruction",
    25: "68,107 routing events + 20,000-item prompt accounting",
    26: "S4 TF-IDF/SBERT item rankings + before/after demand tradeoff",
    27: "C3 deterministic cross-style item rerun + paired endpoints",
    28: "29,120 raw probe responses -> 14 complete L0--L3 profiles",
    29: "historical C3/E4 execution logs + aggregate cross-checks",
}


GROUP_SECTIONS = (
    ("headline", (1, 2, 4, 6, 7, 9, 10, 12, 13)),
    ("robustness", (5, 14, 16, 19, 20, 21, 22, 24, 25, 26, 27)),
    ("audit", (3, 8, 11, 17, 18, 23)),
    ("release-provenance", (15, 28, 29)),
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Recompute the paper's headline number groups from the "
                    "frozen artifacts (CPU-only, stdlib, no network).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="number groups:\n" + "\n".join(
            f"  [{section}]\n" + "\n".join(
                f"  {n:>2}  {GROUP_DESCRIPTIONS[n]}" for n in numbers)
            for section, numbers in GROUP_SECTIONS),
    )
    parser.add_argument(
        "--group", type=int, choices=sorted(GROUP_DESCRIPTIONS), default=None,
        metavar="N", help="run only number group N (default: run all groups)")
    args = parser.parse_args(argv)

    checks_by_group = {
        1: check1, 2: check2, 3: check3, 4: check4, 5: check5,
        6: check6, 7: check7, 8: check8, 9: check9, 10: check10,
        11: check11, 12: check12, 13: check13, 14: check14, 15: check15,
        16: check16, 17: check17, 18: check18, 19: check19, 20: check20,
        21: check21, 22: check22, 23: check23, 24: check24, 25: check25,
        26: check26, 27: check27,
        28: check28, 29: check29,
    }
    if set(checks_by_group) != set(GROUP_DESCRIPTIONS):
        raise RuntimeError("group descriptions and verifier functions differ")
    checks = ([checks_by_group[args.group]] if args.group is not None
              else [checks_by_group[n] for n in sorted(checks_by_group)])
    for c in checks:
        c()
    width = max(len(f"{chk}: {item}") for chk, item, *_ in RESULTS)
    n_fail = 0
    current = None
    for chk, item, expected, got, ok in RESULTS:
        if chk != current:
            print(f"\n== {chk} ==")
            current = chk
        status = "PASS" if ok else "FAIL"
        line = f"  [{status}] {item}"
        if not ok:
            n_fail += 1
            line += f"\n         expected: {expected}\n         got:      {got}"
        print(line)
    total = len(RESULTS)
    print(f"\n{'=' * 60}\n{total - n_fail}/{total} assertions passed"
          + ("" if n_fail == 0 else f", {n_fail} FAILED"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
