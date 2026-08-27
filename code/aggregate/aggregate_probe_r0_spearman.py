#!/usr/bin/env python3
"""Rebuild the manuscript's model-level probe--R0 rank correlations.

The inputs are the two public evidence units:

* the complete 275-function L0--L3 profile for each model; and
* the compact item outcomes for condition A on the 2,000-item benchmark.

Spearman's rho is Pearson correlation applied to ascending average ranks.
Average ranks state the tie convention and keep the implementation valid if a
later panel contains ties; the frozen panels used here contain none.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES = ROOT / "results/supplementary_evidence/probe_profiles.json"
DEFAULT_OUTCOMES = ROOT / "results/raw/main_experiment/primary_outcomes_compact.json"
DEFAULT_OUTPUT = ROOT / "results/aggregates/probe_r0_spearman.json"

LAYERS = ("L0", "L1", "L2", "L3")

# The ten-model open-weight panel defined in the manuscript's experimental
# setup.  Qwen2.5-Coder-0.5B is absent from that panel and has no released
# profile; the 1.5B model is the manuscript's low-end reference.
PANEL10 = (
    ("Qwen2.5-1.5B", "Qwen_Qwen2.5-Coder-1.5B-Instruct"),
    ("Qwen2.5-7B", "Qwen_Qwen2.5-Coder-7B-Instruct"),
    ("Llama-8B", "meta-llama_Llama-3.1-8B-Instruct"),
    ("Qwen2.5-14B", "Qwen_Qwen2.5-Coder-14B-Instruct"),
    ("Qwen2.5-32B", "Qwen_Qwen2.5-Coder-32B-Instruct"),
    ("Qwen3-Next", "Qwen_Qwen3-Coder-Next"),
    ("Llama-70B", "meta-llama_Llama-3.1-70B-Instruct"),
    ("GPT-OSS-120B", "openai_gpt-oss-120b"),
    ("Llama-405B", "meta-llama_Llama-3.1-405B-Instruct"),
    ("Qwen3-480B", "Qwen_Qwen3-Coder-480B-A35B-Instruct"),
)

# The manuscript's five-model, larger-scale sub-panel.  Its stated 32B--405B
# range includes the mid-tier-active-parameter Qwen3-Next point and excludes
# the 480B endpoint.
LARGER5 = (
    "Qwen2.5-32B",
    "Qwen3-Next",
    "Llama-70B",
    "GPT-OSS-120B",
    "Llama-405B",
)

MANUSCRIPT_EXPECTED = {
    "full_panel_10": {"L0": "0.78", "L1": "0.93", "L2": "0.78", "L3": "0.78"},
    "larger_32B_to_405B_5": {"L1": "0.90"},
}


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        stream.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_ranks(values: list[float]) -> list[float]:
    """Return ascending, one-based average ranks with exact-value ties."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = average
        start = stop
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("correlation requires equal vectors with at least two entries")
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_ss = sum((a - x_mean) ** 2 for a in x)
    y_ss = sum((b - y_mean) ** 2 for b in y)
    if x_ss == 0.0 or y_ss == 0.0:
        raise ValueError("correlation is undefined for a constant vector")
    return numerator / math.sqrt(x_ss * y_ss)


def spearman(x: list[float], y: list[float]) -> tuple[float, list[float], list[float]]:
    x_ranks = average_ranks(x)
    y_ranks = average_ranks(y)
    return pearson(x_ranks, y_ranks), x_ranks, y_ranks


def display(value: float) -> str:
    """The manuscript reports rho to two digits after the decimal point."""
    return f"{value:.2f}"


def load_rows(profiles_path: Path, outcomes_path: Path) -> list[dict]:
    profiles = read_json(profiles_path)
    outcomes = read_json(outcomes_path)
    rows = []
    for label, source_model in PANEL10:
        try:
            profile_entry = profiles["models"][label]
        except KeyError as exc:
            raise ValueError(f"profile evidence lacks {label}") from exc
        if profile_entry.get("source_model") != source_model:
            raise ValueError(f"profile model mismatch for {label}")
        profile = profile_entry.get("profile")
        if not isinstance(profile, dict) or len(profile) != 275:
            raise ValueError(f"{label}: expected a complete 275-function profile")

        layer_means = {}
        for layer in LAYERS:
            try:
                values = [float(function_data[layer]["score"])
                          for function_data in profile.values()]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{label}: malformed {layer} score evidence") from exc
            if len(values) != 275 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{label}: incomplete/non-finite {layer} scores")
            layer_means[layer] = sum(values) / len(values)

        key = f"comparison|{source_model}|A"
        try:
            run = outcomes["runs"][key]
        except KeyError as exc:
            raise ValueError(f"compact outcomes lack {key}") from exc
        if (run.get("group"), run.get("model"), run.get("condition")) != (
            "comparison", source_model, "A"
        ):
            raise ValueError(f"run identity mismatch for {key}")
        match_bits = run.get("match_bits")
        if not isinstance(match_bits, str) or len(match_bits) != 2000 or set(match_bits) - {"0", "1"}:
            raise ValueError(f"{key}: expected 2,000 binary item outcomes")
        n_matched = match_bits.count("1")
        snapshots = run.get("round_snapshots") or []
        if len(snapshots) != 1 or int(snapshots[0].get("n_matched", -1)) != n_matched:
            raise ValueError(f"{key}: item outcomes disagree with the R0 snapshot")
        rows.append({
            "model": label,
            "source_model": source_model,
            "probe_mean": layer_means,
            "r0_n": len(match_bits),
            "r0_n_matched": n_matched,
            "r0_accuracy": n_matched / len(match_bits),
        })
    return rows


def analyze(rows: list[dict]) -> dict:
    r0 = [float(row["r0_accuracy"]) for row in rows]
    r0_ranks = average_ranks(r0)
    correlations = {}
    probe_ranks = {}
    for layer in LAYERS:
        values = [float(row["probe_mean"][layer]) for row in rows]
        rho, layer_ranks, check_r0_ranks = spearman(values, r0)
        if check_r0_ranks != r0_ranks:
            raise AssertionError("internal R0-rank inconsistency")
        correlations[layer] = {"rho": rho, "manuscript_display_2dp": display(rho)}
        probe_ranks[layer] = layer_ranks
    ranked_rows = []
    for index, row in enumerate(rows):
        ranked_rows.append({
            **row,
            "ranks": {
                **{layer: probe_ranks[layer][index] for layer in LAYERS},
                "R0": r0_ranks[index],
            },
        })
    return {
        "n_models": len(rows),
        "models": [row["model"] for row in rows],
        "correlations": correlations,
        "model_inputs_and_ranks": ranked_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    profiles_path = args.profiles.resolve()
    outcomes_path = args.outcomes.resolve()
    rows = load_rows(profiles_path, outcomes_path)
    by_model = {row["model"]: row for row in rows}
    larger_rows = [by_model[model] for model in LARGER5]

    analyses = {
        "full_panel_10": analyze(rows),
        "larger_32B_to_405B_5": analyze(larger_rows),
    }
    comparisons = {}
    all_pass = True
    for analysis_name, expected in MANUSCRIPT_EXPECTED.items():
        observed = {
            layer: analyses[analysis_name]["correlations"][layer]["manuscript_display_2dp"]
            for layer in expected
        }
        passed = observed == expected
        comparisons[analysis_name] = {
            "expected_display": expected,
            "observed_display": observed,
            "pass": passed,
        }
        all_pass = all_pass and passed
    if not all_pass:
        raise ValueError(f"recomputed values disagree with manuscript claims: {comparisons}")

    output = {
        "schema_version": 1,
        "source_artifacts": {
            "probe_profiles": {
                "path": profiles_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(profiles_path),
            },
            "primary_compact_item_outcomes": {
                "path": outcomes_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(outcomes_path),
            },
        },
        "definitions": {
            "probe_score": "unweighted arithmetic mean of 275 per-function layer scores",
            "R0": "condition-A exact-match count divided by all 2,000 frozen items",
            "spearman": "Pearson correlation of ascending one-based average ranks; exact-value ties receive their mean rank",
            "rounding": "rho is computed from unrounded inputs and formatted to two decimal places for the manuscript",
            "panel_note": "full_panel_10 is the manuscript open-weight panel; larger_32B_to_405B_5 excludes the 480B endpoint and includes Qwen3-Next",
        },
        "analyses": analyses,
        "manuscript_crosscheck": {"all_pass": all_pass, "claims": comparisons},
    }
    write_json(args.output.resolve(), output)
    print(f"rebuilt probe--R0 Spearman correlations -> {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
