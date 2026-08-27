#!/usr/bin/env python3
"""Render the accuracy--energy serving Pareto scatter from the frozen T11 rows.

The only input is ``results/aggregates/e3_serving_tables_recomputed.json``;
the eleven plotted rows are ``manuscript_tables.main_T11_protocol_a_condensed``
exactly as printed in the manuscript's Table 11 (Protocol A, condition C,
frozen 300-item serving subset).  Before drawing, every row is checked field
by field against the EXPECTED constants below, and the plotted Pareto
front is recomputed from the validated rows and checked against the frozen
front ordering.  Any mismatch raises instead of producing a figure.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "results/aggregates/e3_serving_tables_recomputed.json"
DEFAULT_OUTPUT = ROOT / "assets/fig_serving_pareto.pdf"

# Frozen Table 11 rows (Protocol A condensed): tier -> (model_dir, gpus, tp,
# accuracy_pct, energy_kj_per_successful_task, throughput_tokens_per_s,
# peak_vram_cluster_gb).  Values are the exact frozen aggregate values.
EXPECTED_T11 = {
    "0.5B": ("Qwen_Qwen2.5-Coder-0.5B-Instruct", 1, 1, 1.0,
             1.8277206666666668, 19349.980252999296, 93.72336128),
    "1.5B": ("Qwen_Qwen2.5-Coder-1.5B-Instruct", 1, 1, 9.333333333333334,
             0.13106467857142856, 41642.319575762514, 94.540857344),
    "7B": ("Qwen_Qwen2.5-Coder-7B-Instruct", 1, 1, 20.333333333333332,
           0.1982084262295082, 19311.632602267036, 95.10289408),
    "8B": ("meta-llama_Llama-3.1-8B-Instruct", 1, 1, 22.0,
           0.2332650909090909, 15586.39862032584, 95.26214656),
    "14B": ("Qwen_Qwen2.5-Coder-14B-Instruct", 2, 2, 34.333333333333336,
            0.3052993786407767, 14079.203490186039, 190.840373248),
    "32B": ("Qwen_Qwen2.5-Coder-32B-Instruct", 2, 2, 43.0,
            0.7124169457364341, 5511.754423017072, 194.057601024),
    "70B": ("meta-llama_Llama-3.1-70B-Instruct", 2, 2, 39.0,
            2.084955564102564, 2076.187127764895, 196.618878976),
    "80B": ("Qwen_Qwen3-Coder-Next", 2, 2, 46.666666666666664,
            0.5103176928571428, 5568.273781593592, 203.636146176),
    "120B": ("openai_gpt-oss-120b", 4, 4, 46.666666666666664,
             1.2034136928571428, 7137.990689647345, 385.719468032),
    "405B": ("meta-llama_Llama-3.1-405B-Instruct", 16, 16, 55.0,
             20.48904954545454, 678.8390787803819, 1612.98153472),
    "480B": ("Qwen_Qwen3-Coder-480B-A35B-Instruct", 16, 16, 55.0,
             12.955257836363637, 1132.28239255461, 1605.80960256),
}
# The frozen accuracy--energy Pareto front (dominating tiers, energy ascending).
EXPECTED_FRONT = ("1.5B", "7B", "8B", "14B", "80B", "480B")

LABEL = {
    "0.5B": "0.5B", "1.5B": "1.5B", "7B": "7B", "8B": "8B (Llama)",
    "14B": "14B", "32B": "32B", "70B": "70B (Llama)", "80B": "Qwen3-Next",
    "120B": "GPT-OSS-120B", "405B": "Llama-405B", "480B": "Qwen3-480B",
}


def close(observed: float, expected: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance)


def validate(data: dict) -> list[dict]:
    """Raise unless the frozen T11 rows equal the EXPECTED constants."""
    if data.get("schema_version") != 1:
        raise ValueError("unsupported serving-table schema")
    if data.get("status") != "PASS":
        raise ValueError(f"frozen serving aggregate is not ok: {data.get('status')}")
    rows = data["manuscript_tables"]["main_T11_protocol_a_condensed"]
    if [row["tier"] for row in rows] != list(EXPECTED_T11):
        raise ValueError("T11 tier panel changed")
    for row in rows:
        model_dir, gpus, tp, accuracy, energy, throughput, vram = EXPECTED_T11[row["tier"]]
        if row["model_dir"] != model_dir or int(row["gpus"]) != gpus or int(row["tp"]) != tp:
            raise ValueError(f"T11 configuration changed: {row['tier']}")
        checks = (
            (row["accuracy_pct"], accuracy, "accuracy_pct"),
            (row["energy_kj_per_successful_task"], energy, "energy"),
            (row["throughput_tokens_per_s"], throughput, "throughput"),
            (row["peak_vram_cluster_gb"], vram, "peak VRAM"),
        )
        for observed, expected, label in checks:
            if not close(observed, expected):
                raise ValueError(f"T11 value changed: {row['tier']}/{label}")

    by_energy = sorted(rows, key=lambda row: row["energy_kj_per_successful_task"])
    front, best = [], -math.inf
    for row in by_energy:
        if row["accuracy_pct"] > best:
            front.append(row["tier"])
            best = row["accuracy_pct"]
    if tuple(front) != EXPECTED_FRONT:
        raise ValueError(f"recomputed Pareto front changed: {front}")
    return rows


def build_figure(rows: list[dict]):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.4,
        "axes.labelsize": 7.8,
        "axes.edgecolor": "#8d8d8d",
        "axes.linewidth": 0.55,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(137 / 25.4, 78 / 25.4))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.965, bottom=0.145)

    cells = {row["tier"]: row for row in rows}
    front_rows = [cells[tier] for tier in EXPECTED_FRONT]
    ax.plot(
        [row["energy_kj_per_successful_task"] for row in front_rows],
        [row["accuracy_pct"] for row in front_rows],
        color="#b64b4b", linestyle="--", linewidth=1.0, alpha=0.85, zorder=2,
    )
    size = {1: 26, 2: 52, 4: 104, 16: 240}
    for row in rows:
        on_front = row["tier"] in EXPECTED_FRONT
        ax.scatter(
            row["energy_kj_per_successful_task"], row["accuracy_pct"],
            s=size[int(row["gpus"])],
            facecolor="#4f85bd" if on_front else "#c9c9c9",
            edgecolor="black", linewidth=0.5, alpha=0.92, zorder=4,
        )
    offsets = {
        "0.5B": (6, 2, "left", "bottom"), "1.5B": (0, 7, "center", "bottom"),
        "7B": (0, 8, "center", "bottom"), "8B": (4, -10, "left", "top"),
        "14B": (-2, 9, "center", "bottom"), "32B": (7, -3, "left", "center"),
        "70B": (0, -12, "center", "top"), "80B": (-4, 10, "center", "bottom"),
        "120B": (8, -3, "left", "center"), "405B": (0, -14, "center", "top"),
        "480B": (-2, 12, "center", "bottom"),
    }
    for row in rows:
        dx, dy, horizontal, vertical = offsets[row["tier"]]
        ax.annotate(
            LABEL[row["tier"]],
            (row["energy_kj_per_successful_task"], row["accuracy_pct"]),
            xytext=(dx, dy), textcoords="offset points",
            ha=horizontal, va=vertical, fontsize=6.9, color="#404040",
        )

    ax.set_xscale("log")
    ax.set_xlim(0.08, 42)
    ax.set_ylim(-3, 63)
    ax.set_xlabel("Serving energy per successful task (kJ, log scale)")
    ax.set_ylabel("Result accuracy (%)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.48, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    front_handle = Line2D([], [], color="#b64b4b", linestyle="--", linewidth=1.0,
                          marker="o", markerfacecolor="#4f85bd",
                          markeredgecolor="black", markeredgewidth=0.5,
                          markersize=5.0, label="Pareto front")
    dominated_handle = Line2D([], [], linestyle="", marker="o",
                              markerfacecolor="#c9c9c9", markeredgecolor="black",
                              markeredgewidth=0.5, markersize=5.0, label="dominated")
    size_handles = [
        Line2D([], [], linestyle="", marker="o", markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=0.6,
               markersize=math.sqrt(size[gpus]), label=f"{gpus} GPU{'s' if gpus > 1 else ''}")
        for gpus in (1, 2, 4, 16)
    ]
    legend = ax.legend(handles=[front_handle, dominated_handle], loc="upper left",
                       frameon=False, fontsize=7.2, handletextpad=0.4, labelspacing=0.3)
    ax.add_artist(legend)
    ax.legend(handles=size_handles, title="marker size = GPUs", title_fontsize=7.2,
              loc="lower right", frameon=False, fontsize=7.0,
              handletextpad=0.5, labelspacing=0.85, borderaxespad=0.5)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.data.resolve().open(encoding="utf-8") as stream:
        data = json.load(stream)
    rows = validate(data)
    print(
        f"validated {len(rows)} frozen T11 serving rows x 7 fields and the "
        f"{len(EXPECTED_FRONT)}-point Pareto front {'->'.join(EXPECTED_FRONT)}"
    )
    figure = build_figure(rows)
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fixed_date = datetime(2026, 8, 11, tzinfo=timezone.utc)
    figure.savefig(output, format="pdf", dpi=300, metadata={
        "Title": "PowerCodeBench accuracy-energy serving Pareto scatter",
        "Author": "PowerCodeBench",
        "Creator": "code/build/plot_serving_pareto.py",
        "CreationDate": fixed_date,
        "ModDate": fixed_date,
    })
    print(f"wrote serving Pareto figure -> {output}")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
