#!/usr/bin/env python3
"""Render three manuscript result figures from the public compact aggregate.

The default invocation rebuilds, under ``assets/``:

* ``fig_cross_vendor_decomposition.pdf``;
* ``fig_task_heatmap.pdf``; and
* ``fig_model_scale_scatter.pdf``.

Only ``results/aggregates/primary_results_from_compact.json`` is read.  The
renderer checks the manuscript's cross-vendor table, every plotted endpoint,
and the per-task denominators before drawing.  It therefore fails loudly if a
future aggregate no longer supports the published figures.  Use ``--png`` to
also write deterministic raster previews next to the PDFs.
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
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "results/aggregates/primary_results_from_compact.json"
DEFAULT_OUT_DIR = ROOT / "assets"

PDF_METADATA = {
    "Title": "PowerCodeBench result breakdown",
    "Author": "PowerCodeBench authors",
    "Creator": "PowerCodeBench public renderer",
    "Producer": "Matplotlib",
    "CreationDate": datetime(2026, 8, 8, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 8, 8, tzinfo=timezone.utc),
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 8.0,
        "axes.titlesize": 8.8,
        "axes.labelsize": 8.3,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


DISPLAY = {
    "Qwen_Qwen2.5-Coder-1.5B-Instruct": ("Qwen2.5-1.5B", 1.5, "Qwen"),
    "Qwen_Qwen2.5-Coder-7B-Instruct": ("Qwen2.5-7B", 7.0, "Qwen"),
    "meta-llama_Llama-3.1-8B-Instruct": ("Llama-8B", 8.0, "Llama"),
    "Qwen_Qwen2.5-Coder-14B-Instruct": ("Qwen2.5-14B", 14.0, "Qwen"),
    "Qwen_Qwen2.5-Coder-32B-Instruct": ("Qwen2.5-32B", 32.0, "Qwen"),
    "meta-llama_Llama-3.1-70B-Instruct": ("Llama-70B", 70.0, "Llama"),
    "Qwen_Qwen3-Coder-Next": ("Qwen3-Next", 80.0, "Qwen"),
    "openai_gpt-oss-120b": ("GPT-OSS-120B", 120.0, "GPT-OSS"),
    "meta-llama_Llama-3.1-405B-Instruct": ("Llama-405B", 405.0, "Llama"),
    "Qwen_Qwen3-Coder-480B-A35B-Instruct": ("Qwen3-480B", 480.0, "Qwen"),
    "gpt-5.4-mini": ("GPT-5.4-mini", math.nan, "API"),
    "claude-haiku-4-5": ("Claude-Haiku", math.nan, "API"),
    "deepseek-v4-flash": ("DeepSeek-Flash", math.nan, "API"),
    "gemini-2.5-flash": ("Gemini-Flash", math.nan, "API"),
}

OPEN_MODELS = (
    "Qwen_Qwen2.5-Coder-1.5B-Instruct",
    "Qwen_Qwen2.5-Coder-7B-Instruct",
    "meta-llama_Llama-3.1-8B-Instruct",
    "Qwen_Qwen2.5-Coder-14B-Instruct",
    "Qwen_Qwen2.5-Coder-32B-Instruct",
    "meta-llama_Llama-3.1-70B-Instruct",
    "Qwen_Qwen3-Coder-Next",
    "openai_gpt-oss-120b",
    "meta-llama_Llama-3.1-405B-Instruct",
    "Qwen_Qwen3-Coder-480B-A35B-Instruct",
)

API_MODELS = (
    "gpt-5.4-mini",
    "claude-haiku-4-5",
    "deepseek-v4-flash",
    "gemini-2.5-flash",
)

# The main manuscript's complete cross-vendor table (fractions, not percent).
EXPECTED_T3 = {
    "gpt-5.4-mini": (0.2425, 0.5045, 0.6125, 0.6115, 0.6215),
    "claude-haiku-4-5": (0.1035, 0.3690, 0.6110, 0.6110, 0.6375),
    "deepseek-v4-flash": (0.1105, 0.4065, 0.4730, 0.4730, 0.5590),
    "gemini-2.5-flash": (0.0410, 0.4000, 0.4865, 0.4935, 0.5205),
    "meta-llama_Llama-3.1-405B-Instruct": (0.1345, 0.5145, 0.6475, 0.6475, 0.6870),
    "Qwen_Qwen3-Coder-480B-A35B-Instruct": (0.1505, 0.5665, 0.6380, 0.6375, 0.6590),
    "openai_gpt-oss-120b": (0.1405, 0.4565, 0.5840, 0.5835, 0.6025),
    "Qwen_Qwen3-Coder-Next": (0.0400, 0.4720, 0.5500, 0.5795, 0.5955),
    "meta-llama_Llama-3.1-70B-Instruct": (0.0500, 0.3815, 0.4900, 0.5145, 0.5745),
    "Qwen_Qwen2.5-Coder-32B-Instruct": (0.0380, 0.4270, 0.5075, 0.5275, 0.5275),
}
T3_COLUMNS = ("A", "C", "C_FX", "C_FDR", "C_FDRS")

# Endpoints in the scale figure.  These constants also cover the 1.5B/7B/
# 8B/14B rows that are outside the headline T3 table.
EXPECTED_SCALE = {
    "Qwen_Qwen2.5-Coder-1.5B-Instruct": (0.0000, 0.0810, 0.0960),
    "Qwen_Qwen2.5-Coder-7B-Instruct": (0.0040, 0.2315, 0.3280),
    "meta-llama_Llama-3.1-8B-Instruct": (0.0050, 0.2010, 0.3630),
    "Qwen_Qwen2.5-Coder-14B-Instruct": (0.0160, 0.3420, 0.4860),
    "Qwen_Qwen2.5-Coder-32B-Instruct": (0.0380, 0.4270, 0.5275),
    "meta-llama_Llama-3.1-70B-Instruct": (0.0500, 0.3815, 0.5745),
    "Qwen_Qwen3-Coder-Next": (0.0400, 0.4720, 0.5955),
    "openai_gpt-oss-120b": (0.1405, 0.4565, 0.6025),
    "meta-llama_Llama-3.1-405B-Instruct": (0.1345, 0.5145, 0.6870),
    "Qwen_Qwen3-Coder-480B-A35B-Instruct": (0.1505, 0.5665, 0.6590),
}

TASK_ORDER = (
    "sequential",
    "power_flow",
    "opf",
    "dc_power_flow",
    "diagnose_and_fix",
    "time_series",
    "comparison_opf",
    "comparison",
    "contingency",
    "parallel",
    "short_circuit_3ph",
    "state_estimation",
    "pf_then_sc",
    "short_circuit_2ph",
    "contingency_fix",
)

TASK_LABEL = {
    "sequential": "Sequential workflow",
    "power_flow": "Power flow",
    "opf": "Optimal power flow",
    "dc_power_flow": "DC power flow",
    "diagnose_and_fix": "Diagnose and repair",
    "time_series": "Time-series control",
    "comparison_opf": "OPF comparison",
    "comparison": "Model comparison",
    "contingency": "Contingency analysis",
    "parallel": "Parallel sweep",
    "short_circuit_3ph": "Three-phase short circuit",
    "state_estimation": "State estimation",
    "pf_then_sc": "Power flow then SC",
    "short_circuit_2ph": "Two-phase short circuit",
    "contingency_fix": "Contingency correction",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def close(observed: float, expected: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance)


def group_for(model: str) -> str:
    return "api" if model in API_MODELS else "comparison"


def run_summary(data: dict, model: str, condition: str) -> dict:
    key = f"{group_for(model)}|{model}|{condition}"
    try:
        row = data["run_summaries"][key]
    except KeyError as exc:
        raise ValueError(f"missing public run summary: {key}") from exc
    n = int(row["n"])
    matched = int(row["n_matched"])
    if n != 2000 or not close(row["accuracy"], matched / n):
        raise ValueError(f"invalid count/accuracy identity: {key}")
    per_task = row.get("per_task") or {}
    if per_task:
        if set(per_task) != set(TASK_ORDER):
            raise ValueError(f"unexpected task panel: {key}")
        task_n = sum(int(cell["n"]) for cell in per_task.values())
        task_matched = sum(int(cell["n_matched"]) for cell in per_task.values())
        if task_n != n or task_matched != matched:
            raise ValueError(f"per-task cells do not reconstruct total: {key}")
        for task, cell in per_task.items():
            if not close(cell["accuracy"], cell["n_matched"] / cell["n"]):
                raise ValueError(f"invalid per-task accuracy: {key}/{task}")
    return row


def validate(data: dict) -> None:
    if data.get("schema_version") != 1:
        raise ValueError("unsupported primary compact aggregate schema")

    table = {row["model"]: row for row in data.get("T3_cross_vendor", [])}
    if set(table) != set(EXPECTED_T3):
        raise ValueError("T3 cross-vendor model panel changed")
    for model, expected in EXPECTED_T3.items():
        for condition, target in zip(T3_COLUMNS, expected):
            if not close(table[model][condition], target):
                raise ValueError(f"T3 manuscript value changed: {model}/{condition}")
            actual = run_summary(data, model, condition)["accuracy"]
            if not close(actual, target):
                raise ValueError(f"T3/run-summary mismatch: {model}/{condition}")

    if set(EXPECTED_SCALE) != set(OPEN_MODELS):
        raise ValueError("internal scale-panel definition is incomplete")
    for model, expected in EXPECTED_SCALE.items():
        for condition, target in zip(("A", "C", "C_FDRS"), expected):
            actual = run_summary(data, model, condition)["accuracy"]
            if not close(actual, target):
                raise ValueError(f"scale endpoint changed: {model}/{condition}")

    # The heatmap deliberately excludes only the 1.5B capacity-floor row.
    for model in OPEN_MODELS[1:] + API_MODELS:
        for condition in ("A", "C_FDRS"):
            run_summary(data, model, condition)

    # Preserve the declared, data-derived task order used in the v2 figure.
    heat_models = OPEN_MODELS[1:] + API_MODELS
    score = {}
    for task in TASK_ORDER:
        values = []
        for model in heat_models:
            for condition in ("A", "C_FDRS"):
                values.append(run_summary(data, model, condition)["per_task"][task]["accuracy"])
        score[task] = float(np.mean(values))
    derived_order = tuple(sorted(TASK_ORDER, key=lambda task: -score[task]))
    if derived_order != TASK_ORDER:
        raise ValueError("data-derived per-task order changed")


def save_figure(fig, path: Path, write_png: bool) -> None:
    def display_path(value: Path) -> str:
        try:
            return str(value.relative_to(ROOT))
        except ValueError:
            return str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata=PDF_METADATA)
    if write_png:
        png = path.with_suffix(".png")
        fig.savefig(
            png,
            dpi=220,
            metadata={"Software": "PowerCodeBench public renderer"},
        )
        print(f"wrote {display_path(png)}")
    plt.close(fig)
    print(f"wrote {display_path(path)}")


def cross_vendor_figure(data: dict, include_next: bool = False):
    anchors = (
        "meta-llama_Llama-3.1-405B-Instruct",
        "Qwen_Qwen3-Coder-480B-A35B-Instruct",
        "openai_gpt-oss-120b",
        "meta-llama_Llama-3.1-70B-Instruct",
        "Qwen_Qwen2.5-Coder-32B-Instruct",
    )
    if include_next:
        # The v2 rendering adds the sixth open-weight anchor with the panel's
        # largest A -> C+FDRS lift (+55.55pp); validate() has already checked
        # its endpoints against EXPECTED_T3.
        anchors += ("Qwen_Qwen3-Coder-Next",)
    anchors += API_MODELS
    table = {row["model"]: row for row in data["T3_cross_vendor"]}
    models = sorted(anchors, key=lambda model: -table[model]["C_FDRS"])

    colors = {"base": "#c9c9c9", "proactive": "#5f91c8", "reactive": "#b64b4b"}
    labels = {
        "base": "A baseline",
        "proactive": "+ proactive (C-A)",
        "reactive": "+ reactive (FDRS-C)",
    }
    if include_next:
        # v2 rendering fills the manuscript line width 1:1 (150 mm placed at
        # \linewidth), so every source font size is the typeset size; the
        # absolute label gutter (left margin x width) matches the v1 rendering.
        fig, ax = plt.subplots(figsize=(150 / 25.4, 58 / 25.4))
        fig.subplots_adjust(left=0.21, right=0.985, top=0.98, bottom=0.27)
        # Wider axes mean more mm per data unit; keep the group marker at the
        # same absolute distance from the axis as the v1 gutter.
        marker_x = -0.76
    else:
        fig, ax = plt.subplots(figsize=(110 / 25.4, 67 / 25.4))
        fig.subplots_adjust(left=0.29, right=0.985, top=0.98, bottom=0.24)
        marker_x = -1.15
    y = np.arange(len(models))[::-1]

    for position, model in zip(y, models):
        row = table[model]
        parts = (
            ("base", 100 * row["A"]),
            ("proactive", 100 * (row["C"] - row["A"])),
            ("reactive", 100 * (row["C_FDRS"] - row["C"])),
        )
        left = 0.0
        for component, width in parts:
            if width < -1e-12:
                raise ValueError(f"negative contribution for {model}/{component}")
            ax.barh(
                position,
                width,
                left=left,
                height=0.67,
                color=colors[component],
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
            left += width
        group = DISPLAY[model][2]
        ax.scatter(
            marker_x,
            position,
            marker="s" if group == "API" else "o",
            s=20,
            facecolor="#7b5ca7" if group == "API" else "#247a75",
            edgecolor="black",
            linewidth=0.35,
            clip_on=False,
            zorder=5,
        )
        delta = 100 * (row["C_FDRS"] - row["A"])
        ax.text(left + 0.9, position, f"+{delta:.1f}", va="center", fontsize=7.2, color="#444444")

    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[model][0] for model in models])
    ax.set_xlim(0, 80)
    ax.set_ylim(-0.65, len(models) - 0.35)
    ax.set_xlabel("Accuracy contribution (percentage points)")
    ax.grid(axis="x", linestyle=":", linewidth=0.55, alpha=0.48, zorder=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[key]) for key in ("base", "proactive", "reactive")]
    fig.legend(
        handles,
        [labels[key] for key in ("base", "proactive", "reactive")],
        loc="lower center",
        bbox_to_anchor=(0.61, 0.015),
        ncol=3,
        frameon=False,
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    return fig


def task_heatmap_figure(data: dict, at_linewidth: bool = False):
    open_models = tuple(
        sorted(OPEN_MODELS[1:], key=lambda model: -run_summary(data, model, "C_FDRS")["accuracy"])
    )
    api_models = tuple(
        sorted(API_MODELS, key=lambda model: -run_summary(data, model, "C_FDRS")["accuracy"])
    )
    models = open_models + api_models

    def matrix(condition: str) -> np.ndarray:
        return np.array(
            [
                [100 * run_summary(data, model, condition)["per_task"][task]["accuracy"] for model in models]
                for task in TASK_ORDER
            ]
        )

    baseline = matrix("A")
    full = matrix("C_FDRS")
    short = {
        "meta-llama_Llama-3.1-405B-Instruct": "Llama-405B",
        "Qwen_Qwen3-Coder-480B-A35B-Instruct": "Qwen3-480B",
        "openai_gpt-oss-120b": "OSS-120B",
        "Qwen_Qwen3-Coder-Next": "Qwen3-Next",
        "meta-llama_Llama-3.1-70B-Instruct": "Llama-70B",
        "Qwen_Qwen2.5-Coder-32B-Instruct": "Qwen2.5-32B",
        "Qwen_Qwen2.5-Coder-14B-Instruct": "Qwen2.5-14B",
        "meta-llama_Llama-3.1-8B-Instruct": "Llama-8B",
        "Qwen_Qwen2.5-Coder-7B-Instruct": "Qwen2.5-7B",
        "gpt-5.4-mini": "GPT-5.4",
        "claude-haiku-4-5": "Claude",
        "deepseek-v4-flash": "DeepSeek",
        "gemini-2.5-flash": "Gemini",
    }

    if at_linewidth:
        # v2 rendering fills the manuscript line width 1:1 (the 150 mm source
        # canvas is placed at \linewidth), so every source font size below is
        # within 10% of the typeset size and the model-name labels keep an
        # effective size above 6.5 pt.  Margins are exact: the manuscript no
        # longer needs a trim/clip include.
        fig = plt.figure(figsize=(150 / 25.4, 75 / 25.4))
        grid_kwargs = dict(
            width_ratios=[1.0, 1.0, 0.045],
            wspace=0.12,
            left=0.235,
            right=0.955,
            top=0.86,
            bottom=0.205,
        )
        style = dict(
            model_fontsize=7.3, model_rotation=50, task_fontsize=7.3,
            cell_fontsize=6.2, title_fontsize=8.2, title_pad=16,
            group_fontsize=7.4, group_y=-0.88, bar_label_fontsize=7.6,
        )
    else:
        fig = plt.figure(figsize=(183 / 25.4, 106 / 25.4))
        grid_kwargs = dict(
            width_ratios=[1.0, 1.0, 0.04],
            wspace=0.15,
            left=0.195,
            right=0.955,
            top=0.85,
            bottom=0.24,
        )
        style = dict(
            model_fontsize=7.1, model_rotation=38, task_fontsize=7.4,
            cell_fontsize=6.7, title_fontsize=8.8, title_pad=17,
            group_fontsize=7.6, group_y=-0.88, bar_label_fontsize=8.3,
        )
    grid = fig.add_gridspec(1, 3, **grid_kwargs)
    axes = (fig.add_subplot(grid[0]), fig.add_subplot(grid[1]))
    cax = fig.add_subplot(grid[2])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=100)

    def draw(ax, values: np.ndarray, title: str, show_tasks: bool):
        image = ax.imshow(values, cmap="YlOrRd", norm=norm, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(
            [short[model] for model in models],
            rotation=style["model_rotation"],
            ha="right",
            fontsize=style["model_fontsize"],
        )
        if show_tasks:
            ax.set_yticks(range(len(TASK_ORDER)))
            ax.set_yticklabels([TASK_LABEL[task] for task in TASK_ORDER], fontsize=style["task_fontsize"])
        else:
            ax.set_yticks([])
        ax.axvline(len(open_models) - 0.5, color="black", linewidth=0.6, alpha=0.65)
        ax.set_title(title, pad=style["title_pad"], fontsize=style["title_fontsize"])
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                ax.text(
                    column,
                    row,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=style["cell_fontsize"],
                    color="white" if value >= 55 else "#222222",
                )
        ax.text(
            (len(open_models) - 1) / 2,
            style["group_y"],
            "open-weight",
            ha="center",
            va="bottom",
            fontsize=style["group_fontsize"],
            style="italic",
            color="#444444",
            clip_on=False,
        )
        ax.text(
            len(open_models) + (len(api_models) - 1) / 2,
            style["group_y"],
            "closed-source API",
            ha="center",
            va="bottom",
            fontsize=style["group_fontsize"],
            style="italic",
            color="#444444",
            clip_on=False,
        )
        ax.tick_params(length=2.2, width=0.55)
        return image

    image = draw(axes[0], baseline, "(a)  Baseline A (R0)", True)
    draw(axes[1], full, "(b)  Full method C+FDRS (R3)", False)
    bar = fig.colorbar(image, cax=cax)
    bar.set_label("Result accuracy (%)", fontsize=style["bar_label_fontsize"])
    bar.ax.tick_params(labelsize=7.2, width=0.5, length=2.2)
    bar.outline.set_linewidth(0.45)
    return fig


def model_scale_figure(data: dict):
    rows = []
    for model in OPEN_MODELS:
        name, parameters, family = DISPLAY[model]
        values = tuple(100 * run_summary(data, model, condition)["accuracy"] for condition in ("A", "C", "C_FDRS"))
        rows.append((model, name, parameters, family, *values))

    colors = {"A": "#858585", "C": "#4f85bd", "C_FDRS": "#b64b4b"}
    labels = {"A": "A baseline (R0)", "C": "C proactive (R0)", "C_FDRS": "C+FDRS full method (R3)"}
    markers = {"Qwen": "o", "Llama": "^", "GPT-OSS": "s"}
    fig, ax = plt.subplots(figsize=(120 / 25.4, 84 / 25.4), constrained_layout=True)

    for _, _, parameters, _, a, c, full in rows:
        ax.plot([parameters] * 3, [a, c, full], color="#c5c5c5", linewidth=0.75, zorder=2)

    for index, condition in enumerate(("A", "C", "C_FDRS"), start=4):
        x = np.array([row[2] for row in rows])
        y = np.array([row[index] for row in rows])
        for row in rows:
            ax.scatter(
                row[2],
                row[index],
                marker=markers[row[3]],
                s=35,
                facecolor=colors[condition],
                edgecolor="black",
                linewidth=0.45,
                zorder=4,
            )
        coefficients = np.polyfit(np.log10(x), y, 1)
        reference_x = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)
        ax.plot(
            reference_x,
            coefficients[0] * np.log10(reference_x) + coefficients[1],
            color=colors[condition],
            linestyle="--",
            linewidth=0.95,
            alpha=0.72,
            zorder=3,
        )

    offsets = {
        "Qwen_Qwen2.5-Coder-1.5B-Instruct": (0, 7, "center", "bottom"),
        "Qwen_Qwen2.5-Coder-7B-Instruct": (-5, 7, "right", "bottom"),
        "meta-llama_Llama-3.1-8B-Instruct": (5, 7, "left", "bottom"),
        "Qwen_Qwen2.5-Coder-14B-Instruct": (0, 7, "center", "bottom"),
        "Qwen_Qwen2.5-Coder-32B-Instruct": (0, 7, "center", "bottom"),
        "meta-llama_Llama-3.1-70B-Instruct": (-7, -7, "right", "top"),
        "Qwen_Qwen3-Coder-Next": (-5, 10, "right", "bottom"),
        "openai_gpt-oss-120b": (5, 8, "left", "bottom"),
        "meta-llama_Llama-3.1-405B-Instruct": (-4, 8, "right", "bottom"),
        "Qwen_Qwen3-Coder-480B-A35B-Instruct": (4, 7, "left", "bottom"),
    }
    short = {
        "Qwen_Qwen2.5-Coder-1.5B-Instruct": "1.5B",
        "Qwen_Qwen2.5-Coder-7B-Instruct": "7B",
        "meta-llama_Llama-3.1-8B-Instruct": "8B",
        "Qwen_Qwen2.5-Coder-14B-Instruct": "14B",
        "Qwen_Qwen2.5-Coder-32B-Instruct": "32B",
        "meta-llama_Llama-3.1-70B-Instruct": "70B",
        "Qwen_Qwen3-Coder-Next": "Next",
        "openai_gpt-oss-120b": "120B",
        "meta-llama_Llama-3.1-405B-Instruct": "405B",
        "Qwen_Qwen3-Coder-480B-A35B-Instruct": "480B",
    }
    for model, _, parameters, _, _, _, full in rows:
        dx, dy, horizontal, vertical = offsets[model]
        ax.annotate(
            short[model],
            (parameters, full),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=horizontal,
            va=vertical,
            fontsize=7.1,
            color="#404040",
        )

    api_endpoints = [100 * EXPECTED_T3[model][-1] for model in API_MODELS]
    api_low, api_high = min(api_endpoints), max(api_endpoints)
    ax.axhspan(api_low, api_high, color="#b64b4b", alpha=0.09, zorder=1)
    ax.text(
        1.7,
        (api_low + api_high) / 2,
        f"mid-tier API\nC+FDRS range\n({api_low:.0f}--{api_high:.0f}%)",
        fontsize=7.1,
        color="#854242",
        style="italic",
        ha="left",
        va="center",
    )

    ax.set_xscale("log")
    ax.set_xlim(1.1, 700)
    ax.set_ylim(-3, 82)
    ax.set_xlabel("Parameter count (B, log scale)")
    ax.set_ylabel("Result accuracy (%)")
    ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.5, zorder=0)
    condition_handles = [
        Line2D([], [], marker="o", linestyle="", color=colors[condition], markeredgecolor="black", markeredgewidth=0.4, markersize=5.5, label=labels[condition])
        for condition in ("A", "C", "C_FDRS")
    ]
    condition_legend = ax.legend(handles=condition_handles, loc="upper left", frameon=False, handletextpad=0.35, labelspacing=0.3)
    ax.add_artist(condition_legend)
    family_handles = [
        Line2D([], [], marker=markers[family], linestyle="", markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.6, markersize=5.5, label=family)
        for family in ("Qwen", "Llama", "GPT-OSS")
    ]
    ax.legend(handles=family_handles, title="family", title_fontsize=7.2, loc="lower right", frameon=False, handletextpad=0.35, labelspacing=0.25)
    return fig


FIGURES = {
    "cross-vendor": ("fig_cross_vendor_decomposition.pdf", cross_vendor_figure),
    "cross-vendor-v2": (
        "fig_cross_vendor_decomposition_v2.pdf",
        lambda data: cross_vendor_figure(data, include_next=True),
    ),
    "task-heatmap": ("fig_task_heatmap.pdf", task_heatmap_figure),
    "task-heatmap-v2": (
        "fig_task_heatmap_v2.pdf",
        lambda data: task_heatmap_figure(data, at_linewidth=True),
    ),
    "model-scale": ("fig_model_scale_scatter.pdf", model_scale_figure),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="public compact-derived aggregate")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="output directory")
    parser.add_argument("--figure", choices=("all", *FIGURES), default="all")
    parser.add_argument("--png", action="store_true", help="also write PNG previews")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.data)
    validate(data)
    selected = FIGURES if args.figure == "all" else {args.figure: FIGURES[args.figure]}
    for _, (filename, renderer) in selected.items():
        save_figure(renderer(data), args.out_dir / filename, args.png)


if __name__ == "__main__":
    main()
