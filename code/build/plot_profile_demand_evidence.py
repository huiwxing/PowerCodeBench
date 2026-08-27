#!/usr/bin/env python3
"""Render the knowledge-profile heatmap and demand recall@k figure.

The default invocation is release-only and needs no private repository::

    python3 code/build/plot_profile_demand_evidence.py

It writes ``assets/fig_knowledge_profile_heatmap.pdf``, its line-width 1:1
re-rendering ``assets/fig_knowledge_profile_heatmap_v2.pdf``, and
``assets/fig_demand_recall_k.pdf``; ``--figure`` selects a single output and
``--paper-fig-dir`` optionally copies the rendered PDF bytes into a
manuscript figure directory.

Every displayed heatmap cell and every plotted recall point is checked
against the frozen values used by the manuscript.  The script also verifies
the independently generated public aggregate files before drawing anything.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES = ROOT / "results/supplementary_evidence/probe_profiles.json"
DEFAULT_PROFILE_AGGREGATE = ROOT / "results/aggregates/supplementary_probe_profiles.json"
DEFAULT_DEMAND = ROOT / "results/supplementary_evidence/demand_estimators.json"
DEFAULT_DEMAND_AGGREGATE = ROOT / "results/aggregates/supplementary_demand_evidence.json"
DEFAULT_OUT_DIR = ROOT / "assets"

MM2IN = 1 / 25.4
KS = (1, 3, 5, 10, 20)

MODEL_COLUMNS = (
    ("Qwen2.5-1.5B", "Qwen2.5-1.5B"),
    ("Llama-8B", "Llama-8B"),
    ("Qwen2.5-7B", "Qwen2.5-7B"),
    ("Qwen2.5-14B", "Qwen2.5-14B"),
    ("Qwen2.5-32B", "Qwen2.5-32B"),
    ("Llama-70B", "Llama-70B"),
    ("GPT-OSS-120B", "OSS-120B"),
    ("Llama-405B", "Llama-405B"),
    ("Qwen3-Next", "Qwen3-Next"),
    ("Qwen3-480B", "Qwen3-480B"),
    ("Claude-Haiku-4-5", "Claude"),
    ("Gemini-2.5-Flash", "Gemini"),
    ("GPT-5.4-mini", "GPT-5.4"),
    ("DeepSeek-V4-Flash", "DeepSeek"),
)

CATEGORIES = (
    ("power_flow", "Power flow"),
    ("opf", "Optimal power flow"),
    ("short_circuit", "Short circuit"),
    ("state_estimation", "State estimation"),
    ("time_series_control", "Time-series/control"),
    ("topology", "Topology"),
    ("network_construction", "Network construction"),
    ("io_loading", "I/O and loading"),
    ("plotting", "Plotting"),
    ("utility_misc", "Other utilities"),
)

EXPECTED_CATEGORY_COUNTS = {
    "power_flow": 5,
    "opf": 5,
    "short_circuit": 6,
    "state_estimation": 6,
    "time_series_control": 18,
    "topology": 18,
    "network_construction": 105,
    "io_loading": 57,
    "plotting": 20,
    "utility_misc": 35,
}

# The 150 numbers printed in the manuscript heatmap, in hundredths.  The
# last entry in each row is the all-model mean.  Comparing formatted cells,
# rather than a tolerance around rounded aggregate data, preserves the two
# half-even boundary cells (0.625 -> 0.62 and 0.325 -> 0.32).
EXPECTED_HEATMAP_CENTI = (
    (74, 62, 45, 35, 35, 45, 32, 25, 47, 25, 38, 25, 20, 28, 38),
    (93, 73, 53, 36, 37, 48, 42, 28, 58, 36, 38, 44, 17, 37, 46),
    (85, 70, 59, 62, 59, 51, 56, 51, 58, 48, 44, 42, 37, 55, 55),
    (86, 78, 51, 53, 41, 50, 42, 41, 52, 43, 42, 31, 13, 53, 48),
    (81, 70, 57, 50, 48, 49, 44, 40, 53, 51, 38, 38, 27, 47, 50),
    (73, 64, 44, 43, 35, 40, 40, 32, 45, 35, 43, 39, 32, 35, 43),
    (74, 62, 46, 41, 36, 42, 39, 31, 40, 33, 31, 28, 25, 33, 40),
    (72, 55, 47, 27, 28, 40, 19, 32, 44, 20, 20, 27, 22, 31, 35),
    (80, 63, 51, 45, 44, 45, 43, 49, 43, 39, 44, 33, 41, 43, 47),
    (69, 68, 50, 41, 36, 43, 39, 34, 44, 36, 37, 38, 34, 41, 44),
)

DEMAND_RUNS = (
    ("Hybrid TF-IDF", "Hybrid TF-IDF (proposed)"),
    ("Zero-shot TF-IDF", "Zero-shot TF-IDF"),
    ("Zero-shot SBERT", "Zero-shot SBERT"),
    ("Zero-shot Cross-Encoder", "Zero-shot cross-encoder"),
    ("Pairwise TF-IDF+LogReg", "Pairwise TF-IDF + LogReg"),
    ("Hybrid Cross-Encoder", "Hybrid cross-encoder"),
)

DEMAND_SPLITS = (
    ("augmented_test", "Aug-test", "(a) Aug-test split"),
    ("benchmark_reference_eval_only", "Bench-ref", "(b) Bench-ref split"),
)

# Exact points in the manuscript recall@k figure, ordered by KS.  These
# constants make paper/renderer drift a hard failure rather than a visual
# inspection exercise.
EXPECTED_DEMAND_CURVES = {
    "Hybrid TF-IDF": {
        "Aug-test": (0.1270, 0.2697, 0.3408, 0.4708, 0.6776),
        "Bench-ref": (0.2506, 0.5002, 0.6014, 0.7184, 0.8002),
    },
    "Zero-shot TF-IDF": {
        "Aug-test": (0.0951, 0.1934, 0.2534, 0.3442, 0.4381),
        "Bench-ref": (0.1740, 0.3354, 0.4423, 0.6048, 0.7425),
    },
    "Zero-shot SBERT": {
        "Aug-test": (0.0981, 0.2049, 0.2613, 0.3417, 0.4345),
        "Bench-ref": (0.1308, 0.2056, 0.2521, 0.3313, 0.4552),
    },
    "Zero-shot Cross-Encoder": {
        "Aug-test": (0.1316, 0.2419, 0.2991, 0.3825, 0.4621),
        "Bench-ref": (0.1446, 0.2360, 0.2852, 0.3973, 0.6009),
    },
    "Pairwise TF-IDF+LogReg": {
        "Aug-test": (0.0615, 0.1636, 0.2338, 0.3067, 0.4566),
        "Bench-ref": (0.2198, 0.3083, 0.4181, 0.4885, 0.6082),
    },
    "Hybrid Cross-Encoder": {
        "Aug-test": (0.1394, 0.2817, 0.3565, 0.4799, 0.6075),
        "Bench-ref": (0.1505, 0.2401, 0.3019, 0.4316, 0.6430),
    },
}

PDF_METADATA = {
    "Creator": "PowerCodeBench public renderer",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def function_category(name: str) -> str:
    """Map one pandapower entry point to the manuscript's ten families."""
    n = name
    nl = n.lower()
    if ("_collection" in nl or "_trace" in nl or "plotly" in nl
            or "plot" in nl or "cmap_" in nl or nl.startswith("draw_")
            or "mapbox" in nl or "marker_trace" in nl
            or "annotation_collection" in nl):
        return "plotting"
    if nl in {"runpp", "runpp_3ph", "runpp_pgm", "rundcpp", "lf_info",
              "run_loadflow"}:
        return "power_flow"
    if (nl in {"runopp", "rundcopp", "opf_task"}
            or nl.startswith("create_poly_cost")
            or nl.startswith("create_pwl_cost")):
        return "opf"
    if (nl in {"calc_sc", "create_sc_bus", "calculate_protection_times"}
            or n in {"OCRelay", "Fuse", "SplineCharacteristic"}):
        return "short_circuit"
    if nl in {"estimate", "remove_bad_data", "chi2_analysis",
              "create_measurement", "drop_measurements_at_elements",
              "drop_duplicated_measurements"}:
        return "state_estimation"
    if (nl.endswith("control") or "controller" in nl
            or nl.startswith("characteristic") or nl in {
                "constcontrol", "outputwriter", "dfdata", "run_control",
                "run_timeseries", "log_variable", "create_svc",
                "create_tcsc", "create_vsc", "create_ssc",
            }):
        return "time_series_control"
    if ("nxgraph" in nl or "connected" in nl or "stubs" in nl
            or "boundaries_by_bus_zone" in nl or "set_bus_zone" in nl
            or "get_inner_branches" in nl or "drop_inner_branches" in nl
            or "get_connecting_branches" in nl or "elements_on_path" in nl
            or "next_bus" in nl or "get_equivalent" in nl
            or "calc_distance_to_bus" in nl):
        return "topology"
    if (nl.startswith("case") or nl.startswith("from_")
            or nl.startswith("to_") or nl.startswith("load_")
            or nl.startswith("dump_") or nl.startswith("delete_postgresql")
            or "_network" in nl or nl in {
                "simple_four_bus_system", "simple_mv_open_ring_net",
                "panda_four_load_branch", "mv_oberrhein", "iceland",
                "ieee_european_lv_asymmetric", "example_simple",
                "example_multivoltage", "GBnetwork", "GBreducednetwork",
                "four_loads_with_branches_out",
            } or n in {"GBnetwork", "GBreducednetwork"}):
        return "io_loading"
    if (nl.startswith("create_") or nl.startswith("add_")
            or nl.startswith("drop_") or nl.startswith("replace_")
            or nl.startswith("fuse_") or nl.startswith("attach_")
            or nl.startswith("detach_") or nl.startswith("set_isolated_")
            or "std_type" in nl or "set_data_type" in nl
            or nl.startswith("group") or "group" in nl
            or nl.startswith("change_") or nl.startswith("reindex_")
            or nl.startswith("merge_")):
        return "network_construction"
    return "utility_misc"


def profile_matrix(profiles_path: Path, aggregate_path: Path) -> np.ndarray:
    profiles = load_json(profiles_path)
    aggregate = load_json(aggregate_path)
    models = profiles["models"]
    aggregated_risk = aggregate["risk_profiles"]
    matrix = np.zeros((len(CATEGORIES), len(MODEL_COLUMNS) + 1))

    for model_index, (model, _) in enumerate(MODEL_COLUMNS):
        profile = models[model]["profile"]
        counts = Counter(function_category(fn) for fn in profile)
        if counts != Counter(EXPECTED_CATEGORY_COUNTS):
            raise RuntimeError(f"category membership drift for {model}: {dict(counts)}")

        # Preserve the manuscript renderer's arithmetic order: average each
        # function's score, average those scores by category, then subtract
        # from one.  Averaging per-function risks is algebraically equivalent
        # but changes the last floating-point bit of a 0.325 boundary cell.
        category_scores: dict[str, list[float]] = {key: [] for key, _ in CATEGORIES}
        public_uniform = aggregated_risk[model]["uniform"]["per_function"]
        for function, levels in profile.items():
            scores = [
                float(levels[level]["score"])
                for level in ("L0", "L1", "L2", "L3")
                if isinstance(levels.get(level), dict)
                and levels[level].get("score") is not None
            ]
            if not scores:
                raise RuntimeError(f"{model}/{function} has no defined probe score")
            mean_score = sum(scores) / len(scores)
            risk = 1.0 - mean_score
            if not math.isclose(risk, public_uniform[function], abs_tol=1.1e-6):
                raise RuntimeError(
                    f"raw/aggregate uniform-risk mismatch for {model}/{function}: "
                    f"{risk} != {public_uniform[function]}"
                )
            category_scores[function_category(function)].append(mean_score)

        for category_index, (category, _) in enumerate(CATEGORIES):
            values = category_scores[category]
            matrix[category_index, model_index] = 1.0 - sum(values) / len(values)

    matrix[:, -1] = matrix[:, :-1].mean(axis=1)
    shown = tuple(tuple(f"{value:.2f}" for value in row) for row in matrix)
    expected = tuple(
        tuple(f"{hundredths / 100:.2f}" for hundredths in row)
        for row in EXPECTED_HEATMAP_CENTI
    )
    if shown != expected:
        differences = [
            (CATEGORIES[i][0], j, expected[i][j], shown[i][j])
            for i in range(len(shown))
            for j in range(len(shown[i]))
            if shown[i][j] != expected[i][j]
        ]
        raise RuntimeError(f"heatmap/manuscript cell drift: {differences}")
    return matrix


def demand_curves(demand_path: Path, aggregate_path: Path) -> dict:
    demand = load_json(demand_path)
    aggregate = load_json(aggregate_path)
    if demand.get("suite") != "suite_4428033":
        raise RuntimeError(f"unexpected demand suite: {demand.get('suite')}")
    curves: dict[str, dict[str, tuple[float, ...]]] = {}

    for run, _ in DEMAND_RUNS:
        curves[run] = {}
        for source_split, display_split, _ in DEMAND_SPLITS:
            top_k = demand["runs"][run]["metrics"][source_split]["top_k"]
            values = tuple(float(top_k[str(k)]["recall"]) for k in KS)
            expected = EXPECTED_DEMAND_CURVES[run][display_split]
            if any(not math.isclose(a, b, abs_tol=5e-8)
                   for a, b in zip(values, expected)):
                raise RuntimeError(
                    f"demand curve/manuscript drift for {run}/{display_split}: "
                    f"{values} != {expected}"
                )
            aggregate_values = tuple(
                float(aggregate["s4_recall_hit_curves"][run][display_split][str(k)]["recall"])
                for k in KS
            )
            if values != aggregate_values:
                raise RuntimeError(
                    f"demand evidence/aggregate drift for {run}/{display_split}: "
                    f"{values} != {aggregate_values}"
                )
            curves[run][display_split] = values
    return curves


def save_pdf(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(
            path,
            bbox_inches="tight",
            pad_inches=0.05,
            metadata=PDF_METADATA,
        )
    else:
        # Exact-canvas save: the figure size in the renderer is the physical
        # PDF size, so a source drawn at the manuscript line width is typeset
        # 1:1 and every font size below is the printed size.
        fig.savefig(path, metadata=PDF_METADATA)
    plt.close(fig)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"renderer produced an empty PDF: {path}")
    print(f"wrote {path}")


def render_profile_heatmap(matrix: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(183 * MM2IN, 105 * MM2IN))
    vmin, vmax = 0.10, 0.80
    image = ax.imshow(
        matrix,
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    columns = [label for _, label in MODEL_COLUMNS] + ["All-model\nmean"]
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=35, ha="right", fontsize=7.8)
    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_yticklabels([label for _, label in CATEGORIES], fontsize=8.2)

    midpoint = (vmin + vmax) / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if value >= midpoint + (vmax - midpoint) * 0.5 else "#222222"
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.6,
                color=color,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.022, pad=0.015)
    colorbar.set_label(
        r"Risk: $\rho=1-\mathrm{mean}(L_0,\ldots,L_3)$",
        fontsize=8.2,
    )
    colorbar.ax.tick_params(labelsize=7.8, width=0.5, length=2)
    colorbar.outline.set_linewidth(0.4)
    ax.axvline(9.5, color="black", linewidth=0.7, alpha=0.7)
    ax.axvline(13.5, color="black", linewidth=0.7)
    ax.text(
        4.5,
        -0.85,
        "open-weight",
        ha="center",
        va="bottom",
        fontsize=8.0,
        style="italic",
        color="#444444",
        clip_on=False,
    )
    ax.text(
        11.5,
        -0.85,
        "closed-source API",
        ha="center",
        va="bottom",
        fontsize=8.0,
        style="italic",
        color="#444444",
        clip_on=False,
    )
    ax.tick_params(axis="both", length=2.5, width=0.5)
    fig.subplots_adjust(top=0.88, bottom=0.27, left=0.12, right=0.94)
    save_pdf(fig, path)


def render_profile_heatmap_v2(matrix: np.ndarray, path: Path) -> None:
    """Line-width 1:1 re-rendering of the knowledge-profile heatmap.

    The 150 mm source canvas is placed at ``\\linewidth`` (137.6 mm for the
    manuscript class), so every source font size is within 10% of the typeset
    size; the model-name labels keep an effective size above 6.5 pt.  The
    canvas is saved exactly (no tight bounding box), and the displayed cell
    values are the same 150 validated numbers as the v1 rendering.
    """
    fig, ax = plt.subplots(figsize=(150 * MM2IN, 66 * MM2IN))
    vmin, vmax = 0.10, 0.80
    image = ax.imshow(
        matrix,
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    columns = [label for _, label in MODEL_COLUMNS] + ["All-model mean"]
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_yticklabels([label for _, label in CATEGORIES], fontsize=7.5)

    midpoint = (vmin + vmax) / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if value >= midpoint + (vmax - midpoint) * 0.5 else "#222222"
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color=color,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.022, pad=0.015)
    colorbar.set_label(
        r"Risk: $\rho=1-\mathrm{mean}(L_0,\ldots,L_3)$",
        fontsize=7.5,
    )
    colorbar.ax.tick_params(labelsize=7.2, width=0.5, length=2)
    colorbar.outline.set_linewidth(0.4)
    ax.axvline(9.5, color="black", linewidth=0.7, alpha=0.7)
    ax.axvline(13.5, color="black", linewidth=0.7)
    ax.text(
        4.5,
        -0.72,
        "open-weight",
        ha="center",
        va="bottom",
        fontsize=7.5,
        style="italic",
        color="#444444",
        clip_on=False,
    )
    ax.text(
        11.5,
        -0.72,
        "closed-source API",
        ha="center",
        va="bottom",
        fontsize=7.5,
        style="italic",
        color="#444444",
        clip_on=False,
    )
    ax.tick_params(axis="both", length=2.5, width=0.5)
    fig.subplots_adjust(top=0.885, bottom=0.30, left=0.21, right=0.92)
    save_pdf(fig, path, tight=False)


def render_demand_curves(curves: dict, path: Path) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(183 * MM2IN, 75 * MM2IN),
        sharey=True,
    )
    colors = plt.colormaps["tab10"]
    for panel, (_, display_split, title) in enumerate(DEMAND_SPLITS):
        ax = axes[panel]
        for run_index, (run, label) in enumerate(DEMAND_RUNS):
            ax.plot(
                KS,
                curves[run][display_split],
                marker="o",
                linewidth=1.7,
                label=label,
                color=colors(run_index),
            )
        ax.axvline(10, color="#999999", linestyle=":", linewidth=1.0)
        ax.set_xticks(KS)
        ax.set_xlabel("Top-$k$")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Recall@$k$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save_pdf(fig, path)


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--profile-aggregate", type=Path, default=DEFAULT_PROFILE_AGGREGATE)
    parser.add_argument("--demand", type=Path, default=DEFAULT_DEMAND)
    parser.add_argument("--demand-aggregate", type=Path, default=DEFAULT_DEMAND_AGGREGATE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--figure",
        choices=("all", "profile-heatmap", "profile-heatmap-v2", "demand-recall"),
        default="all",
        help="render every figure (default) or a single named one",
    )
    parser.add_argument(
        "--paper-fig-dir",
        type=Path,
        help="optional manuscript figure directory receiving byte-identical PDF copies",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = profile_matrix(args.profiles.resolve(), args.profile_aggregate.resolve())
    curves = demand_curves(args.demand.resolve(), args.demand_aggregate.resolve())
    configure_matplotlib()

    out_dir = args.out_dir.resolve()
    catalogue = (
        ("profile-heatmap", "fig_knowledge_profile_heatmap.pdf", render_profile_heatmap, matrix),
        ("profile-heatmap-v2", "fig_knowledge_profile_heatmap_v2.pdf", render_profile_heatmap_v2, matrix),
        ("demand-recall", "fig_demand_recall_k.pdf", render_demand_curves, curves),
    )
    outputs = tuple(
        (name, renderer, values)
        for key, name, renderer, values in catalogue
        if args.figure in ("all", key)
    )
    for name, renderer, values in outputs:
        renderer(values, out_dir / name)

    if args.paper_fig_dir:
        paper_dir = args.paper_fig_dir.resolve()
        paper_dir.mkdir(parents=True, exist_ok=True)
        for name, _, _ in outputs:
            source = out_dir / name
            target = paper_dir / name
            shutil.copyfile(source, target)
            print(f"copied {source} -> {target}")

    print("validated 150 heatmap cells and 60 demand-curve points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
