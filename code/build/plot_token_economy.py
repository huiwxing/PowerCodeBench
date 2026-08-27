#!/usr/bin/env python3
"""Build SM Fig. S5 from the public pooled-exact token evidence.

The large diamonds in panels (a)--(b) are only the values in
``T6_token_economy.pooled_exact_display``: totals are pooled over all ten
open-weight models (including Qwen2.5-1.5B) before division.  Small circles
show the individual model cells that those totals pool over, so the diamonds
sit at the pooled value rather than at the mean of the circles.  Panels
(c.1)--(c.2) expose the corresponding per-model prompt and
router-documentation decompositions.

Before drawing, every plotted quantity is validated: each of the ten T6
model cells must be the exact mean of its own frozen totals, each pooled-exact
display value must be the pooled total of those cells, and the routed
documentation shares of panel (c.2) must be rebuilt from the public
item-round router events and agree with both the archived route summaries and
the T6 C+FDRS cells.  Any mismatch raises instead of producing a figure.

Requires matplotlib, and runs on CPU with no network access: the only
inputs are two public JSONs in this repository.
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
from matplotlib.patches import Patch  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIMARY = ROOT / "results/aggregates/primary_results_from_compact.json"
DEFAULT_ROUTER = ROOT / "results/supplementary_evidence/reactive_router_counts.json"
DEFAULT_OUTPUT = ROOT / "assets/fig_token_economy.pdf"
DEFAULT_PNG = ROOT / "assets/fig_token_economy.png"
DEFAULT_COMPACT = ROOT / "assets/fig_token_economy_compact.pdf"

MODELS = (
    ("Qwen_Qwen2.5-Coder-1.5B-Instruct", "Q2.5-1.5B", "Qwen2.5-1.5B"),
    ("Qwen_Qwen2.5-Coder-7B-Instruct", "Q2.5-7B", "Qwen2.5-7B"),
    ("meta-llama_Llama-3.1-8B-Instruct", "L-8B", "Llama-8B"),
    ("Qwen_Qwen2.5-Coder-14B-Instruct", "Q2.5-14B", "Qwen2.5-14B"),
    ("Qwen_Qwen2.5-Coder-32B-Instruct", "Q2.5-32B", "Qwen2.5-32B"),
    ("meta-llama_Llama-3.1-70B-Instruct", "L-70B", "Llama-70B"),
    ("openai_gpt-oss-120b", "OSS-120B", "GPT-OSS-120B"),
    ("Qwen_Qwen3-Coder-Next", "Q3-Next", "Qwen3-Next"),
    ("meta-llama_Llama-3.1-405B-Instruct", "L-405B", "Llama-405B"),
    ("Qwen_Qwen3-Coder-480B-A35B-Instruct", "Q3-480B", "Qwen3-480B"),
)

PROACTIVE = ("A", "B", "C", "X", "R")
REACTIVE = ("C_FX", "C_FD", "C_FDR", "C_FDRS")
COLORS = {
    "A": "#9a9a9a",
    "B": "#666666",
    "C": "#b44747",
    "X": "#326ea8",
    "R": "#8a5a2e",
    "C_FX": "#8c8c8c",
    "C_FD": "#326ea8",
    "C_FDR": "#d18435",
    "C_FDRS": "#b44747",
}
DISPLAY = {
    "A": "A",
    "B": "B",
    "C": "C",
    "X": "X",
    "R": "R",
    "C_FX": "FX",
    "C_FD": "FD",
    "C_FDR": "FDR",
    "C_FDRS": "FDRS",
}
ITEMS_PER_MODEL = 2000
ROUTES = ("api_doc", "boundary_contract", "semantic_fix")
BASIC_FIX_ROUTE = "basic_fix"
ROUTE_COLORS = {
    "api_doc": "#326ea8",
    "boundary_contract": "#d18435",
    "semantic_fix": "#b44747",
}
ROUTE_LABELS = {
    "api_doc": "API docs",
    "boundary_contract": "boundary contract",
    "semantic_fix": "semantic fix",
}

# Frozen T5 reactive-ladder panel means (accuracy fractions over the ten-model
# panel), exactly as printed in the manuscript's Table 5.  The compact figure's
# panel (c) is drawn only after these constants are re-derived from both the
# frozen aggregate block and the underlying per-model run summaries.
LADDER_ROUNDS = ("R0", "FX", "FD", "FDR", "FDRS")
EXPECTED_T5_PANEL_MEAN = {
    "A": {"R0": 0.05785, "FX": 0.1167, "FD": 0.1231, "FDR": 0.1428, "FDRS": 0.1542},
    "C": {"R0": 0.36735, "FX": 0.45265, "FD": 0.47095, "FDR": 0.4703, "FDRS": 0.4919},
    "X": {"R0": 0.3813, "FX": 0.45055, "FD": 0.4608, "FDR": 0.4687, "FDRS": 0.4825},
}


# Scale applied to the hard-coded panel font sizes.  The full-size figure
# keeps 1.0 (output unchanged); the compact variant is typeset at \linewidth
# with a ~0.75 shrink, so it draws its text 1.30x larger (panel titles capped
# at TITLE_SIZE) to keep the typeset sizes readable.
FONT_SCALE = 1.0
TITLE_SIZE = 8.6


def fs(size: float) -> float:
    return round(size * FONT_SCALE, 2)


def load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)


def keyed(rows: list[dict]) -> dict[str, dict]:
    result = {row["model"]: row for row in rows}
    expected = {model for model, _, _ in MODELS}
    if set(result) != expected:
        raise ValueError("T6 per-model rows do not equal the ten-model panel")
    return result


def validate_t6(t6: dict) -> None:
    """Raise if the displayed diamonds are not pooled exact totals."""
    pooled = t6.get("pooled_exact_display") or {}
    proactive = t6.get("proactive") or {}
    reactive = t6.get("reactive") or {}
    if set(proactive) != set(PROACTIVE) or set(reactive) != set(REACTIVE):
        raise ValueError("T6 conditions are incomplete")

    x_prompt = pooled["proactive"]["X"]["prompt_avg"]
    for condition in PROACTIVE:
        row = proactive[condition]
        per_model = keyed(row["per_model"])
        display = pooled["proactive"][condition]
        n_items = ITEMS_PER_MODEL * len(MODELS)
        prompt_total = sum(cell["prompt_total"] for cell in per_model.values())
        successes = sum(cell["n_matched"] for cell in per_model.values())
        # Every plotted model cell must be the exact per-item mean of its own
        # frozen totals; the diamonds are then the pooled totals of those cells.
        for model, cell in per_model.items():
            if not close(cell["prompt_avg_exact_from_items"],
                         cell["prompt_total"] / ITEMS_PER_MODEL):
                raise ValueError(f"{condition}/{model}: per-model prompt_avg is not exact")
        expected_docs = 0.0
        if condition != "A":
            a_rows = keyed(proactive["A"]["per_model"])
            expected_docs = (
                prompt_total - sum(cell["prompt_total"] for cell in a_rows.values())
            ) / n_items
        checks = (
            (display["prompt_avg"], prompt_total / n_items, "prompt_avg"),
            (display["successes_avg_per_model"], successes / len(MODELS), "successes"),
            (display["prompt_tokens_per_success"], prompt_total / successes, "tokens/success"),
            (display["docs_avg"], expected_docs, "added/docs"),
            (display["vs_x_prompt_ratio"], prompt_total / n_items / x_prompt, "vs X"),
        )
        for observed, expected, label in checks:
            if not close(observed, expected):
                raise ValueError(f"{condition}: pooled {label} mismatch")

    for condition in REACTIVE:
        row = reactive[condition]
        per_model = keyed(row["per_model"])
        display = pooled["reactive"][condition]
        attempts = sum(cell["attempts"] for cell in per_model.values())
        docs_total = sum(cell["docs_total"] for cell in per_model.values())
        prompt_total = sum(cell["prompt_total"] for cell in per_model.values())
        successes = sum(cell["successful_fixes"] for cell in per_model.values())
        for model, cell in per_model.items():
            per_model_checks = (
                (cell["prompt_avg_exact"], cell["prompt_total"] / cell["attempts"], "prompt_avg"),
                (cell["docs_avg_exact"], cell["docs_total"] / cell["attempts"], "docs_avg"),
            )
            for observed, expected, label in per_model_checks:
                if not close(observed, expected):
                    raise ValueError(f"{condition}/{model}: per-model {label} is not exact")
        checks = (
            (display["docs_avg"], docs_total / attempts, "docs_avg"),
            (display["prompt_avg"], prompt_total / attempts, "prompt_avg"),
            (display["successes_avg_per_model"], successes / len(MODELS), "successes"),
            (display["prompt_tokens_per_success"], prompt_total / successes, "tokens/success"),
            (display["vs_x_prompt_ratio"], prompt_total / attempts / x_prompt, "vs X"),
        )
        for observed, expected, label in checks:
            if not close(observed, expected):
                raise ValueError(f"{condition}: pooled {label} mismatch")


def route_totals_from_events(condition: dict, routes: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """Re-aggregate the compact item-round event stream into per-route totals.

    ``events`` rows are positional and described by ``event_fields``;
    ``route_index`` addresses the artifact's ``route_dictionary``.
    """
    fields = list(condition["event_fields"])
    try:
        route_at = fields.index("route_index")
        docs_at = fields.index("docs_tokens")
        prompt_at = fields.index("prompt_tokens")
        success_at = fields.index("success")
    except ValueError as exc:
        raise ValueError(f"router event_fields lack a required column: {fields}") from exc
    totals = {
        route: {"attempts": 0, "docs": 0, "prompt": 0, "successful_fixes": 0}
        for route in routes
    }
    for event in condition["events"]:
        route_index = int(event[route_at])
        if not 0 <= route_index < len(routes):
            raise ValueError(f"router event names an unknown route index {route_index}")
        cell = totals[routes[route_index]]
        cell["attempts"] += 1
        cell["docs"] += int(event[docs_at])
        cell["prompt"] += int(event[prompt_at])
        cell["successful_fixes"] += int(event[success_at])
    return totals


def validate_router(router: dict, t6: dict) -> dict[str, dict[str, float]]:
    """Raise unless the router evidence rebuilds every plotted FDRS cell.

    The public artifact stores per-route summaries under
    ``source_summary_route_stats`` and the underlying item-round records under
    ``events``/``event_fields``.  Both levels are recomputed here and checked
    against each other and against the T6 C+FDRS model cells, so panel (c.2)
    is drawn only from route shares that agree with panel (b).
    """
    routes = tuple(router.get("route_dictionary") or ())
    missing = [route for route in ROUTES + (BASIC_FIX_ROUTE,) if route not in routes]
    if missing:
        raise ValueError(f"router route dictionary lacks {missing}")
    evidence = router.get("models") or {}
    fdrs_rows = keyed(t6["reactive"]["C_FDRS"]["per_model"])
    contributions = {}
    for source_model, _, router_label in MODELS:
        try:
            entry = evidence[router_label]
            condition = entry["conditions"]["C+FDRS"]
        except KeyError as exc:
            raise ValueError(f"router evidence lacks {router_label}/C+FDRS") from exc
        if entry.get("source_model") != source_model:
            raise ValueError(f"router source-model mismatch for {router_label}")

        summary = condition["source_summary_route_stats"]
        unknown = sorted(set(summary) - set(routes))
        if unknown:
            raise ValueError(f"router summary names unknown routes {unknown} for {router_label}")
        observed = route_totals_from_events(condition, routes)
        for route in routes:
            recorded = summary.get(route)
            if recorded is None:
                if observed[route]["attempts"]:
                    raise ValueError(
                        f"router events use unsummarised route {route} for {router_label}")
                continue
            expected = {
                "attempts": int(recorded["attempts"]),
                "docs": int(recorded["docs"]["total"]),
                "prompt": int(recorded["prompt"]["total"]),
                "successful_fixes": int(recorded["successful_fixes"]),
            }
            if observed[route] != expected:
                raise ValueError(
                    f"router events do not rebuild the {route} summary for {router_label}: "
                    f"{observed[route]} != {expected}")

        cell = fdrs_rows[source_model]
        attempts = sum(part["attempts"] for part in observed.values())
        t6_checks = (
            (attempts, int(cell["attempts"]), "attempts"),
            (sum(part["docs"] for part in observed.values()), int(cell["docs_total"]), "docs"),
            (sum(part["prompt"] for part in observed.values()), int(cell["prompt_total"]), "prompt"),
            (sum(part["successful_fixes"] for part in observed.values()),
             int(cell["successful_fixes"]), "successful fixes"),
        )
        for got, want, label in t6_checks:
            if got != want:
                raise ValueError(
                    f"router {label} do not match the T6 C+FDRS cell for {router_label}: "
                    f"{got} != {want}")
        if observed[BASIC_FIX_ROUTE]["docs"]:
            raise ValueError(f"basic-fix route unexpectedly carries docs for {router_label}")

        parts = {route: observed[route]["docs"] / attempts for route in ROUTES}
        if not close(sum(parts.values()), float(cell["docs_avg_exact"])):
            raise ValueError(f"router docs do not match T6 for {router_label}")
        contributions[source_model] = parts
    if len(contributions) != len(MODELS):
        raise ValueError("router evidence does not cover all ten models")
    return contributions


def style_axes(ax) -> None:
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.48, zorder=0)
    ax.tick_params(axis="both", length=2.4, width=0.55)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_pooled_scatter(ax, t6: dict, section: str) -> None:
    conditions = PROACTIVE if section == "proactive" else REACTIVE
    pooled = t6["pooled_exact_display"][section]
    rows = t6[section]
    for condition in conditions:
        per_model = keyed(rows[condition]["per_model"])
        if section == "proactive":
            xs = [per_model[model]["prompt_avg_exact_from_items"] for model, _, _ in MODELS]
            ys = [per_model[model]["n_matched"] for model, _, _ in MODELS]
        else:
            xs = [per_model[model]["prompt_avg_exact"] for model, _, _ in MODELS]
            ys = [per_model[model]["successful_fixes"] for model, _, _ in MODELS]
        ax.scatter(
            xs, ys, s=16, facecolors="none", edgecolors=COLORS[condition],
            linewidths=0.7, alpha=0.62, zorder=2,
        )
        marker = pooled[condition]
        x = marker["prompt_avg"]
        y = marker["successes_avg_per_model"]
        ax.scatter(
            [x], [y], marker="D", s=58, color=COLORS[condition],
            edgecolor="black", linewidth=0.65, zorder=5,
        )
        offsets = {
            "A": (7, -2), "B": (7, 5), "C": (7, 5), "X": (7, -11), "R": (-8, 4),
            "C_FX": (-8, -10), "C_FD": (7, -11), "C_FDR": (-8, 5), "C_FDRS": (7, 5),
        }
        dx, dy = offsets[condition]
        ax.annotate(
            DISPLAY[condition], (x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=fs(7.2), fontweight="bold", color=COLORS[condition],
            ha="left" if dx > 0 else "right",
        )

    legend = [
        Line2D([], [], marker="D", linestyle="", color=COLORS[condition],
               markeredgecolor="black", markeredgewidth=0.5,
               markersize=5.2, label=DISPLAY[condition])
        for condition in conditions
    ]
    ax.legend(
        handles=legend, frameon=False, fontsize=fs(6.6), ncol=len(conditions),
        loc="upper center", bbox_to_anchor=(0.5, -0.245),
        handletextpad=0.25, columnspacing=0.7, borderaxespad=0,
    )
    style_axes(ax)


def panel_proactive(ax, t6: dict) -> None:
    panel_pooled_scatter(ax, t6, "proactive")
    pooled = t6["pooled_exact_display"]["proactive"]
    c, x = pooled["C"], pooled["X"]
    success_ratio = c["successes_avg_per_model"] / x["successes_avg_per_model"]
    ax.text(
        0.98, 0.96,
        f"C retains {100 * success_ratio:.1f}% of X successes\n"
        f"at {100 * c['vs_x_prompt_ratio']:.1f}% of X prompt",
        transform=ax.transAxes, ha="right", va="top", fontsize=fs(7.0),
        color="#3f3f3f",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bdbdbd", "linewidth": 0.5},
    )
    ax.text(
        0.02, 0.82, "◆ pooled exact; ○ model cells\n10 models, including 1.5B",
        transform=ax.transAxes, fontsize=fs(6.1), color="#555555", va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )
    ax.set_xlim(0, 4500)
    ax.set_ylim(-30, 1420)
    ax.set_yticks(range(0, 1001, 200))
    ax.set_xlabel("Prompt tokens per item")
    if FONT_SCALE == 1.0:
        ax.set_ylabel("Successful R0 items per model (of 2,000)")
    else:
        # The larger compact text outruns the panel height on one line.
        ax.set_ylabel("Successful R0 items\nper model (of 2,000)")
    ax.set_title("(a) Proactive: prompt vs R0 successes", loc="left", fontsize=TITLE_SIZE)


def panel_reactive(ax, t6: dict) -> None:
    panel_pooled_scatter(ax, t6, "reactive")
    pooled = t6["pooled_exact_display"]["reactive"]
    fd, fdr, fdrs = pooled["C_FD"], pooled["C_FDR"], pooled["C_FDRS"]
    ax.text(
        0.98, 0.96,
        f"FDR: {100 * fdr['successes_avg_per_model'] / fd['successes_avg_per_model']:.1f}% of FD fixes\n"
        f"at {100 * fdr['prompt_avg'] / fd['prompt_avg']:.1f}% of FD prompt\n"
        f"FDRS: {100 * fdrs['successes_avg_per_model'] / fd['successes_avg_per_model']:.1f}% of FD fixes",
        transform=ax.transAxes, ha="right", va="top", fontsize=fs(7.0),
        color="#3f3f3f",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bdbdbd", "linewidth": 0.5},
    )
    # In the compact variant panel (a) drops its copy of this marker key, and
    # the larger text spans the narrow panel, so the key moves into panel
    # (b)'s lower data-free band.
    ax.text(
        0.02, 0.96 if FONT_SCALE == 1.0 else 0.28,
        "◆ pooled exact; ○ model cells\n10 models, including 1.5B",
        transform=ax.transAxes, fontsize=fs(6.1), color="#555555", va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )
    ax.set_xlim(300, 2150)
    ax.set_ylim(0, 420)
    ax.set_xlabel("Fix-round prompt tokens per attempt")
    ax.set_ylabel("Successful fixes per model")
    ax.set_title("(b) Reactive: prompt vs successful fixes", loc="left", fontsize=TITLE_SIZE)


def validate_t5(primary: dict) -> dict[str, dict[str, float]]:
    """Raise unless the frozen T5 ladder equals its EXPECTED constants.

    Each of the fifteen panel means must equal the hard-coded manuscript value
    and must be re-derivable as the mean of the ten per-model run-summary
    accuracies of its condition (with each accuracy equal to its own
    ``n_matched / n`` identity over the full 2,000-item benchmark).
    """
    t5 = primary["T5_reactive_matrix_panel_mean"]
    summaries = primary["run_summaries"]
    if set(t5) != set(EXPECTED_T5_PANEL_MEAN):
        raise ValueError("T5 condition families changed")
    for family, expected_rounds in EXPECTED_T5_PANEL_MEAN.items():
        if set(t5[family]) != set(LADDER_ROUNDS):
            raise ValueError(f"T5 ladder rounds changed for {family}")
        for round_name, expected in expected_rounds.items():
            observed = t5[family][round_name]
            if not close(observed, expected):
                raise ValueError(f"T5 manuscript value changed: {family}/{round_name}")
            condition = family if round_name == "R0" else f"{family}_{round_name}"
            accuracies = []
            for model, _, _ in MODELS:
                row = summaries[f"comparison|{model}|{condition}"]
                if int(row["n"]) != ITEMS_PER_MODEL or not close(
                        row["accuracy"], int(row["n_matched"]) / int(row["n"])):
                    raise ValueError(f"invalid run summary: {model}/{condition}")
                accuracies.append(float(row["accuracy"]))
            if not close(sum(accuracies) / len(accuracies), expected):
                raise ValueError(
                    f"T5 panel mean is not the ten-model mean: {family}/{round_name}")
    return t5


def panel_ladder_lines(ax, t5: dict) -> None:
    positions = range(len(LADDER_ROUNDS))
    for family in ("A", "C", "X"):
        values = [100 * t5[family][round_name] for round_name in LADDER_ROUNDS]
        ax.plot(
            positions, values, color=COLORS[family], linewidth=1.15,
            marker="o", markersize=3.4, markeredgecolor="black",
            markeredgewidth=0.35, zorder=4,
        )
        end_offset = {"A": (5, 0), "C": (5, 5), "X": (5, -6)}[family]
        ax.annotate(
            family, (len(LADDER_ROUNDS) - 1, values[-1]),
            xytext=end_offset, textcoords="offset points", va="center",
            fontsize=fs(7.2), fontweight="bold", color=COLORS[family],
        )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(LADDER_ROUNDS, fontsize=fs(6.4))
    ax.set_xlim(-0.35, len(LADDER_ROUNDS) - 0.45)
    ax.set_ylim(0, 56)
    ax.set_ylabel("Panel-mean\naccuracy (%)")
    ax.set_title("(c) Reactive ladder", loc="left", fontsize=TITLE_SIZE)
    style_axes(ax)


def panel_ladder_delta(ax, t5: dict) -> None:
    positions = range(len(LADDER_ROUNDS))
    deltas = [
        100 * (t5["X"][round_name] - t5["C"][round_name])
        for round_name in LADDER_ROUNDS
    ]
    for position, delta in zip(positions, deltas):
        ax.bar(
            position, delta, width=0.58,
            color=COLORS["X"] if delta > 0 else COLORS["C"],
            edgecolor="white", linewidth=0.3, zorder=3,
        )
        ax.text(
            position, delta + (0.13 if delta > 0 else -0.13),
            f"{delta:+.2f}", ha="center",
            va="bottom" if delta > 0 else "top",
            fontsize=fs(6.0), color="#3f3f3f",
        )
    ax.axhline(0, color="#777777", linewidth=0.6, zorder=2)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(LADDER_ROUNDS, fontsize=fs(6.4))
    ax.set_xlim(-0.35, len(LADDER_ROUNDS) - 0.45)
    ax.set_ylim(-1.9, 2.4)
    ax.set_ylabel("X $-$ C (pp)")
    ax.text(0.97, 0.92, "X ahead", transform=ax.transAxes, ha="right", va="top",
            fontsize=fs(6.0), color=COLORS["X"], style="italic")
    ax.text(0.03, 0.08, "C ahead", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=fs(6.0), color=COLORS["C"], style="italic")
    style_axes(ax)


def build_compact_figure(primary_path: Path):
    """1x3 compact variant: panels (a)-(b) plus the reactive-ladder panel (c)."""
    global FONT_SCALE, TITLE_SIZE
    FONT_SCALE = 1.30
    TITLE_SIZE = 10.4
    primary = load_json(primary_path)
    t6 = primary["T6_token_economy"]
    validate_t6(t6)
    t5 = validate_t5(primary)
    print(
        f"validated {len(MODELS)} T6 model cells x "
        f"{len(PROACTIVE) + len(REACTIVE)} conditions, all pooled-exact display "
        f"totals, and {len(EXPECTED_T5_PANEL_MEAN) * len(LADDER_ROUNDS)} frozen "
        f"T5 ladder panel means rebuilt from per-model run summaries"
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": fs(7.0),
        "axes.labelsize": fs(7.2),
        "axes.edgecolor": "#8d8d8d",
        "axes.linewidth": 0.55,
        "xtick.labelsize": fs(6.6),
        "ytick.labelsize": fs(6.6),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.20, 2.85))
    grid = fig.add_gridspec(
        2, 3, height_ratios=(1.12, 0.88), hspace=0.52, wspace=0.46,
        width_ratios=(1.12, 1.12, 0.95),
        left=0.105, right=0.965, top=0.895, bottom=0.300,
    )
    ax_proactive = fig.add_subplot(grid[:, 0])
    panel_proactive(ax_proactive, t6)
    ax_proactive.set_title("(a) Proactive economy", loc="left", fontsize=TITLE_SIZE)
    ax_reactive = fig.add_subplot(grid[:, 1])
    panel_reactive(ax_reactive, t6)
    ax_reactive.set_title("(b) Reactive economy", loc="left", fontsize=TITLE_SIZE)
    # The marker-key note appears once; panel (b) keeps it (lower data-free
    # band), panel (a) has no scale-1.3 slot that clears both the headline
    # box and the data.
    for text in list(ax_proactive.texts):
        if text.get_text().startswith("◆"):
            text.remove()
    panel_ladder_lines(fig.add_subplot(grid[0, 2]), t5)
    panel_ladder_delta(fig.add_subplot(grid[1, 2]), t5)
    return fig


def panel_proactive_breakdown(ax, t6: dict) -> None:
    proactive = t6["proactive"]
    a_rows = keyed(proactive["A"]["per_model"])
    c_rows = keyed(proactive["C"]["per_model"])
    x_reference = t6["pooled_exact_display"]["proactive"]["X"]["prompt_avg"]
    y_positions = list(range(len(MODELS) + 1))
    ax.barh(0, x_reference, height=0.62, color="#777777", edgecolor="white", linewidth=0.3, zorder=3)
    ax.text(x_reference * 1.02, 0, f"{x_reference:,.0f}", va="center", fontsize=6.6,
            color="#333333", fontweight="bold")
    for index, (model, _, _) in enumerate(MODELS, start=1):
        base = float(a_rows[model]["prompt_avg_exact_from_items"])
        total = float(c_rows[model]["prompt_avg_exact_from_items"])
        added = total - base
        ax.barh(index, base, height=0.62, color="#d7d7d7", edgecolor="white", linewidth=0.3, zorder=3)
        ax.barh(index, added, left=base, height=0.62, color="#b44747", edgecolor="white", linewidth=0.3, zorder=3)
        ax.text(x_reference * 1.02, index, f"{total:,.0f}  ({100 * total / x_reference:.0f}%)",
                va="center", fontsize=6.5, color="#333333")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(["X pooled"] + [short for _, short, _ in MODELS], fontsize=6.7)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.invert_yaxis()
    ax.axhline(0.5, color="#777777", linewidth=0.45)
    ax.set_xlim(0, x_reference * 1.34)
    ax.set_xlabel("Prompt tokens per R0 item")
    ax.set_title("(c.1) C prompt by model", loc="left", fontsize=8.6)
    ax.legend(
        handles=[Patch(color="#d7d7d7", label="base A prompt"),
                 Patch(color="#b44747", label="added prompt (docs + framing)"),
                 Patch(color="#777777", label="X pooled-exact reference")],
        frameon=False, fontsize=6.2, ncol=2, loc="upper center",
        bbox_to_anchor=(0.5, -0.13), handlelength=0.9, columnspacing=0.7,
    )
    style_axes(ax)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.48)


def panel_reactive_breakdown(ax, t6: dict, route_parts: dict[str, dict[str, float]]) -> None:
    fd_reference = t6["pooled_exact_display"]["reactive"]["C_FD"]["docs_avg"]
    fdrs_rows = keyed(t6["reactive"]["C_FDRS"]["per_model"])
    y_positions = list(range(len(MODELS) + 1))
    ax.barh(0, fd_reference, height=0.62, color="#777777", edgecolor="white", linewidth=0.3, zorder=3)
    ax.text(fd_reference * 1.02, 0, f"{fd_reference:,.0f}", va="center", fontsize=6.6,
            color="#333333", fontweight="bold")
    for index, (model, _, _) in enumerate(MODELS, start=1):
        left = 0.0
        for route in ROUTES:
            width = route_parts[model][route]
            if width:
                ax.barh(index, width, left=left, height=0.62,
                        color=ROUTE_COLORS[route], edgecolor="white", linewidth=0.3, zorder=3)
                left += width
        total = float(fdrs_rows[model]["docs_avg_exact"])
        ax.text(fd_reference * 1.02, index,
                f"{total:,.0f}  ({100 * total / fd_reference:.0f}%)",
                va="center", fontsize=6.5, color="#333333")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(["FD pooled"] + [short for _, short, _ in MODELS], fontsize=6.7)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.invert_yaxis()
    ax.axhline(0.5, color="#777777", linewidth=0.45)
    ax.set_xlim(0, fd_reference * 1.37)
    ax.set_xlabel("Documentation tokens per fix attempt")
    ax.set_title("(c.2) FDRS docs by routed destination", loc="left", fontsize=8.6)
    ax.text(0.99, 0.02, "basic-fix route contributes 0 doc tokens",
            transform=ax.transAxes, ha="right", fontsize=6.1, color="#555555")
    handles = [Patch(color=ROUTE_COLORS[route], label=ROUTE_LABELS[route]) for route in ROUTES]
    handles.append(Patch(color="#777777", label="FD pooled-exact reference"))
    ax.legend(
        handles=handles, frameon=False, fontsize=6.1, ncol=2,
        loc="upper center", bbox_to_anchor=(0.5, -0.13),
        handlelength=0.85, columnspacing=0.55,
    )
    style_axes(ax)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.48)


def build_figure(primary_path: Path, router_path: Path):
    primary = load_json(primary_path)
    router = load_json(router_path)
    t6 = primary["T6_token_economy"]
    validate_t6(t6)
    route_parts = validate_router(router, t6)
    events = sum(
        len(router["models"][label]["conditions"]["C+FDRS"]["events"])
        for _, _, label in MODELS
    )
    print(
        f"validated {len(MODELS)} T6 model cells x "
        f"{len(PROACTIVE) + len(REACTIVE)} conditions, all pooled-exact display "
        f"totals, and {events:,} C+FDRS router events rebuilding "
        f"{len(MODELS) * (len(ROUTES) + 1)} route summaries"
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.labelsize": 7.2,
        "axes.edgecolor": "#8d8d8d",
        "axes.linewidth": 0.55,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.20, 6.15))
    grid = fig.add_gridspec(
        2, 2, height_ratios=(0.82, 1.25), hspace=0.42, wspace=0.30,
        left=0.105, right=0.975, top=0.965, bottom=0.105,
    )
    panel_proactive(fig.add_subplot(grid[0, 0]), t6)
    panel_reactive(fig.add_subplot(grid[0, 1]), t6)
    panel_proactive_breakdown(fig.add_subplot(grid[1, 0]), t6)
    panel_reactive_breakdown(fig.add_subplot(grid[1, 1]), t6, route_parts)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--router", type=Path, default=DEFAULT_ROUTER)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--png", nargs="?", const=str(DEFAULT_PNG))
    parser.add_argument(
        "--compact", action="store_true",
        help="write the 1x3 compact variant (panels a-b plus the T5 ladder) "
             "to assets/fig_token_economy_compact.pdf instead")
    args = parser.parse_args()

    if args.compact:
        figure = build_compact_figure(args.primary.resolve())
    else:
        figure = build_figure(args.primary.resolve(), args.router.resolve())
    default_output = DEFAULT_COMPACT if args.compact else DEFAULT_OUTPUT
    output = (args.out or default_output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fixed_date = datetime(2026, 8, 8, 0, 0, 0, tzinfo=timezone.utc)
    metadata = {
        "Title": "PowerCodeBench token-economy Pareto renderings",
        "Author": "PowerCodeBench",
        "Creator": "code/build/plot_token_economy.py",
        "CreationDate": fixed_date,
        "ModDate": fixed_date,
    }
    figure.savefig(output, format="pdf", dpi=300, metadata=metadata)
    print(f"wrote pooled-exact token-economy figure -> {output}")
    if args.png is not None:
        png = Path(args.png).resolve()
        png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(png, format="png", dpi=220,
                       metadata={"Software": "PowerCodeBench plot_token_economy.py"})
        print(f"wrote PNG preview -> {png}")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
