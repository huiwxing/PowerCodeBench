#!/usr/bin/env python3
"""Render the failure-taxonomy stacked bars from the frozen human-review ledger.

Inputs are the two independent 100-item reviewer ledgers and their agreement
summary under ``audit/failure_taxonomy/human_review/``.  The figure shows the
consensus label distribution over the stratified sample as two stacked-bar
panels: (a) by model tier and (b) by injection condition (A-family fix ladder
vs the full method C+FDRS).  The three unadjudicated reviewer disagreements
are drawn as their own hatched segment.

Before drawing, the ledger is validated: every ``sample_id`` must parse
into the frozen 20-cell (4 tiers x 5 conditions x 5 items) design, both
reviewers' marginals, the disagreement set, Cohen's kappa, and every plotted
consensus cross-tab must be rebuilt from the records and agree with the
EXPECTED constants below and with the frozen agreement summary.  Any mismatch
raises instead of producing a figure.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEW_DIR = ROOT / "audit/failure_taxonomy/human_review"
DEFAULT_OUTPUT = ROOT / "assets/fig_failure_taxonomy.pdf"

SAMPLE_ID = re.compile(r"^tier([1-4])_([AC])_(FX|FD|FDR|FDRS)_(\d{2})$")
TIERS = ("tier1", "tier2", "tier3", "tier4")
CONDITIONS = ("condA_FX", "condA_FD", "condA_FDR", "condA_FDRS", "condC_FDRS")
CLASSES = (
    "C1_api_contract",
    "C2_param_misuse",
    "C3_workflow_logic",
    "C4_numerical_extraction",
    "C5_format_env",
    "ambiguous_item_defect",
    "unresolved",
)

EXPECTED_N = 100
EXPECTED_AGREEMENT_PCT = 97.0
EXPECTED_KAPPA = 0.955
# sample_id -> (reviewer A label, reviewer B label); reported, not adjudicated.
EXPECTED_DISAGREEMENTS = {
    "tier1_C_FDRS_05": ("C2_param_misuse", "C1_api_contract"),
    "tier2_C_FDRS_03": ("C4_numerical_extraction", "C5_format_env"),
    "tier4_C_FDRS_01": ("C2_param_misuse", "ambiguous_item_defect"),
}
EXPECTED_MARGINALS = {
    "A": {"C1_api_contract": 47, "C2_param_misuse": 13, "C3_workflow_logic": 28,
          "C4_numerical_extraction": 2, "C5_format_env": 9,
          "ambiguous_item_defect": 1},
    "B": {"C1_api_contract": 48, "C2_param_misuse": 11, "C3_workflow_logic": 28,
          "C4_numerical_extraction": 1, "C5_format_env": 10,
          "ambiguous_item_defect": 2},
}
# Consensus cross-tabs (agreed labels; the three disagreements as unresolved).
EXPECTED_TIER_TABLE = {
    "tier1": {"C1_api_contract": 15, "C2_param_misuse": 1, "C3_workflow_logic": 3,
              "C5_format_env": 5, "unresolved": 1},
    "tier2": {"C1_api_contract": 11, "C2_param_misuse": 3, "C3_workflow_logic": 9,
              "C5_format_env": 1, "unresolved": 1},
    "tier3": {"C1_api_contract": 13, "C2_param_misuse": 5, "C3_workflow_logic": 5,
              "C5_format_env": 1, "ambiguous_item_defect": 1},
    "tier4": {"C1_api_contract": 8, "C2_param_misuse": 2, "C3_workflow_logic": 11,
              "C4_numerical_extraction": 1, "C5_format_env": 2, "unresolved": 1},
}
EXPECTED_CONDITION_TABLE = {
    "condA_FX": {"C1_api_contract": 11, "C2_param_misuse": 2,
                 "C3_workflow_logic": 4, "C5_format_env": 3},
    "condA_FD": {"C1_api_contract": 16, "C2_param_misuse": 2,
                 "C3_workflow_logic": 2},
    "condA_FDR": {"C1_api_contract": 7, "C2_param_misuse": 1,
                  "C3_workflow_logic": 9, "C5_format_env": 2,
                  "ambiguous_item_defect": 1},
    "condA_FDRS": {"C1_api_contract": 10, "C2_param_misuse": 2,
                   "C3_workflow_logic": 5, "C5_format_env": 3},
    "condC_FDRS": {"C1_api_contract": 3, "C2_param_misuse": 4,
                   "C3_workflow_logic": 8, "C4_numerical_extraction": 1,
                   "C5_format_env": 1, "unresolved": 3},
}

# C1/C3 reuse the manuscript-wide blue/red; C2/C4/C5 are desaturated toward
# the restrained grey/blue/red system the other figures share.
COLORS = {
    "C1_api_contract": "#326ea8",
    "C2_param_misuse": "#cf9a52",
    "C3_workflow_logic": "#b64b4b",
    "C4_numerical_extraction": "#4f8f89",
    "C5_format_env": "#8d7cb0",
    "ambiguous_item_defect": "#c9c9c9",
    "unresolved": "white",
}
LABELS = {
    "C1_api_contract": "C1 API contract",
    "C2_param_misuse": "C2 parameter misuse",
    "C3_workflow_logic": "C3 workflow logic",
    "C4_numerical_extraction": "C4 numerical extraction",
    "C5_format_env": "C5 format / environment",
    "ambiguous_item_defect": "ambiguous item defect",
    "unresolved": "reviewer disagreement",
}
DARK_TEXT = {"ambiguous_item_defect", "unresolved"}
TIER_DISPLAY = {"tier1": "Tier 1", "tier2": "Tier 2", "tier3": "Tier 3", "tier4": "Tier 4"}
CONDITION_DISPLAY = {
    "condA_FX": "A+FX", "condA_FD": "A+FD", "condA_FDR": "A+FDR",
    "condA_FDRS": "A+FDRS", "condC_FDRS": "C+FDRS",
}


def load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def empty_table(keys: tuple[str, ...]) -> dict[str, dict[str, int]]:
    return {key: {cls: 0 for cls in CLASSES} for key in keys}


def dense(table: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {key: {cls: int(row.get(cls, 0)) for cls in CLASSES} for key, row in table.items()}


def validate(review_dir: Path) -> tuple[dict, dict]:
    """Raise unless the ledger rebuilds every EXPECTED constant."""
    labels = {}
    for reviewer in ("A", "B"):
        records = load_json(review_dir / f"reviewer_{reviewer}.json")["records"]
        if len(records) != EXPECTED_N:
            raise ValueError(f"reviewer {reviewer} ledger is not {EXPECTED_N} records")
        labels[reviewer] = {row["sample_id"]: row["label"] for row in records}
    if set(labels["A"]) != set(labels["B"]) or len(labels["A"]) != EXPECTED_N:
        raise ValueError("reviewer ledgers do not cover the same sample")

    cells: dict[tuple[str, str], int] = {}
    tier_table = empty_table(TIERS)
    condition_table = empty_table(CONDITIONS)
    disagreements = {}
    for sample_id in labels["A"]:
        match = SAMPLE_ID.match(sample_id)
        if not match:
            raise ValueError(f"sample_id outside the frozen design: {sample_id}")
        tier = f"tier{match.group(1)}"
        condition = f"cond{match.group(2)}_{match.group(3)}"
        cells[(tier, condition)] = cells.get((tier, condition), 0) + 1
        label_a, label_b = labels["A"][sample_id], labels["B"][sample_id]
        if label_a == label_b:
            consensus = label_a
        else:
            consensus = "unresolved"
            disagreements[sample_id] = (label_a, label_b)
        if consensus not in CLASSES:
            raise ValueError(f"unknown label for {sample_id}: {consensus}")
        tier_table[tier][consensus] += 1
        condition_table[condition][consensus] += 1
    if cells != {(t, c): 5 for t in TIERS for c in CONDITIONS}:
        raise ValueError("sample is not the frozen 4x5x5 stratified design")
    if disagreements != EXPECTED_DISAGREEMENTS:
        raise ValueError(f"disagreement set changed: {sorted(disagreements)}")
    for reviewer in ("A", "B"):
        marginal: dict[str, int] = {}
        for label in labels[reviewer].values():
            marginal[label] = marginal.get(label, 0) + 1
        if marginal != EXPECTED_MARGINALS[reviewer]:
            raise ValueError(f"reviewer {reviewer} marginal changed: {marginal}")
    if dense(tier_table) != dense(EXPECTED_TIER_TABLE):
        raise ValueError("consensus tier cross-tab changed")
    if dense(condition_table) != dense(EXPECTED_CONDITION_TABLE):
        raise ValueError("consensus condition cross-tab changed")

    agreed = EXPECTED_N - len(disagreements)
    observed_agreement = agreed / EXPECTED_N
    if not math.isclose(100 * observed_agreement, EXPECTED_AGREEMENT_PCT, abs_tol=1e-9):
        raise ValueError("raw agreement changed")
    chance = sum(
        (EXPECTED_MARGINALS["A"].get(cls, 0) / EXPECTED_N)
        * (EXPECTED_MARGINALS["B"].get(cls, 0) / EXPECTED_N)
        for cls in CLASSES
    )
    kappa = (observed_agreement - chance) / (1 - chance)
    if round(kappa, 3) != EXPECTED_KAPPA:
        raise ValueError(f"recomputed Cohen's kappa changed: {kappa:.6f}")

    summary = load_json(review_dir / "agreement.json")
    frozen_checks = (
        (summary["n_items"], EXPECTED_N, "n_items"),
        (summary["agreement_pct"], EXPECTED_AGREEMENT_PCT, "agreement_pct"),
        (summary["cohens_kappa"], EXPECTED_KAPPA, "cohens_kappa"),
        (summary["n_disagreements"], len(EXPECTED_DISAGREEMENTS), "n_disagreements"),
        (summary["per_reviewer_marginals"],
         {r: {k: v for k, v in m.items()} for r, m in EXPECTED_MARGINALS.items()},
         "per_reviewer_marginals"),
    )
    for observed, expected, name in frozen_checks:
        if observed != expected:
            raise ValueError(f"agreement summary mismatch: {name}")
    return dense(tier_table), dense(condition_table)


def draw_panel(ax, table, keys, display, title, separator_before=None) -> None:
    positions = {key: index for index, key in enumerate(keys)}
    if separator_before is not None:
        for key in keys[keys.index(separator_before):]:
            positions[key] += 0.62
    for key in keys:
        bottom = 0
        for cls in CLASSES:
            count = table[key][cls]
            if not count:
                continue
            ax.bar(
                positions[key], count, width=0.62, bottom=bottom,
                color=COLORS[cls], edgecolor="#777777" if cls == "unresolved" else "white",
                linewidth=0.5 if cls == "unresolved" else 0.35,
                hatch="///" if cls == "unresolved" else None, zorder=3,
            )
            if count >= 2:
                ax.text(
                    positions[key], bottom + count / 2, str(count),
                    ha="center", va="center", fontsize=6.8,
                    color="#333333" if cls in DARK_TEXT else "white",
                )
            bottom += count
    if separator_before is not None:
        boundary = positions[separator_before] - 0.81
        ax.axvline(boundary, color="#8d8d8d", linewidth=0.6, linestyle=":")
        ax.text((min(positions.values()) + boundary) / 2, 27.4, "A-family fix ladder",
                ha="center", fontsize=7.2, style="italic", color="#555555")
        ax.text(positions[separator_before], 27.4, "full\nmethod",
                ha="center", fontsize=7.2, style="italic", color="#555555")
    ax.set_xticks([positions[key] for key in keys])
    ax.set_xticklabels([display[key] for key in keys], fontsize=7.2)
    ax.set_ylim(0, 30)
    ax.set_ylabel("Sampled failures (items)")
    ax.set_title(title, loc="left", fontsize=9.0)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.48, zorder=0)
    ax.tick_params(axis="both", length=2.4, width=0.55)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def build_figure(tier_table: dict, condition_table: dict):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.labelsize": 7.2,
        "axes.edgecolor": "#8d8d8d",
        "axes.linewidth": 0.55,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "hatch.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    # 5.40 x 2.30 in = the manuscript \linewidth 1:1, so every source font
    # size is the typeset size; panel (b) is wider (5 bars + separator gap).
    fig, axes = plt.subplots(
        1, 2, figsize=(5.40, 2.30),
        gridspec_kw={"width_ratios": (1.0, 1.28)},
    )
    fig.subplots_adjust(left=0.088, right=0.99, top=0.845, bottom=0.315, wspace=0.28)
    draw_panel(axes[0], tier_table, TIERS, TIER_DISPLAY,
               "(a) Consensus class by model tier\n(25 items each)")
    draw_panel(axes[1], condition_table, CONDITIONS, CONDITION_DISPLAY,
               "(b) Consensus class by condition\n(20 items each)",
               separator_before="condC_FDRS")
    handles = [
        Patch(facecolor=COLORS[cls],
              edgecolor="#777777" if cls == "unresolved" else "white",
              hatch="///" if cls == "unresolved" else None,
              linewidth=0.5, label=LABELS[cls])
        for cls in CLASSES
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=4, frameon=False, fontsize=6.8, handlelength=1.15,
        handletextpad=0.4, columnspacing=0.8,
    )
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tier_table, condition_table = validate(args.review_dir.resolve())
    print(
        f"validated {EXPECTED_N} double-review records against the frozen "
        f"4x5x5 design, both reviewer marginals, kappa {EXPECTED_KAPPA}, "
        f"{len(EXPECTED_DISAGREEMENTS)} reported disagreements, and both "
        f"consensus cross-tabs"
    )
    figure = build_figure(tier_table, condition_table)
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fixed_date = datetime(2026, 8, 11, tzinfo=timezone.utc)
    figure.savefig(output, format="pdf", dpi=300, metadata={
        "Title": "PowerCodeBench failure-taxonomy consensus stacked bars",
        "Author": "PowerCodeBench",
        "Creator": "code/build/plot_failure_taxonomy.py",
        "CreationDate": fixed_date,
        "ModDate": fixed_date,
    })
    print(f"wrote failure-taxonomy figure -> {output}")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
