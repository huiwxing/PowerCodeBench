#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Builds the four-panel robustness figure of the Supplementary Material
# (risk-weight sensitivity, generation-seed variance, scenario adaptation,
# dense retrieval at C-matched cost) directly from the frozen aggregates in
# this repository.  It replaces four SM tables with one figure; the per-cell
# values stay here, in the JSONs this script reads.
#
# Every panel cross-checks its inputs against the values quoted in the SM
# caption and raises on any disagreement, so the figure stays tied to the
# frozen artifacts.
#
# Usage:  python3 code/build/plot_robustness_panel.py [--out PATH.pdf]
# Needs:  matplotlib (no pandapower, no GPU, no network).
# --------------------------------------------------------------------------
"""Four-panel robustness figure: SM 'Robustness and variance checks'."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "results" / "aggregates"

# -- Palette ---------------------------------------------------------------
# Categorical slots 1-3 of the validated reference palette (light mode); these
# three validate all-pairs (worst CVD dE 9.2, normal-vision 24.0).  Every mark
# also carries a printed value, which is what the low-contrast relief rule
# asks for.
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
BEFORE = "#a8c8ee"          # same hue as S1, lighter step (dumbbell "before")
INK, INK2 = "#0b0b0b", "#52514e"
MUTED, BAND = "#b8b7b0", "#ececea"

# Panel order, smallest model at the bottom.  Shared by all four panels so the
# rows line up across the figure.
MODELS = [
    ("Qwen_Qwen2.5-Coder-1.5B-Instruct", "Qwen2.5-1.5B"),
    ("Qwen_Qwen2.5-Coder-7B-Instruct", "Qwen2.5-7B"),
    ("meta-llama_Llama-3.1-8B-Instruct", "Llama-8B"),
    ("Qwen_Qwen2.5-Coder-14B-Instruct", "Qwen2.5-14B"),
    ("Qwen_Qwen2.5-Coder-32B-Instruct", "Qwen2.5-32B"),
    ("meta-llama_Llama-3.1-70B-Instruct", "Llama-70B"),
    ("openai_gpt-oss-120b", "GPT-OSS-120B"),
    ("Qwen_Qwen3-Coder-Next", "Qwen3-Next"),
    ("meta-llama_Llama-3.1-405B-Instruct", "Llama-405B"),
    ("Qwen_Qwen3-Coder-480B-A35B-Instruct", "Qwen3-480B"),
]
KEYS = [k for k, _ in MODELS]
LABELS = [lab for _, lab in MODELS]
YPOS = {k: i for i, k in enumerate(KEYS)}

# Values quoted in the SM caption / main text; the figure must reproduce them.
EXPECT_MAX_ABS_PP = {           # panel (a)
    "Qwen_Qwen2.5-Coder-14B-Instruct": 0.85,
    "Qwen_Qwen2.5-Coder-32B-Instruct": 0.85,
    "meta-llama_Llama-3.1-70B-Instruct": 0.90,
    "openai_gpt-oss-120b": 0.95,
    "Qwen_Qwen3-Coder-480B-A35B-Instruct": 1.30,
}
EXPECT_SEED_SD = {"median": 0.20, "max": 0.85}          # panel (b)
EXPECT_ADAPT_PANEL_PP = 3.16                            # panel (c)
EXPECT_RSEMB_WORSE = 7                                  # panel (d)


def load(rel: str) -> dict:
    return json.loads((AGG / rel).read_text(encoding="utf-8"))


def close(a: float, b: float, tol: float = 5e-3) -> bool:
    return abs(a - b) <= tol


# --------------------------------------------------------------------------
def panel_a(ax) -> None:
    """Risk-weight sensitivity: accuracy shift away from the deployed weights."""
    grid = load("a53_weight_grid_summary.json")["models"]
    arms = [("uniform", S1, "uniform"),
            ("l3heavy", S2, "L3-heavy"),
            ("l0heavy", S3, "L0-heavy")]

    ax.axvspan(-1.3, 1.3, color=BAND, zorder=0)
    ax.axvline(0, color=MUTED, lw=1.0, zorder=1)

    for key, md in grid.items():
        acc, y = md["acc_pct"], YPOS[key]
        got = max(abs(acc[a] - acc[b]) for a in acc for b in acc)
        exp = EXPECT_MAX_ABS_PP[key]
        if not close(got, exp) or not close(md["max_pairwise_abs_pp"], exp):
            raise RuntimeError(f"(a) max|delta| {key}: {got:.2f} != {exp}")
        for arm, colour, _ in arms:
            ax.plot(acc[arm] - acc["deployed"], y, "o", ms=5.5,
                    color=colour, mec="white", mew=0.7, zorder=3)
        ax.text(1.62, y, f"{exp:.2f}", va="center", ha="left",
                fontsize=6.4, color=INK2)

    ax.text(1.62, len(KEYS) - 0.25, "max|$\\Delta$|", va="center", ha="left",
            fontsize=6.4, color=INK2)
    ax.set_xlim(-1.75, 2.35)
    ax.set_xticks([-1.3, 0, 1.3])
    ax.set_xticklabels(["$-$1.3", "0\n(deployed)", "+1.3"])
    ax.set_xlabel("R0 accuracy shift vs deployed weights (pp)", fontsize=7.5)
    ax.set_title("(a)  L0\u2013L3 risk-weight grid", fontsize=8.5, loc="left",
                 color=INK)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=5, color=c,
                              label=lab) for _, c, lab in arms],
              fontsize=6.4, loc="upper right", bbox_to_anchor=(1.0, 0.085),
              frameon=False, handletextpad=0.2, borderpad=0.2,
              labelspacing=0.2, ncol=3, columnspacing=0.9)


def panel_b(ax) -> None:
    """Generation-seed variance: spread of C+FDRS R3 across 10 vLLM seeds."""
    per = load("action7_multiseed_summary.json")["per_model"]
    by_dir = {v["model_dir"]: v for v in per.values()}
    sds = []
    for key in KEYS:
        row, y = by_dir[key], YPOS[key]
        accs = list(row["per_seed_acc_pct"].values())
        if len(accs) != 10:
            raise RuntimeError(f"(b) {key}: {len(accs)} seeds, expected 10")
        mean = sum(accs) / 10
        sd = (sum((a - mean) ** 2 for a in accs) / 9) ** 0.5
        sds.append(sd)
        lo, hi = min(accs), max(accs)
        ax.plot([0, hi - lo], [y, y], "-", color=BEFORE, lw=2.6,
                solid_capstyle="round", zorder=2)
        ax.plot(hi - lo, y, "o", ms=5.0, color=S1, mec="white", mew=0.7,
                zorder=3)
        ax.text(3.1, y, f"$\\pm${sd:.2f}", va="center", ha="left",
                fontsize=6.4, color=INK2)

    srt = sorted(sds)
    med = (srt[4] + srt[5]) / 2
    if not close(round(med, 2), EXPECT_SEED_SD["median"]) or \
       not close(round(max(sds), 2), EXPECT_SEED_SD["max"]):
        raise RuntimeError(f"(b) sd median/max {med:.2f}/{max(sds):.2f} "
                           f"!= {EXPECT_SEED_SD}")

    ax.text(3.1, len(KEYS) - 0.25, "sd", va="center", ha="left",
            fontsize=6.4, color=INK2)
    ax.set_xlim(-0.12, 4.5)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("C+FDRS R3 spread across 10 seeds (pp)", fontsize=7.5)
    ax.set_title("(b)  Generation-seed variance", fontsize=8.5, loc="left",
                 color=INK)


def _dumbbell(ax, y, x0, x1, lab_x, lab, c0=BEFORE, c1=S1) -> None:
    ax.plot([x0, x1], [y, y], "-", color=MUTED, lw=1.1, zorder=2)
    ax.plot(x0, y, "o", ms=5.0, color=c0, mec="white", mew=0.7, zorder=3)
    ax.plot(x1, y, "o", ms=5.0, color=c1, mec="white", mew=0.7, zorder=4)
    ax.text(lab_x, y, lab, va="center", ha="left", fontsize=6.4, color=INK2)


def panel_c(ax) -> None:
    """Scenario adaptation: C without vs with role-frequency reweighting."""
    rows = load("c_unadapted_summary.json")["rows"]
    tot = 0.0
    for r in rows:
        y = YPOS[r["model"]]
        un, ad = 100 * r["C_unadapted"], 100 * r["C_adapted"]
        d = ad - un
        if not close(d, r["delta_adapted_minus_unadapted"], 1e-6):
            raise RuntimeError(f"(c) delta mismatch {r['model']}")
        tot += d
        _dumbbell(ax, y, un, ad, 62.5, f"{d:+.2f}")
    if not close(round(tot / len(rows), 2), EXPECT_ADAPT_PANEL_PP):
        raise RuntimeError(f"(c) panel mean {tot/len(rows):.2f} "
                           f"!= {EXPECT_ADAPT_PANEL_PP}")

    ax.text(62.5, len(KEYS) - 0.25, "$\\Delta$", va="center", ha="left",
            fontsize=6.4, color=INK2)
    ax.set_xlim(-2, 76)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("Round-0 accuracy (%)", fontsize=7.5)
    ax.set_title("(c)  Scenario adaptation of the demand model", fontsize=8.5,
                 loc="left", color=INK)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=5, color=BEFORE,
               label="C, unadapted"),
        Line2D([], [], marker="o", ls="", ms=5, color=S1, label="C, adapted")],
        fontsize=6.4, loc="upper right", bbox_to_anchor=(1.0, 0.085),
        frameon=False, handletextpad=0.2, borderpad=0.2, labelspacing=0.2,
        ncol=3, columnspacing=0.9)


def panel_d(ax) -> None:
    """Dense vanilla retrieval: nominal budget vs C-matched measured cost."""
    rsem = {m["model"]: m for m in
            load("comparison_e5_semantic_rag/proactive_comparison.json")["models"]}
    cmain = {m["model"]: m for m in
             load("comparison/proactive_comparison.json")["models"]}
    worse = 0
    for key in KEYS:
        y = YPOS[key]
        a, b = 100 * rsem[key]["Rsem"], 100 * rsem[key]["RsemB"]
        c = 100 * cmain[key]["C"]
        if b < a:
            worse += 1
        _dumbbell(ax, y, a, b, 62.5, f"{c - b:+.1f}")
        ax.plot(c, y, "D", ms=4.2, color=INK2, mec="white", mew=0.6, zorder=5)
    if worse != EXPECT_RSEMB_WORSE:
        raise RuntimeError(f"(d) {worse} models drop, expected "
                           f"{EXPECT_RSEMB_WORSE}")

    ax.text(62.5, len(KEYS) - 0.25, "C$-$R$_{\\rm semB}$", va="center",
            ha="left", fontsize=6.4, color=INK2)
    ax.set_xlim(-2, 76)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("Round-0 accuracy (%)", fontsize=7.5)
    ax.set_title("(d)  Dense retrieval at C-matched cost", fontsize=8.5,
                 loc="left", color=INK)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=5, color=BEFORE,
               label="R$_{\\rm sem}$ (nominal)"),
        Line2D([], [], marker="o", ls="", ms=5, color=S1,
               label="R$_{\\rm semB}$ (matched)"),
        Line2D([], [], marker="D", ls="", ms=4, color=INK2, label="C")],
        fontsize=6.4, loc="upper right", bbox_to_anchor=(1.0, 0.085),
        frameon=False, handletextpad=0.2, borderpad=0.2, labelspacing=0.2,
        ncol=3, columnspacing=0.9)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "assets" / "robustness_panel.pdf"))
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "text.usetex": False,
        "mathtext.fontset": "dejavuserif",
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "xtick.major.size": 2.5, "ytick.major.size": 0,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.4))
    for ax, fn in zip(axes.ravel(), (panel_a, panel_b, panel_c, panel_d)):
        ax.set_ylim(-2.1, len(KEYS) - 0.2)
        ax.set_yticks(range(len(KEYS)))
        ax.set_yticklabels(LABELS)
        ax.grid(axis="x", color=BAND, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        fn(ax)

    fig.tight_layout(pad=0.9, w_pad=1.6, h_pad=1.8)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}  (all panel cross-checks passed)")


if __name__ == "__main__":
    main()
