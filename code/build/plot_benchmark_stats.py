# --------------------------------------------------------------------------
# Regenerates the frozen-release composition figure from the release itself:
# reads benchmark.json (repository root), cross-checks every count against
# benchmark/composition_stats.json, measures network sizes by instantiating
# every release network through the release's own registry and loader
# (code/benchmark_generator), and writes assets/benchmark_stats.png plus
# benchmark/fig_benchmark_stats.pdf.
#
# Intentionally strict: any count that disagrees with the frozen
# composition statistics, or any release network that cannot be
# instantiated, aborts the figure.
#
# Usage (from the repository root, pinned environment + matplotlib):
#     python3 code/build/plot_benchmark_stats.py
# --------------------------------------------------------------------------
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "benchmark_generator"))

from benchmark_engine import _load_network  # noqa: E402  (release loader)


def main():
    items = json.loads((ROOT / "benchmark.json").read_text(encoding="utf-8"))
    stats = json.loads(
        (ROOT / "benchmark" / "composition_stats.json").read_text(encoding="utf-8"))

    diff_count = Counter(it["difficulty_level"] for it in items)
    task_count = Counter(it["scenario"]["task"] for it in items)
    net_count = Counter(it["scenario"]["network"] for it in items)

    # -- hard cross-check against the frozen composition statistics --------
    if len(items) != stats["meta"]["n_items"]:
        raise RuntimeError(f"item count {len(items)} != frozen {stats['meta']['n_items']}")
    if dict(diff_count) != stats["per_difficulty"]:
        raise RuntimeError("per-difficulty counts disagree with composition_stats.json")
    if dict(task_count) != stats["per_task_family"]:
        raise RuntimeError("per-task-family counts disagree with composition_stats.json")
    frozen_nets = {k: v["items"] for k, v in stats["per_network"].items()}
    if dict(net_count) != frozen_nets:
        raise RuntimeError("per-network counts disagree with composition_stats.json")
    if len(net_count) != stats["network_pool"]["instantiated_in_release"]:
        raise RuntimeError("network count disagrees with the frozen network pool")

    # -- network sizes via the release's own registry and loader ----------
    network_buses = {}
    for net in sorted(net_count):
        network = _load_network(net)
        n_bus = len(network.bus)
        if n_bus <= 0:
            raise RuntimeError(f"network {net!r} instantiated with {n_bus} buses")
        network_buses[net] = n_bus

    # Let Matplotlib reserve inter-panel space for panel (b)'s long task
    # labels. A fixed wspace allowed those labels to intrude into panel (a).
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20.5, 5.5),
        layout="constrained",
        gridspec_kw={"width_ratios": [0.9, 1.55, 1.0]},
    )

    # (a) difficulty
    ax = axes[0]
    diffs = sorted(diff_count)
    counts = [diff_count[d] for d in diffs]
    bars = ax.bar(range(len(diffs)), counts,
                  color=plt.cm.viridis(np.linspace(0.2, 0.85, len(diffs))))
    ax.set_xticks(range(len(diffs)))
    ax.set_xticklabels([d.split("_", 1)[0] for d in diffs], fontsize=13)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                str(c), ha="center", va="bottom", fontsize=12)
    ax.set_title("(a) Items per difficulty level", fontsize=14)
    ax.set_ylabel("# items", fontsize=12)
    ax.set_ylim(0, max(counts) * 1.12)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, axis="y")

    # (b) task families
    ax = axes[1]
    sorted_tasks = task_count.most_common()
    names = [t for t, _ in sorted_tasks]
    cnts = [c for _, c in sorted_tasks]
    ypos = np.arange(len(names))
    ax.barh(ypos, cnts, color="#4c78a8")
    for y, c in zip(ypos, cnts):
        ax.text(c + 2, y, str(c), va="center", fontsize=11)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_title("(b) Task-type distribution", fontsize=14)
    ax.set_xlabel("# items", fontsize=12)
    ax.set_xlim(0, max(cnts) * 1.12)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, axis="x")

    # (c) network bus-count histogram (log-x)
    ax = axes[2]
    bs = list(network_buses.values())
    bins = np.geomspace(min(bs), max(bs), 12)
    hist_count, _, _ = ax.hist(bs, bins=bins, color="#54a24b",
                               edgecolor="black", linewidth=0.7)
    if int(hist_count.sum()) != len(network_buses):
        raise RuntimeError("histogram does not include every release network")
    ax.set_xscale("log")
    ax.set_xlabel("Network bus count (log scale)", fontsize=12)
    ax.set_ylabel("# networks", fontsize=12)
    ax.set_title(f"(c) Network sizes ({len(network_buses)} networks)", fontsize=14)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)

    for out in (ROOT / "assets" / "benchmark_stats.png",
                ROOT / "benchmark" / "fig_benchmark_stats.pdf"):
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("wrote", out.relative_to(ROOT))
    print(f"panels: (a) {counts}  (b) {len(names)} families  "
          f"(c) {len(network_buses)} networks, buses {min(bs)}..{max(bs)}")


if __name__ == "__main__":
    main()
