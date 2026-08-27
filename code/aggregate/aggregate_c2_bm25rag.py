#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Repository copy of scripts/aggregate/aggregate_c2_bm25rag.py from the frozen
# experimental pipeline. Its inputs are raw per-item run trees too large to
# ship here, so it does not run from this checkout; see ARTIFACT_INDEX.md for
# the outputs it produced.
# --------------------------------------------------------------------------
"""Aggregate C2 BM25-RAG vs proposed-pipeline results.

Reads the R0 accuracy of each arm from ``probe_eval_results/`` and compares:
  R  (BM25-RAG over raw pandapower_docs.json, no boundary cards / anchors)
  C  (proposed: demand x risk gating + library_knowledge contracts)
  A  (no proactive)
on the three representative open-weight models. R0 only.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NEW_R_DIR = ROOT / "probe_eval_results/comparison_bm25rag"
EXISTING_DIR = ROOT / "probe_eval_results/comparison"

MODELS = [
    "Qwen_Qwen2.5-Coder-32B-Instruct",
    "meta-llama_Llama-3.1-70B-Instruct",
    "Qwen_Qwen3-Coder-480B-A35B-Instruct",
]


def _load_acc(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception:
        return None
    summary = data.get("summary") or data
    for key in ("accuracy", "acc", "matched_rate"):
        v = summary.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    matched = summary.get("matched") or summary.get("n_matched")
    n = summary.get("n_items") or summary.get("total")
    if matched is not None and n:
        return matched / n
    return None


def main() -> int:
    print(f"{'Model':<40} {'A (R0)':>8} {'R (R0)':>8} {'C (R0)':>8} {'C-R':>9} {'C-A':>9}")
    print("-" * 88)
    for m in MODELS:
        a = _load_acc(EXISTING_DIR / m / "benchmark_results_condA.json")
        c = _load_acc(EXISTING_DIR / m / "benchmark_results_condC.json")
        r = _load_acc(NEW_R_DIR / m / "benchmark_results_condR.json")
        a_s = f"{a*100:.2f}" if a is not None else "—"
        r_s = f"{r*100:.2f}" if r is not None else "—"
        c_s = f"{c*100:.2f}" if c is not None else "—"
        if c is not None and r is not None:
            cr = (c - r) * 100
            cr_s = f"{cr:+.2f}pp"
        else:
            cr_s = "—"
        if c is not None and a is not None:
            ca = (c - a) * 100
            ca_s = f"{ca:+.2f}pp"
        else:
            ca_s = "—"
        print(f"{m:<40} {a_s:>8} {r_s:>8} {c_s:>8} {cr_s:>9} {ca_s:>9}")

    print()
    print("=== Interpretation ===")
    print("C-R quantifies the gap between vanilla BM25 RAG over raw API docs")
    print("and the proposed pipeline (demand-modelling + risk gating + derived")
    print("contracts). A large positive C-R supports the claim that vanilla")
    print("retrieval is insufficient and motivates the demand+risk machinery.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
