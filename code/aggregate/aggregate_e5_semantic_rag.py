#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Repository copy of scripts/aggregate/aggregate_e5_semantic_rag.py from the
# frozen experimental pipeline. Its inputs are raw per-item run trees too
# large to ship here, so it does not run from this checkout; see
# ARTIFACT_INDEX.md for the outputs it produced.
# --------------------------------------------------------------------------
"""Aggregate E5 dense semantic-RAG (Rsem / RsemB) against the existing panel.

Compares, per open-weight model (R0 only):
  A     (no proactive)
  R     (BM25 lexical RAG over raw pandapower_docs.json)
  Rsem  (SBERT dense RAG, nominal budget 2000 == R; token cost reported as-is)
  RsemB (SBERT dense RAG, tight budget B* calibrated to C's measured cost)
  C     (proposed: demand x risk gating + library_knowledge contracts)

Two views:
  1. Accuracy table (main-table columns): R / Rsem / RsemB / C vs A.
  2. Token-economy table: mean exact prompt tokens for R / Rsem / RsemB / C,
     the axis behind "both lexical and dense vanilla RAG cost more than C".

Rsem and RsemB are the dense retrieval baseline; all arms are read from
``probe_eval_results/``.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXISTING_DIR = ROOT / "probe_eval_results/comparison"
BM25_DIR = ROOT / "probe_eval_results/comparison_bm25rag"
E5_DIR = ROOT / "probe_eval_results/comparison_e5_semantic_rag"

MODELS = [
    "Qwen_Qwen2.5-Coder-1.5B-Instruct",
    "Qwen_Qwen2.5-Coder-7B-Instruct",
    "meta-llama_Llama-3.1-8B-Instruct",
    "Qwen_Qwen2.5-Coder-14B-Instruct",
    "Qwen_Qwen2.5-Coder-32B-Instruct",
    "Qwen_Qwen3-Coder-Next",
    "openai_gpt-oss-120b",
    "meta-llama_Llama-3.1-70B-Instruct",
    "meta-llama_Llama-3.1-405B-Instruct",
    "Qwen_Qwen3-Coder-480B-A35B-Instruct",
]


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def _acc(data: dict | None) -> float | None:
    if data is None:
        return None
    summary = data.get("summary") or data
    for key in ("result_accuracy", "accuracy", "acc", "matched_rate"):
        v = summary.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    matched = summary.get("matched") or summary.get("n_matched")
    n = summary.get("n_items") or summary.get("total")
    if matched is not None and n:
        return matched / n
    return None


def _prompt_tokens_mean(data: dict | None) -> float | None:
    if data is None:
        return None
    items = data.get("item_results") or data.get("items") or []
    pts = [
        it.get("prompt_tokens")
        for it in items
        if it.get("prompt_tokens") is not None
    ]
    if not pts:
        return None
    return sum(pts) / len(pts)


def _pct(v: float | None) -> str:
    return f"{v * 100:.2f}" if v is not None else "—"


def _delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "—"
    return f"{(a - b) * 100:+.2f}pp"


def _tok(v: float | None) -> str:
    return f"{v:.0f}" if v is not None else "—"


def main() -> int:
    print("=== E5 accuracy (R0) ===")
    print(f"{'Model':<40} {'A':>7} {'R':>7} {'Rsem':>7} {'RsemB':>7} {'C':>7} "
          f"{'C-Rsem':>9} {'C-RsemB':>9}")
    print("-" * 100)
    for m in MODELS:
        a = _acc(_load(EXISTING_DIR / m / "benchmark_results_condA.json"))
        c = _acc(_load(EXISTING_DIR / m / "benchmark_results_condC.json"))
        r = _acc(_load(BM25_DIR / m / "benchmark_results_condR.json"))
        rsem = _acc(_load(E5_DIR / m / "benchmark_results_condRsem.json"))
        rsemb = _acc(_load(E5_DIR / m / "benchmark_results_condRsemB.json"))
        print(f"{m:<40} {_pct(a):>7} {_pct(r):>7} {_pct(rsem):>7} {_pct(rsemb):>7} "
              f"{_pct(c):>7} {_delta(c, rsem):>9} {_delta(c, rsemb):>9}")

    print()
    print("=== E5 token economy (mean exact prompt tokens, R0) ===")
    print(f"{'Model':<40} {'R':>8} {'Rsem':>8} {'RsemB':>8} {'C':>8}")
    print("-" * 76)
    for m in MODELS:
        r = _prompt_tokens_mean(_load(BM25_DIR / m / "benchmark_results_condR.json"))
        rsem = _prompt_tokens_mean(_load(E5_DIR / m / "benchmark_results_condRsem.json"))
        rsemb = _prompt_tokens_mean(_load(E5_DIR / m / "benchmark_results_condRsemB.json"))
        c = _prompt_tokens_mean(_load(EXISTING_DIR / m / "benchmark_results_condC.json"))
        print(f"{m:<40} {_tok(r):>8} {_tok(rsem):>8} {_tok(rsemb):>8} {_tok(c):>8}")

    print()
    print("=== Interpretation ===")
    print("Rsem is the dense (SBERT) counterpart to the lexical BM25 baseline R.")
    print("A large positive C-Rsem supports 'both lexical and dense vanilla RAG")
    print("are insufficient vs C'. Rsem's prompt-token mean (nominal budget 2000,")
    print("== R) reports dense RAG's token cost; RsemB is the same retriever at a")
    print("budget calibrated to C's measured cost, isolating selection quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
