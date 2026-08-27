#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Repository copy of scripts/aggregate/aggregate_unified_token_table.py from
# the frozen experimental pipeline. Its inputs are raw per-item run trees too
# large to ship here, so it does not run from this checkout; see
# ARTIFACT_INDEX.md for the outputs it produced.
# --------------------------------------------------------------------------
"""Build the unified token-economy table (proactive R0 + reactive fix rounds).

Replaces the previous split tables (tab:r0_token_economy + tab:token_cost)
with a single block: rows = conditions, columns = Docs / Prompt / Successes
/ Tok./succ. / vs X.

Reads:
  comparison/<model>/benchmark_results_cond{A,B,C,X}.json      (proactive R0)
  comparison_bm25rag/<model>/benchmark_results_condR.json      (BM25 RAG R0)
  comparison/<model>/benchmark_results_cond{C_FX,C_FD,C_FDR,C_FDRS}.json
                                                                (reactive fix rounds)

Token sources (real tokenizer counts, not word-count estimates):
  R0 prompt tokens     -> summary.round0_prompt_token_stats.avg
  R0 matched items      -> summary.n_matched
  Fix prompt tokens    -> summary.fix_round_prompt_token_stats.avg
  Fix doc tokens       -> summary.fix_round_doc_token_stats.avg (when present)
  Fix successful fixes -> summary.n_fix_successes (when present)
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPARISON = ROOT / "probe_eval_results/comparison"
BM25_DIR = ROOT / "probe_eval_results/comparison_bm25rag"

# 10-model open-weight panel (matching paper Table 4)
PANEL = [
    "Qwen_Qwen2.5-Coder-1.5B-Instruct",
    "Qwen_Qwen2.5-Coder-7B-Instruct",
    "meta-llama_Llama-3.1-8B-Instruct",
    "Qwen_Qwen2.5-Coder-14B-Instruct",
    "Qwen_Qwen2.5-Coder-32B-Instruct",
    "meta-llama_Llama-3.1-70B-Instruct",
    "openai_gpt-oss-120b",
    "Qwen_Qwen3-Coder-Next",
    "meta-llama_Llama-3.1-405B-Instruct",
    "Qwen_Qwen3-Coder-480B-A35B-Instruct",
]

PROACTIVE_BASES = ["A", "B", "C", "X"]
REACTIVE_KEYS = ["C_FX", "C_FD", "C_FDR", "C_FDRS"]


def _load_summary(p: Path):
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f).get("summary")


def proactive_r0_stats(model_dir: str, cond: str, base_dir: Path = COMPARISON):
    """Returns (mean_prompt_tokens, n_matched) for a R0-only condition file."""
    s = _load_summary(base_dir / model_dir / f"benchmark_results_cond{cond}.json")
    if not s:
        return None, None
    mean_prompt = (s.get("round0_prompt_token_stats") or {}).get("avg")
    n_matched = s.get("n_matched")
    return mean_prompt, n_matched


def reactive_stats(model_dir: str, key: str):
    """Returns (mean_fix_docs, mean_fix_prompt, n_fix_successes,
    tokens_per_successful_fix) for a reactive run."""
    s = _load_summary(COMPARISON / model_dir / f"benchmark_results_cond{key}.json")
    if not s:
        return None, None, None, None
    fps = s.get("fix_prompt_token_stats") or {}
    fix_docs = (fps.get("docs") or {}).get("avg")
    fix_prompt = (fps.get("prompt") or {}).get("avg")
    n_succ = fps.get("successful_fixes")
    tok_per_succ = fps.get("prompt_tokens_per_successful_fix")
    return fix_docs, fix_prompt, n_succ, tok_per_succ


def main() -> int:
    # ----- proactive R0 panel-mean across the 10 open-weight models -----
    print("=" * 90)
    print("Unified token economy table (panel mean across 10 open-weight models)")
    print("=" * 90)
    print(f"{'Condition':<24} {'Docs (avg)':>11} {'Prompt (avg)':>13} {'Matched/2000':>14} {'Tok./match':>12} {'vs X':>7}")
    print("-" * 90)

    # Collect proactive panel data
    panel_data = {}
    for cond in PROACTIVE_BASES + ["R"]:
        prompts = []
        matches = []
        a_baseline_prompts = []  # used to estimate "docs" portion for proactive
        for m in PANEL:
            base_dir = BM25_DIR if cond == "R" else COMPARISON
            p_mean, n_m = proactive_r0_stats(m, cond if cond != "R" else "R", base_dir)
            if p_mean is None:
                continue
            prompts.append(p_mean)
            matches.append(n_m or 0)
            # For docs estimation: Prompt(this) - Prompt(A) ~ doc tokens injected
            a_p, _ = proactive_r0_stats(m, "A")
            if a_p is not None:
                a_baseline_prompts.append(a_p)
        if not prompts:
            continue
        mean_prompt = st.mean(prompts)
        mean_match = st.mean(matches)
        a_baseline = st.mean(a_baseline_prompts) if a_baseline_prompts else 0
        docs_avg = max(0, mean_prompt - a_baseline) if cond != "A" else 0
        tok_per_match = (mean_prompt * 2000 / mean_match) if mean_match else float("inf")
        panel_data[cond] = {
            "docs": docs_avg, "prompt": mean_prompt, "matched": mean_match,
            "tok_per_match": tok_per_match, "n_models": len(prompts),
        }

    # Reference X prompt for "vs X" column
    x_prompt = panel_data.get("X", {}).get("prompt", 1)

    print(f"--- Round-0 (proactive injection) ---")
    for cond in ["A", "B", "C", "X", "R"]:
        if cond not in panel_data:
            print(f"{cond:<24} {'—':>11} {'—':>13} {'—':>14} {'—':>12} {'—':>7}  (n=0)")
            continue
        d = panel_data[cond]
        vs_x = f"{d['prompt']/x_prompt*100:.0f}%"
        label_map = {"A": "A (no injection)", "B": "B (names only)",
                     "C": "C (proposed)", "X": "X (full-context bound)",
                     "R": "R (BM25 RAG)"}
        label = label_map[cond]
        n = d["n_models"]
        match_str = f"{d['matched']:.1f}" if n == 10 else f"{d['matched']:.1f} (n={n})"
        print(f"{label:<24} {d['docs']:>11.0f} {d['prompt']:>13.0f} "
              f"{match_str:>14} {d['tok_per_match']/1000:>10.2f}k {vs_x:>7}")

    # ----- reactive fix-round panel-mean from production runs -----
    print(f"\n--- Round 1–3 (reactive correction, proactive base C) ---")
    # Skip 0.5B and 1.5B -- they're sub-capacity, paper aggregates on the
    # 10 capable models (1.5B excluded from the reactive cross-model mean).
    REACTIVE_PANEL = [m for m in PANEL if "0.5B" not in m]
    fd_docs_ref = None
    for key in REACTIVE_KEYS:
        d_list, p_list, s_list, t_list = [], [], [], []
        for m in REACTIVE_PANEL:
            d, p, s, t = reactive_stats(m, key)
            if d is None:
                continue
            d_list.append(d); p_list.append(p); s_list.append(s); t_list.append(t)
        if not d_list:
            print(f"{key:<24}  (no data)")
            continue
        d_avg = st.mean(d_list); p_avg = st.mean(p_list)
        s_avg = st.mean(s_list); t_avg = st.mean(t_list)
        if key == "C_FD":
            fd_docs_ref = d_avg
        vs_fd = f"{d_avg/fd_docs_ref:.2f}×" if fd_docs_ref else "—"
        label_map = {"C_FX": "C+FX (self-debug)", "C_FD": "C+FD (always-doc)",
                     "C_FDR": "C+FDR (routed)", "C_FDRS": "C+FDRS (proposed)"}
        label = label_map.get(key, key)
        print(f"{label:<24} {d_avg:>11.0f} {p_avg:>13.0f} "
              f"{s_avg:>14.1f} {t_avg/1000:>10.2f}k {vs_fd:>7}")

    print()
    print("Use these numbers to build a single table block in the paper.")
    print(f"X prompt-token reference: {x_prompt:.0f} (panel mean)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
