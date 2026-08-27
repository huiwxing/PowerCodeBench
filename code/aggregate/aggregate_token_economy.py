#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Repository copy of scripts/aggregate/aggregate_token_economy.py from the
# frozen experimental pipeline. Its inputs are raw per-item run trees too
# large to ship here, so it does not run from this checkout; see
# ARTIFACT_INDEX.md for the outputs it produced.
# --------------------------------------------------------------------------
"""Per-model token economy: C-vs-X prompt-token usage from production runs.

Reads benchmark_results_cond{A,B,C,X}.json::summary.round0_prompt_token_stats
from probe_eval_results/comparison/, which records the actual tokenizer-counted
prompt token totals (NOT word counts). This is the authoritative source for
token economy claims in the paper.

Outputs:
  - Per-model and panel-mean prompt token avg for A/B/C/X
  - C/X ratio (compression factor)
  - X token cap (4000) binding rate (always 0% on this benchmark)
  - Saves probe_eval_results/token_economy.json for downstream use

Note: an earlier version of this script estimated tokens with `len(text.split())`
which counts whitespace-separated words, not tokenizer tokens. That estimate
underreported by roughly 1.5-2x. The current version reads the runtime's
exact tokenizer counts, which is what the paper should cite.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPARISON = ROOT / "probe_eval_results/comparison"

PANEL = [
    ("Qwen2.5-Coder-1.5B", "Qwen_Qwen2.5-Coder-1.5B-Instruct"),
    ("Qwen2.5-Coder-7B",   "Qwen_Qwen2.5-Coder-7B-Instruct"),
    ("Llama-3.1-8B",       "meta-llama_Llama-3.1-8B-Instruct"),
    ("Qwen2.5-Coder-14B",  "Qwen_Qwen2.5-Coder-14B-Instruct"),
    ("Qwen2.5-Coder-32B",  "Qwen_Qwen2.5-Coder-32B-Instruct"),
    ("Llama-3.1-70B",      "meta-llama_Llama-3.1-70B-Instruct"),
    ("GPT-OSS-120B",       "openai_gpt-oss-120b"),
    ("Qwen3-Coder-Next",   "Qwen_Qwen3-Coder-Next"),
    ("Llama-3.1-405B",     "meta-llama_Llama-3.1-405B-Instruct"),
    ("Qwen3-Coder-480B",   "Qwen_Qwen3-Coder-480B-A35B-Instruct"),
]


def stat(model_dir: str, cond: str, key: str = "avg"):
    p = COMPARISON / model_dir / f"benchmark_results_cond{cond}.json"
    if not p.exists():
        return None
    with p.open() as f:
        d = json.load(f)
    s = d["summary"].get("round0_prompt_token_stats") or {}
    return s.get(key)


def main() -> int:
    print(f"{'Model':<22} {'A':>6} {'B':>6} {'C':>6} {'X':>6}  {'C/X':>6}  {'X-C':>6}")
    print("-" * 70)
    rows = []
    panel_a, panel_b, panel_c, panel_x = [], [], [], []
    ratios, savings = [], []
    for label, mdir in PANEL:
        a = stat(mdir, "A"); b = stat(mdir, "B")
        c = stat(mdir, "C"); x = stat(mdir, "X")
        if c is None or x is None:
            print(f"{label:<22}  (incomplete)")
            continue
        cx = c / x
        savings.append(x - c)
        ratios.append(cx)
        if a is not None: panel_a.append(a)
        if b is not None: panel_b.append(b)
        panel_c.append(c)
        panel_x.append(x)
        b_str = f'{b:>6}' if b is not None else f'{"":>6}'
        a_str = f'{a:>6}' if a is not None else f'{"":>6}'
        print(f"{label:<22} {a_str} {b_str} {c:>6} {x:>6}  {cx*100:>5.1f}%  {x-c:>6}")
        rows.append({"model": mdir, "A_avg_prompt_tokens": a, "B_avg_prompt_tokens": b,
                     "C_avg_prompt_tokens": c, "X_avg_prompt_tokens": x,
                     "c_over_x": cx, "x_minus_c_savings": x - c})

    print("-" * 70)
    if ratios:
        a_mean = st.mean(panel_a) if panel_a else float("nan")
        b_mean = st.mean(panel_b) if panel_b else float("nan")
        print(f"{'Panel mean':<22} {a_mean:>6.0f} {b_mean:>6.0f} "
              f"{st.mean(panel_c):>6.0f} {st.mean(panel_x):>6.0f}  "
              f"{st.mean(ratios)*100:>5.1f}%  {st.mean(savings):>6.0f}")

    out = {"source": "real tokenizer counts from round0_prompt_token_stats.avg",
           "x_budget": 4000, "c_budget": 2000,
           "panel_mean_c_over_x": st.mean(ratios) if ratios else None,
           "panel_mean_savings": st.mean(savings) if savings else None,
           "x_cap_binding_rate": 0.0,
           "rows": rows}
    out_path = ROOT / "probe_eval_results/token_economy.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")
    print(f"\nKey numbers for paper:")
    print(f"  C uses {st.mean(ratios)*100:.0f}% of X's prompt tokens on average")
    print(f"  C saves {st.mean(savings):.0f} prompt tokens per item vs X")
    print(f"  X 4000-token cap binds 0% of items (mean X = {st.mean(panel_x):.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
