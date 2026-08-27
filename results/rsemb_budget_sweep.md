# R_semB budget calibration — full per-budget sweep record

This is the complete per-budget sweep behind the frozen R_semB token budget
B* = 600. Read it to check how that budget was chosen, or to see the measured
prompt-token distribution at every other budget in the grid. The record is
transcribed from the frozen E5 calibration log (2026-07-24 entries, ratified
the same day with B* = 600 frozen). The production runs consuming B* are the
experimental pipeline's `probe_eval_results/comparison_e5_semantic_rag/` run
trees, whose frozen aggregate is archived here as
`results/aggregates/comparison_e5_semantic_rag/proactive_comparison.json`.
Cited in the supplementary material as SM S5, "B* calibration protocol".

## Protocol (frozen)

- Offline, GPU-free, token-only; calibration-run accuracy discarded by
  protocol.
- Sample: `_select_benchmark_items(max_items=100, sample_mode="stratified",
  sample_seed=22, sample_strata="difficulty")` — same seed/stratification
  as runtime; strata D1:30 / D2:30 / D3:20 / D4:20.
- Tokenizer: `Qwen/Qwen2.5-Coder-32B-Instruct`; metering =
  `len(tokenizer.apply_chat_template(msgs, tokenize=True,
  add_generation_prompt=True))`, byte-identical to the runtime accounting
  (`llm_backend.count_chat_prompt_tokens`, vLLM branch).
- Metering fidelity: the 100 reconstructed condition-C prompts match the
  runtime `prompt_tokens` in `condC.json` exactly, 100/100, max|diff| = 0.
- Reference points: offline C mean on the 100-item sample = 1,414.7;
  runtime C mean over all 2,000 items (32B) = 1,464.8; frozen target
  window = 1,450 +/- 10% = [1,305, 1,595].

## Per-budget sweep (R_semB nominal budget -> measured prompt-token distribution, n=100)

| Nominal budget | mean | median | std | p10 | p90 | min | max | Hits [1305, 1595]? |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 400 | 976.4 | 975.5 | 84.0 | 879 | 1083 | 740 | 1218 | no (under) |
| 500 | 1210.6 | 1216.5 | 92.9 | 1088 | 1332 | 994 | 1409 | no (under, -16.5%) |
| **600** | **1459.9** | 1473.0 | 105.9 | 1311 | 1600 | 1248 | 1707 | **yes (+0.7% vs 1450)** |
| 700 | 1711.9 | 1716.5 | 130.8 | 1549 | 1872 | 1396 | 2070 | no (over, +18.1%) |
| 800 | 1916.8 | 1933.0 | 159.3 | 1706 | 2102 | 1534 | 2372 | no |
| 1000 | 2263.5 | 2295.0 | 217.2 | 1954 | 2539 | 1724 | 2715 | no |
| 1200 | 2522.8 | 2535.5 | 242.0 | 2169 | 2814 | 1984 | 3071 | no |

The pre-registered grid was {600, 800, 1000, 1200}. {400, 500, 700} were added
downward in the same calibration round to characterise the curve, which is
monotone with 600 the unique window hit.

## Freeze ruling

B* = 600 frozen (2026-07-24): mean 1,459.9 tokens = +0.7% vs the frozen
1,450 target, -0.3% vs the 32B runtime condition-C mean (1,464.8), and +3.2%
vs the offline sample C mean (1,414.7). All three references sit within
+/-10%, while the neighbouring budgets miss by -16.5% (500) and +18.1% (700).
A single global B* serves the whole panel. R_semB's distribution at fixed
budget is much tighter than C's, std ~106 vs ~379, because C's risk gating
produces high-variance injection volumes. Measured-cost matching is by mean,
per the frozen ruling.
