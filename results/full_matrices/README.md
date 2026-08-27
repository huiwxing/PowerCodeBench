# results/full_matrices/ — full per-model result matrices

This directory holds the complete per-model result matrices behind the
fix-trajectory, per-difficulty, and per-task numbers in the paper. Look here
to read a cell that the manuscript and SM tables report only in summarised or
rounded form. All values are exported read-only from the frozen result files
under the experimental pipeline's
`probe_eval_results/comparison/<model>/benchmark_results_cond*.json` run
trees, which are excluded from this repository by size policy. Cited in the
main text (fix trajectory and per-difficulty discussion), in SM S5.1, and in
the SM per-task compact table.

| File | Contents | Source condition | Verifies |
|------|----------|------------------|----------|
| `perdiff_R3.csv` | Full model x D1-D4 matrix, Acc and Acc\|Exec at R3 (plus raw total/executed/matched counts) | `condA_FX` (bare baseline + up to 3 naive fix rounds, no intervention) | old SM Table S11 (`tab:sm_diff_r3`), main-text per-difficulty discussion |
| `fix_trajectory_R0R3.csv` | Per-round (R0-R3) execution rate and accuracy per model | `condA_FX` round snapshots | SM Table `tab:sm_fix_trajectory` (S9), main-text R0->R3 quotes |
| `pertask_C_R0.csv` | Full per-model matrix over all 15 task families (Round-0, proactive base C), incl. Qwen3-Coder-Next | `condC` | SM `tab:sm_pertask_compact` (left column group) |
| `perdiff_C_R0.csv` | Per-difficulty Round-0 matrix under proactive base C | `condC` | SM `tab:sm_pertask_compact` (right column group) |

Accuracy columns in `perdiff_R3.csv` / `fix_trajectory_R0R3.csv` are printed
with 3 decimals, identical to the values quoted in the manuscript and SM
tables. `pertask_C_R0.csv` / `perdiff_C_R0.csv` have 4 decimals; the SM
compact table rounds these to whole percent or one decimal.
