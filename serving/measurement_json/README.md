# serving/measurement_json/ — E3 serving energy/latency measurement JSONs

This directory holds the E3 serving measurements: the frozen per-tier energy
and power aggregate, the per-cell and per-repetition measurement windows, and
the online-serving concurrency gradient. Look here to check a reported serving
energy, power, or latency number. All files are read-only copies of the frozen
E3 artifacts under the experimental pipeline's
`probe_eval_results/e3_protocolA*/` and `probe_eval_results/e3_protocolB/` run
trees. The full generation transcripts (`benchmark_results_*.json`) stay in
those run trees, which the repository excludes by size policy. Cited in the
supplementary material as SM S8, "Full serving tables".

| Path | Contents | Verifies |
|------|----------|----------|
| `e3_serving_table.json` | Frozen per-tier aggregate: per-token energy, per-item energy, per-GPU avg/peak power, utilisation, KV-cache allocation (GiB / tokens), throughput, window provenance, for conditions C and C+FDRS. Old SM Table S27's full energy decomposition; E ratio = median `j_per_item` (C+FDRS) / median `j_per_item` (C). | `tab:sm_serving_protocol_a` (incl. E-ratio column), old Table S27, Pareto-reading KV-cache and power figures |
| `protocolA/<model>/<C\|C_FDRS>/cell_summary.json` | Per-cell summary across repetitions (median of 3; anchors 1 repetition) | SM S8 protocol A |
| `protocolA/<model>/<cond>/rep*/{aggregate,run_meta,energy_*}.json` | Per-repetition measurement window: NVML cumulative total-energy counter readings per GPU (`energy_<jobid>_<node>.json`), run metadata (TP/nodes/window timestamps), aggregate metrics | SM S8 protocol A + measurement caveats |
| `protocolA_anchor_retest/Qwen_Qwen3-Coder-480B.../C/` | 480B anchor replication runs with hardened probe instrumentation (incl. the two documented incomplete attempts, kept for provenance) | SM S8 caveat "single-repetition anchors" |
| `protocolB/<model>/cell_summary.json` and `protocolB/<model>/conc{1,4,16}/{aggregate,bench_serve,run_meta,energy_*}.json` | Online-serving concurrency gradient (vllm bench serve, 200-request frozen pool): TTFT/TPOT/ITL/e2e latency, throughput, per-request energy and energy-window split | `tab:sm_serving_latency`, SM S8 protocol B |

Hardware and window conventions (GH200 nodes, NVML counters, window scopes,
anchor amendments) are documented inside `e3_serving_table.json` under
`hardware` and `meta`, and in each `run_meta.json`.
