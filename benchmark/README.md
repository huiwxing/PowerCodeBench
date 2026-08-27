# benchmark/ — PowerCodeBench frozen release and generator

This directory holds the frozen PowerCodeBench release, the generator that
produced it, and the split and query resources that go with it. Look here
for the item inventory, the composition statistics behind the release
figure, and the transfer-pilot mini-benchmarks. Cited in the main text
benchmark coverage section and in the supplementary material as SM S1.

Key entries:

| Path | Contents |
|------|----------|
| `../benchmark.json` (repository root) | The frozen 2,000-item release used throughout the paper (full item-level inventory: query, reference code, scalar ground truth, scenario spec) |
| repository root (`README.md`, `LICENSE`, `CITATION.cff`, `assets/`) | The public release packaging of the same frozen dataset — the frozen experimental pipeline's `benchmark/powercodebench_release/` staging directory became this repository |
| `fig_benchmark_stats.pdf` | Frozen-release composition plots: items per difficulty (D1-D4), task-family frequency across the 15 families, grid network size distribution across the 39 instantiated networks (regenerated from the release by `../code/build/plot_benchmark_stats.py`) |
| `composition_stats.json` | Per-difficulty, per-family, and per-network composition statistics derived read-only from `benchmark.json` (counts + registry display names) |
| `../code/benchmark_generator/benchmark_config.py`, `../code/benchmark_generator/benchmark_engine.py` | Parameterised generator: network registry, task templates, modification operators, admission/feasibility filters |
| `c3_splits/`, `../code/benchmark_generator/expanded_nl_templates.json`, `naturalistic_holdout/` | Frozen split definitions and query-style resources |
| `e2_opendss/`, `e2_pypsa/`, `e2_freeze_manifest.json` | E2 transfer-pilot mini-benchmarks (90 items each) and their freeze/re-freeze manifest (see `transfer/`) |
| `e4_layer2a/`, `e4_layer2b/` | External-query construction artifacts (archived copies with README under `external_queries/`) |
