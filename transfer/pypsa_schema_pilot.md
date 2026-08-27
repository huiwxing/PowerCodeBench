# Schema-level PyPSA migration pilot (archived)

This is the earlier schema-level PyPSA pilot, old SM Table S5
(`tab:sm_pypsa_schema_pilot`), archived here with its content unchanged.
The end-to-end OpenDSS/PyPSA transfer pilot replaced it: that pilot covers
the same ground with mini-benchmarks, per-model profiles, and
A/C/R_sem/R_semB conditions, and its artifacts live under `benchmark/e2_*`,
`dataset/{opendss,pypsa}_docs.json`, and the frozen experimental pipeline's
`probe_eval_results/e2_*` run trees. Cited in the supplementary material as
the SM S10 introduction.

For one documented power-flow entry, the pilot showed that the same L0-L3
probe templates used for pandapower can be instantiated after replacing
only the backend-specific documentation fields and value contracts. That is
the schema-level portability claim, which the later end-to-end pilot
upgraded to executed evidence.

| Schema element | PyPSA pilot instance | Example probe or contract |
|---|---|---|
| API entry | `pypsa.Network.pf` | Description: full non-linear power flow for a network instance. Signature: `pf(snapshots=None, skip_pre=False, x_tol=1e-06, use_seed=False, distribute_slack=False, slack_weights='p_set')`. |
| L0 recognition | Function existence | Four-choice item: identify `Network.pf` as the real PyPSA power-flow method among fabricated near-neighbours. |
| L1 recall | Path and required arguments | Free-response item: full callable path `pypsa.Network.pf`; no required user-supplied argument beyond an existing `Network` instance; optional fields include `snapshots`, `x_tol`, and `distribute_slack`. |
| L2 comprehension | Parameter and return semantics | Multiple-choice item: what `distribute_slack` changes, or what the returned dictionary records. Contract keys: `n_iter`, `converged`, `error`. |
| L3 application | Minimal executable use | Code-generation item: create a small `Network`, add buses, loads, generators, and lines with `n.add(...)`, call `n.pf()`, and print a convergence flag or angle scalar. |
| Result-table contract | Value-error tracing | Scalar tasks would replace pandapower result paths with PyPSA paths such as `n.buses_t.v_ang`, `n.generators_t.p`, and `n.lines_t.p0`. |

The end-to-end pilot supplies 67 curated PyPSA entries in the shared
nine-field schema, executed probes, and a frozen 90-item mini-benchmark,
so this file serves as the archival record the SM points to. No result in
the manuscript rests on it.
