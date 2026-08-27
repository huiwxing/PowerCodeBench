# Instrument migration ledger — components beyond pandapower

This file records what it takes to point the instrument at a simulation
library other than pandapower: which parts are reused unchanged, and which
have to be regenerated. The OpenDSS/PyPSA transfer pilot under `transfer/`
instantiates the ledger end to end. Cited in the main text under backend
transfer; the supplementary material summarises the six-component
breakdown under "Backend migration scope" in SM Sec. S9.3, and the pilot
itself is SM Sec. S9 (Tables S18--S19).

## Scope split

The expected migration workload beyond pandapower splits along one line.
The boundary-probing and demand-risk architecture is library-agnostic and
is reused at the framework level: probe templates, risk aggregation, the
demand-ranking formulation, injection scoring, and exception routing. The
backend-facing artifact layer must be regenerated per simulation library:
the documentation parser into the shared nine-field schema, executable
minimal-use examples, the role taxonomy, layer snippets and boundary
cards, backend exception classes and result-table contracts, and the
benchmark generator's loaders and feasibility filters.

## Six-component breakdown

| Component | Library-agnostic part | Backend-specific replacement |
|-----------|----------------------|------------------------------|
| API-spec schema and L0-L3 probes | Function name, path, signature, parameter, return, description, and example fields; L0-L3 probe templates. | Parser or scraper that maps the target library's documentation into the shared schema; backend-specific distractors and executable minimal-use examples. |
| Knowledge profile and risk score | Per-model probe scoring, L0-L3 aggregation, and layer-wise risk definition. | No architectural rewrite expected; scores are recomputed from the target library's probe suite. |
| Demand estimation | Query-to-API ranking formulation, hybrid retrieval features, and role-frequency reweighting. | Backend role taxonomy and labelled query-API pairs generated from the target benchmark or documentation examples. |
| Proactive intervention | Demand-risk scoring, token-budgeted layer selection, and prompt renderer. | Layer snippets, boundary cards, and API contracts rendered from the target backend's documentation. |
| Reactive correction | Exception routing, reference-free code-error repair, and validation-enabled value-error branch. | Backend exception classes, result-table schemas, and function-output contracts used to trace numerical mismatches. |
| Benchmark generator | Difficulty-axis design, executable reference-code protocol, and frozen-release discipline. | Network/case loader, task templates, feasibility filters, and ground-truth extraction logic for the target simulator. |

## Pilot instantiation evidence (OpenDSS / PyPSA)

Porting required regenerating only the backend-facing artifacts. The
probing chain ran with one environment hook, and the benchmark chain ran
with no code change beyond the library selection variable. Bring-up
exposed three backend-neutral instrument defects, all fixed and
regression-verified to leave pandapower probe behaviour unchanged (full
defect write-ups: `transfer/DEFECT_LEDGER.md`):

1. the L3 scorer now matches underscore-joined spec-name variants of
   dotted call paths (OpenDSS's `Module_Method` naming);
2. target-API detection was extended by entry type for PyPSA's instance
   methods and attribute entries;
3. L3 scoring execution was moved to an isolated subprocess because
   OpenDSS's native engine can segfault the runner (negative return codes
   recorded as native crashes).

One environment prerequisite accompanies the artifact swap: the target
library must be installed in the pinned execution environment. Here that
is `opendssdirect` 0.9.4 and `pypsa` 1.2.3, with the frozen pandapower
stack held at its pinned versions.

Backend artifacts produced by the pilot: `dataset/opendss_docs.json`
(76 curated entries) and `dataset/pypsa_docs.json` (67 curated entries),
both isomorphic to the pandapower nine-field spec schema
(`dataset/pandapower_docs.json`, 275 functions); mini-benchmarks under
`benchmark/e2_opendss/` and `benchmark/e2_pypsa/` (90 items each, frozen,
hashes in `benchmark/e2_freeze_manifest.json`).
