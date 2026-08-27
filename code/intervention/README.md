# Knowledge injection (intervention) module layout

This directory is the repository copy of the `knowledge_injection/`
package from the private experiment repository the runs were executed in,
imported here as `intervention`. It holds the intervention logic used by
the benchmark runner (`code/backend/probe_runner.py`); the API-spec and
derived library-knowledge artifacts it consumes are archived under
`dataset/`.

- `library_spec.py`: loads the standardized API-spec JSON plus one derived
  library-knowledge artifact.  Raw docs stay close to `functions`/`attributes`;
  runtime routing metadata and interface contracts are read from that artifact.
- `library_knowledge.py`: generates the sidecar artifact from raw docs.  It
  derives import aliases, symbol aliases, workflow intents, boundary cards, and
  function/schema/state/output contracts in one pass.
- `contract_index.py`: lower-level derivation helpers for function,
  object-schema, state-mutation, and output-observable contracts.
- `proactive.py`: first-pass injection for A/B/C/X.  It combines task demand,
  per-model L0-L3 risk, token budgets, and library-spec boundary contracts.
- `reactive.py`: fix-round injection for FX/FR/FD/FDR/FS/FDRS/FE.  FDR routes
  compact error evidence into basic fixes, API docs, or boundary contracts.
- `semantic.py`: trace-grounded repair context for executed-but-wrong value
  bugs.  It extracts a compact implementation trace from generated code, then
  attaches derived contracts for the APIs/tables/results actually touched by
  that trace.  It can flag missing docs-derived workflow anchors and
  high-confidence code-only invariants such as self-difference expressions or
  final outputs that do not visibly read documented result tables.  What it
  reads is the original task text, the failure message, and the code trace;
  benchmark query templates, query/code target contracts, and the reference
  answer stay outside that evidence.
- `rag_baseline.py` / `semantic_rag.py`: the BM25 and semantic-embedding
  retrieval baselines used by the C2/E5 RAG-comparison ablations.

Design rule: intervention algorithms should stay library-adaptable.  Raw API
knowledge belongs in the API-spec JSON.  Generated runtime and contract
knowledge lives beside it as one explicit sidecar, keeping it out of the raw
docs.

## Task-need signals and their layer increments

The task-need multiplier `a_l(q)` of the injection score starts from
`base_l` and adds the increments below.  Every signal is derived from the
query alone, by regex, by intent match, or through the query-only demand
candidates; the reference code, the difficulty label, and execution feedback
are outside its inputs.  The SM states the increment table; the matching
vocabulary is here.  Canonical source: `proactive.py`
(`_task_layer_profile`).

| Signal | Fires when the query contains | L1 | L2 | L3 |
|---|---|---:|---:|---:|
| `workflow_intent` | a match against a documented workflow executor (time-series, short-circuit, contingency, OPF, DC power flow) | — | +0.10 | +0.22 |
| `sequential` | `first`, `then`, `after`, `before`, `following`, `once`, `next`, `step`, `subsequent`, `again`, `rerun`, `re-run` | — | — | +0.22 |
| `conditional` | `if`, `whether`, `when`, `check`, `verify`, `ensure`, `exceed(s)`, `violate`, `violation`, `threshold`, `above`, `below`, `greater`, `less` | — | +0.22 * | — |
| `aggregation` | `max(imum)`, `min(imum)`, `highest`, `lowest`, `largest`, `smallest`, `count`, `total`, `sum`, `average`, `mean`, `which`, `index`, `argmax`, `argmin`, `rank` | — | +0.22 * | — |
| `multi_modification` | ≥ 2 modification verbs from `set`, `change`, `modify`, `adjust`, `update`, `increase`, `decrease`, `scale`, `multiply`, `add`, `remove`, `disconnect`, `reconnect`, `connect`, `install`, `curtail`, `replace`, `switch`, `open`, `close`, `out of service` | — | +0.10 | +0.12 |
| `constructor_or_executor_role` | a retrieved candidate carries a construction/network or analysis/executor role | +0.10 | — | — |
| `long_query` | ≥ 35 whitespace tokens | — | +0.05 | +0.05 |

\* `conditional` and `aggregation` share one +0.22 L2 bump: the rule fires on
their disjunction, not once each. L0 takes no task-need increment. The result
is clipped to `cap_l`; `base_l` and `cap_l` are in the SM constants table.
The count of active procedural triggers additionally caps retained L3 examples
at 1, 2, or 3.

## The three proactive artifact classes

- **Query anchors** — high-precision substring matches on network-loader and
  construction-API names; they fire the `tau_anchor` term of the deficit floor
  so a demanded function stays in contention even when its probe deficit is
  small.
- **Intent-consistency filter** — the workflow executors are mutually
  exclusive, so an OPF anchor must not pull in a time-series executor card.
  Exclusion sets derive from the `api_intents` field of
  `dataset/pandapower_library_knowledge.json` (14 intents) rather than from
  benchmark-specific rules.
- **DataFrame boundary cards** — four contracts in the same artifact
  (`runtime.boundary_contracts`): `table_schema`, `time_series`,
  `short_circuit`, `opf_cost`. Each states a convention that spans several APIs
  (e.g. which `net.res_*` column carries line loading) rather than documenting
  one function.

All three are regenerated from the API spec by `library_knowledge.py` rather
than hand-written per benchmark item.
