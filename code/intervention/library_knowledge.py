# --------------------------------------------------------------------------
# Repository copy of knowledge_injection/library_knowledge.py, with imports
# and data-path constants adapted to this repository's layout. Runs CPU-only
# against the archived corpus/spec files in this repository. See
# code/README.md for the module map.
# --------------------------------------------------------------------------
"""Derived library-knowledge artifact generation.

The raw API spec should stay close to documentation: functions, attributes,
and one link to a generated artifact.  This module creates that artifact:
runtime routing metadata plus interface contracts derived from the same raw
docs.  Downstream injection code should consume this single artifact instead
of mixing ad hoc profile and contract sidecars.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from intervention.contract_index import derive_contract_index


def _dedup(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _function_names(docs: Dict) -> Set[str]:
    return {
        str(item.get("name"))
        for item in list(docs.get("functions") or [])
        if isinstance(item, dict) and item.get("name")
    }


def _has_function(docs: Dict, name: str) -> bool:
    return name in _function_names(docs)


def _derive_import_aliases(docs: Dict) -> List[str]:
    library = str(docs.get("library") or docs.get("library_name") or "").strip()
    aliases: List[str] = []
    if library:
        aliases.append(library)
    if not library:
        return _dedup(aliases)

    import_pattern = re.compile(
        rf"\bimport\s+{re.escape(library)}\s+as\s+([A-Za-z_][A-Za-z0-9_]*)"
    )
    for fn in docs.get("functions") or []:
        if not isinstance(fn, dict):
            continue
        for example in fn.get("examples") or []:
            for match in import_pattern.finditer(str(example)):
                aliases.append(match.group(1))
    return _dedup(aliases)


def _derive_system_prompt_code(docs: Dict) -> str:
    library = str(docs.get("library") or docs.get("library_name") or "the target library").strip()
    doc_text = " ".join(
        str(part or "")
        for fn in docs.get("functions") or []
        if isinstance(fn, dict)
        for part in (fn.get("description"), fn.get("category"), fn.get("signature"))
    ).lower()
    attr_text = " ".join(
        str(part or "")
        for attr in docs.get("attributes") or []
        if isinstance(attr, dict)
        for part in (attr.get("description"), attr.get("type"), attr.get("access_pattern"))
    ).lower()
    corpus = f"{doc_text} {attr_text}"
    is_power_system = any(
        term in corpus
        for term in ("power system", "power flow", "electrical network", "bus/node", "active power")
    )
    if is_power_system:
        role = f"Power System Python code generator for the {library} library"
        first_step = "Loads the specified network"
        last_step = "Prints the final numeric result using print()"
        last_print = "requested numeric value"
    else:
        role = f"Python code generator for the {library} library"
        first_step = "Loads or creates the requested input data"
        last_step = "Prints the final requested result using print()"
        last_print = "requested value"

    return (
        f"You are a {role}. Given a natural language task description, write complete, "
        "executable Python code that:\n"
        f"1. {first_step}\n"
        "2. Applies any requested modifications\n"
        "3. Runs the specified analysis or computation\n"
        f"4. {last_step}\n\n"
        "Output ONLY executable Python code. Include all necessary imports. The last print() "
        f"statement should output ONLY the {last_print}."
    )


def _intent(
    intent_id: str,
    patterns: Sequence[str],
    *,
    anchors: Sequence[str] = (),
    gated: Sequence[str] = (),
    exclude_patterns: Sequence[str] = (),
    suppress: Sequence[str] = (),
    context_if: Sequence[str] = (),
    context_add: Sequence[str] = (),
) -> Dict:
    item: Dict = {
        "id": intent_id,
        "patterns": list(patterns),
    }
    if exclude_patterns:
        item["exclude_patterns"] = list(exclude_patterns)
    if anchors:
        item["anchor_functions"] = list(anchors)
    if gated:
        item["gated_functions"] = list(gated)
    if suppress:
        item["suppress_functions"] = list(suppress)
    if context_if:
        item["context_if_functions"] = list(context_if)
    if context_add:
        item["context_add_functions"] = list(context_add)
    return item


def _derive_alias_candidates(docs: Dict) -> Dict[str, str]:
    """Generate conservative symbol-normalization candidates from API names."""
    names = _function_names(docs)
    compact_to_name: Dict[str, str] = {}
    duplicate_compacts: Set[str] = set()
    for name in names:
        compact = re.sub(r"[^a-z0-9]+", "", name.lower())
        if not compact:
            continue
        if compact in compact_to_name and compact_to_name[compact] != name:
            duplicate_compacts.add(compact)
        compact_to_name[compact] = name

    aliases: Dict[str, str] = {}
    for name in names:
        compact = re.sub(r"[^a-z0-9]+", "", name.lower())
        no_underscore = name.replace("_", "")
        if compact and compact not in duplicate_compacts and no_underscore != name:
            aliases[no_underscore] = name

    if "run_timeseries" in names:
        aliases.update({
            "run_time_series": "run_timeseries",
            "run_time_series_simulation": "run_timeseries",
            "timeseries": "run_timeseries",
            "runtimeseries": "run_timeseries",
        })
    if "rundcpp" in names:
        aliases.update({
            "run_dcpp": "rundcpp",
            "run_dcpf": "rundcpp",
            "rundcopp": "rundcpp",
            "dc_power_flow": "rundcpp",
        })
    if "run_contingency" in names:
        aliases.update({
            "contingency_analysis": "run_contingency",
            "run_contingency_analysis": "run_contingency",
            "security_analysis": "run_contingency",
            "run_security_analysis": "run_contingency",
        })
    if "calc_sc" in names:
        aliases.update({
            "short_circuit": "calc_sc",
            "shortcircuit": "calc_sc",
            "run_short_circuit": "calc_sc",
            "calc_short_circuit": "calc_sc",
        })
    if "estimate" in names:
        aliases.update({
            "state_estimation": "estimate",
            "run_state_estimation": "estimate",
            "wls_estimation": "estimate",
        })
    for entity in ("load", "shunt", "measurement"):
        fn = f"create_{entity}"
        if fn in names:
            aliases[f"add_{entity}"] = fn
            aliases[f"create_{entity}s"] = fn
    if "create_gen" in names:
        aliases.update({"create_generator": "create_gen", "add_generator": "create_gen"})
    return {k: v for k, v in sorted(aliases.items())}


def _derive_intent_candidates(docs: Dict) -> List[Dict]:
    """Generate workflow intent candidates from available API symbols."""
    intents: List[Dict] = []
    names = _function_names(docs)

    if "rundcpp" in names:
        intents.append(_intent(
            "dc_power_flow",
            [r"\bdc\s+(?:power|load)\s+flow\b", r"\brundcpp\b", r"algorithm=['\"](?:dc|dcpp|dcpf)['\"]"],
            anchors=["rundcpp"],
            gated=["rundcpp"],
            context_if=["runpp"] if "runpp" in names else [],
            context_add=["rundcpp"],
        ))
    if "runpp" in names:
        intents.append(_intent(
            "ac_power_flow",
            [r"\b(?:power\s+flow|load\s+flow)\b"],
            anchors=["runpp"],
            exclude_patterns=[r"\bdc\s+(?:power|load)\s+flow\b"],
        ))
    if "runopp" in names:
        intents.append(_intent(
            "opf",
            [r"\b(?:opf|optimal\s+power\s+flow)\b"],
            anchors=["runopp"],
            gated=["runopp"],
        ))
    if "calc_sc" in names:
        intents.append(_intent(
            "short_circuit",
            [r"\b(?:short[-_ ]?circuit|short circuit|shortcircuit)\b"],
            anchors=["calc_sc"],
            gated=["calc_sc"],
        ))
    if "run_contingency" in names:
        intents.append(_intent(
            "contingency",
            [r"\b(?:n-1|contingenc(?:y|ies))\b"],
            anchors=["run_contingency"],
            gated=["run_contingency"],
        ))
    if {"DFData", "ConstControl", "OutputWriter", "run_timeseries"} <= names:
        suppress = [
            fn for fn in ["runpp", "runpp_3ph", "rundcpp", "runopp", "run_contingency"]
            if fn in names
        ]
        intents.append(_intent(
            "time_series",
            [r"\btime[-_ ]?series\b", r"\btime steps?\b", r"\b(?:24|48|96)[- ]?step\b", r"\brun_timeseries\b"],
            anchors=["DFData", "ConstControl", "OutputWriter", "run_timeseries"],
            gated=["DFData", "ConstControl", "OutputWriter", "run_timeseries"],
            suppress=suppress,
        ))
    if "estimate" in names:
        anchors = ["estimate"]
        if "create_measurement" in names:
            anchors.append("create_measurement")
        intents.append(_intent(
            "state_estimation",
            [r"\b(?:state\s+estimation|state\s+estimate|wls|measurement)\b"],
            anchors=anchors,
            gated=anchors,
        ))
    if "runpp_3ph" in names:
        intents.append(_intent(
            "three_phase",
            [r"\b(?:3[- ]?phase|three[- ]?phase|unbalanced|asymmetric)\b"],
            exclude_patterns=[r"\bshort[-_ ]?circuit\b", r"\bshortcircuit\b"],
            gated=["runpp_3ph"],
        ))
    if "create_load" in names:
        intents.append(_intent(
            "create_load",
            [
                r"\b(?:add|adding|introduce|place|install|create)\s+(?:a\s+)?(?:new\s+)?(?:\d+(?:\.\d+)?\s*(?:mw|mvar)\s+)?load\b",
                r"\bcreate_?load\b",
            ],
            anchors=["create_load"],
        ))
    if "create_gen" in names:
        intents.append(_intent(
            "create_gen",
            [
                r"\b(?:add|adding|introduce|place|install|create)\s+(?:a\s+)?(?:new\s+)?(?:\d+(?:\.\d+)?\s*mw\s+)?(?:generator|gen)\b",
                r"\bcreate_?gen\b",
                r"\b(?:generator|gen)\b.{0,50}\b(?:added|introduced|placed|installed)\b",
            ],
            anchors=["create_gen"],
        ))
    if "create_shunt" in names:
        intents.append(_intent(
            "create_shunt",
            [
                r"\b(?:add|adding|introduce|place|install|create)\s+(?:a\s+)?(?:new\s+)?(?:\d+(?:\.\d+)?\s*mvar\s+)?shunt\b",
                r"\bcreate_?shunt\b",
                r"\bshunt\b.{0,50}\b(?:added|introduced|placed|installed)\b",
            ],
            anchors=["create_shunt"],
        ))
    if "create_sgen" in names:
        intents.append(_intent(
            "create_sgen",
            [r"\b(?:sgen|static generator|static-generator)\b"],
            anchors=["create_sgen"],
        ))
    if any(str(a.get("type")) in {"input_table", "result_table"} for a in docs.get("attributes") or []):
        intents.append(_intent(
            "table_schema_access",
            [
                r"\b(?:line|trafo|transformer)\b.{0,60}\b(?:loading|loaded|overload|exceed)",
                r"\b(?:maximum|max|minimum|min|peak)\s+(?:line|trafo|transformer)\s+loading",
                r"\b(?:bus\s+)?voltage(?:\s+magnitude)?\b",
                r"\bvm[_ ]?pu\b",
                r"\bmost\s+loaded\s+line\b",
                r"\b(?:set|change|modify|adjust|update|disconnect|reconnect|take|switch|open|close)\b.{0,90}\b(?:bus|line|load|generator|gen|sgen|trafo|transformer|external grid|ext_grid)\b",
                r"\b(?:minimum|min|maximum|max)\s+voltage\s+limit\b",
                r"\b(?:load\s+)?scaling\b",
                r"\bparallel\s+units?\b",
                r"\bout\s+of\s+service\b",
                r"\b(?:res_|net\.)",
                r"\b(?:loading_percent|vm_pu|p_mw|q_mvar|ikss_ka|res_cost)\b",
            ],
        ))
    if any(str(a.get("name")) == "res_cost" for a in docs.get("attributes") or []):
        intents.append(_intent(
            "opf_cost",
            [r"\b(?:opf|optimal\s+power\s+flow)\b.{0,80}\b(?:cost|objective|generation cost)\b"],
        ))
    return intents


def _fields_for(docs: Dict, name: str) -> List[str]:
    for attr in docs.get("attributes") or []:
        if isinstance(attr, dict) and attr.get("name") == name:
            return [
                str(p.get("name"))
                for p in list(attr.get("parameters") or [])
                if isinstance(p, dict) and p.get("name")
            ]
    return []


def _derive_boundary_candidates(docs: Dict) -> List[Dict]:
    """Generate conservative boundary-card candidates from schema/workflow docs."""
    cards: List[Dict] = []
    attrs = {str(a.get("name")): a for a in docs.get("attributes") or [] if isinstance(a, dict)}
    names = _function_names(docs)

    if attrs:
        common_columns = []
        if {"min_vm_pu", "max_vm_pu"} <= set(_fields_for(docs, "bus")):
            common_columns.append("`net.bus['min_vm_pu'/'max_vm_pu']`")
        if "scaling" in _fields_for(docs, "load"):
            common_columns.append("`net.load['scaling']`")
        gen_fields = set(_fields_for(docs, "gen"))
        if {"p_mw", "min_p_mw", "max_p_mw", "in_service"} <= gen_fields:
            common_columns.append("`net.gen['p_mw'/'min_p_mw'/'max_p_mw'/'in_service']`")
        if {"in_service", "parallel"} <= set(_fields_for(docs, "line")):
            common_columns.append("`net.line['in_service'/'parallel']`")
        if "parallel" in _fields_for(docs, "trafo"):
            common_columns.append("`net.trafo['parallel']`")
        result_reads = []
        if "vm_pu" in _fields_for(docs, "res_bus"):
            result_reads.append("`net.res_bus['vm_pu']`")
        if "loading_percent" in _fields_for(docs, "res_line"):
            result_reads.append("`net.res_line['loading_percent']`")
        if "loading_percent" in _fields_for(docs, "res_trafo"):
            result_reads.append("`net.res_trafo['loading_percent']`")
        cards.append({
            "id": "table_schema",
            "title": "DataFrame/Table Contract",
            "related_functions": [fn for fn in ("runpp", "runopp") if fn in names],
            "intent_ids": ["table_schema_access"],
            "evidence_terms": _dedup([
                "res_line.loading",
                "has no attribute 'loading'",
                "bus_min_vm_pu",
                "bus_max_vm_pu",
                "loading_percent",
                "min_vm_pu",
                "max_vm_pu",
                "res_bus",
                "res_line",
                "line loading",
                "most loaded line",
                "voltage limit",
                "res_",
                "net.",
                "vm_pu",
                "p_mw",
                "q_mvar",
                "ikss_ka",
            ]),
            "contract": [
                "pandapower element and result tables are pandas DataFrames under `net.<table>` and `net.res_<table>`; use `.loc[index, 'column']` to update element values and `['column']` to read result columns.",
                "Element references such as bus, line, load, generator, and transformer IDs are pandapower table index labels; `.loc`/`.at` and constructor arguments such as `bus=` should use those labels directly, not a 1-based to 0-based conversion.",
                "Do not invent aggregate attributes such as `net.bus_min_vm_pu`, `net.bus_max_vm_pu`, or `net.res_line.loading`.",
                "Common element columns: " + ", ".join(common_columns) + ".",
                "After `runpp`/`runopp`, read common results with " + ", ".join(result_reads) + "; use `.idxmax()`/`.idxmin()` on those Series when an index is requested.",
            ],
            "source": "derived_schema",
        })
    if {"DFData", "ConstControl", "OutputWriter", "run_timeseries"} <= names:
        cards.append({
            "id": "time_series",
            "title": "Time-Series API Contract",
            "related_functions": ["DFData", "ConstControl", "OutputWriter", "run_timeseries"],
            "intent_ids": ["time_series"],
            "evidence_terms": ["time series", "run_timeseries", "DFData", "ConstControl", "OutputWriter", "profile_name"],
            "contract": [
                "Use `DFData(profiles)` where the DataFrame columns exactly match `ConstControl(..., profile_name=...)`.",
                "For load scaling across all loads, use `element_index=net.load.index` and `profile_name=net.load.index.tolist()` in one vectorized `ConstControl` call.",
                "Do not loop over loads with `profile_name=str(load_index)` unless the profile DataFrame columns are strings with the same labels; mixed integer/string labels cause KeyError during `run_timeseries`.",
                "Create `OutputWriter(net, time_steps=range(n_steps))`, then register outputs with `ow.log_variable('res_bus', 'vm_pu')` and, when needed, `ow.log_variable('res_line', 'loading_percent')`.",
                "Run `run_timeseries(net, time_steps=range(n_steps), verbose=False)` and read multi-step results from `ow.output['res_bus.vm_pu']` or `ow.output['res_line.loading_percent']`.",
            ],
            "source": "derived_workflow",
        })
    if "calc_sc" in names:
        ext_grid_fields = set(_fields_for(docs, "ext_grid"))
        has_ext_grid_sc = {"s_sc_max_mva", "s_sc_min_mva", "rx_max", "rx_min"} <= ext_grid_fields
        gen_fields = set(_fields_for(docs, "gen"))
        sgen_fields = set(_fields_for(docs, "sgen"))
        has_bus_sc = "ikss_ka" in _fields_for(docs, "res_bus_sc")
        contract = [
            "Use `pandapower.shortcircuit.calc_sc` or import it from `pandapower.shortcircuit`.",
        ]
        if has_ext_grid_sc:
            contract.append(
                "Before `calc_sc`, ensure `net.ext_grid` has `s_sc_max_mva`, `s_sc_min_mva`, `rx_max`, and `rx_min` when max/min cases may be used."
            )
        if "in_service" in gen_fields and "in_service" in sgen_fields:
            contract.append(
                "Many network cases have `gen` or `sgen` tables without the short-circuit `vn_kv`/impedance fields expected by `calc_sc`; if those fields are missing, set `net.gen['in_service'] = False` and `net.sgen['in_service'] = False` before the calculation."
            )
        read_line = " and read bus currents from `net.res_bus_sc['ikss_ka']`" if has_bus_sc else ""
        contract.append("Call `calc_sc(net, fault='3ph' or '2ph', case='max' or 'min')`" + read_line + ".")
        cards.append({
            "id": "short_circuit",
            "title": "Short-Circuit API Contract",
            "related_functions": ["calc_sc"],
            "intent_ids": ["short_circuit"],
            "evidence_terms": ["short circuit", "calc_sc", "res_bus_sc", "ikss", "fault"],
            "contract": contract,
            "source": "derived_workflow",
        })
    if "runopp" in names and "res_cost" in attrs:
        cards.append({
            "id": "opf_cost",
            "title": "OPF Cost Result Contract",
            "related_functions": ["runopp"],
            "intent_ids": ["opf_cost"],
            "evidence_terms": ["runopp", "opf", "optimal power flow", "res_cost", "cost", "objective"],
            "contract": [
                "After `runopp(net)`, pandapower stores the total OPF objective in `net.res_cost`, which is commonly a scalar float.",
                "Read it with `float(net.res_cost)`; do not use `net.res_cost.total_cost` or `net.res_cost['total_cost']` unless you have verified that it is a table.",
            ],
            "source": "derived_schema",
        })
    return cards


# ---------------------------------------------------------------------------
# Analysis function → result-table output mapping.
#
# This is the curated piece of library knowledge that captures, for each
# analysis routine, the canonical `net.res_*` columns where pandapower writes
# the post-analysis state. The mapping is library-level (no benchmark task
# information) and small enough to maintain by hand; switching to a different
# library only requires replacing this table together with the docs.
#
# Each row uses the form:
#     function_name -> [
#         (result_table, [column, ...]),
#         ...
#     ]
# Empty column lists denote scalar attributes (e.g. ``net.res_cost``).
# ---------------------------------------------------------------------------

_ANALYSIS_FUNCTION_OUTPUTS: Dict[str, List[tuple]] = {
    "runpp": [
        ("res_bus", ["vm_pu", "va_degree", "p_mw", "q_mvar"]),
        ("res_line", ["loading_percent", "p_from_mw", "q_from_mvar"]),
        ("res_trafo", ["loading_percent"]),
        ("res_gen", ["p_mw", "q_mvar", "vm_pu"]),
        ("res_load", ["p_mw", "q_mvar"]),
        ("res_ext_grid", ["p_mw", "q_mvar"]),
    ],
    "rundcpp": [
        ("res_bus", ["va_degree", "p_mw"]),
        ("res_line", ["loading_percent", "p_from_mw"]),
        ("res_trafo", ["loading_percent"]),
    ],
    "runopp": [
        ("res_bus", ["vm_pu", "va_degree", "p_mw", "q_mvar"]),
        ("res_line", ["loading_percent"]),
        ("res_gen", ["p_mw", "q_mvar"]),
        ("res_cost", []),
    ],
    "calc_sc": [
        ("res_bus_sc", ["ikss_ka", "ip_ka", "ith_ka"]),
        ("res_line_sc", ["ikss_ka"]),
        ("res_trafo_sc", ["ikss_ka"]),
    ],
    "run_contingency": [
        ("res_line", ["max_loading_percent", "min_loading_percent", "loading_percent"]),
        ("res_trafo", ["max_loading_percent", "min_loading_percent", "loading_percent"]),
        ("res_bus", ["max_vm_pu", "min_vm_pu", "vm_pu"]),
    ],
    "estimate": [
        ("res_bus_est", ["vm_pu", "va_degree"]),
        ("res_line_est", ["loading_percent", "p_from_mw"]),
        ("res_trafo_est", ["loading_percent"]),
    ],
    "runpp_3ph": [
        ("res_bus_3ph", ["vm_a_pu", "vm_b_pu", "vm_c_pu", "va_a_degree"]),
        ("res_line_3ph", ["loading_percent_a", "loading_percent_b", "loading_percent_c"]),
    ],
    # run_timeseries does not write into net.res_* directly; multi-step results
    # live in OutputWriter. The contract below makes that explicit so the model
    # does not look for aggregated time-series values inside net.res_*.
    "run_timeseries": [],
}


def _result_columns_present(docs: Dict, table: str, columns: Sequence[str]) -> List[str]:
    """Filter a column list down to those that actually appear in the docs."""
    declared = set(_fields_for(docs, table))
    if not declared:
        return list(columns)
    return [col for col in columns if col in declared]


def _function_output_contract(
    fn_name: str,
    rows: Sequence[tuple],
    docs: Dict,
) -> Optional[Dict]:
    """Build a contract that names where an analysis function writes results."""
    if fn_name not in _function_names(docs):
        return None

    table_lines: List[str] = []
    column_phrases: List[str] = []
    for table, columns in rows:
        present = _result_columns_present(docs, table, list(columns or []))
        if columns and not present:
            continue
        if columns and present:
            cols = ", ".join(f"'{c}'" for c in present[:6])
            phrase = f"`net.{table}[{cols}]`"
            table_lines.append(f"`net.{table}` columns: {cols}.")
            column_phrases.append(phrase)
        elif not columns:
            phrase = f"`net.{table}` (scalar)"
            table_lines.append(f"`net.{table}` is a scalar attribute on the network object.")
            column_phrases.append(phrase)
        else:
            phrase = f"`net.{table}`"
            table_lines.append(f"`net.{table}` (post-analysis result table).")
            column_phrases.append(phrase)

    contract: List[str] = []
    if column_phrases:
        # Headline line carries both the read locations and the anti-pattern,
        # since downstream prompts may render only the first line per contract.
        contract.append(
            f"After `{fn_name}(net, ...)`, read final scalar/aggregate values "
            "from the post-analysis tables "
            + ", ".join(column_phrases)
            + "; do not read aggregated metrics from the function's Python return "
            "value or from any intermediate dictionary it produces."
        )
        contract.extend(table_lines)
        contract.append(
            "Apply the requested operator (point read, .max(), .min(), .idxmax(), "
            ".sum(), or before/after difference) directly on those columns."
        )
    else:
        # run_timeseries-style: results live elsewhere
        contract.append(
            f"`{fn_name}` does not populate `net.res_*` with aggregated multi-step "
            "values; record outputs through `OutputWriter` (e.g. "
            "`ow.log_variable('res_bus', 'vm_pu')`) and read them back from "
            "`ow.output['<table>.<column>']` after the run."
        )
        contract.append(
            "When the task asks for a min/max/sum across the full simulation, "
            "aggregate over `ow.output[...]` rather than the post-loop "
            "`net.res_*` snapshot, which only reflects the final time step."
        )

    query_patterns = [rf"\b{re.escape(fn_name)}\b"]
    code_patterns = [rf"\b{re.escape(fn_name)}\s*\("]

    return {
        "id": f"function_outputs:{fn_name}",
        "kind": "function_outputs",
        "name": fn_name,
        "function": fn_name,
        "query_patterns": query_patterns,
        "code_patterns": code_patterns,
        "contract": [line for line in contract if line],
        "source": "function_outputs",
        "confidence": 0.92,
    }


def _derive_function_output_contracts(docs: Dict) -> List[Dict]:
    """Return curated function_outputs contracts for analysis routines."""
    out: List[Dict] = []
    for fn_name, rows in _ANALYSIS_FUNCTION_OUTPUTS.items():
        contract = _function_output_contract(fn_name, rows, docs)
        if contract:
            out.append(contract)
    return out


def derive_library_knowledge(docs: Dict, *, source_docs_path: str = "") -> Dict:
    """Create a single derived library-knowledge artifact from raw docs."""
    library = docs.get("library") or docs.get("library_name") or "unknown"
    functions = list(docs.get("functions") or [])
    attributes = list(docs.get("attributes") or [])
    contract_index = derive_contract_index(
        docs,
        source_docs_path=source_docs_path,
        attribute_source_path=source_docs_path,
    )

    contracts = list(contract_index.get("contracts") or [])
    contracts.extend(_derive_function_output_contracts(docs))

    summary = dict(contract_index.get("summary") or {})
    by_kind = dict(summary.get("by_kind") or {})
    for c in contracts:
        kind = str(c.get("kind") or "")
        if kind == "function_outputs":
            by_kind[kind] = by_kind.get(kind, 0) + 1
    summary["by_kind"] = by_kind
    summary["total_contracts"] = len(contracts)

    return {
        "version": "library_knowledge_v1",
        "library": library,
        "source_docs_path": source_docs_path,
        "generated_from": {
            "functions": len(functions),
            "attributes": len(attributes),
        },
        "metadata": {
            "runtime_policy": "derived",
            "notes": [
                "All runtime metadata and interface contracts are generated from raw functions, attributes, and examples.",
                "The raw API spec should not embed aliases, intents, boundary cards, or semantic contracts.",
                "function_outputs contracts encode where each curated analysis routine writes its result tables; this is library-level metadata, not benchmark-specific.",
            ],
        },
        "runtime": {
            "import_aliases": _derive_import_aliases(docs),
            "system_prompt_code": _derive_system_prompt_code(docs),
            "aliases": _derive_alias_candidates(docs),
            "api_intents": _derive_intent_candidates(docs),
            "boundary_contracts": _derive_boundary_candidates(docs),
        },
        "contract_summary": summary,
        "contracts": contracts,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a derived library-knowledge artifact.")
    parser.add_argument("--docs", required=True, help="Base API-spec JSON path")
    parser.add_argument("--output", required=True, help="Output library-knowledge JSON path")
    args = parser.parse_args(argv)

    docs_path = Path(args.docs)
    out_path = Path(args.output)
    docs = _load_json(docs_path)
    artifact = derive_library_knowledge(docs, source_docs_path=str(docs_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
        f.write("\n")
    runtime = artifact.get("runtime") or {}
    summary = {
        "aliases": len(runtime.get("aliases") or {}),
        "api_intents": len(runtime.get("api_intents") or []),
        "boundary_contracts": len(runtime.get("boundary_contracts") or []),
        "contracts": len(artifact.get("contracts") or []),
        "contract_summary": artifact.get("contract_summary") or {},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
