# --------------------------------------------------------------------------
# Repository copy of benchmark/benchmark_engine.py, unmodified apart from this
# header. Runs CPU-only against the archived corpus/spec files in this
# repository. See code/README.md for the module map.
# --------------------------------------------------------------------------
"""
Power System Analysis Benchmark Engine (v7)
============================================
v7 adds:
  - D3_semantic / D4_compound difficulty levels
  - Derived semantic rewriting for D3/D4 items
  - Diagnostic failure decomposition for D1/D2
  - Difficulty labels on all BenchmarkItems
"""
 
import copy
import json
import random
import hashlib
import logging
import io
import contextlib
import time
import os
import math
import re
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Dict, List
from collections import Counter
from string import Formatter
 
try:
    from .benchmark_config import (
        NETWORK_REGISTRY, SIMBENCH_NETWORKS, FAST_NETWORKS, LARGE_NETWORKS,
        MODIFICATION_DEFS, MODIFICATION_PARAM_DEFAULTS,
        TASK_CONFIG, EXTRACT_CODE_TEMPLATES, NL_TEMPLATES,
        SAMPLING_CONFIG, TIME_SERIES_CONFIG, STATE_ESTIMATION_CONFIG,
        LLM_EXPANSION_CONFIG, EXPANDED_TEMPLATES_PATH,
        MULTI_TASK_NL_TEMPLATES, SEMANTIC_NL_TEMPLATES, COMPOUND_NL_TEMPLATES,
        TASK_FUNCTION_PATTERNS, SEMANTIC_BALANCE_CONFIG,
    )
except ImportError:
    from benchmark_config import (
        NETWORK_REGISTRY, SIMBENCH_NETWORKS, FAST_NETWORKS, LARGE_NETWORKS,
        MODIFICATION_DEFS, MODIFICATION_PARAM_DEFAULTS,
        TASK_CONFIG, EXTRACT_CODE_TEMPLATES, NL_TEMPLATES,
        SAMPLING_CONFIG, TIME_SERIES_CONFIG, STATE_ESTIMATION_CONFIG,
        LLM_EXPANSION_CONFIG, EXPANDED_TEMPLATES_PATH,
        MULTI_TASK_NL_TEMPLATES, SEMANTIC_NL_TEMPLATES, COMPOUND_NL_TEMPLATES,
        TASK_FUNCTION_PATTERNS, SEMANTIC_BALANCE_CONFIG,
    )
 
logger = logging.getLogger(__name__)
BENCHMARK_DIR = Path(__file__).resolve().parent
 
# ---------------------------------------------------------------------------
# TEMPLATE MERGING — auto-loads expanded templates if available
# ---------------------------------------------------------------------------
_merged_templates_cache = None
_merged_multi_cache = None
_merged_semantic_cache = None
_merged_compound_cache = None

_FORMATTER = Formatter()
_MOD_DESC_FIELDS = ("mod_description", "mod_description_lower")

def _parse_template_key(key):
    """Convert normalized repr(tuple) keys back to tuple form when needed."""
    if isinstance(key, str) and key.startswith("("):
        try:
            return ast.literal_eval(key)
        except Exception:
            return key
    return key

def _template_fields(template: str) -> set:
    """Extract placeholder field names from a Python format string."""
    return {
        field_name
        for _, field_name, _, _ in _FORMATTER.parse(template)
        if field_name
    }

def _section_template_source(section: str) -> dict:
    """Return base template dict for one section."""
    return {
        "single": NL_TEMPLATES,
        "multi": MULTI_TASK_NL_TEMPLATES,
        "semantic": SEMANTIC_NL_TEMPLATES,
        "compound": COMPOUND_NL_TEMPLATES,
    }.get(section, {})

def _required_template_fields(section: str, key) -> tuple:
    """Infer required placeholders from base templates for one template key."""
    parsed_key = _parse_template_key(key)
    source = _section_template_source(section)
    templates = source.get(parsed_key, [])
    fields = set()
    for tmpl in templates:
        fields.update(_template_fields(tmpl))

    if section in ("single", "multi"):
        fields.difference_update(_MOD_DESC_FIELDS)
    return tuple(sorted(fields))

def _required_template_hint(section: str, key) -> str:
    """Human-readable placeholder hint for API expansion prompts."""
    fields = [f"{{{name}}}" for name in _required_template_fields(section, key)]
    if section in ("single", "multi"):
        if fields:
            return (
                "all of "
                + ", ".join(fields)
                + ", and exactly one of {mod_description} or {mod_description_lower}"
            )
        return "exactly one of {mod_description} or {mod_description_lower}"
    return "all of " + ", ".join(fields) if fields else "{network_display}"

def _template_dummy_kwargs() -> dict:
    """All placeholder kwargs used to validate template format strings."""
    return {
        "network_display": "TEST_NET",
        "mod_description": "Test mod. ",
        "mod_description_lower": "test mod, ",
        "filter_idx": 0,
        "threshold": 0.95,
        "column": "vm_pu",
        "n_steps": 24,
        "load_lo": 0.6,
        "load_hi": 1.4,
        "time_step": 5,
        "v_std": 0.01,
        "p_std": 1.0,
        "se_seed": 42,
        "scenario_phrase": "simulate a stressed operating condition",
        "v_threshold": 0.95,
        "shunt_mvar": 10,
        "redispatch_mw": 15.0,
        "metric_name": "minimum bus voltage (p.u.)",
        "baseline_metric": "the minimum bus voltage (p.u.)",
        "modification_desc": "increase one selected load",
        "condition_metric": "the maximum line loading",
        "condition_op_word": "exceeds",
        "condition_threshold": 95,
        "action_desc": "disconnect the most loaded line",
        "final_query_desc": "the final minimum bus voltage (p.u.)",
        "target_bus": 0,
        "fix_gen_mw": 10,
        "fix_bus": 0,
        "task_a_name": "power flow",
        "task_b_name": "DC power flow",
        "metric_a_desc": "the minimum bus voltage (p.u.)",
        "metric_b_desc": "the maximum line loading (%)",
        "overload_threshold": 95,
        "voltage_threshold": 0.95,
    }

def _template_validation_error(section: str, key, tmpl: str) -> Optional[str]:
    """Return a validation error message for one template, or None if valid."""
    if section in ("single", "multi"):
        has_mod = any("{" + field + "}" in tmpl for field in _MOD_DESC_FIELDS)
        if not has_mod:
            return "Missing {mod_description} or {mod_description_lower}"

    missing_fields = [
        field for field in _required_template_fields(section, key)
        if "{" + field + "}" not in tmpl
    ]
    if missing_fields:
        return "Missing required placeholders: " + ", ".join(
            "{" + field + "}" for field in missing_fields
        )

    try:
        tmpl.format(**_template_dummy_kwargs())
    except (KeyError, IndexError, ValueError) as e:
        return f"format() failed: {e}"
    return None

def _repair_expanded_template(section: str, tmpl: str) -> str:
    """Repair common placeholder omissions in LLM-generated templates."""
    if section in ("single", "multi"):
        has_mod = any("{" + field + "}" in tmpl for field in _MOD_DESC_FIELDS)
        if not has_mod:
            tmpl = "{mod_description}" + tmpl.lstrip()
    return tmpl
 
def _get_merged_templates() -> dict:
    """Return NL_TEMPLATES merged with LLM-expanded single-task templates."""
    global _merged_templates_cache
    if _merged_templates_cache is not None:
        return _merged_templates_cache
    merged = {k: list(v) for k, v in NL_TEMPLATES.items()}
    _load_expanded_into(merged, section="single")
    _merged_templates_cache = merged
    return merged
 
def _get_merged_multi_templates() -> dict:
    """Return MULTI_TASK_NL_TEMPLATES merged with LLM-expanded multi-task templates."""
    global _merged_multi_cache
    if _merged_multi_cache is not None:
        return _merged_multi_cache
    merged = {k: list(v) for k, v in MULTI_TASK_NL_TEMPLATES.items()}
    _load_expanded_into(merged, section="multi")
    _merged_multi_cache = merged
    return merged

def _get_merged_semantic_templates() -> dict:
    """Return SEMANTIC_NL_TEMPLATES merged with expanded D3 variants."""
    global _merged_semantic_cache
    if _merged_semantic_cache is not None:
        return _merged_semantic_cache
    merged = {k: list(v) for k, v in SEMANTIC_NL_TEMPLATES.items()}
    _load_expanded_into(merged, section="semantic")
    _merged_semantic_cache = merged
    return merged

def _get_merged_compound_templates() -> dict:
    """Return COMPOUND_NL_TEMPLATES merged with expanded D4 variants."""
    global _merged_compound_cache
    if _merged_compound_cache is not None:
        return _merged_compound_cache
    merged = {k: list(v) for k, v in COMPOUND_NL_TEMPLATES.items()}
    _load_expanded_into(merged, section="compound")
    _merged_compound_cache = merged
    return merged
 
def _load_expanded_into(merged: dict, section: str):
    """Load expanded templates from JSON file into merged dict."""
    if not os.path.exists(EXPANDED_TEMPLATES_PATH):
        return
    try:
        with open(EXPANDED_TEMPLATES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Handle both old format (flat dict) and new format ({"single": ..., "multi": ...})
        if section in data and isinstance(data[section], dict):
            expanded = data[section]
        elif section == "single" and "single" not in data:
            expanded = data  # old format: flat dict of single-task templates
        else:
            return
 
        for key_str, templates in expanded.items():
            key = _parse_template_key(key_str)
            if key not in merged:
                merged[key] = []
            seen = set(merged[key])
            for t in templates:
                if t not in seen:
                    merged[key].append(t)
                    seen.add(t)
 
        n_orig_single = sum(len(v) for v in NL_TEMPLATES.values())
        n_orig_multi = sum(len(v) for v in MULTI_TASK_NL_TEMPLATES.values())
        n_merged = sum(len(v) for v in merged.values())
        logger.info(f"Loaded expanded [{section}] templates: {n_merged} total")
    except Exception as e:
        logger.warning(f"Failed to load {EXPANDED_TEMPLATES_PATH} [{section}]: {e}")
 
# ---------------------------------------------------------------------------
# 1. DATA STRUCTURES
# ---------------------------------------------------------------------------
 
@dataclass
class Modification:
    op: str
    params: dict = field(default_factory=dict)
    def to_dict(self):
        return {"op": self.op, **self.params}
 
@dataclass
class QueryTarget:
    qtype: str
    table: str
    column: str
    filter_idx: Optional[int] = None
    threshold: Optional[float] = None
    operator: Optional[str] = None
    # Extra fields for TS/SE
    time_step: Optional[int] = None       # for ts_point_at_step
    se_seed: Optional[int] = None         # for SE reproducibility
    n_steps: Optional[int] = None         # for TS
    load_scaling_range: Optional[tuple] = None  # for TS
    def to_dict(self):
        d = {"qtype": self.qtype, "table": self.table, "column": self.column}
        for k in ("filter_idx", "threshold", "operator", "time_step",
                   "se_seed", "n_steps", "load_scaling_range"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v if not isinstance(v, tuple) else list(v)
        return d
 
@dataclass
class Scenario:
    network: str
    task: str
    modifications: list = field(default_factory=list)
    query_target: QueryTarget = None
    task_options: dict = field(default_factory=dict)
    description: str = ""
    def to_dict(self):
        d = {
            "network": self.network, "task": self.task,
            "modifications": [m.to_dict() for m in self.modifications],
            "query_target": self.query_target.to_dict(),
        }
        if self.task_options:
            d["task_options"] = self.task_options
        if self.description:
            d["description"] = self.description
        return d
 
@dataclass
class BenchmarkItem:
    id: str
    scenario: "Scenario"          # keep existing Scenario reference
    natural_language_query: str
    reference_code: str
    ground_truth: Any
    ground_truth_type: str
    difficulty_level: str = "D1_basic"  # ← NEW FIELD
    # Optional: for D3/D4 items that have behavioral eval criteria
    eval_criteria: dict = field(default_factory=dict)  # ← NEW FIELD

    def __post_init__(self):
        self.natural_language_query = normalize_nl_query_text(
            self.natural_language_query
        )
 
    def to_dict(self):
        d = {
            "id": self.id, "scenario": self.scenario.to_dict(),
            "natural_language_query": self.natural_language_query,
            "reference_code": self.reference_code,
            "ground_truth": self.ground_truth,
            "ground_truth_type": self.ground_truth_type,
            "difficulty_level": self.difficulty_level,  # ← NEW
        }
        if self.eval_criteria:
            d["eval_criteria"] = self.eval_criteria   # ← NEW
        return d
 
# ---------------------------------------------------------------------------
# 2. NETWORK LOADING
# ---------------------------------------------------------------------------
 
def _all_registries():
    return {**NETWORK_REGISTRY, **SIMBENCH_NETWORKS}
 
def _load_network(network_name):
    import pandapower.networks as pn
    try:
        import simbench as sb
    except ImportError:
        sb = None
    return eval(_all_registries()[network_name]["loader"])
 
# ---------------------------------------------------------------------------
# 3. MODIFICATION HANDLERS (unchanged from v2)
# ---------------------------------------------------------------------------
 
def apply_modification(net, mod: Modification):
    p, op = mod.params, mod.op
    if op.startswith("set_"):
        _apply_set_op(net, op, p)
    elif op.startswith("scale_all_"):
        if op == "scale_all_loads":
            net.load["scaling"] = p["factor"]
    elif op.startswith("disconnect_"):
        getattr(net, op.split("_", 1)[1]).at[p["idx"], "in_service"] = False
    elif op in ("open_switch", "close_switch"):
        net.switch.at[p["idx"], "closed"] = (op == "close_switch")
    elif op.startswith("add_"):
        _apply_add_op(net, op, p)
    else:
        raise ValueError(f"No handler for op: {op}")
 
_SET_MAPPING = {
    "set_load_p": ("load", "p_mw"), "set_load_q": ("load", "q_mvar"),
    "set_load_scaling": ("load", "scaling"), "set_gen_p": ("gen", "p_mw"),
    "set_gen_vm": ("gen", "vm_pu"), "set_ext_grid_vm": ("ext_grid", "vm_pu"),
    "set_gen_max_p": ("gen", "max_p_mw"), "set_gen_min_p": ("gen", "min_p_mw"),
    "set_line_length": ("line", "length_km"), "set_line_max_i": ("line", "max_i_ka"),
    "set_line_r": ("line", "r_ohm_per_km"), "set_line_x": ("line", "x_ohm_per_km"),
    "set_line_parallel": ("line", "parallel"),
    "set_trafo_tap_pos": ("trafo", "tap_pos"),
    "set_trafo_parallel": ("trafo", "parallel"), "set_trafo_shift": ("trafo", "shift_degree"),
    "set_bus_max_vm": ("bus", "max_vm_pu"), "set_bus_min_vm": ("bus", "min_vm_pu"),
    "set_gen_out_of_service": ("gen", "in_service"),
}
 
def _apply_set_op(net, op, p):
    if op in _SET_MAPPING:
        tbl, col = _SET_MAPPING[op]
        val = False if op == "set_gen_out_of_service" else p["value"]
        getattr(net, tbl).at[p["idx"], col] = val
    else:
        raise ValueError(f"No set handler for: {op}")
 
def _apply_add_op(net, op, p):
    import pandapower as pp
    if op == "add_load":
        pp.create_load(net, bus=p["bus"], p_mw=p["p_mw"], q_mvar=p.get("q_mvar", 0))
    elif op == "add_sgen":
        pp.create_sgen(net, bus=p["bus"], p_mw=p["p_mw"], q_mvar=p.get("q_mvar", 0))
    elif op == "add_shunt":
        pp.create_shunt(net, bus=p["bus"], q_mvar=p["q_mvar"], p_mw=p.get("p_mw", 0))
    elif op == "add_storage":
        pp.create_storage(net, bus=p["bus"], p_mw=p["p_mw"], max_e_mwh=p.get("max_e_mwh", 1.0))
    elif op == "add_gen":
        pp.create_gen(net, bus=p["bus"], p_mw=p["p_mw"], vm_pu=p.get("vm_pu", 1.0))
 
def mod_to_code(mod: Modification) -> str:
    mdef = MODIFICATION_DEFS.get(mod.op)
    if mdef is None:
        return f"# Unknown op: {mod.op}"
    return mdef["code"].format(**{**MODIFICATION_PARAM_DEFAULTS, **mod.params})
 
def describe_modification(mod: Modification) -> str:
    mdef = MODIFICATION_DEFS.get(mod.op)
    if mdef is None:
        return f"apply modification: {mod.op}"
    return mdef["describe"].format(**{**MODIFICATION_PARAM_DEFAULTS, **mod.params})
 
 
# ---------------------------------------------------------------------------
# 4. TASK EXECUTION — standard tasks
# ---------------------------------------------------------------------------
 
def run_task(net, task: str, task_options: dict = None):
    """Execute standard tasks (PF, DC, SC, OPF, Contingency)."""
    import pandapower as pp
    import pandapower.shortcircuit
    tc = TASK_CONFIG[task]
 
    if tc.get("needs_setup_fn"):
        raise ValueError(f"Task '{task}' requires setup_fn, use dedicated executor")
 
    opts = task_options or {}
    setup_lines = tc.get("setup_code", [])
    if setup_lines:
        exec("\n".join(setup_lines), {"net": net, "pp": pp, "len": len})
 
    ns = {"net": net, "pp": pp}
    for imp_line in tc.get("imports", []):
        exec(imp_line, ns)
    exec(tc["run_code"], ns)
 
 
# ---------------------------------------------------------------------------
# 5. TIME SERIES — setup, execute, extract, codegen
# ---------------------------------------------------------------------------
 
def _execute_time_series(net, qt: QueryTarget, seed: int = 42):
    """
    Run time series simulation.
    Returns: (ow_results_dict, ground_truth)
    """
    import numpy as np
    import pandas as pd
    from pandapower.timeseries import DFData, OutputWriter, run_timeseries
    from pandapower.control import ConstControl
 
    rng_np = np.random.RandomState(seed)
 
    n_steps = qt.n_steps or TIME_SERIES_CONFIG["n_steps"]
    lo, hi = qt.load_scaling_range or TIME_SERIES_CONFIG["load_scaling_range"]
 
    n_loads = len(net.load)
    if n_loads == 0:
        raise ValueError("Network has no loads for time series")
 
    # Generate load scaling profiles
    profiles = pd.DataFrame(
        rng_np.uniform(lo, hi, size=(n_steps, n_loads)),
        columns=net.load.index
    )
 
    ds = DFData(profiles)
    ConstControl(net, element='load', variable='scaling',
                 element_index=net.load.index,
                 data_source=ds, profile_name=net.load.index.tolist())
 
    ow = OutputWriter(net, time_steps=range(n_steps))
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
 
    run_timeseries(net, time_steps=range(n_steps), verbose=False)
 
    # Collect results
    ow_results = {
        "res_bus.vm_pu": ow.output["res_bus.vm_pu"],
        "res_line.loading_percent": ow.output["res_line.loading_percent"],
    }
 
    # Extract ground truth based on qtype
    key = f"{qt.table}.{qt.column}"
    df = ow_results[key]
 
    if qt.qtype == "ts_min_value":
        gt = round(float(df.min().min()), 6)
    elif qt.qtype == "ts_max_value":
        gt = round(float(df.max().max()), 6)
    elif qt.qtype == "ts_point_at_step":
        gt = round(float(df.iloc[qt.time_step, qt.filter_idx]), 6)
    else:
        raise ValueError(f"Unknown TS query type: {qt.qtype}")
 
    return ow_results, gt
 
 
def _generate_ts_reference_code(scenario: Scenario) -> str:
    """Generate complete reference code for time series task."""
    reg = _all_registries()
    net_info = reg[scenario.network]
    qt = scenario.query_target
    n_steps = qt.n_steps or TIME_SERIES_CONFIG["n_steps"]
    lo, hi = qt.load_scaling_range or TIME_SERIES_CONFIG["load_scaling_range"]
    seed = qt.se_seed or 42  # reuse se_seed field for TS seed
 
    lines = [
        "import pandapower as pp",
        "import pandapower.networks as pn",
        "import numpy as np",
        "import pandas as pd",
        "from pandapower.timeseries import DFData, OutputWriter, run_timeseries",
        "from pandapower.control import ConstControl",
        "",
        f"# Load {net_info['display']}",
        f"net = {net_info['loader']}",
        "",
    ]
 
    if scenario.modifications:
        lines.append("# Apply modifications")
        for mod in scenario.modifications:
            lines.append(mod_to_code(mod))
        lines.append("")
 
    lines += [
        "# Create time-varying load profiles",
        f"np.random.seed({seed})",
        f"n_steps = {n_steps}",
        f"profiles = pd.DataFrame(",
        f"    np.random.uniform({lo}, {hi}, size=(n_steps, len(net.load))),",
        f"    columns=net.load.index",
        f")",
        "",
        "# Bind profiles to network",
        "ds = DFData(profiles)",
        "ConstControl(net, element='load', variable='scaling',",
        "             element_index=net.load.index,",
        "             data_source=ds, profile_name=net.load.index.tolist())",
        "",
        "# Set up output logging",
        "ow = OutputWriter(net, time_steps=range(n_steps))",
        "ow.log_variable('res_bus', 'vm_pu')",
        "ow.log_variable('res_line', 'loading_percent')",
        "",
        "# Run time series simulation",
        "run_timeseries(net, time_steps=range(n_steps), verbose=False)",
        "",
        "# Extract result",
    ]
 
    if qt.qtype == "ts_min_value":
        lines.append(f'result = ow.output["{qt.table}.{qt.column}"].min().min()')
    elif qt.qtype == "ts_max_value":
        lines.append(f'result = ow.output["{qt.table}.{qt.column}"].max().max()')
    elif qt.qtype == "ts_point_at_step":
        lines.append(f'result = ow.output["{qt.table}.{qt.column}"].iloc[{qt.time_step}, {qt.filter_idx}]')
 
    lines.append("print(result)")
    return "\n".join(lines)
 
 
# ---------------------------------------------------------------------------
# 6. STATE ESTIMATION — setup, execute, extract, codegen
# ---------------------------------------------------------------------------
 
def _execute_state_estimation(net, qt: QueryTarget, seed: int = 42):
    """
    Run state estimation: PF for true values → noisy measurements → WLS estimate.
    Returns: ground_truth
    """
    import pandapower as pp
    import numpy as np
    from pandapower.estimation import estimate
 
    sc = STATE_ESTIMATION_CONFIG
    rng_np = np.random.RandomState(seed)
 
    # Step 1: Run true power flow
    pp.runpp(net)
 
    # Step 2: Place measurements — V at all buses, P+Q injection at all buses
    in_service_buses = net.bus[net.bus.in_service].index
 
    for bus_idx in in_service_buses:
        true_v = net.res_bus.at[bus_idx, "vm_pu"]
        noisy_v = true_v + rng_np.normal(0, sc["noise_scale_v"])
        pp.create_measurement(net, "v", "bus", noisy_v, sc["voltage_std_dev"],
                              element=bus_idx)
 
    for bus_idx in in_service_buses:
        true_p = net.res_bus.at[bus_idx, "p_mw"]
        true_q = net.res_bus.at[bus_idx, "q_mvar"]
        pp.create_measurement(net, "p", "bus",
                              true_p + rng_np.normal(0, sc["noise_scale_p"]),
                              sc["power_bus_std_dev"], element=bus_idx)
        pp.create_measurement(net, "q", "bus",
                              true_q + rng_np.normal(0, sc["noise_scale_q"]),
                              sc["power_bus_std_dev"], element=bus_idx)
 
    # Step 3: Run WLS state estimation
    success = estimate(net, algorithm='wls')
    if not success or (isinstance(success, dict) and not success.get("success")):
        raise RuntimeError("State estimation did not converge")
 
    # Step 4: Extract ground truth from res_bus_est
    table = getattr(net, qt.table)
    col = table[qt.column]
 
    if qt.qtype == "point_value":
        return round(float(col.at[qt.filter_idx]), 6)
    elif qt.qtype == "max_value":
        return round(float(col.max()), 6)
    elif qt.qtype == "min_value":
        return round(float(col.min()), 6)
    else:
        raise ValueError(f"Unknown SE query type: {qt.qtype}")
 
 
def _generate_se_reference_code(scenario: Scenario) -> str:
    """Generate complete reference code for state estimation task."""
    reg = _all_registries()
    net_info = reg[scenario.network]
    qt = scenario.query_target
    seed = qt.se_seed or 42
    sc = STATE_ESTIMATION_CONFIG
 
    lines = [
        "import pandapower as pp",
        "import pandapower.networks as pn",
        "import numpy as np",
        "from pandapower.estimation import estimate",
        "",
        f"# Load {net_info['display']}",
        f"net = {net_info['loader']}",
        "",
    ]
 
    if scenario.modifications:
        lines.append("# Apply modifications")
        for mod in scenario.modifications:
            lines.append(mod_to_code(mod))
        lines.append("")
 
    lines += [
        "# Step 1: Run true power flow",
        "pp.runpp(net)",
        "",
        "# Step 2: Create noisy measurements from true PF results",
        f"np.random.seed({seed})",
        "in_service_buses = net.bus[net.bus.in_service].index",
        "",
        "# Voltage measurements at all buses",
        "for bus_idx in in_service_buses:",
        f'    true_v = net.res_bus.at[bus_idx, "vm_pu"]',
        f'    noisy_v = true_v + np.random.normal(0, {sc["noise_scale_v"]})',
        f'    pp.create_measurement(net, "v", "bus", noisy_v, {sc["voltage_std_dev"]}, element=bus_idx)',
        "",
        "# Power injection measurements at all buses",
        "for bus_idx in in_service_buses:",
        f'    true_p = net.res_bus.at[bus_idx, "p_mw"]',
        f'    true_q = net.res_bus.at[bus_idx, "q_mvar"]',
        f'    pp.create_measurement(net, "p", "bus", true_p + np.random.normal(0, {sc["noise_scale_p"]}), {sc["power_bus_std_dev"]}, element=bus_idx)',
        f'    pp.create_measurement(net, "q", "bus", true_q + np.random.normal(0, {sc["noise_scale_q"]}), {sc["power_bus_std_dev"]}, element=bus_idx)',
        "",
        "# Step 3: Run WLS state estimation",
        "estimate(net, algorithm='wls')",
        "",
        "# Step 4: Extract result",
    ]
 
    if qt.qtype == "point_value":
        lines.append(f'result = net.{qt.table}.at[{qt.filter_idx}, "{qt.column}"]')
    elif qt.qtype == "max_value":
        lines.append(f'result = net.{qt.table}["{qt.column}"].max()')
    elif qt.qtype == "min_value":
        lines.append(f'result = net.{qt.table}["{qt.column}"].min()')
 
    lines.append("print(result)")
    return "\n".join(lines)
 
 
# ---------------------------------------------------------------------------
# 7. RESULT EXTRACTION (standard tasks)
# ---------------------------------------------------------------------------
 
def extract_result(net, qt: QueryTarget):
    table = getattr(net, qt.table)
    col = table[qt.column]
    extractors = {
        "point_value":      lambda: round(float(col.at[qt.filter_idx]), 6),
        "max_value":        lambda: round(float(col.max()), 6),
        "min_value":        lambda: round(float(col.min()), 6),
        "argmax":           lambda: int(col.idxmax()),
        "argmin":           lambda: int(col.idxmin()),
        "total":            lambda: round(float(col.sum()), 6),
        "count_violations": lambda: int((col > qt.threshold).sum() if qt.operator == ">"
                                        else (col < qt.threshold).sum() if qt.operator == "<"
                                        else (col >= qt.threshold).sum() if qt.operator == ">="
                                        else (col <= qt.threshold).sum()),
        "bool_check":       lambda: bool((col > qt.threshold).any() if qt.operator == ">"
                                         else (col < qt.threshold).any()),
        "full_table":       lambda: {int(k): round(float(v), 6) for k, v in col.items()},
    }
    return extractors[qt.qtype]()
 
 
# ---------------------------------------------------------------------------
# 8. UNIFIED REFERENCE CODE GENERATION
# ---------------------------------------------------------------------------
 
def generate_reference_code(scenario: Scenario) -> str:
    if scenario.task == "time_series":
        return _generate_ts_reference_code(scenario)
    elif scenario.task == "state_estimation":
        return _generate_se_reference_code(scenario)
 
    # Standard tasks
    tc = TASK_CONFIG[scenario.task]
    reg = _all_registries()
    net_info = reg[scenario.network]
 
    lines = ["import pandapower as pp", "import pandapower.networks as pn"]
    if scenario.network.startswith("sb_"):
        lines.append("import simbench as sb")
    for imp in tc.get("imports", []):
        lines.append(imp)
    lines.append("")
 
    lines.append(f"# Load {net_info['display']}")
    lines.append(f"net = {net_info['loader']}")
    lines.append("")
 
    if scenario.modifications:
        lines.append("# Apply modifications")
        for mod in scenario.modifications:
            lines.append(mod_to_code(mod))
        lines.append("")
 
    lines.append("# Run analysis")
    for s in tc.get("setup_code", []):
        lines.append(s)
    for r in tc["run_code"].split("\n"):
        lines.append(r)
    lines.append("")
 
    if scenario.query_target:
        qt = scenario.query_target
        lines.append("# Extract result")
        tmpl = EXTRACT_CODE_TEMPLATES.get(qt.qtype, "# unknown")
        lines.append(tmpl.format(**qt.to_dict()))
        lines.append("print(result)")
 
    return "\n".join(lines)
 
 
# ---------------------------------------------------------------------------
# 9. UNIFIED EXECUTION
# ---------------------------------------------------------------------------
 
class _SuppressPandapowerLogs:
    """Context manager to suppress pandapower's internal ERROR/WARNING logs
    and stderr traceback noise (using OS-level fd redirect, not sys.stderr)."""
    def __init__(self):
        self._saved = []
        self._old_stderr_fd = None
        self._devnull_fd = None
    def __enter__(self):
        # Logger-level suppression
        for name, lg in logging.Logger.manager.loggerDict.items():
            if "pandapower" in name and isinstance(lg, logging.Logger):
                self._saved.append((lg, lg.level))
                lg.setLevel(logging.CRITICAL)
        root = logging.getLogger("pandapower")
        self._saved.append((root, root.level))
        root.setLevel(logging.CRITICAL)
        # OS-level stderr redirect (catches traceback.print_exc)
        self._old_stderr_fd = os.dup(2)
        self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull_fd, 2)
        return self
    def __exit__(self, *args):
        # Restore stderr
        if self._old_stderr_fd is not None:
            os.dup2(self._old_stderr_fd, 2)
            os.close(self._old_stderr_fd)
        if self._devnull_fd is not None:
            os.close(self._devnull_fd)
        # Restore loggers
        for lg, old_level in self._saved:
            lg.setLevel(old_level)
 
 
def _is_valid_gt(gt) -> bool:
    """Check if ground truth is a valid, non-NaN, non-Inf value."""
    import math
    if gt is None:
        return False
    if isinstance(gt, float):
        return not (math.isnan(gt) or math.isinf(gt))
    if isinstance(gt, dict):
        return all(_is_valid_gt(v) for v in gt.values())
    return True  # int, bool, etc.
 
 
def execute_scenario(scenario: Scenario):
    """Execute any scenario. Returns (ground_truth, type_str).
    Raises ValueError if result is NaN/Inf/None."""
    net = _load_network(scenario.network)
    for mod in scenario.modifications:
        apply_modification(net, mod)
 
    qt = scenario.query_target
    seed = qt.se_seed or 42
 
    with _SuppressPandapowerLogs():
        if scenario.task == "time_series":
            _, gt = _execute_time_series(net, qt, seed=seed)
        elif scenario.task == "state_estimation":
            gt = _execute_state_estimation(net, qt, seed=seed)
        else:
            run_task(net, scenario.task, scenario.task_options)
            gt = extract_result(net, qt)
 
    # Guard: reject NaN / Inf / None
    if not _is_valid_gt(gt):
        raise ValueError(f"Invalid ground truth: {gt} (NaN/Inf/None)")
 
    if isinstance(gt, bool):
        return gt, "bool"
    elif isinstance(gt, int):
        return gt, "int"
    elif isinstance(gt, float):
        return gt, "float"
    elif isinstance(gt, dict):
        return gt, "dict"
    return gt, type(gt).__name__
 
 
# ---------------------------------------------------------------------------
# 10. NL QUERY GENERATION
# ---------------------------------------------------------------------------

def normalize_nl_query_text(text: str) -> str:
    """Clean up common mechanical template-composition artifacts."""
    if not text:
        return text

    replacements = [
        (r"\bbegin by identify\b", "begin by identifying"),
        (r"\bbegin by locate\b", "begin by locating"),
        (r"\bbegin by find\b", "begin by finding"),
        (r"\bperform identify\b", "identify"),
        (r"\bperform locate\b", "locate"),
        (r"\bperform find\b", "find"),
        (r"\bconduct identify\b", "identify"),
        (r"\bconduct locate\b", "locate"),
        (r"\bconduct find\b", "find"),
        (r"\bimplement identify\b", "identify"),
        (r"\bimplement locate\b", "locate"),
        (r"\bimplement find\b", "find"),
        (r"\bproceed with identify\b", "proceed to identify"),
        (r"\bproceed with locate\b", "proceed to locate"),
        (r"\bproceed with find\b", "proceed to find"),
        (r"\blet's identify\b", "Let's identify"),
        (r"\blet's locate\b", "Let's locate"),
        (r"\blet's find\b", "Let's find"),
        (r"\bafter set\b", "after you set"),
        (r"\bafter add\b", "after you add"),
        (r"\bafter disconnect\b", "after you disconnect"),
        (r"\bafter take\b", "after you take"),
        (r"\bafter scale\b", "after you scale"),
        (r"\bonce set\b", "once you set"),
        (r"\bonce add\b", "once you add"),
        (r"\bonce disconnect\b", "once you disconnect"),
        (r"\bonce take\b", "once you take"),
        (r"\bonce scale\b", "once you scale"),
        (r"\bfollowing set\b", "then set"),
        (r"\bfollowing add\b", "then add"),
        (r"\bfollowing disconnect\b", "then disconnect"),
        (r"\bfollowing take\b", "then take"),
        (r"\bfollowing scale\b", "then scale"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    text = re.sub(r"\busing\s+for\s+", "for ", text, flags=re.IGNORECASE)
    text = re.sub(r"\busing\s+,?\s*on\s+the\b", "on the", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(carry out|consider|use)\s+(Locate|Find|Identify)\b",
        lambda m: f"{m.group(1)} the following setup: {m.group(2).lower()}",
        text,
        flags=re.IGNORECASE,
    )

    def _repair_using_the_mod(match):
        verb = match.group(1)
        return "using the following setup: " + verb.lower()

    text = re.sub(
        r"\busing the\s+(Add|Set|Disconnect|Open|Close|Take|Scale)\b",
        _repair_using_the_mod,
        text,
    )

    text = re.sub(r"\s+([,.;?!])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\.\s*\.", ".", text)
    return text.strip()
 
def generate_nl_query(scenario: Scenario, rng: random.Random = None) -> str:
    if rng is None:
        rng = random.Random()
 
    reg = _all_registries()
    net_info = reg[scenario.network]
    qt = scenario.query_target
 
    if scenario.modifications:
        descs = [describe_modification(m) for m in scenario.modifications]
        joined = ", ".join(descs)
        mod_desc = joined[0].upper() + joined[1:] + ". "
        mod_desc_lower = joined + ", "
    else:
        mod_desc, mod_desc_lower = "", ""
 
    key = (scenario.task, qt.qtype, qt.table, qt.column)
    all_templates = _get_merged_templates()
    templates = all_templates.get(key)
    if templates is None:
        key_no_col = (scenario.task, qt.qtype, qt.table, None)
        templates = all_templates.get(key_no_col)
    if templates is None:
        return (f"On the {net_info['display']}, {mod_desc_lower}"
                f"run {scenario.task.replace('_', ' ')} and report "
                f"{qt.column} from {qt.table} ({qt.qtype}).")
 
    # Build format kwargs with all possible fields
    ts_cfg = TIME_SERIES_CONFIG
    se_cfg = STATE_ESTIMATION_CONFIG
    lo, hi = qt.load_scaling_range or ts_cfg.get("load_scaling_range", (0.6, 1.4))
 
    fmt = dict(
        network_display=net_info["display"],
        mod_description=mod_desc,
        mod_description_lower=mod_desc_lower,
        filter_idx=qt.filter_idx if qt.filter_idx is not None else "",
        threshold=qt.threshold if qt.threshold is not None else "",
        column=qt.column or "",
        n_steps=qt.n_steps or ts_cfg.get("n_steps", 24),
        load_lo=lo, load_hi=hi,
        time_step=qt.time_step if qt.time_step is not None else "",
        v_std=se_cfg.get("voltage_std_dev", 0.01),
        p_std=se_cfg.get("power_bus_std_dev", 1.0),
        se_seed=qt.se_seed or 42,
    )
    return normalize_nl_query_text(rng.choice(templates).format(**fmt))


def generate_semantic_nl_query(scenario: Scenario, scenario_phrase: str,
                               rng: random.Random = None) -> str:
    """
    Generate a D3-style query by reusing D1 templates while replacing explicit
    modification text with a rule-grounded semantic phrase.
    """
    if rng is None:
        rng = random.Random()

    reg = _all_registries()
    net_info = reg[scenario.network]
    qt = scenario.query_target

    phrase = (scenario_phrase or "").strip()
    if phrase:
        mod_desc = phrase[0].upper() + phrase[1:] + ". "
        mod_desc_lower = phrase + ", "
    else:
        mod_desc, mod_desc_lower = "", ""

    key = (scenario.task, qt.qtype, qt.table, qt.column)
    templates = _get_merged_templates().get(key)
    if templates is None:
        key_no_col = (scenario.task, qt.qtype, qt.table, None)
        templates = _get_merged_templates().get(key_no_col)
    if templates is None:
        return normalize_nl_query_text(
            f"On the {net_info['display']}, {mod_desc_lower}"
            f"run {scenario.task.replace('_', ' ')} and report "
            f"{qt.column} from {qt.table} ({qt.qtype})."
        )

    ts_cfg = TIME_SERIES_CONFIG
    se_cfg = STATE_ESTIMATION_CONFIG
    lo, hi = qt.load_scaling_range or ts_cfg.get("load_scaling_range", (0.6, 1.4))
    fmt = dict(
        network_display=net_info["display"],
        mod_description=mod_desc,
        mod_description_lower=mod_desc_lower,
        filter_idx=qt.filter_idx if qt.filter_idx is not None else "",
        threshold=qt.threshold if qt.threshold is not None else "",
        column=qt.column or "",
        n_steps=qt.n_steps or ts_cfg.get("n_steps", 24),
        load_lo=lo,
        load_hi=hi,
        time_step=qt.time_step if qt.time_step is not None else "",
        v_std=se_cfg.get("voltage_std_dev", 0.01),
        p_std=se_cfg.get("power_bus_std_dev", 1.0),
        se_seed=qt.se_seed or 42,
    )
    return normalize_nl_query_text(rng.choice(templates).format(**fmt))
 
 
# ---------------------------------------------------------------------------
# 11. SAMPLING
# ---------------------------------------------------------------------------
 
def get_net_metadata(network_name):
    net = _load_network(network_name)
    return {
        "n_bus": len(net.bus), "n_line": len(net.line), "n_trafo": len(net.trafo),
        "n_gen": len(net.gen), "n_load": len(net.load), "n_sgen": len(net.sgen),
        "n_ext_grid": len(net.ext_grid), "n_switch": len(net.switch),
        "bus_indices": net.bus.index.tolist(),
        "line_indices": net.line.index.tolist(),
        "trafo_indices": net.trafo.index.tolist(),
        "gen_indices": net.gen.index.tolist(),
        "load_indices": net.load.index.tolist(),
        "has_cost": (len(getattr(net, "poly_cost", [])) > 0
                     or len(getattr(net, "pwl_cost", [])) > 0),
    }
 
def sample_modifications(meta, rng):
    """Sample random modifications. All MODIFICATION_DEFS ops are reachable."""
    sc = SAMPLING_CONFIG
    n_mods = rng.choices([0, 1, 2, 3], weights=sc["n_mods_weights"])[0]
    mods, pool = [], []
    if meta["n_load"] > 0: pool.append("load_change")
    if meta["n_line"] > 1: pool.append("disconnect_line")
    if meta["n_line"] > 0: pool.append("line_param")
    if meta["n_gen"] > 0: pool.append("gen_change")
    if meta["n_ext_grid"] > 0: pool.append("ext_grid_vm")
    if meta["n_trafo"] > 0: pool.append("trafo_change")
    if meta["n_bus"] > 2: pool.append("add_element")
    if meta["n_bus"] > 2: pool.append("bus_limit")
    avail_lines = list(meta["line_indices"])
 
    for _ in range(n_mods):
        if not pool: break
        cat = rng.choice(pool)
 
        if cat == "load_change":
            idx = rng.choice(meta["load_indices"])
            # Mix of scaling vs absolute value changes
            choice = rng.choice(["set_load_scaling", "set_load_p", "set_load_q"])
            if choice == "set_load_scaling":
                lo, hi = sc["load_scaling_range"]
                mods.append(Modification("set_load_scaling", {"idx": idx, "value": round(rng.uniform(lo, hi), 2)}))
            elif choice == "set_load_p":
                lo, hi = sc["load_p_range"]
                mods.append(Modification("set_load_p", {"idx": idx, "value": round(rng.uniform(lo, hi), 1)}))
            else:
                lo, hi = sc["load_q_range"]
                mods.append(Modification("set_load_q", {"idx": idx, "value": round(rng.uniform(lo, hi), 1)}))
 
        elif cat == "disconnect_line":
            idx = rng.choice(avail_lines)
            mods.append(Modification("disconnect_line", {"idx": idx}))
            avail_lines = [i for i in avail_lines if i != idx]
            if len(avail_lines) <= 1: pool.remove("disconnect_line")
 
        elif cat == "line_param":
            idx = rng.choice(meta["line_indices"])
            choice = rng.choice(["set_line_r", "set_line_x", "set_line_parallel"])
            if choice == "set_line_r":
                lo, hi = sc["line_r_range"]
                mods.append(Modification("set_line_r", {"idx": idx, "value": round(rng.uniform(lo, hi), 3)}))
            elif choice == "set_line_x":
                lo, hi = sc["line_x_range"]
                mods.append(Modification("set_line_x", {"idx": idx, "value": round(rng.uniform(lo, hi), 3)}))
            else:
                mods.append(Modification("set_line_parallel", {"idx": idx, "value": rng.choice([2, 3])}))
 
        elif cat == "gen_change":
            idx = rng.choice(meta["gen_indices"])
            choice = rng.choice(["set_gen_vm", "set_gen_p", "set_gen_out_of_service",
                                  "set_gen_max_p", "set_gen_min_p"])
            if choice == "set_gen_vm":
                lo, hi = sc["gen_vm_range"]
                mods.append(Modification("set_gen_vm", {"idx": idx, "value": round(rng.uniform(lo, hi), 3)}))
            elif choice == "set_gen_p":
                lo, hi = sc["gen_p_range"]
                mods.append(Modification("set_gen_p", {"idx": idx, "value": round(rng.uniform(lo, hi), 1)}))
            elif choice == "set_gen_out_of_service":
                # Only if there are at least 2 gens (don't take out the only one)
                if meta["n_gen"] > 1:
                    mods.append(Modification("set_gen_out_of_service", {"idx": idx}))
            elif choice == "set_gen_max_p":
                lo, hi = sc["gen_p_range"]
                mods.append(Modification("set_gen_max_p", {"idx": idx, "value": round(rng.uniform(lo, hi), 1)}))
            elif choice == "set_gen_min_p":
                mods.append(Modification("set_gen_min_p", {"idx": idx, "value": round(rng.uniform(0, 20), 1)}))
 
        elif cat == "ext_grid_vm":
            lo, hi = sc["ext_grid_vm_range"]
            mods.append(Modification("set_ext_grid_vm", {"idx": 0, "value": round(rng.uniform(lo, hi), 3)}))
 
        elif cat == "trafo_change":
            idx = rng.choice(meta["trafo_indices"])
            choice = rng.choice(["set_trafo_tap_pos", "set_trafo_parallel", "set_trafo_shift"])
            if choice == "set_trafo_tap_pos":
                mods.append(Modification("set_trafo_tap_pos", {"idx": idx, "value": rng.choice([-2, -1, 0, 1, 2])}))
            elif choice == "set_trafo_parallel":
                mods.append(Modification("set_trafo_parallel", {"idx": idx, "value": 2}))
            else:
                lo, hi = sc["trafo_shift_range"]
                mods.append(Modification("set_trafo_shift", {"idx": idx, "value": round(rng.uniform(lo, hi), 1)}))
 
        elif cat == "bus_limit":
            idx = rng.choice(meta["bus_indices"])
            choice = rng.choice(["set_bus_max_vm", "set_bus_min_vm"])
            if choice == "set_bus_max_vm":
                lo, hi = sc["bus_vm_max_range"]
                mods.append(Modification("set_bus_max_vm", {"idx": idx, "value": round(rng.uniform(lo, hi), 3)}))
            else:
                lo, hi = sc["bus_vm_min_range"]
                mods.append(Modification("set_bus_min_vm", {"idx": idx, "value": round(rng.uniform(lo, hi), 3)}))
 
        elif cat == "add_element":
            bus = rng.choice(meta["bus_indices"])
            elem = rng.choice(["add_load", "add_sgen", "add_gen"])
            if elem == "add_load":
                lo_p, hi_p = sc["new_load_p_range"]
                lo_q, hi_q = sc["new_load_q_range"]
                mods.append(Modification("add_load", {"bus": bus,
                    "p_mw": round(rng.uniform(lo_p, hi_p), 1),
                    "q_mvar": round(rng.uniform(lo_q, hi_q), 1)}))
            elif elem == "add_sgen":
                lo_p, hi_p = sc["new_sgen_p_range"]
                mods.append(Modification("add_sgen",
                    {"bus": bus, "p_mw": round(rng.uniform(lo_p, hi_p), 1), "q_mvar": 0}))
            else:
                lo_p, hi_p = sc["new_gen_p_range"]
                lo_v, hi_v = sc["new_gen_vm_range"]
                mods.append(Modification("add_gen",
                    {"bus": bus, "p_mw": round(rng.uniform(lo_p, hi_p), 1),
                     "vm_pu": round(rng.uniform(lo_v, hi_v), 3)}))
    return mods
 
 
def sample_query_target(task, meta, rng):
    sc = SAMPLING_CONFIG
    opts = []
 
    if task in ("power_flow", "dc_power_flow"):
        bus_idx = rng.choice(meta["bus_indices"])
        line_idx = rng.choice(meta["line_indices"]) if meta["n_line"] > 0 else None
        opts.append(QueryTarget("point_value", "res_bus", "vm_pu", filter_idx=bus_idx))
        opts.append(QueryTarget("point_value", "res_bus", "va_degree", filter_idx=bus_idx))
        if line_idx is not None:
            opts.append(QueryTarget("point_value", "res_line", "loading_percent", filter_idx=line_idx))
        if meta["n_trafo"] > 0:
            trafo_idx = rng.choice(meta["trafo_indices"])
            opts.append(QueryTarget("point_value", "res_trafo", "loading_percent", filter_idx=trafo_idx))
            opts.append(QueryTarget("max_value", "res_trafo", "loading_percent"))
        opts.append(QueryTarget("max_value", "res_line", "loading_percent"))
        opts.append(QueryTarget("min_value", "res_bus", "vm_pu"))
        opts.append(QueryTarget("argmax", "res_line", "loading_percent"))
        opts.append(QueryTarget("total", "res_line", "pl_mw"))
        opts.append(QueryTarget("point_value", "res_ext_grid", "p_mw", filter_idx=0))
        opts.append(QueryTarget("point_value", "res_ext_grid", "q_mvar", filter_idx=0))
        vt = rng.choice(sc["voltage_violation_thresholds"])
        opts.append(QueryTarget("count_violations", "res_bus", "vm_pu", threshold=vt, operator="<"))
        lt = rng.choice(sc["loading_violation_thresholds"])
        opts.append(QueryTarget("bool_check", "res_line", "loading_percent", threshold=lt, operator=">"))
        if task == "dc_power_flow":
            opts.append(QueryTarget("total", "res_line", "pl_mw"))
 
    elif task.startswith("short_circuit"):
        bus_idx = rng.choice(meta["bus_indices"])
        opts.append(QueryTarget("point_value", "res_bus_sc", "ikss_ka", filter_idx=bus_idx))
        opts.append(QueryTarget("max_value", "res_bus_sc", "ikss_ka"))
        opts.append(QueryTarget("min_value", "res_bus_sc", "ikss_ka"))
        opts.append(QueryTarget("argmax", "res_bus_sc", "ikss_ka"))
 
    elif task == "opf":
        if meta["n_gen"] > 0:
            gen_idx = rng.choice(meta["gen_indices"])
            opts.append(QueryTarget("point_value", "res_gen", "p_mw", filter_idx=gen_idx))
            opts.append(QueryTarget("point_value", "res_gen", "q_mvar", filter_idx=gen_idx))
            opts.append(QueryTarget("total", "res_gen", "p_mw"))
        opts.append(QueryTarget("min_value", "res_bus", "vm_pu"))
        opts.append(QueryTarget("max_value", "res_bus", "vm_pu"))
 
    elif task == "contingency":
        lt = rng.choice(sc["loading_violation_thresholds"])
        opts.append(QueryTarget("max_value", "res_line", "max_loading_percent"))
        opts.append(QueryTarget("bool_check", "res_line", "max_loading_percent",
                                threshold=lt, operator=">"))
        opts.append(QueryTarget("min_value", "res_bus", "min_vm_pu"))
 
    elif task == "time_series":
        ts_cfg = TIME_SERIES_CONFIG
        n_steps = ts_cfg["n_steps"]
        lo, hi = ts_cfg["load_scaling_range"]
        se_seed = rng.randint(1, 9999)
        bus_idx = rng.choice(meta["bus_indices"])
 
        opts.append(QueryTarget("ts_min_value", "res_bus", "vm_pu",
                                n_steps=n_steps, load_scaling_range=(lo, hi), se_seed=se_seed))
        opts.append(QueryTarget("ts_max_value", "res_line", "loading_percent",
                                n_steps=n_steps, load_scaling_range=(lo, hi), se_seed=se_seed))
        step = rng.randint(0, n_steps - 1)
        opts.append(QueryTarget("ts_point_at_step", "res_bus", "vm_pu",
                                filter_idx=bus_idx, time_step=step,
                                n_steps=n_steps, load_scaling_range=(lo, hi), se_seed=se_seed))
 
    elif task == "state_estimation":
        bus_idx = rng.choice(meta["bus_indices"])
        se_seed = rng.randint(1, 9999)
        opts.append(QueryTarget("point_value", "res_bus_est", "vm_pu",
                                filter_idx=bus_idx, se_seed=se_seed))
        opts.append(QueryTarget("max_value", "res_bus_est", "vm_pu", se_seed=se_seed))
        opts.append(QueryTarget("min_value", "res_bus_est", "vm_pu", se_seed=se_seed))
 
    if not opts:
        raise ValueError(f"No query targets for task: {task}")
    return rng.choice(opts)


def sample_semantic_query_target(task, meta, rng):
    """Task-family-specific query targets for D3 semantic items."""
    opts = []

    if task == "power_flow":
        if meta["n_line"] > 0:
            line_idx = rng.choice(meta["line_indices"])
            opts.append(QueryTarget("point_value", "res_line", "loading_percent", filter_idx=line_idx))
            opts.append(QueryTarget("max_value", "res_line", "loading_percent"))
            opts.append(QueryTarget("argmax", "res_line", "loading_percent"))
            opts.append(QueryTarget("total", "res_line", "pl_mw"))
        if meta["n_trafo"] > 0:
            trafo_idx = rng.choice(meta["trafo_indices"])
            opts.append(QueryTarget("point_value", "res_trafo", "loading_percent", filter_idx=trafo_idx))
            opts.append(QueryTarget("max_value", "res_trafo", "loading_percent"))
        opts.append(QueryTarget("min_value", "res_bus", "vm_pu"))
        opts.append(QueryTarget("point_value", "res_ext_grid", "p_mw", filter_idx=0))

    elif task == "dc_power_flow":
        bus_idx = rng.choice(meta["bus_indices"])
        opts.append(QueryTarget("point_value", "res_bus", "va_degree", filter_idx=bus_idx))
        if meta["n_line"] > 0:
            line_idx = rng.choice(meta["line_indices"])
            opts.append(QueryTarget("point_value", "res_line", "loading_percent", filter_idx=line_idx))
            opts.append(QueryTarget("max_value", "res_line", "loading_percent"))
            opts.append(QueryTarget("argmax", "res_line", "loading_percent"))
            opts.append(QueryTarget("total", "res_line", "pl_mw"))
        if meta["n_trafo"] > 0:
            trafo_idx = rng.choice(meta["trafo_indices"])
            opts.append(QueryTarget("point_value", "res_trafo", "loading_percent", filter_idx=trafo_idx))
            opts.append(QueryTarget("max_value", "res_trafo", "loading_percent"))
        opts.append(QueryTarget("point_value", "res_ext_grid", "p_mw", filter_idx=0))

    elif task.startswith("short_circuit"):
        bus_idx = rng.choice(meta["bus_indices"])
        opts.append(QueryTarget("point_value", "res_bus_sc", "ikss_ka", filter_idx=bus_idx))
        opts.append(QueryTarget("max_value", "res_bus_sc", "ikss_ka"))
        opts.append(QueryTarget("min_value", "res_bus_sc", "ikss_ka"))
        opts.append(QueryTarget("argmax", "res_bus_sc", "ikss_ka"))

    elif task == "opf":
        if meta["n_gen"] > 0:
            gen_idx = rng.choice(meta["gen_indices"])
            opts.append(QueryTarget("point_value", "res_gen", "p_mw", filter_idx=gen_idx))
            opts.append(QueryTarget("point_value", "res_gen", "q_mvar", filter_idx=gen_idx))
            opts.append(QueryTarget("total", "res_gen", "p_mw"))
        opts.append(QueryTarget("min_value", "res_bus", "vm_pu"))
        opts.append(QueryTarget("max_value", "res_bus", "vm_pu"))

    elif task == "contingency":
        opts.append(QueryTarget("max_value", "res_line", "max_loading_percent"))
        opts.append(QueryTarget("min_value", "res_bus", "min_vm_pu"))
        lt = rng.choice(SAMPLING_CONFIG["loading_violation_thresholds"])
        opts.append(QueryTarget("bool_check", "res_line", "max_loading_percent",
                                threshold=lt, operator=">"))

    elif task == "time_series":
        ts_cfg = TIME_SERIES_CONFIG
        n_steps = ts_cfg["n_steps"]
        lo, hi = ts_cfg["load_scaling_range"]
        se_seed = rng.randint(1, 9999)
        bus_idx = rng.choice(meta["bus_indices"])
        opts.append(QueryTarget("ts_min_value", "res_bus", "vm_pu",
                                n_steps=n_steps, load_scaling_range=(lo, hi), se_seed=se_seed))
        if meta["n_line"] > 0:
            opts.append(QueryTarget("ts_max_value", "res_line", "loading_percent",
                                    n_steps=n_steps, load_scaling_range=(lo, hi), se_seed=se_seed))
        step = rng.randint(0, n_steps - 1)
        opts.append(QueryTarget("ts_point_at_step", "res_bus", "vm_pu",
                                filter_idx=bus_idx, time_step=step,
                                n_steps=n_steps, load_scaling_range=(lo, hi), se_seed=se_seed))

    elif task == "state_estimation":
        bus_idx = rng.choice(meta["bus_indices"])
        se_seed = rng.randint(1, 9999)
        opts.append(QueryTarget("point_value", "res_bus_est", "vm_pu",
                                filter_idx=bus_idx, se_seed=se_seed))
        opts.append(QueryTarget("max_value", "res_bus_est", "vm_pu", se_seed=se_seed))
        opts.append(QueryTarget("min_value", "res_bus_est", "vm_pu", se_seed=se_seed))

    if not opts:
        return sample_query_target(task, meta, rng)
    return rng.choice(opts)
 
 
# ---------------------------------------------------------------------------
# 12. CONSISTENCY VALIDATION
# ---------------------------------------------------------------------------
 
def validate_item(item: BenchmarkItem, tolerance: float = 1e-4) -> dict:
    """
    Validate that reference_code produces the same result as ground_truth.
    Executes reference_code in isolated namespace, captures print output.
    """
    import warnings
    warnings.filterwarnings("ignore")
 
    code = item.reference_code
    gt = item.ground_truth

    # D3/D4 semantic items are evaluated behaviorally and may not have
    # executable reference code for exact-match consistency checks.
    if not code or not code.strip():
        return {
            "match": False,
            "code_output": None,
            "ground_truth": gt,
            "error": "No reference code available for exact-match validation",
            "skipped": True,
        }
 
    stdout_capture = io.StringIO()
    ns = {}
    try:
        with _SuppressPandapowerLogs(), contextlib.redirect_stdout(stdout_capture):
            exec(code, ns)
        code_output = stdout_capture.getvalue().strip()
    except Exception as e:
        return {"match": False, "code_output": None,
                "ground_truth": gt, "error": f"{type(e).__name__}: {e}"}
 
    try:
        if item.ground_truth_type == "float":
            import math
            code_val = float(code_output)
            if math.isnan(code_val) and math.isnan(gt):
                match = True
            else:
                match = abs(code_val - gt) < tolerance
        elif item.ground_truth_type == "int":
            code_val = int(float(code_output))
            match = code_val == gt
        elif item.ground_truth_type == "bool":
            code_val = code_output.strip() in ("True", "true", "1")
            match = code_val == gt
        else:
            match = str(code_output) == str(gt)
    except Exception as e:
        return {"match": False, "code_output": code_output,
                "ground_truth": gt, "error": f"Comparison failed: {e}"}
 
    return {"match": match, "code_output": code_output,
            "ground_truth": gt, "error": None}
 
 
def validate_batch(items: list, tolerance: float = 1e-4,
                   max_items: int = None, verbose: bool = True) -> dict:
    """Validate a batch of items with progress display. Returns summary dict."""
    to_check = items[:max_items] if max_items else items
    results = []
    n_total = len(to_check)
    n_skipped = 0
    if verbose:
        print(f"Validating {n_total} items...")
 
    with _SuppressPandapowerLogs():
        for i, item in enumerate(to_check):
            r = validate_item(item, tolerance)
            r["index"] = i
            r["task"] = item.scenario.task
            r["network"] = item.scenario.network
            r["difficulty_level"] = item.difficulty_level
            results.append(r)
            if r.get("skipped"):
                n_skipped += 1
                if verbose:
                    print(f"  SKIP [{i}] {r['difficulty_level']}/{r['task']}/{r['network']}: "
                          f"{r['error']}")
                continue
            if verbose and not r["match"]:
                print(f"  FAIL [{i}] {r['task']}/{r['network']}: "
                      f"gt={r['ground_truth']}, out={repr(r['code_output'])[:50]}, "
                      f"err={r['error']}")
            if verbose and (i + 1) % 20 == 0:
                ok = sum(1 for r2 in results if r2["match"])
                print(f"  ... {i+1}/{n_total} checked ({ok} ok, {n_skipped} skipped)")
 
    n_match = sum(1 for r in results if r["match"])
    n_error = sum(1 for r in results if r["error"] and not r.get("skipped"))
    mismatches = [r for r in results if not r["match"] and not r["error"] and not r.get("skipped")]
    n_effective = n_total - n_skipped
 
    if verbose:
        if n_effective > 0:
            print(f"Consistency: {n_match}/{n_effective} ({n_match/n_effective*100:.0f}%)")
        else:
            print("No exact-match items to validate")
 
    return {
        "total": n_total, "match": n_match,
        "effective_total": n_effective,
        "skipped": n_skipped,
        "mismatch": len(mismatches), "error": n_error,
        "consistency_rate": n_match / n_effective if n_effective else 0,
        "details": results,
    }
 
 
# ---------------------------------------------------------------------------
# 13. NL TEMPLATE VALIDATION
# ---------------------------------------------------------------------------
 
def validate_templates(templates_dict: dict = None) -> dict:
    """
    Validate NL templates: check all placeholders are present and format() works.
 
    Args:
        templates_dict: dict to validate. Defaults to NL_TEMPLATES from config.
 
    Returns:
        {"valid": int, "invalid": int, "errors": list}
    """
    if templates_dict is None:
        templates_dict = NL_TEMPLATES
 
    errors = []
    valid = 0
 
    for key, templates in templates_dict.items():
        for i, tmpl in enumerate(templates):
            err = _template_validation_error("single", key, tmpl)
            if err is None:
                valid += 1
            else:
                errors.append({
                    "key": key, "index": i, "template": tmpl[:80],
                    "error": err,
                })
 
    return {"valid": valid, "invalid": len(errors), "errors": errors}


def _validate_templates_with_required_fields(templates_dict: dict, required_fields: tuple) -> dict:
    """Validate templates using required placeholder presence + format() success."""
    errors = []
    valid = 0

    for key, templates in templates_dict.items():
        for i, tmpl in enumerate(templates):
            missing = [field for field in required_fields if field not in tmpl]
            if missing:
                errors.append({
                    "key": key, "index": i, "template": tmpl[:80],
                    "error": f"Missing required placeholders: {', '.join(missing)}",
                })
                continue
            try:
                tmpl.format(**_template_dummy_kwargs())
                valid += 1
            except (KeyError, IndexError, ValueError) as e:
                errors.append({
                    "key": key, "index": i, "template": tmpl[:80],
                    "error": f"format() failed: {e}",
                })

    return {"valid": valid, "invalid": len(errors), "errors": errors}


def validate_semantic_templates(templates_dict: dict = None) -> dict:
    """Validate D3 semantic templates."""
    return _validate_templates_with_required_fields(
        templates_dict or SEMANTIC_NL_TEMPLATES,
        required_fields=("{network_display}", "{scenario_phrase}"),
    )


def validate_compound_templates(templates_dict: dict = None) -> dict:
    """Validate D4 compound templates with type-specific placeholders."""
    if templates_dict is None:
        templates_dict = COMPOUND_NL_TEMPLATES

    errors = []
    valid = 0
    requirements = {
        "semantic_then_check": ("{network_display}", "{scenario_phrase}", "{threshold}"),
        "semantic_then_compensate": ("{network_display}", "{scenario_phrase}", "{v_threshold}", "{shunt_mvar}"),
        "semantic_comparison": ("{network_display}", "{scenario_phrase}", "{metric_name}"),
        "semantic_then_redispatch": (
            "{network_display}", "{scenario_phrase}", "{v_threshold}", "{redispatch_mw}",
        ),
        "two_stage_rank_then_act": ("{network_display}", "{scenario_phrase}"),
        "dual_condition_choose_action": (
            "{network_display}", "{scenario_phrase}", "{threshold}", "{v_threshold}", "{shunt_mvar}",
        ),
    }

    for key, templates in templates_dict.items():
        vr = _validate_templates_with_required_fields(
            {key: templates},
            required_fields=requirements.get(key, ("{network_display}", "{scenario_phrase}")),
        )
        valid += vr["valid"]
        errors.extend(vr["errors"])

    return {"valid": valid, "invalid": len(errors), "errors": errors}
 
 
def get_net_metadata_from_net(net):
    """Extract metadata directly from a loaded net object."""
    return {
        "n_bus": len(net.bus), "n_line": len(net.line), "n_trafo": len(net.trafo),
        "n_gen": len(net.gen), "n_load": len(net.load), "n_sgen": len(net.sgen),
        "n_ext_grid": len(net.ext_grid), "n_switch": len(net.switch),
        "bus_indices": net.bus.index.tolist(),
        "line_indices": net.line.index.tolist(),
        "trafo_indices": net.trafo.index.tolist(),
        "gen_indices": net.gen.index.tolist(),
        "load_indices": net.load.index.tolist(),
    }


# ---------------------------------------------------------------------------
# 15. DIAGNOSTIC CHECK FUNCTIONS (for D1/D2 failure decomposition)
# ---------------------------------------------------------------------------

def check_task_function(code: str, task: str) -> bool:
    """Check if generated code calls the correct analysis function."""
    patterns = TASK_FUNCTION_PATTERNS.get(task, [])
    if not patterns:
        return True
    return any(p in code for p in patterns)


def check_network_loader(code: str, network: str) -> bool:
    """Check if generated code loads the correct network."""
    reg = _all_registries()
    info = reg.get(network)
    if not info:
        return True
    fn_name = info["loader"].split("(")[0].split(".")[-1]
    return fn_name in code


def check_modifications_applied(code: str, modifications: list) -> float:
    """Check what fraction of expected modifications are present in code.
    Returns: float in [0, 1]."""
    if not modifications:
        return 1.0
    found = 0
    for mod_dict in modifications:
        op = mod_dict.get("op", "")
        if op.startswith("set_") and "idx" in mod_dict:
            if str(mod_dict["idx"]) in code:
                found += 1
        elif op == "scale_all_loads":
            if "scaling" in code or "scale" in code.lower():
                found += 1
        elif op.startswith("disconnect_"):
            if "in_service" in code and "False" in code:
                found += 1
        elif op.startswith("add_"):
            element = op.replace("add_", "")
            if f"create_{element}" in code:
                found += 1
        elif op in ("open_switch", "close_switch"):
            if "closed" in code:
                found += 1
        else:
            found += 0.5
    return round(found / len(modifications), 4)


def diagnose_item(code: str, scenario_dict: dict) -> dict:
    """Full diagnostic decomposition for a D1/D2 benchmark item."""
    return {
        "correct_task_fn": check_task_function(code, scenario_dict.get("task", "")),
        "correct_network": check_network_loader(code, scenario_dict.get("network", "")),
        "modifications_applied": check_modifications_applied(
            code, scenario_dict.get("modifications", [])),
    }


def diagnostic_checks(code: str, task: str, network: str,
                      modifications: list = None) -> dict:
    """Compatibility wrapper used by probe_runner."""
    diag = diagnose_item(code, {
        "task": task,
        "network": network,
        "modifications": modifications or [],
    })
    return {
        "correct_task_fn": bool(diag["correct_task_fn"]),
        "correct_network": bool(diag["correct_network"]),
        "modifications_applied": diag["modifications_applied"] >= 1.0,
    }


def semantic_rewrite_checks(query: str, modifications: List[dict],
                            eval_criteria: Dict) -> dict:
    """
    Prompt-quality diagnostics for rule-grounded D3/D4 items.

    This is only for debugging generated benchmark quality; leaderboard scoring
    remains exact-match on execution outputs.
    """
    semantic_phrases = [
        p.strip().lower()
        for p in (eval_criteria or {}).get("semantic_phrases", [])
        if isinstance(p, str) and p.strip()
    ]
    q_lower = (query or "").lower()

    n_phrase_hits = sum(1 for p in semantic_phrases if p in q_lower)
    leak_scan_text = q_lower
    if semantic_phrases:
        focused = []
        for phrase in semantic_phrases:
            start = 0
            while True:
                pos = q_lower.find(phrase, start)
                if pos < 0:
                    break
                # Scan only the semantic setup span, not the whole sentence:
                # D3/D4 prompts often mention result targets before/after the
                # hidden setup phrase.
                suffix = q_lower[pos:]
                boundary = re.search(r"[,.;!?]", suffix)
                end = pos + boundary.start() if boundary else len(q_lower)
                focused.append(q_lower[pos:end])
                start = pos + len(phrase)
        if focused:
            leak_scan_text = " ".join(focused)

    leaked_tokens = []
    semantic_source = (eval_criteria or {}).get("semantic_source")
    mods_to_check = list(modifications or [])
    if semantic_source == "rule_grounded_d4" and mods_to_check:
        # D4 contains a hidden semantic setup followed by explicit multi-step
        # actions from the D2 generator. Only the hidden setup should count as
        # semantic leakage.
        mods_to_check = mods_to_check[:1]

    def _target_labels(op: str, key: str) -> list[str]:
        if key == "bus":
            return ["bus"]
        if key != "idx":
            return []
        if op.startswith("set_load"):
            return ["load"]
        if op.startswith("set_gen"):
            return ["gen", "generator"]
        if op.startswith("set_line") or op == "disconnect_line":
            return ["line"]
        if op.startswith("set_trafo") or op == "disconnect_trafo":
            return ["trafo", "transformer"]
        if op.startswith("set_bus"):
            return ["bus"]
        if op in ("open_switch", "close_switch"):
            return ["switch"]
        if op == "set_ext_grid_vm":
            return ["ext_grid", "external grid"]
        return []

    def _mentions_index(label: str, token: str) -> bool:
        label_pattern = re.escape(label).replace(r"\ ", r"\s+")
        return bool(re.search(
            rf"\b{label_pattern}(?:\s+(?:index|idx))?\s*{re.escape(token)}\b",
            leak_scan_text,
        ))

    for mod in mods_to_check:
        op = mod.get("op", "")
        params = {k: v for k, v in mod.items() if k != "op"}

        for key, value in params.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                token = str(value)
                for label in _target_labels(op, key):
                    if _mentions_index(label, token):
                        leaked_tokens.append(f"{key}={token}")
                        break
                # Bare integer values are too ambiguous in prompts that also
                # contain 2-phase/3-phase, thresholds, bus ids, and step counts.
            elif isinstance(value, float):
                token_patterns = sorted({
                    f"{value}",
                    f"{value:.2f}",
                    f"{value:.3f}",
                })
                for token in token_patterns:
                    if "." not in token:
                        continue
                    if token and re.search(rf"(?<![\d.]){re.escape(token)}(?![\d.])", leak_scan_text):
                        leaked_tokens.append(f"{key}={token}")

    seen = set()
    leaked_tokens = [t for t in leaked_tokens if not (t in seen or seen.add(t))]

    return {
        "semantic_source": (eval_criteria or {}).get("semantic_source"),
        "semantic_rule": (eval_criteria or {}).get("semantic_rule"),
        "has_semantic_phrase": n_phrase_hits > 0 if semantic_phrases else False,
        "n_semantic_phrase_hits": n_phrase_hits,
        "leaks_explicit_params": bool(leaked_tokens),
        "leaked_tokens": leaked_tokens[:12],
    }


# ---------------------------------------------------------------------------
# 17. BENCHMARK ENGINE
# ---------------------------------------------------------------------------
 
class BenchmarkEngine:
    def __init__(self, seed=42, networks=None, include_large=False, large_ratio=0.3):
        """
        Args:
            seed: random seed
            networks: explicit network list. If None, uses FAST_NETWORKS.
            include_large: if True and networks is None, appends LARGE_NETWORKS.
        """
        self.rng = random.Random(seed)
        if networks is not None:
            self.networks = networks
        elif include_large:
            n_f, n_l = len(FAST_NETWORKS), len(LARGE_NETWORKS)
            repeat = max(1, round(n_l * (1 - large_ratio) / (n_f * large_ratio)))
            self.networks = FAST_NETWORKS * repeat + LARGE_NETWORKS
        else:
            self.networks = FAST_NETWORKS
        self._meta_cache = {}
        self._semantic_balance = {
            "d3_task": Counter(),
            "d3_rule": Counter(),
            "d4_task": Counter(),
            "d4_rule": Counter(),
        }
 
    def _get_meta(self, name):
        if name not in self._meta_cache:
            self._meta_cache[name] = get_net_metadata(name)
        return copy.deepcopy(self._meta_cache[name])
 
    def _make_id(self, scenario):
        raw = json.dumps(scenario.to_dict(), sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _make_variant_id(self, item: BenchmarkItem, suffix: str, query: str) -> str:
        raw = json.dumps({
            "base_id": item.id,
            "suffix": suffix,
            "query": query,
        }, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()[:12]
 
    def generate_one(self, task=None, network=None):
        available_tasks = list(TASK_CONFIG.keys())
        if task is None:
            task = self.rng.choice(available_tasks)
 
        tc = TASK_CONFIG[task]
        task_filter = tc["filter"]
 
        if network is None:
            candidates = []
            for name in self.networks:
                try:
                    meta = self._get_meta(name)
                    if task_filter(meta):
                        candidates.append(name)
                except Exception:
                    continue
            if not candidates:
                raise ValueError(f"No suitable networks for task: {task}")
            network = self.rng.choice(candidates)
 
        meta = self._get_meta(network)
        mods = sample_modifications(meta, self.rng)
        qt = sample_query_target(task, meta, self.rng)
        scenario = Scenario(network=network, task=task, modifications=mods, query_target=qt)
 
        try:
            gt, gt_type = execute_scenario(scenario)
        except Exception as e:
            logger.info(f"Retry without mods ({network}/{task}): {e}")
            scenario.modifications = []
            gt, gt_type = execute_scenario(scenario)
 
        ref_code = generate_reference_code(scenario)
        nl_query = generate_nl_query(scenario, self.rng)
 
        return BenchmarkItem(
            id=self._make_id(scenario), scenario=scenario,
            natural_language_query=nl_query, reference_code=ref_code,
            ground_truth=gt, ground_truth_type=gt_type,
        )
 
    # -----------------------------------------------------------------
    # MULTI-TASK GENERATION
    # -----------------------------------------------------------------
 
    def _run_pf(self, network, modifications=None):
        """Helper: load net, apply mods, run PF, return net."""
        net = _load_network(network)
        for mod in (modifications or []):
            apply_modification(net, mod)
        with _SuppressPandapowerLogs():
            import pandapower as pp
            pp.runpp(net)
        return net
 
    def _build_mod_desc(self, mods):
        """Build mod_description / mod_description_lower from a list of Modifications."""
        if not mods:
            return "", ""
        descs = [describe_modification(m) for m in mods]
        j = ", ".join(descs)
        return j[0].upper() + j[1:] + ". ", j + ", "
 
    def _dominant_idxmax(self, series, min_gap_pct=5.0, min_abs_gap=1e-6):
        """Return idxmax only if the maximum is clearly separated from the runner-up."""
        if len(series) < 2:
            return int(series.idxmax())
        sorted_vals = series.sort_values(ascending=False)
        first, second = sorted_vals.iloc[0], sorted_vals.iloc[1]
        abs_gap = abs(first - second)
        rel_gap = abs_gap / max(abs(second), 1e-9) * 100
        if abs_gap < min_abs_gap and rel_gap < min_gap_pct:
            raise ValueError(f"Max not dominant: {first:.4f} vs {second:.4f}")
        return int(sorted_vals.index[0])

    def _dominant_idxmin(self, series, min_gap_pct=2.0, min_abs_gap=1e-4):
        """Return idxmin only if the minimum is clearly separated from the 2nd minimum."""
        if len(series) < 2:
            return int(series.idxmin())
        sorted_vals = series.sort_values(ascending=True)
        first, second = sorted_vals.iloc[0], sorted_vals.iloc[1]
        abs_gap = abs(second - first)
        rel_gap = abs_gap / max(abs(second), 1e-9) * 100
        if abs_gap < min_abs_gap and rel_gap < min_gap_pct:
            raise ValueError(f"Min not dominant: {first:.6f} vs {second:.6f}")
        return int(sorted_vals.index[0])

    def _sample_explicit_pf_action(self, meta, net_state):
        """Sample one explicit PF-side action while downweighting line outages."""
        actions, weights = [], []

        if meta["n_line"] > 1:
            worst_line = int(net_state.res_line["loading_percent"].idxmax())
            actions.append(Modification("disconnect_line", {"idx": worst_line}))
            weights.append(0.12)
            actions.append(Modification("set_line_parallel", {
                "idx": worst_line,
                "value": int(net_state.line.at[worst_line, "parallel"]) + 1,
            }))
            weights.append(0.26)

        if meta["n_load"] > 0:
            target_load = int(net_state.load["p_mw"].abs().idxmax())
            actions.append(Modification("set_load_scaling", {
                "idx": target_load,
                "value": round(self.rng.choice([0.65, 0.8, 1.25, 1.5]), 2),
            }))
            weights.append(0.26)

        if meta["n_gen"] > 0:
            online_gen = net_state.gen[net_state.gen["in_service"]]
            if len(online_gen) > 0:
                target_gen = int(online_gen["p_mw"].abs().idxmax())
                current_p = float(net_state.gen.at[target_gen, "p_mw"])
                factor = self.rng.choice([0.8, 1.2])
                new_p = round(current_p * factor, 1)
                if abs(new_p - current_p) < 1e-6:
                    new_p = round(current_p + (10.0 if factor > 1 else -10.0), 1)
                actions.append(Modification("set_gen_p", {
                    "idx": target_gen,
                    "value": new_p,
                }))
                weights.append(0.2)

        if meta["n_bus"] > 2:
            weak_bus = int(net_state.res_bus["vm_pu"].idxmin())
            shunt_mvar = round(self.rng.uniform(5.0, 20.0), 0)
            actions.append(Modification("add_shunt", {
                "bus": weak_bus,
                "q_mvar": -shunt_mvar,
                "p_mw": 0,
            }))
            weights.append(0.16)

        if not actions:
            raise ValueError("No explicit PF actions available")
        return self.rng.choices(actions, weights=weights, k=1)[0]
 
    def generate_comparison(self, network=None, base_mods=None,
                            mod_desc_override=None):
        """2-step: PF -> modify -> PF -> report diff."""
        if network is None:
            network = self.rng.choice(self.networks)
        meta = self._get_meta(network)
        reg = _all_registries()
        net_display = reg[network]["display"]
 
        base_mods = (
            list(base_mods) if base_mods is not None
            else sample_modifications(meta, self.rng)
        )
        net_base = self._run_pf(network, base_mods)
 
        metric_choices = [("res_bus", "vm_pu", "min", "minimum bus voltage (p.u.)")]
        if meta["n_line"] > 0:
            metric_choices.append(("res_line", "loading_percent", "max", "maximum line loading (%)"))
        table, col, agg, metric_name = self.rng.choice(metric_choices)
        base_val = float(getattr(net_base, table)[col].min() if agg == "min"
                         else getattr(net_base, table)[col].max())
 
        extra_mod = self._sample_explicit_pf_action(meta, net_base)
 
        net_mod = self._run_pf(network, base_mods + [extra_mod])
        mod_val = float(getattr(net_mod, table)[col].min() if agg == "min"
                        else getattr(net_mod, table)[col].max())
        gt = round(abs(base_val - mod_val), 6)
        if not _is_valid_gt(gt):
            raise ValueError(f"Invalid GT: {gt}")
 
        mod_desc, mod_desc_lower = (
            mod_desc_override
            if mod_desc_override is not None
            else self._build_mod_desc(base_mods)
        )
        nl = self.rng.choice(_get_merged_multi_templates()["comparison"]).format(
            network_display=net_display, mod_description=mod_desc,
            mod_description_lower=mod_desc_lower,
            baseline_metric=metric_name, baseline_value=round(base_val, 4),
            modification_desc=describe_modification(extra_mod))
 
        lines = ["import pandapower as pp", "import pandapower.networks as pn", "",
                 f"# Load {net_display}", f"net = {reg[network]['loader']}", ""]
        if base_mods:
            lines.append("# Baseline modifications")
            for m in base_mods: lines.append(mod_to_code(m))
            lines.append("")
        lines += ["# Step 1: baseline power flow", "pp.runpp(net)",
                  f'baseline = net.{table}["{col}"].{agg}()', "",
                  f"# Step 2: {describe_modification(extra_mod)}", mod_to_code(extra_mod), "",
                  "# Re-run power flow", "pp.runpp(net)",
                  f'modified = net.{table}["{col}"].{agg}()', "",
                  "result = abs(baseline - modified)", "print(result)"]
 
        scenario = Scenario(network=network, task="comparison",
                            modifications=base_mods + [extra_mod],
                            query_target=QueryTarget("comparison", table, col))
        return BenchmarkItem(id=self._make_id(scenario), scenario=scenario,
                             natural_language_query=nl, reference_code="\n".join(lines),
                             ground_truth=gt, ground_truth_type="float")
 
    def generate_comparison_opf(self, network=None, base_mods=None,
                                mod_desc_override=None):
        """2-step: OPF -> modify load -> OPF -> report cost diff."""
        if network is None:
            candidates = [n for n in self.networks if self._get_meta(n)["has_cost"]]
            if not candidates: raise ValueError("No OPF-capable networks")
            network = self.rng.choice(candidates)
        meta = self._get_meta(network)
        reg = _all_registries()
        net_display = reg[network]["display"]
 
        import pandapower as pp
        # Step 1: baseline OPF
        net1 = _load_network(network)
        for m in (base_mods or []):
            apply_modification(net1, m)
        with _SuppressPandapowerLogs():
            pp.runopp(net1)
        cost1 = float(net1.res_cost)
 
        # Step 2: modify a load, re-run OPF
        if meta["n_load"] == 0: raise ValueError("No loads")
        idx = self.rng.choice(meta["load_indices"])
        factor = round(self.rng.uniform(1.3, 1.8), 2)
        extra_mod = Modification("set_load_scaling", {"idx": idx, "value": factor})
 
        net2 = _load_network(network)
        for m in (base_mods or []):
            apply_modification(net2, m)
        apply_modification(net2, extra_mod)
        with _SuppressPandapowerLogs():
            pp.runopp(net2)
        cost2 = float(net2.res_cost)
 
        gt = round(abs(cost1 - cost2), 6)
        if not _is_valid_gt(gt): raise ValueError(f"Invalid GT: {gt}")
 
        mod_desc, mod_desc_lower = (
            mod_desc_override
            if mod_desc_override is not None
            else self._build_mod_desc(base_mods or [])
        )
        nl = self.rng.choice(_get_merged_multi_templates()["comparison_opf"]).format(
            network_display=net_display, mod_description=mod_desc,
            mod_description_lower=mod_desc_lower,
            modification_desc=describe_modification(extra_mod))
 
        lines = ["import pandapower as pp", "import pandapower.networks as pn", "",
                 f"net = {reg[network]['loader']}", "",
                 ]
        if base_mods:
            lines.append("# Baseline modifications")
            for m in base_mods:
                lines.append(mod_to_code(m))
            lines.append("")
        lines += ["# Step 1: baseline OPF", "pp.runopp(net)",
                 "cost_before = net.res_cost", "",
                 f"# Step 2: {describe_modification(extra_mod)}",
                 f"net2 = {reg[network]['loader']}",
                 ]
        if base_mods:
            lines.append("# Re-apply baseline modifications")
            for m in base_mods:
                lines.append(mod_to_code(m).replace("net.", "net2."))
        lines += [mod_to_code(extra_mod).replace("net.", "net2."), "",
                  "pp.runopp(net2)", "cost_after = net2.res_cost", "",
                 "result = abs(cost_before - cost_after)", "print(result)"]
 
        scenario = Scenario(network=network, task="comparison_opf",
                            modifications=list(base_mods or []) + [extra_mod],
                            query_target=QueryTarget("comparison_opf", "res_cost", "total"))
        return BenchmarkItem(id=self._make_id(scenario), scenario=scenario,
                             natural_language_query=nl, reference_code="\n".join(lines),
                             ground_truth=gt, ground_truth_type="float")
 
    def generate_sequential(self, network=None, base_mods=None,
                            mod_desc_override=None):
        """2-step: PF -> check condition -> act -> PF. Uses dominance guard."""
        if network is None:
            candidates = [n for n in self.networks if self._get_meta(n)["n_line"] >= 3]
            if not candidates: raise ValueError("No networks with enough lines")
            network = self.rng.choice(candidates)
        meta = self._get_meta(network)
        reg = _all_registries()
        net_display = reg[network]["display"]
 
        base_mods = (
            list(base_mods) if base_mods is not None
            else sample_modifications(meta, self.rng)
        )
        net_base = self._run_pf(network, base_mods)
 
        # Dominance guard: max loading must be clearly above 2nd
        max_line_idx = self._dominant_idxmax(
            net_base.res_line["loading_percent"], min_gap_pct=3.0)
        max_loading = net_base.res_line.at[max_line_idx, "loading_percent"]
 
        threshold = round(max_loading * self.rng.uniform(0.5, 0.85), 1)
        if threshold <= 0: threshold = 1.0
 
        action_mod = self._sample_explicit_pf_action(meta, net_base)
        net_after = self._run_pf(network, base_mods + [action_mod])
        gt = round(float(net_after.res_bus["vm_pu"].min()), 6)
        if not _is_valid_gt(gt): raise ValueError(f"Invalid GT: {gt}")
 
        mod_desc, mod_desc_lower = (
            mod_desc_override
            if mod_desc_override is not None
            else self._build_mod_desc(base_mods)
        )
        nl = self.rng.choice(_get_merged_multi_templates()["sequential"]).format(
            network_display=net_display, mod_description=mod_desc,
            mod_description_lower=mod_desc_lower,
            condition_metric="any line's loading", condition_op_word="exceeds",
            condition_threshold=f"{threshold}%",
            action_desc=describe_modification(action_mod),
            final_query_desc="the minimum bus voltage (p.u.)")
 
        lines = ["import pandapower as pp", "import pandapower.networks as pn", "",
                 f"net = {reg[network]['loader']}", ""]
        if base_mods:
            lines.append("# Apply modifications")
            for m in base_mods: lines.append(mod_to_code(m))
            lines.append("")
        lines += ["# Step 1: run power flow", "pp.runpp(net)", "",
                  f"# Step 2: check condition",
                  f'max_loading = net.res_line["loading_percent"].max()',
                  f"if max_loading > {threshold}:",
                  f"    {mod_to_code(action_mod)}",
                  f"    pp.runpp(net)", "",
                  'result = net.res_bus["vm_pu"].min()', "print(result)"]
 
        scenario = Scenario(network=network, task="sequential",
                            modifications=base_mods + [action_mod],
                            query_target=QueryTarget("sequential", "res_bus", "vm_pu",
                                                     threshold=threshold))
        return BenchmarkItem(id=self._make_id(scenario), scenario=scenario,
                             natural_language_query=nl, reference_code="\n".join(lines),
                             ground_truth=gt, ground_truth_type="float")
 
    def generate_pf_then_sc(self, network=None, base_mods=None,
                            mod_desc_override=None):
        """2-step: PF -> SC (superposition method). Tests LLM knowledge of SC prerequisites."""
        if network is None:
            network = self.rng.choice(self.networks)
        meta = self._get_meta(network)
        reg = _all_registries()
        net_display = reg[network]["display"]
 
        import pandapower as pp
        import pandapower.shortcircuit
 
        base_mods = (
            list(base_mods) if base_mods is not None
            else sample_modifications(meta, self.rng)
        )
        target_bus = self.rng.choice(meta["bus_indices"])
 
        net = _load_network(network)
        for m in base_mods: apply_modification(net, m)
        with _SuppressPandapowerLogs():
            pp.runpp(net)
            # SC setup
            if "s_sc_max_mva" not in net.ext_grid.columns or net.ext_grid["s_sc_max_mva"].isna().any():
                net.ext_grid["s_sc_max_mva"] = 1000
                net.ext_grid["s_sc_min_mva"] = 800
                net.ext_grid["rx_max"] = 0.1
                net.ext_grid["rx_min"] = 0.1
            if len(net.gen) > 0 and "vn_kv" not in net.gen.columns:
                net.gen["in_service"] = False
            if len(net.sgen) > 0:
                net.sgen["in_service"] = False
            pp.shortcircuit.calc_sc(net, fault="3ph", case="max",
                                    use_pre_fault_voltage=True)
 
        gt = round(float(net.res_bus_sc.at[target_bus, "ikss_ka"]), 6)
        if not _is_valid_gt(gt): raise ValueError(f"Invalid GT: {gt}")
 
        mod_desc, mod_desc_lower = (
            mod_desc_override
            if mod_desc_override is not None
            else self._build_mod_desc(base_mods)
        )
        nl = self.rng.choice(_get_merged_multi_templates()["pf_then_sc"]).format(
            network_display=net_display, mod_description=mod_desc,
            mod_description_lower=mod_desc_lower, target_bus=target_bus)
 
        tc_sc = TASK_CONFIG["short_circuit_3ph"]
        lines = ["import pandapower as pp", "import pandapower.networks as pn",
                 "import pandapower.shortcircuit", "",
                 f"net = {reg[network]['loader']}", ""]
        if base_mods:
            for m in base_mods: lines.append(mod_to_code(m))
            lines.append("")
        lines += ["# Step 1: run power flow", "pp.runpp(net)", "",
                  "# Step 2: SC with superposition method"]
        for s in tc_sc.get("setup_code", []): lines.append(s)
        lines += ['pp.shortcircuit.calc_sc(net, fault="3ph", case="max", use_pre_fault_voltage=True)',
                  "", f'result = net.res_bus_sc.at[{target_bus}, "ikss_ka"]', "print(result)"]
 
        scenario = Scenario(network=network, task="pf_then_sc",
                            modifications=base_mods,
                            query_target=QueryTarget("pf_then_sc", "res_bus_sc", "ikss_ka",
                                                     filter_idx=target_bus))
        return BenchmarkItem(id=self._make_id(scenario), scenario=scenario,
                             natural_language_query=nl, reference_code="\n".join(lines),
                             ground_truth=gt, ground_truth_type="float")
 
    def generate_contingency_fix(self, network=None, base_mods=None,
                                 mod_desc_override=None):
        """2-step: Contingency -> find worst -> add gen -> PF under worst case -> verify."""
        if network is None:
            candidates = [n for n in self.networks
                          if self._get_meta(n)["n_line"] >= 5
                          and self._get_meta(n)["n_bus"] <= 60]  # small only — contingency is O(n_lines)
            if not candidates: raise ValueError("No suitable networks")
            network = self.rng.choice(candidates)
        meta = self._get_meta(network)
        reg = _all_registries()
        net_display = reg[network]["display"]
 
        import pandapower as pp
        from pandapower.contingency import run_contingency
 
        net = _load_network(network)
        for m in (base_mods or []):
            apply_modification(net, m)
        with _SuppressPandapowerLogs():
            nminus1 = {"line": {"index": net.line.index.tolist()}}
            run_contingency(net, nminus1)
 
        if "max_loading_percent" not in net.res_line.columns:
            raise ValueError("Contingency did not produce max_loading_percent")
 
        worst_line = self._dominant_idxmax(
            net.res_line["max_loading_percent"], min_gap_pct=2.0)
 
        # Choose a bus and gen size for remediation
        fix_bus = self.rng.choice(meta["bus_indices"])
        fix_gen_mw = round(self.rng.uniform(10, 40), 0)
 
        # Apply fix and verify
        net2 = _load_network(network)
        for m in (base_mods or []):
            apply_modification(net2, m)
        net2.line.at[worst_line, "in_service"] = False
        pp.create_gen(net2, bus=fix_bus, p_mw=fix_gen_mw, vm_pu=1.0)
        with _SuppressPandapowerLogs():
            pp.runpp(net2)
 
        gt = round(float(net2.res_line["loading_percent"].max()), 6)
        if not _is_valid_gt(gt): raise ValueError(f"Invalid GT: {gt}")
 
        mod_desc, mod_desc_lower = (
            mod_desc_override
            if mod_desc_override is not None
            else self._build_mod_desc(base_mods or [])
        )
        nl = self.rng.choice(_get_merged_multi_templates()["contingency_fix"]).format(
            network_display=net_display, mod_description=mod_desc,
            mod_description_lower=mod_desc_lower,
            fix_gen_mw=int(fix_gen_mw), fix_bus=fix_bus)
 
        lines = ["import pandapower as pp", "import pandapower.networks as pn",
                 "from pandapower.contingency import run_contingency", "",
                 f"net = {reg[network]['loader']}", "",
                 ]
        if base_mods:
            lines.append("# Apply semantic/base modifications")
            for m in base_mods:
                lines.append(mod_to_code(m))
            lines.append("")
        lines += [
                 "# Step 1: N-1 contingency analysis",
                 'nminus1 = {"line": {"index": net.line.index.tolist()}}',
                 "run_contingency(net, nminus1)", "",
                 "# Find worst contingency",
                 'worst_line = net.res_line["max_loading_percent"].idxmax()', "",
                 f"# Step 2: add {int(fix_gen_mw)} MW gen at bus {fix_bus}, disconnect worst line",
                 f"net.line.at[worst_line, 'in_service'] = False",
                 f"pp.create_gen(net, bus={fix_bus}, p_mw={fix_gen_mw}, vm_pu=1.0)", "",
                 "# Verify", "pp.runpp(net)", "",
                 'result = net.res_line["loading_percent"].max()', "print(result)"]
 
        fix_mods = list(base_mods or []) + [
            Modification("disconnect_line", {"idx": worst_line}),
            Modification("add_gen", {"bus": fix_bus, "p_mw": fix_gen_mw, "vm_pu": 1.0}),
        ]
        scenario = Scenario(network=network, task="contingency_fix",
                    modifications=fix_mods,
                    query_target=QueryTarget("contingency_fix", "res_line",
                                             "loading_percent"))
        return BenchmarkItem(id=self._make_id(scenario), scenario=scenario,
                             natural_language_query=nl, reference_code="\n".join(lines),
                             ground_truth=gt, ground_truth_type="float")
 
    def generate_parallel(self, network=None, base_mods=None,
                          mod_desc_override=None):
        """2-step: PF + SC parallel analysis."""
        if network is None:
            network = self.rng.choice(self.networks)
        meta = self._get_meta(network)
        reg = _all_registries()
        net_display = reg[network]["display"]
 
        base_mods = (
            list(base_mods) if base_mods is not None
            else sample_modifications(meta, self.rng)
        )
 
        net_pf = self._run_pf(network, base_mods)
 
        import pandapower as pp
        import pandapower.shortcircuit
        net_sc = _load_network(network)
        for m in base_mods: apply_modification(net_sc, m)
        with _SuppressPandapowerLogs():
            run_task(net_sc, "short_circuit_3ph")
        sc_val = round(float(net_sc.res_bus_sc["ikss_ka"].max()), 6)
        if not _is_valid_gt(sc_val): raise ValueError(f"Invalid GT: {sc_val}")
 
        mod_desc, mod_desc_lower = (
            mod_desc_override
            if mod_desc_override is not None
            else self._build_mod_desc(base_mods)
        )
        nl = self.rng.choice(_get_merged_multi_templates()["parallel"]).format(
            network_display=net_display, mod_description=mod_desc,
            mod_description_lower=mod_desc_lower,
            task_a_name="power flow", task_b_name="3-phase short circuit analysis",
            metric_a_desc="the maximum line loading (%)",
            metric_b_desc="the maximum short circuit current (ikss in kA)")
 
        tc_sc = TASK_CONFIG["short_circuit_3ph"]
        lines = ["import pandapower as pp", "import pandapower.networks as pn",
                 "import pandapower.shortcircuit", "",
                 f"net = {reg[network]['loader']}", ""]
        if base_mods:
            for m in base_mods: lines.append(mod_to_code(m))
            lines.append("")
        lines += ["# Task A: power flow", "pp.runpp(net)",
                  'max_loading = net.res_line["loading_percent"].max()', "",
                  "# Task B: short circuit"]
        for s in tc_sc.get("setup_code", []): lines.append(s)
        lines += [tc_sc["run_code"], "",
                  'result = net.res_bus_sc["ikss_ka"].max()', "print(result)"]
 
        scenario = Scenario(network=network, task="parallel",
                            modifications=base_mods,
                            query_target=QueryTarget("parallel", "res_bus_sc", "ikss_ka"))
        return BenchmarkItem(id=self._make_id(scenario), scenario=scenario,
                             natural_language_query=nl, reference_code="\n".join(lines),
                             ground_truth=sc_val, ground_truth_type="float")
 
    def generate_diagnose_and_fix(self, network=None, base_mods=None,
                                  mod_desc_override=None):
        """3-step: PF -> overload check -> disconnect -> PF -> voltage check -> add shunt -> PF."""
        if network is None:
            candidates = [n for n in self.networks
                          if self._get_meta(n)["n_line"] >= 5
                          and self._get_meta(n)["n_bus"] <= 60]  # small only — 3 PF runs
            if not candidates: raise ValueError("No suitable networks")
            network = self.rng.choice(candidates)
        meta = self._get_meta(network)
        reg = _all_registries()
        net_display = reg[network]["display"]
 
        import pandapower as pp
        base_mods = (
            list(base_mods) if base_mods is not None
            else sample_modifications(meta, self.rng)
        )
 
        # Stage 1: PF, check overload
        net = _load_network(network)
        for m in base_mods: apply_modification(net, m)
        with _SuppressPandapowerLogs():
            pp.runpp(net)
 
        overload_threshold = round(net.res_line["loading_percent"].max() *
                                   self.rng.uniform(0.5, 0.85), 1)
        if overload_threshold <= 0: overload_threshold = 1.0
        max_line = self._dominant_idxmax(net.res_line["loading_percent"], min_gap_pct=3.0)
 
        # Stage 2: disconnect worst line, re-run PF
        net.line.at[max_line, "in_service"] = False
        with _SuppressPandapowerLogs():
            pp.runpp(net)
 
        # Check voltage
        voltage_threshold = round(self.rng.uniform(0.93, 0.99), 2)
        min_v_bus = int(net.res_bus["vm_pu"].idxmin())
        min_v_val = net.res_bus.at[min_v_bus, "vm_pu"]
 
        # Stage 3: add shunt if voltage is low, re-run PF
        shunt_mvar = round(self.rng.uniform(5, 20), 0)
        need_shunt = min_v_val < voltage_threshold
        if need_shunt:
            pp.create_shunt(net, bus=min_v_bus, q_mvar=-shunt_mvar, p_mw=0)
        with _SuppressPandapowerLogs():
            pp.runpp(net)
 
        gt = round(float(net.res_bus["vm_pu"].min()), 6)
        if not _is_valid_gt(gt): raise ValueError(f"Invalid GT: {gt}")
 
        mod_desc, mod_desc_lower = (
            mod_desc_override
            if mod_desc_override is not None
            else self._build_mod_desc(base_mods)
        )
        nl = self.rng.choice(_get_merged_multi_templates()["diagnose_and_fix"]).format(
            network_display=net_display, mod_description=mod_desc,
            mod_description_lower=mod_desc_lower,
            overload_threshold=overload_threshold,
            voltage_threshold=voltage_threshold,
            shunt_mvar=int(shunt_mvar))
 
        lines = ["import pandapower as pp", "import pandapower.networks as pn", "",
                 f"net = {reg[network]['loader']}", ""]
        if base_mods:
            for m in base_mods: lines.append(mod_to_code(m))
            lines.append("")
        lines += [
            "# Stage 1: run PF, check overloads", "pp.runpp(net)",
            f'max_loading = net.res_line["loading_percent"].max()',
            f"if max_loading > {overload_threshold}:",
            f'    worst_line = net.res_line["loading_percent"].idxmax()',
            f'    net.line.at[worst_line, "in_service"] = False', "",
            "# Stage 2: re-run PF, check voltages", "pp.runpp(net)",
            f'min_v = net.res_bus["vm_pu"].min()',
            f"if min_v < {voltage_threshold}:",
            f'    weak_bus = net.res_bus["vm_pu"].idxmin()',
            f"    pp.create_shunt(net, bus=weak_bus, q_mvar={-shunt_mvar}, p_mw=0)", "",
            "# Stage 3: final PF", "pp.runpp(net)", "",
            'result = net.res_bus["vm_pu"].min()', "print(result)"]
 
        actual_mods = list(base_mods)
        actual_mods.append(Modification("disconnect_line", {"idx": max_line}))  # Stage 1 always fires
        if need_shunt:
            actual_mods.append(Modification("add_shunt", {"bus": min_v_bus, "q_mvar": -shunt_mvar, "p_mw": 0}))

        scenario = Scenario(network=network, task="diagnose_and_fix",
                            modifications=actual_mods,  
                            query_target=QueryTarget("diagnose_and_fix", "res_bus", "vm_pu",
                                                    threshold=voltage_threshold))
        
        return BenchmarkItem(id=self._make_id(scenario), scenario=scenario,
                             natural_language_query=nl, reference_code="\n".join(lines),
                             ground_truth=gt, ground_truth_type="float")
 
    def _multi_generators(self):
        return {
            "comparison": self.generate_comparison,
            "comparison_opf": self.generate_comparison_opf,
            "sequential": self.generate_sequential,
            "pf_then_sc": self.generate_pf_then_sc,
            "contingency_fix": self.generate_contingency_fix,
            "parallel": self.generate_parallel,
            "diagnose_and_fix": self.generate_diagnose_and_fix,
        }

    def _pf_result_specs(self, meta):
        """Result metrics allowed for rule-grounded PF-style D3/D4 items."""
        specs = [
            ("min_value", "res_bus", "vm_pu", "the minimum bus voltage (p.u.)", "min()"),
            ("total", "res_ext_grid", "p_mw", "the total external grid active power injection (MW)", "sum()"),
        ]
        if meta["n_line"] > 0:
            specs.append(("max_value", "res_line", "loading_percent",
                          "the maximum line loading (%)", "max()"))
            specs.append(("total", "res_line", "pl_mw",
                          "the total active power losses across all lines (MW)", "sum()"))
        return specs

    def _load_solved_pf(self, network):
        import pandapower as pp
        net = _load_network(network)
        with _SuppressPandapowerLogs():
            pp.runpp(net)
        return net

    def _semantic_setup_rule_specs(self):
        """Final rule-grounded setup library for D3 and D4."""
        return {
            "largest_load_peak": {
                "category": "load_scenario",
                "requirements": {"min_load": 1},
                "d4_wrappers": {
                    "semantic_then_check",
                    "semantic_then_compensate",
                    "semantic_comparison",
                    "semantic_then_redispatch",
                    "two_stage_rank_then_act",
                    "dual_condition_choose_action",
                },
            },
            "largest_load_curtailment": {
                "category": "load_scenario",
                "requirements": {"min_load": 1},
                "d4_wrappers": {
                    "semantic_comparison",
                    "semantic_then_check",
                    "semantic_then_compensate",
                    "semantic_then_redispatch",
                    "two_stage_rank_then_act",
                    "dual_condition_choose_action",
                },
            },
            "lowest_voltage_load_curtailment": {
                "category": "load_scenario",
                "requirements": {"min_load": 1, "min_bus": 3},
                "d4_wrappers": {
                    "semantic_then_check",
                    "semantic_then_compensate",
                    "semantic_comparison",
                    "semantic_then_redispatch",
                    "two_stage_rank_then_act",
                    "dual_condition_choose_action",
                },
            },
            "most_loaded_line_outage": {
                "category": "topo_scenario",
                "requirements": {"min_line": 2},
                "d4_wrappers": {
                    "semantic_then_check",
                    "semantic_then_compensate",
                    "semantic_comparison",
                    "semantic_then_redispatch",
                    "two_stage_rank_then_act",
                    "dual_condition_choose_action",
                },
            },
            "most_loaded_line_reinforcement": {
                "category": "topo_scenario",
                "requirements": {"min_line": 2},
                "d4_wrappers": {
                    "semantic_comparison",
                    "semantic_then_check",
                    "semantic_then_compensate",
                    "semantic_then_redispatch",
                    "two_stage_rank_then_act",
                    "dual_condition_choose_action",
                },
            },
            "longest_line_outage": {
                "category": "topo_scenario",
                "requirements": {"min_line": 2},
                "d4_wrappers": {
                    "semantic_then_check",
                    "semantic_then_compensate",
                    "semantic_comparison",
                    "semantic_then_redispatch",
                    "two_stage_rank_then_act",
                    "dual_condition_choose_action",
                },
            },
            "largest_generator_trip": {
                "category": "gen_scenario",
                "requirements": {"min_gen": 2},
                "d4_wrappers": {
                    "semantic_then_check",
                    "semantic_then_compensate",
                    "semantic_comparison",
                    "semantic_then_redispatch",
                    "two_stage_rank_then_act",
                    "dual_condition_choose_action",
                },
            },
            "largest_generator_redispatch_up": {
                "category": "gen_scenario",
                "requirements": {"min_gen": 1},
                "d4_wrappers": {
                    "semantic_comparison",
                    "semantic_then_check",
                    "semantic_then_compensate",
                    "semantic_then_redispatch",
                    "two_stage_rank_then_act",
                    "dual_condition_choose_action",
                },
            },
            "largest_generator_redispatch_down": {
                "category": "gen_scenario",
                "requirements": {"min_gen": 1},
                "d4_wrappers": {
                    "semantic_comparison",
                    "semantic_then_check",
                    "semantic_then_compensate",
                    "semantic_then_redispatch",
                    "two_stage_rank_then_act",
                    "dual_condition_choose_action",
                },
            },
            "lowest_voltage_shunt_support": {
                "category": "add_element_scenario",
                "requirements": {"min_bus": 3},
                "d4_wrappers": {
                    "semantic_comparison",
                    "semantic_then_check",
                    "semantic_then_compensate",
                    "semantic_then_redispatch",
                    "two_stage_rank_then_act",
                    "dual_condition_choose_action",
                },
            },
            "highest_voltage_shunt_absorption": {
                "category": "add_element_scenario",
                "requirements": {"min_bus": 3},
                "d4_wrappers": {
                    "semantic_comparison",
                    "semantic_then_check",
                    "semantic_then_compensate",
                    "semantic_then_redispatch",
                    "two_stage_rank_then_act",
                    "dual_condition_choose_action",
                },
            },
            "largest_trafo_tap_adjustment": {
                "category": "voltage_scenario",
                "requirements": {"min_trafo": 1},
                "d4_wrappers": {
                    "semantic_comparison",
                    "semantic_then_check",
                    "semantic_then_compensate",
                    "semantic_then_redispatch",
                    "two_stage_rank_then_act",
                    "dual_condition_choose_action",
                },
            },
        }

    def _network_supports_semantic_rule(self, network: str, rule_name: str) -> bool:
        req = self._semantic_setup_rule_specs()[rule_name]["requirements"]
        try:
            meta = self._get_meta(network)
        except Exception:
            return False
        return (
            meta["n_load"] >= req.get("min_load", 0)
            and meta["n_line"] >= req.get("min_line", 0)
            and meta["n_gen"] >= req.get("min_gen", 0)
            and meta["n_bus"] >= req.get("min_bus", 0)
            and meta["n_trafo"] >= req.get("min_trafo", 0)
        )

    def _semantic_mod_desc_override(self, scenario_phrase: str) -> tuple:
        phrase = (scenario_phrase or "").strip()
        if not phrase:
            return "", ""
        return phrase[0].upper() + phrase[1:] + ". ", phrase + ", "

    def _weighted_need_order(self, names, counter_key: str, weight_map: dict | None = None):
        """Return names ordered from most underused to most overused."""
        counts = self._semantic_balance[counter_key]
        weight_map = weight_map or {}
        keyed = []
        for idx, name in enumerate(names):
            target_weight = float(weight_map.get(name, 1.0))
            score = counts[name] / target_weight
            keyed.append((score, counts[name], idx, name))
        keyed.sort()
        return [name for _, _, _, name in keyed]

    def _semantic_rules_for_single_task(self, task_name: str) -> list:
        """Return the subset of semantic setup rules that fit one D1 task family."""
        common_load_gen = [
            "largest_load_peak",
            "largest_load_curtailment",
            "largest_generator_redispatch_up",
            "largest_generator_redispatch_down",
        ]
        if task_name == "power_flow":
            return list(self._semantic_setup_rule_specs().keys())
        if task_name == "dc_power_flow":
            return common_load_gen + [
                "most_loaded_line_outage",
                "most_loaded_line_reinforcement",
                "longest_line_outage",
                "largest_trafo_tap_adjustment",
            ]
        if task_name == "opf":
            return common_load_gen + [
                "lowest_voltage_load_curtailment",
                "most_loaded_line_reinforcement",
                "lowest_voltage_shunt_support",
                "largest_trafo_tap_adjustment",
            ]
        if task_name in ("short_circuit_3ph", "short_circuit_2ph"):
            return common_load_gen + [
                "largest_generator_trip",
                "most_loaded_line_outage",
                "longest_line_outage",
                "largest_trafo_tap_adjustment",
            ]
        if task_name == "time_series":
            return common_load_gen + [
                "lowest_voltage_load_curtailment",
                "most_loaded_line_reinforcement",
                "lowest_voltage_shunt_support",
                "highest_voltage_shunt_absorption",
                "largest_trafo_tap_adjustment",
            ]
        if task_name == "state_estimation":
            return common_load_gen + [
                "lowest_voltage_load_curtailment",
                "lowest_voltage_shunt_support",
                "highest_voltage_shunt_absorption",
                "largest_trafo_tap_adjustment",
            ]
        if task_name == "contingency":
            return common_load_gen + [
                "most_loaded_line_reinforcement",
                "lowest_voltage_shunt_support",
                "largest_trafo_tap_adjustment",
            ]
        return list(self._semantic_setup_rule_specs().keys())

    def _semantic_rules_for_multi_task(self, task_name: str) -> list:
        """Return the subset of semantic setup rules that fit one D2 family."""
        common_ops = [
            "largest_load_peak",
            "largest_load_curtailment",
            "largest_generator_redispatch_up",
            "largest_generator_redispatch_down",
            "lowest_voltage_shunt_support",
            "largest_trafo_tap_adjustment",
        ]
        if task_name == "comparison":
            return common_ops + [
                "most_loaded_line_outage",
                "most_loaded_line_reinforcement",
                "highest_voltage_shunt_absorption",
                "largest_generator_trip",
            ]
        if task_name == "comparison_opf":
            return common_ops + [
                "lowest_voltage_load_curtailment",
                "most_loaded_line_reinforcement",
            ]
        if task_name == "sequential":
            return common_ops + [
                "lowest_voltage_load_curtailment",
                "most_loaded_line_reinforcement",
                "largest_generator_trip",
            ]
        if task_name == "pf_then_sc":
            return [
                "largest_load_peak",
                "largest_load_curtailment",
                "most_loaded_line_outage",
                "most_loaded_line_reinforcement",
                "largest_generator_trip",
                "largest_generator_redispatch_up",
                "largest_generator_redispatch_down",
                "largest_trafo_tap_adjustment",
            ]
        if task_name == "contingency_fix":
            return common_ops + [
                "lowest_voltage_load_curtailment",
                "most_loaded_line_reinforcement",
            ]
        if task_name == "parallel":
            return common_ops + [
                "most_loaded_line_outage",
                "most_loaded_line_reinforcement",
                "highest_voltage_shunt_absorption",
                "largest_generator_trip",
            ]
        if task_name == "diagnose_and_fix":
            return common_ops + [
                "lowest_voltage_load_curtailment",
                "most_loaded_line_reinforcement",
            ]
        return list(self._semantic_setup_rule_specs().keys())

    def _pick_semantic_single_task_and_network(self, network=None):
        """Pick one D1 task + network pair that supports both the task and a semantic rule."""
        tasks = list(TASK_CONFIG.keys())
        self.rng.shuffle(tasks)
        tasks = self._weighted_need_order(
            tasks,
            "d3_task",
            SEMANTIC_BALANCE_CONFIG.get("d3_task_weights"),
        )

        for task in tasks:
            task_filter = TASK_CONFIG[task]["filter"]
            rule_names = self._weighted_need_order(
                self._semantic_rules_for_single_task(task),
                "d3_rule",
                SEMANTIC_BALANCE_CONFIG.get("d3_rule_weights"),
            )
            if network is not None:
                try:
                    meta = self._get_meta(network)
                except Exception:
                    continue
                if not task_filter(meta):
                    continue
                rule_candidates = [
                    r for r in rule_names
                    if self._network_supports_semantic_rule(network, r)
                ]
                if rule_candidates:
                    return task, network, self.rng.choice(rule_candidates)
                continue

            net_candidates = []
            for name in self.networks:
                try:
                    meta = self._get_meta(name)
                except Exception:
                    continue
                if not task_filter(meta):
                    continue
                rule_candidates = [
                    r for r in rule_names
                    if self._network_supports_semantic_rule(name, r)
                ]
                if rule_candidates:
                    net_candidates.append((name, rule_candidates))
            if net_candidates:
                picked_network, picked_rules = self.rng.choice(net_candidates)
                return task, picked_network, self.rng.choice(picked_rules)
        raise ValueError("No semantic D1 task/network/rule combination available")

    def _semantic_multi_task_filter(self, task_name: str, meta: dict) -> bool:
        """Check whether a network can support one D2 generator family."""
        if task_name == "comparison_opf":
            return bool(meta["has_cost"]) and meta["n_load"] > 0
        if task_name == "sequential":
            return meta["n_line"] >= 3
        if task_name in ("contingency_fix", "diagnose_and_fix"):
            return meta["n_line"] >= 5 and meta["n_bus"] <= 60
        if task_name == "pf_then_sc":
            return meta["n_bus"] > 0
        if task_name == "parallel":
            return meta["n_bus"] > 0
        if task_name == "comparison":
            return meta["n_bus"] > 0
        return False

    def _pick_semantic_multi_task_and_network(self, network=None):
        """Pick one D2 generator family + network + semantic rule triplet."""
        multi_tasks = list(self._multi_generators().keys())
        self.rng.shuffle(multi_tasks)
        multi_tasks = self._weighted_need_order(
            multi_tasks,
            "d4_task",
            SEMANTIC_BALANCE_CONFIG.get("d4_task_weights"),
        )

        for task_name in multi_tasks:
            rule_names = self._weighted_need_order(
                self._semantic_rules_for_multi_task(task_name),
                "d4_rule",
                SEMANTIC_BALANCE_CONFIG.get("d4_rule_weights"),
            )
            if network is not None:
                try:
                    meta = self._get_meta(network)
                except Exception:
                    continue
                if not self._semantic_multi_task_filter(task_name, meta):
                    continue
                rule_candidates = [
                    r for r in rule_names
                    if self._network_supports_semantic_rule(network, r)
                ]
                if rule_candidates:
                    return task_name, network, self.rng.choice(rule_candidates)
                continue

            net_candidates = []
            for name in self.networks:
                try:
                    meta = self._get_meta(name)
                except Exception:
                    continue
                if not self._semantic_multi_task_filter(task_name, meta):
                    continue
                rule_candidates = [
                    r for r in rule_names
                    if self._network_supports_semantic_rule(name, r)
                ]
                if rule_candidates:
                    net_candidates.append((name, rule_candidates))
            if net_candidates:
                picked_network, picked_rules = self.rng.choice(net_candidates)
                return task_name, picked_network, self.rng.choice(picked_rules)
        raise ValueError("No semantic D2 task/network/rule combination available")

    def _pick_semantic_rule(self, network=None, allowed_wrappers=None):
        """Pick one setup rule and a compatible network."""
        specs = self._semantic_setup_rule_specs()
        if allowed_wrappers:
            rule_pool = [
                k for k, v in specs.items()
                if any(w in v["d4_wrappers"] for w in allowed_wrappers)
            ]
        else:
            rule_pool = list(specs.keys())
        self.rng.shuffle(rule_pool)

        if network is not None:
            candidates = [
                r for r in rule_pool
                if self._network_supports_semantic_rule(network, r)
            ]
            if not candidates:
                raise ValueError(f"No semantic rules fit network: {network}")
            return network, self.rng.choice(candidates)

        pair_pool = []
        for name in self.networks:
            for rule_name in rule_pool:
                if self._network_supports_semantic_rule(name, rule_name):
                    pair_pool.append((name, rule_name))
        if not pair_pool:
            raise ValueError("No (network, rule) pair satisfies semantic requirements")
        return self.rng.choice(pair_pool)

    def _d4_wrapper_specs(self):
        """Wrapper-level requirements for compound semantic tasks."""
        return {
            "semantic_then_check": {"min_line": 2},
            "semantic_then_compensate": {"min_bus": 3},
            "semantic_comparison": {},
            "semantic_then_redispatch": {"min_gen": 1},
            "two_stage_rank_then_act": {"min_line": 2},
            "dual_condition_choose_action": {"min_line": 2, "min_bus": 3},
        }

    def _pick_compound_wrapper(self, network: str, setup_rule_name: str) -> str:
        """Pick one D4 wrapper compatible with both the setup rule and network."""
        meta = self._get_meta(network)
        allowed = self._semantic_setup_rule_specs()[setup_rule_name]["d4_wrappers"]
        candidates = []
        for wrapper_name in allowed:
            req = self._d4_wrapper_specs().get(wrapper_name, {})
            if (meta["n_load"] >= req.get("min_load", 0)
                    and meta["n_line"] >= req.get("min_line", 0)
                    and meta["n_gen"] >= req.get("min_gen", 0)
                    and meta["n_bus"] >= req.get("min_bus", 0)
                    and meta["n_trafo"] >= req.get("min_trafo", 0)):
                candidates.append(wrapper_name)
        if not candidates:
            raise ValueError(
                f"No D4 wrappers fit network={network}, setup_rule={setup_rule_name}"
            )
        wrapper_weights = {
            "semantic_comparison": 1.35,
            "semantic_then_compensate": 1.25,
            "semantic_then_redispatch": 1.25,
            "dual_condition_choose_action": 1.0,
            "semantic_then_check": 0.75,
            "two_stage_rank_then_act": 0.7,
        }
        weights = [wrapper_weights.get(name, 1.0) for name in candidates]
        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def _build_semantic_setup_rule(self, net, rule_name: str) -> dict:
        """Build one deterministic hidden-action setup rule from the current network state."""
        if rule_name == "largest_load_peak":
            target_idx = self._dominant_idxmax(
                net.load["p_mw"].abs(), min_gap_pct=3.0, min_abs_gap=1e-4
            )
            factor = round(self.rng.uniform(1.45, 1.9), 2)
            return {
                "category": "load_scenario",
                "scenario_phrase": (
                    "identify the currently largest load and raise only that "
                    "demand point to a peak-demand operating level"
                ),
                "modifications": [Modification(
                    "set_load_scaling", {"idx": target_idx, "value": factor}
                )],
                "setup_lines": [
                    "# Apply semantic setup: peak-load increase at the largest existing load",
                    'target_load = net.load["p_mw"].abs().idxmax()',
                    f'net.load.at[target_load, "scaling"] = {factor}',
                    "",
                ],
                "metadata": {"target_load": target_idx, "load_factor": factor},
            }

        if rule_name == "largest_load_curtailment":
            target_idx = self._dominant_idxmax(
                net.load["p_mw"].abs(), min_gap_pct=3.0, min_abs_gap=1e-4
            )
            factor = round(self.rng.uniform(0.55, 0.85), 2)
            return {
                "category": "load_scenario",
                "scenario_phrase": (
                    "identify the currently largest load and curtail only that "
                    "demand point to a low-demand operating level"
                ),
                "modifications": [Modification(
                    "set_load_scaling", {"idx": target_idx, "value": factor}
                )],
                "setup_lines": [
                    "# Apply semantic setup: demand curtailment at the largest existing load",
                    'target_load = net.load["p_mw"].abs().idxmax()',
                    f'net.load.at[target_load, "scaling"] = {factor}',
                    "",
                ],
                "metadata": {"target_load": target_idx, "load_factor": factor},
            }

        if rule_name == "lowest_voltage_load_curtailment":
            load_buses = set(net.load["bus"].astype(int).tolist())
            bus_vm = net.res_bus.loc[net.res_bus.index.isin(load_buses), "vm_pu"]
            if bus_vm.empty:
                raise ValueError("No load buses available for low-voltage curtailment")
            target_bus = self._dominant_idxmin(
                bus_vm, min_gap_pct=0.2, min_abs_gap=1e-4
            )
            load_rows = net.load[net.load["bus"] == target_bus]
            if load_rows.empty:
                raise ValueError("No load connected to the selected lowest-voltage bus")
            if len(load_rows) > 1:
                target_idx = self._dominant_idxmax(
                    load_rows["p_mw"].abs(), min_gap_pct=2.0, min_abs_gap=1e-4
                )
            else:
                target_idx = int(load_rows.index[0])
            factor = round(self.rng.uniform(0.55, 0.85), 2)
            return {
                "category": "load_scenario",
                "scenario_phrase": (
                    "locate the lowest-voltage bus that has load connected and "
                    "curtail the largest load at that bus to a low-demand level"
                ),
                "modifications": [Modification(
                    "set_load_scaling", {"idx": target_idx, "value": factor}
                )],
                "setup_lines": [
                    "# Apply semantic setup: curtail the largest load at the lowest-voltage load bus",
                    "pp.runpp(net)",
                    'load_buses = net.load["bus"].astype(int).unique()',
                    'target_bus = net.res_bus.loc[load_buses, "vm_pu"].idxmin()',
                    'loads_at_bus = net.load[net.load["bus"] == target_bus]',
                    'target_load = loads_at_bus["p_mw"].abs().idxmax()',
                    f'net.load.at[target_load, "scaling"] = {factor}',
                    "",
                ],
                "metadata": {
                    "target_bus": target_bus,
                    "target_load": target_idx,
                    "load_factor": factor,
                },
            }

        if rule_name == "most_loaded_line_outage":
            target_idx = self._dominant_idxmax(
                net.res_line["loading_percent"], min_gap_pct=2.0, min_abs_gap=1e-4
            )
            return {
                "category": "topo_scenario",
                "scenario_phrase": (
                    "identify the most heavily loaded line under the current "
                    "operating point and take that line out of service"
                ),
                "modifications": [Modification(
                    "disconnect_line", {"idx": target_idx}
                )],
                "setup_lines": [
                    "# Apply semantic setup: outage of the currently most loaded line",
                    "pp.runpp(net)",
                    'target_line = net.res_line["loading_percent"].idxmax()',
                    'net.line.at[target_line, "in_service"] = False',
                    "",
                ],
                "metadata": {"target_line": target_idx},
            }

        if rule_name == "most_loaded_line_reinforcement":
            target_idx = self._dominant_idxmax(
                net.res_line["loading_percent"], min_gap_pct=2.0, min_abs_gap=1e-4
            )
            new_parallel = int(net.line.at[target_idx, "parallel"]) + 1
            return {
                "category": "topo_scenario",
                "scenario_phrase": (
                    "identify the most heavily loaded line under the current "
                    "operating point and reinforce that line by adding one parallel circuit"
                ),
                "modifications": [Modification(
                    "set_line_parallel", {"idx": target_idx, "value": new_parallel}
                )],
                "setup_lines": [
                    "# Apply semantic setup: reinforce the currently most loaded line",
                    "pp.runpp(net)",
                    'target_line = net.res_line["loading_percent"].idxmax()',
                    f'net.line.at[target_line, "parallel"] = {new_parallel}',
                    "",
                ],
                "metadata": {
                    "target_line": target_idx,
                    "parallel": new_parallel,
                },
            }

        if rule_name == "longest_line_outage":
            target_idx = self._dominant_idxmax(
                net.line["length_km"].fillna(0.0), min_gap_pct=2.0, min_abs_gap=1e-4
            )
            return {
                "category": "topo_scenario",
                "scenario_phrase": (
                    "find the physically longest line in the network and take "
                    "that line out of service"
                ),
                "modifications": [Modification(
                    "disconnect_line", {"idx": target_idx}
                )],
                "setup_lines": [
                    "# Apply semantic setup: outage of the physically longest line",
                    'target_line = net.line["length_km"].fillna(0.0).idxmax()',
                    'net.line.at[target_line, "in_service"] = False',
                    "",
                ],
                "metadata": {"target_line": target_idx},
            }

        if rule_name in (
            "largest_generator_trip",
            "largest_generator_redispatch_up",
            "largest_generator_redispatch_down",
        ):
            online_gen = net.gen[net.gen["in_service"]]
            if len(online_gen) < (2 if rule_name == "largest_generator_trip" else 1):
                raise ValueError("Not enough online generators for semantic rule")
            target_idx = self._dominant_idxmax(
                online_gen["p_mw"].abs(), min_gap_pct=2.0, min_abs_gap=1e-4
            )
            if rule_name == "largest_generator_trip":
                return {
                    "category": "gen_scenario",
                    "scenario_phrase": (
                        "find the online generator with the highest active power "
                        "dispatch and trip that unit offline"
                    ),
                    "modifications": [Modification(
                        "set_gen_out_of_service", {"idx": target_idx}
                    )],
                    "setup_lines": [
                        "# Apply semantic setup: trip the online generator with the largest dispatch",
                        'online_gen = net.gen[net.gen["in_service"]]',
                        'target_gen = online_gen["p_mw"].abs().idxmax()',
                        'net.gen.at[target_gen, "in_service"] = False',
                        "",
                    ],
                    "metadata": {"target_gen": target_idx},
                }

            current_p = float(net.gen.at[target_idx, "p_mw"])
            if rule_name == "largest_generator_redispatch_up":
                factor = self.rng.uniform(1.2, 1.6)
                phrase = (
                    "find the online generator with the highest active power "
                    "dispatch and raise that unit to a higher dispatch level"
                )
                comment = "# Apply semantic setup: redispatch the largest online generator upward"
            else:
                factor = self.rng.uniform(0.5, 0.85)
                phrase = (
                    "find the online generator with the highest active power "
                    "dispatch and reduce that unit to a lower dispatch level"
                )
                comment = "# Apply semantic setup: redispatch the largest online generator downward"
            new_p = round(current_p * factor, 1)
            if abs(new_p - current_p) < 1e-6:
                new_p = round(current_p + (10.0 if factor > 1 else -10.0), 1)
            return {
                "category": "gen_scenario",
                "scenario_phrase": phrase,
                "modifications": [Modification(
                    "set_gen_p", {"idx": target_idx, "value": new_p}
                )],
                "setup_lines": [
                    comment,
                    'online_gen = net.gen[net.gen["in_service"]]',
                    'target_gen = online_gen["p_mw"].abs().idxmax()',
                    f'net.gen.at[target_gen, "p_mw"] = {new_p}',
                    "",
                ],
                "metadata": {
                    "target_gen": target_idx,
                    "p_mw": new_p,
                },
            }

        if rule_name == "lowest_voltage_shunt_support":
            target_bus = self._dominant_idxmin(
                net.res_bus["vm_pu"], min_gap_pct=0.2, min_abs_gap=1e-4
            )
            shunt_mvar = round(self.rng.uniform(5.0, 20.0), 0)
            return {
                "category": "add_element_scenario",
                "scenario_phrase": (
                    "locate the lowest-voltage bus from the current operating "
                    "point and install capacitive shunt support at that bus"
                ),
                "modifications": [Modification(
                    "add_shunt", {
                        "bus": target_bus,
                        "q_mvar": -shunt_mvar,
                        "p_mw": 0,
                    }
                )],
                "setup_lines": [
                    "# Apply semantic setup: add capacitive support at the lowest-voltage bus",
                    "pp.runpp(net)",
                    'target_bus = net.res_bus["vm_pu"].idxmin()',
                    f"pp.create_shunt(net, bus=target_bus, q_mvar={-shunt_mvar}, p_mw=0)",
                    "",
                ],
                "metadata": {
                    "target_bus": target_bus,
                    "shunt_mvar": -shunt_mvar,
                },
            }

        if rule_name == "highest_voltage_shunt_absorption":
            target_bus = self._dominant_idxmax(
                net.res_bus["vm_pu"], min_gap_pct=0.2, min_abs_gap=1e-4
            )
            shunt_mvar = round(self.rng.uniform(5.0, 20.0), 0)
            return {
                "category": "add_element_scenario",
                "scenario_phrase": (
                    "locate the highest-voltage bus from the current operating "
                    "point and install inductive shunt absorption at that bus"
                ),
                "modifications": [Modification(
                    "add_shunt", {
                        "bus": target_bus,
                        "q_mvar": shunt_mvar,
                        "p_mw": 0,
                    }
                )],
                "setup_lines": [
                    "# Apply semantic setup: add inductive absorption at the highest-voltage bus",
                    "pp.runpp(net)",
                    'target_bus = net.res_bus["vm_pu"].idxmax()',
                    f"pp.create_shunt(net, bus=target_bus, q_mvar={shunt_mvar}, p_mw=0)",
                    "",
                ],
                "metadata": {
                    "target_bus": target_bus,
                    "shunt_mvar": shunt_mvar,
                },
            }

        if rule_name == "largest_trafo_tap_adjustment":
            if len(net.trafo) < 1:
                raise ValueError("No transformers available")
            if len(net.res_trafo) == 0:
                raise ValueError("No transformer result rows available")
            target_idx = self._dominant_idxmax(
                net.res_trafo["loading_percent"].abs(),
                min_gap_pct=2.0,
                min_abs_gap=1e-4,
            )
            current_tap = net.trafo.at[target_idx, "tap_pos"]
            current_tap = 0 if current_tap != current_tap else int(current_tap)
            new_tap = current_tap + self.rng.choice([-1, 1])
            return {
                "category": "voltage_scenario",
                "scenario_phrase": (
                    "find the most heavily loaded transformer and move its tap "
                    "position by one step"
                ),
                "modifications": [Modification(
                    "set_trafo_tap_pos", {"idx": target_idx, "value": new_tap}
                )],
                "setup_lines": [
                    "# Apply semantic setup: one-step tap adjustment on the most loaded transformer",
                    "pp.runpp(net)",
                    'target_trafo = net.res_trafo["loading_percent"].abs().idxmax()',
                    'current_tap = net.trafo.at[target_trafo, "tap_pos"]',
                    'if current_tap != current_tap:',
                    '    current_tap = 0',
                    f'net.trafo.at[target_trafo, "tap_pos"] = int(current_tap) + {new_tap - current_tap}',
                    "",
                ],
                "metadata": {
                    "target_trafo": target_idx,
                    "tap_pos": new_tap,
                },
            }

        raise ValueError(f"Unknown semantic setup rule: {rule_name}")

    def _build_d3_rule_case(self, network, rule_name=None):
        """Create one rule-grounded D3 case from network state + deterministic selector."""
        meta = self._get_meta(network)
        net = self._load_solved_pf(network)

        reg = _all_registries()
        net_display = reg[network]["display"]
        qtype, table, column, result_desc, agg_expr = self.rng.choice(self._pf_result_specs(meta))
        if rule_name is None:
            _, rule_name = self._pick_semantic_rule(network=network)
        rule_case = self._build_semantic_setup_rule(net, rule_name)

        pre_lines = [
            "import pandapower as pp",
            "import pandapower.networks as pn",
        ]
        if network.startswith("sb_"):
            pre_lines.append("import simbench as sb")
        pre_lines += [
            "",
            f"net = {reg[network]['loader']}",
            "",
        ]

        templates = _get_merged_semantic_templates().get((rule_case["category"], {
            ("min_value", "res_bus", "vm_pu"): "min_voltage",
            ("max_value", "res_line", "loading_percent"): "max_loading",
            ("total", "res_line", "pl_mw"): "total_loss",
            ("total", "res_ext_grid", "p_mw"): "ext_grid_p",
        }[(qtype, table, column)]), [])
        if templates:
            nl = self.rng.choice(templates).format(
                network_display=net_display,
                scenario_phrase=rule_case["scenario_phrase"],
            )
        else:
            nl = (
                f"On the {net_display}, {rule_case['scenario_phrase']}. "
                f"Run power flow and report {result_desc}."
            )

        ref_lines = pre_lines + rule_case["setup_lines"] + [
            "# Run the final power flow after the rule-grounded modification",
            "pp.runpp(net)",
            "",
            f'result = net.{table}["{column}"].{agg_expr}',
            "print(result)",
        ]

        scenario = Scenario(
            network=network,
            task="power_flow",
            modifications=rule_case["modifications"],
            query_target=QueryTarget(qtype, table, column),
            task_options={
                "semantic_source": "rule_grounded_d3",
                "semantic_rule": rule_name,
                "rule_metadata": rule_case.get("metadata", {}),
            },
        )
        return scenario, nl, "\n".join(ref_lines), [rule_case["scenario_phrase"]]

    def generate_derived_semantic_item(self, network=None, max_tries: int = 40):
        """Generate a rule-grounded D3 item aligned to one of the D1 task families."""
        for _ in range(max_tries):
            try:
                task_name, picked_network, rule_name = (
                    self._pick_semantic_single_task_and_network(network=network)
                )
                base_net = self._load_solved_pf(picked_network)
                rule_case = self._build_semantic_setup_rule(base_net, rule_name)
                scenario = Scenario(
                    network=picked_network,
                    task=task_name,
                    modifications=rule_case["modifications"],
                    query_target=sample_semantic_query_target(
                        task_name,
                        self._get_meta(picked_network),
                        self.rng,
                    ),
                    task_options={
                        "semantic_source": "rule_grounded_d3",
                        "semantic_rule": rule_name,
                        "rule_metadata": rule_case.get("metadata", {}),
                    },
                )
                gt, gt_type = execute_scenario(scenario)
                ref_code = generate_reference_code(scenario)
                nl = generate_semantic_nl_query(
                    scenario,
                    rule_case["scenario_phrase"],
                    self.rng,
                )
            except Exception:
                continue

            item = BenchmarkItem(
                id=self._make_variant_id(
                    BenchmarkItem(
                        id=self._make_id(scenario), scenario=scenario,
                        natural_language_query=nl, reference_code=ref_code,
                        ground_truth=gt, ground_truth_type=gt_type,
                    ),
                    "rule_d3",
                    nl,
                ),
                scenario=scenario,
                natural_language_query=nl,
                reference_code=ref_code,
                ground_truth=gt,
                ground_truth_type=gt_type,
                difficulty_level="D3_semantic",
                eval_criteria={
                    "semantic_source": "rule_grounded_d3",
                    "semantic_rule": scenario.task_options.get("semantic_rule"),
                    "semantic_phrases": [rule_case["scenario_phrase"]],
                },
            )
            self._semantic_balance["d3_task"][task_name] += 1
            self._semantic_balance["d3_rule"][scenario.task_options.get("semantic_rule")] += 1
            return item

        raise ValueError("Could not generate a rule-grounded D3 item")

    def _build_d4_rule_case(self, network, setup_rule_name=None):
        """Create one rule-grounded D4 case with a semantic setup and conditional action."""
        meta = self._get_meta(network)
        if meta["n_bus"] < 3:
            raise ValueError("Need at least three buses for D4 semantic rules")

        net = self._load_solved_pf(network)
        reg = _all_registries()
        net_display = reg[network]["display"]

        if setup_rule_name is None:
            _, setup_rule_name = self._pick_semantic_rule(
                network=network,
                allowed_wrappers=set(self._d4_wrapper_specs().keys()),
            )
        compound_type = self._pick_compound_wrapper(network, setup_rule_name)
        rule_case = self._build_semantic_setup_rule(net, setup_rule_name)

        ref_lines = [
            "import pandapower as pp",
            "import pandapower.networks as pn",
        ]
        if network.startswith("sb_"):
            ref_lines.append("import simbench as sb")
        ref_lines += [
            "",
            f"net = {reg[network]['loader']}",
            "",
        ]

        scenario_mods = list(rule_case["modifications"])
        if compound_type == "semantic_then_check":
            if meta["n_line"] < 2:
                raise ValueError("Need at least two lines for semantic_then_check")
            # Threshold is chosen from the post-setup loading level so the branch is meaningful.
            net_mod = _load_network(network)
            for mod in rule_case["modifications"]:
                apply_modification(net_mod, mod)
            with _SuppressPandapowerLogs():
                import pandapower as pp
                pp.runpp(net_mod)
            threshold = round(float(net_mod.res_line["loading_percent"].max()) *
                              self.rng.uniform(0.75, 0.95), 1)
            if threshold <= 0:
                threshold = 1.0
            if float(net_mod.res_line["loading_percent"].max()) > threshold:
                worst_line = self._dominant_idxmax(
                    net_mod.res_line["loading_percent"], min_gap_pct=2.0
                )
                scenario_mods.append(Modification("disconnect_line", {"idx": worst_line}))

            nl = self.rng.choice(
                _get_merged_compound_templates()["semantic_then_check"]
            ).format(
                network_display=net_display,
                scenario_phrase=rule_case["scenario_phrase"],
                threshold=threshold,
            )
            ref_lines += rule_case["setup_lines"] + [
                "# Run PF and shed the currently most loaded line if the threshold is exceeded",
                "pp.runpp(net)",
                'max_loading = net.res_line["loading_percent"].max()',
                f"if max_loading > {threshold}:",
                '    worst_line = net.res_line["loading_percent"].idxmax()',
                '    net.line.at[worst_line, "in_service"] = False',
                "    pp.runpp(net)",
                "",
                'result = net.res_bus["vm_pu"].min()',
                "print(result)",
            ]
            qt = QueryTarget("compound", "res_bus", "vm_pu", threshold=threshold)

        elif compound_type == "semantic_then_compensate":
            net_mod = _load_network(network)
            for mod in rule_case["modifications"]:
                apply_modification(net_mod, mod)
            with _SuppressPandapowerLogs():
                import pandapower as pp
                pp.runpp(net_mod)
            v_threshold = round(float(net_mod.res_bus["vm_pu"].min()) +
                                self.rng.uniform(0.002, 0.02), 3)
            shunt_mvar = round(self.rng.uniform(5, 20), 0)
            if float(net_mod.res_bus["vm_pu"].min()) < v_threshold:
                weak_bus = self._dominant_idxmin(
                    net_mod.res_bus["vm_pu"], min_gap_pct=0.2, min_abs_gap=1e-4
                )
                scenario_mods.append(Modification("add_shunt", {
                    "bus": weak_bus, "q_mvar": -shunt_mvar, "p_mw": 0
                }))

            nl = self.rng.choice(
                _get_merged_compound_templates()["semantic_then_compensate"]
            ).format(
                network_display=net_display,
                scenario_phrase=rule_case["scenario_phrase"],
                v_threshold=v_threshold,
                shunt_mvar=int(shunt_mvar),
            )
            ref_lines += rule_case["setup_lines"] + [
                "# Run PF and reinforce the weakest bus if voltage drops below threshold",
                "pp.runpp(net)",
                'min_v = net.res_bus["vm_pu"].min()',
                f"if min_v < {v_threshold}:",
                '    weak_bus = net.res_bus["vm_pu"].idxmin()',
                f"    pp.create_shunt(net, bus=weak_bus, q_mvar={-shunt_mvar}, p_mw=0)",
                "    pp.runpp(net)",
                "",
                'result = net.res_bus["vm_pu"].min()',
                "print(result)",
            ]
            qt = QueryTarget("compound", "res_bus", "vm_pu", threshold=v_threshold)

        elif compound_type == "semantic_then_redispatch":
            if meta["n_gen"] < 1:
                raise ValueError("Need at least one generator for semantic_then_redispatch")
            net_mod = _load_network(network)
            for mod in rule_case["modifications"]:
                apply_modification(net_mod, mod)
            with _SuppressPandapowerLogs():
                import pandapower as pp
                pp.runpp(net_mod)
            v_threshold = round(float(net_mod.res_bus["vm_pu"].min()) +
                                self.rng.uniform(0.002, 0.02), 3)
            online_gen = net_mod.gen[net_mod.gen["in_service"]]
            if len(online_gen) == 0:
                raise ValueError("No online generators left after setup rule")
            target_gen = self._dominant_idxmax(
                online_gen["p_mw"].abs(), min_gap_pct=2.0, min_abs_gap=1e-4
            )
            redispatch_mw = round(self.rng.uniform(5.0, 30.0), 1)
            if float(net_mod.res_bus["vm_pu"].min()) < v_threshold:
                new_p = round(float(net_mod.gen.at[target_gen, "p_mw"]) + redispatch_mw, 1)
                scenario_mods.append(Modification("set_gen_p", {
                    "idx": target_gen,
                    "value": new_p,
                }))

            nl = self.rng.choice(
                _get_merged_compound_templates()["semantic_then_redispatch"]
            ).format(
                network_display=net_display,
                scenario_phrase=rule_case["scenario_phrase"],
                v_threshold=v_threshold,
                redispatch_mw=redispatch_mw,
            )
            ref_lines += rule_case["setup_lines"] + [
                "# Run PF and redispatch the largest online generator if voltage is too low",
                "pp.runpp(net)",
                'min_v = net.res_bus["vm_pu"].min()',
                f"if min_v < {v_threshold}:",
                '    online_gen = net.gen[net.gen["in_service"]]',
                '    target_gen = online_gen["p_mw"].abs().idxmax()',
                f'    net.gen.at[target_gen, "p_mw"] = net.gen.at[target_gen, "p_mw"] + {redispatch_mw}',
                "    pp.runpp(net)",
                "",
                'result = net.res_bus["vm_pu"].min()',
                "print(result)",
            ]
            qt = QueryTarget("compound", "res_bus", "vm_pu", threshold=v_threshold)

        elif compound_type == "two_stage_rank_then_act":
            if meta["n_line"] < 2:
                raise ValueError("Need at least two lines for two_stage_rank_then_act")
            net_mod = _load_network(network)
            for mod in rule_case["modifications"]:
                apply_modification(net_mod, mod)
            with _SuppressPandapowerLogs():
                import pandapower as pp
                pp.runpp(net_mod)
            ranked = net_mod.res_line["loading_percent"].sort_values(ascending=False)
            if len(ranked) < 2:
                raise ValueError("Need at least two line loading results")
            first_val, second_val = float(ranked.iloc[0]), float(ranked.iloc[1])
            if abs(first_val - second_val) < 1e-4:
                raise ValueError("Top-2 line loadings are too close for stable ranking")
            second_line = int(ranked.index[1])
            scenario_mods.append(Modification("disconnect_line", {"idx": second_line}))

            nl = self.rng.choice(
                _get_merged_compound_templates()["two_stage_rank_then_act"]
            ).format(
                network_display=net_display,
                scenario_phrase=rule_case["scenario_phrase"],
            )
            ref_lines += rule_case["setup_lines"] + [
                "# Run PF, rank lines by loading, and disconnect the second-most loaded line",
                "pp.runpp(net)",
                'ranked_lines = net.res_line["loading_percent"].sort_values(ascending=False)',
                "second_line = ranked_lines.index[1]",
                'net.line.at[second_line, "in_service"] = False',
                "pp.runpp(net)",
                "",
                'result = net.res_line["loading_percent"].max()',
                "print(result)",
            ]
            qt = QueryTarget("compound", "res_line", "loading_percent")

        elif compound_type == "dual_condition_choose_action":
            if meta["n_line"] < 2 or meta["n_bus"] < 3:
                raise ValueError("Need at least two lines and three buses for dual_condition_choose_action")
            net_mod = _load_network(network)
            for mod in rule_case["modifications"]:
                apply_modification(net_mod, mod)
            with _SuppressPandapowerLogs():
                import pandapower as pp
                pp.runpp(net_mod)

            max_loading = float(net_mod.res_line["loading_percent"].max())
            min_v = float(net_mod.res_bus["vm_pu"].min())
            shunt_mvar = round(self.rng.uniform(5.0, 20.0), 0)
            branch_choice = self.rng.choice(["line_action", "voltage_action"])

            if branch_choice == "line_action":
                threshold = round(max_loading * self.rng.uniform(0.8, 0.95), 1)
                if threshold <= 0:
                    threshold = 1.0
                v_threshold = round(max(min_v - self.rng.uniform(0.01, 0.03), 0.8), 3)
                worst_line = self._dominant_idxmax(
                    net_mod.res_line["loading_percent"], min_gap_pct=2.0, min_abs_gap=1e-4
                )
                scenario_mods.append(Modification("disconnect_line", {"idx": worst_line}))
            else:
                threshold = round(max_loading * self.rng.uniform(1.05, 1.25) + 1.0, 1)
                v_threshold = round(min(min_v + self.rng.uniform(0.002, 0.02), 1.2), 3)
                weak_bus = self._dominant_idxmin(
                    net_mod.res_bus["vm_pu"], min_gap_pct=0.2, min_abs_gap=1e-4
                )
                scenario_mods.append(Modification("add_shunt", {
                    "bus": weak_bus,
                    "q_mvar": -shunt_mvar,
                    "p_mw": 0,
                }))

            nl = self.rng.choice(
                _get_merged_compound_templates()["dual_condition_choose_action"]
            ).format(
                network_display=net_display,
                scenario_phrase=rule_case["scenario_phrase"],
                threshold=threshold,
                v_threshold=v_threshold,
                shunt_mvar=int(shunt_mvar),
            )
            ref_lines += rule_case["setup_lines"] + [
                "# Run PF and choose one deterministic corrective action from two conditions",
                "pp.runpp(net)",
                'max_loading = net.res_line["loading_percent"].max()',
                'min_v = net.res_bus["vm_pu"].min()',
                f"if max_loading > {threshold}:",
                '    worst_line = net.res_line["loading_percent"].idxmax()',
                '    net.line.at[worst_line, "in_service"] = False',
                "    pp.runpp(net)",
                f"elif min_v < {v_threshold}:",
                '    weak_bus = net.res_bus["vm_pu"].idxmin()',
                f"    pp.create_shunt(net, bus=weak_bus, q_mvar={-shunt_mvar}, p_mw=0)",
                "    pp.runpp(net)",
                "",
                'result = net.res_bus["vm_pu"].min()',
                "print(result)",
            ]
            qt = QueryTarget("compound", "res_bus", "vm_pu", threshold=v_threshold)

        else:
            baseline_metric = self.rng.choice([
                ("minimum bus voltage (p.u.)", "res_bus", "vm_pu", "min()"),
                ("maximum line loading (%)", "res_line", "loading_percent", "max()"),
            ])
            metric_name, table, column, agg_expr = baseline_metric
            nl = self.rng.choice(
                _get_merged_compound_templates()["semantic_comparison"]
            ).format(
                network_display=net_display,
                scenario_phrase=rule_case["scenario_phrase"],
                metric_name=metric_name,
            )
            ref_lines += [
                "# Run baseline PF and record the requested metric",
                "pp.runpp(net)",
                f'baseline = net.{table}["{column}"].{agg_expr}',
                "",
                "# Apply the semantic setup rule and re-run PF",
            ] + rule_case["setup_lines"] + [
                "pp.runpp(net)",
                f'modified = net.{table}["{column}"].{agg_expr}',
                "",
                "result = abs(modified - baseline)",
                "print(result)",
            ]
            qt = QueryTarget("comparison", table, column)

        scenario = Scenario(
            network=network,
            task=f"compound_{compound_type}",
            modifications=scenario_mods,
            query_target=qt,
            task_options={
                "semantic_source": "rule_grounded_d4",
                "semantic_rule": setup_rule_name,
                "compound_type": compound_type,
                "rule_metadata": rule_case.get("metadata", {}),
            },
        )
        return scenario, nl, "\n".join(ref_lines), [rule_case["scenario_phrase"]]

    def _execute_reference_code(self, code: str):
        """Run reference code and parse its printed scalar result."""
        stdout_capture = io.StringIO()
        ns = {}
        with _SuppressPandapowerLogs(), contextlib.redirect_stdout(stdout_capture):
            exec(code, ns)
        out = stdout_capture.getvalue().strip()
        try:
            return round(float(out), 6), "float"
        except ValueError:
            if out in ("True", "False"):
                return out == "True", "bool"
            return int(float(out)), "int"

    def generate_derived_compound_item(self, network=None, max_tries: int = 40):
        """Generate a rule-grounded D4 item aligned to one D2 generator family."""
        for _ in range(max_tries):
            try:
                task_name, picked_network, setup_rule_name = (
                    self._pick_semantic_multi_task_and_network(network=network)
                )
                base_net = self._load_solved_pf(picked_network)
                rule_case = self._build_semantic_setup_rule(
                    base_net,
                    setup_rule_name,
                )
                mod_desc_override = self._semantic_mod_desc_override(
                    rule_case["scenario_phrase"]
                )
                item = self._multi_generators()[task_name](
                    network=picked_network,
                    base_mods=rule_case["modifications"],
                    mod_desc_override=mod_desc_override,
                )
                item.scenario.task_options = {
                    **(item.scenario.task_options or {}),
                    "semantic_source": "rule_grounded_d4",
                    "semantic_rule": setup_rule_name,
                    "rule_metadata": rule_case.get("metadata", {}),
                    "base_task": task_name,
                }
            except Exception:
                continue

            out = BenchmarkItem(
                id=self._make_variant_id(
                    item,
                    "rule_d4",
                    item.natural_language_query,
                ),
                scenario=item.scenario,
                natural_language_query=item.natural_language_query,
                reference_code=item.reference_code,
                ground_truth=item.ground_truth,
                ground_truth_type=item.ground_truth_type,
                difficulty_level="D4_compound",
                eval_criteria={
                    "semantic_source": "rule_grounded_d4",
                    "semantic_rule": item.scenario.task_options.get("semantic_rule"),
                    "semantic_phrases": [rule_case["scenario_phrase"]],
                    "base_task": task_name,
                },
            )
            self._semantic_balance["d4_task"][task_name] += 1
            self._semantic_balance["d4_rule"][item.scenario.task_options.get("semantic_rule")] += 1
            return out
        raise ValueError("Could not generate a rule-grounded D4 item")

    def generate_semantic_item(self, network=None):
        """Generate one rule-grounded D3 item with hidden-but-deterministic setup."""
        return self.generate_derived_semantic_item(network=network)

    def generate_compound_item(self, network=None):
        """Generate one rule-grounded D4 item with semantic setup + multi-step logic."""
        return self.generate_derived_compound_item(network=network)


    def generate(self, n=100, tasks=None, multi_task_ratio=0.0,
                    semantic_ratio=0.0,     # ← NEW: fraction of D3 items
                    compound_ratio=0.0,     # ← NEW: fraction of D4 items
                    auto_validate=True, tolerance=1e-3, verbose=True,
                    calibrate=True):        # kept for API compatibility; unused
        """
        Generate n benchmark items across D1-D4 difficulty levels.
    
        Args:
            n: total number of items
            tasks: single-task types for D1 (default: all)
            multi_task_ratio: fraction for D2 multi-step
            semantic_ratio: fraction for D3 semantic
            compound_ratio: fraction for D4 compound
            auto_validate: validate D1/D2 items
            tolerance: float matching tolerance for D1/D2
            calibrate: legacy argument kept for API compatibility
        """
        import warnings
        warnings.filterwarnings("ignore")
    
        if tasks is None:
            tasks = list(TASK_CONFIG.keys())
    
        # Item count per difficulty
        n_semantic = int(n * semantic_ratio)
        n_compound = int(n * compound_ratio)
        n_multi = int(n * multi_task_ratio)
        n_single = n - n_multi - n_semantic - n_compound
    
        items = []
        stats = {
            "attempts": 0, "success": 0, "gen_fail": 0,
            "validate_fail": 0, "duplicate": 0,
            "gen_fail_detail": {}, "validate_fail_detail": [],
            "per_difficulty": {"D1_basic": 0, "D2_multi_step": 0,
                            "D3_semantic": 0, "D4_compound": 0},
        }
        seen_ids = set()
        progress_every = max(10, n // 20)
        attempt_every = max(100, n // 10)
        progress_time_sec = 120.0
        last_progress_success = 0
        last_progress_attempt = 0
        last_progress_time = time.time()

        def _report_progress(stage, force=False):
            nonlocal last_progress_success, last_progress_attempt, last_progress_time
            if not verbose:
                return
            now = time.time()
            should_print = (
                force
                or stats["success"] - last_progress_success >= progress_every
                or stats["attempts"] - last_progress_attempt >= attempt_every
                or now - last_progress_time >= progress_time_sec
            )
            if not should_print:
                return
            last_progress_success = stats["success"]
            last_progress_attempt = stats["attempts"]
            last_progress_time = now
            d1 = stats["per_difficulty"]["D1_basic"]
            d2 = stats["per_difficulty"]["D2_multi_step"]
            d3 = stats["per_difficulty"]["D3_semantic"]
            d4 = stats["per_difficulty"]["D4_compound"]
            print(
                f"  [{stage}] done={stats['success']}/{n} "
                f"(D1={d1}/{n_single}, D2={d2}/{n_multi}, "
                f"D3={d3}/{n_semantic}, D4={d4}/{n_compound}) "
                f"attempts={stats['attempts']} gen_fail={stats['gen_fail']} "
                f"validate_fail={stats['validate_fail']} duplicate={stats['duplicate']}",
                flush=True,
            )
    
        def _try_gen(label, gen_fn, difficulty, need_validate=True):
            stats["attempts"] += 1
            try:
                item = gen_fn()
            except Exception as e:
                stats["gen_fail"] += 1
                stats["gen_fail_detail"][label] = stats["gen_fail_detail"].get(label, 0) + 1
                return None
    
            if item.id in seen_ids:
                stats["duplicate"] += 1
                return None
            seen_ids.add(item.id)
    
            # Set difficulty
            item.difficulty_level = difficulty
    
            # Validate every exact-match item that has reference code.
            if need_validate and auto_validate and item.reference_code.strip():
                r = validate_item(item, tolerance)
                if not r["match"]:
                    stats["validate_fail"] += 1
                    return None
    
            stats["success"] += 1
            stats["per_difficulty"][difficulty] += 1
            _report_progress(difficulty)
            return item
    
        if verbose:
            print(f"Generating {n} items: D1={n_single} D2={n_multi} "
                f"D3={n_semantic} D4={n_compound}", flush=True)
    
        # ── D1: single-task ──
        for _ in range(n_single * 3):
            if stats["per_difficulty"]["D1_basic"] >= n_single:
                break
            task = self.rng.choice(tasks)
            item = _try_gen(task, lambda t=task: self.generate_one(task=t), "D1_basic")
            if item:
                items.append(item)
            else:
                _report_progress("D1_basic")
    
        # ── D2: multi-task ──
        multi_generators = {
            "comparison": self.generate_comparison,
            "comparison_opf": self.generate_comparison_opf,
            "sequential": self.generate_sequential,
            "pf_then_sc": self.generate_pf_then_sc,
            "contingency_fix": self.generate_contingency_fix,
            "parallel": self.generate_parallel,
            "diagnose_and_fix": self.generate_diagnose_and_fix,
        }
        for _ in range(n_multi * 5):
            if stats["per_difficulty"]["D2_multi_step"] >= n_multi:
                break
            mt = self.rng.choice(list(multi_generators.keys()))
            item = _try_gen(mt, multi_generators[mt], "D2_multi_step")
            if item:
                items.append(item)
            else:
                _report_progress("D2_multi_step")
    
        # ── D3: semantic ──
        for _ in range(n_semantic * 4):
            if stats["per_difficulty"]["D3_semantic"] >= n_semantic:
                break
            item = _try_gen("semantic", self.generate_semantic_item, "D3_semantic")
            if item:
                items.append(item)
            else:
                _report_progress("D3_semantic")
    
        # ── D4: compound ──
        for _ in range(n_compound * 5):
            if stats["per_difficulty"]["D4_compound"] >= n_compound:
                break
            item = _try_gen("compound", self.generate_compound_item, "D4_compound")
            if item:
                items.append(item)
            else:
                _report_progress("D4_compound")
    
        if verbose:
            _report_progress("final", force=True)
            print(f"\nDone: {len(items)}/{n} items")
            for d, c in stats["per_difficulty"].items():
                print(f"  {d}: {c}")
            print(f"  attempts={stats['attempts']} gen_fail={stats['gen_fail']} "
                f"validate_fail={stats['validate_fail']} duplicate={stats['duplicate']}",
                flush=True)
    
        return items
 
 
    @staticmethod
    def print_stats(items):
        """Print comprehensive statistics about generated benchmark items."""
        from collections import Counter

        def fmt_counter(c, tag_set=None, tag="[M]"):
            """Format a counter as a single horizontal string."""
            parts = []
            for k, v in sorted(c.items(), key=lambda x: -x[1]):
                suffix = f" {tag}" if (tag_set and k not in tag_set) else ""
                parts.append(f"{k}:{v}{suffix}")
            return "  ".join(parts)

        single_tasks = set(TASK_CONFIG.keys())
        multi_items  = [i for i in items if i.scenario.task not in single_tasks]
        single_items = [i for i in items if i.scenario.task in single_tasks]

        print(f"\n{'='*60}")
        print(f"BENCHMARK STATISTICS ({len(items)} items)")
        print(f"{'='*60}")
        print(f"  Single-task: {len(single_items)}  |  Multi-task: {len(multi_items)}")

        tc = Counter(i.scenario.task for i in items)
        print(f"\n  Tasks ({len(tc)} types):")
        print(f"    {fmt_counter(tc, tag_set=single_tasks)}")

        dc = Counter(i.difficulty_level for i in items)
        print(f"\n  Difficulty Levels ({len(dc)} levels):")
        for d in ["D1_basic", "D2_multi_step", "D3_semantic", "D4_compound"]:
            c = dc.get(d, 0)
            pct = c / len(items) * 100 if items else 0
            bar = "█" * max(1, round(pct / 5)) + "░" * (20 - max(1, round(pct / 5)))
            print(f"    {d:20s}: {c:4d} ({pct:5.1f}%)  {bar}")

        nc = Counter(i.scenario.network for i in items)
        print(f"\n  Networks ({len(nc)} unique):")
        print(f"    {fmt_counter(nc)}")

        qc = Counter(i.scenario.query_target.qtype for i in items)
        print(f"  Query types:     {fmt_counter(qc)}")

        mc = Counter(len(i.scenario.modifications) for i in items)
        print(f"  # Modifications: {fmt_counter(mc)}")

        gc = Counter(i.ground_truth_type for i in items)
        print(f"  GT types:        {fmt_counter(gc)}")

        mod_ops = Counter(m.op for i in items for m in i.scenario.modifications)
        if mod_ops:
            print(f"\n  Mod ops ({len(mod_ops)} types):")
            print(f"    {fmt_counter(mod_ops)}")

        code_lengths = [len(i.reference_code.split("\n")) for i in items]
        nl_lengths   = [len(i.natural_language_query) for i in items]
        print(f"\n  Ref code: avg={sum(code_lengths)/len(code_lengths):.0f} lines  "
            f"min={min(code_lengths)}  max={max(code_lengths)}")
        print(f"  NL query: avg={sum(nl_lengths)/len(nl_lengths):.0f} chars  "
            f"min={min(nl_lengths)}  max={max(nl_lengths)}")
        print(f"{'='*60}")
    
 
    @staticmethod
    def save(items, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([item.to_dict() for item in items], f, indent=2, ensure_ascii=False)
        print(f"Saved {len(items)} benchmark items to {filepath}")
 
    @staticmethod
    def load(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
 
    # ----- LLM Template Expansion -----
 
    @staticmethod
    def expand_templates(api_key=None, n_per_template=5, output_path=None,
                         delay=0.5, reset_existing=False):
        """
        Expand D1/D2/D3/D4 NL templates via OpenAI API.
        Saves to JSON file that is auto-loaded on next generate() call.
 
        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            n_per_template: minimum number of expanded paraphrases to keep per template key.
            output_path: where to save (default: EXPANDED_TEMPLATES_PATH from config).
            delay: seconds between API calls (rate limiting).
            reset_existing: if True, ignore previous expanded templates and rebuild from base templates.
        """
        import urllib.request
 
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            print("ERROR: Provide api_key or set OPENAI_API_KEY env variable.")
            return

        cfg = LLM_EXPANSION_CONFIG
        save_path = output_path or EXPANDED_TEMPLATES_PATH

        def _normalize_keys(section_dict):
            normalized = {}
            for key, templates in section_dict.items():
                normalized[repr(key) if not isinstance(key, str) else key] = list(templates)
            return normalized

        def _denormalize_keys(section_dict):
            denorm = {}
            for key, templates in section_dict.items():
                parsed_key = eval(key) if isinstance(key, str) and key.startswith("(") else key
                denorm[parsed_key] = list(templates)
            return denorm

        def _merge_base_and_expanded(base_dict, expanded_dict):
            merged = {}
            for key, base_templates in base_dict.items():
                normalized_key = repr(key) if not isinstance(key, str) else key
                merged[normalized_key] = list(base_templates)
                seen = set(merged[normalized_key])
                for t in expanded_dict.get(normalized_key, []):
                    if t not in seen:
                        merged[normalized_key].append(t)
                        seen.add(t)
            return merged

        def _required_placeholder_hint(section, label_key):
            return _required_template_hint(section, label_key)

        existing_data = {}
        if not reset_existing and os.path.exists(save_path):
            try:
                with open(save_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = {}
 
        # Gather all templates across D1-D4.
        base_sections = {
            "single": _normalize_keys(NL_TEMPLATES),
            "multi": _normalize_keys(MULTI_TASK_NL_TEMPLATES),
            "semantic": _normalize_keys(SEMANTIC_NL_TEMPLATES),
            "compound": _normalize_keys(COMPOUND_NL_TEMPLATES),
        }

        all_source = {}
        for section, templates_dict in base_sections.items():
            for normalized_key, templates in templates_dict.items():
                all_source[(section, normalized_key)] = templates

        existing_sections = {}
        for section in ("single", "multi", "semantic", "compound"):
            raw_section = existing_data.get(section, {})
            existing_sections[section] = {
                key: [t for t in value if isinstance(t, str)]
                for key, value in raw_section.items()
                if isinstance(value, list)
            }

        def call_api(messages):
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
            body = json.dumps({
                "model": cfg["model"], "messages": messages,
                "temperature": 0.9, "max_tokens": 2000,
            }).encode()
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=body, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]

        expanded = {}
        all_keys = list(all_source.keys())
        print(f"Expanding {len(all_keys)} template keys to >= {n_per_template} expanded variants each...")
 
        for i, (cat, normalized_key) in enumerate(all_keys):
            base_templates = all_source[(cat, normalized_key)]
            current_expanded = list(existing_sections.get(cat, {}).get(normalized_key, []))
            missing = max(0, n_per_template - len(current_expanded))
            expanded[(cat, normalized_key)] = current_expanded

            label_key = _parse_template_key(normalized_key)
            print(f"  [{i+1}/{len(all_keys)}] [{cat}] {label_key} "
                  f"(base={len(base_templates)} expanded={len(current_expanded)} missing={missing})...",
                  end=" ", flush=True)

            if missing == 0:
                print("skip")
                continue

            existing_str = "\n".join(f"  - {t}" for t in (base_templates + current_expanded))
 
            accepted = 0
            rejected = 0
            last_error = None
            seen = set(current_expanded)
            max_rounds = 3
            for _round in range(max_rounds):
                need_now = max(0, n_per_template - len(current_expanded))
                if need_now == 0:
                    break
                try:
                    messages = [
                        {"role": "system", "content": cfg["system_prompt"]},
                        {"role": "user", "content": cfg["user_template"].format(
                            key=str(label_key),
                            existing=existing_str,
                            n=max(need_now, missing),
                            required_placeholders=_required_placeholder_hint(cat, label_key),
                        )},
                    ]
                    text = call_api(messages).strip()
                    if text.startswith("```"):
                        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
                    new = json.loads(text)
                    for tmpl in new:
                        if not isinstance(tmpl, str) or tmpl in seen:
                            continue
                        tmpl = _repair_expanded_template(cat, tmpl)
                        if tmpl in seen:
                            continue
                        err = _template_validation_error(cat, label_key, tmpl)
                        if err is not None:
                            rejected += 1
                            last_error = err
                            continue
                        current_expanded.append(tmpl)
                        seen.add(tmpl)
                        accepted += 1
                    existing_str = "\n".join(
                        f"  - {t}" for t in (base_templates + current_expanded)
                    )
                except Exception as e:
                    last_error = str(e)
                    break

            expanded[(cat, normalized_key)] = current_expanded
            if len(current_expanded) >= n_per_template:
                print(
                    f"got {len(current_expanded)} total expanded "
                    f"(+{accepted} accepted, {rejected} rejected)"
                )
            else:
                suffix = f", last_error={last_error}" if last_error else ""
                print(
                    f"partial {len(current_expanded)}/{n_per_template} "
                    f"(+{accepted} accepted, {rejected} rejected{suffix})"
                )
 
            if delay > 0 and i < len(all_keys) - 1:
                time.sleep(delay)
 
        # Merge base + expanded per section.
        merged_single = _merge_base_and_expanded(base_sections["single"], {
            k: v for (cat, k), v in expanded.items() if cat == "single"
        })
        merged_multi = _merge_base_and_expanded(base_sections["multi"], {
            k: v for (cat, k), v in expanded.items() if cat == "multi"
        })
        merged_semantic = _merge_base_and_expanded(base_sections["semantic"], {
            k: v for (cat, k), v in expanded.items() if cat == "semantic"
        })
        merged_compound = _merge_base_and_expanded(base_sections["compound"], {
            k: v for (cat, k), v in expanded.items() if cat == "compound"
        })

        # Validate templates before saving.
        validation_plan = [
            ("single", merged_single, validate_templates),
            ("semantic", _denormalize_keys(merged_semantic), validate_semantic_templates),
            ("compound", _denormalize_keys(merged_compound), validate_compound_templates),
        ]
        for section_name, section_templates, validator in validation_plan:
            vr = validator(section_templates)
            if vr["invalid"] > 0:
                print(f"WARNING: {vr['invalid']} invalid {section_name} templates detected")
                for err in vr["errors"]:
                    key = err["key"]
                    normalized_key = repr(key) if not isinstance(key, str) else key
                    merged_target = {
                        "single": merged_single,
                        "semantic": merged_semantic,
                        "compound": merged_compound,
                    }[section_name]
                    if normalized_key in merged_target:
                        merged_target[normalized_key] = [
                            t for t in merged_target[normalized_key]
                            if t[:80] != err["template"]
                        ]

        # Save as JSON (auto-loaded on next generate() call)
        json_data = {
            "single": merged_single,
            "multi": merged_multi,
            "semantic": merged_semantic,
            "compound": merged_compound,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
 
        # Invalidate cache so next generate() picks up the new templates
        global _merged_templates_cache, _merged_multi_cache
        global _merged_semantic_cache, _merged_compound_cache
        _merged_templates_cache = None
        _merged_multi_cache = None
        _merged_semantic_cache = None
        _merged_compound_cache = None
 
        n_before = sum(len(v) for v in NL_TEMPLATES.values()) + \
                   sum(len(v) for v in MULTI_TASK_NL_TEMPLATES.values()) + \
                   sum(len(v) for v in SEMANTIC_NL_TEMPLATES.values()) + \
                   sum(len(v) for v in COMPOUND_NL_TEMPLATES.values())
        n_after = sum(len(v) for v in merged_single.values()) + \
                  sum(len(v) for v in merged_multi.values()) + \
                  sum(len(v) for v in merged_semantic.values()) + \
                  sum(len(v) for v in merged_compound.values())
        print(f"\nBefore: {n_before} templates | After: {n_after} | Added: {n_after - n_before}")
        print(f"Saved to {save_path} (auto-loaded on next generate)")

if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="Power System Benchmark Engine")

    # --- Pipeline selection ---
    parser.add_argument("--generate", action="store_true", help="Run generate pipeline")
    parser.add_argument("--expand", action="store_true", help="Run expand pipeline")

    # --- generate arguments ---
    parser.add_argument("-n", type=int, default=20)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-large", action="store_true", help="Include large networks")
    parser.add_argument("-o", "--output", default="benchmark.json")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--multi-task-ratio", type=float, default=0.0, help="Ratio of multi-task scenarios")
    parser.add_argument("--semantic-ratio", type=float, default=0.0, help="Ratio of D3 semantic scenarios")
    parser.add_argument("--compound-ratio", type=float, default=0.0, help="Ratio of D4 compound scenarios")
    parser.add_argument("--large-ratio", type=float, default=0.3, help="Ratio of large networks")

    # --- expand arguments ---
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    parser.add_argument("--n-per-template", type=int, default=5)
    parser.add_argument("--expand-output", default=EXPANDED_TEMPLATES_PATH,
                        help="Output path for expanded templates")
    parser.add_argument("--expand-reset", action="store_true",
                        help="Rebuild expanded templates from base templates and ignore existing JSON")

    # Hard-coded defaults for the frozen release; override on the CLI if needed.
    parser.set_defaults(
        generate=True,
        expand=False,
        include_large=True,
        validate=True,
        n=2000,
        seed=42,
        multi_task_ratio=0.3,
        semantic_ratio=0.20,
        compound_ratio=0.20,
        large_ratio=0.3,
        output=str(BENCHMARK_DIR / "benchmark.json"),
        expand_output=EXPANDED_TEMPLATES_PATH,
        expand_reset=False,
        n_per_template=8,
        api_key=None,
    )

    args = parser.parse_args()

    # If neither flag is passed, default to generate-only.
    if not args.generate and not args.expand:
        args.generate = True

    # --- Run generate ---
    if args.generate:
        print("=== Running: generate ===")
        engine = BenchmarkEngine(seed=args.seed, include_large=args.include_large, large_ratio=args.large_ratio)
        items = engine.generate(n=args.n, tasks=args.tasks, multi_task_ratio=args.multi_task_ratio,
                                semantic_ratio=args.semantic_ratio,    
                                compound_ratio=args.compound_ratio,      
                                auto_validate=args.validate, verbose=True)
        engine.save(items, args.output)
        engine.print_stats(items)
        print(f"Saved {len(items)} items to {args.output}")

        if args.validate:
            print("\nRunning consistency validation...")
            report = validate_batch(items, max_items=len(items))
            print(f"Consistency: {report['match']}/{report['effective_total']} "
                f"({report['consistency_rate']:.0%}), skipped={report['skipped']}")

    # --- Run expand ---
    if args.expand:
        print("=== Running: expand ===")
        BenchmarkEngine.expand_templates(
            api_key=args.api_key,
            n_per_template=args.n_per_template,
            output_path=args.expand_output,
            reset_existing=args.expand_reset,
        )
        validation_reports = {
            "single": validate_templates(),
            "semantic": validate_semantic_templates(_get_merged_semantic_templates()),
            "compound": validate_compound_templates(_get_merged_compound_templates()),
        }
        for section, report in validation_reports.items():
            print(
                f"{section} templates: "
                f"{report['valid']} valid, {report['invalid']} invalid"
            )
            for err in report["errors"]:
                print(f"  {err['key']}: {err['error']}")
                print(f"    {err['template']}...")



    if args.generate:
        from collections import Counter
        print(f'Total: {len(items)} items')
        print(f'Tasks:    {dict(Counter(i.scenario.task for i in items))}')
        print(f'Networks: {dict(Counter(i.scenario.network for i in items))}')
        print(f'Q-types:  {dict(Counter(i.scenario.query_target.qtype for i in items))}')
        print(f'# Mods:   {dict(Counter(len(i.scenario.modifications) for i in items))}')
        print(f'GT types: {dict(Counter(i.ground_truth_type for i in items))}')
