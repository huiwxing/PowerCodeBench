# --------------------------------------------------------------------------
# Repository copy of benchmark/benchmark_config.py, unmodified apart from this
# header. Runs CPU-only against the archived corpus/spec files in this
# repository. See code/README.md for the module map.
# --------------------------------------------------------------------------
"""
Benchmark Engine Configuration (v6)
====================================
Changes from v5:
  - Added DIFFICULTY_LEVELS taxonomy (D1-D4)
  - Added SEMANTIC_NL_TEMPLATES for D3 query rewriting
  - Added COMPOUND_NL_TEMPLATES for D4 query rewriting
  - Added TASK_FUNCTION_PATTERNS for diagnostic checks
  - All existing D1/D2 structures unchanged

Difficulty taxonomy:
  D1_basic:      Single-step, explicit parameters, direct query
  D2_multi_step: Multi-step, explicit parameters, conditional/comparison logic
  D3_semantic:   Single-step, derived semantic rewrite with hidden-but-recoverable intent
  D4_compound:   Multi-step, derived semantic rewrite with hidden-but-recoverable intent
"""

from pathlib import Path

# ===========================================================================
# 1. DIFFICULTY LEVELS  (NEW)
# ===========================================================================

DIFFICULTY_LEVELS = {
    "D1_basic":      {"label": "D1: Basic Explicit",       "description": "Single-step with explicit parameters"},
    "D2_multi_step": {"label": "D2: Multi-Step Explicit",  "description": "Multi-step with explicit parameters"},
    "D3_semantic":   {"label": "D3: Semantic Single-Step",  "description": "Single-step with scenario-driven implicit parameters"},
    "D4_compound":   {"label": "D4: Semantic Compound",     "description": "Multi-step with semantic inference + conditional logic"},
}

# Map multi-task generator names → D2
MULTI_TASK_DIFFICULTY = {
    "comparison": "D2_multi_step", "comparison_opf": "D2_multi_step",
    "sequential": "D2_multi_step", "pf_then_sc": "D2_multi_step",
    "contingency_fix": "D2_multi_step", "parallel": "D2_multi_step",
    "diagnose_and_fix": "D2_multi_step",
}

SEMANTIC_BALANCE_CONFIG = {
    "d3_task_weights": {
        "power_flow": 1.0,
        "dc_power_flow": 1.0,
        "opf": 1.25,
        "short_circuit_3ph": 1.0,
        "short_circuit_2ph": 1.0,
        "contingency": 1.0,
        "time_series": 1.0,
        "state_estimation": 1.0,
    },
    "d4_task_weights": {
        "comparison": 1.0,
        "comparison_opf": 1.4,
        "sequential": 1.0,
        "pf_then_sc": 0.85,
        "contingency_fix": 1.25,
        "parallel": 1.0,
        "diagnose_and_fix": 1.0,
    },
    "d3_rule_weights": {
        "largest_generator_trip": 1.25,
        "longest_line_outage": 1.15,
    },
    "d4_rule_weights": {
        "largest_generator_trip": 1.2,
        "highest_voltage_shunt_absorption": 1.15,
    },
}


# ===========================================================================
# 2. SEMANTIC NL TEMPLATES (D3)
# ===========================================================================

SEMANTIC_NL_TEMPLATES = {
    ("load_scenario", "min_voltage"): [
        "On the {network_display}, please {scenario_phrase}. Then run a power flow analysis and report the minimum bus voltage (p.u.) across the network.",
        "Using the {network_display}, first {scenario_phrase}. After running load flow, what is the lowest bus voltage magnitude in per-unit?",
        "For the {network_display}, {scenario_phrase}. Perform power flow and return the minimum bus voltage (p.u.).",
    ],
    ("load_scenario", "max_loading"): [
        "On the {network_display}, please {scenario_phrase}. Run power flow and report the maximum line loading percentage.",
        "For the {network_display}, {scenario_phrase}. What is the peak line loading (%) after running power flow?",
    ],
    ("load_scenario", "total_loss"): [
        "On the {network_display}, please {scenario_phrase}. Run power flow and report the total active power losses (MW) across all lines.",
        "For the {network_display}, {scenario_phrase}. What are the total line losses in MW after running power flow?",
    ],
    ("load_scenario", "ext_grid_p"): [
        "On the {network_display}, please {scenario_phrase}. Run power flow and report the active power supplied by the external grid (MW).",
    ],
    ("gen_scenario", "min_voltage"): [
        "On the {network_display}, please {scenario_phrase}. Run a power flow and report the minimum bus voltage (p.u.).",
        "For the {network_display}, {scenario_phrase}. Perform power flow and report the lowest voltage in the network (p.u.).",
    ],
    ("gen_scenario", "max_loading"): [
        "On the {network_display}, please {scenario_phrase}. Run power flow and report the maximum line loading (%).",
    ],
    ("gen_scenario", "ext_grid_p"): [
        "On the {network_display}, please {scenario_phrase}. Run power flow and report the total external grid active power injection (MW).",
    ],
    ("topo_scenario", "max_loading"): [
        "On the {network_display}, please {scenario_phrase}. Run power flow and report the maximum line loading (%).",
    ],
    ("topo_scenario", "min_voltage"): [
        "On the {network_display}, please {scenario_phrase}. Run power flow and report the minimum bus voltage (p.u.).",
    ],
    ("voltage_scenario", "min_voltage"): [
        "On the {network_display}, please {scenario_phrase}. Run power flow and report the minimum bus voltage (p.u.).",
    ],
    ("add_element_scenario", "min_voltage"): [
        "On the {network_display}, please {scenario_phrase}. Run power flow and report the minimum bus voltage (p.u.).",
    ],
    ("add_element_scenario", "max_loading"): [
        "On the {network_display}, please {scenario_phrase}. Run power flow and report the maximum line loading (%).",
    ],
    ("add_element_scenario", "ext_grid_p"): [
        "On the {network_display}, please {scenario_phrase}. Run power flow and report the external grid active power injection (MW).",
    ],
}


# ===========================================================================
# 3. COMPOUND NL TEMPLATES (D4)
# ===========================================================================

COMPOUND_NL_TEMPLATES = {
    "semantic_then_check": [
        "On the {network_display}, first {scenario_phrase}. Run power flow. If any line exceeds {threshold}% loading, disconnect the most loaded line and re-run power flow. Report the final minimum bus voltage (p.u.).",
        "Using the {network_display}, please {scenario_phrase}. Perform load flow. If the maximum line loading exceeds {threshold}%, take the worst-loaded line out of service, re-run power flow, and report the minimum bus voltage (p.u.).",
    ],
    "semantic_then_compensate": [
        "On the {network_display}, first {scenario_phrase}. Run power flow. If the minimum bus voltage drops below {v_threshold} p.u., add a {shunt_mvar} Mvar shunt capacitor at the lowest-voltage bus and re-run power flow. Report the final minimum bus voltage (p.u.).",
    ],
    "semantic_comparison": [
        "On the {network_display}, first run a baseline power flow and record the {metric_name}. Then {scenario_phrase}, re-run power flow, and report the change in {metric_name}.",
        "Using the {network_display}, compare the {metric_name} before and after this rule-based adjustment: {scenario_phrase}. Report the absolute difference.",
    ],
    "semantic_then_redispatch": [
        "On the {network_display}, first {scenario_phrase}. Run power flow. If the minimum bus voltage is below {v_threshold} p.u., find the online generator with the highest active power dispatch, increase that generator by {redispatch_mw} MW, re-run power flow, and report the final minimum bus voltage (p.u.).",
    ],
    "two_stage_rank_then_act": [
        "On the {network_display}, first {scenario_phrase}. Run power flow, rank all lines by loading, disconnect the second-most loaded in-service line, re-run power flow, and report the final maximum line loading (%).",
    ],
    "dual_condition_choose_action": [
        "On the {network_display}, first {scenario_phrase}. Run power flow. If the maximum line loading exceeds {threshold}%, disconnect the most loaded line; otherwise, if the minimum bus voltage is below {v_threshold} p.u., add a {shunt_mvar} Mvar shunt capacitor at the lowest-voltage bus. Re-run power flow if an action was taken and report the final minimum bus voltage (p.u.).",
    ],
}

# ===========================================================================
# 4. DIAGNOSTIC CHECK CONFIG
# ===========================================================================

TASK_FUNCTION_PATTERNS = {
    "power_flow":        [r"pp\.runpp\s*\(", r"runpp\s*\("],
    "dc_power_flow":     [r"pp\.rundcpp\s*\(", r"rundcpp\s*\("],
    "opf":               [r"pp\.runopp\s*\(", r"runopp\s*\("],
    "short_circuit_3ph": [r"calc_sc\s*\("],
    "short_circuit_2ph": [r"calc_sc\s*\("],
    "contingency":       [r"run_contingency\s*\("],
    "time_series":       [r"run_timeseries\s*\("],
    "state_estimation":  [r"estimate\s*\("],
}


# ===========================================================================
# 7–15. UNCHANGED FROM v5  (NETWORK_REGISTRY, MODIFICATION_DEFS, etc.)
# ===========================================================================

NETWORK_REGISTRY = {
    "case4gs":             {"loader": "pn.case4gs()",             "display": "4-bus test case"},
    "case5":               {"loader": "pn.case5()",               "display": "5-bus PJM case"},
    "case6ww":             {"loader": "pn.case6ww()",             "display": "6-bus Ward-Hale case"},
    "case9":               {"loader": "pn.case9()",               "display": "9-bus WSCC case"},
    "case14":              {"loader": "pn.case14()",              "display": "IEEE 14-bus case"},
    "case24_ieee_rts":     {"loader": "pn.case24_ieee_rts()",    "display": "IEEE 24-bus RTS"},
    "case30":              {"loader": "pn.case30()",              "display": "30-bus test case"},
    "case_ieee30":         {"loader": "pn.case_ieee30()",        "display": "IEEE 30-bus case"},
    "case33bw":            {"loader": "pn.case33bw()",           "display": "33-bus Baran-Wu case"},
    "case39":              {"loader": "pn.case39()",             "display": "IEEE 39-bus New England"},
    "case57":              {"loader": "pn.case57()",             "display": "IEEE 57-bus case"},
    "case89pegase":        {"loader": "pn.case89pegase()",       "display": "PEGASE 89-bus"},
    "case118":             {"loader": "pn.case118()",            "display": "IEEE 118-bus case"},
    "case145":             {"loader": "pn.case145()",            "display": "145-bus case"},
    "case200_illinois":    {"loader": "pn.case_illinois200()",   "display": "Illinois 200-bus"},
    "case300":             {"loader": "pn.case300()",            "display": "300-bus test case"},
    "case1354pegase":      {"loader": "pn.case1354pegase()",     "display": "PEGASE 1354-bus"},
    "case2869pegase":      {"loader": "pn.case2869pegase()",     "display": "PEGASE 2869-bus"},
    "case9241pegase":      {"loader": "pn.case9241pegase()",     "display": "PEGASE 9241-bus"},
    "case1888rte":         {"loader": "pn.case1888rte()",        "display": "RTE 1888-bus French grid"},
    "case2848rte":         {"loader": "pn.case2848rte()",        "display": "RTE 2848-bus French grid"},
    "case6470rte":         {"loader": "pn.case6470rte()",        "display": "RTE 6470-bus French grid"},
    "case3120sp":          {"loader": "pn.case3120sp()",         "display": "Polish 3120-bus grid"},
    "GBnetwork":           {"loader": "pn.GBnetwork()",          "display": "GB 2224-bus network"},
    "GBreducednetwork":    {"loader": "pn.GBreducednetwork()",   "display": "GB reduced 29-bus network"},
    "iceland":             {"loader": "pn.iceland()",            "display": "Iceland 189-bus grid"},
    "simple_four_bus":         {"loader": "pn.simple_four_bus_system()", "display": "simple 4-bus system"},
    "example_simple":          {"loader": "pn.example_simple()",        "display": "pandapower example network"},
    "example_multivoltage":    {"loader": "pn.example_multivoltage()",  "display": "multi-voltage example"},
    "four_loads_branches":     {"loader": "pn.four_loads_with_branches_out()", "display": "4-loads-with-branches network"},
    "synth_volt_ctrl_lv":      {"loader": "pn.create_synthetic_voltage_control_lv_network()", "display": "synthetic voltage control LV network"},
    "cigre_mv":  {"loader": "pn.create_cigre_network_mv()", "display": "CIGRE MV benchmark"},
    "cigre_lv":  {"loader": "pn.create_cigre_network_lv()", "display": "CIGRE LV benchmark"},
    "cigre_hv":  {"loader": "pn.create_cigre_network_hv()", "display": "CIGRE HV benchmark"},
    "kerber_dorfnetz":         {"loader": "pn.create_kerber_dorfnetz()",                  "display": "Kerber village (Dorf) LV network"},
    "kerber_landnetz_f1":      {"loader": "pn.create_kerber_landnetz_freileitung_1()",    "display": "Kerber rural overhead LV network"},
    "kerber_landnetz_k1":      {"loader": "pn.create_kerber_landnetz_kabel_1()",          "display": "Kerber rural cable LV network"},
    "kerber_vorstadtnetz_k1":  {"loader": "pn.create_kerber_vorstadtnetz_kabel_1()",      "display": "Kerber suburban cable LV network"},
    "oberrhein":             {"loader": 'pn.mv_oberrhein(scenario="generation")', "display": "MV Oberrhein network"},
    "simple_mv_open_ring":   {"loader": "pn.simple_mv_open_ring_net()",           "display": "simple MV open ring"},
}

SIMBENCH_NETWORKS = {
    "sb_lv_rural":  {"loader": 'sb.get_simbench_net("1-LV-rural1--0-sw")',  "display": "SimBench LV rural"},
    "sb_lv_urban":  {"loader": 'sb.get_simbench_net("1-LV-urban6--0-sw")',  "display": "SimBench LV urban"},
    "sb_mv_rural":  {"loader": 'sb.get_simbench_net("1-MV-rural--0-sw")',   "display": "SimBench MV rural"},
    "sb_mv_urban":  {"loader": 'sb.get_simbench_net("1-MV-urban--0-sw")',   "display": "SimBench MV urban"},
    "sb_mv_comm":   {"loader": 'sb.get_simbench_net("1-MV-comm--0-sw")',    "display": "SimBench MV commercial"},
    "sb_hv_urban":  {"loader": 'sb.get_simbench_net("1-HV-urban--0-sw")',   "display": "SimBench HV urban"},
    "sb_hv_mixed":  {"loader": 'sb.get_simbench_net("1-HV-mixed--0-sw")',   "display": "SimBench HV mixed"},
    "sb_ehv":       {"loader": 'sb.get_simbench_net("1-EHV-mixed--0-sw")',  "display": "SimBench EHV mixed"},
}

FAST_NETWORKS = [
    "case4gs", "case5", "case6ww", "case9", "case14",
    "case24_ieee_rts", "case30", "case_ieee30", "case33bw",
    "case39", "case57", "case89pegase", "case118",
    "case145", "case200_illinois", "case300",
    "GBreducednetwork", "iceland",
    "simple_four_bus", "example_simple", "four_loads_branches",
    "synth_volt_ctrl_lv", "simple_mv_open_ring",
    "cigre_mv", "cigre_lv", "cigre_hv",
    "kerber_dorfnetz", "kerber_landnetz_f1", "kerber_landnetz_k1",
    "kerber_vorstadtnetz_k1",
]

LARGE_NETWORKS = [
    "case1354pegase", "case2869pegase", "case9241pegase",
    "case1888rte", "case2848rte", "case6470rte", "case3120sp",
    "GBnetwork", "oberrhein",
]


MODIFICATION_DEFS = {
    "set_load_p":             {"code": 'net.load.at[{idx}, "p_mw"] = {value}',                  "describe": "set the active power of load {idx} to {value} MW"},
    "set_load_q":             {"code": 'net.load.at[{idx}, "q_mvar"] = {value}',                "describe": "set the reactive power of load {idx} to {value} Mvar"},
    "set_load_scaling":       {"code": 'net.load.at[{idx}, "scaling"] = {value}',               "describe": "set the scaling of load {idx} to {value}"},
    "scale_all_loads":        {"code": 'net.load["scaling"] = {factor}',                        "describe": "scale all loads by a factor of {factor}"},
    "set_gen_p":              {"code": 'net.gen.at[{idx}, "p_mw"] = {value}',                   "describe": "set generator {idx} active power to {value} MW"},
    "set_gen_vm":             {"code": 'net.gen.at[{idx}, "vm_pu"] = {value}',                  "describe": "set generator {idx} voltage setpoint to {value} p.u."},
    "set_gen_out_of_service": {"code": 'net.gen.at[{idx}, "in_service"] = False',               "describe": "take generator {idx} out of service"},
    "set_gen_max_p":          {"code": 'net.gen.at[{idx}, "max_p_mw"] = {value}',               "describe": "set generator {idx} maximum active power to {value} MW"},
    "set_gen_min_p":          {"code": 'net.gen.at[{idx}, "min_p_mw"] = {value}',               "describe": "set generator {idx} minimum active power to {value} MW"},
    "set_ext_grid_vm":        {"code": 'net.ext_grid.at[{idx}, "vm_pu"] = {value}',             "describe": "set external grid {idx} voltage to {value} p.u."},
    "disconnect_line":        {"code": 'net.line.at[{idx}, "in_service"] = False',              "describe": "disconnect line {idx} (take it out of service)"},
    "disconnect_trafo":       {"code": 'net.trafo.at[{idx}, "in_service"] = False',             "describe": "disconnect transformer {idx}"},
    "open_switch":            {"code": 'net.switch.at[{idx}, "closed"] = False',                "describe": "open switch {idx}"},
    "close_switch":           {"code": 'net.switch.at[{idx}, "closed"] = True',                 "describe": "close switch {idx}"},
    "set_line_length":        {"code": 'net.line.at[{idx}, "length_km"] = {value}',             "describe": "change line {idx} length to {value} km"},
    "set_line_max_i":         {"code": 'net.line.at[{idx}, "max_i_ka"] = {value}',              "describe": "set line {idx} maximum current to {value} kA"},
    "set_line_r":             {"code": 'net.line.at[{idx}, "r_ohm_per_km"] = {value}',          "describe": "set line {idx} resistance to {value} Ohm/km"},
    "set_line_x":             {"code": 'net.line.at[{idx}, "x_ohm_per_km"] = {value}',          "describe": "set line {idx} reactance to {value} Ohm/km"},
    "set_line_parallel":      {"code": 'net.line.at[{idx}, "parallel"] = {value}',              "describe": "set line {idx} to {value} parallel circuits"},
    "set_trafo_tap_pos":      {"code": 'net.trafo.at[{idx}, "tap_pos"] = {value}',              "describe": "set transformer {idx} tap position to {value}"},
    "set_trafo_parallel":     {"code": 'net.trafo.at[{idx}, "parallel"] = {value}',             "describe": "set transformer {idx} to {value} parallel units"},
    "set_trafo_shift":        {"code": 'net.trafo.at[{idx}, "shift_degree"] = {value}',         "describe": "set transformer {idx} phase shift to {value} degrees"},
    "set_bus_max_vm":         {"code": 'net.bus.at[{idx}, "max_vm_pu"] = {value}',              "describe": "set bus {idx} maximum voltage limit to {value} p.u."},
    "set_bus_min_vm":         {"code": 'net.bus.at[{idx}, "min_vm_pu"] = {value}',              "describe": "set bus {idx} minimum voltage limit to {value} p.u."},
    "add_load":               {"code": "pp.create_load(net, bus={bus}, p_mw={p_mw}, q_mvar={q_mvar})", "describe": "add a new load at bus {bus} ({p_mw} MW, {q_mvar} Mvar)"},
    "add_sgen":               {"code": "pp.create_sgen(net, bus={bus}, p_mw={p_mw}, q_mvar={q_mvar})", "describe": "add a static generator at bus {bus} ({p_mw} MW)"},
    "add_shunt":              {"code": "pp.create_shunt(net, bus={bus}, q_mvar={q_mvar}, p_mw={p_mw})", "describe": "add a shunt at bus {bus} ({q_mvar} Mvar)"},
    "add_storage":            {"code": "pp.create_storage(net, bus={bus}, p_mw={p_mw}, max_e_mwh={max_e_mwh})", "describe": "add a storage at bus {bus} ({p_mw} MW, {max_e_mwh} MWh)"},
    "add_gen":                {"code": "pp.create_gen(net, bus={bus}, p_mw={p_mw}, vm_pu={vm_pu})", "describe": "add a generator at bus {bus} ({p_mw} MW, vm={vm_pu} p.u.)"},
}

MODIFICATION_PARAM_DEFAULTS = {
    "q_mvar": 0, "p_mw": 0, "max_e_mwh": 1.0, "vm_pu": 1.0, "value": 0,
}


_SC_SETUP = [
    'if "s_sc_max_mva" not in net.ext_grid.columns or net.ext_grid["s_sc_max_mva"].isna().any():',
    '    net.ext_grid["s_sc_max_mva"] = 1000',
    '    net.ext_grid["s_sc_min_mva"] = 800',
    '    net.ext_grid["rx_max"] = 0.1',
    '    net.ext_grid["rx_min"] = 0.1',
    'if len(net.gen) > 0 and "vn_kv" not in net.gen.columns:',
    '    net.gen["in_service"] = False',
    'if len(net.sgen) > 0:',
    '    net.sgen["in_service"] = False',
]

TASK_CONFIG = {
    "power_flow":        {"run_code": "pp.runpp(net)",     "imports": [], "setup_code": [], "filter": lambda m: m["n_bus"] >= 3},
    "dc_power_flow":     {"run_code": "pp.rundcpp(net)",   "imports": [], "setup_code": [], "filter": lambda m: m["n_bus"] >= 3},
    "opf":               {"run_code": "pp.runopp(net)",    "imports": [], "setup_code": [], "filter": lambda m: m["has_cost"]},
    "short_circuit_3ph": {"run_code": 'pp.shortcircuit.calc_sc(net, fault="3ph", case="max")', "imports": ["import pandapower.shortcircuit"], "setup_code": _SC_SETUP, "filter": lambda m: m["n_bus"] >= 3},
    "short_circuit_2ph": {"run_code": 'pp.shortcircuit.calc_sc(net, fault="2ph", case="max")', "imports": ["import pandapower.shortcircuit"], "setup_code": _SC_SETUP, "filter": lambda m: m["n_bus"] >= 3},
    "contingency":       {"run_code": 'nminus1_cases = {"line": {"index": net.line.index.tolist()}}\nrun_contingency(net, nminus1_cases)', "imports": ["from pandapower.contingency import run_contingency"], "setup_code": [], "filter": lambda m: m["n_line"] >= 3 and m["n_bus"] <= 300},
    "time_series":       {"run_code": None, "imports": ["import numpy as np", "import pandas as pd", "from pandapower.timeseries import DFData, OutputWriter, run_timeseries", "from pandapower.control import ConstControl"], "setup_code": [], "filter": lambda m: m["n_load"] >= 1 and m["n_bus"] <= 300, "needs_setup_fn": True},
    "state_estimation":  {"run_code": None, "imports": ["import numpy as np", "from pandapower.estimation import estimate"], "setup_code": [], "filter": lambda m: m["n_bus"] >= 3 and m["n_bus"] <= 200, "needs_setup_fn": True},
}

EXTRACT_CODE_TEMPLATES = {
    "point_value":      'result = net.{table}.at[{filter_idx}, "{column}"]',
    "max_value":        'result = net.{table}["{column}"].max()',
    "min_value":        'result = net.{table}["{column}"].min()',
    "argmax":           'result = net.{table}["{column}"].idxmax()',
    "argmin":           'result = net.{table}["{column}"].idxmin()',
    "total":            'result = net.{table}["{column}"].sum()',
    "count_violations": 'result = (net.{table}["{column}"] {operator} {threshold}).sum()',
    "bool_check":       'result = (net.{table}["{column}"] {operator} {threshold}).any()',
    "full_table":       'result = net.{table}["{column}"]',
    "ts_min_value":     'result = ow_results["{table}.{column}"].min().min()',
    "ts_max_value":     'result = ow_results["{table}.{column}"].max().max()',
    "ts_point_at_step": 'result = ow_results["{table}.{column}"].iloc[{time_step}, {filter_idx}]',
}

TIME_SERIES_CONFIG = {"n_steps": 24, "load_scaling_range": (0.6, 1.4), "noise_std": 0.02}
STATE_ESTIMATION_CONFIG = {"voltage_std_dev": 0.01, "power_bus_std_dev": 1.0, "power_line_std_dev": 0.5, "noise_scale_v": 0.003, "noise_scale_p": 0.5, "noise_scale_q": 0.3}


NL_TEMPLATES = {
    ("power_flow", "point_value", "res_bus", "vm_pu"): [
        "Load the {network_display} in pandapower. {mod_description}Run a power flow analysis and report the voltage magnitude (in p.u.) at bus {filter_idx}.",
        "Using the {network_display}, {mod_description_lower}perform a load flow calculation. What is the voltage at bus {filter_idx} in per-unit?",
        "On the {network_display}: {mod_description}After running power flow, what is the p.u. voltage magnitude at bus {filter_idx}?",
        "{mod_description}Set up the {network_display} and run power flow. Report the voltage magnitude at bus {filter_idx} (p.u.).",
    ],
    ("power_flow", "point_value", "res_bus", "va_degree"): [
        "On the {network_display}, {mod_description_lower}run power flow. What is the voltage angle (degrees) at bus {filter_idx}?",
        "{mod_description}Using the {network_display}, perform power flow and report the voltage angle at bus {filter_idx} in degrees.",
    ],
    ("power_flow", "point_value", "res_line", "loading_percent"): [
        "On the {network_display}, {mod_description_lower}run power flow and report the loading percentage of line {filter_idx}.",
        "Using the {network_display}: {mod_description}Calculate power flow. What is line {filter_idx}'s loading in percent?",
    ],
    ("power_flow", "point_value", "res_trafo", "loading_percent"): [
        "On the {network_display}, {mod_description_lower}run power flow. What is the loading percentage of transformer {filter_idx}?",
        "{mod_description}Using the {network_display}, run power flow and report transformer {filter_idx}'s loading (%).",
    ],
    ("power_flow", "max_value", "res_line", "loading_percent"): [
        "Load the {network_display}. {mod_description}Run a power flow and find the maximum line loading percentage across all lines.",
        "{mod_description}On the {network_display}, after running load flow, what is the highest line loading (%) in the network?",
        "Using the {network_display}, {mod_description_lower}perform power flow analysis. Report the peak line loading percentage.",
    ],
    ("power_flow", "max_value", "res_trafo", "loading_percent"): [
        "On the {network_display}, {mod_description_lower}run power flow. What is the maximum transformer loading (%) in the network?",
    ],
    ("power_flow", "min_value", "res_bus", "vm_pu"): [
        "On the {network_display}, {mod_description_lower}run power flow and find the minimum bus voltage magnitude (p.u.) in the network.",
        "{mod_description}Load the {network_display} and run a load flow. What is the lowest voltage (p.u.) across all buses?",
    ],
    ("power_flow", "argmax", "res_line", "loading_percent"): [
        "On the {network_display}, {mod_description_lower}run power flow. Which line has the highest loading percentage? Report its index.",
        "{mod_description}Using the {network_display}, perform load flow analysis. Identify the most heavily loaded line (return its index).",
    ],
    ("power_flow", "count_violations", "res_bus", "vm_pu"): [
        "On the {network_display}, {mod_description_lower}run power flow. How many buses have voltage below {threshold} p.u.?",
        "{mod_description}Load the {network_display}. After power flow, count the number of buses where voltage magnitude is less than {threshold} p.u.",
    ],
    ("power_flow", "bool_check", "res_line", "loading_percent"): [
        "On the {network_display}, {mod_description_lower}run power flow. Are there any lines with loading above {threshold}%?",
        "{mod_description}Using the {network_display}, run a load flow and check: does any line exceed {threshold}% loading?",
    ],
    ("power_flow", "total", "res_line", "pl_mw"): [
        "On the {network_display}, {mod_description_lower}run power flow. What are the total active power losses (MW) across all lines?",
        "{mod_description}Load the {network_display} and run power flow. Report the sum of active power losses on all lines in MW.",
    ],
    ("power_flow", "point_value", "res_ext_grid", "p_mw"): [
        "On the {network_display}, {mod_description_lower}run power flow. What is the active power supplied by the external grid (MW)?",
        "{mod_description}Using the {network_display}, run power flow and report the ext_grid active power injection (MW).",
    ],
    ("power_flow", "point_value", "res_ext_grid", "q_mvar"): [
        "On the {network_display}, {mod_description_lower}run power flow. What is the reactive power from the external grid (Mvar)?",
    ],
    ("dc_power_flow", "point_value", "res_bus", "va_degree"): [
        "Using the {network_display}, {mod_description_lower}run a DC power flow. What is the voltage angle (degrees) at bus {filter_idx}?",
    ],
    ("dc_power_flow", "point_value", "res_bus", "vm_pu"): [
        "On the {network_display}, {mod_description_lower}run a DC power flow and report the voltage magnitude (p.u.) at bus {filter_idx}.",
    ],
    ("dc_power_flow", "max_value", "res_line", "loading_percent"): [
        "On the {network_display}, {mod_description_lower}run DC power flow and report the maximum line loading (%).",
    ],
    ("dc_power_flow", "min_value", "res_bus", "vm_pu"): [
        "On the {network_display}, {mod_description_lower}run DC power flow and report the minimum bus voltage (p.u.) across all buses.",
    ],
    ("dc_power_flow", "argmax", "res_line", "loading_percent"): [
        "On the {network_display}, {mod_description_lower}run DC power flow. Which line has the highest loading percentage? Report its index.",
    ],
    ("dc_power_flow", "count_violations", "res_bus", "vm_pu"): [
        "On the {network_display}, {mod_description_lower}run DC power flow. How many buses have voltage magnitude below {threshold} p.u.?",
    ],
    ("dc_power_flow", "bool_check", "res_line", "loading_percent"): [
        "On the {network_display}, {mod_description_lower}run DC power flow. Does any line loading exceed {threshold}%?",
    ],
    ("dc_power_flow", "point_value", "res_line", "loading_percent"): [
        "On the {network_display}, {mod_description_lower}run DC power flow. What is the loading of line {filter_idx} in percent?",
    ],
    ("dc_power_flow", "point_value", "res_trafo", "loading_percent"): [
        "On the {network_display}, {mod_description_lower}run DC power flow. What is the loading of transformer {filter_idx} in percent?",
    ],
    ("dc_power_flow", "max_value", "res_trafo", "loading_percent"): [
        "On the {network_display}, {mod_description_lower}run DC power flow. What is the maximum transformer loading (%)?",
    ],
    ("dc_power_flow", "total", "res_line", "pl_mw"): [
        "On the {network_display}, {mod_description_lower}run DC power flow. What are the total line losses (MW)?",
    ],
    ("dc_power_flow", "point_value", "res_ext_grid", "p_mw"): [
        "On the {network_display}, {mod_description_lower}run DC power flow. What is the active power exchange at the external grid (MW)?",
    ],
    ("short_circuit_3ph", "point_value", "res_bus_sc", "ikss_ka"): [
        "On the {network_display}, {mod_description_lower}calculate the 3-phase short circuit current at bus {filter_idx}. Report ikss in kA.",
        "Using the {network_display}, {mod_description_lower}perform a 3-phase short circuit analysis. What is the initial symmetrical short-circuit current (ikss) at bus {filter_idx} in kA?",
        "{mod_description}Run a three-phase short circuit calculation on the {network_display}. Report ikss_ka at bus {filter_idx}.",
    ],
    ("short_circuit_3ph", "max_value", "res_bus_sc", "ikss_ka"): [
        "On the {network_display}, {mod_description_lower}run 3-phase short circuit analysis. What is the maximum ikss (kA) across all buses?",
    ],
    ("short_circuit_3ph", "argmax", "res_bus_sc", "ikss_ka"): [
        "On the {network_display}, {mod_description_lower}run 3-phase short circuit analysis. Which bus has the highest short circuit current? Report its index.",
    ],
    ("short_circuit_3ph", "min_value", "res_bus_sc", "ikss_ka"): [
        "On the {network_display}, {mod_description_lower}run 3-phase short circuit analysis. What is the minimum ikss (kA) across all buses?",
    ],
    ("short_circuit_2ph", "point_value", "res_bus_sc", "ikss_ka"): [
        "On the {network_display}, {mod_description_lower}calculate the 2-phase short circuit current at bus {filter_idx}. Report ikss in kA.",
        "{mod_description}Run a two-phase short circuit calculation on the {network_display}. What is ikss_ka at bus {filter_idx}?",
    ],
    ("short_circuit_2ph", "max_value", "res_bus_sc", "ikss_ka"): [
        "On the {network_display}, {mod_description_lower}run 2-phase short circuit analysis. What is the maximum ikss (kA)?",
    ],
    ("short_circuit_2ph", "min_value", "res_bus_sc", "ikss_ka"): [
        "On the {network_display}, {mod_description_lower}run 2-phase short circuit analysis. What is the minimum ikss (kA) across all buses?",
    ],
    ("short_circuit_2ph", "argmax", "res_bus_sc", "ikss_ka"): [
        "On the {network_display}, {mod_description_lower}run 2-phase short circuit analysis. Which bus has the highest short circuit current? Report its index.",
    ],
    ("opf", "point_value", "res_gen", "p_mw"): [
        "On the {network_display}, {mod_description_lower}run an optimal power flow. What is the optimal active power dispatch (MW) for generator {filter_idx}?",
        "Using the {network_display}, {mod_description_lower}perform OPF. Report the active power output of generator {filter_idx} in MW.",
    ],
    ("opf", "point_value", "res_gen", "q_mvar"): [
        "On the {network_display}, {mod_description_lower}run OPF. What is the reactive power output (Mvar) of generator {filter_idx}?",
    ],
    ("opf", "total", "res_gen", "p_mw"): [
        "On the {network_display}, {mod_description_lower}run OPF. What is the total generation (MW) from all generators?",
    ],
    ("opf", "min_value", "res_bus", "vm_pu"): [
        "On the {network_display}, {mod_description_lower}run optimal power flow. What is the minimum bus voltage (p.u.) in the optimal solution?",
    ],
    ("opf", "max_value", "res_bus", "vm_pu"): [
        "On the {network_display}, {mod_description_lower}run OPF. What is the maximum bus voltage (p.u.) in the optimal dispatch?",
    ],
    ("contingency", "max_value", "res_line", "max_loading_percent"): [
        "On the {network_display}, {mod_description_lower}run N-1 contingency analysis for all lines. What is the worst-case maximum line loading (%) across all contingencies?",
    ],
    ("contingency", "bool_check", "res_line", "max_loading_percent"): [
        "On the {network_display}, {mod_description_lower}run N-1 contingency analysis. Does any line exceed {threshold}% loading under any single line outage?",
    ],
    ("contingency", "min_value", "res_bus", "min_vm_pu"): [
        "On the {network_display}, {mod_description_lower}run N-1 contingency analysis. What is the lowest bus voltage (p.u.) observed across all contingency cases?",
    ],
    ("time_series", "ts_min_value", "res_bus", "vm_pu"): [
        "On the {network_display}, {mod_description_lower}run a {n_steps}-step time series simulation with random load scaling between {load_lo}x and {load_hi}x (seed={se_seed}). What is the minimum bus voltage (p.u.) observed across all buses and all time steps?",
    ],
    ("time_series", "ts_max_value", "res_line", "loading_percent"): [
        "On the {network_display}, {mod_description_lower}run a {n_steps}-step time series simulation with random load profiles (scaling {load_lo}-{load_hi}, seed={se_seed}). What is the peak line loading (%) across all lines and time steps?",
    ],
    ("time_series", "ts_point_at_step", "res_bus", "vm_pu"): [
        "On the {network_display}, {mod_description_lower}run a {n_steps}-step time series simulation with random load profiles (scaling {load_lo}-{load_hi}, seed={se_seed}). What is the voltage (p.u.) at bus {filter_idx} at time step {time_step}?",
    ],
    ("state_estimation", "point_value", "res_bus_est", "vm_pu"): [
        "On the {network_display}, {mod_description_lower}first run a power flow to obtain true bus values. Then place noisy measurements (voltage at all buses with std_dev={v_std}, bus power injections with std_dev={p_std} MW, seed={se_seed}) and run WLS state estimation. Report the estimated voltage magnitude (p.u.) at bus {filter_idx}.",
    ],
    ("state_estimation", "max_value", "res_bus_est", "vm_pu"): [
        "On the {network_display}, {mod_description_lower}set up a state estimation with noisy measurements from a power flow solution (voltage std_dev={v_std}, power std_dev={p_std}, seed={se_seed}). What is the maximum estimated bus voltage (p.u.)?",
    ],
    ("state_estimation", "min_value", "res_bus_est", "vm_pu"): [
        "On the {network_display}, {mod_description_lower}run state estimation with noisy measurements derived from power flow results (seed={se_seed}). What is the minimum estimated bus voltage (p.u.)?",
    ],
}


LLM_EXPANSION_CONFIG = {
    "model": "gpt-4o-mini",
    "system_prompt": """You are helping generate diverse natural language prompts for a power systems analysis benchmark.

Given an example prompt template (with {placeholders}), generate N new paraphrases that:
1. Keep ALL {placeholders} exactly as-is (same names, same braces)
2. Vary the sentence structure, vocabulary, and style
3. Include a mix of: imperative ("Run..."), question ("What is..."), conversational ("Can you...")
4. Some should be more technical, some more casual
5. Each must be a single string, no line breaks inside
6. Keep all task-specific required placeholders listed by the user exactly as written

Return ONLY a JSON array of strings. No explanation, no markdown.""",

    "user_template": """Here are existing templates for the key {key}:

{existing}

Generate {n} NEW diverse paraphrases. Keep all {{placeholders}} exactly as they appear.
Every template MUST include these placeholders exactly as written: {required_placeholders}.
If the requirement says "exactly one of {{a}} or {{b}}", include one and only one of those two placeholders.
Return a JSON array of strings only.""",
}

BENCHMARK_DIR = Path(__file__).resolve().parent
EXPANDED_TEMPLATES_PATH = str(BENCHMARK_DIR / "expanded_nl_templates.json")


MULTI_TASK_NL_TEMPLATES = {
    "comparison": [
        "On the {network_display}, {mod_description_lower}run power flow and record {baseline_metric}. Then {modification_desc}, re-run power flow, and report the change in {baseline_metric}.",
        "Using the {network_display}, {mod_description_lower}first run a power flow to get {baseline_metric}. After that, {modification_desc} and run power flow again. What is the difference in {baseline_metric} between the two runs?",
        "{mod_description}Load the {network_display}. Run power flow. Then {modification_desc} and re-run. Report how much {baseline_metric} changed.",
        "On the {network_display}, first apply this operating-rule change: {mod_description_lower}run power flow and record {baseline_metric}. Next, {modification_desc}, run power flow again, and report the change in {baseline_metric}.",
    ],
    "comparison_opf": [
        "On the {network_display}, {mod_description_lower}run OPF and record the total generation cost. Then {modification_desc}, re-run OPF, and report how much the cost changed.",
        "On the {network_display}, first apply this operating-rule change: {mod_description_lower}run OPF and record the total generation cost. Next, {modification_desc}, re-run OPF, and report the cost difference.",
        "Using the {network_display}, start from this semantic operating condition: {mod_description_lower}solve an OPF and note the total cost. Then {modification_desc}, solve OPF again, and report how much the cost changes.",
    ],
    "sequential": [
        "On the {network_display}, {mod_description_lower}run power flow. If {condition_metric} {condition_op_word} {condition_threshold}, {action_desc} and re-run power flow. Report {final_query_desc}.",
        "Using the {network_display}, {mod_description_lower}perform a power flow analysis. Check whether {condition_metric} {condition_op_word} {condition_threshold}. If so, {action_desc}, then re-run power flow and report {final_query_desc}.",
        "On the {network_display}, begin from this semantic setup: {mod_description_lower}run power flow. If {condition_metric} {condition_op_word} {condition_threshold}, then {action_desc}, re-run the analysis, and report {final_query_desc}.",
    ],
    "pf_then_sc": [
        "On the {network_display}, {mod_description_lower}first run a power flow to establish the operating point. Then run a 3-phase short circuit calculation using the superposition method (use_pre_fault_voltage=True) and report the short circuit current (ikss in kA) at bus {target_bus}.",
        "Using the {network_display}, start from this semantic operating setup: {mod_description_lower}run a power flow to establish the pre-fault operating point. Then perform a 3-phase short circuit calculation with superposition and report ikss at bus {target_bus}.",
        "On the {network_display}, first enforce this rule-based operating condition: {mod_description_lower}determine the operating point with power flow. After that, run a 3-phase short circuit study with use_pre_fault_voltage=True and report the short-circuit current at bus {target_bus}.",
    ],
    "contingency_fix": [
        "On the {network_display}, {mod_description_lower}run N-1 contingency analysis for all lines. Identify the contingency causing the highest line loading. To mitigate, add a {fix_gen_mw} MW generator at bus {fix_bus}. Re-run power flow with that worst-case line disconnected and the new generator. Report the new maximum line loading (%).",
        "Using the {network_display}, begin from this semantic operating setup: {mod_description_lower}run an N-1 contingency analysis for all lines. Find the contingency with the highest loading, add a {fix_gen_mw} MW generator at bus {fix_bus} as the corrective measure, then re-run power flow with that worst-case line disconnected and report the new maximum line loading (%).",
    ],
    "parallel": [
        "On the {network_display}, {mod_description_lower}run {task_a_name} and record {metric_a_desc}. Then run {task_b_name} and record {metric_b_desc}. Report {metric_b_desc}.",
        "On the {network_display}, first apply this semantic setup: {mod_description_lower}then run {task_a_name} and record {metric_a_desc}. Afterward, run {task_b_name}, record {metric_b_desc}, and report {metric_b_desc}.",
    ],
    "diagnose_and_fix": [
        "On the {network_display}, {mod_description_lower}run power flow. If any line exceeds {overload_threshold}% loading, disconnect the most loaded line. Run power flow again. If any bus voltage is below {voltage_threshold} p.u., add a {shunt_mvar} Mvar shunt at the lowest-voltage bus. Run power flow a final time and report the minimum bus voltage (p.u.).",
        "Using the {network_display}, start from this semantic operating condition: {mod_description_lower}run power flow. If any line exceeds {overload_threshold}% loading, disconnect the worst-loaded line and re-run. If any bus voltage is then below {voltage_threshold} p.u., add a {shunt_mvar} Mvar shunt at the weakest bus, run power flow again, and report the final minimum bus voltage (p.u.).",
    ],
}


SAMPLING_CONFIG = {
    "n_mods_weights": [0.2, 0.4, 0.3, 0.1],
    "load_scaling_range": (0.5, 2.0),
    "load_p_range": (5.0, 100.0),
    "load_q_range": (1.0, 30.0),
    "gen_vm_range": (0.95, 1.08),
    "gen_p_range": (10.0, 200.0),
    "ext_grid_vm_range": (0.97, 1.06),
    "new_load_p_range": (1.0, 20.0),
    "new_load_q_range": (0.0, 5.0),
    "new_sgen_p_range": (1.0, 15.0),
    "new_gen_p_range": (5.0, 50.0),
    "new_gen_vm_range": (0.98, 1.05),
    "line_r_range": (0.01, 0.5),
    "line_x_range": (0.05, 0.8),
    "trafo_shift_range": (-30.0, 30.0),
    "bus_vm_max_range": (1.05, 1.10),
    "bus_vm_min_range": (0.90, 0.95),
    "voltage_violation_thresholds": [0.90, 0.95, 0.97],
    "loading_violation_thresholds": [80, 100, 120],
}
