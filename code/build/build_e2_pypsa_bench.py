#!/usr/bin/env python
# --------------------------------------------------------------------------
# Repository copy of scripts/build/build_e2_pypsa_bench.py, with imports and
# data-path constants adapted to this repository's layout. Requires the pinned
# environment plus pypsa; archived to document the frozen E2 mini-bench
# construction. See code/README.md for the module map.
# --------------------------------------------------------------------------
"""Build the E2 PyPSA mini-bench (benchmark/e2_pypsa/e2_pypsa_bench.json).

Protocol authority: E2_backend_transfer.md
  - 5 families x {D1_basic, D3_semantic} x ~9 = ~90 items.
  - Families: power_flow / linear_power_flow / opf / network_modification / result_extraction.
  - Network base: self-built small PyPSA networks (3-10 buses, diverse topologies /
    mixed components) + a few offline PyPSA example networks (ac-dc-meshed).
  - GT protocol (E2 §2.3): every reference_code is a self-contained script; GT is produced
    by running it in an isolated subprocess (NOT hardcoded, anti-contamination), passed
    through an admission filter (n_bus>=3, pf converged / opf optimal & finite), and rounded
    to 6 decimals.
  - Query discipline (absorbing E1 pilot lessons): the queried scalar must be sensitive to
    the item's modification (baseline vs modified scalar differs > tolerance); deterministic
    parameters only (no RNG); no bool queries.
  - D3 construction: the semantic target is resolved against the built network state FIRST,
    then hardcoded into the reference (same construction method as the main benchmark); the
    resolved target index is recorded in scenario.resolved_target for E1-style audit.

Schema aligned to the main benchmark item (benchmark.json):
  id / scenario{task,network,modifications,query_target,resolved_target?} /
  difficulty_level / natural_language_query / reference_code /
  ground_truth / ground_truth_type.

Resource discipline: login-node, serial single-thread (OMP/OPENBLAS/MKL=1); per-item GT run
in an isolated subprocess; incremental checkpoint every 15 items. No background processes.

Reproducible / re-runnable: fully deterministic (no RNG); SEED recorded for provenance.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# --- resource discipline: single-thread BLAS for any in-process resolution -------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import logging  # noqa: E402

logging.disable(logging.WARNING)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

pd.set_option("future.infer_string", False)  # PyPSA/xarray + arrow-string incompat guard
import pypsa  # noqa: E402

SEED = 20260724  # frozen provenance marker (construction is deterministic; no RNG used)
REPO = Path(__file__).resolve().parents[2]  # repository root
OUT_DIR = REPO / "benchmark" / "e2_pypsa"
OUT_JSON = OUT_DIR / "e2_pypsa_bench.json"
SCRATCH = Path(
    os.environ.get("POWERCODEBENCH_SCRATCH", tempfile.gettempdir())
).expanduser()
SCRATCH.mkdir(parents=True, exist_ok=True)
SUBPROC_TIMEOUT = 120  # s per isolated GT run
SENS_ABS_TOL = 1e-4    # queried scalar must move at least this much under the modification
SENS_REL_TOL = 1e-4

FAMILIES = [
    "power_flow",
    "linear_power_flow",
    "opf",
    "network_modification",
    "result_extraction",
]
DIFFS = ["D1_basic", "D3_semantic"]
N_PER_CELL = 9  # target items per (family, difficulty)

# ============================================================================================
# Network builders -> (recipe_lines, description). Deterministic; no RNG.
# recipe_lines fully construct `n` (self-built start with n=pypsa.Network(); example nets
# reassign n). The same lines are exec'd live (resolution/sensitivity) and embedded verbatim
# into reference_code, so the audited network and the executed network are one source.
# ============================================================================================

def b_radial3():
    return [
        "n = pypsa.Network()",
        "n.add('Bus', 'b1', v_nom=20.0)",
        "n.add('Bus', 'b2', v_nom=20.0)",
        "n.add('Bus', 'b3', v_nom=20.0)",
        "n.add('Generator', 'slack', bus='b1', control='Slack', p_nom=200.0)",
        "n.add('Load', 'ld2', bus='b2', p_set=40.0, q_set=12.0)",
        "n.add('Load', 'ld3', bus='b3', p_set=70.0, q_set=20.0)",
        "n.add('Line', 'l12', bus0='b1', bus1='b2', r=0.03, x=0.09, s_nom=150.0)",
        "n.add('Line', 'l23', bus0='b2', bus1='b3', r=0.04, x=0.12, s_nom=120.0)",
    ], "self-built 3-bus radial AC network (slack + 2 loads, 2 lines)"


def b_radial4():
    return [
        "n = pypsa.Network()",
        "n.add('Bus', 'b1', v_nom=20.0)",
        "n.add('Bus', 'b2', v_nom=20.0)",
        "n.add('Bus', 'b3', v_nom=20.0)",
        "n.add('Bus', 'b4', v_nom=20.0)",
        "n.add('Generator', 'slack', bus='b1', control='Slack', p_nom=250.0)",
        "n.add('Load', 'ld2', bus='b2', p_set=30.0, q_set=9.0)",
        "n.add('Load', 'ld3', bus='b3', p_set=45.0, q_set=14.0)",
        "n.add('Load', 'ld4', bus='b4', p_set=55.0, q_set=16.0)",
        "n.add('Line', 'l12', bus0='b1', bus1='b2', r=0.03, x=0.08, s_nom=150.0)",
        "n.add('Line', 'l23', bus0='b2', bus1='b3', r=0.04, x=0.10, s_nom=120.0)",
        "n.add('Line', 'l34', bus0='b3', bus1='b4', r=0.05, x=0.12, s_nom=90.0)",
    ], "self-built 4-bus radial AC network (20 kV, slack + 3 loads, 3 series lines)"


def b_radial5():
    return [
        "n = pypsa.Network()",
        "n.add('Bus', 'b1', v_nom=20.0)",
        "n.add('Bus', 'b2', v_nom=20.0)",
        "n.add('Bus', 'b3', v_nom=20.0)",
        "n.add('Bus', 'b4', v_nom=20.0)",
        "n.add('Bus', 'b5', v_nom=20.0)",
        "n.add('Generator', 'slack', bus='b1', control='Slack', p_nom=300.0)",
        "n.add('Load', 'ld2', bus='b2', p_set=25.0, q_set=8.0)",
        "n.add('Load', 'ld3', bus='b3', p_set=30.0, q_set=9.0)",
        "n.add('Load', 'ld4', bus='b4', p_set=35.0, q_set=11.0)",
        "n.add('Load', 'ld5', bus='b5', p_set=40.0, q_set=12.0)",
        "n.add('Line', 'l12', bus0='b1', bus1='b2', r=0.025, x=0.07, s_nom=170.0)",
        "n.add('Line', 'l23', bus0='b2', bus1='b3', r=0.03, x=0.08, s_nom=140.0)",
        "n.add('Line', 'l34', bus0='b3', bus1='b4', r=0.035, x=0.09, s_nom=110.0)",
        "n.add('Line', 'l45', bus0='b4', bus1='b5', r=0.04, x=0.10, s_nom=90.0)",
    ], "self-built 5-bus radial AC network (20 kV, slack + 4 loads, 4 series lines)"


def b_mesh4():
    return [
        "n = pypsa.Network()",
        "n.add('Bus', 'b1', v_nom=110.0)",
        "n.add('Bus', 'b2', v_nom=110.0)",
        "n.add('Bus', 'b3', v_nom=110.0)",
        "n.add('Bus', 'b4', v_nom=110.0)",
        "n.add('Generator', 'slack', bus='b1', control='Slack', p_nom=400.0)",
        "n.add('Generator', 'g2', bus='b2', control='PV', p_nom=150.0, p_set=50.0)",
        "n.add('Load', 'ld3', bus='b3', p_set=80.0, q_set=25.0)",
        "n.add('Load', 'ld4', bus='b4', p_set=60.0, q_set=18.0)",
        "n.add('Line', 'l12', bus0='b1', bus1='b2', r=0.02, x=0.06, s_nom=200.0)",
        "n.add('Line', 'l23', bus0='b2', bus1='b3', r=0.02, x=0.06, s_nom=200.0)",
        "n.add('Line', 'l34', bus0='b3', bus1='b4', r=0.03, x=0.08, s_nom=180.0)",
        "n.add('Line', 'l41', bus0='b4', bus1='b1', r=0.02, x=0.06, s_nom=200.0)",
        "n.add('Line', 'l13', bus0='b1', bus1='b3', r=0.025, x=0.07, s_nom=180.0)",
    ], "self-built 4-bus meshed AC network (slack + PV gen + 2 loads, 5 lines incl. diagonal)"


def b_ring5():
    return [
        "n = pypsa.Network()",
        "n.add('Bus', 'b1', v_nom=110.0)",
        "n.add('Bus', 'b2', v_nom=110.0)",
        "n.add('Bus', 'b3', v_nom=110.0)",
        "n.add('Bus', 'b4', v_nom=110.0)",
        "n.add('Bus', 'b5', v_nom=110.0)",
        "n.add('Generator', 'slack', bus='b1', control='Slack', p_nom=400.0)",
        "n.add('Generator', 'g3', bus='b3', control='PV', p_nom=120.0, p_set=40.0)",
        "n.add('Load', 'ld2', bus='b2', p_set=55.0, q_set=15.0)",
        "n.add('Load', 'ld3', bus='b3', p_set=65.0, q_set=20.0)",
        "n.add('Load', 'ld4', bus='b4', p_set=70.0, q_set=22.0)",
        "n.add('Load', 'ld5', bus='b5', p_set=45.0, q_set=12.0)",
        "n.add('Line', 'l12', bus0='b1', bus1='b2', r=0.02, x=0.06, s_nom=200.0)",
        "n.add('Line', 'l23', bus0='b2', bus1='b3', r=0.02, x=0.06, s_nom=200.0)",
        "n.add('Line', 'l34', bus0='b3', bus1='b4', r=0.03, x=0.08, s_nom=180.0)",
        "n.add('Line', 'l45', bus0='b4', bus1='b5', r=0.03, x=0.08, s_nom=180.0)",
        "n.add('Line', 'l51', bus0='b5', bus1='b1', r=0.02, x=0.06, s_nom=200.0)",
    ], "self-built 5-bus ring AC network (slack + PV gen + 4 loads, 5 ring lines)"


def b_trafo6():
    return [
        "n = pypsa.Network()",
        "n.add('Bus', 'hv1', v_nom=110.0)",
        "n.add('Bus', 'hv2', v_nom=110.0)",
        "n.add('Bus', 'hv3', v_nom=110.0)",
        "n.add('Bus', 'lv1', v_nom=20.0)",
        "n.add('Bus', 'lv2', v_nom=20.0)",
        "n.add('Bus', 'lv3', v_nom=20.0)",
        "n.add('Generator', 'slack', bus='hv1', control='Slack', p_nom=300.0)",
        "n.add('Load', 'ld_hv2', bus='hv2', p_set=40.0, q_set=12.0)",
        "n.add('Load', 'ld_lv2', bus='lv2', p_set=30.0, q_set=9.0)",
        "n.add('Load', 'ld_lv3', bus='lv3', p_set=25.0, q_set=7.0)",
        "n.add('Line', 'lhv12', bus0='hv1', bus1='hv2', r=0.02, x=0.08, s_nom=200.0)",
        "n.add('Line', 'lhv23', bus0='hv2', bus1='hv3', r=0.02, x=0.08, s_nom=200.0)",
        "n.add('Line', 'lhv13', bus0='hv1', bus1='hv3', r=0.025, x=0.09, s_nom=180.0)",
        "n.add('Transformer', 'T1', bus0='hv3', bus1='lv1', s_nom=100.0, r=0.01, x=0.06)",
        "n.add('Line', 'llv12', bus0='lv1', bus1='lv2', r=0.05, x=0.10, s_nom=80.0)",
        "n.add('Line', 'llv23', bus0='lv2', bus1='lv3', r=0.06, x=0.12, s_nom=70.0)",
    ], "self-built 6-bus two-voltage-level AC network (110/20 kV, 1 transformer, 3 loads)"


def b_opf_econ5():
    return [
        "n = pypsa.Network()",
        "n.add('Bus', 'b1', v_nom=110.0)",
        "n.add('Bus', 'b2', v_nom=110.0)",
        "n.add('Bus', 'b3', v_nom=110.0)",
        "n.add('Bus', 'b4', v_nom=110.0)",
        "n.add('Bus', 'b5', v_nom=110.0)",
        "n.add('Generator', 'cheap', bus='b1', p_nom=120.0, marginal_cost=10.0)",
        "n.add('Generator', 'mid', bus='b3', p_nom=90.0, marginal_cost=35.0)",
        "n.add('Generator', 'peak', bus='b5', p_nom=120.0, marginal_cost=70.0)",
        "n.add('Load', 'ld2', bus='b2', p_set=90.0)",
        "n.add('Load', 'ld4', bus='b4', p_set=110.0)",
        "n.add('Line', 'l12', bus0='b1', bus1='b2', r=0.01, x=0.06, s_nom=150.0)",
        "n.add('Line', 'l23', bus0='b2', bus1='b3', r=0.01, x=0.06, s_nom=150.0)",
        "n.add('Line', 'l34', bus0='b3', bus1='b4', r=0.01, x=0.06, s_nom=150.0)",
        "n.add('Line', 'l45', bus0='b4', bus1='b5', r=0.01, x=0.06, s_nom=150.0)",
        "n.add('Line', 'l51', bus0='b5', bus1='b1', r=0.01, x=0.06, s_nom=150.0)",
        "n.add('Line', 'l13', bus0='b1', bus1='b3', r=0.012, x=0.07, s_nom=120.0)",
    ], "self-built 5-bus economic-dispatch OPF network (3 dispatchable gens, 2 loads, meshed)"


def b_opf_mesh4():
    return [
        "n = pypsa.Network()",
        "n.add('Bus', 'b1', v_nom=110.0)",
        "n.add('Bus', 'b2', v_nom=110.0)",
        "n.add('Bus', 'b3', v_nom=110.0)",
        "n.add('Bus', 'b4', v_nom=110.0)",
        "n.add('Generator', 'cheap', bus='b1', p_nom=100.0, marginal_cost=12.0)",
        "n.add('Generator', 'exp', bus='b4', p_nom=100.0, marginal_cost=55.0)",
        "n.add('Load', 'ld2', bus='b2', p_set=60.0)",
        "n.add('Load', 'ld3', bus='b3', p_set=70.0)",
        "n.add('Line', 'l12', bus0='b1', bus1='b2', r=0.01, x=0.06, s_nom=70.0)",
        "n.add('Line', 'l23', bus0='b2', bus1='b3', r=0.01, x=0.06, s_nom=70.0)",
        "n.add('Line', 'l34', bus0='b3', bus1='b4', r=0.01, x=0.06, s_nom=120.0)",
        "n.add('Line', 'l14', bus0='b1', bus1='b4', r=0.012, x=0.07, s_nom=120.0)",
    ], "self-built 4-bus OPF network with a binding line limit (cheap vs expensive gen, meshed)"


def b_opf_link6():
    return [
        "n = pypsa.Network()",
        "n.add('Bus', 'a1', v_nom=110.0)",
        "n.add('Bus', 'a2', v_nom=110.0)",
        "n.add('Bus', 'a3', v_nom=110.0)",
        "n.add('Bus', 'd1', v_nom=110.0)",
        "n.add('Bus', 'd2', v_nom=110.0)",
        "n.add('Bus', 'd3', v_nom=110.0)",
        "n.add('Generator', 'genA', bus='a1', p_nom=180.0, marginal_cost=15.0)",
        "n.add('Generator', 'genB', bus='d1', p_nom=180.0, marginal_cost=45.0)",
        "n.add('Load', 'lda3', bus='a3', p_set=60.0)",
        "n.add('Load', 'ldd2', bus='d2', p_set=70.0)",
        "n.add('Load', 'ldd3', bus='d3', p_set=50.0)",
        "n.add('Line', 'la12', bus0='a1', bus1='a2', r=0.01, x=0.06, s_nom=200.0)",
        "n.add('Line', 'la23', bus0='a2', bus1='a3', r=0.01, x=0.06, s_nom=200.0)",
        "n.add('Line', 'ld12', bus0='d1', bus1='d2', r=0.01, x=0.06, s_nom=200.0)",
        "n.add('Line', 'ld23', bus0='d2', bus1='d3', r=0.01, x=0.06, s_nom=200.0)",
        "n.add('Link', 'hvdc', bus0='a2', bus1='d1', p_nom=120.0, marginal_cost=1.0)",
    ], "self-built 6-bus two-area OPF network joined by a controllable HVDC Link"


def b_acdc():
    return [
        "import pypsa.examples as ex",
        "n = ex.ac_dc_meshed()",
        "n.set_snapshots(n.snapshots[:1])",
    ], "PyPSA ac-dc-meshed example network (offline, first snapshot)"


BUILDERS = {
    "radial3": b_radial3,
    "radial4": b_radial4,
    "radial5": b_radial5,
    "mesh4": b_mesh4,
    "ring5": b_ring5,
    "trafo6": b_trafo6,
    "opf_econ5": b_opf_econ5,
    "opf_mesh4": b_opf_mesh4,
    "opf_link6": b_opf_link6,
    "acdc": b_acdc,
}

# ============================================================================================
# Modification & query renderers (single-source: rendered lines are both exec'd live and
# embedded into reference_code).
# ============================================================================================

def render_mod(m):
    op = m["op"]
    if op == "set_attr":
        return f"n.{m['comp']}.loc[{m['name']!r}, {m['attr']!r}] = {m['value']!r}"
    if op == "scale_attr":
        return f"n.{m['comp']}.loc[{m['name']!r}, {m['attr']!r}] *= {m['factor']!r}"
    if op == "add":
        kw = ", ".join(f"{k}={v!r}" for k, v in m["attrs"].items())
        return f"n.add({m['comp']!r}, {m['name']!r}, {kw})"
    if op == "remove":
        return f"n.remove({m['comp']!r}, {m['name']!r})"
    raise ValueError(f"unknown mod op {op}")


def render_query(q):
    qt = q["qtype"]
    sn = "n.snapshots[0]"
    if qt == "objective":
        return "result = n.objective"
    t = q["table"]
    if qt == "point":
        return f"result = n.{t}.loc[{sn}, {q['elem']!r}]"
    if qt == "point_abs":
        return f"result = abs(n.{t}.loc[{sn}, {q['elem']!r}])"
    if qt == "point_deg":
        return f"result = np.rad2deg(n.{t}.loc[{sn}, {q['elem']!r}])"
    if qt == "min":
        return f"result = n.{t}.loc[{sn}].min()"
    if qt == "max":
        return f"result = n.{t}.loc[{sn}].max()"
    if qt == "max_abs":
        return f"result = n.{t}.loc[{sn}].abs().max()"
    raise ValueError(f"unknown qtype {qt}")


HEADER = (
    "import os\n"
    "os.environ['OMP_NUM_THREADS'] = '1'\n"
    "os.environ['OPENBLAS_NUM_THREADS'] = '1'\n"
    "os.environ['MKL_NUM_THREADS'] = '1'\n"
    "import warnings; warnings.filterwarnings('ignore')\n"
    "import logging; logging.disable(logging.WARNING)\n"
    "import pandas as pd; pd.set_option('future.infer_string', False)\n"
    "import numpy as np\n"
    "import pypsa"
)

_OPF_OPTS = "{'output_flag': False, 'threads': 1}"
ANALYSIS = {
    "pf": [
        "_conv = n.pf()['converged']",
        "assert bool(np.asarray(_conv).all()), 'power flow did not converge'",
    ],
    "lpf": ["n.lpf()"],
    "opf": [
        f"_status = n.optimize(solver_name='highs', solver_options={_OPF_OPTS})",
        "assert _status[1] == 'optimal', 'OPF not optimal'",
    ],
}


def assemble_code(recipe, mod_lines, analysis, query):
    parts = [HEADER, "", "# Build network"]
    parts += recipe
    if mod_lines:
        parts += ["", "# Apply modifications"]
        parts += mod_lines
    parts += ["", "# Run analysis"]
    parts += ANALYSIS[analysis]
    parts += ["", "# Extract result", render_query(query), "print(round(float(result), 6))"]
    return "\n".join(parts) + "\n"


# ============================================================================================
# Live build + isolated-subprocess runner
# ============================================================================================

def build_live(recipe, mod_lines=None):
    ns = {"pypsa": pypsa, "np": np, "pd": pd}
    exec("\n".join(recipe), ns)
    if mod_lines:
        exec("\n".join(mod_lines), ns)
    return ns["n"]


def solve_live(n, analysis):
    if analysis == "pf":
        n.pf()
    elif analysis == "lpf":
        n.lpf()
    elif analysis == "opf":
        n.optimize(solver_name="highs", solver_options={"output_flag": False, "threads": 1})
    return n.snapshots[0]


def _get_table(n, path):
    obj = n
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def eval_query_live(n, q):
    """In-process mirror of render_query, for the sensitivity baseline (QC only)."""
    qt = q["qtype"]
    if qt == "objective":
        return float(n.objective)
    sn = n.snapshots[0]
    tbl = _get_table(n, q["table"])
    if qt == "point":
        return float(tbl.loc[sn, q["elem"]])
    if qt == "point_abs":
        return abs(float(tbl.loc[sn, q["elem"]]))
    if qt == "point_deg":
        return float(np.rad2deg(tbl.loc[sn, q["elem"]]))
    if qt == "min":
        return float(tbl.loc[sn].min())
    if qt == "max":
        return float(tbl.loc[sn].max())
    if qt == "max_abs":
        return float(tbl.loc[sn].abs().max())
    raise ValueError(f"unknown qtype {qt}")


def argsel_element(n, q):
    """For semantic (min/max/max_abs) queries, return the element the query resolves to on the
    given (modified) network -- i.e. the argmin/argmax index. Recorded for D3 items so the
    resolved target index is auditable (E1-style: verify the NL does not leak this element)."""
    qt = q["qtype"]
    if qt not in ("min", "max", "max_abs"):
        return None
    sn = n.snapshots[0]
    row = _get_table(n, q["table"]).loc[sn]
    if qt == "min":
        return str(row.astype(float).idxmin())
    if qt == "max":
        return str(row.astype(float).idxmax())
    return str(row.abs().idxmax())  # max_abs


def run_isolated(code):
    """Run reference_code in an isolated subprocess; return (value_or_None, err)."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", dir=str(SCRATCH), delete=False
    ) as fh:
        fh.write(code)
        path = fh.name
    try:
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "1"
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=SUBPROC_TIMEOUT, env=env,
        )
        if proc.returncode != 0:
            return None, proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "nonzero exit"
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        if not lines:
            return None, "no stdout"
        try:
            return float(lines[-1]), None
        except ValueError:
            return None, f"unparsable stdout tail: {lines[-1][:80]!r}"
    except subprocess.TimeoutExpired:
        return None, "timeout"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def sensitive(gt, gt_base):
    if gt_base is None or gt is None:
        return False, None
    d = abs(gt - gt_base)
    rel = d / abs(gt_base) if abs(gt_base) > 1e-12 else float("inf")
    return (d > SENS_ABS_TOL and (abs(gt_base) < 1e-9 or rel > SENS_REL_TOL)), d


# ============================================================================================
# Semantic resolvers (D3): resolve target element against the built (pre-modification) network.
# ============================================================================================

def resolve(kind, recipe, analysis):
    n = build_live(recipe)
    if kind == "heaviest_load":
        return str(n.loads["p_set"].astype(float).idxmax())
    if kind == "highest_load_bus":
        # bus carrying the greatest TOTAL load -> return the load element sitting on it
        # (networks here have <=1 load/bus, so this is the load to scale; resolver must
        #  return a *load* index because the mod indexes into n.loads).
        top_bus = str(n.loads.groupby("bus")["p_set"].sum().astype(float).idxmax())
        loads_here = list(n.loads.index[n.loads["bus"] == top_bus])
        return str(loads_here[0])
    if kind == "cheapest_gen":
        return str(n.generators["marginal_cost"].astype(float).idxmin())
    if kind == "most_expensive_gen":
        return str(n.generators["marginal_cost"].astype(float).idxmax())
    if kind == "highest_reactance_line":
        return str(n.lines["x"].astype(float).idxmax())
    if kind == "most_loaded_line":
        sn = solve_live(n, analysis)
        return str(n.lines_t.p0.loc[sn].abs().idxmax())
    if kind == "lowest_voltage_bus":
        sn = solve_live(n, "pf")
        return str(n.buses_t.v_mag_pu.loc[sn].astype(float).idxmin())
    raise ValueError(f"unknown resolve kind {kind}")


# ============================================================================================
# Candidate item specification.
# Each candidate: dict(family, diff, builder, analysis, mods, query, nl, [resolved]).
# `mods` entries may carry {"resolve": <kind>} to fill name from the built network state (D3).
# ============================================================================================

def _mk(family, diff, builder, analysis, mods, query, nl, resolved=None):
    return dict(family=family, diff=diff, builder=builder, analysis=analysis,
                mods=mods, query=query, nl=nl, resolved=resolved)


def candidates():
    """Sensitivity-robust candidate pool (E1 discipline: the queried scalar must move under
    the item's modification). Query-type choice is governed by physics:
      - line/transformer/link FLOW magnitudes and OPF objective/dispatch/LMP move robustly on
        any network -> used freely on meshed nets;
      - bus voltage MAGNITUDE only moves appreciably on electrically weak buses -> restricted to
        the radial (20 kV) feeders and the transformer LV side;
      - no bool queries; no RNG; every scalar is deterministic.
    Each cell is over-generated (>=10 candidates for 9 slots) so the sensitivity/GT filter can
    cull without shrinking a cell below target.
    """
    C = []

    # ============================ power_flow (n.pf) ============================
    # D1: explicit params + explicit query element.
    C += [
        _mk("power_flow", "D1_basic", "radial3", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld3", "attr": "p_set", "value": 120.0}],
            {"qtype": "point", "table": "buses_t.v_mag_pu", "elem": "b3"},
            "Set the active power demand of load 'ld3' to 120 MW, run an AC power flow, and "
            "report the voltage magnitude (p.u.) at bus 'b3'."),
        _mk("power_flow", "D1_basic", "radial4", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld4", "attr": "p_set", "value": 85.0}],
            {"qtype": "point", "table": "buses_t.v_mag_pu", "elem": "b4"},
            "After setting load 'ld4' to 85 MW, execute a non-linear power flow and report the "
            "voltage magnitude (p.u.) at the feeder-end bus 'b4'."),
        _mk("power_flow", "D1_basic", "radial4", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld3", "attr": "p_set", "value": 70.0}],
            {"qtype": "point", "table": "buses_t.v_mag_pu", "elem": "b3"},
            "Change the demand at load 'ld3' to 70 MW, solve the AC power flow, and return the "
            "voltage magnitude (p.u.) at bus 'b3'."),
        _mk("power_flow", "D1_basic", "radial5", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld5", "attr": "p_set", "value": 65.0}],
            {"qtype": "point", "table": "buses_t.v_mag_pu", "elem": "b5"},
            "Set load 'ld5' to 65 MW, run the power flow, and report the voltage magnitude (p.u.) "
            "at the remote bus 'b5'."),
        _mk("power_flow", "D1_basic", "trafo6", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld_lv3", "attr": "p_set", "value": 45.0}],
            {"qtype": "point", "table": "buses_t.v_mag_pu", "elem": "lv3"},
            "Set the load 'ld_lv3' to 45 MW, run the power flow, and report the voltage magnitude "
            "(p.u.) at the low-voltage bus 'lv3'."),
        _mk("power_flow", "D1_basic", "trafo6", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld_lv2", "attr": "p_set", "value": 42.0}],
            {"qtype": "point", "table": "buses_t.v_mag_pu", "elem": "lv2"},
            "Set the low-voltage load 'ld_lv2' to 42 MW, solve the AC power flow, and report the "
            "voltage magnitude (p.u.) at bus 'lv2'."),
        _mk("power_flow", "D1_basic", "mesh4", "pf",
            [{"op": "set_attr", "comp": "generators", "name": "g2", "attr": "p_set", "value": 90.0}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l12"},
            "Set the PV generator 'g2' to dispatch 90 MW, run the AC power flow, and report the "
            "magnitude of the active power flow (MW) on line 'l12'."),
        _mk("power_flow", "D1_basic", "radial3", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld2", "attr": "q_set", "value": 35.0}],
            {"qtype": "point_abs", "table": "lines_t.q0", "elem": "l12"},
            "Set the reactive demand of load 'ld2' to 35 MVAr, solve the power flow, and report "
            "the magnitude of the reactive power flow (MVAr) on line 'l12'."),
        _mk("power_flow", "D1_basic", "ring5", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld3", "attr": "p_set", "value": 100.0}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l23"},
            "Raise the demand of load 'ld3' to 100 MW, run the power flow, and report the "
            "magnitude of the active power flow (MW) on line 'l23'."),
        _mk("power_flow", "D1_basic", "trafo6", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld_lv2", "attr": "p_set", "value": 40.0}],
            {"qtype": "point_abs", "table": "transformers_t.p1", "elem": "T1"},
            "Set the low-voltage load 'ld_lv2' to 40 MW, solve the AC power flow, and report the "
            "magnitude of the active power (MW) at the low-voltage side (p1) of transformer 'T1'."),
        _mk("power_flow", "D1_basic", "mesh4", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld3", "attr": "p_set", "value": 120.0}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l23"},
            "Set load 'ld3' to 120 MW, run the AC power flow, and report the magnitude of the "
            "active power flow (MW) on line 'l23'."),
        _mk("power_flow", "D1_basic", "ring5", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld4", "attr": "p_set", "value": 90.0}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l34"},
            "Set the demand at load 'ld4' to 90 MW, solve the power flow, and report the magnitude "
            "of the active power flow (MW) on line 'l34'."),
    ]
    # D3: semantic modification target resolved against the built (pre-mod) network, then
    #     hardcoded; queries stay robust (flow-max on meshed nets, voltage-min on weak feeders).
    C += [
        _mk("power_flow", "D3_semantic", "mesh4", "pf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "heaviest_load", "attr": "p_set", "factor": 1.25}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Take the network's most heavily loaded demand and increase its active power by 25%, "
            "then run the AC power flow and report the largest active power flow magnitude (MW) on "
            "any line.", "heaviest_load"),
        _mk("power_flow", "D3_semantic", "ring5", "pf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "heaviest_load", "attr": "p_set", "factor": 1.40}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Increase the single largest active-power load by 40%, solve the power flow, and report "
            "the maximum active power flow magnitude (MW) across all lines.", "heaviest_load"),
        _mk("power_flow", "D3_semantic", "mesh4", "pf",
            [{"op": "scale_attr", "comp": "lines", "resolve": "most_loaded_line", "attr": "x", "factor": 0.5}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Identify the most heavily loaded line under the current operating point and halve its "
            "reactance to reinforce it; then run the power flow and report the largest active power "
            "flow magnitude (MW) on any line.", "most_loaded_line"),
        _mk("power_flow", "D3_semantic", "ring5", "pf",
            [{"op": "scale_attr", "comp": "lines", "resolve": "most_loaded_line", "attr": "x", "factor": 0.5}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Reinforce the most heavily loaded line by cutting its reactance in half, then solve the "
            "AC power flow and report the largest active power flow magnitude (MW) on any line.",
            "most_loaded_line"),
        _mk("power_flow", "D3_semantic", "mesh4", "pf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "heaviest_load", "attr": "p_set", "factor": 1.20}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Raise the heaviest active-power demand in the system by 20%, run the power flow, and "
            "report the largest line active power flow magnitude (MW).", "heaviest_load"),
        _mk("power_flow", "D3_semantic", "ring5", "pf",
            [{"op": "scale_attr", "comp": "lines", "resolve": "highest_reactance_line", "attr": "x", "factor": 0.5}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Upgrade the line with the highest reactance by halving its reactance, then run the power "
            "flow and report the largest active power flow magnitude (MW) on any line.",
            "highest_reactance_line"),
        _mk("power_flow", "D3_semantic", "radial4", "pf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "heaviest_load", "attr": "p_set", "factor": 1.30}],
            {"qtype": "min", "table": "buses_t.v_mag_pu"},
            "Increase the biggest demand on the feeder by 30%, then run the AC power flow and report "
            "the lowest bus voltage magnitude (p.u.) in the network.", "heaviest_load"),
        _mk("power_flow", "D3_semantic", "radial5", "pf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "heaviest_load", "attr": "p_set", "factor": 1.40}],
            {"qtype": "min", "table": "buses_t.v_mag_pu"},
            "Grow the largest load on the feeder by 40%, solve the power flow, and report the minimum "
            "voltage magnitude (p.u.) across all buses.", "heaviest_load"),
        _mk("power_flow", "D3_semantic", "radial4", "pf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "highest_load_bus", "attr": "p_set", "factor": 1.35}],
            {"qtype": "min", "table": "buses_t.v_mag_pu"},
            "Find the bus carrying the greatest total load and increase that load by 35%, then run "
            "the power flow and report the lowest bus voltage magnitude (p.u.) in the network.",
            "highest_load_bus"),
        _mk("power_flow", "D3_semantic", "radial3", "pf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "heaviest_load", "attr": "p_set", "factor": 1.30}],
            {"qtype": "min", "table": "buses_t.v_mag_pu"},
            "Increase the largest demand in the network by 30%, solve the AC power flow, and report "
            "the minimum bus voltage magnitude (p.u.).", "heaviest_load"),
        _mk("power_flow", "D3_semantic", "trafo6", "pf",
            [{"op": "scale_attr", "comp": "lines", "resolve": "most_loaded_line", "attr": "x", "factor": 0.5}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Halve the reactance of the most heavily loaded line, then run the AC power flow and "
            "report the largest active power flow magnitude (MW) on any line.", "most_loaded_line"),
    ]

    # ============================ linear_power_flow (n.lpf) ============================
    # DC (linear) flow holds bus voltage magnitudes at nominal, so all queries are line flows.
    C += [
        _mk("linear_power_flow", "D1_basic", "mesh4", "lpf",
            [{"op": "set_attr", "comp": "loads", "name": "ld3", "attr": "p_set", "value": 120.0}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l23"},
            "Set load 'ld3' to 120 MW, run a linear (DC) power flow, and report the magnitude of the "
            "active power flow (MW) on line 'l23'."),
        _mk("linear_power_flow", "D1_basic", "ring5", "lpf",
            [{"op": "set_attr", "comp": "loads", "name": "ld4", "attr": "p_set", "value": 95.0}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l34"},
            "After changing load 'ld4' to 95 MW, solve the linear power flow and report the magnitude "
            "of the active power flow (MW) on line 'l34'."),
        _mk("linear_power_flow", "D1_basic", "trafo6", "lpf",
            [{"op": "set_attr", "comp": "loads", "name": "ld_hv2", "attr": "p_set", "value": 70.0}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "lhv12"},
            "Set the load 'ld_hv2' to 70 MW, run the DC power flow, and report the magnitude of the "
            "active power flow (MW) on line 'lhv12'."),
        _mk("linear_power_flow", "D1_basic", "mesh4", "lpf",
            [{"op": "set_attr", "comp": "loads", "name": "ld4", "attr": "p_set", "value": 110.0}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l41"},
            "Change load 'ld4' to 110 MW, solve the linear power flow, and report the magnitude of "
            "the active power flow (MW) on line 'l41'."),
        _mk("linear_power_flow", "D1_basic", "ring5", "lpf",
            [{"op": "set_attr", "comp": "loads", "name": "ld2", "attr": "p_set", "value": 85.0}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l12"},
            "Set load 'ld2' to 85 MW, run the DC power flow, and report the magnitude of the active "
            "power flow (MW) on line 'l12'."),
        _mk("linear_power_flow", "D1_basic", "acdc", "lpf",
            [{"op": "set_attr", "comp": "lines", "name": "1", "attr": "x", "value": 0.5}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "0"},
            "In the ac-dc-meshed network, set the reactance of line '1' to 0.5 and run a linear power "
            "flow; report the magnitude of the active power flow (MW) on line '0'."),
        _mk("linear_power_flow", "D1_basic", "mesh4", "lpf",
            [{"op": "set_attr", "comp": "generators", "name": "g2", "attr": "p_set", "value": 100.0}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l12"},
            "Dispatch PV generator 'g2' at 100 MW, run the linear power flow, and report the magnitude "
            "of the active power flow (MW) on line 'l12'."),
        _mk("linear_power_flow", "D1_basic", "trafo6", "lpf",
            [{"op": "set_attr", "comp": "loads", "name": "ld_lv2", "attr": "p_set", "value": 45.0}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "lhv13"},
            "Set load 'ld_lv2' to 45 MW, run the DC power flow, and report the magnitude of the active "
            "power flow (MW) on line 'lhv13'."),
        _mk("linear_power_flow", "D1_basic", "ring5", "lpf",
            [{"op": "set_attr", "comp": "loads", "name": "ld5", "attr": "p_set", "value": 80.0}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l51"},
            "Set load 'ld5' to 80 MW, solve the linear power flow, and report the magnitude of the "
            "active power flow (MW) on line 'l51'."),
        _mk("linear_power_flow", "D1_basic", "radial4", "lpf",
            [{"op": "set_attr", "comp": "loads", "name": "ld4", "attr": "p_set", "value": 80.0}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l12"},
            "Set load 'ld4' to 80 MW, run the DC power flow, and report the magnitude of the active "
            "power flow (MW) on line 'l12'."),
        _mk("linear_power_flow", "D1_basic", "mesh4", "lpf",
            [{"op": "set_attr", "comp": "loads", "name": "ld3", "attr": "p_set", "value": 130.0}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l34"},
            "Set load 'ld3' to 130 MW, run a DC power flow, and report the magnitude of the active "
            "power flow (MW) on line 'l34'."),
    ]
    C += [
        _mk("linear_power_flow", "D3_semantic", "mesh4", "lpf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "heaviest_load", "attr": "p_set", "factor": 1.30}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Increase the network's largest active-power demand by 30%, run the linear power flow, and "
            "report the largest active power flow magnitude (MW) on any line.", "heaviest_load"),
        _mk("linear_power_flow", "D3_semantic", "ring5", "lpf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "heaviest_load", "attr": "p_set", "factor": 1.50}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Grow the single biggest load by 50%, solve the DC power flow, and report the maximum active "
            "power flow magnitude (MW) across all lines.", "heaviest_load"),
        _mk("linear_power_flow", "D3_semantic", "mesh4", "lpf",
            [{"op": "scale_attr", "comp": "lines", "resolve": "most_loaded_line", "attr": "x", "factor": 0.5}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Find the line carrying the largest flow and halve its reactance, then re-run the linear power "
            "flow and report the largest active power flow magnitude (MW) on any line.", "most_loaded_line"),
        _mk("linear_power_flow", "D3_semantic", "ring5", "lpf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "highest_load_bus", "attr": "p_set", "factor": 1.40}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Increase the load at the most heavily loaded bus by 40%, run the DC power flow, and report the "
            "maximum active power flow magnitude (MW) over all lines.", "highest_load_bus"),
        _mk("linear_power_flow", "D3_semantic", "trafo6", "lpf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "heaviest_load", "attr": "p_set", "factor": 1.60}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Take the largest demand in the system and increase it by 60%, then solve the linear power flow "
            "and report the largest line active power flow magnitude (MW).", "heaviest_load"),
        _mk("linear_power_flow", "D3_semantic", "acdc", "lpf",
            [{"op": "scale_attr", "comp": "lines", "resolve": "most_loaded_line", "attr": "x", "factor": 0.5}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "In the ac-dc-meshed network, halve the reactance of the most heavily loaded AC line, then run "
            "a linear power flow and report the largest line active power flow magnitude (MW).",
            "most_loaded_line"),
        _mk("linear_power_flow", "D3_semantic", "mesh4", "lpf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "highest_load_bus", "attr": "p_set", "factor": 1.35}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Increase the total demand at the most heavily loaded bus by 35%, run the linear power flow, and "
            "report the maximum line active power flow magnitude (MW).", "highest_load_bus"),
        _mk("linear_power_flow", "D3_semantic", "ring5", "lpf",
            [{"op": "scale_attr", "comp": "lines", "resolve": "highest_reactance_line", "attr": "x", "factor": 0.5}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Upgrade the highest-reactance line by halving its reactance, then run the DC power flow and report "
            "the maximum active power flow magnitude (MW) on any line.", "highest_reactance_line"),
        _mk("linear_power_flow", "D3_semantic", "trafo6", "lpf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "highest_load_bus", "attr": "p_set", "factor": 1.45}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Increase the load at the most heavily loaded bus by 45%, solve the linear power flow, and report "
            "the largest line active power flow magnitude (MW).", "highest_load_bus"),
        _mk("linear_power_flow", "D3_semantic", "ring5", "lpf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "heaviest_load", "attr": "p_set", "factor": 1.35}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Increase the biggest load by 35%, run the DC power flow, and report the maximum active power flow "
            "magnitude (MW) across all lines.", "heaviest_load"),
    ]

    # ============================ opf (n.optimize / LOPF) ============================
    C += [
        _mk("opf", "D1_basic", "opf_econ5", "opf",
            [{"op": "set_attr", "comp": "loads", "name": "ld4", "attr": "p_set", "value": 140.0}],
            {"qtype": "objective"},
            "Set load 'ld4' to 140 MW and solve the OPF (LOPF); report the total optimal system cost "
            "(objective value)."),
        _mk("opf", "D1_basic", "opf_mesh4", "opf",
            [{"op": "set_attr", "comp": "loads", "name": "ld3", "attr": "p_set", "value": 90.0}],
            {"qtype": "objective"},
            "Change load 'ld3' to 90 MW, run the OPF, and report the minimized total generation cost "
            "(objective value)."),
        _mk("opf", "D1_basic", "opf_econ5", "opf",
            [{"op": "set_attr", "comp": "loads", "name": "ld2", "attr": "p_set", "value": 120.0}],
            {"qtype": "point", "table": "generators_t.p", "elem": "mid"},
            "Set load 'ld2' to 120 MW, solve the OPF, and report the optimal active power dispatch (MW) "
            "of generator 'mid'."),
        _mk("opf", "D1_basic", "opf_mesh4", "opf",
            [{"op": "set_attr", "comp": "loads", "name": "ld2", "attr": "p_set", "value": 90.0}],
            {"qtype": "point", "table": "generators_t.p", "elem": "exp"},
            "Set load 'ld2' to 90 MW, run the OPF, and report the optimal active power dispatch (MW) of "
            "the expensive generator 'exp'."),
        _mk("opf", "D1_basic", "opf_link6", "opf",
            [{"op": "set_attr", "comp": "loads", "name": "ldd2", "attr": "p_set", "value": 110.0}],
            {"qtype": "point_abs", "table": "links_t.p0", "elem": "hvdc"},
            "Set load 'ldd2' to 110 MW and solve the OPF; report the magnitude of the optimal power flow "
            "(MW) on the HVDC link 'hvdc'."),
        _mk("opf", "D1_basic", "opf_econ5", "opf",
            [{"op": "set_attr", "comp": "generators", "name": "cheap", "attr": "marginal_cost", "value": 40.0}],
            {"qtype": "objective"},
            "Raise the marginal cost of generator 'cheap' to 40 currency/MWh, solve the OPF, and report the "
            "total optimal system cost (objective value)."),
        _mk("opf", "D1_basic", "opf_mesh4", "opf",
            [{"op": "set_attr", "comp": "generators", "name": "cheap", "attr": "p_nom", "value": 60.0}],
            {"qtype": "objective"},
            "Reduce the capacity (p_nom) of generator 'cheap' to 60 MW, run the OPF, and report the "
            "minimized total generation cost (objective value)."),
        _mk("opf", "D1_basic", "opf_link6", "opf",
            [{"op": "set_attr", "comp": "loads", "name": "ldd2", "attr": "p_set", "value": 120.0}],
            {"qtype": "objective"},
            "Set load 'ldd2' to 120 MW, solve the OPF, and report the total optimal system cost "
            "(objective value)."),
        _mk("opf", "D1_basic", "opf_econ5", "opf",
            [{"op": "set_attr", "comp": "loads", "name": "ld4", "attr": "p_set", "value": 100.0}],
            {"qtype": "point", "table": "generators_t.p", "elem": "mid"},
            "Set load 'ld4' to 100 MW, run the OPF, and report the optimal active power dispatch (MW) of "
            "generator 'mid'."),
        _mk("opf", "D1_basic", "opf_econ5", "opf",
            [{"op": "set_attr", "comp": "loads", "name": "ld2", "attr": "p_set", "value": 130.0}],
            {"qtype": "point", "table": "buses_t.marginal_price", "elem": "b2"},
            "Set load 'ld2' to 130 MW, solve the OPF, and report the locational marginal price at bus 'b2'."),
        _mk("opf", "D1_basic", "opf_mesh4", "opf",
            [{"op": "set_attr", "comp": "generators", "name": "exp", "attr": "marginal_cost", "value": 30.0}],
            {"qtype": "objective"},
            "Lower the marginal cost of generator 'exp' to 30 currency/MWh, run the OPF, and report the "
            "total optimal system cost (objective value)."),
    ]
    C += [
        _mk("opf", "D3_semantic", "opf_econ5", "opf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "heaviest_load", "attr": "p_set", "factor": 1.30}],
            {"qtype": "objective"},
            "Increase the largest load in the system by 30% and re-solve the OPF; report the total optimal "
            "system cost (objective value).", "heaviest_load"),
        _mk("opf", "D3_semantic", "opf_mesh4", "opf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "heaviest_load", "attr": "p_set", "factor": 1.40}],
            {"qtype": "objective"},
            "Grow the biggest demand by 40%, run the OPF, and report the minimized total generation cost "
            "(objective value).", "heaviest_load"),
        _mk("opf", "D3_semantic", "opf_econ5", "opf",
            [{"op": "set_attr", "comp": "generators", "resolve": "cheapest_gen", "attr": "marginal_cost", "value": 90.0}],
            {"qtype": "objective"},
            "Find the cheapest generator and raise its marginal cost to 90 currency/MWh (making it the most "
            "expensive), then re-optimize and report the total optimal system cost (objective value).",
            "cheapest_gen"),
        _mk("opf", "D3_semantic", "opf_econ5", "opf",
            [{"op": "scale_attr", "comp": "generators", "resolve": "cheapest_gen", "attr": "p_nom", "factor": 0.5}],
            {"qtype": "objective"},
            "Halve the capacity of the cheapest generator, solve the OPF, and report the total optimal system "
            "cost (objective value).", "cheapest_gen"),
        _mk("opf", "D3_semantic", "opf_mesh4", "opf",
            [{"op": "set_attr", "comp": "generators", "resolve": "cheapest_gen", "attr": "p_nom", "value": 50.0}],
            {"qtype": "point", "table": "generators_t.p", "elem_resolve": "most_expensive_gen"},
            "Cap the capacity of the cheapest generator at 50 MW, re-solve the OPF, and report the optimal "
            "active power dispatch (MW) of the most expensive generator.", "cheapest_gen"),
        _mk("opf", "D3_semantic", "opf_econ5", "opf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "heaviest_load", "attr": "p_set", "factor": 1.25}],
            {"qtype": "point", "table": "generators_t.p", "elem": "peak"},
            "Increase the heaviest load by 25%, run the OPF, and report the optimal active power dispatch (MW) "
            "of the peaking generator 'peak'.", "heaviest_load"),
        _mk("opf", "D3_semantic", "opf_link6", "opf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "highest_load_bus", "attr": "p_set", "factor": 1.35}],
            {"qtype": "objective"},
            "Increase the load at the most heavily loaded bus by 35%, solve the OPF, and report the total "
            "optimal system cost (objective value).", "highest_load_bus"),
        _mk("opf", "D3_semantic", "opf_mesh4", "opf",
            [{"op": "set_attr", "comp": "generators", "resolve": "most_expensive_gen", "attr": "marginal_cost", "value": 20.0}],
            {"qtype": "objective"},
            "Cut the marginal cost of the most expensive generator down to 20 currency/MWh, re-optimize, and "
            "report the total optimal system cost (objective value).", "most_expensive_gen"),
        _mk("opf", "D3_semantic", "opf_mesh4", "opf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "heaviest_load", "attr": "p_set", "factor": 1.50}],
            {"qtype": "objective"},
            "Increase the largest demand by 50%, run the OPF, and report the total optimal system cost "
            "(objective value).", "heaviest_load"),
        _mk("opf", "D3_semantic", "opf_link6", "opf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "heaviest_load", "attr": "p_set", "factor": 1.30}],
            {"qtype": "objective"},
            "Increase the biggest load in the two-area system by 30%, solve the OPF, and report the total "
            "optimal system cost (objective value).", "heaviest_load"),
        _mk("opf", "D3_semantic", "opf_econ5", "opf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "heaviest_load", "attr": "p_set", "factor": 1.20}],
            {"qtype": "point", "table": "generators_t.p", "elem": "mid"},
            "Increase the largest load by 20%, run the OPF, and report the optimal active power dispatch (MW) "
            "of the mid-merit generator 'mid'.", "heaviest_load"),
    ]

    # ==================== network_modification (add/remove/attr-modify -> pf) ====================
    C += [
        _mk("network_modification", "D1_basic", "mesh4", "pf",
            [{"op": "add", "comp": "Line", "name": "lnew",
              "attrs": {"bus0": "b2", "bus1": "b4", "r": 0.03, "x": 0.08, "s_nom": 150.0}}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l13"},
            "Add a new line 'lnew' between bus 'b2' and bus 'b4' with r=0.03, x=0.08 and s_nom=150 MVA, "
            "then run the AC power flow and report the magnitude of the active power flow (MW) on line 'l13'."),
        _mk("network_modification", "D1_basic", "mesh4", "pf",
            [{"op": "remove", "comp": "Line", "name": "l13"}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l23"},
            "Remove line 'l13' from the network, run the AC power flow, and report the magnitude of the "
            "active power flow (MW) on line 'l23'."),
        _mk("network_modification", "D1_basic", "ring5", "pf",
            [{"op": "remove", "comp": "Line", "name": "l45"}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l51"},
            "Take line 'l45' out of the network, solve the power flow, and report the magnitude of the "
            "active power flow (MW) on line 'l51'."),
        _mk("network_modification", "D1_basic", "radial3", "pf",
            [{"op": "add", "comp": "Load", "name": "ld_extra",
              "attrs": {"bus": "b3", "p_set": 30.0, "q_set": 10.0}}],
            {"qtype": "point", "table": "buses_t.v_mag_pu", "elem": "b3"},
            "Add a new load 'ld_extra' of 30 MW / 10 MVAr at bus 'b3', run the power flow, and report the "
            "voltage magnitude (p.u.) at bus 'b3'."),
        _mk("network_modification", "D1_basic", "mesh4", "pf",
            [{"op": "add", "comp": "Generator", "name": "g_extra",
              "attrs": {"bus": "b4", "control": "PV", "p_nom": 100.0, "p_set": 40.0}}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l34"},
            "Add a PV generator 'g_extra' at bus 'b4' dispatching 40 MW (p_nom 100 MW), run the AC power "
            "flow, and report the magnitude of the active power flow (MW) on line 'l34'."),
        _mk("network_modification", "D1_basic", "ring5", "pf",
            [{"op": "add", "comp": "Line", "name": "l24",
              "attrs": {"bus0": "b2", "bus1": "b4", "r": 0.025, "x": 0.07, "s_nom": 160.0}}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l23"},
            "Add a tie line 'l24' between bus 'b2' and bus 'b4' (r=0.025, x=0.07, s_nom=160 MVA), run the "
            "power flow, and report the magnitude of the active power flow (MW) on line 'l23'."),
        _mk("network_modification", "D1_basic", "trafo6", "pf",
            [{"op": "set_attr", "comp": "transformers", "name": "T1", "attr": "x", "value": 0.12}],
            {"qtype": "point", "table": "buses_t.v_mag_pu", "elem": "lv2"},
            "Increase the transformer 'T1' reactance (x) to 0.12 p.u., run the AC power flow, and report the "
            "voltage magnitude (p.u.) at the low-voltage bus 'lv2'."),
        _mk("network_modification", "D1_basic", "mesh4", "pf",
            [{"op": "remove", "comp": "Line", "name": "l34"}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l23"},
            "Remove line 'l34', solve the AC power flow, and report the magnitude of the active power flow "
            "(MW) on line 'l23'."),
        _mk("network_modification", "D1_basic", "radial3", "pf",
            [{"op": "add", "comp": "Line", "name": "l13",
              "attrs": {"bus0": "b1", "bus1": "b3", "r": 0.05, "x": 0.13, "s_nom": 120.0}}],
            {"qtype": "point", "table": "buses_t.v_mag_pu", "elem": "b3"},
            "Add a new line 'l13' directly between bus 'b1' and bus 'b3' (r=0.05, x=0.13, s_nom=120 MVA), "
            "run the power flow, and report the voltage magnitude (p.u.) at bus 'b3'."),
        _mk("network_modification", "D1_basic", "radial4", "pf",
            [{"op": "add", "comp": "Load", "name": "ld_extra",
              "attrs": {"bus": "b4", "p_set": 25.0, "q_set": 8.0}}],
            {"qtype": "point", "table": "buses_t.v_mag_pu", "elem": "b4"},
            "Add an extra load 'ld_extra' of 25 MW / 8 MVAr at the feeder-end bus 'b4', run the AC power "
            "flow, and report the voltage magnitude (p.u.) at bus 'b4'."),
        _mk("network_modification", "D1_basic", "trafo6", "pf",
            [{"op": "set_attr", "comp": "transformers", "name": "T1", "attr": "x", "value": 0.15}],
            {"qtype": "point", "table": "buses_t.v_mag_pu", "elem": "lv3"},
            "Set the transformer 'T1' reactance (x) to 0.15 p.u., solve the AC power flow, and report the "
            "voltage magnitude (p.u.) at the low-voltage bus 'lv3'."),
    ]
    C += [
        _mk("network_modification", "D3_semantic", "mesh4", "pf",
            [{"op": "add_parallel_line", "resolve": "most_loaded_line", "name": "lreinf"}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Reinforce the network by adding a parallel circuit alongside the most heavily loaded line "
            "(identical parameters), then run the power flow and report the largest active power flow "
            "magnitude (MW) on any line.", "most_loaded_line"),
        _mk("network_modification", "D3_semantic", "ring5", "pf",
            [{"op": "add_parallel_line", "resolve": "most_loaded_line", "name": "lreinf"}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Add a parallel circuit to the most heavily loaded line to reinforce it, then solve the power "
            "flow and report the largest active power flow magnitude (MW) on any line.", "most_loaded_line"),
        _mk("network_modification", "D3_semantic", "mesh4", "pf",
            [{"op": "remove_line", "resolve": "most_loaded_line"}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Simulate an outage of the most heavily loaded line by removing it, then run the AC power flow "
            "and report the largest active power flow magnitude (MW) among the remaining lines.",
            "most_loaded_line"),
        _mk("network_modification", "D3_semantic", "ring5", "pf",
            [{"op": "scale_attr", "comp": "lines", "resolve": "most_loaded_line", "attr": "s_nom", "factor": 2.0},
             {"op": "scale_attr", "comp": "lines", "resolve": "most_loaded_line", "attr": "x", "factor": 0.5}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Upgrade the most heavily loaded line by doubling its rating and halving its reactance, then run "
            "the power flow and report the largest active power flow magnitude (MW) on any line.",
            "most_loaded_line"),
        _mk("network_modification", "D3_semantic", "mesh4", "pf",
            [{"op": "add_parallel_line", "resolve": "highest_reactance_line", "name": "lreinf"}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Add a parallel circuit to the highest-reactance line, then run the power flow and report the "
            "largest active power flow magnitude (MW) on any line.", "highest_reactance_line"),
        _mk("network_modification", "D3_semantic", "ring5", "pf",
            [{"op": "remove_line", "resolve": "highest_reactance_line"}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Remove the line with the highest reactance, solve the power flow, and report the largest active "
            "power flow magnitude (MW) among the remaining lines.", "highest_reactance_line"),
        _mk("network_modification", "D3_semantic", "ring5", "pf",
            [{"op": "add_parallel_line", "resolve": "highest_reactance_line", "name": "lreinf"}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Strengthen the highest-reactance line by adding an identical parallel circuit, then run the AC "
            "power flow and report the largest active power flow magnitude (MW) on any line.",
            "highest_reactance_line"),
        _mk("network_modification", "D3_semantic", "radial4", "pf",
            [{"op": "add_parallel_line", "resolve": "most_loaded_line", "name": "lreinf"}],
            {"qtype": "min", "table": "buses_t.v_mag_pu"},
            "Reinforce the feeder by adding a parallel circuit alongside its most heavily loaded line, then "
            "solve the power flow and report the minimum bus voltage magnitude (p.u.).", "most_loaded_line"),
        _mk("network_modification", "D3_semantic", "radial4", "pf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "heaviest_load", "attr": "p_set", "factor": 1.40}],
            {"qtype": "min", "table": "buses_t.v_mag_pu"},
            "Increase the largest load on the feeder by 40%, then run the AC power flow and report the lowest "
            "bus voltage magnitude (p.u.).", "heaviest_load"),
        _mk("network_modification", "D3_semantic", "radial5", "pf",
            [{"op": "add_parallel_line", "resolve": "most_loaded_line", "name": "lreinf"}],
            {"qtype": "min", "table": "buses_t.v_mag_pu"},
            "Add a parallel circuit to the busiest line of the feeder to strengthen it, then solve the power "
            "flow and report the minimum bus voltage magnitude (p.u.).", "most_loaded_line"),
        _mk("network_modification", "D3_semantic", "radial5", "pf",
            [{"op": "scale_attr", "comp": "loads", "resolve": "heaviest_load", "attr": "p_set", "factor": 1.35}],
            {"qtype": "min", "table": "buses_t.v_mag_pu"},
            "Increase the largest load on the feeder by 35%, solve the AC power flow, and report the minimum "
            "bus voltage magnitude (p.u.) in the network.", "heaviest_load"),
    ]

    # ==================== result_extraction (navigate result tables for a scalar) ====================
    C += [
        _mk("result_extraction", "D1_basic", "mesh4", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld3", "attr": "p_set", "value": 95.0}],
            {"qtype": "point_abs", "table": "lines_t.q0", "elem": "l23"},
            "Set load 'ld3' to 95 MW, run the AC power flow, and read the magnitude of the reactive power "
            "flow (MVAr) on line 'l23' from the results."),
        _mk("result_extraction", "D1_basic", "ring5", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld4", "attr": "p_set", "value": 85.0}],
            {"qtype": "point_abs", "table": "lines_t.p1", "elem": "l45"},
            "Set load 'ld4' to 85 MW, run the power flow, and read the magnitude of the active power (MW) at "
            "the receiving end (p1) of line 'l45' from the results."),
        _mk("result_extraction", "D1_basic", "trafo6", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld_lv2", "attr": "p_set", "value": 38.0}],
            {"qtype": "point_abs", "table": "transformers_t.p1", "elem": "T1"},
            "Set the low-voltage load 'ld_lv2' to 38 MW, run the AC power flow, and read the magnitude of the "
            "active power (MW) at the low-voltage side (p1) of transformer 'T1'."),
        _mk("result_extraction", "D1_basic", "opf_econ5", "opf",
            [{"op": "set_attr", "comp": "loads", "name": "ld4", "attr": "p_set", "value": 125.0}],
            {"qtype": "point", "table": "buses_t.marginal_price", "elem": "b4"},
            "Set load 'ld4' to 125 MW, solve the OPF, and read the locational marginal price at bus 'b4' "
            "from the results."),
        _mk("result_extraction", "D1_basic", "mesh4", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld4", "attr": "q_set", "value": 40.0}],
            {"qtype": "point_abs", "table": "generators_t.q", "elem": "g2"},
            "Set the reactive demand of load 'ld4' to 40 MVAr, run the AC power flow, and read the magnitude "
            "of the reactive power output (MVAr) of generator 'g2'."),
        _mk("result_extraction", "D1_basic", "opf_mesh4", "opf",
            [{"op": "set_attr", "comp": "loads", "name": "ld3", "attr": "p_set", "value": 85.0}],
            {"qtype": "point", "table": "buses_t.marginal_price", "elem": "b3"},
            "Set load 'ld3' to 85 MW, run the OPF, and read the locational marginal price at bus 'b3'."),
        _mk("result_extraction", "D1_basic", "ring5", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld2", "attr": "p_set", "value": 75.0}],
            {"qtype": "point_abs", "table": "lines_t.p1", "elem": "l12"},
            "Set load 'ld2' to 75 MW, run the power flow, and read the magnitude of the active power (MW) at "
            "the receiving end (p1) of line 'l12' from the results."),
        _mk("result_extraction", "D1_basic", "opf_link6", "opf",
            [{"op": "set_attr", "comp": "loads", "name": "ldd3", "attr": "p_set", "value": 80.0}],
            {"qtype": "point", "table": "generators_t.p", "elem": "genB"},
            "Set load 'ldd3' to 80 MW, solve the OPF, and read the optimal active power dispatch (MW) of "
            "generator 'genB'."),
        _mk("result_extraction", "D1_basic", "radial4", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld4", "attr": "p_set", "value": 75.0}],
            {"qtype": "point", "table": "buses_t.v_mag_pu", "elem": "b4"},
            "Set load 'ld4' to 75 MW, run the AC power flow, and read the voltage magnitude (p.u.) at the "
            "feeder-end bus 'b4' from the results."),
        _mk("result_extraction", "D1_basic", "mesh4", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld3", "attr": "p_set", "value": 105.0}],
            {"qtype": "point_abs", "table": "lines_t.p0", "elem": "l12"},
            "Set load 'ld3' to 105 MW, run the AC power flow, and read the magnitude of the active power "
            "flow (MW) on line 'l12' from the results."),
        _mk("result_extraction", "D1_basic", "opf_econ5", "opf",
            [{"op": "set_attr", "comp": "loads", "name": "ld2", "attr": "p_set", "value": 110.0}],
            {"qtype": "point", "table": "generators_t.p", "elem": "mid"},
            "Set load 'ld2' to 110 MW, solve the OPF, and read the optimal active power dispatch (MW) of "
            "generator 'mid' from the results."),
    ]
    C += [
        _mk("result_extraction", "D3_semantic", "mesh4", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld3", "attr": "p_set", "value": 110.0}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Set load 'ld3' to 110 MW, run the AC power flow, and report the active power flow magnitude (MW) "
            "on the most heavily loaded line."),
        _mk("result_extraction", "D3_semantic", "ring5", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld4", "attr": "p_set", "value": 95.0}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Set load 'ld4' to 95 MW, run the power flow, and report the active power flow magnitude (MW) on "
            "the most heavily loaded line in the network."),
        _mk("result_extraction", "D3_semantic", "opf_econ5", "opf",
            [{"op": "set_attr", "comp": "loads", "name": "ld4", "attr": "p_set", "value": 130.0}],
            {"qtype": "max", "table": "buses_t.marginal_price"},
            "Set load 'ld4' to 130 MW, solve the OPF, and report the highest locational marginal price across "
            "all buses."),
        _mk("result_extraction", "D3_semantic", "radial4", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld4", "attr": "p_set", "value": 80.0}],
            {"qtype": "min", "table": "buses_t.v_mag_pu"},
            "Set load 'ld4' to 80 MW, run the AC power flow, and report the voltage magnitude (p.u.) at the "
            "bus with the lowest voltage in the network."),
        _mk("result_extraction", "D3_semantic", "radial5", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld5", "attr": "p_set", "value": 60.0}],
            {"qtype": "min", "table": "buses_t.v_mag_pu"},
            "Set load 'ld5' to 60 MW, run the power flow, and report the voltage magnitude (p.u.) at the "
            "weakest (lowest-voltage) bus."),
        _mk("result_extraction", "D3_semantic", "opf_mesh4", "opf",
            [{"op": "set_attr", "comp": "loads", "name": "ld3", "attr": "p_set", "value": 95.0}],
            {"qtype": "max", "table": "buses_t.marginal_price"},
            "Set load 'ld3' to 95 MW, solve the OPF, and report the maximum locational marginal price over all "
            "buses."),
        _mk("result_extraction", "D3_semantic", "trafo6", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld_lv3", "attr": "p_set", "value": 45.0}],
            {"qtype": "min", "table": "buses_t.v_mag_pu"},
            "Set the low-voltage load 'ld_lv3' to 45 MW, run the AC power flow, and report the lowest bus "
            "voltage magnitude (p.u.) in the network."),
        _mk("result_extraction", "D3_semantic", "opf_econ5", "opf",
            [{"op": "set_attr", "comp": "loads", "name": "ld2", "attr": "p_set", "value": 105.0}],
            {"qtype": "max", "table": "generators_t.p"},
            "Set load 'ld2' to 105 MW, solve the OPF, and report the largest optimal active power dispatch (MW) "
            "among all generators."),
        _mk("result_extraction", "D3_semantic", "acdc", "lpf",
            [{"op": "set_attr", "comp": "lines", "name": "1", "attr": "x", "value": 0.4}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "In the ac-dc-meshed network, set the reactance of line '1' to 0.4 and run a linear power flow; "
            "report the largest active power flow magnitude (MW) on any AC line."),
        _mk("result_extraction", "D3_semantic", "radial4", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld3", "attr": "p_set", "value": 70.0}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Set load 'ld3' to 70 MW, run the AC power flow, and report the active power flow magnitude (MW) "
            "on the most heavily loaded line of the feeder."),
        _mk("result_extraction", "D3_semantic", "opf_link6", "opf",
            [{"op": "set_attr", "comp": "loads", "name": "ldd2", "attr": "p_set", "value": 100.0}],
            {"qtype": "max", "table": "generators_t.p"},
            "Set load 'ldd2' to 100 MW, solve the OPF, and report the largest optimal active power dispatch "
            "(MW) among all generators."),
        # extra robust D3 buffer (max-over-table on OPF dispatch/LMP can pin to a bound -> the
        # over-generated pool below leans on flow-max and weak-net voltage-min, which move reliably)
        _mk("result_extraction", "D3_semantic", "mesh4", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld4", "attr": "p_set", "value": 95.0}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Set load 'ld4' to 95 MW, run the AC power flow, and report the active power flow magnitude (MW) "
            "on the most heavily loaded line in the network."),
        _mk("result_extraction", "D3_semantic", "ring5", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld3", "attr": "p_set", "value": 100.0}],
            {"qtype": "max_abs", "table": "lines_t.p0"},
            "Set load 'ld3' to 100 MW, run the power flow, and report the active power flow magnitude (MW) on "
            "the busiest line in the network."),
        _mk("result_extraction", "D3_semantic", "radial5", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld4", "attr": "p_set", "value": 55.0}],
            {"qtype": "min", "table": "buses_t.v_mag_pu"},
            "Set load 'ld4' to 55 MW, run the AC power flow, and report the voltage magnitude (p.u.) at the "
            "weakest (lowest-voltage) bus in the network."),
        _mk("result_extraction", "D3_semantic", "trafo6", "pf",
            [{"op": "set_attr", "comp": "loads", "name": "ld_lv2", "attr": "p_set", "value": 42.0}],
            {"qtype": "min", "table": "buses_t.v_mag_pu"},
            "Set the low-voltage load 'ld_lv2' to 42 MW, run the AC power flow, and report the lowest bus "
            "voltage magnitude (p.u.) in the network."),
        _mk("result_extraction", "D3_semantic", "opf_econ5", "opf",
            [{"op": "set_attr", "comp": "loads", "name": "ld2", "attr": "p_set", "value": 115.0}],
            {"qtype": "max", "table": "buses_t.marginal_price"},
            "Set load 'ld2' to 115 MW, solve the OPF, and report the highest locational marginal price across "
            "all buses."),
    ]
    return C


# ============================================================================================
# Item realization (resolve D3 targets, render, run GT, admission, sensitivity).
# ============================================================================================


def realize_mods(spec):
    """Return (rendered_mod_lines, structured_mods, resolved_record)."""
    recipe, _ = BUILDERS[spec["builder"]]()
    rendered, structured = [], []
    resolved_record = None
    for m in spec["mods"]:
        m = dict(m)
        if "resolve" in m:
            kind = m.pop("resolve")
            elem = resolve(kind, recipe, spec["analysis"])
            resolved_record = {"semantic": spec.get("resolved") or kind, "resolved_element": elem}
            op = m["op"]
            if op == "add_parallel_line":
                # add a circuit with identical parameters to the resolved line
                base = build_live(recipe)
                row = base.lines.loc[elem]
                attrs = {"bus0": str(row["bus0"]), "bus1": str(row["bus1"]),
                         "r": float(row["r"]), "x": float(row["x"]), "s_nom": float(row["s_nom"])}
                sm = {"op": "add", "comp": "Line", "name": m["name"], "attrs": attrs,
                      "parallel_to": elem}
                structured.append(sm)
                rendered.append(render_mod({"op": "add", "comp": "Line", "name": m["name"], "attrs": attrs}))
                continue
            if op == "remove_line":
                sm = {"op": "remove", "comp": "Line", "name": elem}
                structured.append(sm)
                rendered.append(render_mod(sm))
                continue
            m["name"] = elem
        structured.append(m)
        rendered.append(render_mod(m))
    return rendered, structured, resolved_record


def realize_query(spec, recipe):
    """Resolve elem_resolve queries (D3) against the built network; return (query, extra_resolved)."""
    q = dict(spec["query"])
    extra = None
    if "elem_resolve" in q:
        kind = q.pop("elem_resolve")
        elem = resolve(kind, recipe, spec["analysis"])
        q["elem"] = elem
        extra = {"query_semantic": kind, "query_resolved_element": elem}
    return q, extra


def make_id(family, diff, code):
    h = hashlib.sha1(f"{family}|{diff}|{code}".encode()).hexdigest()[:12]
    return h


def realize(spec):
    recipe, netdesc = BUILDERS[spec["builder"]]()
    mod_lines, structured_mods, resolved_rec = realize_mods(spec)
    query, query_extra = realize_query(spec, recipe)
    code = assemble_code(recipe, mod_lines, spec["analysis"], query)

    # GROUND TRUTH: isolated subprocess (protocol §2.3, anti-contamination) -- authoritative.
    gt, err = run_isolated(code)
    if gt is None:
        return None, f"GT run failed: {err}"
    if not np.isfinite(gt):
        return None, f"GT non-finite: {gt}"
    gt = round(gt, 6)

    # SENSITIVITY BASELINE: in-process (QC only, not shipped as GT) -- keeps wall-clock bounded.
    try:
        nb = build_live(recipe)
        solve_live(nb, spec["analysis"])
        gt_base = round(eval_query_live(nb, query), 6)
        del nb
        gc.collect()
    except Exception as e:  # noqa: BLE001
        gt_base = None
        berr = f"{type(e).__name__}: {e}"
    else:
        berr = None
    ok, delta = sensitive(gt, gt_base)
    if not ok:
        return None, (f"NOT sensitive (gt={gt}, base={gt_base}, "
                      f"delta={delta}, berr={berr})")

    # D3 semantic-QUERY items (min/max/max_abs, no explicit elem): record which element the
    # query resolves to on the GT (modified, solved) network -- the resolved target index the
    # model must implicitly recover (E1 audit: confirm the NL does not name this element).
    query_argsel = None
    if spec["diff"] == "D3_semantic" and query["qtype"] in ("min", "max", "max_abs"):
        try:
            nm = build_live(recipe, mod_lines)
            solve_live(nm, spec["analysis"])
            elem = argsel_element(nm, query)
            del nm
            gc.collect()
        except Exception:  # noqa: BLE001
            elem = None
        if elem is not None:
            query_argsel = {"query_selector": f"{query['qtype']}({query['table']})",
                            "query_resolved_element": elem}

    scenario = {
        "task": spec["family"],
        "network": netdesc,
        "analysis": spec["analysis"],
        "modifications": structured_mods,
        "query_target": query,
    }
    if resolved_rec is not None or query_extra is not None or query_argsel is not None:
        rt = {}
        if resolved_rec is not None:
            rt.update(resolved_rec)
        if query_extra is not None:
            rt.update(query_extra)
        if query_argsel is not None:
            rt.update(query_argsel)
        scenario["resolved_target"] = rt

    item = {
        "id": "e2pypsa_" + make_id(spec["family"], spec["diff"], code),
        "scenario": scenario,
        "difficulty_level": spec["diff"],
        "natural_language_query": spec["nl"],
        "reference_code": code,
        "ground_truth": gt,
        "ground_truth_type": "float",
        "_audit": {
            "baseline_ground_truth": None if gt_base is None else round(gt_base, 6),
            "sensitivity_delta": None if delta is None else round(delta, 6),
        },
    }
    return item, None


def load_existing():
    if OUT_JSON.exists():
        try:
            return json.loads(OUT_JSON.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # CLI: optional family filter (process one/few families per invocation to bound wall-clock).
    req = [a for a in sys.argv[1:] if not a.startswith("-")]
    target_families = req if req else list(FAMILIES)
    for f in target_families:
        if f not in FAMILIES:
            raise SystemExit(f"unknown family {f!r}; choose from {FAMILIES}")

    # Merge discipline: keep items of families NOT being processed; replace processed families.
    existing = load_existing()
    preserved = [it for it in existing if it["scenario"]["task"] not in target_families]

    specs = candidates()
    from collections import defaultdict

    by_cell = defaultdict(list)
    for s in specs:
        by_cell[(s["family"], s["diff"])].append(s)

    accepted, rejected = [], []
    t0 = time.time()
    for family in target_families:
        for diff in DIFFS:
            cell = by_cell[(family, diff)]
            kept = 0
            for s in cell:
                if kept >= N_PER_CELL:
                    break
                item, err = realize(s)
                if item is None:
                    rejected.append((family, diff, s["builder"], err))
                    print(f"  [reject] {family}/{diff} {s['builder']}: {err}", flush=True)
                    continue
                accepted.append(item)
                kept += 1
                print(f"  [ok {len(accepted):>2}] {family}/{diff} {s['builder']} "
                      f"gt={item['ground_truth']}", flush=True)
                if len(accepted) % 15 == 0:
                    _write(preserved + accepted)
                    print(f"  --- checkpoint written ({len(accepted)} new items) ---", flush=True)
            if kept < N_PER_CELL:
                print(f"  [WARN] {family}/{diff} only {kept}/{N_PER_CELL} accepted", flush=True)

    _write(preserved + accepted)
    dt = time.time() - t0
    _report(preserved + accepted, rejected, dt)


def _write(items):
    # _audit is a namespaced, audit-only field (E1-style provenance); the main benchmark loader
    # ignores unknown keys, so it is harmless to ship.
    OUT_JSON.write_text(json.dumps(items, indent=2))


def _report(accepted, rejected, dt):
    from collections import Counter

    cells = Counter((it["scenario"]["task"], it["difficulty_level"]) for it in accepted)
    print("\n" + "=" * 70)
    print(f"E2 PyPSA mini-bench: {len(accepted)} items in {dt:.1f}s  (rejected {len(rejected)})")
    print("-" * 70)
    print(f"{'family':<22}{'D1_basic':>10}{'D3_semantic':>13}{'total':>8}")
    for fam in FAMILIES:
        d1 = cells.get((fam, "D1_basic"), 0)
        d3 = cells.get((fam, "D3_semantic"), 0)
        print(f"{fam:<22}{d1:>10}{d3:>13}{d1 + d3:>8}")
    print("-" * 70)
    print(f"{'TOTAL':<22}{sum(1 for i in accepted if i['difficulty_level']=='D1_basic'):>10}"
          f"{sum(1 for i in accepted if i['difficulty_level']=='D3_semantic'):>13}{len(accepted):>8}")
    print("=" * 70)
    if rejected:
        print("Rejected candidates (over-generated buffer, expected):")
        for r in rejected:
            print("  ", r)
    print(f"\nWritten: {OUT_JSON}")


if __name__ == "__main__":
    main()
