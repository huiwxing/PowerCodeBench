#!/usr/bin/env python
# --------------------------------------------------------------------------
# Repository copy of scripts/build/build_e2_opendss_bench.py, with imports and
# data-path constants adapted to this repository's layout. Requires the pinned
# environment plus opendssdirect; archived to document the frozen E2
# mini-bench construction. See code/README.md for the module map.
# --------------------------------------------------------------------------
"""Build the E2 OpenDSS mini-bench (benchmark/e2_opendss/e2_opendss_bench.json).

Protocol authority: E2_backend_transfer.md
  - 5 families x {D1_basic, D3_semantic} x ~9 = ~90 items.
  - OpenDSS families (E2 §2.2, no native OPF): power_flow_3ph / network_modification /
    result_extraction / control_device / fault_analysis.
  - Network base: self-built diverse distribution feeders constructed through the OpenDSS
    text-command DSL (different phase counts / voltage levels / component mixes: 3-phase
    radial + unbalanced single-phase laterals + looped feeder + step-down transformer +
    capacitor/CapControl + regulator/RegControl + fault feeder). Construction idioms reuse
    the 76 verified examples in dataset/opendss_docs.json.
  - GT protocol (E2 §2.3): every reference_code is a self-contained script (text commands
    that build the circuit + accessor extraction). GT is produced by running it in an ISOLATED
    subprocess (NOT hardcoded, anti-contamination), passed through the admission filter
    (circuit compiles + >=1 Vsource + >=1 Load + Solution.Converged()==True + control family
    requires a CapControl/RegControl object), and rounded to 6 decimals. Non-convergent items
    are discarded (over-generation absorbs the loss).
  - Query discipline (absorbing the PyPSA-round lessons): the queried scalar must be sensitive
    to the item's modification (baseline vs modified scalar differs > tolerance); deterministic
    only (no RNG); no bool queries.
  - OpenDSS-specific pitfalls handled by construction: (a) every recipe issues
    `Set voltagebases=...` + `CalcVoltageBases` before Solve so per-unit accessors are correct;
    (b) control-family feeders instantiate the CapControl/RegControl object up front (admission);
    (c) fault-family analysis issues `Set mode=faultstudy` before Solve.
  - D3 construction: the semantic target (largest load, longest line, most-loaded line,
    feeder-end/weakest bus, control-device rating/set-point) is resolved against the built /
    solved circuit FIRST, then hardcoded into the reference; the resolved element is recorded in
    scenario.resolved_target for E1-style audit (the NL must not name it).

Schema aligned to the E2 PyPSA mini-bench (benchmark/e2_pypsa/e2_pypsa_bench.json), which in
turn mirrors the main benchmark item:
  id / scenario{task,network,analysis,modifications,query_target,resolved_target?} /
  difficulty_level / natural_language_query / reference_code /
  ground_truth / ground_truth_type / _audit.

Resource discipline: login-node, serial single-thread (OMP/OPENBLAS/MKL=1); per-item GT run in
an isolated subprocess (opendssdirect 0.9.4, CPU-light); per-family invocation + incremental
checkpoint every 15 items. No background processes.

Reproducible / re-runnable: fully deterministic (no RNG); SEED recorded for provenance only.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# --- resource discipline: single-thread for any in-process resolution --------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import opendssdirect as dss  # noqa: E402

SEED = 20260724  # frozen provenance marker (construction is deterministic; no RNG used)
REPO = Path(__file__).resolve().parents[2]  # repository root
OUT_DIR = REPO / "benchmark" / "e2_opendss"
OUT_JSON = OUT_DIR / "e2_opendss_bench.json"
SCRATCH = Path(
    os.environ.get("POWERCODEBENCH_SCRATCH", tempfile.gettempdir())
).expanduser()
SCRATCH.mkdir(parents=True, exist_ok=True)
SUBPROC_TIMEOUT = 120  # s per isolated GT run
SENS_ABS_TOL = 1e-4    # queried scalar must move at least this much under the modification
SENS_REL_TOL = 1e-4

FAMILIES = [
    "power_flow_3ph",
    "network_modification",
    "result_extraction",
    "control_device",
    "fault_analysis",
]
DIFFS = ["D1_basic", "D3_semantic"]
N_PER_CELL = 9  # target items per (family, difficulty)

# ==============================================================================================
# DSL construction helpers (standard distribution overhead impedances, ohm/km).
# ==============================================================================================
_RM3 = "rmatrix=(0.3 | 0.1 0.3 | 0.1 0.1 0.3)"
_XM3 = "xmatrix=(0.7 | 0.3 0.7 | 0.3 0.3 0.7)"
_RM3_LV = "rmatrix=(0.2 | 0.05 0.2 | 0.05 0.05 0.2)"
_XM3_LV = "xmatrix=(0.4 | 0.1 0.4 | 0.1 0.1 0.4)"


def L3(name, b1, b2, length, lv=False):
    rm, xm = (_RM3_LV, _XM3_LV) if lv else (_RM3, _XM3)
    return f"New Line.{name} bus1={b1} bus2={b2} length={length} units=km phases=3 {rm} {xm}"


def L1(name, b1, b2, length):
    return f"New Line.{name} bus1={b1} bus2={b2} length={length} units=km phases=1 r1=0.5 x1=0.9"


def LD3(name, bus, kw, pf=0.92, kv=12.47):
    return f"New Load.{name} bus1={bus} phases=3 kV={kv} kW={kw} pf={pf}"


def LD1(name, busnode, kw, pf=0.95, kv=7.2):
    return f"New Load.{name} bus1={busnode} phases=1 kV={kv} kW={kw} pf={pf}"


# ==============================================================================================
# Network builders -> (recipe_dsl_lines, description).
# recipe_dsl_lines are OpenDSS DSL command strings (first is always 'Clear'); the SAME strings
# are exec'd live (resolution/sensitivity) via dss.Text.Command and embedded verbatim (wrapped
# as dss.Text.Command(...)) into reference_code, so the audited circuit and the executed circuit
# are one source. Recipes stop before Solve (the analysis step solves); each ends with
# voltagebases + CalcVoltageBases (OpenDSS per-unit correctness pitfall).
# ==============================================================================================

def b_radial3():
    return [
        "Clear",
        "New Circuit.src basekv=12.47 phases=3 bus1=src",
        L3("l1", "src", "m1", 3),
        L3("l2", "m1", "endb", 4),
        LD3("lda", "m1", 700),
        LD3("ldb", "endb", 1400),
        "Set voltagebases=[12.47]",
        "CalcVoltageBases",
    ], "self-built 3-bus 12.47 kV three-phase radial feeder (source + 2 loads, 2 line sections)"


def b_radial4():
    return [
        "Clear",
        "New Circuit.src basekv=12.47 phases=3 bus1=src",
        L3("l1", "src", "m1", 3),
        L3("l2", "m1", "m2", 3),
        L3("l3", "m2", "endb", 4),
        LD3("lda", "m1", 600),
        LD3("ldb", "m2", 900),
        LD3("ldc", "endb", 1300),
        "Set voltagebases=[12.47]",
        "CalcVoltageBases",
    ], "self-built 4-bus 12.47 kV three-phase radial feeder (source + 3 loads, 3 line sections)"


def b_radial5():
    return [
        "Clear",
        "New Circuit.src basekv=12.47 phases=3 bus1=src",
        L3("l1", "src", "b1", 2),
        L3("l2", "b1", "b2", 3),
        L3("l3", "b2", "b3", 3),
        L3("l4", "b3", "b4", 4),
        LD3("lda", "b1", 500),
        LD3("ldb", "b2", 700),
        LD3("ldc", "b3", 900),
        LD3("ldd", "b4", 1200),
        "Set voltagebases=[12.47]",
        "CalcVoltageBases",
    ], "self-built 5-bus 12.47 kV three-phase radial feeder (source + 4 loads, 4 line sections)"


def b_unbal3():
    return [
        "Clear",
        "New Circuit.src basekv=12.47 phases=3 bus1=src",
        L3("l0", "src", "j1", 2),
        L1("la", "j1.1", "ea.1", 3),
        L1("lb", "j1.2", "eb.2", 3),
        L1("lc", "j1.3", "ec.3", 4),
        LD1("lda", "ea.1", 400),
        LD1("ldb", "eb.2", 800),
        LD1("ldc", "ec.3", 1200),
        "Set voltagebases=[12.47]",
        "CalcVoltageBases",
    ], ("self-built 12.47 kV three-phase feeder with unbalanced single-phase laterals "
        "(one load per phase, 400/800/1200 kW), yielding per-phase voltage asymmetry")


def b_loop4():
    return [
        "Clear",
        "New Circuit.src basekv=12.47 phases=3 bus1=src",
        L3("l1", "src", "a", 2),
        L3("l2", "a", "b", 3),
        L3("l3", "b", "c", 3),
        L3("ltie", "a", "c", 3),
        LD3("ldb", "b", 1200),
        LD3("ldc", "c", 1000),
        "Set voltagebases=[12.47]",
        "CalcVoltageBases",
    ], ("self-built 12.47 kV three-phase looped feeder (source -> a -> b -> c with an a-c tie "
        "line, 2 loads) so line flows redistribute around the loop")


def b_trafo():
    return [
        "Clear",
        "New Circuit.src basekv=12.47 phases=3 bus1=src",
        L3("l1", "src", "hv", 2),
        ("New Transformer.t1 phases=3 windings=2 buses=[hv, lv] conns=[wye, wye] "
         "kvs=[12.47, 4.16] kvas=[5000, 5000] xhl=6"),
        L3("l2", "lv", "lend", 1, lv=True),
        LD3("ldhv", "hv", 800),
        LD3("ldlv", "lend", 1200, kv=4.16),
        "Set voltagebases=[12.47, 4.16]",
        "CalcVoltageBases",
    ], ("self-built two-voltage-level feeder (12.47 kV source, step-down 12.47/4.16 kV "
        "transformer, 4.16 kV load line, 2 loads)")


def b_cap():
    return [
        "Clear",
        "New Circuit.src basekv=12.47 phases=3 bus1=src",
        L3("l1", "src", "b1", 3),
        L3("l2", "b1", "b2", 4),
        LD3("lda", "b1", 900),
        LD3("ldb", "b2", 1800),
        "New Capacitor.cap1 bus1=b2 phases=3 kvar=600 kv=12.47 numsteps=3 states=[1, 0, 0]",
        ("New CapControl.cc1 capacitor=cap1 element=Line.l2 terminal=1 type=voltage "
         "ptratio=60 onsetting=118 offsetting=126"),
        "Set voltagebases=[12.47]",
        "CalcVoltageBases",
    ], ("self-built 12.47 kV three-phase feeder with a 3-step shunt capacitor bank and a "
        "voltage-mode CapControl at the remote bus (2 loads)")


def b_cap2():
    return [
        "Clear",
        "New Circuit.src basekv=12.47 phases=3 bus1=src",
        L3("l1", "src", "n1", 2),
        L3("l2", "n1", "n2", 3),
        L3("l3", "n2", "n3", 3),
        LD3("lda", "n1", 700),
        LD3("ldb", "n2", 1000),
        LD3("ldc", "n3", 1500),
        "New Capacitor.cap1 bus1=n3 phases=3 kvar=900 kv=12.47 numsteps=3 states=[1, 1, 0]",
        ("New CapControl.cc1 capacitor=cap1 element=Line.l3 terminal=1 type=kvar "
         "ptratio=60 onsetting=200 offsetting=-200"),
        "Set voltagebases=[12.47]",
        "CalcVoltageBases",
    ], ("self-built 12.47 kV three-phase radial feeder with a 3-step shunt capacitor bank and a "
        "kvar-mode CapControl at the feeder end (3 loads)")


def b_reg():
    return [
        "Clear",
        "New Circuit.src basekv=12.47 phases=3 bus1=src",
        ("New Transformer.reg1 phases=3 windings=2 buses=[src, regbus] conns=[wye, wye] "
         "kvs=[12.47, 12.47] kvas=[6000, 6000] xhl=0.5"),
        "New RegControl.creg1 transformer=reg1 winding=2 vreg=122 band=2 ptratio=60",
        L3("l1", "regbus", "m1", 3),
        L3("l2", "m1", "endb", 4),
        LD3("lda", "m1", 900),
        LD3("ldb", "endb", 1600),
        "Set voltagebases=[12.47]",
        "CalcVoltageBases",
    ], ("self-built 12.47 kV three-phase feeder with an in-line voltage regulator (RegControl on "
        "a 12.47/12.47 kV regulator transformer, vreg=122) feeding 2 downstream loads")


def b_reg2():
    return [
        "Clear",
        "New Circuit.src basekv=12.47 phases=3 bus1=src",
        ("New Transformer.reg1 phases=3 windings=2 buses=[src, regbus] conns=[wye, wye] "
         "kvs=[12.47, 12.47] kvas=[8000, 8000] xhl=0.5"),
        "New RegControl.creg1 transformer=reg1 winding=2 vreg=120 band=3 ptratio=60",
        L3("l1", "regbus", "p1", 2),
        L3("l2", "p1", "p2", 3),
        L3("l3", "p2", "p3", 4),
        LD3("lda", "p1", 700),
        LD3("ldb", "p2", 1000),
        LD3("ldc", "p3", 1400),
        "Set voltagebases=[12.47]",
        "CalcVoltageBases",
    ], ("self-built 12.47 kV three-phase feeder with an in-line voltage regulator (RegControl, "
        "vreg=120, band=3) feeding a 3-load radial")


def b_fault():
    return [
        "Clear",
        "New Circuit.src basekv=12.47 phases=3 bus1=src MVAsc3=15000 MVAsc1=12000",
        L3("l1", "src", "b1", 2),
        L3("l2", "b1", "b2", 3),
        L3("l3", "b2", "b3", 3),
        L3("llat", "b1", "lat", 2),
        LD3("lda", "b2", 900),
        LD3("ldb", "b3", 1100),
        LD3("ldc", "lat", 700),
        "Set voltagebases=[12.47]",
        "CalcVoltageBases",
    ], ("self-built 12.47 kV three-phase feeder for fault study (source with defined short-circuit "
        "MVA, main line src->b1->b2->b3 plus a b1->lat lateral, 3 loads)")


def b_fault2():
    return [
        "Clear",
        "New Circuit.src basekv=12.47 phases=3 bus1=src MVAsc3=20000 MVAsc1=16000",
        L3("l1", "src", "n1", 3),
        L3("l2", "n1", "n2", 4),
        L3("l3", "n2", "n3", 5),
        LD3("lda", "n1", 800),
        LD3("ldb", "n2", 1000),
        LD3("ldc", "n3", 1200),
        "Set voltagebases=[12.47]",
        "CalcVoltageBases",
    ], ("self-built 12.47 kV three-phase radial feeder for fault study (source with defined "
        "short-circuit MVA, three series line sections of increasing length, 3 loads)")


BUILDERS = {
    "radial3": b_radial3, "radial4": b_radial4, "radial5": b_radial5, "unbal3": b_unbal3,
    "loop4": b_loop4, "trafo": b_trafo, "cap": b_cap, "cap2": b_cap2, "reg": b_reg,
    "reg2": b_reg2, "fault": b_fault, "fault2": b_fault2,
}

# ==============================================================================================
# Modification renderers -> list of python statement strings (single-source: rendered lines are
# both exec'd live and embedded into reference_code).
# ==============================================================================================

def render_mod(m):
    op = m["op"]
    if op in ("text", "add_line", "add_load", "disable_line"):
        return [f"dss.Text.Command({m['command']!r})"]
    if op == "load_kw":
        return [f"dss.Loads.Name({m['load']!r})", f"dss.Loads.kW({m['value']!r})"]
    if op == "line_length":
        return [f"dss.Lines.Name({m['line']!r})", f"dss.Lines.Length({m['value']!r})"]
    if op == "trafo_tap":
        return [f"dss.Transformers.Name({m['trafo']!r})",
                f"dss.Transformers.Wdg({m['wdg']!r})",
                f"dss.Transformers.Tap({m['value']!r})"]
    if op == "reg_vreg":
        return [f"dss.RegControls.Name({m['reg']!r})", f"dss.RegControls.ForwardVreg({m['value']!r})"]
    if op == "reg_band":
        return [f"dss.RegControls.Name({m['reg']!r})", f"dss.RegControls.ForwardBand({m['value']!r})"]
    if op == "cap_kvar":
        # NB: the Capacitors.kvar() setter is a silent no-op in opendssdirect 0.9.4 (getter works);
        # resize the bank through the text DSL, which does take effect.
        return [f"dss.Text.Command('Edit Capacitor.{m['cap']} kvar={m['value']}')"]
    if op == "cap_addstep":
        return [f"dss.Capacitors.Name({m['cap']!r})"] + ["dss.Capacitors.AddStep()"] * int(m["n"])
    if op == "cap_substep":
        return [f"dss.Capacitors.Name({m['cap']!r})"] + ["dss.Capacitors.SubtractStep()"] * int(m["n"])
    raise ValueError(f"unknown mod op {op}")


# ==============================================================================================
# Query renderers -> list of python statement strings ending in `result = <scalar>`.
# No bool queries; every scalar is a deterministic float.
# ==============================================================================================

def render_query(q):
    qt = q["qtype"]
    if qt == "bus_v":
        ph = int(q.get("phase", 1))
        # index by the phase's position among the bus's present nodes -> correct on both
        # single-phase buses (one node) and three-phase buses.
        return [f"dss.Circuit.SetActiveBus({q['bus']!r})",
                "_nodes = list(dss.Bus.Nodes())",
                "_v = dss.Bus.puVmagAngle()",
                f"result = _v[2 * _nodes.index({ph})]"]
    if qt == "min_node_v":
        return ["result = min(dss.Circuit.AllBusMagPu())"]
    if qt == "supplied_p":
        return ["result = -dss.Circuit.TotalPower()[0]"]
    if qt == "supplied_q":
        return ["result = -dss.Circuit.TotalPower()[1]"]
    if qt == "losses_p":
        return ["result = dss.Circuit.Losses()[0] / 1000.0"]
    if qt == "line_i":
        return [f"dss.Circuit.SetActiveElement('Line.{q['line']}')",
                "_c = dss.CktElement.Currents()",
                "result = (_c[0] ** 2 + _c[1] ** 2) ** 0.5"]
    if qt == "max_line_i":
        return ["_mags = []",
                "_i = dss.Lines.First()",
                "while _i:",
                "    dss.Circuit.SetActiveElement('Line.' + dss.Lines.Name())",
                "    _c = dss.CktElement.Currents()",
                "    _mags.append((_c[0] ** 2 + _c[1] ** 2) ** 0.5)",
                "    _i = dss.Lines.Next()",
                "result = max(_mags)"]
    if qt == "elem_p":
        return [f"dss.Circuit.SetActiveElement({q['elem']!r})",
                "result = dss.CktElement.Powers()[0]"]
    if qt == "isc":
        return [f"dss.Circuit.SetActiveBus({q['bus']!r})",
                "_isc = dss.Bus.Isc()",
                "result = (_isc[0] ** 2 + _isc[1] ** 2) ** 0.5"]
    if qt == "zsc1":
        return [f"dss.Circuit.SetActiveBus({q['bus']!r})",
                "_r, _x = dss.Bus.Zsc1()",
                "result = (_r ** 2 + _x ** 2) ** 0.5"]
    if qt == "tap":
        return [f"dss.RegControls.Name({q['reg']!r})",
                "result = float(dss.RegControls.TapNumber())"]
    raise ValueError(f"unknown qtype {qt}")


HEADER = (
    "import os\n"
    "os.environ['OMP_NUM_THREADS'] = '1'\n"
    "os.environ['OPENBLAS_NUM_THREADS'] = '1'\n"
    "os.environ['MKL_NUM_THREADS'] = '1'\n"
    "import opendssdirect as dss"
)

ANALYSIS = {
    "snap": ["dss.Solution.Solve()",
             "assert dss.Solution.Converged(), 'power flow did not converge'"],
    # open-loop snapshot: the CapControl object stays in the circuit (admission), but automatic
    # control actions are disabled so a MANUAL capacitor switch/resize is deterministic and moves
    # the solution (otherwise the auto-control re-decides the switching state on every solve and
    # cancels the modification). Regulator items keep closed-loop 'snap'.
    "snap_nocontrol": ["dss.Text.Command('Set controlmode=off')",
                       "dss.Solution.Solve()",
                       "assert dss.Solution.Converged(), 'power flow did not converge'"],
    "fault": ["dss.Text.Command('Set mode=faultstudy')",
              "dss.Solution.Solve()",
              "assert dss.Solution.Converged(), 'fault study did not converge'"],
}


def recipe_to_py(recipe):
    return [f"dss.Text.Command({c!r})" for c in recipe]


def assemble_code(recipe, mod_lines, analysis, query):
    parts = [HEADER, "", "# Build circuit (text-command DSL)"]
    parts += recipe_to_py(recipe)
    if mod_lines:
        parts += ["", "# Apply modifications"]
        parts += mod_lines
    parts += ["", "# Run analysis"]
    parts += ANALYSIS[analysis]
    parts += ["", "# Extract result"]
    parts += render_query(query)
    parts += ["print(round(float(result), 6))"]
    return "\n".join(parts) + "\n"


# ==============================================================================================
# In-process runner (live baseline / resolvers / argsel). opendssdirect is a global singleton;
# every recipe starts with 'Clear', so each exec resets engine state -> deterministic.
# ==============================================================================================

def run_body_inproc(recipe, mod_lines, analysis, query=None):
    """Exec recipe(+mods)+analysis(+query) live; return ns (with 'result' if query given).
    Leaves the dss engine in the built/modified/solved state for post-hoc argsel reads."""
    lines = list(recipe_to_py(recipe))
    if mod_lines:
        lines += mod_lines
    lines += ANALYSIS[analysis]
    if query is not None:
        lines += render_query(query)
    ns = {"dss": dss, "min": min, "max": max, "float": float, "abs": abs}
    exec("\n".join(lines), ns)
    return ns


def run_isolated(code):
    """Run reference_code in an isolated subprocess; return (value_or_None, err)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=str(SCRATCH), delete=False) as fh:
        fh.write(code)
        path = fh.name
    try:
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "1"
        proc = subprocess.run([sys.executable, path], capture_output=True, text=True,
                              timeout=SUBPROC_TIMEOUT, env=env)
        if proc.returncode != 0:
            tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "nonzero exit"
            return None, tail
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


# ==============================================================================================
# Admission (structural, in-process): compiled + >=1 Vsource + >=1 Load + control obj if needed.
# ==============================================================================================

def admit(recipe, family):
    try:
        run_body_inproc(recipe, None, "snap")  # build + snap solve
    except Exception as e:  # noqa: BLE001
        return False, f"build/solve failed: {type(e).__name__}: {e}"
    if dss.Vsources.Count() < 1:
        return False, "no Vsource"
    if dss.Loads.Count() < 1:
        return False, "no Load"
    if family == "control_device" and dss.CapControls.Count() < 1 and dss.RegControls.Count() < 1:
        return False, "control family without CapControl/RegControl"
    return True, None


# ==============================================================================================
# Semantic resolvers (D3): resolve target against the built / solved circuit.
# Each returns (element_name, base_value_or_None).
# ==============================================================================================

def _iter_loads():
    out = []
    i = dss.Loads.First()
    while i:
        out.append((dss.Loads.Name(), dss.Loads.kW()))
        i = dss.Loads.Next()
    return out


def _iter_lines():
    out = []
    i = dss.Lines.First()
    while i:
        out.append((dss.Lines.Name(), dss.Lines.Length()))
        i = dss.Lines.Next()
    return out


def resolve(kind, recipe, analysis):
    if kind == "largest_load":
        run_body_inproc(recipe, None, "snap")
        return max(_iter_loads(), key=lambda t: t[1])
    if kind == "longest_line":
        run_body_inproc(recipe, None, "snap")
        return max(_iter_lines(), key=lambda t: t[1])
    if kind == "cap_rating":
        run_body_inproc(recipe, None, "snap")
        dss.Capacitors.First()
        return dss.Capacitors.Name(), dss.Capacitors.kvar()
    if kind == "reg_setpoint":
        run_body_inproc(recipe, None, "snap")
        dss.RegControls.First()
        return dss.RegControls.Name(), dss.RegControls.ForwardVreg()
    if kind in ("most_loaded_line", "most_loaded_nonhead"):
        # nonhead: exclude the substation head line(s) (Bus1 == source bus), so an N-1 outage of
        # the resolved line cannot island the whole feeder (which would give a degenerate 0-current
        # de-energized network).
        run_body_inproc(recipe, None, analysis)
        src_bus = dss.Circuit.AllBusNames()[0]  # source bus (bus1 of New Circuit)
        best, bi = None, -1.0
        i = dss.Lines.First()
        while i:
            nm = dss.Lines.Name()
            dss.Circuit.SetActiveElement("Line." + nm)
            b1 = dss.Lines.Bus1().split(".")[0]
            c = dss.CktElement.Currents()
            mag = (c[0] ** 2 + c[1] ** 2) ** 0.5
            if kind == "most_loaded_nonhead" and b1 == src_bus:
                i = dss.Lines.Next()
                continue
            if mag > bi:
                bi, best = mag, nm
            i = dss.Lines.Next()
        return best, bi
    if kind == "lowest_voltage_bus":
        run_body_inproc(recipe, None, "snap")
        mags = dss.Circuit.AllBusMagPu()
        nodes = dss.Circuit.AllNodeNames()
        j = min(range(len(mags)), key=lambda k: mags[k])
        return nodes[j].split(".")[0], mags[j]
    raise ValueError(f"unknown resolve kind {kind}")


def line_endpoints(recipe, line):
    run_body_inproc(recipe, None, "snap")
    dss.Lines.Name(line)
    return dss.Lines.Bus1(), dss.Lines.Bus2(), dss.Lines.Length(), dss.Lines.Phases()


def argsel(query):
    """Read the argmin/argmax element from the CURRENT (modified, solved) engine state."""
    qt = query["qtype"]
    if qt == "min_node_v":
        mags = dss.Circuit.AllBusMagPu()
        nodes = dss.Circuit.AllNodeNames()
        j = min(range(len(mags)), key=lambda k: mags[k])
        return f"min_node_v", nodes[j]
    if qt == "max_line_i":
        best, bi = None, -1.0
        i = dss.Lines.First()
        while i:
            nm = dss.Lines.Name()
            dss.Circuit.SetActiveElement("Line." + nm)
            c = dss.CktElement.Currents()
            mag = (c[0] ** 2 + c[1] ** 2) ** 0.5
            if mag > bi:
                bi, best = mag, nm
            i = dss.Lines.Next()
        return "max_line_i", best
    return None, None


# ==============================================================================================
# Candidate specification.
# ==============================================================================================

def _mk(family, diff, builder, analysis, mods, query, nl, resolved=None):
    return dict(family=family, diff=diff, builder=builder, analysis=analysis,
                mods=mods, query=query, nl=nl, resolved=resolved)


def candidates():
    C = []

    # ================================ power_flow_3ph (snap) ================================
    # D1: explicit named load change + explicit query element on electrically weak feeders.
    C += [
        _mk("power_flow_3ph", "D1_basic", "radial3", "snap",
            [{"op": "load_kw", "load": "ldb", "value": 2600.0}],
            {"qtype": "bus_v", "bus": "endb"},
            "Set the active power of load 'ldb' to 2600 kW, run the three-phase power flow, and "
            "report the phase-1 per-unit voltage magnitude at the feeder-end bus 'endb'."),
        _mk("power_flow_3ph", "D1_basic", "radial4", "snap",
            [{"op": "load_kw", "load": "ldc", "value": 2400.0}],
            {"qtype": "bus_v", "bus": "endb"},
            "Change load 'ldc' to 2400 kW, solve the three-phase power flow, and report the phase-1 "
            "per-unit voltage magnitude at the feeder-end bus 'endb'."),
        _mk("power_flow_3ph", "D1_basic", "radial5", "snap",
            [{"op": "load_kw", "load": "ldd", "value": 2200.0}],
            {"qtype": "bus_v", "bus": "b4"},
            "Set load 'ldd' to 2200 kW, run the three-phase power flow, and report the phase-1 "
            "per-unit voltage magnitude at the remote bus 'b4'."),
        _mk("power_flow_3ph", "D1_basic", "radial3", "snap",
            [{"op": "load_kw", "load": "ldb", "value": 2400.0}],
            {"qtype": "min_node_v"},
            "Set load 'ldb' to 2400 kW, run the three-phase power flow, and report the lowest "
            "per-unit node voltage magnitude anywhere in the circuit."),
        _mk("power_flow_3ph", "D1_basic", "radial4", "snap",
            [{"op": "load_kw", "load": "ldc", "value": 2000.0}],
            {"qtype": "supplied_p"},
            "Set load 'ldc' to 2000 kW, solve the three-phase power flow, and report the total active "
            "power (kW) supplied into the circuit at the source."),
        _mk("power_flow_3ph", "D1_basic", "unbal3", "snap",
            [{"op": "load_kw", "load": "ldc", "value": 1800.0}],
            {"qtype": "bus_v", "bus": "ec", "phase": 3},
            "Set the single-phase load 'ldc' to 1800 kW, run the three-phase power flow, and report "
            "the phase-3 per-unit voltage magnitude at its bus 'ec'."),
        _mk("power_flow_3ph", "D1_basic", "unbal3", "snap",
            [{"op": "load_kw", "load": "ldb", "value": 1400.0}],
            {"qtype": "bus_v", "bus": "eb", "phase": 2},
            "Set the single-phase load 'ldb' to 1400 kW, solve the three-phase power flow, and report "
            "the phase-2 per-unit voltage magnitude at bus 'eb'."),
        _mk("power_flow_3ph", "D1_basic", "loop4", "snap",
            [{"op": "load_kw", "load": "ldc", "value": 2000.0}],
            {"qtype": "line_i", "line": "l3"},
            "Set load 'ldc' to 2000 kW, run the three-phase power flow, and report the phase-1 current "
            "magnitude (A) on line 'l3'."),
        _mk("power_flow_3ph", "D1_basic", "radial5", "snap",
            [{"op": "load_kw", "load": "ldc", "value": 1800.0}],
            {"qtype": "min_node_v"},
            "Set load 'ldc' to 1800 kW, solve the three-phase power flow, and report the minimum "
            "per-unit node voltage magnitude in the circuit."),
        _mk("power_flow_3ph", "D1_basic", "trafo", "snap",
            [{"op": "load_kw", "load": "ldlv", "value": 2200.0}],
            {"qtype": "bus_v", "bus": "lend"},
            "Set the low-voltage load 'ldlv' to 2200 kW, run the three-phase power flow, and report the "
            "phase-1 per-unit voltage magnitude at the 4.16 kV bus 'lend'."),
        _mk("power_flow_3ph", "D1_basic", "radial4", "snap",
            [{"op": "load_kw", "load": "ldb", "value": 1800.0}],
            {"qtype": "line_i", "line": "l1"},
            "Set load 'ldb' to 1800 kW, run the three-phase power flow, and report the phase-1 current "
            "magnitude (A) on the head line 'l1'."),
        _mk("power_flow_3ph", "D1_basic", "loop4", "snap",
            [{"op": "load_kw", "load": "ldb", "value": 2200.0}],
            {"qtype": "supplied_p"},
            "Set load 'ldb' to 2200 kW, solve the three-phase power flow, and report the total active "
            "power (kW) drawn from the source."),
    ]
    # D3: semantic modification target resolved against the built network (not named in the NL).
    C += [
        _mk("power_flow_3ph", "D3_semantic", "radial4", "snap",
            [{"op": "load_kw", "resolve": "largest_load", "scale": 1.6}],
            {"qtype": "min_node_v"},
            "Take the largest active-power load on the feeder and increase it by 60%, then run the "
            "three-phase power flow and report the lowest per-unit node voltage magnitude in the "
            "circuit.", "largest_load"),
        _mk("power_flow_3ph", "D3_semantic", "radial5", "snap",
            [{"op": "load_kw", "resolve": "largest_load", "scale": 1.7}],
            {"qtype": "min_node_v"},
            "Increase the single biggest load on the feeder by 70%, solve the three-phase power flow, "
            "and report the minimum per-unit node voltage magnitude in the circuit.", "largest_load"),
        _mk("power_flow_3ph", "D3_semantic", "radial3", "snap",
            [{"op": "load_kw", "resolve": "largest_load", "scale": 1.8}],
            {"qtype": "min_node_v"},
            "Grow the heaviest load in the feeder by 80%, run the three-phase power flow, and report "
            "the lowest per-unit node voltage magnitude.", "largest_load"),
        _mk("power_flow_3ph", "D3_semantic", "radial4", "snap",
            [{"op": "load_kw", "resolve": "largest_load", "scale": 1.5}],
            {"qtype": "supplied_p"},
            "Increase the largest load on the feeder by 50%, solve the three-phase power flow, and "
            "report the total active power (kW) supplied at the source.", "largest_load"),
        _mk("power_flow_3ph", "D3_semantic", "loop4", "snap",
            [{"op": "load_kw", "resolve": "largest_load", "scale": 1.5}],
            {"qtype": "max_line_i"},
            "Increase the heaviest load in the looped feeder by 50%, run the three-phase power flow, "
            "and report the largest phase-1 line current magnitude (A) on any line.", "largest_load"),
        _mk("power_flow_3ph", "D3_semantic", "radial5", "snap",
            [{"op": "load_kw", "resolve": "largest_load", "scale": 1.4}],
            {"qtype": "supplied_p"},
            "Grow the biggest load on the feeder by 40%, solve the three-phase power flow, and report "
            "the total active power (kW) drawn from the source.", "largest_load"),
        _mk("power_flow_3ph", "D3_semantic", "loop4", "snap",
            [{"op": "load_kw", "resolve": "largest_load", "scale": 1.7}],
            {"qtype": "max_line_i"},
            "Take the largest load in the loop and increase it by 70%, then run the three-phase power "
            "flow and report the maximum phase-1 line current magnitude (A) across all lines.",
            "largest_load"),
        _mk("power_flow_3ph", "D3_semantic", "unbal3", "snap",
            [{"op": "load_kw", "resolve": "largest_load", "scale": 1.5}],
            {"qtype": "min_node_v"},
            "Increase the biggest single-phase load on the feeder by 50%, solve the three-phase power "
            "flow, and report the lowest per-unit node voltage magnitude in the circuit.",
            "largest_load"),
        _mk("power_flow_3ph", "D3_semantic", "radial4", "snap",
            [{"op": "load_kw", "resolve": "largest_load", "scale": 1.9}],
            {"qtype": "min_node_v"},
            "Grow the largest load in the feeder by 90%, run the three-phase power flow, and report "
            "the minimum per-unit node voltage magnitude.", "largest_load"),
        _mk("power_flow_3ph", "D3_semantic", "trafo", "snap",
            [{"op": "load_kw", "resolve": "largest_load", "scale": 1.6}],
            {"qtype": "min_node_v"},
            "Increase the largest load in the network by 60%, solve the three-phase power flow, and "
            "report the lowest per-unit node voltage magnitude anywhere in the circuit.",
            "largest_load"),
        _mk("power_flow_3ph", "D3_semantic", "radial5", "snap",
            [{"op": "load_kw", "resolve": "largest_load", "scale": 1.6}],
            {"qtype": "max_line_i"},
            "Increase the heaviest load on the feeder by 60%, run the three-phase power flow, and "
            "report the largest phase-1 line current magnitude (A) on any line.", "largest_load"),
    ]

    # ================================ network_modification (snap) ================================
    # D1: explicit topology / parameter edits (add line, disable line, edit length, transformer tap).
    C += [
        _mk("network_modification", "D1_basic", "loop4", "snap",
            [{"op": "disable_line", "command": "Disable Line.ltie"}],
            {"qtype": "line_i", "line": "l3"},
            "Take the tie line 'ltie' out of service (disable it), run the three-phase power flow, and "
            "report the phase-1 current magnitude (A) on line 'l3'."),
        _mk("network_modification", "D1_basic", "loop4", "snap",
            [{"op": "add_line",
              "command": "New Line.lnew bus1=b bus2=src length=5 units=km phases=3 "
                         + _RM3 + " " + _XM3}],
            {"qtype": "line_i", "line": "l1"},
            "Add a new tie line 'lnew' from bus 'b' back to the source bus 'src' (5 km, three-phase), "
            "run the power flow, and report the phase-1 current magnitude (A) on line 'l1'."),
        _mk("network_modification", "D1_basic", "radial4", "snap",
            [{"op": "line_length", "line": "l1", "value": 8.0}],
            {"qtype": "min_node_v"},
            "Change the length of head line 'l1' to 8 km, run the three-phase power flow, and report "
            "the lowest per-unit node voltage magnitude in the circuit."),
        _mk("network_modification", "D1_basic", "radial3", "snap",
            [{"op": "add_load",
              "command": "New Load.ldx bus1=endb phases=3 kV=12.47 kW=800 pf=0.9"}],
            {"qtype": "supplied_p"},
            "Add a new three-phase load 'ldx' of 800 kW at bus 'endb', solve the three-phase power "
            "flow, and report the total active power (kW) supplied at the source."),
        _mk("network_modification", "D1_basic", "trafo", "snap",
            [{"op": "trafo_tap", "trafo": "t1", "wdg": 2, "value": 1.05}],
            {"qtype": "bus_v", "bus": "lend"},
            "Set the tap of the low-voltage winding (winding 2) of transformer 't1' to 1.05, run the "
            "three-phase power flow, and report the phase-1 per-unit voltage magnitude at bus 'lend'."),
        _mk("network_modification", "D1_basic", "radial5", "snap",
            [{"op": "line_length", "line": "l4", "value": 8.0}],
            {"qtype": "bus_v", "bus": "b4"},
            "Change the length of line 'l4' to 8 km, run the three-phase power flow, and report the "
            "phase-1 per-unit voltage magnitude at the feeder-end bus 'b4'."),
        _mk("network_modification", "D1_basic", "loop4", "snap",
            [{"op": "line_length", "line": "ltie", "value": 0.5}],
            {"qtype": "line_i", "line": "l3"},
            "Shorten the tie line 'ltie' to 0.5 km, run the three-phase power flow, and report the "
            "phase-1 current magnitude (A) on line 'l3'."),
        _mk("network_modification", "D1_basic", "radial4", "snap",
            [{"op": "add_line",
              "command": "New Line.lbypass bus1=m1 bus2=endb length=6 units=km phases=3 "
                         + _RM3 + " " + _XM3}],
            {"qtype": "bus_v", "bus": "endb"},
            "Add a bypass line 'lbypass' directly from bus 'm1' to bus 'endb' (6 km, three-phase), "
            "run the power flow, and report the phase-1 per-unit voltage magnitude at bus 'endb'."),
        _mk("network_modification", "D1_basic", "radial3", "snap",
            [{"op": "line_length", "line": "l2", "value": 9.0}],
            {"qtype": "min_node_v"},
            "Set the length of line 'l2' to 9 km, solve the three-phase power flow, and report the "
            "lowest per-unit node voltage magnitude in the circuit."),
        _mk("network_modification", "D1_basic", "trafo", "snap",
            [{"op": "add_load",
              "command": "New Load.ldx bus1=lend phases=3 kV=4.16 kW=900 pf=0.9"}],
            {"qtype": "supplied_p"},
            "Add a new 900 kW three-phase load 'ldx' at the 4.16 kV bus 'lend', run the power flow, "
            "and report the total active power (kW) supplied at the source."),
        _mk("network_modification", "D1_basic", "radial5", "snap",
            [{"op": "add_line",
              "command": "New Line.ltie2 bus1=b2 bus2=b4 length=6 units=km phases=3 "
                         + _RM3 + " " + _XM3}],
            {"qtype": "bus_v", "bus": "b4"},
            "Add a tie line 'ltie2' between bus 'b2' and bus 'b4' (6 km, three-phase), run the power "
            "flow, and report the phase-1 per-unit voltage magnitude at bus 'b4'."),
        _mk("network_modification", "D1_basic", "trafo", "snap",
            [{"op": "trafo_tap", "trafo": "t1", "wdg": 2, "value": 0.95}],
            {"qtype": "min_node_v"},
            "Set the tap of winding 2 of transformer 't1' to 0.95, solve the three-phase power flow, "
            "and report the lowest per-unit node voltage magnitude in the circuit."),
    ]
    # D3: semantic target (most-loaded / longest line) resolved, not named.
    C += [
        _mk("network_modification", "D3_semantic", "loop4", "snap",
            [{"op": "add_parallel_line", "resolve": "most_loaded_line", "name": "lreinf"}],
            {"qtype": "max_line_i"},
            "Reinforce the network by adding a parallel circuit (identical parameters) alongside the "
            "most heavily loaded line, then run the three-phase power flow and report the largest "
            "phase-1 line current magnitude (A) on any line.", "most_loaded_line"),
        _mk("network_modification", "D3_semantic", "radial5", "snap",
            [{"op": "add_parallel_line", "resolve": "most_loaded_line", "name": "lreinf"}],
            {"qtype": "min_node_v"},
            "Add a parallel circuit to the most heavily loaded line to strengthen the feeder, then "
            "solve the three-phase power flow and report the lowest per-unit node voltage magnitude.",
            "most_loaded_line"),
        _mk("network_modification", "D3_semantic", "loop4", "snap",
            [{"op": "disable_line", "resolve": "most_loaded_nonhead"}],
            {"qtype": "max_line_i"},
            "Simulate an outage of the most heavily loaded feeder line downstream of the substation "
            "by taking it out of service, then run the three-phase power flow and report the largest "
            "phase-1 line current magnitude (A) among the remaining lines.", "most_loaded_nonhead"),
        _mk("network_modification", "D3_semantic", "radial4", "snap",
            [{"op": "line_length", "resolve": "longest_line", "scale": 0.5}],
            {"qtype": "min_node_v"},
            "Reinforce the feeder by halving the length of its longest line section, then solve the "
            "three-phase power flow and report the lowest per-unit node voltage magnitude.",
            "longest_line"),
        _mk("network_modification", "D3_semantic", "radial5", "snap",
            [{"op": "line_length", "resolve": "longest_line", "scale": 0.5}],
            {"qtype": "bus_v", "bus": "b4"},
            "Halve the length of the longest line section on the feeder, then run the three-phase "
            "power flow and report the phase-1 per-unit voltage magnitude at the feeder-end bus 'b4'.",
            "longest_line"),
        _mk("network_modification", "D3_semantic", "radial4", "snap",
            [{"op": "line_length", "resolve": "longest_line", "scale": 2.0}],
            {"qtype": "min_node_v"},
            "Double the length of the longest line section on the feeder, solve the three-phase power "
            "flow, and report the lowest per-unit node voltage magnitude in the circuit.",
            "longest_line"),
        _mk("network_modification", "D3_semantic", "loop4", "snap",
            [{"op": "add_parallel_line", "resolve": "longest_line", "name": "lreinf"}],
            {"qtype": "max_line_i"},
            "Add a parallel circuit alongside the longest line in the loop, then run the three-phase "
            "power flow and report the largest phase-1 line current magnitude (A) on any line.",
            "longest_line"),
        _mk("network_modification", "D3_semantic", "radial5", "snap",
            [{"op": "line_length", "resolve": "longest_line", "scale": 2.0}],
            {"qtype": "min_node_v"},
            "Double the length of the feeder's longest line, solve the three-phase power flow, and "
            "report the minimum per-unit node voltage magnitude.", "longest_line"),
        _mk("network_modification", "D3_semantic", "radial4", "snap",
            [{"op": "add_parallel_line", "resolve": "most_loaded_line", "name": "lreinf"}],
            {"qtype": "min_node_v"},
            "Strengthen the feeder by adding a parallel circuit to its most heavily loaded line, then "
            "run the three-phase power flow and report the lowest per-unit node voltage magnitude.",
            "most_loaded_line"),
        _mk("network_modification", "D3_semantic", "radial3", "snap",
            [{"op": "line_length", "resolve": "longest_line", "scale": 0.5}],
            {"qtype": "bus_v", "bus": "endb"},
            "Halve the length of the longest line section of the feeder, run the three-phase power "
            "flow, and report the phase-1 per-unit voltage magnitude at the feeder-end bus 'endb'.",
            "longest_line"),
        _mk("network_modification", "D3_semantic", "radial5", "snap",
            [{"op": "add_parallel_line", "resolve": "longest_line", "name": "lreinf"}],
            {"qtype": "min_node_v"},
            "Add a parallel circuit alongside the longest line section of the feeder, then solve the "
            "three-phase power flow and report the lowest per-unit node voltage magnitude.",
            "longest_line"),
    ]

    # ================================ result_extraction (snap) ================================
    # D1: simple named load change; the point is the specific accessor extraction path.
    C += [
        _mk("result_extraction", "D1_basic", "radial4", "snap",
            [{"op": "load_kw", "load": "ldc", "value": 1800.0}],
            {"qtype": "elem_p", "elem": "Line.l1"},
            "Set load 'ldc' to 1800 kW, run the three-phase power flow, and read the sending-end "
            "phase-1 active power (kW) flowing into line 'l1' from the element power results."),
        _mk("result_extraction", "D1_basic", "loop4", "snap",
            [{"op": "load_kw", "load": "ldc", "value": 1600.0}],
            {"qtype": "elem_p", "elem": "Line.l3"},
            "Set load 'ldc' to 1600 kW, solve the three-phase power flow, and read the sending-end "
            "phase-1 active power (kW) into line 'l3' from the element power results."),
        _mk("result_extraction", "D1_basic", "radial4", "snap",
            [{"op": "load_kw", "load": "ldb", "value": 1600.0}],
            {"qtype": "losses_p"},
            "Set load 'ldb' to 1600 kW, run the three-phase power flow, and report the total active "
            "power loss (kW) of the circuit."),
        _mk("result_extraction", "D1_basic", "unbal3", "snap",
            [{"op": "load_kw", "load": "ldc", "value": 1600.0}],
            {"qtype": "bus_v", "bus": "ec", "phase": 3},
            "Set the single-phase load 'ldc' to 1600 kW, run the three-phase power flow, and read the "
            "phase-3 per-unit voltage magnitude at bus 'ec' from the bus voltage results."),
        _mk("result_extraction", "D1_basic", "radial5", "snap",
            [{"op": "load_kw", "load": "ldd", "value": 1800.0}],
            {"qtype": "line_i", "line": "l4"},
            "Set load 'ldd' to 1800 kW, solve the three-phase power flow, and read the phase-1 current "
            "magnitude (A) on line 'l4' from the element current results."),
        _mk("result_extraction", "D1_basic", "loop4", "snap",
            [{"op": "load_kw", "load": "ldb", "value": 1800.0}],
            {"qtype": "supplied_q"},
            "Set load 'ldb' to 1800 kW, run the three-phase power flow, and read the total reactive "
            "power (kvar) supplied into the circuit at the source."),
        _mk("result_extraction", "D1_basic", "trafo", "snap",
            [{"op": "load_kw", "load": "ldlv", "value": 1800.0}],
            {"qtype": "elem_p", "elem": "Transformer.t1"},
            "Set the low-voltage load 'ldlv' to 1800 kW, solve the three-phase power flow, and read "
            "the phase-1 active power (kW) flowing into the primary terminal of transformer 't1'."),
        _mk("result_extraction", "D1_basic", "radial3", "snap",
            [{"op": "load_kw", "load": "ldb", "value": 2000.0}],
            {"qtype": "losses_p"},
            "Set load 'ldb' to 2000 kW, run the three-phase power flow, and report the total active "
            "power loss (kW) of the circuit."),
        _mk("result_extraction", "D1_basic", "radial5", "snap",
            [{"op": "load_kw", "load": "ldc", "value": 1400.0}],
            {"qtype": "elem_p", "elem": "Line.l2"},
            "Set load 'ldc' to 1400 kW, run the three-phase power flow, and read the sending-end "
            "phase-1 active power (kW) into line 'l2' from the element power results."),
        _mk("result_extraction", "D1_basic", "radial4", "snap",
            [{"op": "load_kw", "load": "ldc", "value": 1600.0}],
            {"qtype": "bus_v", "bus": "endb"},
            "Set load 'ldc' to 1600 kW, solve the three-phase power flow, and read the phase-1 "
            "per-unit voltage magnitude at bus 'endb' from the bus voltage results."),
        _mk("result_extraction", "D1_basic", "loop4", "snap",
            [{"op": "load_kw", "load": "ldc", "value": 1400.0}],
            {"qtype": "line_i", "line": "ltie"},
            "Set load 'ldc' to 1400 kW, run the three-phase power flow, and read the phase-1 current "
            "magnitude (A) on the tie line 'ltie' from the element current results."),
    ]
    # D3: extract a semantically-specified scalar (lowest voltage / most-loaded line).
    C += [
        _mk("result_extraction", "D3_semantic", "radial4", "snap",
            [{"op": "load_kw", "load": "ldc", "value": 1800.0}],
            {"qtype": "min_node_v"},
            "Set load 'ldc' to 1800 kW, run the three-phase power flow, and report the per-unit "
            "voltage magnitude at the weakest (lowest-voltage) node in the circuit."),
        _mk("result_extraction", "D3_semantic", "radial5", "snap",
            [{"op": "load_kw", "load": "ldd", "value": 1600.0}],
            {"qtype": "min_node_v"},
            "Set load 'ldd' to 1600 kW, solve the three-phase power flow, and report the per-unit "
            "voltage magnitude at the lowest-voltage node in the circuit."),
        _mk("result_extraction", "D3_semantic", "loop4", "snap",
            [{"op": "load_kw", "load": "ldc", "value": 1800.0}],
            {"qtype": "max_line_i"},
            "Set load 'ldc' to 1800 kW, run the three-phase power flow, and report the phase-1 current "
            "magnitude (A) on the most heavily loaded line in the network."),
        _mk("result_extraction", "D3_semantic", "radial4", "snap",
            [{"op": "load_kw", "load": "ldc", "value": 2000.0}],
            {"qtype": "max_line_i"},
            "Set load 'ldc' to 2000 kW, solve the three-phase power flow, and report the phase-1 "
            "current magnitude (A) on the busiest line of the feeder."),
        _mk("result_extraction", "D3_semantic", "radial5", "snap",
            [{"op": "load_kw", "load": "ldc", "value": 1400.0}],
            {"qtype": "max_line_i"},
            "Set load 'ldc' to 1400 kW, run the three-phase power flow, and report the phase-1 current "
            "magnitude (A) on the most heavily loaded line in the circuit."),
        _mk("result_extraction", "D3_semantic", "unbal3", "snap",
            [{"op": "load_kw", "load": "ldc", "value": 1600.0}],
            {"qtype": "min_node_v"},
            "Set the single-phase load 'ldc' to 1600 kW, solve the three-phase power flow, and report "
            "the per-unit voltage magnitude at the weakest node in the circuit."),
        _mk("result_extraction", "D3_semantic", "trafo", "snap",
            [{"op": "load_kw", "load": "ldlv", "value": 2000.0}],
            {"qtype": "min_node_v"},
            "Set the low-voltage load 'ldlv' to 2000 kW, run the three-phase power flow, and report "
            "the per-unit voltage magnitude at the lowest-voltage node in the circuit."),
        _mk("result_extraction", "D3_semantic", "radial3", "snap",
            [{"op": "load_kw", "load": "ldb", "value": 2200.0}],
            {"qtype": "max_line_i"},
            "Set load 'ldb' to 2200 kW, solve the three-phase power flow, and report the phase-1 "
            "current magnitude (A) on the most heavily loaded line in the feeder."),
        _mk("result_extraction", "D3_semantic", "loop4", "snap",
            [{"op": "load_kw", "load": "ldb", "value": 2000.0}],
            {"qtype": "min_node_v"},
            "Set load 'ldb' to 2000 kW, run the three-phase power flow, and report the per-unit "
            "voltage magnitude at the weakest node in the looped feeder."),
        _mk("result_extraction", "D3_semantic", "radial5", "snap",
            [{"op": "load_kw", "load": "ldb", "value": 1600.0}],
            {"qtype": "min_node_v"},
            "Set load 'ldb' to 1600 kW, solve the three-phase power flow, and report the per-unit "
            "voltage magnitude at the lowest-voltage node in the circuit."),
        _mk("result_extraction", "D3_semantic", "radial4", "snap",
            [{"op": "load_kw", "load": "ldb", "value": 2000.0}],
            {"qtype": "max_line_i"},
            "Set load 'ldb' to 2000 kW, run the three-phase power flow, and report the phase-1 current "
            "magnitude (A) on the busiest line of the feeder."),
    ]

    # ================================ control_device (snap) ================================
    # D1: explicit control-device operations (capacitor bank sizing / switching, regulator set-point).
    C += [
        _mk("control_device", "D1_basic", "cap", "snap_nocontrol",
            [{"op": "cap_kvar", "cap": "cap1", "value": 1500.0}],
            {"qtype": "supplied_q"},
            "Set the total rating of the switched shunt capacitor bank 'cap1' to 1500 kvar, re-run the "
            "three-phase power flow, and report the total reactive power (kvar) supplied at the source."),
        _mk("control_device", "D1_basic", "cap", "snap_nocontrol",
            [{"op": "cap_addstep", "cap": "cap1", "n": 2}],
            {"qtype": "min_node_v"},
            "Switch in two additional steps of the capacitor bank 'cap1', re-run the three-phase power "
            "flow, and report the lowest per-unit node voltage magnitude in the circuit."),
        _mk("control_device", "D1_basic", "reg", "snap",
            [{"op": "reg_vreg", "reg": "creg1", "value": 126.0}],
            {"qtype": "tap", "reg": "creg1"},
            "Raise the forward voltage set-point of the regulator control 'creg1' to 126 V, re-solve "
            "the three-phase power flow, and report the settled integer tap position of the regulator."),
        _mk("control_device", "D1_basic", "reg", "snap",
            [{"op": "reg_vreg", "reg": "creg1", "value": 125.0}],
            {"qtype": "bus_v", "bus": "regbus"},
            "Set the forward voltage set-point of regulator control 'creg1' to 125 V, re-run the power "
            "flow, and report the phase-1 per-unit voltage magnitude at the regulated bus 'regbus'."),
        _mk("control_device", "D1_basic", "cap2", "snap_nocontrol",
            [{"op": "cap_kvar", "cap": "cap1", "value": 1800.0}],
            {"qtype": "min_node_v"},
            "Set the total rating of the shunt capacitor bank 'cap1' to 1800 kvar, re-run the "
            "three-phase power flow, and report the lowest per-unit node voltage magnitude."),
        _mk("control_device", "D1_basic", "reg2", "snap",
            [{"op": "reg_vreg", "reg": "creg1", "value": 125.0}],
            {"qtype": "tap", "reg": "creg1"},
            "Raise the forward voltage set-point of regulator control 'creg1' to 125 V, re-solve the "
            "power flow, and report the settled integer tap position of the regulator."),
        _mk("control_device", "D1_basic", "cap", "snap_nocontrol",
            [{"op": "cap_kvar", "cap": "cap1", "value": 300.0}],
            {"qtype": "supplied_q"},
            "Reduce the total rating of capacitor bank 'cap1' to 300 kvar, re-run the three-phase "
            "power flow, and report the total reactive power (kvar) supplied at the source."),
        _mk("control_device", "D1_basic", "reg", "snap",
            [{"op": "reg_band", "reg": "creg1", "value": 1.0}],
            {"qtype": "bus_v", "bus": "endb"},
            "Tighten the regulation bandwidth of regulator control 'creg1' to 1 V, re-run the power "
            "flow, and report the phase-1 per-unit voltage magnitude at bus 'endb'."),
        _mk("control_device", "D1_basic", "cap2", "snap_nocontrol",
            [{"op": "cap_addstep", "cap": "cap1", "n": 1}],
            {"qtype": "supplied_q"},
            "Switch in one more step of the capacitor bank 'cap1', re-run the three-phase power flow, "
            "and report the total reactive power (kvar) supplied at the source."),
        _mk("control_device", "D1_basic", "reg2", "snap",
            [{"op": "reg_vreg", "reg": "creg1", "value": 124.0}],
            {"qtype": "bus_v", "bus": "regbus"},
            "Set the forward voltage set-point of regulator control 'creg1' to 124 V, re-solve the "
            "power flow, and report the phase-1 per-unit voltage magnitude at the regulated bus "
            "'regbus'."),
        _mk("control_device", "D1_basic", "cap", "snap_nocontrol",
            [{"op": "cap_kvar", "cap": "cap1", "value": 1200.0}],
            {"qtype": "min_node_v"},
            "Set the total rating of capacitor bank 'cap1' to 1200 kvar, re-run the three-phase power "
            "flow, and report the lowest per-unit node voltage magnitude in the circuit."),
        _mk("control_device", "D1_basic", "cap2", "snap_nocontrol",
            [{"op": "cap_kvar", "cap": "cap1", "value": 300.0}],
            {"qtype": "supplied_q"},
            "Reduce the shunt capacitor bank 'cap1' rating to 300 kvar, re-run the three-phase power "
            "flow, and report the total reactive power (kvar) supplied at the source."),
    ]
    # D3: semantic control target (the bank's current rating / the regulator's current set-point).
    C += [
        _mk("control_device", "D3_semantic", "cap", "snap_nocontrol",
            [{"op": "cap_kvar", "resolve": "cap_rating", "scale": 2.0}],
            {"qtype": "supplied_q"},
            "Double the reactive rating of the feeder's switched shunt capacitor bank, re-run the "
            "three-phase power flow, and report the total reactive power (kvar) supplied at the "
            "source.", "cap_rating"),
        _mk("control_device", "D3_semantic", "cap", "snap_nocontrol",
            [{"op": "cap_kvar", "resolve": "cap_rating", "scale": 2.5}],
            {"qtype": "min_node_v"},
            "Increase the capacitor bank's reactive rating to 2.5 times its present value, re-solve "
            "the three-phase power flow, and report the lowest per-unit node voltage magnitude.",
            "cap_rating"),
        _mk("control_device", "D3_semantic", "reg", "snap",
            [{"op": "reg_vreg", "resolve": "reg_setpoint", "delta": 4.0}],
            {"qtype": "tap", "reg": "creg1"},
            "Raise the voltage regulator's forward set-point by 4 V above its current value, re-solve "
            "the three-phase power flow, and report the settled integer tap position of the "
            "regulator.", "reg_setpoint"),
        _mk("control_device", "D3_semantic", "reg", "snap",
            [{"op": "reg_vreg", "resolve": "reg_setpoint", "delta": 3.0}],
            {"qtype": "min_node_v"},
            "Increase the voltage regulator's forward set-point by 3 V, re-run the three-phase power "
            "flow, and report the lowest per-unit node voltage magnitude in the circuit.",
            "reg_setpoint"),
        _mk("control_device", "D3_semantic", "cap2", "snap_nocontrol",
            [{"op": "cap_kvar", "resolve": "cap_rating", "scale": 1.8}],
            {"qtype": "min_node_v"},
            "Increase the switched shunt capacitor bank's rating to 1.8 times its present value, "
            "re-run the three-phase power flow, and report the minimum per-unit node voltage "
            "magnitude.", "cap_rating"),
        _mk("control_device", "D3_semantic", "reg2", "snap",
            [{"op": "reg_vreg", "resolve": "reg_setpoint", "delta": 5.0}],
            {"qtype": "tap", "reg": "creg1"},
            "Raise the voltage regulator's forward set-point by 5 V above its present value, re-solve "
            "the power flow, and report the settled integer tap position of the regulator.",
            "reg_setpoint"),
        _mk("control_device", "D3_semantic", "cap", "snap_nocontrol",
            [{"op": "cap_kvar", "resolve": "cap_rating", "scale": 0.5}],
            {"qtype": "supplied_q"},
            "Halve the reactive rating of the feeder's switched shunt capacitor bank, re-run the "
            "three-phase power flow, and report the total reactive power (kvar) supplied at the "
            "source.", "cap_rating"),
        _mk("control_device", "D3_semantic", "reg2", "snap",
            [{"op": "reg_vreg", "resolve": "reg_setpoint", "delta": 4.0}],
            {"qtype": "min_node_v"},
            "Increase the voltage regulator's forward set-point by 4 V, solve the three-phase power "
            "flow, and report the lowest per-unit node voltage magnitude in the circuit.",
            "reg_setpoint"),
        _mk("control_device", "D3_semantic", "cap2", "snap_nocontrol",
            [{"op": "cap_kvar", "resolve": "cap_rating", "scale": 2.2}],
            {"qtype": "supplied_q"},
            "Increase the switched shunt capacitor bank's rating to 2.2 times its present value, "
            "re-run the three-phase power flow, and report the total reactive power (kvar) supplied "
            "at the source.", "cap_rating"),
        _mk("control_device", "D3_semantic", "reg", "snap",
            [{"op": "reg_vreg", "resolve": "reg_setpoint", "delta": 5.0}],
            {"qtype": "bus_v", "bus": "regbus"},
            "Raise the voltage regulator's forward set-point by 5 V, re-solve the power flow, and "
            "report the phase-1 per-unit voltage magnitude at the regulated bus 'regbus'.",
            "reg_setpoint"),
        _mk("control_device", "D3_semantic", "cap", "snap_nocontrol",
            [{"op": "cap_kvar", "resolve": "cap_rating", "scale": 3.0}],
            {"qtype": "min_node_v"},
            "Triple the reactive rating of the feeder's capacitor bank, re-run the three-phase power "
            "flow, and report the minimum per-unit node voltage magnitude in the circuit.",
            "cap_rating"),
        _mk("control_device", "D3_semantic", "cap2", "snap_nocontrol",
            [{"op": "cap_kvar", "resolve": "cap_rating", "scale": 0.5}],
            {"qtype": "supplied_q"},
            "Halve the switched shunt capacitor bank's reactive rating, re-run the three-phase power "
            "flow, and report the total reactive power (kvar) supplied at the source.", "cap_rating"),
    ]

    # ================================ fault_analysis (fault) ================================
    # D1: explicit named modification (line length / source strength / added path) + named fault bus.
    C += [
        _mk("fault_analysis", "D1_basic", "fault", "fault",
            [{"op": "line_length", "line": "l2", "value": 8.0}],
            {"qtype": "isc", "bus": "b2"},
            "Change the length of line 'l2' to 8 km, run a fault study, and report the phase-1 "
            "short-circuit current magnitude (A) at bus 'b2'."),
        _mk("fault_analysis", "D1_basic", "fault2", "fault",
            [{"op": "line_length", "line": "l3", "value": 10.0}],
            {"qtype": "isc", "bus": "n3"},
            "Set the length of line 'l3' to 10 km, run a fault study, and report the phase-1 "
            "short-circuit current magnitude (A) at the feeder-end bus 'n3'."),
        _mk("fault_analysis", "D1_basic", "fault", "fault",
            [{"op": "line_length", "line": "l2", "value": 6.0}],
            {"qtype": "zsc1", "bus": "b2"},
            "Change the length of line 'l2' to 6 km, run a fault study, and report the magnitude "
            "(ohm) of the positive-sequence short-circuit (Thevenin) impedance at bus 'b2'."),
        _mk("fault_analysis", "D1_basic", "fault2", "fault",
            [{"op": "line_length", "line": "l2", "value": 8.0}],
            {"qtype": "isc", "bus": "n2"},
            "Set the length of line 'l2' to 8 km, run a fault study, and report the phase-1 "
            "short-circuit current magnitude (A) at bus 'n2'."),
        _mk("fault_analysis", "D1_basic", "fault", "fault",
            [{"op": "text",
              "command": "New Line.lpar bus1=src bus2=b1 length=2 units=km phases=3 "
                         + _RM3 + " " + _XM3}],
            {"qtype": "isc", "bus": "b1"},
            "Add a parallel line 'lpar' between the source 'src' and bus 'b1' (2 km, three-phase), run "
            "a fault study, and report the phase-1 short-circuit current magnitude (A) at bus 'b1'."),
        _mk("fault_analysis", "D1_basic", "fault2", "fault",
            [{"op": "line_length", "line": "l1", "value": 6.0}],
            {"qtype": "zsc1", "bus": "n1"},
            "Change the length of head line 'l1' to 6 km, run a fault study, and report the magnitude "
            "(ohm) of the positive-sequence Thevenin impedance at bus 'n1'."),
        _mk("fault_analysis", "D1_basic", "fault", "fault",
            [{"op": "line_length", "line": "l3", "value": 7.0}],
            {"qtype": "isc", "bus": "b3"},
            "Set the length of line 'l3' to 7 km, run a fault study, and report the phase-1 "
            "short-circuit current magnitude (A) at the feeder-end bus 'b3'."),
        _mk("fault_analysis", "D1_basic", "fault2", "fault",
            [{"op": "text",
              "command": "New Line.lpar bus1=src bus2=n1 length=3 units=km phases=3 "
                         + _RM3 + " " + _XM3}],
            {"qtype": "isc", "bus": "n1"},
            "Add a parallel line 'lpar' between the source 'src' and bus 'n1' (3 km, three-phase), run "
            "a fault study, and report the phase-1 short-circuit current magnitude (A) at bus 'n1'."),
        _mk("fault_analysis", "D1_basic", "fault", "fault",
            [{"op": "line_length", "line": "llat", "value": 6.0}],
            {"qtype": "isc", "bus": "lat"},
            "Change the length of the lateral line 'llat' to 6 km, run a fault study, and report the "
            "phase-1 short-circuit current magnitude (A) at the lateral bus 'lat'."),
        _mk("fault_analysis", "D1_basic", "fault2", "fault",
            [{"op": "line_length", "line": "l3", "value": 8.0}],
            {"qtype": "zsc1", "bus": "n3"},
            "Set the length of line 'l3' to 8 km, run a fault study, and report the magnitude (ohm) of "
            "the positive-sequence Thevenin impedance at the feeder-end bus 'n3'."),
        _mk("fault_analysis", "D1_basic", "fault", "fault",
            [{"op": "line_length", "line": "l1", "value": 5.0}],
            {"qtype": "isc", "bus": "b1"},
            "Change the length of head line 'l1' to 5 km, run a fault study, and report the phase-1 "
            "short-circuit current magnitude (A) at bus 'b1'."),
    ]
    # D3: semantic modification target (longest line) resolved; feeder-end fault bus resolved.
    C += [
        _mk("fault_analysis", "D3_semantic", "fault2", "fault",
            [{"op": "line_length", "resolve": "longest_line", "scale": 1.6}],
            {"qtype": "isc", "resolve": "lowest_voltage_bus"},
            "Extend the longest line section of the feeder by 60%, then run a fault study and report "
            "the phase-1 short-circuit current magnitude (A) at the electrically weakest (feeder-end) "
            "bus.", "longest_line+lowest_voltage_bus"),
        _mk("fault_analysis", "D3_semantic", "fault2", "fault",
            [{"op": "line_length", "resolve": "longest_line", "scale": 2.0}],
            {"qtype": "isc", "resolve": "lowest_voltage_bus"},
            "Double the length of the longest feeder section, then run a fault study and report the "
            "phase-1 short-circuit current magnitude (A) at the weakest (feeder-end) bus.",
            "longest_line+lowest_voltage_bus"),
        _mk("fault_analysis", "D3_semantic", "fault", "fault",
            [{"op": "line_length", "resolve": "longest_line", "scale": 1.8}],
            {"qtype": "isc", "resolve": "lowest_voltage_bus"},
            "Increase the length of the longest line section by 80%, then run a fault study and report "
            "the phase-1 short-circuit current magnitude (A) at the electrically weakest bus.",
            "longest_line+lowest_voltage_bus"),
        _mk("fault_analysis", "D3_semantic", "fault2", "fault",
            [{"op": "line_length", "resolve": "longest_line", "scale": 1.5}],
            {"qtype": "zsc1", "resolve": "lowest_voltage_bus"},
            "Extend the longest feeder section by 50%, then run a fault study and report the magnitude "
            "(ohm) of the positive-sequence Thevenin impedance at the weakest (feeder-end) bus.",
            "longest_line+lowest_voltage_bus"),
        _mk("fault_analysis", "D3_semantic", "fault", "fault",
            [{"op": "line_length", "resolve": "longest_line", "scale": 2.0}],
            {"qtype": "isc", "resolve": "lowest_voltage_bus"},
            "Double the length of the longest line section, then run a fault study and report the "
            "phase-1 short-circuit current magnitude (A) at the electrically weakest bus.",
            "longest_line+lowest_voltage_bus"),
        _mk("fault_analysis", "D3_semantic", "fault2", "fault",
            [{"op": "line_length", "resolve": "longest_line", "scale": 1.4}],
            {"qtype": "isc", "resolve": "lowest_voltage_bus"},
            "Extend the longest feeder section by 40%, then run a fault study and report the phase-1 "
            "short-circuit current magnitude (A) at the weakest (feeder-end) bus.",
            "longest_line+lowest_voltage_bus"),
        _mk("fault_analysis", "D3_semantic", "fault", "fault",
            [{"op": "line_length", "resolve": "longest_line", "scale": 1.5}],
            {"qtype": "zsc1", "resolve": "lowest_voltage_bus"},
            "Increase the longest line section by 50%, then run a fault study and report the magnitude "
            "(ohm) of the positive-sequence Thevenin impedance at the electrically weakest bus.",
            "longest_line+lowest_voltage_bus"),
        _mk("fault_analysis", "D3_semantic", "fault2", "fault",
            [{"op": "line_length", "resolve": "longest_line", "scale": 1.7}],
            {"qtype": "isc", "resolve": "lowest_voltage_bus"},
            "Extend the longest feeder section by 70%, then run a fault study and report the phase-1 "
            "short-circuit current magnitude (A) at the weakest bus.",
            "longest_line+lowest_voltage_bus"),
        _mk("fault_analysis", "D3_semantic", "fault", "fault",
            [{"op": "line_length", "resolve": "longest_line", "scale": 1.6}],
            {"qtype": "zsc1", "resolve": "lowest_voltage_bus"},
            "Extend the longest line section by 60%, then run a fault study and report the magnitude "
            "(ohm) of the positive-sequence Thevenin impedance at the weakest (feeder-end) bus.",
            "longest_line+lowest_voltage_bus"),
        _mk("fault_analysis", "D3_semantic", "fault2", "fault",
            [{"op": "line_length", "resolve": "longest_line", "scale": 2.2}],
            {"qtype": "isc", "resolve": "lowest_voltage_bus"},
            "Increase the longest feeder section length by 120%, then run a fault study and report the "
            "phase-1 short-circuit current magnitude (A) at the electrically weakest bus.",
            "longest_line+lowest_voltage_bus"),
        _mk("fault_analysis", "D3_semantic", "fault", "fault",
            [{"op": "line_length", "resolve": "longest_line", "scale": 1.9}],
            {"qtype": "isc", "resolve": "lowest_voltage_bus"},
            "Extend the longest line section by 90%, then run a fault study and report the phase-1 "
            "short-circuit current magnitude (A) at the weakest bus.",
            "longest_line+lowest_voltage_bus"),
    ]

    return C


# ==============================================================================================
# Item realization (resolve D3 targets, render, run GT, admission, sensitivity).
# ==============================================================================================

def realize_mods(spec, recipe):
    """Return (rendered_mod_lines, structured_mods, resolved_record)."""
    rendered, structured = [], []
    resolved = {}
    for m in spec["mods"]:
        m = dict(m)
        if "resolve" in m:
            kind = m.pop("resolve")
            op = m["op"]
            if op == "add_parallel_line":
                name, _ = resolve(kind, recipe, spec["analysis"])
                b1, b2, ln, ph = line_endpoints(recipe, name)
                cmd = (f"New Line.{m['name']} bus1={b1} bus2={b2} length={ln} units=km "
                       f"phases={ph} {_RM3} {_XM3}")
                sm = {"op": "add_line", "command": cmd, "parallel_to": name}
                structured.append(sm)
                rendered += render_mod(sm)
                resolved[spec.get("resolved") or kind] = name
                continue
            if op == "disable_line":
                name, _ = resolve(kind, recipe, spec["analysis"])
                sm = {"op": "disable_line", "command": f"Disable Line.{name}", "resolved_line": name}
                structured.append(sm)
                rendered += render_mod(sm)
                resolved[spec.get("resolved") or kind] = name
                continue
            name, base = resolve(kind, recipe, spec["analysis"])
            if "scale" in m:
                val = round(float(base) * float(m.pop("scale")), 6)
            elif "delta" in m:
                val = round(float(base) + float(m.pop("delta")), 6)
            else:
                raise ValueError("resolve mod needs scale or delta")
            if op == "load_kw":
                m = {"op": "load_kw", "load": name, "value": val}
            elif op == "line_length":
                m = {"op": "line_length", "line": name, "value": val}
            elif op == "cap_kvar":
                m = {"op": "cap_kvar", "cap": name, "value": val}
            elif op == "reg_vreg":
                m = {"op": "reg_vreg", "reg": name, "value": val}
            else:
                raise ValueError(f"unsupported resolve op {op}")
            m["resolved_from"] = {"semantic": spec.get("resolved") or kind,
                                  "element": name, "base_value": round(float(base), 6)}
            resolved[spec.get("resolved") or kind] = name
            structured.append(m)
            rendered += render_mod({k: v for k, v in m.items() if k != "resolved_from"})
            continue
        structured.append(m)
        rendered += render_mod(m)
    return rendered, structured, (resolved or None)


def realize_query(spec, recipe):
    """Resolve semantic query location (e.g. feeder-end fault bus) against the built network."""
    q = dict(spec["query"])
    extra = None
    if "resolve" in q:
        kind = q.pop("resolve")
        bus, _ = resolve(kind, recipe, "snap")
        q["bus"] = bus
        extra = {"query_semantic": kind, "query_resolved_element": bus}
    return q, extra


def make_id(family, diff, code):
    return hashlib.sha1(f"{family}|{diff}|{code}".encode()).hexdigest()[:12]


def realize(spec):
    recipe, netdesc = BUILDERS[spec["builder"]]()

    # structural admission (family-specific) on the base circuit.
    ok, aerr = admit(recipe, spec["family"])
    if not ok:
        return None, f"admission failed: {aerr}"

    mod_lines, structured_mods, resolved_rec = realize_mods(spec, recipe)
    query, query_extra = realize_query(spec, recipe)
    code = assemble_code(recipe, mod_lines, spec["analysis"], query)

    # GROUND TRUTH: isolated subprocess (protocol §2.3, anti-contamination) -- authoritative.
    gt, err = run_isolated(code)
    if gt is None:
        return None, f"GT run failed: {err}"
    if not (gt == gt and abs(gt) != float("inf")):  # finite check
        return None, f"GT non-finite: {gt}"
    gt = round(gt, 6)

    # SENSITIVITY BASELINE: in-process, unmodified circuit (QC only, not shipped as GT).
    try:
        ns = run_body_inproc(recipe, None, spec["analysis"], query)
        gt_base = round(float(ns["result"]), 6)
    except Exception as e:  # noqa: BLE001
        gt_base, berr = None, f"{type(e).__name__}: {e}"
    else:
        berr = None
    ok, delta = sensitive(gt, gt_base)
    if not ok:
        return None, f"NOT sensitive (gt={gt}, base={gt_base}, delta={delta}, berr={berr})"

    # D3 semantic-QUERY argsel: record which element a min/max query resolves to on the modified,
    # solved circuit (the implicit target the model must recover; NL must not name it).
    query_argsel = None
    if spec["diff"] == "D3_semantic" and query["qtype"] in ("min_node_v", "max_line_i"):
        try:
            run_body_inproc(recipe, mod_lines, spec["analysis"])  # leaves modified+solved engine
            sel, elem = argsel(query)
            if elem is not None:
                query_argsel = {"query_selector": sel, "query_resolved_element": elem}
        except Exception:  # noqa: BLE001
            pass

    scenario = {
        "task": spec["family"],
        "network": netdesc,
        "analysis": spec["analysis"],
        "modifications": structured_mods,
        "query_target": query,
    }
    rt = {}
    if resolved_rec:
        rt.update(resolved_rec)
    if query_extra:
        rt.update(query_extra)
    if query_argsel:
        rt.update(query_argsel)
    if rt:
        scenario["resolved_target"] = rt

    item = {
        "id": "e2opendss_" + make_id(spec["family"], spec["diff"], code),
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


def _write(items):
    OUT_JSON.write_text(json.dumps(items, indent=2))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    req = [a for a in sys.argv[1:] if not a.startswith("-")]
    target_families = req if req else list(FAMILIES)
    for f in target_families:
        if f not in FAMILIES:
            raise SystemExit(f"unknown family {f!r}; choose from {FAMILIES}")

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


def _report(accepted, rejected, dt):
    from collections import Counter

    cells = Counter((it["scenario"]["task"], it["difficulty_level"]) for it in accepted)
    print("\n" + "=" * 72)
    print(f"E2 OpenDSS mini-bench: {len(accepted)} items in {dt:.1f}s  (rejected {len(rejected)})")
    print("-" * 72)
    print(f"{'family':<24}{'D1_basic':>10}{'D3_semantic':>13}{'total':>8}")
    for fam in FAMILIES:
        d1 = cells.get((fam, "D1_basic"), 0)
        d3 = cells.get((fam, "D3_semantic"), 0)
        print(f"{fam:<24}{d1:>10}{d3:>13}{d1 + d3:>8}")
    print("-" * 72)
    print(f"{'TOTAL':<24}{sum(1 for i in accepted if i['difficulty_level']=='D1_basic'):>10}"
          f"{sum(1 for i in accepted if i['difficulty_level']=='D3_semantic'):>13}{len(accepted):>8}")
    print("=" * 72)
    if rejected:
        print("Rejected candidates (over-generated buffer, expected):")
        for r in rejected:
            print("  ", r)
    print(f"\nWritten: {OUT_JSON}")


if __name__ == "__main__":
    main()
