# --------------------------------------------------------------------------
# Repository copy of audit/sandbox.py. The frozen audit implementation is
# unchanged except for the benchmark-engine package path, which is adapted to
# this repository's code/ layout.
# --------------------------------------------------------------------------
"""T1 -- instrumented re-execution sandbox for the E1 validity audit.

Re-executes a single code string (a model's ``generated_code`` or the
benchmark's ``reference_code``) and captures the raw evidence the five
checks (T2, out of scope here) will later interpret:

  * the resulting pandapower ``net`` object(s) left in the namespace, as a
    structural signature (via ``benchmark_engine.get_net_metadata_from_net``)
    plus a small raw state block (converged flag, populated result tables,
    voltage range);
  * a solver-call trace: which pandapower solvers were invoked, in order,
    with the physics-relevant kwargs (fault/case for short-circuit, etc.)
    and each call's convergence / exception outcome;
  * captured stdout (the benchmark's scalar answer is printed here);
  * a hard wall-clock timeout.

This is a *capture* layer, not a *judge* layer: it records facts (which
solver ran, did it converge, what does the net look like) and leaves the
"is this the expected solver / a valid physical state" rulings to T2.

Design mirrors ``utils.execute_code_safely`` (matplotlib/webbrowser mocked,
log/stderr silenced, exit() stripped, SIGALRM timeout, run in a scratch cwd)
so that re-execution reproduces the environment under which the on-disk
results were originally generated -- but unlike that helper it (a) returns
the post-exec namespace so we can recover ``net``, and (b) traces solvers.

Existing engine code is imported and reused unmodified.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import os
import re
import signal
import sys
import tempfile
import warnings
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

# Reuse (do not reimplement) the engine's net-signature extractor.
from benchmark_generator.benchmark_engine import get_net_metadata_from_net

import pandapower as pp
import pandapower.shortcircuit  # noqa: F401  (registers pp.shortcircuit)
import pandapower.estimation    # noqa: F401
import pandapower.timeseries    # noqa: F401
import pandapower.contingency   # noqa: F401


# --------------------------------------------------------------------------
# Which solvers to trace.  The first six are the protocol-named set
# (E1 doc s2.2 check (2)); rundcopp and run_contingency are traced in
# addition because they are the analysis entry points of the opf/contingency
# families (GAP-7) and cost nothing extra to capture.  The T2 checker decides
# which solver is "expected" per task family.
# --------------------------------------------------------------------------
PROTOCOL_SOLVERS = ("runpp", "rundcpp", "runopp", "calc_sc", "estimate", "run_timeseries")
_EXTRA_SOLVERS = ("rundcopp", "run_contingency")

# name -> (holder_module, attribute) ; the attribute is looked up at call
# time by user code, so patching it here (before exec) intercepts both
# ``pp.<name>(...)`` and ``from <mod> import <name>`` usage.
_PATCH_TARGETS = {
    "runpp": (pp, "runpp"),
    "rundcpp": (pp, "rundcpp"),
    "runopp": (pp, "runopp"),
    "rundcopp": (pp, "rundcopp"),
    "calc_sc": (pp.shortcircuit, "calc_sc"),
    "estimate": (pp.estimation, "estimate"),
    "run_timeseries": (pp.timeseries, "run_timeseries"),
    "run_contingency": (pp.contingency, "run_contingency"),
}

# kwargs worth recording per solver (physics-distinguishing, GAP-7).
# ``method`` is captured because ``runpp`` has no ``method`` parameter (pandapower
# 3.4.0): ``runpp(method="dc")`` is a no-op kwarg silently absorbed into
# ``**kwargs`` that still solves AC.  Recording it lets the T2 solver-identity
# check flag the silently-dropped DC request (``noop_kwargs``) rather than
# mistake it for a genuine DC power flow (E1 doc s8 D-(1)).
_PARAMS_OF_INTEREST = ("fault", "case", "algorithm", "calculate_voltage_angles",
                       "use_pre_fault_voltage", "elements", "distributed_slack",
                       "method")

_RESULT_TABLES = ("res_bus", "res_line", "res_trafo", "res_gen", "res_load",
                  "res_ext_grid", "res_sgen", "res_bus_sc", "res_line_sc",
                  "res_cost")

# Upper bound on how many pre-solve modification-fingerprint snapshots to keep
# per run (E1 doc s8 N6/L1).  The first snapshot per run is the one check (1)
# uses for the compound families; the rest are retained as evidence.  Bounded so
# a hand-written per-step solver loop (which may call runpp hundreds of times)
# cannot make capture unbounded.
_PRESOLVE_MAX = 6


def _is_net(obj: Any) -> bool:
    return type(obj).__name__ == "pandapowerNet"


# pandapower stores convergence under different keys per solver family;
# short-circuit has no scalar convergence flag (success == result table present).
_CONVERGED_KEY = {
    "runpp": "converged", "rundcpp": "converged",
    "run_timeseries": "converged", "run_contingency": "converged",
    "runopp": "OPF_converged", "rundcopp": "OPF_converged",
    "calc_sc": None,
}


def _read_converged(name: str, net: Any, ret: Any) -> Optional[bool]:
    """Best-effort convergence read for one solver call (raw capture)."""
    if name == "estimate":
        # pandapower estimate() returns a success bool (possibly numpy.bool_).
        if ret is None:
            return None
        try:
            return bool(ret)
        except Exception:
            return None
    key = _CONVERGED_KEY.get(name, "converged")
    if key and _is_net(net):
        try:
            if key in net:
                return bool(net[key])
        except Exception:
            pass
    return None


def _select_params(args: tuple, kwargs: dict) -> Dict[str, Any]:
    out = {}
    for k in _PARAMS_OF_INTEREST:
        if k in kwargs:
            v = kwargs[k]
            out[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return out


def _install_solver_trace(trace: List[dict],
                          presolve: Optional[List[dict]] = None):
    """Patch traced solvers to append to ``trace``; return an undo callable.

    When ``presolve`` is supplied, the modification fingerprint of the net
    handed to each of the first ``_PRESOLVE_MAX`` solver calls is captured
    *before* the solve runs and appended to it, in call order.  This gives
    check (1) a pre-solve view of the network for the compound families
    (comparison / comparison_opf / time_series), where the post-exec net state
    is a double-net / controller-residue artifact and diffing it produces false
    modification divergences (E1 doc s8 N6/L1).  The capture is additive: with
    ``presolve=None`` behaviour is byte-identical to before.
    """
    saved = []

    def make_wrapper(name, orig):
        def wrapper(*args, **kwargs):
            entry = {"solver": name, "params": _select_params(args, kwargs)}
            net = args[0] if args else kwargs.get("net")
            if presolve is not None and _is_net(net) \
                    and len(presolve) < _PRESOLVE_MAX:
                try:
                    presolve.append({
                        "solve_order": len(presolve), "solver": name,
                        "n_bus": len(net["bus"]),
                        "fingerprint": _mod_fingerprint(net),
                    })
                except Exception:
                    pass
            try:
                ret = orig(*args, **kwargs)
                entry["raised"] = False
                entry["error_type"] = None
                entry["converged"] = _read_converged(name, net, ret)
                trace.append(entry)
                return ret
            except BaseException as e:  # record then re-raise, behaviour unchanged
                entry["raised"] = True
                entry["error_type"] = type(e).__name__
                entry["converged"] = _read_converged(name, net, None)
                trace.append(entry)
                raise
        wrapper.__name__ = getattr(orig, "__name__", name)
        return wrapper

    for name, (holder, attr) in _PATCH_TARGETS.items():
        orig = getattr(holder, attr)
        saved.append((holder, attr, orig))
        setattr(holder, attr, make_wrapper(name, orig))

    def undo():
        for holder, attr, orig in saved:
            setattr(holder, attr, orig)

    return undo


# --------------------------------------------------------------------------
# Optional load perturbation (check (5) v2 -- perturbation re-execution).
#
# All access paths a model uses to load a network (``pp.networks.caseX``,
# ``import pandapower.networks as pn; pn.caseX``, ``nw.caseX`` ...) resolve to
# attributes of the single ``pandapower.networks`` module object.  Wrapping the
# public loaders on that module therefore perturbs whatever base network the
# code loads, regardless of the alias it used.  The wrapper multiplies every
# load's ``scaling`` (falling back to generator ``p_mw`` when the net has no
# load) by ``factor`` on the returned net, so a genuinely-simulated output moves
# while a hard-coded / constant output does not.  ``undo`` restores the module
# exactly, so nothing leaks into the next item's execution.
# --------------------------------------------------------------------------
import pandapower.networks as _pp_networks


def _install_load_perturbation(factor: float):
    saved = []

    def make_wrapper(orig):
        def wrapper(*args, **kwargs):
            net = orig(*args, **kwargs)
            try:
                if _is_net(net):
                    if len(net["load"]) > 0 and "scaling" in net["load"].columns:
                        net["load"]["scaling"] = net["load"]["scaling"] * factor
                    elif len(net["gen"]) > 0 and "p_mw" in net["gen"].columns:
                        net["gen"]["p_mw"] = net["gen"]["p_mw"] * factor
            except Exception:
                pass
            return net
        wrapper.__name__ = getattr(orig, "__name__", "loader")
        return wrapper

    for name in dir(_pp_networks):
        if name.startswith("_"):
            continue
        orig = getattr(_pp_networks, name)
        if not callable(orig) or isinstance(orig, type):
            continue
        saved.append((name, orig))
        setattr(_pp_networks, name, make_wrapper(orig))

    def undo():
        for name, orig in saved:
            setattr(_pp_networks, name, orig)

    return undo


# --------------------------------------------------------------------------
# Modification fingerprint (check (1)) and power-balance inputs (check (3)).
# Both are compact, size-bounded digests captured from the post-exec net so a
# generated net can be diffed against a reference net without shipping whole
# tables (honours the OOM discipline for large grids).
# --------------------------------------------------------------------------
_FULL_DICT_MAX_ROWS = 64   # capture per-index values only for small tables

# (table, column) whose value is a modification target; each entry says how to
# reduce it to a bounded fingerprint.  "outset" -> sorted indices where flagged
# out of service; "nondefault" -> {idx: value} for entries != default; "digest"
# -> rounded sum + row count + order-independent per-(index,value) content hash
# (+ per-index dict when the table is small).
#
# Coverage is audited against benchmark_config.MODIFICATION_DEFS: every column
# any modification operator writes is fingerprinted (checker defect D-3 --
# shunt.* and gen.max_p_mw were missing and made wrong-sign / wrong-bus /
# wrong-limit modifications invisible to check (1) whenever element counts
# still matched).  Operators covered per entry are annotated.
_MOD_COLUMNS = (
    ("line", "in_service", "outset"),            # disconnect_line
    ("trafo", "in_service", "outset"),           # disconnect_trafo
    ("gen", "in_service", "outset"),             # set_gen_out_of_service
    ("sgen", "in_service", "outset"),            # (SC setup / model toggles)
    ("shunt", "in_service", "outset"),           # add_shunt end-state (D-3)
    ("switch", "closed", "outset"),              # open_switch / close_switch
    ("line", "parallel", "nondefault", 1),       # set_line_parallel
    ("trafo", "parallel", "nondefault", 1),      # set_trafo_parallel
    ("line", "length_km", "digest"),             # set_line_length
    ("line", "r_ohm_per_km", "digest"),          # set_line_r
    ("line", "x_ohm_per_km", "digest"),          # set_line_x
    ("line", "max_i_ka", "digest"),              # set_line_max_i (D-3 class)
    ("trafo", "tap_pos", "nondefault", 0),       # set_trafo_tap_pos
    ("trafo", "shift_degree", "nondefault", 0),  # set_trafo_shift
    ("load", "scaling", "nondefault", 1.0),      # set_load_scaling / scale_all_loads
    ("load", "p_mw", "digest"),                  # set_load_p / add_load
    ("load", "q_mvar", "digest"),                # set_load_q / add_load
    ("load", "bus", "digest"),                   # add_load target bus (D-3 class)
    ("gen", "p_mw", "digest"),                   # set_gen_p / add_gen
    ("gen", "vm_pu", "digest"),                  # set_gen_vm / add_gen
    ("gen", "max_p_mw", "digest"),               # set_gen_max_p (D-3b, observed)
    ("gen", "min_p_mw", "digest"),               # set_gen_min_p (D-3 class)
    ("gen", "bus", "digest"),                    # add_gen target bus (D-3 class)
    ("sgen", "p_mw", "digest"),                  # add_sgen (D-3 class)
    ("sgen", "q_mvar", "digest"),                # add_sgen (D-3 class)
    ("sgen", "bus", "digest"),                   # add_sgen target bus (D-3 class)
    ("shunt", "q_mvar", "digest"),               # add_shunt (D-3, observed)
    ("shunt", "p_mw", "digest"),                 # add_shunt (D-3)
    ("shunt", "bus", "digest"),                  # add_shunt target bus (D-3, observed)
    ("storage", "p_mw", "digest"),               # add_storage (D-3 class)
    ("storage", "max_e_mwh", "digest"),          # add_storage (D-3 class)
    ("storage", "bus", "digest"),                # add_storage target bus (D-3 class)
    ("ext_grid", "vm_pu", "digest"),             # set_ext_grid_vm
    ("bus", "max_vm_pu", "digest"),              # set_bus_max_vm
    ("bus", "min_vm_pu", "digest"),              # set_bus_min_vm
)


def _col_fingerprint(net: Any, table: str, column: str, kind: str,
                     default: Any = None) -> Optional[Dict[str, Any]]:
    """Bounded fingerprint of one (table, column); None if absent/empty."""
    try:
        t = net[table]
        if column not in getattr(t, "columns", []):
            return None
        s = t[column]
        n = len(s)
        if n == 0:
            return None
    except Exception:
        return None
    try:
        if kind == "outset":
            # boolean columns (in_service / switch.closed): record indices whose
            # flag is falsy (out of service / open).
            idx = sorted(int(i) for i, v in s.items() if not bool(v))
            return {"kind": "outset", "n": int(n), "out": idx}
        if kind == "nondefault":
            nd = {}
            for i, v in s.items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                # NaN is not a modification and, being != itself, would make
                # two identical fingerprints compare unequal -> skip it.
                if fv != fv:
                    continue
                if fv != float(default):
                    nd[int(i)] = round(fv, 6)
            return {"kind": "nondefault", "n": int(n), "default": default,
                    "values": nd}
        # digest
        vals = s.dropna()
        rsum = round(float(vals.astype(float).sum()), 6) if len(vals) else 0.0
        out: Dict[str, Any] = {"kind": "digest", "n": int(n),
                               "sum": rsum, "n_nan": int(n - len(vals))}
        # Order-independent per-(index, value) content hash (checker defect
        # D-5): two columns with the same sum and count but the value at a
        # different index (e.g. min_vm_pu set on bus 11 instead of bus 12) no
        # longer compare equal.  Bounded: one 16-hex digest whatever the table
        # size, so the index sensitivity also covers tables larger than
        # _FULL_DICT_MAX_ROWS where no per-index dict is captured.
        h = hashlib.blake2b(digest_size=8)
        for i, v in sorted((int(i), round(float(v), 6))
                           for i, v in vals.items()):
            h.update(f"{i}:{v!r};".encode())
        out["vhash"] = h.hexdigest()
        if n <= _FULL_DICT_MAX_ROWS:
            out["values"] = {int(i): round(float(v), 6)
                             for i, v in vals.items()}
        return out
    except Exception:
        return None


def _mod_fingerprint(net: Any) -> Dict[str, Any]:
    fp: Dict[str, Any] = {}
    for spec in _MOD_COLUMNS:
        table, column, kind = spec[0], spec[1], spec[2]
        default = spec[3] if len(spec) > 3 else None
        res = _col_fingerprint(net, table, column, kind, default)
        if res is not None:
            fp[f"{table}.{column}"] = res
    return fp


def _balance_capture(net: Any) -> Dict[str, Any]:
    """Power-balance inputs + result-range summaries for check (3)."""
    bal: Dict[str, Any] = {}

    def col_sum(table, column):
        try:
            t = net[table]
            if column in getattr(t, "columns", []) and len(t):
                v = t[column].dropna()
                return round(float(v.astype(float).sum()), 6) if len(v) else 0.0
        except Exception:
            return None
        return None

    def col_stat(table, column, fn):
        try:
            t = net[table]
            if column in getattr(t, "columns", []) and len(t):
                v = t[column].dropna()
                if len(v):
                    return round(float(fn(v.astype(float))), 6)
        except Exception:
            return None
        return None

    bal["p_gen"] = col_sum("res_gen", "p_mw")
    bal["p_sgen"] = col_sum("res_sgen", "p_mw")
    bal["p_ext_grid"] = col_sum("res_ext_grid", "p_mw")
    bal["p_load"] = col_sum("res_load", "p_mw")
    bal["p_loss_line"] = col_sum("res_line", "pl_mw")
    bal["p_loss_trafo"] = col_sum("res_trafo", "pl_mw")
    # further consumption terms (needed for a closed balance: e.g. case145 has
    # 97 shunts drawing ~77 GW -- omitting them fakes a huge residual).
    bal["p_shunt"] = col_sum("res_shunt", "p_mw")
    bal["p_storage"] = col_sum("res_storage", "p_mw")
    bal["p_ward"] = col_sum("res_ward", "p_mw")
    bal["loading_line_max"] = col_stat("res_line", "loading_percent", lambda v: v.max())
    bal["loading_trafo_max"] = col_stat("res_trafo", "loading_percent", lambda v: v.max())
    # contingency envelope columns
    bal["max_loading_line_max"] = col_stat("res_line", "max_loading_percent", lambda v: v.max())
    bal["min_vm_bus_min"] = col_stat("res_bus", "min_vm_pu", lambda v: v.min())
    # short-circuit
    bal["ikss_min"] = col_stat("res_bus_sc", "ikss_ka", lambda v: v.min())
    bal["ikss_max"] = col_stat("res_bus_sc", "ikss_ka", lambda v: v.max())
    # OPF cost (scalar attribute, not a table)
    try:
        rc = net["res_cost"]
        bal["res_cost"] = float(rc) if rc is not None else None
    except Exception:
        bal["res_cost"] = None
    return bal


def _net_capture(net: Any) -> Dict[str, Any]:
    """Structural signature (reused engine fn) + raw state observations.

    ``state`` additionally carries a compact modification fingerprint
    (``mod_fingerprint``, for check (1)'s state diff) and power-balance inputs
    (``balance``, for check (3)'s residual / sanity), both size-bounded.
    """
    sig: Dict[str, Any] = {}
    try:
        sig["metadata"] = get_net_metadata_from_net(net)
    except Exception as e:
        sig["metadata"] = None
        sig["metadata_error"] = f"{type(e).__name__}: {e}"
    # element counts beyond the reused metadata helper (n_shunt / n_storage /
    # n_switch matter for the "added element" part of check (1)).
    try:
        sig["extra_counts"] = {
            "n_shunt": len(net["shunt"]), "n_storage": len(net["storage"]),
            "n_switch": len(net["switch"]), "n_sgen": len(net["sgen"]),
            "n_ext_grid": len(net["ext_grid"]),
        }
    except Exception:
        sig["extra_counts"] = None
    state: Dict[str, Any] = {}
    for flag in ("converged", "OPF_converged"):
        try:
            state[flag] = bool(net[flag]) if flag in net else None
        except Exception:
            state[flag] = None
    present = []
    for t in _RESULT_TABLES:
        try:
            tbl = net[t]
            if tbl_len(tbl) > 0:
                present.append(t)
        except Exception:
            pass
    state["res_tables_present"] = present
    try:
        rb = net["res_bus"]
        if tbl_len(rb) > 0 and "vm_pu" in rb:
            col = rb["vm_pu"].dropna()
            if len(col) > 0:
                state["vm_pu_min"] = float(col.min())
                state["vm_pu_max"] = float(col.max())
    except Exception:
        pass
    state["mod_fingerprint"] = _mod_fingerprint(net)
    state["balance"] = _balance_capture(net)
    sig["state"] = state
    return sig


def tbl_len(tbl) -> int:
    try:
        return len(tbl)
    except Exception:
        return 0


# code transforms applied by the original harness (utils.execute_code_safely);
# reproduced so re-execution matches how results were first generated.
_RE_EXIT = re.compile(r"\n\s*exit\(.*?\)")
_RE_SYSEXIT = re.compile(r"\n\s*sys\.exit\(.*?\)")
_RE_OUTPATH = re.compile(r"output_path\s*=\s*['\"][^'\"]*['\"]")


def run_instrumented(code: str, timeout: int = 90,
                     suppress_logs: bool = True,
                     perturb_factor: Optional[float] = None) -> Dict[str, Any]:
    """Execute ``code`` with solver tracing and namespace/net capture.

    ``perturb_factor`` (check (5) v2): when set, every base network loaded by the
    code has its loads (or generators, if load-free) scaled by this factor before
    the code sees it; the module is restored afterwards.  Default ``None`` leaves
    behaviour byte-identical to an unperturbed run.

    Returns a dict:
        success (bool), error, error_type, traceback,
        stdout (str), timed_out (bool),
        solver_trace (list of call entries, in invocation order),
        presolve_fingerprints (list of pre-solve modification fingerprints, in
            call order; check (1) uses the first for the compound families),
        nets ({var_name: signature}), primary_net_var (str|None),
        primary_net (signature|None).
    """
    result: Dict[str, Any] = {
        "success": False, "error": None, "error_type": None, "traceback": None,
        "stdout": "", "timed_out": False,
        "solver_trace": [], "presolve_fingerprints": [],
        "nets": {}, "primary_net_var": None, "primary_net": None,
    }
    if not code or not code.strip():
        result["error"] = "empty code"
        result["error_type"] = "EmptyCode"
        return result

    trace: List[dict] = []
    result["solver_trace"] = trace
    presolve: List[dict] = []
    result["presolve_fingerprints"] = presolve

    out_buf, err_buf = io.StringIO(), io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    original_cwd = os.getcwd()
    original_logging_disable = logging.root.manager.disable

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution exceeded {timeout} seconds")

    old_alarm_handler = signal.getsignal(signal.SIGALRM)
    undo_trace = None
    undo_perturb = None
    workdir = tempfile.mkdtemp(prefix="e1_sandbox_")

    src = _RE_EXIT.sub("\n# exit() removed", code)
    src = _RE_SYSEXIT.sub("\n# sys.exit() removed", src)
    src = _RE_OUTPATH.sub("output_path='./output'", src)

    namespace: Dict[str, Any] = {}
    try:
        if suppress_logs:
            logging.disable(logging.CRITICAL)
        sys.stdout, sys.stderr = out_buf, err_buf
        undo_trace = _install_solver_trace(trace, presolve)
        if perturb_factor is not None:
            undo_perturb = _install_load_perturbation(perturb_factor)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        try:
            with warnings.catch_warnings():
                if suppress_logs:
                    warnings.simplefilter("ignore")
                with patch("matplotlib.pyplot.show", MagicMock()), \
                        patch("matplotlib.pyplot.savefig", MagicMock()), \
                        patch("matplotlib.pyplot.figure", MagicMock()), \
                        patch("webbrowser.open", MagicMock()), \
                        patch("webbrowser.open_new_tab", MagicMock()), \
                        patch("webbrowser.open_new", MagicMock()):
                    exec("import matplotlib", namespace)
                    exec("import matplotlib.pyplot as plt", namespace)
                    exec("matplotlib.use('Agg')", namespace)
                    os.chdir(workdir)
                    exec(src, namespace)
            result["success"] = True
        finally:
            signal.alarm(0)
    except TimeoutError as e:
        result["error"] = str(e)
        result["error_type"] = "TimeoutError"
        result["timed_out"] = True
    except BaseException as e:
        import traceback as _tb
        result["error"] = str(e)
        result["error_type"] = type(e).__name__
        result["traceback"] = _tb.format_exc()
    finally:
        os.chdir(original_cwd)
        if undo_perturb is not None:
            undo_perturb()
        if undo_trace is not None:
            undo_trace()
        signal.signal(signal.SIGALRM, old_alarm_handler)
        sys.stdout, sys.stderr = old_stdout, old_stderr
        logging.disable(original_logging_disable)
        result["stdout"] = out_buf.getvalue()
        result["stderr"] = err_buf.getvalue()
        # -- capture net objects left in the namespace --
        nets = {}
        for name, val in namespace.items():
            if _is_net(val):
                nets[name] = _net_capture(val)
        result["nets"] = nets
        if nets:
            primary = "net" if "net" in nets else sorted(nets)[0]
            result["primary_net_var"] = primary
            result["primary_net"] = nets[primary]
        # best-effort scratch cleanup
        try:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass
    return result


TRACED_SOLVERS = tuple(_PATCH_TARGETS.keys())
