# --------------------------------------------------------------------------
# Repository copy of audit/checkers.py from the frozen experimental pipeline.
# Its inputs are raw per-item run trees too large to ship here, so it does not
# run from this checkout; see ARTIFACT_INDEX.md for the outputs it produced.
# --------------------------------------------------------------------------
"""T2 -- the five engineering-validity checks for the E1 audit.

Implements E1 doc s2.2 checks (1)-(5) as an automatic layer that produces, per
matched item, a ``pass`` / ``fail`` / ``ambiguous`` verdict plus structured
evidence for each check.  The verdicts and evidence feed the downstream
dual-judge + adversarial-refute pass (E1 doc s2.4, out of scope here): this
module makes *no* final acceptance/pollution ruling and computes *no* FAR.

Check summary
-------------
(1) correct network + modifications consistent with intent
      -- structural signature + modification-fingerprint diff of the re-executed
         generated ``net`` against the re-executed reference ``net``
         (self-consistent, no external modification spec needed; GAP-4).
(2) expected analysis program invoked and converged
      -- rebuilt solver-identity table (15 families x expected solver identity,
         DC-vs-AC and 2ph-vs-3ph distinguished, multi-step compound families),
         cross-checked against the reference's own solver trace.  Does NOT reuse
         the on-disk ``TASK_FUNCTION_PATTERNS`` (GAP-7).
(3) power-balance residual + task-relevant sanity
      -- re-implements the construction-time admission sanity as a *posterior*
         validator on the model output's net (GAP-3), per-family applicability.
(4) correct result table / column / aggregation / index extracted
      -- AST parse of both extraction expressions + query_type cross-check;
         non-trivial derived expressions are marked ambiguous.
(5) output genuinely comes from simulation (not hard-coded)
      -- v1 static literal scan + v2 perturbation re-execution (scale one input,
         check the output moves); families where a perturbation cannot be
         reliably constructed fall back to v1 with an annotation.

Everything re-executes inside the T1 ``sandbox`` (pandapower 3.4.0); only engine
code is imported, and it is reused unmodified.  Per-item hard timeouts and
sequential execution honour the login-node OOM discipline (E1 doc s7).
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from audit import records, sandbox
from audit.records import PROJECT_ROOT, cond_path

# ==========================================================================
# Frozen constants
# ==========================================================================
PERTURB_FACTOR = 1.10          # check (5) v2 load-scaling perturbation
LARGE_NET_BUS = 2500           # skip the extra perturbation exec above this size
DEFAULT_TIMEOUT = 120          # per-exec hard wall-clock (s)
RESIDUAL_ABS_TOL = 0.5         # MW; power-balance residual tolerance floor
RESIDUAL_REL_TOL = 0.005       # + 0.5% of served load
VM_HARD_LO, VM_HARD_HI = 0.5, 1.5     # grossly-implausible voltage band -> fail
VM_SOFT_LO, VM_SOFT_HI = 0.90, 1.10   # nominal band (construction interval)
GT_REL_TOL, GT_ABS_TOL = 1e-3, 1e-4   # numeric equality for the hard-code scan
PERTURB_MIN_REL, PERTURB_MIN_ABS = 1e-4, 1e-6  # output-moved threshold

PASS, FAIL, AMBIG = "pass", "fail", "ambiguous"

# --------------------------------------------------------------------------
# (2) Solver-identity table (rebuilt for the audit; GAP-7).
#
# ``identities`` = the canonical solver-physics identity/identities the family's
# workflow must exhibit (see ``_classify_solver``).  ``min_calls`` = minimum
# number of solver invocations of the primary identity (2 => baseline+modified
# for the comparison families).  ``engine`` names a canonical multi-network
# engine (run_contingency / run_timeseries) whose replacement by a hand-written
# runpp loop is a known equivalent-path candidate -> ambiguous, not fail.
# For per-item precision the *reference's own* solver trace is the authoritative
# expectation; this table is the family-level cross-check and the degraded-path
# fallback when the reference trace is unavailable.
# --------------------------------------------------------------------------
SOLVER_IDENTITY: Dict[str, Dict[str, Any]] = {
    "power_flow":        {"identities": ["ac_pf"], "min_calls": 1},
    "dc_power_flow":     {"identities": ["dc_pf"], "min_calls": 1},
    "opf":               {"identities": ["ac_opf"], "min_calls": 1, "opf": True},
    "short_circuit_3ph": {"identities": ["sc_3ph"], "min_calls": 1, "sc": True},
    "short_circuit_2ph": {"identities": ["sc_2ph"], "min_calls": 1, "sc": True},
    "contingency":       {"identities": ["contingency"], "min_calls": 1,
                          "engine": "contingency"},
    "time_series":       {"identities": ["timeseries"], "min_calls": 1,
                          "engine": "timeseries"},
    "state_estimation":  {"identities": ["estimate"], "prereq": ["ac_pf"],
                          "min_calls": 1},
    # -- compound (D2/D4) --
    "comparison":        {"identities": ["ac_pf"], "min_calls": 2},
    "comparison_opf":    {"identities": ["ac_opf"], "min_calls": 2, "opf": True},
    "sequential":        {"identities": ["ac_pf"], "min_calls": 1},
    "pf_then_sc":        {"identities": ["ac_pf", "sc_3ph"], "ordered": True,
                          "sc": True},
    "contingency_fix":   {"identities": ["contingency", "ac_pf"],
                          "engine": "contingency"},
    "parallel":          {"identities": "reference_derived", "min_calls": 1},
    "diagnose_and_fix":  {"identities": ["ac_pf"], "min_calls": 1},
}

# --------------------------------------------------------------------------
# (1) Families whose post-exec net state is a double-net / controller-residue
# artifact.  For these, the modification fingerprint is diffed at *pre-solve*
# time (the first solver call's snapshot) instead of the post-exec net, so a
# baseline-vs-modified accounting difference (comparison_opf keeps baseline in
# ``net`` and modified in ``net2``) or a time-series controller's last-step
# scaling does not masquerade as a modification divergence (E1 doc s8 N6/L1).
# --------------------------------------------------------------------------
_C1_PRESOLVE = {"comparison", "comparison_opf", "time_series"}

# --------------------------------------------------------------------------
# (3) Which sanity sub-checks apply to which family.
# --------------------------------------------------------------------------
_C3_RESIDUAL_AC = {"power_flow", "sequential", "diagnose_and_fix", "comparison",
                   "contingency_fix", "parallel", "state_estimation"}
_C3_RESIDUAL_DC = {"dc_power_flow"}
_C3_OPF = {"opf", "comparison_opf"}
_C3_SC = {"short_circuit_3ph", "short_circuit_2ph", "pf_then_sc"}
_C3_CONTINGENCY = {"contingency"}
_C3_VOLTAGE = _C3_RESIDUAL_AC | _C3_OPF | {"contingency"}
_C3_LOADING = _C3_RESIDUAL_AC | _C3_RESIDUAL_DC

# families where the check (5) perturbation cannot be reliably constructed:
# a time-series controller re-imposes the load profile every step, masking a
# base-load perturbation.
_C5_NO_PERTURB = {"time_series"}
# query types whose output is discrete/insensitive: perturbation invariance is
# not evidence of hard-coding for these.
_DISCRETE_QTYPES = {"argmax", "argmin", "bool_check", "count_violations",
                    "sequential"}


# ==========================================================================
# Small numeric / trace helpers
# ==========================================================================
def _last_number(stdout: str) -> Optional[float]:
    """Parse the numeric value from the last parseable line of stdout.

    Model code sometimes prints intermediate lines (e.g. a metric label) before
    the final answer; the benchmark scores the last line.
    """
    if not stdout:
        return None
    for line in reversed([l for l in stdout.splitlines() if l.strip()]):
        tok = line.strip().split()[-1] if line.strip().split() else line.strip()
        for cand in (line.strip(), tok):
            try:
                return float(cand)
            except ValueError:
                pass
        low = line.strip().lower()
        if low in ("true", "false"):
            return 1.0 if low == "true" else 0.0
    return None


def _num_eq(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return False
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= max(GT_ABS_TOL, GT_REL_TOL * abs(b))


def _classify_solver(entry: dict) -> str:
    """Map a raw solver-trace entry to a canonical physics identity.

    ``runpp`` is *always* AC.  pandapower's ``runpp`` has no ``method`` parameter
    (verified, pandapower 3.4.0), so ``runpp(method='dc')`` is a no-op kwarg that
    is silently absorbed into ``**kwargs`` and still solves AC -- bit-identical to
    a plain ``runpp`` and *not* equal to ``rundcpp``.  Classifying it as ``dc_pf``
    would be a factual error (E1 doc s8 D-(1) / frozen case 2).  The silently
    dropped DC request is instead surfaced as a ``noop_kwargs`` flag on check (2)
    evidence (see ``_noop_solver_kwargs``); the identity here stays ``ac_pf`` so a
    DC-family task correctly registers an AC/DC swap.
    """
    name = entry.get("solver")
    params = entry.get("params") or {}
    if name == "runpp":
        return "ac_pf"          # runpp always solves AC; method= is a no-op
    if name == "rundcpp":
        return "dc_pf"
    if name == "runopp":
        return "ac_opf"
    if name == "rundcopp":
        return "dc_opf"
    if name == "calc_sc":
        fault = str(params.get("fault", "3ph")).lower()  # pandapower default 3ph
        return f"sc_{fault}"
    if name == "run_contingency":
        return "contingency"
    if name == "run_timeseries":
        return "timeseries"
    if name == "estimate":
        return "estimate"
    return f"other:{name}"


def _call_ok(entry: dict) -> bool:
    """Did one solver call succeed (not raise, and converge where meaningful)?"""
    if entry.get("raised"):
        return False
    conv = entry.get("converged")
    if conv is None:      # calc_sc has no scalar convergence flag
        return True
    return bool(conv)


def _identity_multiset(trace: List[dict]) -> Dict[str, Dict[str, int]]:
    """{identity: {"total": n, "ok": n_converged}} over a solver trace."""
    out: Dict[str, Dict[str, int]] = {}
    for e in trace or []:
        ident = _classify_solver(e)
        d = out.setdefault(ident, {"total": 0, "ok": 0})
        d["total"] += 1
        if _call_ok(e):
            d["ok"] += 1
    return out


def _noop_solver_kwargs(trace: List[dict]) -> List[Dict[str, Any]]:
    """Solver kwargs a model passed hoping to change the physics but which
    pandapower silently absorbs into ``**kwargs``, leaving the physics unchanged.

    The identity classifier keys only on ``method`` (runpp) and ``fault``
    (calc_sc).  ``fault`` is a *valid* calc_sc parameter, so of the two the sole
    silent no-op is ``runpp(method=...)``: runpp has no ``method`` parameter
    (verified, pandapower 3.4.0), so the DC request is dropped and AC is solved
    regardless (E1 doc s8 D-(1), frozen case 2).  No other
    classification-affecting kwarg is a silent no-op, so this scan is
    deliberately narrow (fix the classification, do not expand scope)."""
    flags: List[Dict[str, Any]] = []
    for e in trace or []:
        if e.get("solver") == "runpp":
            params = e.get("params") or {}
            if "method" in params:
                flags.append({
                    "solver": "runpp", "kwarg": "method",
                    "value": params["method"],
                    "note": "runpp has no 'method' parameter; silently absorbed "
                            "into **kwargs and solved AC (no-op)",
                })
    return flags


# ==========================================================================
# Extraction-expression AST parsing (checks (4) and (5) v1)
# ==========================================================================
_RES_TABLES = ("res_bus_sc", "res_line_sc", "res_bus_est", "res_line_est",
               "res_bus", "res_line", "res_trafo", "res_gen", "res_load",
               "res_ext_grid", "res_sgen", "res_cost", "res_shunt", "res_storage")
_AGG_METHODS = {"max": "max", "min": "min", "sum": "sum", "mean": "mean",
                "idxmax": "argmax", "idxmin": "argmin", "any": "any",
                "all": "all", "count": "count"}


def _name_is_accumulated(tree: ast.Module, name: str) -> bool:
    """True if ``name`` is mutated by ``+=`` etc. or assigned inside a loop --
    i.e. its final value is not its literal initializer (guards against reading a
    loop-accumulated variable, such as a violation counter, as a hard-coded
    constant)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == name:
            return True
        if isinstance(node, (ast.For, ast.While)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Assign, ast.AugAssign)):
                    tgts = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                    for t in tgts:
                        if isinstance(t, ast.Name) and t.id == name:
                            return True
    return False


def _resolve_name(tree: ast.Module, name: str, depth: int = 3) -> Optional[ast.AST]:
    """Return the RHS of the last assignment to ``name`` (shallow chase).

    Returns ``None`` for loop-accumulated / augmented names -- their runtime
    value is not the assigned RHS, so resolving them would mislead checks (4)/(5).
    """
    if _name_is_accumulated(tree, name):
        return None
    val = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    val = node.value
    if isinstance(val, ast.Name) and depth > 0:
        deeper = _resolve_name(tree, val.id, depth - 1)
        return deeper if deeper is not None else val
    return val


def _expr_has_result_source(tree: ast.Module, expr: ast.AST) -> bool:
    """Does ``expr`` (after one level of name resolution) touch a res_* table
    or an OutputWriter ``output``?  Used to pick the reporting print among
    branch prints."""
    node = expr
    if isinstance(expr, ast.Name):
        r = _resolve_name(tree, expr.id)
        if r is not None:
            node = r
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr in _RES_TABLES:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr == "output":
            return True
        if isinstance(sub, ast.Subscript):
            s = _str_const(sub.slice)
            if s and (s in _RES_TABLES or s.split(".")[0] in _RES_TABLES):
                return True
    return False


def _final_print_expr(tree: ast.Module) -> Optional[ast.AST]:
    """The reporting ``print(...)`` argument.

    Prefer (a) the last print whose expression is sourced from a result table
    (handles branch prints where a non-taken branch prints a string), then
    (b) the last non-string-constant print, then (c) the last print.
    """
    args = [n.args[0] for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "print" and n.args]
    if not args:
        return None
    sourced = [a for a in args if _expr_has_result_source(tree, a)]
    if sourced:
        return sourced[-1]
    non_str = [a for a in args
               if not (isinstance(a, ast.Constant) and isinstance(a.value, str))]
    return non_str[-1] if non_str else args[-1]


def _str_const(node: ast.AST) -> Optional[str]:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _find_table_and_column(node: ast.AST) -> Tuple[Optional[str], Optional[str]]:
    """Locate a res_* table and its column anywhere inside ``node``."""
    table = col = None
    for sub in ast.walk(node):
        # net.res_xxx  (attribute) or net["res_xxx"] (subscript)
        if isinstance(sub, ast.Attribute) and sub.attr in _RES_TABLES and table is None:
            table = sub.attr
        if isinstance(sub, ast.Subscript):
            s = _str_const(sub.slice)
            if s in _RES_TABLES and table is None:
                table = s
            # ow.output["table.col"] form
            if s and "." in s and s.split(".")[0] in _RES_TABLES:
                table, col = s.split(".", 1)
    # column: string subscript on the table, or attribute access after table,
    # or the column element of a ``.at[idx, "col"]`` / ``.loc[idx, "col"]``.
    for sub in ast.walk(node):
        if isinstance(sub, ast.Subscript):
            s = _str_const(sub.slice)
            if s and s not in _RES_TABLES and "." not in s:
                base = sub.value
                if (isinstance(base, ast.Attribute) and base.attr == table) or \
                   (isinstance(base, ast.Subscript) and _str_const(base.slice) == table):
                    col = col or s
            # .at[idx, "col"] / .loc[idx, "col"] : column is the 2nd tuple element
            if isinstance(sub.value, ast.Attribute) and sub.value.attr in ("at", "loc") \
                    and isinstance(sub.slice, ast.Tuple) and len(sub.slice.elts) >= 2:
                cell = _str_const(sub.slice.elts[1])
                if cell:
                    col = col or cell
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Attribute):
            if sub.value.attr == table and sub.attr not in _AGG_METHODS \
               and sub.attr not in ("at", "loc", "iloc"):
                col = col or sub.attr
    return table, col


def _aggregation(node: ast.AST) -> Optional[str]:
    """Outermost aggregation / accessor of the extraction expression."""
    # method call: X.max(), X.idxmax(), ...
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        m = node.func.attr
        if m in _AGG_METHODS:
            return _AGG_METHODS[m]
    # subscript accessor: .at[..] / .loc[..] / .iloc[..] / [scalar]
    if isinstance(node, ast.Subscript):
        base = node.value
        if isinstance(base, ast.Attribute) and base.attr in ("at", "loc"):
            return "point"
        if isinstance(base, ast.Attribute) and base.attr == "iloc":
            return "point_positional"
        return "point"
    # nested (e.g. .max().max() for time-series frames)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        inner = _aggregation(node.func.value)
        if inner:
            return inner
    return None


def _is_derived(node: ast.AST) -> bool:
    """True if the extraction is a non-trivial derived expression (BinOp,
    abs()/round() wrapper over differences, positional ts indexing) -> judge."""
    if isinstance(node, ast.BinOp):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in ("abs", "round"):
        # abs(a - b): derived; abs(net.res...max()) alone is not
        for a in node.args:
            if isinstance(a, ast.BinOp):
                return True
    # positional ts indexing ow.output[...].iloc[i, j]
    if _aggregation(node) == "point_positional":
        return True
    # ``series.idxmax()`` as the OUTERMOST call is a direct argmax/argmin -- it is
    # the queried aggregation itself, not a derived idiom (N5).
    if _aggregation(node) in ("argmax", "argmin"):
        return False
    # derived index: .loc[series.idxmin(), col] reads the min/max *via* an argmin
    # -- mathematically an aggregation, but a non-trivial idiom -> judge / N5
    # equivalence normalisation.
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                and sub.func.attr in ("idxmin", "idxmax"):
            return True
    return False


def parse_extraction(code: str) -> Dict[str, Any]:
    """Parse the result-extraction expression of a benchmark/model script.

    Returns {parse_ok, table, column, aggregation, index, derived, raw_expr}.
    """
    out: Dict[str, Any] = {"parse_ok": False, "table": None, "column": None,
                           "aggregation": None, "index": None, "derived": False,
                           "raw_expr": None}
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        out["error"] = f"SyntaxError: {e}"
        return out
    expr = _final_print_expr(tree)
    if expr is None:
        out["error"] = "no print() found"
        return out
    if isinstance(expr, ast.Name):
        resolved = _resolve_name(tree, expr.id)
        if resolved is not None:
            expr = resolved
    out["parse_ok"] = True
    try:
        out["raw_expr"] = ast.unparse(expr)[:240]
    except Exception:
        pass
    if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float, bool)):
        out["literal_constant"] = expr.value       # bare literal -> check (5) v1
        return out
    out["derived"] = _is_derived(expr)
    out["table"], out["column"] = _find_table_and_column(expr)
    out["aggregation"] = _aggregation(expr)
    # index literal inside .at[idx, ...] / [idx]
    for sub in ast.walk(expr):
        if isinstance(sub, ast.Subscript):
            sl = sub.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, (int, str)):
                if sl.value not in _RES_TABLES:
                    out["index"] = sl.value
            elif isinstance(sl, ast.Tuple) and sl.elts:
                first = sl.elts[0]
                if isinstance(first, ast.Constant):
                    out["index"] = first.value
    return out


def _scan_hardcoded_gt(code: str, gt: Any) -> Dict[str, Any]:
    """Static v1: is the ground-truth value a literal in the extraction path?"""
    ev: Dict[str, Any] = {"bare_literal": False, "literal_near_gt": False}
    parsed = parse_extraction(code)
    lit = parsed.get("literal_constant")
    try:
        gt_f = float(gt)
    except (TypeError, ValueError):
        gt_f = None
    if lit is not None and gt_f is not None:
        ev["bare_literal"] = _num_eq(float(lit), gt_f)
        ev["bare_literal_value"] = lit
    # any numeric literal in the code equal to GT (weaker signal)
    if gt_f is not None:
        try:
            for n in ast.walk(ast.parse(code)):
                if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) \
                        and not isinstance(n.value, bool):
                    if _num_eq(float(n.value), gt_f):
                        ev["literal_near_gt"] = True
                        break
        except SyntaxError:
            pass
    return ev


# ==========================================================================
# (4) N5 equivalence-pattern table + extraction normaliser.
#
# Each pattern recognises a mathematically-equivalent extraction idiom and
# normalises it to a canonical (table, column, aggregation) triple, so an
# equivalent gen/ref pair compares equal instead of being flagged as a derived
# expression (E1 doc s8 N5, extending the frozen equivalence case #4).  The
# table is data (not scattered comments) so the set of accepted equivalences is
# auditable in one place.
# ==========================================================================
EQUIV_PATTERNS: List[Dict[str, str]] = [
    {"name": "direct",
     "desc": "plain aggregation on a result table (argmax/argmin included)"},
    {"name": "argfn_index_is_minmax",
     "desc": ".loc[col.idxmin(), col] / .at[col.idxmax()] == col.min()/.max()"},
    {"name": "from_plus_to_is_ploss",
     "desc": "(res_line.p_from_mw + res_line.p_to_mw).sum() == res_line.pl_mw.sum()"},
    {"name": "abs_diff_symmetry",
     "desc": "abs(a - b) == abs(b - a) when both operands reduce to triples"},
]


def _pat_argfn_index(expr: ast.AST) -> Optional[Dict[str, Any]]:
    """min/max obtained via ``.loc``/``.at`` indexed by ``col.idxmin/idxmax()``."""
    if not isinstance(expr, ast.Subscript):
        return None
    base = expr.value
    if not (isinstance(base, ast.Attribute) and base.attr in ("loc", "at")):
        return None
    elts = expr.slice.elts if isinstance(expr.slice, ast.Tuple) else [expr.slice]
    idx_call = None
    col = None
    for e in elts:
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) \
                and e.func.attr in ("idxmin", "idxmax"):
            idx_call = e
        elif isinstance(e, ast.Constant) and isinstance(e.value, str):
            col = col or e.value
    if idx_call is None:
        return None
    agg = "min" if idx_call.func.attr == "idxmin" else "max"
    table, tcol = _find_table_and_column(base.value)
    if col is None:
        col = tcol
    if table is None:
        table, ccol = _find_table_and_column(idx_call)
        col = col or ccol
    if table is None:
        return None
    return {"table": table, "column": col, "aggregation": agg,
            "pattern": "argfn_index_is_minmax"}


def _pat_from_plus_to(expr: ast.AST) -> Optional[Dict[str, Any]]:
    """Total loss via ``(p_from_mw + p_to_mw).sum()`` == ``pl_mw.sum()``."""
    if not (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add)):
        return None
    tables, cols, aggs = set(), set(), set()
    for operand in (expr.left, expr.right):
        t, c = _find_table_and_column(operand)
        a = _aggregation(operand)
        if t:
            tables.add(t)
        if c:
            cols.add(c)
        if a:
            aggs.add(a)
    if tables == {"res_line"} and cols == {"p_from_mw", "p_to_mw"} \
            and aggs <= {"sum"}:
        return {"table": "res_line", "column": "pl_mw", "aggregation": "sum",
                "pattern": "from_plus_to_is_ploss"}
    return None


def _pat_abs_diff(expr: ast.AST, tree: ast.Module) -> Optional[Dict[str, Any]]:
    """``abs(a - b)`` canonicalised to a symmetric form -- only when *both*
    operands reduce to comparable extraction triples (otherwise a signed vs
    abs, or intermediate-variable, difference is left for the judge)."""
    if not (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
            and expr.func.id == "abs" and len(expr.args) == 1
            and isinstance(expr.args[0], ast.BinOp)
            and isinstance(expr.args[0].op, ast.Sub)):
        return None
    parts = []
    for operand in (expr.args[0].left, expr.args[0].right):
        node = operand
        if isinstance(operand, ast.Name):
            rv = _resolve_name(tree, operand.id)
            if rv is not None:
                node = rv
        t, c = _find_table_and_column(node)
        a = _aggregation(node)
        if t is None:
            return None
        parts.append((t, c, a))
    parts.sort(key=lambda p: (p[0] or "", p[1] or "", p[2] or ""))
    return {"table": f"absdiff[{parts[0]}|{parts[1]}]", "column": None,
            "aggregation": "abs_symmetric_diff", "pattern": "abs_diff_symmetry"}


def _normalize_equiv(code: str) -> Optional[Dict[str, Any]]:
    """Reduce an extraction to a canonical {table, column, aggregation, pattern}
    triple via a frozen equivalence pattern, or ``None`` if none apply."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    expr = _final_print_expr(tree)
    if expr is None:
        return None
    if isinstance(expr, ast.Name):
        rv = _resolve_name(tree, expr.id)
        if rv is not None:
            expr = rv
    res = _pat_argfn_index(expr)
    if res:
        return res
    res = _pat_from_plus_to(expr)
    if res:
        return res
    table, col = _find_table_and_column(expr)
    agg = _aggregation(expr)
    if table and agg and not _is_derived(expr):
        return {"table": table, "column": col, "aggregation": agg,
                "pattern": "direct"}
    res = _pat_abs_diff(expr, tree)
    if res:
        return res
    return None


def _n1_extract_target(g: Dict[str, Any], r: Dict[str, Any],
                       query_type: str) -> Dict[str, Any]:
    """N1 extraction-target comparison (model computes the queried element vs
    reference hard-codes it).  Evidence only; the verdict is the caller's."""
    g_idx, r_idx = g.get("index"), r.get("index")
    g_agg, r_agg = g.get("aggregation"), r.get("aggregation")
    dyn = {"argmax", "argmin"}
    g_dynamic = g_agg in dyn or (g_agg == "point" and g_idx is None)
    r_dynamic = r_agg in dyn or (r_agg == "point" and r_idx is None)
    both_dynamic = g_dynamic and r_dynamic
    applicable = (g_idx != r_idx) or (g_dynamic != r_dynamic) \
        or (g_agg in dyn and r_agg in dyn)
    if not applicable:
        return {"applicable": False}

    def _label(idx, agg, is_dyn):
        # an argmax/argmin computes its target at run time; ignore any spurious
        # column-string index the parser may have captured for it.
        if agg in dyn:
            return f"dynamic({agg})"
        return idx if idx is not None else f"dynamic({agg})"

    return {
        "applicable": True,
        "ref_target": _label(r_idx, r_agg, r_dynamic),
        "gen_target": _label(g_idx, g_agg, g_dynamic),
        "both_dynamic": both_dynamic,
        "conclusion": (
            "both sides compute the target dynamically by the same rule "
            "(equivalent iff the rule matches; N1)" if both_dynamic else
            "model computes the target dynamically while the reference "
            "hard-codes it (or vice versa); the runtime index is not captured "
            "here -> judge (N1)"),
    }


# ==========================================================================
# Check (1): network + modification consistency
# ==========================================================================
def _counts(sig: dict) -> Dict[str, int]:
    md = (sig or {}).get("metadata") or {}
    ex = (sig or {}).get("extra_counts") or {}
    c = {k: md.get(k) for k in ("n_bus", "n_line", "n_trafo", "n_gen", "n_load")}
    c["n_sgen"] = (ex or {}).get("n_sgen", md.get("n_sgen"))
    c["n_shunt"] = (ex or {}).get("n_shunt")
    c["n_storage"] = (ex or {}).get("n_storage")
    return c


def _fp_diff(gen_fp: dict, ref_fp: dict) -> List[str]:
    """Names of (table.column) fingerprints that differ between gen and ref.

    Digest entries are compared per (index, value) -- via the full ``values``
    dict when captured (small tables) and the order-independent ``vhash``
    content hash always -- not only by the (sum, count) aggregate.  The
    aggregate-only comparison was checker defect D-5: the same value written at
    a different index left the sum unchanged and the divergence invisible.
    """
    diffs = []
    keys = set(gen_fp or {}) | set(ref_fp or {})
    for k in sorted(keys):
        g, r = (gen_fp or {}).get(k), (ref_fp or {}).get(k)
        if g == r:
            continue
        if g is None or r is None:
            diffs.append(k)
            continue
        kind = r.get("kind")
        if kind == "outset":
            if set(g.get("out", [])) != set(r.get("out", [])):
                diffs.append(k)
        elif kind == "nondefault":
            if g.get("values") != r.get("values"):
                diffs.append(k)
        else:  # digest
            if g.get("sum") != r.get("sum") or g.get("n") != r.get("n"):
                diffs.append(k)
            elif g.get("values") is not None and r.get("values") is not None \
                    and g.get("values") != r.get("values"):
                diffs.append(k)          # D-5: same aggregate, different index
            elif g.get("vhash") is not None and r.get("vhash") is not None \
                    and g.get("vhash") != r.get("vhash"):
                diffs.append(k)          # D-5, tables too large for a values dict
    return diffs


def _fp_targets(entry: Optional[dict]) -> Optional[List[int]]:
    """The set of element indices a fingerprint entry marks as modified."""
    if not entry:
        return None
    kind = entry.get("kind")
    if kind == "outset":
        return sorted(entry.get("out", []))
    if kind == "nondefault":
        return sorted(int(i) for i in (entry.get("values") or {}))
    if kind == "digest":
        v = entry.get("values")
        return sorted(int(i) for i in v) if v else None
    return None


def _n1_mod_target(fp_diffs: List[str], gen_fp: dict,
                   ref_fp: dict) -> Dict[str, Any]:
    """N1 modification-target comparison: which elements gen vs ref modify.
    Evidence for the dual-judge -- a dynamic-computed target (e.g. the most-
    loaded line) vs a hard-coded one shows up here as a target-index divergence."""
    cols: Dict[str, Any] = {}
    for k in fp_diffs:
        gt = _fp_targets((gen_fp or {}).get(k))
        rt = _fp_targets((ref_fp or {}).get(k))
        cols[k] = {"gen_targets": gt, "ref_targets": rt,
                   "same_target": gt is not None and gt == rt}
    # this helper is only called on columns that already diverge; if every
    # diverging column targets the same element index, the divergence is a
    # modification VALUE/magnitude difference on the same target rather than a
    # target-selection (dynamic vs hard-coded) difference.
    same = all(v["same_target"] for v in cols.values()) if cols else None
    return {"columns": cols, "same_target_all": same,
            "conclusion": (
                "gen and ref target the same element(s) but the fingerprint "
                "still diverges -> modification value/magnitude differs on the "
                "same target; judge" if same else
                "gen and ref modify different element targets (dynamic-computed "
                "vs hard-coded target possible; N1) -> judge")}


def _best_matching_net(nets: dict,
                       ref_fp: dict) -> Optional[Tuple[str, dict, int]]:
    """Among namespace nets, the one whose fingerprint is closest to ``ref_fp``
    (fewest diffs).  Fallback net-candidate selection for the compound families
    when no pre-solve snapshot is available (E1 doc s8 N6)."""
    best = None
    for var, sig in (nets or {}).items():
        fp = (sig.get("state") or {}).get("mod_fingerprint", {})
        ndiff = len(_fp_diff(fp, ref_fp))
        if best is None or ndiff < best[2]:
            best = (var, fp, ndiff)
    return best


def _mod_fp_for_check1(task: str, gen_exec: dict, ref_exec: dict, gen_net: dict,
                       ref_net: dict) -> Tuple[dict, dict, str, Optional[dict]]:
    """(gen_fp, ref_fp, source, selection) for the modification diff.

    Compound families (comparison/comparison_opf/time_series) use the first
    pre-solve snapshot (N6/L1); if unavailable, the namespace net best matching
    the reference; otherwise every family uses the post-exec net state."""
    if task in _C1_PRESOLVE:
        g_snaps = gen_exec.get("presolve_fingerprints") or []
        r_snaps = ref_exec.get("presolve_fingerprints") or []
        if g_snaps and r_snaps:
            return (g_snaps[0]["fingerprint"], r_snaps[0]["fingerprint"],
                    "pre_solve_first_snapshot",
                    {"basis": "first pre-solve snapshot (compound family)",
                     "gen_snaps": len(g_snaps), "ref_snaps": len(r_snaps)})
        ref_fp = ref_net["state"].get("mod_fingerprint", {})
        best = _best_matching_net(gen_exec.get("nets") or {}, ref_fp)
        if best is not None:
            var, fp, ndiff = best
            return (fp, ref_fp, "best_matching_namespace_net",
                    {"basis": "best-matching namespace net (no pre-solve "
                              "snapshot)", "selected_var": var,
                     "n_fingerprint_diffs": ndiff})
    return (gen_net["state"].get("mod_fingerprint", {}),
            ref_net["state"].get("mod_fingerprint", {}),
            "post_exec_net", None)


def check1_network_mods(task: str, gen_exec: dict,
                        ref_exec: dict) -> Dict[str, Any]:
    gen_net, ref_net = gen_exec.get("primary_net"), ref_exec.get("primary_net")
    ev: Dict[str, Any] = {"net_recovered": gen_net is not None,
                          "ref_net_recovered": ref_net is not None,
                          "degraded": False}
    if ref_net is None:
        return {"status": AMBIG, "evidence": {**ev,
                "note": "reference net unavailable; cannot diff"}}
    if gen_net is None:
        # degraded AST fallback (GAP-5): loader name presence only
        ev["degraded"] = True
        return {"status": AMBIG, "evidence": {**ev,
                "note": "generated net not recovered; degrade to AST/judge"}}

    gc_, rc_ = _counts(gen_net), _counts(ref_net)
    ev["gen_counts"], ev["ref_counts"] = gc_, rc_
    gmd, rmd = gen_net.get("metadata") or {}, ref_net.get("metadata") or {}
    bus_match = set(gmd.get("bus_indices", [])) == set(rmd.get("bus_indices", []))
    line_match = set(gmd.get("line_indices", [])) == set(rmd.get("line_indices", []))
    counts_match = all(gc_.get(k) == rc_.get(k) for k in gc_)
    ev["bus_index_match"], ev["line_index_match"] = bus_match, line_match

    gen_fp, ref_fp, fp_source, selection = _mod_fp_for_check1(
        task, gen_exec, ref_exec, gen_net, ref_net)
    ev["mod_fp_source"] = fp_source
    if selection:
        ev["net_selection"] = selection
    fp_diffs = _fp_diff(gen_fp, ref_fp)
    ev["fingerprint_diffs"] = fp_diffs
    ev["fingerprint_detail"] = {k: {"gen": (gen_fp or {}).get(k),
                                    "ref": (ref_fp or {}).get(k)}
                                for k in fp_diffs}
    if fp_diffs:
        ev["n1_mod_target"] = _n1_mod_target(fp_diffs, gen_fp, ref_fp)

    if bus_match and line_match and counts_match:
        if not fp_diffs:
            return {"status": PASS, "evidence": ev}
        # right base network, but modifications diverge -> judge
        presolve_tag = (" (pre-solve fingerprint)"
                        if fp_source == "pre_solve_first_snapshot" else "")
        return {"status": AMBIG, "evidence": {**ev,
                "note": "base network matches; modification state diverges "
                        f"on {fp_diffs}{presolve_tag}"}}

    # structural mismatch: distinguish spurious/omitted addable element from a
    # wrong base network.
    g_bus, r_bus = set(gmd.get("bus_indices", [])), set(rmd.get("bus_indices", []))
    g_line, r_line = set(gmd.get("line_indices", [])), set(rmd.get("line_indices", []))
    subset_like = (g_bus >= r_bus or r_bus >= g_bus) and \
                  (g_line >= r_line or r_line >= g_line)
    if subset_like:
        return {"status": AMBIG, "evidence": {**ev,
                "note": "structural counts differ by an added/omitted element "
                        "(subset-like); judge"}}
    return {"status": FAIL, "evidence": {**ev,
            "note": "base network topology differs (bus/line index sets); "
                    "wrong network loaded"}}


# ==========================================================================
# (2) N2 hand-written-loop equivalence criteria (evidence for the dual-judge).
#
# When a canonical multi-network engine (run_contingency / run_timeseries) is
# replaced by a hand-written per-step ``runpp`` loop, N2 makes equivalence
# conditional on four criteria all being verifiable/satisfied; the automatic
# layer surfaces their checkable status (per-step solver identity, scenario set,
# non-convergence handling, aggregation semantics) and leaves the ruling to the
# judge (E1 doc s8 N2).
# ==========================================================================
_PF_CALL_NAMES = {"runpp", "rundcpp", "runopp", "calc_sc"}


def _call_names_in(node: ast.AST) -> set:
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Attribute):
                names.add(f.attr)
            elif isinstance(f, ast.Name):
                names.add(f.id)
    return names


def _solver_loops(tree: ast.Module) -> List[ast.AST]:
    return [n for n in ast.walk(tree)
            if isinstance(n, (ast.For, ast.While))
            and (_call_names_in(n) & _PF_CALL_NAMES)]


def _contains_break(loop: ast.AST) -> bool:
    return any(isinstance(s, ast.Break) for s in ast.walk(loop))


def _loop_assigns(loop: ast.AST, attrs: tuple) -> bool:
    for sub in ast.walk(loop):
        if isinstance(sub, (ast.Assign, ast.AugAssign)):
            tgts = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
            for t in tgts:
                for a in ast.walk(t):
                    if isinstance(a, ast.Attribute) and a.attr in attrs:
                        return True
                    if isinstance(a, ast.Constant) and a.value in attrs:
                        return True
    return False


def _handles_convergence(loop: ast.AST) -> bool:
    for sub in ast.walk(loop):
        if isinstance(sub, ast.Try):
            return True
        if isinstance(sub, ast.Attribute) and sub.attr == "converged":
            return True
        if isinstance(sub, ast.Subscript) and _str_const(sub.slice) == "converged":
            return True
    return False


def _n2_loop_criteria(gen_code: str, aggregation: Optional[str]) -> Dict[str, Any]:
    """Report the four N2 criteria for a hand-written solver loop (evidence)."""
    try:
        tree = ast.parse(gen_code)
    except SyntaxError:
        return {"parse_ok": False}
    loops = _solver_loops(tree)
    out: Dict[str, Any] = {"parse_ok": True,
                           "has_manual_solver_loop": bool(loops)}
    if not loops:
        return out
    early_break = any(_contains_break(l) for l in loops)
    in_loop_profile = any(_loop_assigns(l, ("p_mw", "q_mvar", "scaling"))
                          for l in loops)
    in_loop_toggle = any(_loop_assigns(l, ("in_service",)) for l in loops)
    nonconv = any(_handles_convergence(l) for l in loops)
    out["criteria"] = {
        "c1_per_step_solver_identity": {
            "gen_step_solver": "ac_pf (runpp)", "verifiable": True,
            "note": "per-step physics is AC power flow; must match the "
                    "reference engine's per-step physics"},
        "c2_scenario_set": {
            "early_break": early_break, "in_loop_element_toggle": in_loop_toggle,
            "in_loop_profile_assignment": in_loop_profile,
            "satisfied": (not early_break),
            "note": ("early break truncates the scenario/step set -> criterion "
                     "NOT satisfied (N2 (2))" if early_break else
                     "no early break; full enumeration plausible -- profile/"
                     "scenario application method still to verify (N2 (2))")},
        "c3_nonconvergence_handling": {
            "explicit_handling": nonconv,
            "note": "whether non-convergent steps are handled without changing "
                    "the queried quantity (N2 (3))"},
        "c4_aggregation_semantics": {
            "aggregation": aggregation,
            "note": "aggregation over steps must match the reference (N2 (4))"},
    }
    return out


# ==========================================================================
# Check (2): expected solver invoked and converged
# ==========================================================================
def check2_solver_identity(task: str, gen_exec: dict, ref_exec: dict,
                           gen_code: str = "") -> Dict[str, Any]:
    spec = SOLVER_IDENTITY.get(task, {})
    gen_ms = _identity_multiset(gen_exec.get("solver_trace", []))
    ref_ms = _identity_multiset(ref_exec.get("solver_trace", []))
    ev: Dict[str, Any] = {"gen_identities": gen_ms, "ref_identities": ref_ms,
                          "family_expected": spec.get("identities")}

    # D-(1): flag runpp(method=...) -- a silently-dropped DC request that still
    # runs AC.  The identity above already registers it as ac_pf; this makes the
    # no-op explicit as evidence for the dual-judge (E1 doc s8 D-(1)).
    noop = _noop_solver_kwargs(gen_exec.get("solver_trace", []))
    if noop:
        ev["noop_kwargs"] = noop

    # authoritative expectation = reference-derived identities (excluding the
    # generic prereq ac_pf for families whose headline solver is elsewhere).
    ref_present = {i for i, d in ref_ms.items() if d["total"] > 0
                   and not i.startswith("other:")}
    expected = spec.get("identities")
    if expected == "reference_derived":
        expected = sorted(ref_present) or []
        ev["family_expected"] = f"reference_derived={expected}"
    expected = list(expected or [])
    # cross-check: reference should exhibit the family's expected identities
    ev["reference_supports_expected"] = all(
        e in ref_present or (e == "ac_pf" and "dc_pf" in ref_present) for e in expected) \
        if expected and ref_present else None

    if not gen_exec.get("solver_trace") and not gen_exec.get("success"):
        return {"status": AMBIG, "evidence": {**ev,
                "note": "generated code did not execute; solver trace empty"}}

    reasons: List[str] = []
    status = PASS
    for ident in expected:
        got = gen_ms.get(ident)
        # DC/AC swap detection
        if ident == "dc_pf" and not got and gen_ms.get("ac_pf"):
            reasons.append("expected DC power flow but AC runpp used (physics "
                           "model changed)")
            status = FAIL
            continue
        if ident == "ac_pf" and not got and gen_ms.get("dc_pf"):
            reasons.append("expected AC power flow but DC solver used")
            status = FAIL
            continue
        # short-circuit fault mismatch
        if ident.startswith("sc_") and not got:
            other_sc = [i for i in gen_ms if i.startswith("sc_")]
            if other_sc:
                reasons.append(f"expected {ident} but {other_sc} used "
                               "(different fault physics)")
                status = FAIL
                continue
        if not got:
            # canonical engine replaced by a manual runpp loop?
            engine = spec.get("engine")
            if engine and ident in ("contingency", "timeseries") \
                    and gen_ms.get("ac_pf", {}).get("total", 0) >= 1:
                reasons.append(f"canonical {ident} engine replaced by manual "
                               "runpp loop (possible equivalent-path)")
                if status != FAIL:
                    status = AMBIG
                continue
            reasons.append(f"expected solver identity {ident} not invoked")
            status = FAIL
            continue
        if got["ok"] == 0:
            reasons.append(f"{ident} invoked but did not converge")
            status = FAIL

    # min_calls (comparison families need baseline + modified)
    min_calls = spec.get("min_calls", 1)
    if status == PASS and expected and min_calls > 1:
        primary = expected[0]
        n = gen_ms.get(primary, {}).get("total", 0)
        if n < min_calls:
            reasons.append(f"{primary} invoked {n}x, expected >= {min_calls} "
                           "(baseline+modified)")
            status = AMBIG

    ev["reasons"] = reasons
    # N2: for the multi-network-engine families, surface the hand-written-loop
    # equivalence criteria whenever a manual per-step solver loop replaced the
    # canonical engine (evidence for the dual-judge; verdict stays as decided).
    if spec.get("engine") and gen_code:
        n2 = _n2_loop_criteria(gen_code, parse_extraction(gen_code).get("aggregation"))
        if n2.get("has_manual_solver_loop"):
            ev["n2_equivalence"] = n2
    return {"status": status, "evidence": ev}


# ==========================================================================
# Check (3): power-balance residual + task-relevant sanity
# ==========================================================================
def check3_physical_sanity(task: str, gen_exec: dict) -> Dict[str, Any]:
    net = gen_exec.get("primary_net")
    ev: Dict[str, Any] = {"net_recovered": net is not None, "subchecks": {}}
    if net is None:
        return {"status": AMBIG, "evidence": {**ev,
                "note": "net not recovered; N/A-pass-with-flag (judge)"}}
    bal = net["state"].get("balance", {})
    state = net["state"]
    sub = ev["subchecks"]
    status = PASS

    def fail(msg):
        nonlocal status
        status = FAIL
        ev.setdefault("reasons", []).append(msg)

    # -- power balance residual --
    if task in _C3_RESIDUAL_AC or task in _C3_RESIDUAL_DC or task in _C3_OPF:
        inj = sum(x for x in (bal.get("p_gen"), bal.get("p_sgen"),
                              bal.get("p_ext_grid")) if x is not None)
        load = bal.get("p_load")
        # consumption beyond load: line/trafo losses + shunt/storage/ward draw
        loss = sum(x for x in (bal.get("p_loss_line"), bal.get("p_loss_trafo"),
                               bal.get("p_shunt"), bal.get("p_storage"),
                               bal.get("p_ward")) if x is not None)
        if load is not None and (bal.get("p_gen") is not None
                                 or bal.get("p_ext_grid") is not None):
            resid = inj - load - loss
            tol = max(RESIDUAL_ABS_TOL, RESIDUAL_REL_TOL * abs(load))
            sub["power_balance"] = {"injection": round(inj, 4),
                                    "load": round(load, 4),
                                    "loss": round(loss, 4),
                                    "residual": round(resid, 4), "tol": round(tol, 4)}
            if abs(resid) > tol:
                fail(f"power-balance residual {resid:.3f} MW exceeds tol {tol:.3f}")
        else:
            sub["power_balance"] = "inputs unavailable"
            if status == PASS:
                status = AMBIG

    # -- voltage sanity --
    if task in _C3_VOLTAGE:
        vmin, vmax = state.get("vm_pu_min"), state.get("vm_pu_max")
        sub["voltage"] = {"vm_min": vmin, "vm_max": vmax}
        if vmin is not None and vmax is not None:
            if vmin < VM_HARD_LO or vmax > VM_HARD_HI:
                fail(f"bus voltages implausible (min={vmin}, max={vmax})")
            elif vmin < VM_SOFT_LO or vmax > VM_SOFT_HI:
                sub["voltage"]["outside_nominal"] = True

    # -- line loading finite --
    if task in _C3_LOADING:
        lmax = bal.get("loading_line_max")
        sub["loading_line_max"] = lmax
        if lmax is not None and (math.isnan(lmax) or math.isinf(lmax) or lmax < 0):
            fail(f"line loading not finite/non-negative ({lmax})")

    # -- OPF feasibility --
    if task in _C3_OPF:
        opf_conv = state.get("OPF_converged")
        sub["opf_converged"] = opf_conv
        sub["res_cost"] = bal.get("res_cost")
        if opf_conv is not True:
            fail(f"OPF not converged (OPF_converged={opf_conv})")

    # -- short-circuit existence --
    if task in _C3_SC:
        present = "res_bus_sc" in state.get("res_tables_present", [])
        sub["res_bus_sc_present"] = present
        sub["ikss_max"] = bal.get("ikss_max")
        if not present:
            fail("short-circuit result table res_bus_sc absent")
        elif bal.get("ikss_max") is not None and bal["ikss_max"] <= 0:
            fail("short-circuit current non-positive")

    # -- contingency sanity --
    if task in _C3_CONTINGENCY:
        ml = bal.get("max_loading_line_max")
        mv = bal.get("min_vm_bus_min")
        sub["contingency"] = {"max_loading_line_max": ml, "min_vm_bus_min": mv}
        # manual-loop implementations may not populate the envelope columns;
        # only fail on a present-but-broken value.
        for name, val in (("max_loading", ml), ("min_vm", mv)):
            if val is not None and (math.isnan(val) or math.isinf(val)):
                fail(f"contingency {name} not finite ({val})")

    return {"status": status, "evidence": ev}


# ==========================================================================
# Check (4): extraction-expression alignment
# ==========================================================================
def check4_extraction(query_type: str, gen_code: str,
                      ref_code: str) -> Dict[str, Any]:
    g, r = parse_extraction(gen_code), parse_extraction(ref_code)
    ev: Dict[str, Any] = {"gen": g, "ref": r, "query_type": query_type}
    # query_type <-> reference aggregation self-consistency (evidence only)
    ev["query_type_consistent"] = _qtype_agg_consistent(query_type, r.get("aggregation"))

    if not g.get("parse_ok") or not r.get("parse_ok"):
        return {"status": AMBIG, "evidence": {**ev,
                "note": "extraction expression not parseable; judge"}}

    # N5: normalise equivalent idioms to a canonical (table, column, agg) triple
    # so an equivalent derived expression compares equal instead of being flagged.
    g_norm, r_norm = _normalize_equiv(gen_code), _normalize_equiv(ref_code)
    ev["gen_normalized"], ev["ref_normalized"] = g_norm, r_norm
    # N1: extraction-target comparison (evidence only; verdict below).
    ev["n1_target"] = _n1_extract_target(g, r, query_type)

    # Short-circuit to PASS only when an equivalence *idiom* was actually applied
    # (a derived expression normalised to the same triple).  Two plain "direct"
    # extractions still go through the index-aware comparison below, so the N1
    # dynamic-vs-hard-coded index case is not swallowed by triple equality.
    if g_norm and r_norm and not (g_norm["pattern"] == "direct"
                                  and r_norm["pattern"] == "direct"):
        gt = (g_norm["table"], g_norm["column"], g_norm["aggregation"])
        rt = (r_norm["table"], r_norm["column"], r_norm["aggregation"])
        if gt == rt:
            return {"status": PASS, "evidence": {**ev,
                    "equiv_pattern": [g_norm["pattern"], r_norm["pattern"]],
                    "note": "extractions equivalent via N5 pattern "
                            f"{g_norm['pattern']}/{r_norm['pattern']}"}}

    if g.get("derived") or r.get("derived"):
        return {"status": AMBIG, "evidence": {**ev,
                "note": "non-trivial derived extraction expression; judge"}}
    if g.get("table") is None or r.get("table") is None:
        # extraction not reducible to a result table on one side (intermediate
        # variable, loop-accumulated value, unusual idiom) -> judge, not fail.
        return {"status": AMBIG, "evidence": {**ev,
                "note": "extraction not reducible to a result table on one "
                        "side; judge"}}
    reasons = []
    if g.get("table") != r.get("table"):
        reasons.append(f"table {g.get('table')} != {r.get('table')}")
    if g.get("column") != r.get("column"):
        reasons.append(f"column {g.get('column')} != {r.get('column')}")
    if g.get("aggregation") != r.get("aggregation"):
        reasons.append(f"aggregation {g.get('aggregation')} != {r.get('aggregation')}")
    ev["reasons"] = reasons
    if reasons:
        # table/column/agg divergence -> reads a different quantity
        return {"status": FAIL, "evidence": ev}
    # same table/column/agg but the model computes the point index dynamically
    # while the reference hard-codes it -> the runtime index is not captured
    # here, so its coincidence cannot be certified (N1); judge.
    if g.get("aggregation") == "point" and g.get("index") is None \
            and r.get("index") is not None:
        return {"status": AMBIG, "evidence": {**ev,
                "note": "same table/column/agg but model computes the index "
                        f"dynamically vs reference hard-coded index "
                        f"{r.get('index')} (N1); judge"}}
    # same table/column/agg; index divergence is often label-vs-position idiom
    if g.get("aggregation") == "point" and g.get("index") is not None \
            and r.get("index") is not None and g.get("index") != r.get("index"):
        return {"status": AMBIG, "evidence": {**ev,
                "note": f"same table/column/agg but index {g.get('index')} != "
                        f"{r.get('index')} (label/position idiom or wrong element)"}}
    return {"status": PASS, "evidence": ev}


def _qtype_agg_consistent(query_type: str, agg: Optional[str]) -> Optional[bool]:
    expect = {"point_value": "point", "max_value": "max", "min_value": "min",
              "total": "sum", "argmax": "argmax", "argmin": "argmin",
              "bool_check": "any", "count_violations": "count",
              "ts_max_value": "max", "ts_min_value": "min",
              "ts_point_at_step": "point_positional"}.get(query_type)
    if expect is None or agg is None:
        return None
    return expect == agg


# ==========================================================================
# Check (5): hard-code scan (v1) + perturbation re-execution (v2)
# ==========================================================================
def check5_provenance(task: str, query_type: str, gen_code: str, gt: Any,
                      gen_exec: dict, pert_exec: Optional[dict],
                      pert_skip_reason: Optional[str],
                      check2_status: Optional[str] = None,
                      check3_status: Optional[str] = None) -> Dict[str, Any]:
    v1 = _scan_hardcoded_gt(gen_code, gt)
    solver_called = bool(gen_exec.get("solver_trace"))
    ev: Dict[str, Any] = {"v1": v1, "solver_called": solver_called}

    # v2 perturbation
    v2: Dict[str, Any] = {"constructible": False}
    orig = _last_number(gen_exec.get("stdout", ""))
    v2["orig_output"] = orig
    if pert_skip_reason:
        v2["skip_reason"] = pert_skip_reason
    elif pert_exec is not None:
        pert_out = _last_number(pert_exec.get("stdout", ""))
        v2["perturbed_output"] = pert_out
        v2["perturbed_success"] = pert_exec.get("success")
        if orig is not None and pert_out is not None and pert_exec.get("success"):
            v2["constructible"] = True
            delta = abs(pert_out - orig)
            v2["delta"] = delta
            v2["sensitive"] = delta > max(PERTURB_MIN_ABS, PERTURB_MIN_REL * abs(orig))
        else:
            v2["skip_reason"] = "perturbed re-exec produced no comparable output"
    ev["v2"] = v2

    # -- ruling --
    # 1) bare hard-coded literal == GT  -> fail (high confidence)
    if v1.get("bare_literal"):
        return {"status": FAIL, "evidence": {**ev,
                "note": "printed value is a literal equal to ground truth"}}
    # 2) no solver at all + output == GT -> fail
    if not solver_called and _num_eq(orig, _to_float(gt)):
        return {"status": FAIL, "evidence": {**ev,
                "note": "no solver invoked yet output equals ground truth"}}
    # 3) perturbation moved the output -> genuinely simulated
    if v2.get("sensitive"):
        return {"status": PASS, "evidence": ev}
    # 4) perturbation invariant
    if v2.get("constructible") and v2.get("sensitive") is False:
        # N4 stiff exemption: an output that does not move under a load
        # perturbation is never pollution evidence.  If the expected solver ran
        # and converged (check 2) and the physical sanity holds (check 3), the
        # quantity is stiff/insensitive (slack-bus voltage, short-circuit
        # current, ...) -> auto-pass, do not send to judge (E1 doc s8 N4).
        if check2_status == PASS and check3_status == PASS:
            return {"status": PASS, "evidence": {**ev, "stiff_exempt": True,
                    "note": "perturbation-invariant output but expected solver "
                            "converged (check 2) and sanity passed (check 3) -> "
                            "stiff/insensitive quantity (N4 stiff-exempt)"}}
        if query_type in _DISCRETE_QTYPES or task in _C5_NO_PERTURB:
            # discrete/insensitive output: invariance uninformative
            if solver_called:
                return {"status": PASS, "evidence": {**ev,
                        "note": "discrete/insensitive output; perturbation "
                                "invariance uninformative; solver ran"}}
            return {"status": AMBIG, "evidence": {**ev,
                    "note": "invariant discrete output, no solver; judge"}}
        # continuous quantity that did not move: could be a stiff quantity
        # (slack-bus voltage, etc.) -> flag for judge, do not auto-fail
        return {"status": AMBIG, "evidence": {**ev,
                "note": "continuous output invariant under load perturbation "
                        "(possible stiff quantity or hard-code); judge"}}
    # 5) perturbation not constructible -> v1-only
    if solver_called and not v1.get("literal_near_gt"):
        return {"status": PASS, "evidence": {**ev,
                "note": "perturbation not constructible; solver ran and output "
                        "not a GT literal"}}
    if v1.get("literal_near_gt"):
        return {"status": AMBIG, "evidence": {**ev,
                "note": "a code literal equals GT and perturbation inconclusive; judge"}}
    return {"status": AMBIG, "evidence": {**ev, "note": "provenance inconclusive"}}


def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ==========================================================================
# Per-item orchestration
# ==========================================================================
def audit_item(record: dict, *, timeout: int = DEFAULT_TIMEOUT,
               do_perturb: bool = True) -> Dict[str, Any]:
    """Run the T1 re-executions and the five checks on one matched record."""
    task = record.get("task")
    qtype = record.get("query_type")
    gen_code = record.get("generated_code") or ""
    ref_code = record.get("reference_code") or ""
    gt = record.get("ground_truth")

    ref_exec = sandbox.run_instrumented(ref_code, timeout=timeout)
    gen_exec = sandbox.run_instrumented(gen_code, timeout=timeout)

    # per-item environment sub-gate: does reference reproduce ground truth?
    ref_out = _last_number(ref_exec.get("stdout", ""))
    ref_reproduces = _num_eq(ref_out, _to_float(gt))

    # perturbation exec (check (5) v2), skipping the largest nets / ts family
    pert_exec = None
    pert_skip = None
    gen_net = gen_exec.get("primary_net")
    n_bus = ((gen_net or {}).get("metadata") or {}).get("n_bus")
    if not do_perturb:
        pert_skip = "perturbation disabled"
    elif task in _C5_NO_PERTURB:
        pert_skip = "time-series controller overrides load perturbation"
    elif n_bus is not None and n_bus > LARGE_NET_BUS:
        pert_skip = f"large network (n_bus={n_bus}) skipped for OOM discipline"
    elif not gen_exec.get("success"):
        pert_skip = "generated code did not execute successfully"
    else:
        pert_exec = sandbox.run_instrumented(gen_code, timeout=timeout,
                                             perturb_factor=PERTURB_FACTOR)

    # check2 / check3 are computed before check5 so their verdicts can drive the
    # N4 stiff exemption (perturbation-invariant output + solver converged +
    # sanity passed -> stiff-exempt pass).
    c1 = check1_network_mods(task, gen_exec, ref_exec)
    c2 = check2_solver_identity(task, gen_exec, ref_exec, gen_code)
    c3 = check3_physical_sanity(task, gen_exec)
    c4 = check4_extraction(qtype, gen_code, ref_code)
    c5 = check5_provenance(task, qtype, gen_code, gt, gen_exec, pert_exec,
                           pert_skip, check2_status=c2["status"],
                           check3_status=c3["status"])
    checks = {
        "check1_network_mods": c1,
        "check2_solver_identity": c2,
        "check3_physical_sanity": c3,
        "check4_extraction": c4,
        "check5_provenance": c5,
    }

    statuses = [c["status"] for c in checks.values()]
    item_result = {
        "checks": checks,
        "check_status": {k: v["status"] for k, v in checks.items()},
        "any_fail": any(s == FAIL for s in statuses),
        "any_flag": any(s in (FAIL, AMBIG) for s in statuses),
        "exec": {
            "ref_success": ref_exec.get("success"),
            "ref_reproduces_gt": ref_reproduces,
            "ref_output": ref_out,
            "gen_success": gen_exec.get("success"),
            "gen_error_type": gen_exec.get("error_type"),
            "gen_output": _last_number(gen_exec.get("stdout", "")),
            "gen_net_recovered": gen_net is not None,
            "gen_n_bus": n_bus,
            "perturb_run": pert_exec is not None,
            "perturb_skip_reason": pert_skip,
        },
    }
    del ref_exec, gen_exec, pert_exec
    gc.collect()
    return item_result


# ==========================================================================
# Pilot runner (E1 doc s2.5 / s8): run the full automatic layer on the 20
# independent pilot pointers.
# ==========================================================================
def run_pilot(manifest_path: Path, out_path: Path, *,
              timeout: int = DEFAULT_TIMEOUT, verbose: bool = True) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    pilot = manifest["samples"]["pilot"]

    items = []
    degr = {"n": len(pilot), "gen_net_recovered": 0, "ast_fallback": 0,
            "perturb_constructible": 0, "perturb_skipped": 0,
            "ref_reproduces_gt": 0, "gen_exec_failed": 0}
    dist = {c: {PASS: 0, FAIL: 0, AMBIG: 0} for c in
            ("check1_network_mods", "check2_solver_identity",
             "check3_physical_sanity", "check4_extraction", "check5_provenance")}

    for i, ptr in enumerate(pilot):
        api = ptr["tier"] == "API"
        path = cond_path(ptr["model"], ptr["condition"], api=api)
        rec = records.fetch_record(path, bench_index=ptr["bench_index"],
                                   item_id=ptr.get("item_id"))
        if rec is None:
            if verbose:
                print(f"[{i:2d}] MISSING {ptr['model']} bi={ptr['bench_index']}")
            continue
        if verbose:
            print(f"[{i:2d}] {ptr['tier']}/{ptr['condition']} {rec['task']:16s} "
                  f"bi={ptr['bench_index']} net={rec.get('network')}", flush=True)
        res = audit_item(rec, timeout=timeout)
        res["pointer"] = ptr
        res["meta"] = {k: rec.get(k) for k in
                       ("bench_index", "item_id", "task", "network", "query_type",
                        "n_modifications", "difficulty_level", "ground_truth",
                        "ground_truth_type")}
        items.append(res)

        for c in dist:
            dist[c][res["check_status"][c]] += 1
        ex = res["exec"]
        degr["gen_net_recovered"] += int(bool(ex["gen_net_recovered"]))
        degr["ast_fallback"] += int(res["checks"]["check1_network_mods"]
                                    ["evidence"].get("degraded", False))
        degr["ref_reproduces_gt"] += int(bool(ex["ref_reproduces_gt"]))
        degr["gen_exec_failed"] += int(not ex["gen_success"])
        v2 = res["checks"]["check5_provenance"]["evidence"]["v2"]
        degr["perturb_constructible"] += int(bool(v2.get("constructible")))
        degr["perturb_skipped"] += int(v2.get("skip_reason") is not None)
        if verbose:
            print("       " + "  ".join(f"{c.split('_')[0]}={res['check_status'][c]}"
                                        for c in dist), flush=True)

    n = len(items)
    rates = {
        "net_recovery_rate": round(degr["gen_net_recovered"] / n, 3) if n else None,
        "ast_fallback_rate": round(degr["ast_fallback"] / n, 3) if n else None,
        "perturb_unconstructible_rate":
            round(1 - degr["perturb_constructible"] / n, 3) if n else None,
        "ref_reproduces_gt_rate": round(degr["ref_reproduces_gt"] / n, 3) if n else None,
        "gen_exec_fail_rate": round(degr["gen_exec_failed"] / n, 3) if n else None,
    }
    out = {
        "meta": {
            "protocol": "E1_validity_audit.md "
                        "s2.2/s2.5 (T2 automatic layer)",
            "generated": _dt.datetime.now().isoformat(timespec="seconds"),
            "pandapower": "3.4.0", "perturb_factor": PERTURB_FACTOR,
            "timeout_s": timeout, "n_pilot": n,
            "note": "Automatic layer only: per-check {pass/fail/ambiguous} + "
                    "evidence. NOT a final clean/pollution ruling; no FAR. "
                    "Feeds dual-judge + refute (E1 doc s2.4).",
        },
        "distribution": dist,
        "degradation": {**degr, "rates": rates},
        "items": items,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    if verbose:
        print(f"\nwritten: {out_path}")
        print("per-check distribution:", json.dumps(dist))
        print("degradation rates:", json.dumps(rates))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the E1 T2 five-check "
                                             "automatic layer on the pilot 20.")
    ap.add_argument("--manifest",
                    default=str(PROJECT_ROOT / "audit" / "e1_sample_manifest.json"))
    ap.add_argument("--out",
                    default=str(PROJECT_ROOT / "audit" / "pilot20_auto_results.json"))
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run_pilot(Path(args.manifest), Path(args.out),
              timeout=args.timeout, verbose=not args.quiet)


if __name__ == "__main__":
    main()
