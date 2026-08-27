# --------------------------------------------------------------------------
# Repository copy of knowledge_injection/semantic.py, with imports and
# data-path constants adapted to this repository's layout. Runs CPU-only
# against the archived corpus/spec files in this repository. See
# code/README.md for the module map.
# --------------------------------------------------------------------------
"""
Trace-grounded value repair context for reactive fix rounds.

This module handles the "code executed but returned the wrong value" branch.
It does not inspect the reference answer and does not parse benchmark-specific
query templates.  For value bugs, it extracts a compact implementation trace
from the generated code and attaches docs-derived library contracts for the
APIs/tables the code actually touched, plus high-confidence docs-derived
workflow anchors that are absent from the trace.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from intervention.library_spec import LibraryKnowledgeBase


ROUTE_SEMANTIC = "semantic_fix"
SOURCE_SEMANTIC = "trace_grounded_value_repair"

_BUILTINS = frozenset({
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len",
    "list", "max", "min", "print", "range", "round", "set", "sorted", "str",
    "sum", "zip",
})

_OBJECT_HANDLE_NAMES = frozenset({"net", "network", "grid", "model", "case"})
_ANALYSIS_NAME_HINTS = (
    "runpp", "rundcpp", "runopp", "run_timeseries", "run_contingency",
    "calc_sc", "estimate", "runpp_3ph",
)


def _dedup(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _matches(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) is not None
    except re.error:
        return pattern.lower() in text.lower()


def _matches_any(patterns: Iterable[str], text: str) -> bool:
    return any(_matches(p, text or "") for p in patterns or [])


def _short(text: str, limit: int = 220) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _fmt_line(item: Dict) -> str:
    line = item.get("line")
    return f"L{line}: " if line else ""


@dataclass
class CodeTrace:
    called_functions: List[str] = field(default_factory=list)
    called_symbols: List[str] = field(default_factory=list)
    call_sequence: List[Dict] = field(default_factory=list)
    analysis_calls: List[Dict] = field(default_factory=list)
    state_writes: List[Dict] = field(default_factory=list)
    object_creations: List[Dict] = field(default_factory=list)
    result_reads: List[Dict] = field(default_factory=list)
    final_reads: List[Dict] = field(default_factory=list)
    final_print: str = ""
    final_expression: str = ""
    expanded_final_expression: str = ""
    code_alerts: List[str] = field(default_factory=list)

    def compact(self) -> Dict:
        return {
            "called_functions": self.called_functions,
            "call_sequence": self.call_sequence[:16],
            "analysis_calls": self.analysis_calls[:10],
            "state_writes": self.state_writes[:12],
            "object_creations": self.object_creations[:8],
            "result_reads": self.result_reads[:12],
            "final_expression": _short(self.expanded_final_expression or self.final_expression),
            "final_reads": self.final_reads[:6],
            "code_alerts": self.code_alerts,
        }


@dataclass
class SemanticFailurePlan:
    active: bool
    reason: str
    signals: List[str] = field(default_factory=list)
    repair_actions: List[str] = field(default_factory=list)
    contract_ids: List[str] = field(default_factory=list)
    matched_intents: List[str] = field(default_factory=list)
    anchor_functions: List[str] = field(default_factory=list)
    observed_functions: List[str] = field(default_factory=list)
    expected_contract: Dict = field(default_factory=dict)
    observed_contract: Dict = field(default_factory=dict)
    route: str = ROUTE_SEMANTIC
    source: str = SOURCE_SEMANTIC


class SemanticDemandAnalyzer:
    """Build compact value-level repair context from generated-code trace."""

    VALUE_BUG_PATTERNS = (
        r"\bincorrect numeric output\b",
        r"\bwrong result\b",
        r"\bvalue mismatch\b",
    )
    # Output-format failures: code executed successfully, the printed payload
    # could not be parsed as the expected scalar / boolean type. These are still
    # value-level failures (not runtime/code errors), so they belong on the
    # FDRS branch but skip trace/contract enrichment in favour of a focused
    # output-format instruction.
    OUTPUT_FORMAT_PATTERNS = (
        "cannot_parse",
        "empty_output",
    )
    # Failures we still cannot reason about with trace + contracts and where
    # output format alone is unlikely to help; let the basic fix loop handle
    # them.
    NON_VALUE_BUG_TERMS = (
        "nan_or_inf",
        "unknown_gt_type",
        "comparison_error",
    )
    MUTATION_CODE_PATTERNS = (
        r"\.loc\s*\[[^\n]*\]\s*(?:[+\-*/%]?=)",
        r"\.at\s*\[[^\n]*\]\s*(?:[+\-*/%]?=)",
        r"\.iloc\s*\[[^\n]*\]\s*(?:[+\-*/%]?=)",
        r"\bcreate_[A-Za-z_][A-Za-z0-9_]*\s*\(",
        r"\bdrop\s*\(",
        r"\[['\"](?:in_service|scaling|p_mw|q_mvar|vm_pu|min_vm_pu|max_vm_pu|parallel|tap_pos)['\"]\]\s*(?:[+\-*/%]?=)",
        r"\b(?:in_service|scaling|p_mw|q_mvar|vm_pu|min_vm_pu|max_vm_pu|parallel|tap_pos)\s*(?:[+\-*/%]?=)",
    )

    def __init__(
        self,
        knowledge_base: Optional[LibraryKnowledgeBase] = None,
        max_signals: int = 4,
    ):
        self.kb = knowledge_base or LibraryKnowledgeBase({"functions": []})
        self.max_signals = max(1, int(max_signals or 4))

    def is_value_bug(self, error_type: str = "", error_msg: str = "") -> bool:
        if (error_type or "").strip() not in {"", "default", "ValueMismatch"}:
            return False
        msg = error_msg or ""
        lower = msg.lower()
        if any(term in lower for term in self.NON_VALUE_BUG_TERMS):
            return False
        if any(term in lower for term in self.OUTPUT_FORMAT_PATTERNS):
            return True
        return _matches_any(self.VALUE_BUG_PATTERNS, msg)

    def is_output_format_failure(self, error_msg: str = "") -> bool:
        lower = (error_msg or "").lower()
        return any(term in lower for term in self.OUTPUT_FORMAT_PATTERNS)

    def analyze(
        self,
        original_query: str,
        generated_code: str,
        error_msg: str = "",
        error_type: str = "default",
    ) -> SemanticFailurePlan:
        if not self.is_value_bug(error_type=error_type, error_msg=error_msg):
            return SemanticFailurePlan(
                active=False,
                reason="failure is not an executed-but-wrong value bug",
            )

        # Output-format sub-path: code executed successfully but printed a
        # value the matcher could not parse. Trace and contracts are not
        # actionable here; the model needs a clear output-format instruction.
        if self.is_output_format_failure(error_msg):
            return SemanticFailurePlan(
                active=True,
                reason="value-level failure: output could not be parsed as expected scalar",
                signals=[
                    "The previous code executed but printed a value the matcher could not parse as the expected scalar."
                ],
                repair_actions=["fix_output_format"],
                contract_ids=[],
                matched_intents=[],
                anchor_functions=[],
                observed_functions=[],
                expected_contract={
                    "principle": "Output format failures are still value-level bugs; reactive intervention is the format instruction itself.",
                },
                observed_contract={"failure_kind": "output_format"},
            )

        query = original_query or ""
        code = generated_code or ""
        trace = self._extract_code_trace(code)
        matched_intents = self.kb.matched_workflow_ids(query)
        missing_anchors = self._missing_workflow_anchors(trace, matched_intents)
        contract_ids = self._trace_contract_ids(trace, missing_anchors)
        signals = self._trace_signals(trace, missing_anchors)
        repair_actions = self._trace_repair_actions(trace, missing_anchors)

        return SemanticFailurePlan(
            active=True,
            reason="trace-grounded value repair context generated",
            signals=_dedup(signals)[: self.max_signals],
            repair_actions=_dedup(repair_actions)[:5],
            contract_ids=_dedup(contract_ids)[:6],
            matched_intents=matched_intents,
            anchor_functions=missing_anchors,
            observed_functions=trace.called_functions,
            expected_contract={
                "demand_intents": matched_intents,
                "missing_anchor_functions": missing_anchors,
                "principle": "Use the original task text plus the code trace; no reference answer is used.",
            },
            observed_contract=trace.compact(),
        )

    def render(self, plan: SemanticFailurePlan) -> str:
        if not plan.active:
            return ""

        # Output-format failure: render only the focused format instruction.
        if "fix_output_format" in (plan.repair_actions or []):
            return "\n".join([
                "[Value-Level Output Format Repair]",
                (
                    "The previous code executed without raising an exception, but "
                    "the printed payload could not be parsed as the expected scalar."
                ),
                "Repair instruction:",
                (
                    "- The last line must print exactly one numeric scalar value — "
                    "no labels, no f-strings wrapping the value, no multi-line text, "
                    "no DataFrame/Series/array dump."
                ),
                "Keep the rest of the analysis logic unchanged.",
            ])

        trace = plan.observed_contract or {}
        lines = [
            "[Trace-Grounded Contract Context]",
            (
                "The previous code executed and printed a numeric value, but it did "
                "not match the requested result. Compare the original task with this "
                "implementation trace and the library contracts below. No reference "
                "answer is used."
            ),
        ]
        if plan.signals:
            lines.append("Structural trace signals:")
            lines.extend(f"- {signal}" for signal in plan.signals if signal)

        obligations = self._repair_obligations(plan)
        if obligations:
            lines.append("High-confidence trace invariants:")
            lines.extend(f"- {line}" for line in obligations)

        trace_lines = self._render_trace_lines(trace)
        if trace_lines:
            lines.append("Implementation trace:")
            lines.extend(f"- {line}" for line in trace_lines)

        contract_texts = self.kb.render_contract_ids(
            plan.contract_ids,
            max_lines_per_contract=1,
        )
        if contract_texts:
            lines.append("Trace-touched interface contracts:")
            lines.extend(contract_texts)
        lines.append(
            "Return a minimal code repair whose trace satisfies the original task "
            "and the contracts. Do not guess the expected numeric value."
        )
        return "\n".join(lines)

    def _render_trace_lines(self, trace: Dict) -> List[str]:
        lines: List[str] = []
        calls = [
            f"{_fmt_line(c)}{c.get('function')}"
            for c in list(trace.get("call_sequence") or [])[:10]
            if c.get("function")
        ]
        if calls:
            lines.append("Library/API call order: " + " -> ".join(calls))

        writes = []
        for write in list(trace.get("state_writes") or [])[:8]:
            target = write.get("target") or self._table_field_label(write)
            value = write.get("value")
            if value:
                writes.append(f"{_fmt_line(write)}{target} = {_short(value, 80)}")
            elif target:
                writes.append(f"{_fmt_line(write)}{target}")
        if writes:
            lines.append("State/table writes: " + " | ".join(writes))

        creations = [
            f"{_fmt_line(c)}{c.get('function')}({_short(c.get('args', ''), 80)})"
            for c in list(trace.get("object_creations") or [])[:6]
            if c.get("function")
        ]
        if creations:
            lines.append("Object creation calls: " + " | ".join(creations))

        reads = [
            f"{_fmt_line(r)}{self._table_field_label(r)}"
            for r in list(trace.get("result_reads") or [])[:8]
            if self._table_field_label(r)
        ]
        if reads:
            lines.append("Result-table reads seen in code: " + " | ".join(_dedup(reads)))

        final_reads = [
            self._table_field_label(r)
            for r in list(trace.get("final_reads") or [])[:6]
            if self._table_field_label(r)
        ]
        final_expr = trace.get("final_expression") or ""
        if final_reads:
            lines.append("Final print reads: " + " | ".join(_dedup(final_reads)))
        if final_expr:
            lines.append("Final print expression: " + _short(final_expr, 180))
        return lines

    def _repair_obligations(self, plan: SemanticFailurePlan) -> List[str]:
        actions = set(plan.repair_actions or [])
        obligations: List[str] = []
        if "fix_self_difference" in actions:
            obligations.append(
                "Final arithmetic must not subtract an expression from itself; the trace needs distinct observations if the task asks for a change or comparison."
            )
        if "verify_missing_workflow_anchor" in actions and plan.anchor_functions:
            obligations.append(
                "The trace lacks docs-derived workflow anchor API(s): "
                + ", ".join(plan.anchor_functions[:5])
                + "."
            )
        if "verify_final_result_source" in actions:
            trace = plan.observed_contract or {}
            if not trace.get("final_reads"):
                obligations.append(
                    "The final print should be traceable to a scalar value computed from the intended final analysis state."
                )
        return _dedup(obligations)

    def _trace_signals(self, trace: CodeTrace, missing_anchors: List[str]) -> List[str]:
        signals = []
        signals.extend(trace.code_alerts)
        if missing_anchors:
            signals.append(
                "Docs-derived task intent suggests missing workflow anchor API(s): "
                + ", ".join(missing_anchors[:5])
                + "."
            )
        if not trace.final_reads and trace.final_expression:
            signals.append(
                "The final print expression does not clearly read a documented result table/field."
            )
        return signals

    def _trace_repair_actions(
        self,
        trace: CodeTrace,
        missing_anchors: List[str],
    ) -> List[str]:
        actions = ["trace_grounded_value_repair"]
        if missing_anchors:
            actions.append("verify_missing_workflow_anchor")
        if any("subtract" in alert and "same expression" in alert for alert in trace.code_alerts):
            actions.append("fix_self_difference")
        if trace.final_expression:
            actions.append("verify_final_result_source")
        return _dedup(actions)

    def _missing_workflow_anchors(
        self,
        trace: CodeTrace,
        matched_intents: List[str],
    ) -> List[str]:
        missing: List[str] = []
        matched = set(matched_intents or [])
        seen = set(trace.called_functions or [])
        seen_lower = {s.lower() for s in trace.called_symbols or []}
        for intent in self.kb.api_intents:
            if not isinstance(intent, dict):
                continue
            intent_id = str(intent.get("id") or "")
            if intent_id not in matched:
                continue
            anchors = [
                self.kb.canonical_name(fn)
                for fn in list(intent.get("anchor_functions") or [])
                if self.kb.known(self.kb.canonical_name(fn))
            ]
            gated = [
                self.kb.canonical_name(fn)
                for fn in list(intent.get("gated_functions") or [])
                if self.kb.known(self.kb.canonical_name(fn))
            ]
            if not anchors or not gated:
                continue
            present = [
                fn for fn in anchors
                if fn in seen or fn.lower() in seen_lower
            ]
            if not present:
                missing.extend(anchors)
        return _dedup(missing)

    def _trace_contract_ids(
        self,
        trace: CodeTrace,
        missing_anchors: List[str],
    ) -> List[str]:
        ids: List[str] = []
        # Analysis routines come first so the prompt always shows where the
        # function actually writes its result tables. This catches cases where
        # the model reads from a Python return value (e.g. contingency_results
        # dict) instead of net.res_*.
        for call in trace.analysis_calls or []:
            fn = self.kb.canonical_name(call.get("function") or "")
            cid = f"function_outputs:{fn}"
            if fn and self.kb.contract_for_id(cid):
                ids.append(cid)
        for read in list(trace.final_reads or []) + list(trace.result_reads or []):
            cid = self._output_contract_id(read)
            if cid and self.kb.contract_for_id(cid):
                ids.append(cid)
        priority_functions = list(missing_anchors or [])
        priority_functions.extend(c.get("function") for c in trace.analysis_calls or [])
        for fn in priority_functions:
            cid = f"function:{self.kb.canonical_name(fn)}"
            if self.kb.contract_for_id(cid):
                ids.append(cid)
        for write in trace.state_writes or []:
            cid = self._state_contract_id(write)
            if cid and self.kb.contract_for_id(cid):
                ids.append(cid)
            schema_id = f"schema:{write.get('table')}"
            if write.get("table") and self.kb.contract_for_id(schema_id):
                ids.append(schema_id)
        function_order = []
        function_order.extend(c.get("function") for c in trace.object_creations or [])
        function_order.extend(trace.called_functions or [])
        for fn in function_order:
            cid = f"function:{self.kb.canonical_name(fn)}"
            if self.kb.contract_for_id(cid):
                ids.append(cid)
        return _dedup(ids)

    def _extract_code_trace(self, code: str) -> CodeTrace:
        calls: List[str] = []
        symbols: List[str] = []
        call_sequence: List[Dict] = []
        analysis_calls: List[Dict] = []
        state_writes: List[Dict] = []
        object_creations: List[Dict] = []
        assignments: Dict[str, str] = {}
        print_calls: List[Tuple[int, str, str]] = []

        try:
            tree = ast.parse(code or "")
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    line = getattr(node, "lineno", 0)
                    targets = self._assignment_targets(code, node)
                    value = self._assignment_value(code, node)
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and value:
                                assignments[target.id] = value
                    elif isinstance(node, ast.AnnAssign):
                        if isinstance(node.target, ast.Name) and value:
                            assignments[node.target.id] = value
                    for target_text in targets:
                        write = self._parse_state_write(target_text, value, line)
                        if write:
                            state_writes.append(write)
                elif isinstance(node, ast.Call):
                    call_name = self._call_name(node.func)
                    if not call_name:
                        continue
                    symbols.append(call_name)
                    last = call_name.split(".")[-1]
                    canonical = self.kb.canonical_name(last)
                    if self.kb.known(canonical):
                        calls.append(canonical)
                        entry = {"line": getattr(node, "lineno", 0), "function": canonical}
                        call_sequence.append(entry)
                        if self._is_analysis_call(canonical):
                            analysis_calls.append(entry)
                        if canonical.startswith("create_"):
                            object_creations.append({
                                "line": getattr(node, "lineno", 0),
                                "function": canonical,
                                "args": self._call_args_text(code, node),
                            })
                    if last == "print":
                        seg = ast.get_source_segment(code, node) or ""
                        expr = ""
                        if node.args:
                            expr = ast.get_source_segment(code, node.args[0]) or ""
                        if seg:
                            print_calls.append((getattr(node, "lineno", 0), seg, expr))
        except SyntaxError:
            pass

        if not calls:
            calls.extend(self._regex_library_calls(code))
        symbols.extend(self._regex_call_symbols(code))
        call_sequence = sorted(call_sequence, key=lambda item: int(item.get("line") or 0))
        analysis_calls = sorted(analysis_calls, key=lambda item: int(item.get("line") or 0))
        state_writes = sorted(state_writes, key=lambda item: int(item.get("line") or 0))
        object_creations = sorted(object_creations, key=lambda item: int(item.get("line") or 0))

        final_print = ""
        final_expression = ""
        if print_calls:
            _, final_print, final_expression = sorted(print_calls, key=lambda x: x[0])[-1]
        final_print = final_print or self._last_print_regex(code)
        final_expression = final_expression or self._print_expr_from_text(final_print)
        expanded_expression = self._expand_expression(final_expression, assignments)
        result_reads = self._extract_result_reads(code or "")
        final_reads = self._extract_result_reads(
            " ".join(s for s in [final_expression, expanded_expression, final_print] if s)
        )

        trace = CodeTrace(
            called_functions=_dedup(calls),
            called_symbols=_dedup(symbols),
            call_sequence=call_sequence,
            analysis_calls=analysis_calls,
            state_writes=state_writes,
            object_creations=object_creations,
            result_reads=result_reads,
            final_reads=final_reads,
            final_print=final_print,
            final_expression=final_expression,
            expanded_final_expression=expanded_expression,
        )
        trace.code_alerts = self._code_alerts(trace)
        return trace

    def _assignment_targets(self, code: str, node: ast.AST) -> List[str]:
        targets: List[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        return [
            ast.get_source_segment(code, target) or ""
            for target in targets
        ]

    def _assignment_value(self, code: str, node: ast.AST) -> str:
        if isinstance(node, ast.Assign):
            return ast.get_source_segment(code, node.value) or ""
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            return ast.get_source_segment(code, node.value) or ""
        if isinstance(node, ast.AugAssign):
            op = type(node.op).__name__
            value = ast.get_source_segment(code, node.value) or ""
            return f"<augassign:{op}> {value}".strip()
        return ""

    def _parse_state_write(self, target_text: str, value: str, line: int) -> Dict:
        target = target_text or ""
        if not target or not target.startswith("net."):
            return {}
        if re.search(r"\bnet\.res_", target):
            return {}
        table = ""
        field = ""
        patterns = [
            r"\bnet\.([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(?:at|loc|iat|iloc)\s*\[[^\]]*?,\s*['\"]([^'\"]+)['\"]",
            r"\bnet\.([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*['\"]([^'\"]+)['\"]\s*\]",
            r"\bnet\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, target)
            if match:
                table, field = match.group(1), match.group(2)
                break
        if not table:
            match = re.search(r"\bnet\.([A-Za-z_][A-Za-z0-9_]*)\b", target)
            if match:
                table = match.group(1)
        return {
            "line": line,
            "table": table,
            "field": field,
            "target": _short(target, 120),
            "value": _short(value, 120),
        }

    def _extract_result_reads(self, text: str) -> List[Dict]:
        reads: List[Dict] = []
        text = text or ""
        patterns = [
            r"\b(?:net\.)?(res_[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(?:at|loc|iat|iloc)\s*\[[^\n]*?,\s*['\"]([^'\"]+)['\"]",
            r"\b(?:net\.)?(res_[A-Za-z_][A-Za-z0-9_]*)\s*\[\s*['\"]([^'\"]+)['\"]\s*\]",
            r"\b(?:net\.)?(res_[A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
            r"\boutput\s*\[\s*['\"](res_[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)['\"]\s*\]",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                if "." in match.group(1):
                    table, field = match.group(1).split(".", 1)
                else:
                    table, field = match.group(1), match.group(2)
                if field in {"at", "loc", "iat", "iloc"}:
                    continue
                reads.append({
                    "table": table,
                    "field": field,
                    "target": f"{table}.{field}",
                })
        return self._dedup_table_fields(reads)

    def _code_alerts(self, trace: CodeTrace) -> List[str]:
        alerts: List[str] = []
        expr = trace.expanded_final_expression or trace.final_expression
        if self._is_self_difference(expr):
            alerts.append(
                "The final expression appears to subtract the same expression from itself."
            )
        if trace.state_writes and trace.analysis_calls and trace.final_reads:
            last_run = max(int(c.get("line") or 0) for c in trace.analysis_calls)
            late_writes = [
                w for w in trace.state_writes
                if int(w.get("line") or 0) > last_run
            ]
            if late_writes:
                alerts.append(
                    "Some state/table writes appear after the last analysis call; verify that the analysis is rerun after the final mutation."
                )
        if trace.final_expression and not trace.final_reads and re.search(r"\bnet\.", trace.final_expression):
            alerts.append(
                "The final expression touches `net` but does not clearly read a documented result table."
            )
        return _dedup(alerts)

    def _is_self_difference(self, expr: str) -> bool:
        compact = re.sub(r"\s+", "", expr or "")
        if not compact or "-" not in compact:
            return False
        compact = compact.strip()
        while compact.startswith("(") and compact.endswith(")"):
            compact = compact[1:-1].strip()
        depth = 0
        split_at = -1
        for idx, char in enumerate(compact):
            if char == "(":
                depth += 1
            elif char == ")":
                depth = max(0, depth - 1)
            elif char == "-" and depth == 0:
                split_at = idx
                break
        if split_at < 0:
            return False
        left = compact[:split_at].strip("()")
        right = compact[split_at + 1:].strip("()")
        return bool(left and left == right)

    def _is_analysis_call(self, canonical: str) -> bool:
        name = str(canonical or "")
        return name in _ANALYSIS_NAME_HINTS or name.startswith("run")

    def _table_field_label(self, item: Dict) -> str:
        table = str(item.get("table") or "")
        field = str(item.get("field") or "")
        return f"{table}.{field}" if table and field else table

    def _output_contract_id(self, read: Dict) -> str:
        label = self._table_field_label(read)
        return f"output_observable:{label}" if label else ""

    def _state_contract_id(self, write: Dict) -> str:
        table = str(write.get("table") or "")
        field = str(write.get("field") or "")
        return f"state_mutation:{table}.{field}" if table and field else ""

    def _dedup_table_fields(self, items: List[Dict]) -> List[Dict]:
        out: List[Dict] = []
        seen: Set[Tuple[str, str]] = set()
        for item in items:
            table = str(item.get("table") or "")
            field = str(item.get("field") or "")
            key = (table, field)
            if table and field and key not in seen:
                seen.add(key)
                out.append(item)
        return out

    def _call_args_text(self, code: str, node: ast.Call) -> str:
        pieces = []
        for arg in list(node.args or [])[:4]:
            pieces.append(ast.get_source_segment(code, arg) or "")
        for kw in list(node.keywords or [])[:6]:
            if kw.arg:
                pieces.append(f"{kw.arg}={ast.get_source_segment(code, kw.value) or ''}")
        return ", ".join(p for p in pieces if p)

    def _call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._call_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return ""

    def _regex_library_calls(self, code: str) -> List[str]:
        calls: List[str] = []
        aliases = [re.escape(a) for a in self.kb.import_aliases if a]
        if aliases:
            pattern = r"\b(?:" + "|".join(aliases) + r")\.([A-Za-z_][A-Za-z0-9_]*)\s*\("
            for match in re.finditer(pattern, code or ""):
                fn = self.kb.canonical_name(match.group(1))
                if self.kb.known(fn):
                    calls.append(fn)
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", code or ""):
            name = match.group(1)
            if name in _BUILTINS:
                continue
            fn = self.kb.canonical_name(name)
            if self.kb.known(fn):
                calls.append(fn)
        return _dedup(calls)

    def _regex_call_symbols(self, code: str) -> List[str]:
        names = [
            match.group(1)
            for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", code or "")
            if match.group(1) not in _BUILTINS
        ]
        attrs = [
            match.group(1)
            for match in re.finditer(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", code or "")
            if match.group(1) not in _BUILTINS
        ]
        return _dedup(names + attrs)

    def _last_print_regex(self, code: str) -> str:
        prints = re.findall(r"print\s*\((.*?)\)\s*$", code or "", flags=re.MULTILINE)
        return f"print({prints[-1]})" if prints else ""

    def _print_expr_from_text(self, print_text: str) -> str:
        m = re.search(r"print\s*\((.*)\)\s*$", print_text or "", flags=re.DOTALL)
        return m.group(1).strip() if m else ""

    def _expand_expression(self, expr: str, assignments: Dict[str, str]) -> str:
        if not expr:
            return ""
        expanded = expr.strip()
        for _ in range(4):
            next_expr = self._expand_expression_once(expanded, assignments)
            if not next_expr or next_expr == expanded:
                break
            expanded = next_expr
            if len(expanded) > 700:
                break
        return expanded

    def _expand_expression_once(self, expr: str, assignments: Dict[str, str]) -> str:
        name = (expr or "").strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) and name in assignments:
            return assignments.get(name) or expr
        try:
            tree = ast.parse(expr or "", mode="eval")
        except SyntaxError:
            return expr

        class _InlineNames(ast.NodeTransformer):
            def __init__(self, mapping: Dict[str, str]):
                self.mapping = mapping
                self.changed = False

            def visit_Name(self, node: ast.Name):  # noqa: N802 - ast API name
                if node.id in _OBJECT_HANDLE_NAMES:
                    return node
                replacement = self.mapping.get(node.id)
                if not replacement:
                    return node
                try:
                    repl_node = ast.parse(replacement, mode="eval").body
                except SyntaxError:
                    return node
                self.changed = True
                return ast.copy_location(repl_node, node)

        inliner = _InlineNames(assignments)
        new_tree = inliner.visit(tree)
        if not inliner.changed:
            return expr
        ast.fix_missing_locations(new_tree)
        try:
            return ast.unparse(new_tree)
        except Exception:
            return expr
