# --------------------------------------------------------------------------
# Repository copy of knowledge_injection/reactive.py, with imports and
# data-path constants adapted to this repository's layout. Runs CPU-only
# against the archived corpus/spec files in this repository. See
# code/README.md for the module map.
# --------------------------------------------------------------------------
"""
Reactive Knowledge Injection
==============================
Parses model execution errors and injects targeted library API documentation
into fix-round prompts, replacing generic RAG retrieval with error-class-aware
documentation lookup.

Fix Conditions:
  FX  – No reactive injection (error message only) — current baseline
  FR  – L0+L1 injection (function name + signature)
  FD  – L0+L1+L2 injection (name + signature + description/params) — MAIN condition
  FDR – Fix-demand routed injection (basic/API-doc/boundary routing)
  FS  – Semantic diagnosis for executed-but-wrong value bugs
  FDRS – FDR plus semantic diagnosis for value bugs
  FE  – L0+L1+L2+L3 injection (full docs with code examples)

Combination Conditions (proactive_condition + fix_condition):
  A+FX  = standard benchmark + plain fix → benchmark_results_condA_FX.json
  A+FD  = baseline + reactive fix   → benchmark_results_condA_FD.json  — pure reactive
  C+FX  = proactive C + plain fix   → benchmark_results_condC_FX.json
  C+FD  = proactive C + reactive fix→ benchmark_results_condC_FD.json  — MAIN experiment
  C+FDR = proactive C + routed fix  → benchmark_results_condC_FDR.json — low-cost candidate
  C+FDRS = proactive C + routed/semantic fix → benchmark_results_condC_FDRS.json
  X+FD  = proactive X + reactive fix→ benchmark_results_condX_FD.json  — upper bound

Design principles vs ProactiveInjector:
  - Operates on FIX ROUNDS only (not round 0)
  - Uses the same demand-aware intervention idea, but with error-side evidence
  - Keeps legacy FD as an always-doc baseline for ablation
  - FDR routes each failure into basic fix, API docs, or boundary contract
  - FDRS routes runtime/API bugs like FDR and value bugs to semantic diagnosis

Usage (standalone):
    from intervention.reactive import ReactiveInjector
    injector = ReactiveInjector.from_paths("dataset/pandapower_docs.json", fix_condition="FD")
    docs_str = injector.get_injection(error_type, error_msg, generated_code)
    # Pass docs_str as retrieved_docs to build_fix_prompt()

Usage (integration in _run_benchmark_with_backend):
    reactive_injector=injector  param → auto-used in fix rounds instead of error_retriever
"""

import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from intervention.library_spec import LibraryKnowledgeBase
from intervention.semantic import (
    ROUTE_SEMANTIC,
    SOURCE_SEMANTIC,
    SemanticDemandAnalyzer,
)
from probing.probe_framework import InjectionGenerator, InjectionSnippet


LAYER_ORDER = ["L0", "L1", "L2", "L3"]

ROUTE_BASIC = "basic_fix"
ROUTE_API_DOC = "api_doc"
ROUTE_BOUNDARY = "boundary_contract"
ROUTE_STOP = "stop_retry"

# FDR is deliberately tighter than legacy FD. It should repair only the API
# contract that the error implicates, not every library call in the script.
ROUTED_MAX_API_FUNCS = 2

NO_DOC_ERROR_CLASSES = {
    "SyntaxError",
    "IndentationError",
    "code_extraction_failed",
    "TimeoutError",
    "TypingError",
    "UFuncTypeError",
    "NotImplementedError",
    "AlgorithmUnknown",
    "ZeroDivisionError",
}

API_DOC_ERROR_CLASSES = {
    "AttributeError",
    "TypeError",
    "NameError",
    "ImportError",
    "ModuleNotFoundError",
    "IndexError",
}

API_CONTRACT_ERROR_CLASSES = {
    "ValueError",
    "KeyError",
    "InvalidIndexError",
    "RuntimeError",
    "Exception",
    "UserWarning",
    "LoadflowNotConverged",
    "OPFNotConverged",
}

BOUNDARY_ERROR_CLASSES = {
    "KeyError",
    "InvalidIndexError",
    "LoadflowNotConverged",
    "OPFNotConverged",
}

@dataclass
class FixDemandPlan:
    """Structured decision for one fix-round intervention."""

    route: str
    reason: str
    error_class: str
    functions: List[str] = field(default_factory=list)
    n_layers: int = 0
    layers: List[str] = field(default_factory=list)
    boundary_cards: List[str] = field(default_factory=list)
    source: str = "fix_demand"

# Error class → injection depth (number of layers) for FD condition
# 0 = no injection (FX fallback)
ERROR_CLASS_LAYERS: Dict[str, int] = {
    "AttributeError":        3,   # wrong method name → name + signature + description
    "TypeError":             3,   # wrong args/kwargs → name + signature + description
    "NameError":             2,   # undefined name → name + signature is enough
    "SyntaxError":           0,   # syntax issue → doc injection doesn't help → FX fallback
    "ImportError":           1,   # wrong import → name only
    "ValueError":            3,   # bad parameter value → name + signature + description
    "KeyError":              4,   # wrong key access pattern → full docs + example needed
    "IndexError":            1,   # indexing error → name only
    "RuntimeError":          3,
    "LoadflowNotConverged":  4,   # wrong ConstControl variable/data → full docs + example
    "default":               3,   # unknown error class → full description (safe default)
}

# Python builtins and common names to exclude from function-name extraction
_BUILTINS = frozenset({
    "print", "len", "range", "list", "dict", "set", "str", "int", "float",
    "type", "None", "True", "False", "zip", "map", "filter", "sum", "max",
    "min", "abs", "round", "open", "sorted", "enumerate", "isinstance",
    "hasattr", "getattr", "setattr", "any", "all", "next", "iter",
})

# ─────────────────────────────────────────────────────────────────────────────
# Error Parser
# ─────────────────────────────────────────────────────────────────────────────

class ErrorParser:
    """
    Extracts target-library function names from execution errors.

    Priority order:
      1. AttributeError  → regex 'has no attribute X' → validate → inject
      2. TypeError       → 'unexpected keyword argument' or 'X() takes' → scan code
      3. NameError       → 'name X is not defined' → validate → inject
      4. SyntaxError     → return [] (doc injection doesn't help syntax errors)
      5. Fallback        → scan generated code for library-qualified call sites
    """

    _ATTR_ERR = re.compile(r"has no attribute '([^']+)'")
    _TYPE_FUNC = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(\)")
    _TYPE_KW   = re.compile(r"got an unexpected keyword argument '([^']+)'")
    _NAME_ERR  = re.compile(r"name '([^']+)' is not defined")
    _LINE_HEAD = re.compile(r"^Line\s+(\d+)$")

    def __init__(
        self,
        known_functions: Optional[Set[str]] = None,
        knowledge_base: LibraryKnowledgeBase = None,
    ):
        """
        Args:
            known_functions: set of known target-library function names for validation.
                             When provided, only names in this set are accepted.
                             When None, uses heuristic filtering.
        """
        self.known_functions = known_functions
        self.kb = knowledge_base or LibraryKnowledgeBase({"functions": []})

    def parse(
        self,
        error_type: str,
        error_msg: str,
        generated_code: str = "",
    ) -> Tuple[List[str], str]:
        """
        Extract implicated function names and normalised error class.

        Returns:
            (func_names, error_class)
              func_names  – ordered list of target-library functions to inject docs for
              error_class – error type string used for ERROR_CLASS_LAYERS lookup
        """
        details = self.parse_details(error_type, error_msg, generated_code)
        return details["functions"], details["error_class"]

    def parse_details(
        self,
        error_type: str,
        error_msg: str,
        generated_code: str = "",
    ) -> Dict:
        """Extract structured bug evidence from compact parse_error() text."""
        error_class = self.normalise_error_class(error_type, error_msg)
        line_no, failing_line, bare_error = self._split_compact_error(error_msg)

        if error_class == "SyntaxError":
            return {
                "functions": [],
                "line_functions": [],
                "error_class": error_class,
                "line_no": line_no,
                "failing_line": failing_line,
                "bare_error": bare_error,
            }

        func_names: List[str] = []
        line_functions = self._scan_code(failing_line, limit=3) if failing_line else []

        if error_class == "AttributeError":
            m = self._ATTR_ERR.search(error_msg or "")
            if m:
                name = self._canonical(m.group(1))
                if self._valid(name):
                    func_names = [name]
            if not func_names and line_functions:
                func_names = line_functions

        elif error_class == "TypeError":
            # Case 1: unexpected keyword argument → scan caller site in code
            if self._TYPE_KW.search(error_msg or ""):
                func_names = line_functions or self._scan_code(generated_code, limit=3)
            else:
                # Case 2: "X() takes N arguments"
                m = self._TYPE_FUNC.search(error_msg or "")
                if m:
                    name = self._canonical(m.group(1))
                    if self._valid(name):
                        func_names = [name]
                if not func_names:
                    func_names = line_functions or self._scan_code(generated_code, limit=3)

        elif error_class == "NameError":
            m = self._NAME_ERR.search(error_msg or "")
            if m:
                name = self._canonical(m.group(1))
                if self._valid(name):
                    func_names = [name]

        else:
            # ValueError, KeyError, RuntimeError, etc.
            func_names = line_functions or self._scan_code(generated_code, limit=3)

        # Final fallback: scan code
        if not func_names:
            func_names = self._scan_code(generated_code, limit=5)

        return {
            "functions": _dedup(func_names),
            "line_functions": _dedup(line_functions),
            "error_class": error_class,
            "line_no": line_no,
            "failing_line": failing_line,
            "bare_error": bare_error,
        }

    def normalise_error_class(self, error_type: str, error_msg: str = "") -> str:
        error_class = (error_type or "").strip()
        if not error_class or error_class == "default":
            last = (error_msg or "").rsplit(" | ", 1)[-1]
            m = re.match(r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Warning|Converged|Unknown))\s*:", last)
            if m:
                error_class = m.group(1)
        if not error_class:
            error_class = "default"
        if "." in error_class:
            error_class = error_class.rsplit(".", 1)[-1]
        return error_class

    def _split_compact_error(self, error_msg: str) -> Tuple[Optional[int], str, str]:
        """Parse utils.parse_error() output: Line N | code line | ErrorType: msg."""
        if not error_msg:
            return None, "", ""
        parts = [p.strip() for p in str(error_msg).split(" | ")]
        line_no = None
        failing_line = ""
        if parts:
            m = self._LINE_HEAD.match(parts[0])
            if m:
                line_no = int(m.group(1))
                if len(parts) >= 3:
                    failing_line = parts[1]
        bare_error = parts[-1] if parts else str(error_msg)
        return line_no, failing_line, bare_error

    def _canonical(self, name: str) -> str:
        clean = (name or "").strip()
        return self.kb.canonical_name(clean)

    def _valid(self, name: str) -> bool:
        """True if name is a plausible target-library function name."""
        name = self._canonical(name)
        if not name or len(name) < 3 or name in _BUILTINS:
            return False
        if not name[0].isalpha():
            return False
        if self.known_functions is not None:
            return name in self.known_functions
        return True

    def _scan_code(self, code: str, limit: int = 5) -> List[str]:
        """Scan code for library-qualified calls plus direct known-function calls."""
        if not code:
            return []
        aliases = [re.escape(a) for a in self.kb.import_aliases if a]
        if aliases:
            call_re = re.compile(
                rf"(?:{'|'.join(aliases)})(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
                r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\("
            )
            candidates = [self._canonical(c) for c in call_re.findall(code)]
        else:
            candidates = []
        # Secondary: if known_functions is set, also scan for any direct call `FuncName(`
        # that appears in the code and is a known target-library function.
        if self.known_functions is not None:
            direct_call = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(')
            for m in direct_call.finditer(code):
                name = m.group(1)
                name = self._canonical(name)
                if name in self.known_functions and name not in _BUILTINS:
                    candidates.append(name)

        if self.known_functions is not None:
            candidates = [c for c in candidates if c in self.known_functions]
        else:
            candidates = [c for c in candidates if self._valid(c)]
        return _dedup(candidates)[:limit]


# ─────────────────────────────────────────────────────────────────────────────
# Fix-Demand Router
# ─────────────────────────────────────────────────────────────────────────────

class FixDemandRouter:
    """Decide whether one failed fix needs docs, a boundary contract, or plain fix.

    The router is intentionally evidence-based and model-agnostic: it uses
    exception class, compact failing-line evidence, and target-library API
    symbols. The original query remains in the fix prompt, but FDR does not use
    query intent alone to trigger reactive docs; query-side demand belongs to the
    proactive stage.
    """

    def __init__(self, parser: ErrorParser, knowledge_base: LibraryKnowledgeBase = None):
        self.parser = parser
        self.kb = knowledge_base or parser.kb

    def route(
        self,
        error_type: str,
        error_msg: str,
        generated_code: str = "",
        original_query: str = "",
        max_funcs: int = ROUTED_MAX_API_FUNCS,
    ) -> Tuple[FixDemandPlan, Dict]:
        evidence = self.parser.parse_details(error_type, error_msg, generated_code)
        error_class = evidence["error_class"]
        boundary_cards = self._detect_boundary_cards(
            error_msg=error_msg,
            generated_code=generated_code,
            original_query=original_query,
            failing_line=evidence.get("failing_line", ""),
        )
        funcs = self._prioritise_functions(evidence, max_funcs=max_funcs)
        line_funcs = _dedup(evidence.get("line_functions") or [])
        funcs = self._expand_context_functions(
            funcs=funcs,
            evidence=evidence,
            error_msg=error_msg,
            generated_code=generated_code,
            original_query=original_query,
            max_funcs=max_funcs,
        )

        if error_class in NO_DOC_ERROR_CLASSES:
            return FixDemandPlan(
                route=ROUTE_BASIC,
                reason=f"{error_class} is normally a code/local-runtime fix, not an API-doc demand",
                error_class=error_class,
                source="fix_demand_basic",
            ), evidence

        if error_class == "default":
            return FixDemandPlan(
                route=ROUTE_BASIC,
                reason="wrong-result/default failure should preserve task semantics with plain fix unless an exception gives precise API evidence",
                error_class=error_class,
                source="fix_demand_basic",
            ), evidence

        if error_class in BOUNDARY_ERROR_CLASSES:
            if line_funcs:
                return self._api_doc_plan(
                    error_class, funcs,
                    f"{error_class} occurs on a concrete {self.kb.library_name} API call",
                    boundary_cards=boundary_cards,
                ), evidence
            if boundary_cards:
                return self._boundary_plan(
                    error_class, boundary_cards,
                    f"{error_class} points to API result/profile/boundary semantics",
                ), evidence
            return FixDemandPlan(
                route=ROUTE_BASIC,
                reason=f"{error_class} without a recognized API boundary is usually local indexing/data handling",
                error_class=error_class,
                source="fix_demand_basic",
            ), evidence

        if error_class == "AttributeError":
            if boundary_cards and not evidence.get("line_functions"):
                return self._boundary_plan(
                    error_class, boundary_cards,
                    "attribute failure matches a known result-object boundary",
                ), evidence
            if funcs:
                return self._api_doc_plan(
                    error_class, funcs,
                    f"attribute/name mismatch implicates a concrete {self.kb.library_name} API",
                ), evidence
            if boundary_cards:
                return self._boundary_plan(
                    error_class, boundary_cards,
                    "attribute failure matches a known boundary pattern",
                ), evidence
            return FixDemandPlan(
                route=ROUTE_BASIC,
                reason=f"attribute failure did not resolve to a known {self.kb.library_name} API",
                error_class=error_class,
                source="fix_demand_basic",
            ), evidence

        if error_class in API_DOC_ERROR_CLASSES:
            if funcs:
                return self._api_doc_plan(
                    error_class, funcs,
                    f"{error_class} implicates a concrete {self.kb.library_name} API on the failing line",
                    boundary_cards=boundary_cards,
                ), evidence
            if boundary_cards:
                return self._boundary_plan(
                    error_class, boundary_cards,
                    f"{error_class} has boundary evidence but no concrete function",
                ), evidence
            return FixDemandPlan(
                route=ROUTE_BASIC,
                reason=f"{error_class} has no recognized {self.kb.library_name} API symbol",
                error_class=error_class,
                source="fix_demand_basic",
            ), evidence

        if error_class in API_CONTRACT_ERROR_CLASSES:
            if line_funcs:
                return self._api_doc_plan(
                    error_class, funcs,
                    f"{error_class} indicates an API contract/signature/value issue on the failing line",
                    boundary_cards=boundary_cards,
                ), evidence
            if boundary_cards:
                return self._boundary_plan(
                    error_class, boundary_cards,
                    f"{error_class} occurs at a high-risk API boundary",
                ), evidence
            return FixDemandPlan(
                route=ROUTE_BASIC,
                reason=f"{error_class} often reflects network/data state; avoid broad code-scan docs",
                error_class=error_class,
                source="fix_demand_basic",
            ), evidence

        if boundary_cards:
            return self._boundary_plan(
                error_class, boundary_cards,
                "fallback boundary evidence from query/code/error text",
            ), evidence

        return FixDemandPlan(
            route=ROUTE_BASIC,
            reason=f"{error_class} has insufficient API-doc demand evidence",
            error_class=error_class,
            source="fix_demand_basic",
        ), evidence

    def _api_doc_plan(
        self,
        error_class: str,
        funcs: List[str],
        reason: str,
        boundary_cards: Optional[List[str]] = None,
    ) -> FixDemandPlan:
        n_layers = ERROR_CLASS_LAYERS.get(error_class, 2)
        # Routed docs never escalate to examples; boundary contracts handle workflow shape.
        n_layers = min(max(n_layers, 1), 3)
        layers = LAYER_ORDER[:n_layers]
        return FixDemandPlan(
            route=ROUTE_API_DOC,
            reason=reason,
            error_class=error_class,
            functions=funcs,
            n_layers=n_layers,
            layers=layers,
            boundary_cards=boundary_cards or [],
            source="fix_demand_api_doc",
        )

    def _boundary_plan(
        self, error_class: str, boundary_cards: List[str], reason: str
    ) -> FixDemandPlan:
        return FixDemandPlan(
            route=ROUTE_BOUNDARY,
            reason=reason,
            error_class=error_class,
            boundary_cards=boundary_cards,
            source="fix_demand_boundary",
        )

    def _prioritise_functions(self, evidence: Dict, max_funcs: int) -> List[str]:
        max_funcs = max(1, min(int(max_funcs or ROUTED_MAX_API_FUNCS), ROUTED_MAX_API_FUNCS))
        line_funcs = evidence.get("line_functions") or []
        all_funcs = evidence.get("functions") or []
        # Failing-line symbols are stronger evidence than whole-code symbols.
        return _dedup(line_funcs + all_funcs)[:max_funcs]

    def _expand_context_functions(
        self,
        funcs: List[str],
        evidence: Dict,
        error_msg: str,
        generated_code: str,
        original_query: str,
        max_funcs: int,
    ) -> List[str]:
        """Add small, error-grounded companion APIs without broad code scan."""
        max_funcs = max(1, min(int(max_funcs or ROUTED_MAX_API_FUNCS), ROUTED_MAX_API_FUNCS))
        text = " ".join(
            s for s in [
                error_msg or "",
                evidence.get("failing_line", "") or "",
            ]
            if s
        )
        return self.kb.expand_context_functions(funcs, text, max_funcs=max_funcs)

    def _detect_boundary_cards(
        self,
        error_msg: str,
        generated_code: str,
        original_query: str,
        failing_line: str,
    ) -> List[str]:
        return self.kb.boundary_contract_ids_for_evidence(
            error_msg=error_msg,
            generated_code="",
            original_query="",
            failing_line=failing_line,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Reactive Injector
# ─────────────────────────────────────────────────────────────────────────────

class ReactiveInjector:
    """
    Builds reactive fix-round doc injection strings for augmented fix prompts.

    Unlike ProactiveInjector, ReactiveInjector:
      - Is model-agnostic (no knowledge profiles, no demand predictions)
      - Operates ONLY on fix rounds, not round 0
      - Keeps legacy always-doc conditions (FR/FD/FE) for ablations
      - Adds FDR to route fixes by error-side demand evidence
      - Adds FS/FDRS to diagnose executed-but-wrong value failures

    The injected string is passed as `retrieved_docs` to `build_fix_prompt()`.
    """

    # Total layers to inject per fix condition (overridden per-error-class in FD)
    CONDITION_LAYERS: Dict[str, int] = {
        "FX": 0,   # no injection — pass-through to build_fix_prompt with empty docs
        "FR": 2,   # L0 + L1 (name + signature)
        "FD": -1,  # error-class-specific depth via ERROR_CLASS_LAYERS (see get_injection)
        "FDR": -2, # demand-routed basic/API-doc/boundary decision
        "FS": -3,  # semantic diagnosis only for executed-but-wrong value bugs
        "FDRS": -4,# FDR for runtime/API bugs, semantic diagnosis for value bugs
        "FE": 4,   # L0 + L1 + L2 + L3 (all layers including examples)
    }

    def __init__(
        self,
        snippets: Dict[str, Dict[str, InjectionSnippet]],
        fix_condition: str = "FD",
        known_functions: Optional[Set[str]] = None,
        max_funcs: int = 5,
        knowledge_base: LibraryKnowledgeBase = None,
    ):
        """
        Args:
            snippets:        {layer: {func_name: InjectionSnippet}} from InjectionGenerator
            fix_condition:   one of FX/FR/FD/FDR/FS/FDRS/FE
            known_functions: set of valid target-library function names (from snippet keys)
        """
        if fix_condition not in self.CONDITION_LAYERS:
            raise ValueError(
                f"Invalid fix_condition {fix_condition!r}. Use one of: FX, FR, FD, FDR, FS, FDRS, FE"
            )
        self.snippets = snippets
        self.fix_condition = fix_condition
        self.max_funcs = max(1, int(max_funcs))
        self.kb = knowledge_base or LibraryKnowledgeBase({"functions": []})
        self.parser = ErrorParser(known_functions=known_functions, knowledge_base=self.kb)
        self.router = FixDemandRouter(self.parser, knowledge_base=self.kb)
        self.semantic_analyzer = SemanticDemandAnalyzer(knowledge_base=self.kb)

    @classmethod
    def from_paths(
        cls,
        docs_path: str,
        fix_condition: str = "FD",
        library_name: str = None,
        max_funcs: int = 5,
    ) -> "ReactiveInjector":
        """
        Build a ReactiveInjector by loading and generating snippets from docs JSON.

        Args:
            docs_path:     path to a standardized API-spec JSON
            fix_condition: FX / FR / FD / FDR / FS / FDRS / FE  (default: FD)
            library_name:  library prefix for snippet generation (default: JSON metadata)
            max_funcs:     maximum implicated functions to document per fix prompt
        """
        kb = LibraryKnowledgeBase.from_json(docs_path)
        library_name = library_name or kb.library_name
        print(f"  [ReactiveInjector] Generating snippets from: {docs_path}  cond={fix_condition}")
        gen = InjectionGenerator(docs_path, library_name=library_name)
        snippets = gen.generate_all_snippets()

        # Collect all known function names across all layers
        known: Set[str] = set()
        for layer_dict in snippets.values():
            known.update(layer_dict.keys())

        return cls(
            snippets=snippets,
            fix_condition=fix_condition,
            known_functions=known,
            max_funcs=max_funcs,
            knowledge_base=kb,
        )

    def get_injection(
        self,
        error_type: str,
        error_msg: str,
        generated_code: str = "",
        original_query: str = "",
        max_funcs: int = None,
    ) -> str:
        """
        Build the documentation injection string for one fix-round prompt.

        Pass the returned string as `retrieved_docs` in `build_fix_prompt()`.

        Args:
            error_type:      e.g. "AttributeError", "TypeError" (from exec_result["error_type"])
            error_msg:       the compact error string (from parse_error() in utils.py)
            generated_code:  the buggy generated code (for fallback API-call scanning)
            original_query:  original NL task, used only by FDR boundary routing
            max_funcs:       maximum functions to inject docs for (default: self.max_funcs)

        Returns:
            Documentation string, or "" if FX condition or no relevant functions found.
        """
        return self.get_injection_details(
            error_type=error_type,
            error_msg=error_msg,
            generated_code=generated_code,
            original_query=original_query,
            max_funcs=max_funcs,
        )["docs"]

    def get_injection_details(
        self,
        error_type: str,
        error_msg: str,
        generated_code: str = "",
        original_query: str = "",
        max_funcs: int = None,
    ) -> Dict:
        """
        Build documentation plus a compact accounting record for one fix prompt.

        Returns:
            {
              "docs": str,
              "functions": [str],
              "error_class": str,
              "layers": [str],
              "n_layers": int,
              "route": str,
              "route_reason": str,
              "boundary_cards": [str],
              "semantic_signals": [str],
              "doc_chars": int,  # token counts are added by probe_runner's backend tokenizer
            }
        """
        max_funcs = self.max_funcs if max_funcs is None else max(1, int(max_funcs))
        error_class = self.parser.normalise_error_class(error_type, error_msg)
        empty = {
            "docs": "",
            "functions": [],
            "error_class": error_class,
            "layers": [],
            "n_layers": 0,
            "route": ROUTE_BASIC if self.fix_condition in ("FX", "FDR", "FS", "FDRS") else "legacy_empty",
            "route_reason": "no reactive injection" if self.fix_condition == "FX" else "",
            "boundary_cards": [],
            "semantic_signals": [],
            "semantic_repair_actions": [],
            "semantic_contract_ids": [],
            "semantic_expected_contract": {},
            "semantic_observed_contract": {},
            "source": "none" if self.fix_condition == "FX" else "reactive_empty",
            "doc_chars": 0,
        }

        if self.fix_condition == "FX":
            return empty

        if self.fix_condition == "FDR":
            return self._get_routed_injection_details(
                error_type=error_type,
                error_msg=error_msg,
                generated_code=generated_code,
                original_query=original_query,
                max_funcs=max_funcs,
            )

        if self.fix_condition == "FS":
            return self._get_semantic_injection_details(
                error_type=error_type,
                error_msg=error_msg,
                generated_code=generated_code,
                original_query=original_query,
            )

        if self.fix_condition == "FDRS":
            return self._get_routed_semantic_injection_details(
                error_type=error_type,
                error_msg=error_msg,
                generated_code=generated_code,
                original_query=original_query,
                max_funcs=max_funcs,
            )

        n_layers = self.CONDITION_LAYERS[self.fix_condition]

        func_names, error_class = self.parser.parse(
            error_type, error_msg, generated_code
        )

        if not func_names:
            empty["error_class"] = error_class
            return empty

        # Determine layer depth
        if self.fix_condition == "FD":
            n_layers = ERROR_CLASS_LAYERS.get(error_class, ERROR_CLASS_LAYERS["default"])
        # FX already returned above; n_layers is now set for FR/FE/FD

        if n_layers == 0:
            # SyntaxError or FX → no injection
            empty["functions"] = func_names[:max_funcs]
            empty["error_class"] = error_class
            empty["route"] = "legacy_no_doc"
            empty["route_reason"] = f"{error_class} maps to zero documentation layers"
            return empty

        layers_to_include = LAYER_ORDER[:n_layers]

        # Build one block per function
        blocks = self._build_function_blocks(func_names[:max_funcs], layers_to_include)

        if not blocks:
            empty["functions"] = func_names[:max_funcs]
            empty["error_class"] = error_class
            empty["layers"] = layers_to_include
            empty["n_layers"] = n_layers
            empty["route"] = "legacy_empty"
            empty["route_reason"] = "no snippets found for implicated functions"
            return empty

        header = (
            "[Relevant API Documentation]\n"
            f"{self.kb.library_name} documentation for functions implicated in the {error_class} "
            f"(layers: {', '.join(layers_to_include)}):\n\n"
        )
        docs = header + "\n\n".join(blocks)
        return {
            "docs": docs,
            "functions": func_names[:max_funcs],
            "error_class": error_class,
            "layers": layers_to_include,
            "n_layers": n_layers,
            "route": "legacy_api_doc",
            "route_reason": f"legacy {self.fix_condition} always-doc reactive injection",
            "boundary_cards": [],
            "semantic_signals": [],
            "semantic_repair_actions": [],
            "semantic_contract_ids": [],
            "semantic_expected_contract": {},
            "semantic_observed_contract": {},
            "source": "reactive",
            "doc_chars": len(docs),
        }

    def _get_routed_injection_details(
        self,
        error_type: str,
        error_msg: str,
        generated_code: str = "",
        original_query: str = "",
        max_funcs: int = None,
    ) -> Dict:
        max_funcs = self.max_funcs if max_funcs is None else max(1, int(max_funcs))
        plan, evidence = self.router.route(
            error_type=error_type,
            error_msg=error_msg,
            generated_code=generated_code,
            original_query=original_query,
            max_funcs=max_funcs,
        )
        plan_dict = asdict(plan)
        docs = ""

        if plan.route == ROUTE_API_DOC:
            evidence_text = " ".join(
                part
                for part in [
                    original_query or "",
                    error_msg or "",
                    evidence.get("failing_line", "") or "",
                    generated_code or "",
                ]
                if part
            )
            blocks = self._build_function_blocks(
                plan.functions,
                plan.layers,
                compact_l2=True,
                evidence_text=evidence_text,
            )
            if blocks:
                boundary_docs = ""
                cards = self.kb.boundary_card_texts(plan.boundary_cards)
                if cards:
                    boundary_docs = (
                        "\n\n[Fix-Demand Boundary Contract]\n"
                        + "\n\n".join(cards)
                    )
                docs = (
                    "[Fix-Demand API Documentation]\n"
                    f"Route: {plan.route}. Reason: {plan.reason}.\n"
                    f"{self.kb.library_name} documentation for implicated functions "
                    f"(layers: {', '.join(plan.layers)}):\n\n"
                    + "\n\n".join(blocks)
                    + boundary_docs
                )
            else:
                plan_dict["source"] = "fix_demand_empty"

        elif plan.route == ROUTE_BOUNDARY:
            cards = self.kb.boundary_card_texts(plan.boundary_cards)
            if cards:
                docs = (
                    "[Fix-Demand Boundary Contract]\n"
                    f"Route: {plan.route}. Reason: {plan.reason}.\n\n"
                    + "\n\n".join(cards)
                )
            else:
                plan_dict["source"] = "fix_demand_empty"

        return {
            "docs": docs,
            "functions": plan.functions,
            "error_class": plan.error_class,
            "layers": plan.layers,
            "n_layers": plan.n_layers,
            "route": plan.route,
            "route_reason": plan.reason,
            "boundary_cards": plan.boundary_cards,
            "semantic_signals": [],
            "semantic_repair_actions": [],
            "semantic_contract_ids": [],
            "semantic_expected_contract": {},
            "semantic_observed_contract": {},
            "source": plan_dict.get("source", plan.source),
            "line_no": evidence.get("line_no"),
            "failing_line": evidence.get("failing_line", ""),
            "doc_chars": len(docs),
        }

    def _get_semantic_injection_details(
        self,
        error_type: str,
        error_msg: str,
        generated_code: str = "",
        original_query: str = "",
    ) -> Dict:
        error_class = self.parser.normalise_error_class(error_type, error_msg)
        plan = self.semantic_analyzer.analyze(
            original_query=original_query,
            generated_code=generated_code,
            error_msg=error_msg,
            error_type=error_class,
        )
        if not plan.active:
            return {
                "docs": "",
                "functions": [],
                "error_class": error_class,
                "layers": [],
                "n_layers": 0,
                "route": ROUTE_BASIC,
                "route_reason": plan.reason,
                "boundary_cards": [],
                "semantic_signals": [],
                "semantic_repair_actions": [],
                "semantic_contract_ids": [],
                "source": "semantic_empty",
                "semantic_expected_contract": plan.expected_contract,
                "semantic_observed_contract": plan.observed_contract,
                "doc_chars": 0,
            }

        docs = self.semantic_analyzer.render(plan)
        evidence_text = " ".join(
            part
            for part in [original_query or "", error_msg or "", generated_code or ""]
            if part
        )
        boundary_ids = self.kb.boundary_contract_ids_for_semantic_context(
            plan.contract_ids,
            intent_ids=plan.matched_intents,
        )
        boundary_cards = self.kb.boundary_card_texts(boundary_ids)
        if boundary_cards:
            docs += (
                "\n\n[Trace-Grounded Boundary Context]\n"
                + "\n\n".join(boundary_cards)
            )

        context_functions = self._semantic_context_functions(plan)
        api_blocks = []
        if context_functions:
            api_blocks = self._build_function_blocks(
                context_functions[: max(1, min(self.max_funcs, ROUTED_MAX_API_FUNCS))],
                ["L1", "L2"],
                compact_l2=True,
                evidence_text=evidence_text,
            )
        if api_blocks:
            docs += (
                "\n\n[Trace-Grounded API Context]\n"
                + "\n\n".join(api_blocks)
            )
        return {
            "docs": docs,
            "functions": list(context_functions),
            "error_class": error_class,
            "layers": ["L1", "L2"] if api_blocks else [],
            "n_layers": 2 if api_blocks else 0,
            "route": ROUTE_SEMANTIC,
            "route_reason": plan.reason,
            "boundary_cards": boundary_ids,
            "semantic_signals": plan.signals,
            "semantic_repair_actions": plan.repair_actions,
            "semantic_contract_ids": plan.contract_ids,
            "semantic_intents": plan.matched_intents,
            "semantic_anchor_functions": plan.anchor_functions,
            "semantic_expected_contract": plan.expected_contract,
            "semantic_observed_contract": plan.observed_contract,
            "source": "trace_grounded_value_repair",
            "doc_chars": len(docs),
        }

    def _semantic_context_functions(self, plan) -> List[str]:
        funcs: List[str] = []
        funcs.extend(plan.anchor_functions or [])
        if getattr(plan, "source", "") != "trace_grounded_value_repair":
            funcs.extend(self.kb.intent_anchor_functions(plan.matched_intents or []))
            for cid in plan.contract_ids or []:
                contract = self.kb.contract_for_id(cid)
                if not contract:
                    continue
                if str(contract.get("kind") or "") == "function_contract":
                    funcs.append(str(contract.get("name") or ""))
        return [
            fn for fn in _dedup(self.kb.canonical_name(f) for f in funcs)
            if self.kb.known(fn)
        ]

    def _get_routed_semantic_injection_details(
        self,
        error_type: str,
        error_msg: str,
        generated_code: str = "",
        original_query: str = "",
        max_funcs: int = None,
    ) -> Dict:
        semantic = self._get_semantic_injection_details(
            error_type=error_type,
            error_msg=error_msg,
            generated_code=generated_code,
            original_query=original_query,
        )
        if semantic.get("docs"):
            return semantic
        return self._get_routed_injection_details(
            error_type=error_type,
            error_msg=error_msg,
            generated_code=generated_code,
            original_query=original_query,
            max_funcs=max_funcs,
        )

    def _build_function_blocks(
        self,
        func_names: List[str],
        layers: List[str],
        compact_l2: bool = False,
        evidence_text: str = "",
    ) -> List[str]:
        blocks: List[str] = []
        for fn in func_names:
            lines: List[str] = []
            if compact_l2 and "L2" in layers:
                compact = self._compact_l2_snippet_block(fn, evidence_text=evidence_text)
                if compact:
                    lines.append(compact)
                    if "L3" in layers:
                        snip = self.snippets.get("L3", {}).get(fn)
                        if snip and "No example available" not in snip.content:
                            lines.append(snip.content)
                    blocks.append("\n".join(lines))
                    continue
            for layer in layers:
                if compact_l2 and layer == "L0" and "L1" in layers:
                    continue
                snip = self.snippets.get(layer, {}).get(fn)
                if snip:
                    lines.append(snip.content)
            hint = self._build_call_path_hint(fn) if "L1" in layers else ""
            if hint:
                insert_at = 1 if lines else 0
                lines.insert(insert_at, hint)
            if lines:
                blocks.append("\n".join(lines))
        return blocks

    def _first_sentence(self, text: str, limit: int = 220) -> str:
        clean = " ".join(str(text or "").split())
        if not clean:
            return ""
        match = re.search(r"(.{1,%d}?[.!?])(?:\s|$)" % limit, clean)
        return match.group(1) if match else clean[:limit].rstrip()

    def _shorten(self, text: str, limit: int = 150) -> str:
        clean = " ".join(str(text or "").split())
        if len(clean) <= limit:
            return clean
        return clean[: max(0, limit - 3)].rstrip() + "..."

    def _param_is_required(self, param: dict) -> bool:
        return bool(param.get("required"))

    def _param_matches_text(self, param: dict, text: str) -> bool:
        evidence = (text or "").lower()
        name = str(param.get("name") or "").lower()
        if not name or name.startswith("*"):
            return False
        pieces = [name] + [p for p in re.split(r"[_\W]+", name) if len(p) >= 3]
        return any(piece and piece in evidence for piece in pieces)

    def _compact_l2_snippet_block(self, func_name: str, evidence_text: str = "") -> str:
        """Compact FDR L2 renderer: only the contract needed by the failure evidence."""
        func_info = self.kb.func_map.get(func_name) or {}
        if not func_info:
            return ""

        desc = self._first_sentence(func_info.get("description", ""))
        full_path = func_info.get("full_path") or f"{self.kb.library_name}.{func_name}"
        params = [
            p for p in list(func_info.get("parameters") or [])
            if (
                isinstance(p, dict)
                and not str(p.get("name", "")).startswith("*")
                and str(p.get("name", "")).lower() not in {"kwargs", "**kwargs"}
            )
        ]

        required = [p for p in params if self._param_is_required(p)]
        evidence_params = [p for p in params if self._param_matches_text(p, evidence_text)]
        selected_params: List[dict] = []
        for param in required + evidence_params:
            if param not in selected_params:
                selected_params.append(param)
        for param in params:
            if len(selected_params) >= 6:
                break
            if param not in selected_params:
                selected_params.append(param)

        call_args = [str(p.get("name")) for p in selected_params if p.get("name")]
        omitted = len(params) > len(selected_params)
        ellipsis = ", ..." if omitted and call_args else "..." if omitted else ""
        call = f"{full_path}({', '.join(call_args)}{ellipsis})"

        lines = [f"[{self.kb.library_name}.{func_name} | L2: compact API contract]"]
        if desc:
            lines.append(f"Purpose: {desc}")
        lines.append(f"Call contract: `{call}`")
        if selected_params:
            lines.append("Key parameters:")
            for param in selected_params:
                name = str(param.get("name") or "")
                ptype = str(param.get("type") or "").strip()
                default = str(param.get("default", "")).strip()
                desc_text = self._shorten(param.get("description", ""), 130)
                tags = []
                if self._param_is_required(param):
                    tags.append("required")
                elif default and default.lower() not in {"none", "null"}:
                    tags.append(f"default={default}")
                detail = f" ({ptype})" if ptype and ptype.lower() != "any" else ""
                suffix = f" [{'; '.join(tags)}]" if tags else ""
                if desc_text:
                    lines.append(f"- `{name}`{detail}: {desc_text}{suffix}")
                else:
                    lines.append(f"- `{name}`{detail}{suffix}")
        for ret in list(func_info.get("return_info") or []):
            if isinstance(ret, dict) and ret.get("description"):
                lines.append(f"Return/side effect: {self._shorten(ret.get('description'), 160)}")
                break
        return "\n".join(lines)

    def _build_call_path_hint(self, func_name: str) -> str:
        """Derive a concise import/call hint from the documented full path."""
        snip = self.snippets.get("L1", {}).get(func_name)
        if not snip:
            return ""
        lib = re.escape(self.kb.library_name)
        m = re.search(rf"`({lib}(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\s*\(", snip.content)
        if not m:
            return ""
        full_path = m.group(1)
        parts = full_path.split(".")
        if len(parts) <= 2:
            return ""
        module_path = ".".join(parts[:-1])
        call_name = parts[-1]
        return (
            f"Call path hint: `{call_name}` is documented at `{full_path}`. "
            f"Use `from {module_path} import {call_name}` or call it through "
            f"that module path; do not invent a root-level alias call "
            "unless the signature documents it there."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dedup(seq: List[str]) -> List[str]:
    """Deduplicate a list preserving insertion order."""
    seen: Set[str] = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
