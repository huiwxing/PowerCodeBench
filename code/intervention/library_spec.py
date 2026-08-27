# --------------------------------------------------------------------------
# Repository copy of knowledge_injection/library_spec.py, unmodified apart
# from this header. Runs CPU-only against the archived corpus/spec files in
# this repository. See code/README.md for the module map.
# --------------------------------------------------------------------------
"""
Library specification helpers for knowledge injection.

The intervention algorithms should not hardcode one library's API folklore.
This module loads a standardized API-spec JSON plus one derived
library-knowledge artifact and exposes generic matching primitives for
proactive and reactive injection.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _dedup(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


def _matches_pattern(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    except re.error:
        return pattern.lower() in text.lower()


def _matches_any(patterns: Sequence[str], text: str) -> bool:
    return any(_matches_pattern(p, text) for p in patterns or [])


def _contains_any(terms: Sequence[str], text: str) -> bool:
    lower = text.lower()
    return any((term or "").lower() in lower for term in terms or [])


@dataclass(frozen=True)
class BoundaryContract:
    """A compact cross-API or object-schema contract for one target library."""

    id: str
    title: str
    contract: List[str]
    related_functions: List[str]
    intent_ids: List[str]
    intent_patterns: List[str]
    evidence_terms: List[str]
    evidence_patterns: List[str]
    task_tags: List[str]

    @classmethod
    def from_dict(cls, raw: Dict) -> "BoundaryContract":
        return cls(
            id=str(raw.get("id") or ""),
            title=str(raw.get("title") or raw.get("id") or "Boundary Contract"),
            contract=list(raw.get("contract") or raw.get("contract_lines") or []),
            related_functions=list(raw.get("related_functions") or []),
            intent_ids=list(raw.get("intent_ids") or []),
            intent_patterns=list(raw.get("intent_patterns") or []),
            evidence_terms=list(raw.get("evidence_terms") or []),
            evidence_patterns=list(raw.get("evidence_patterns") or []),
            task_tags=list(raw.get("task_tags") or []),
        )

    def render(self, library_name: str = "") -> str:
        prefix = f"{library_name} " if library_name else ""
        lines = [f"[Boundary Card: {prefix}{self.title}]"]
        lines.extend(f"- {line}" for line in self.contract if line)
        return "\n".join(lines)


class LibraryKnowledgeBase:
    """
    Data-backed library knowledge used by injection algorithms.

    Required JSON field:
      - functions: standardized function/class API entries

    Optional top-level fields:
      - library / library_name
      - derived_knowledge_path / library_knowledge
    """

    def __init__(self, data: Dict, source_path: str = ""):
        self.data = data
        self.source_path = source_path
        self.library_name = (
            data.get("library")
            or data.get("library_name")
            or ("pandapower" if "pandapower" in str(source_path).lower() else "unknown")
        )
        self.functions = list(data.get("functions") or [])
        self.function_names = {
            f.get("name")
            for f in self.functions
            if isinstance(f, dict) and f.get("name")
        }
        self.func_map = {
            f.get("name"): f
            for f in self.functions
            if isinstance(f, dict) and f.get("name")
        }
        self.library_knowledge = self._load_library_knowledge()
        self.knowledge_metadata = dict(self.library_knowledge.get("metadata") or {})
        runtime = dict(self.library_knowledge.get("runtime") or {})
        self.aliases = {
            str(k).lower(): str(v)
            for k, v in (runtime.get("aliases") or {}).items()
            if k and v
        }
        self.import_aliases = _dedup(
            [self.library_name]
            + [str(a) for a in (runtime.get("import_aliases") or []) if a]
        )
        self.system_prompt = str(runtime.get("system_prompt_code") or "")
        self.api_intents = [
            i for i in list(runtime.get("api_intents") or [])
            if isinstance(i, dict)
        ]
        self.context_expansions = [
            i for i in list(runtime.get("context_expansions") or [])
            if isinstance(i, dict)
        ]
        self.contract_index = {
            "version": self.library_knowledge.get("version", "library_knowledge_v1"),
            "library": self.library_knowledge.get("library", self.library_name),
            "summary": self.library_knowledge.get("contract_summary") or {},
            "contracts": list(self.library_knowledge.get("contracts") or []),
        }
        self.derived_contracts = [
            c for c in list(self.contract_index.get("contracts") or [])
            if isinstance(c, dict) and c.get("id")
        ]
        self.derived_contract_map = {
            str(c.get("id")): c for c in self.derived_contracts
        }
        self.boundary_contracts = [
            BoundaryContract.from_dict(c)
            for c in list(runtime.get("boundary_contracts") or [])
            if isinstance(c, dict) and c.get("id")
        ]

    def _load_library_knowledge(self) -> Dict:
        """Load the derived library-knowledge artifact attached to the API spec."""
        inline = self.data.get("library_knowledge")
        if isinstance(inline, dict):
            return inline
        path_value = self.data.get("derived_knowledge_path")
        if not path_value:
            return {}
        path = str(path_value)
        candidates = []
        if self.source_path:
            candidates.append(PathLike(self.source_path).parent_join(path))
        candidates.append(path)
        for candidate in candidates:
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    return loaded
            except FileNotFoundError:
                continue
            except Exception:
                continue
        return {}

    @classmethod
    def from_json(cls, path: str) -> "LibraryKnowledgeBase":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data, source_path=path)

    def system_prompt_code(self) -> str:
        if self.system_prompt:
            return self.system_prompt
        return (
            f"You are a Python code generator for the {self.library_name} library. "
            "Given a natural language task description, write complete, executable Python code. "
            "Output ONLY executable Python code. Include all necessary imports. "
            "The last print() statement should output ONLY the requested numeric value."
        )

    def canonical_name(self, name: str) -> str:
        clean = (name or "").strip()
        return self.aliases.get(clean.lower(), clean)

    def known(self, name: str) -> bool:
        return self.canonical_name(name) in self.function_names

    def query_anchor_functions(
        self,
        query: str,
        known_predicate: Optional[Callable[[str], bool]] = None,
        extra_functions: Optional[Iterable[str]] = None,
    ) -> List[str]:
        q = query or ""
        funcs: List[str] = list(extra_functions or [])
        for intent in self.api_intents:
            if not isinstance(intent, dict):
                continue
            patterns = list(intent.get("patterns") or [])
            exclude_patterns = list(intent.get("exclude_patterns") or [])
            if patterns and not _matches_any(patterns, q):
                continue
            if exclude_patterns and _matches_any(exclude_patterns, q):
                continue
            funcs.extend(intent.get("anchor_functions") or [])

        out: List[str] = []
        for fn in _dedup(self.canonical_name(f) for f in funcs):
            if known_predicate is None or known_predicate(fn):
                out.append(fn)
        return out

    def prune_intent_mismatches(self, query: str, merged: Dict[str, dict]) -> None:
        q = query or ""
        for item in self.api_intents:
            if not isinstance(item, dict):
                continue
            funcs = [self.canonical_name(f) for f in item.get("gated_functions") or []]
            if not funcs:
                continue
            patterns = list(item.get("patterns") or [])
            exclude_patterns = list(item.get("exclude_patterns") or [])
            keep = (not patterns or _matches_any(patterns, q)) and not _matches_any(exclude_patterns, q)
            if keep:
                continue
            for fn in funcs:
                merged.pop(fn, None)

    def matched_workflow_ids(self, query: str) -> List[str]:
        q = query or ""
        ids: List[str] = []
        for intent in self.api_intents:
            if not isinstance(intent, dict):
                continue
            if _matches_any(list(intent.get("patterns") or []), q) and not _matches_any(list(intent.get("exclude_patterns") or []), q):
                ids.append(str(intent.get("id") or "intent"))
        return _dedup(ids)

    def workflow_suppressed_functions(self, query: str) -> List[str]:
        q = query or ""
        funcs: List[str] = []
        for intent in self.api_intents:
            if not isinstance(intent, dict):
                continue
            if _matches_any(list(intent.get("patterns") or []), q) and not _matches_any(list(intent.get("exclude_patterns") or []), q):
                funcs.extend(intent.get("suppress_functions") or [])
        return _dedup(self.canonical_name(f) for f in funcs)

    def expand_context_functions(
        self,
        funcs: List[str],
        evidence_text: str,
        max_funcs: int,
    ) -> List[str]:
        expanded = list(funcs)
        canonical = set(self.canonical_name(f) for f in funcs)
        rules = list(self.context_expansions or []) + list(self.api_intents or [])
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            required = {
                self.canonical_name(f)
                for f in (rule.get("if_functions") or rule.get("context_if_functions") or [])
            }
            if required and not (required & canonical):
                continue
            patterns = list(rule.get("patterns") or [])
            terms = list(rule.get("terms") or [])
            if patterns and not _matches_any(patterns, evidence_text):
                continue
            if terms and not _contains_any(terms, evidence_text):
                continue
            add = rule.get("add_functions") or rule.get("context_add_functions") or []
            expanded = [self.canonical_name(f) for f in add] + expanded
        return _dedup(expanded)[:max(1, int(max_funcs))]

    def boundary_card_entries(
        self,
        query: str,
        funcs: Optional[Iterable[str]] = None,
        task: str = "",
    ) -> List[Tuple[str, str, int]]:
        matched = self.boundary_contract_ids_for_query(query, funcs=funcs, task=task)
        return [
            (contract.id, contract.render(self.library_name), _token_estimate(contract.render(self.library_name)))
            for contract in self.boundary_contracts
            if contract.id in matched
        ]

    def boundary_contract_ids_for_query(
        self,
        query: str,
        funcs: Optional[Iterable[str]] = None,
        task: str = "",
    ) -> List[str]:
        q = query or ""
        task_norm = (task or "").lower()
        func_set = {self.canonical_name(f) for f in funcs or []}
        matched_intents = set(self.matched_workflow_ids(q))
        matched: List[str] = []
        for contract in self.boundary_contracts:
            has_task = bool(task_norm and task_norm in {t.lower() for t in contract.task_tags})
            contract_intents = set(getattr(contract, "intent_ids", []))
            has_intent = bool(contract_intents & matched_intents) or _matches_any(contract.intent_patterns, q)
            has_related_func = bool(func_set and func_set & set(contract.related_functions))
            if has_task or has_intent:
                matched.append(contract.id)
            elif has_related_func and _matches_any(contract.intent_patterns, q):
                matched.append(contract.id)
        return _dedup(matched)

    def boundary_contract_ids_for_evidence(
        self,
        error_msg: str = "",
        generated_code: str = "",
        original_query: str = "",
        failing_line: str = "",
    ) -> List[str]:
        text = " ".join(
            s for s in [error_msg or "", generated_code or "", original_query or "", failing_line or ""]
            if s
        )
        matched: List[str] = []
        for contract in self.boundary_contracts:
            if (
                _contains_any(contract.evidence_terms, text)
                or _matches_any(contract.evidence_patterns, text)
                or _matches_any(contract.intent_patterns, original_query or "")
            ):
                matched.append(contract.id)
        return _dedup(matched)

    def boundary_card_texts(self, ids: Iterable[str]) -> List[str]:
        wanted = set(ids or [])
        return [
            contract.render(self.library_name)
            for contract in self.boundary_contracts
            if contract.id in wanted
        ]

    def boundary_contract_ids_for_semantic_context(
        self,
        contract_ids: Iterable[str],
        intent_ids: Iterable[str] = (),
    ) -> List[str]:
        """Find boundary cards that make semantic contract repairs actionable."""
        wanted: List[str] = []
        matched_intents = set(str(i) for i in intent_ids or [])
        for contract in self.boundary_contracts:
            if set(contract.intent_ids or []) & matched_intents:
                wanted.append(contract.id)

        for cid in _dedup(str(c) for c in contract_ids or []):
            contract = self.contract_for_id(cid)
            if not contract:
                continue
            kind = str(contract.get("kind") or "")
            name = str(contract.get("name") or "")
            table = str(contract.get("table") or "")
            field = str(contract.get("field") or "")
            text = " ".join(
                part for part in [
                    cid,
                    kind,
                    name,
                    table,
                    field,
                    str(contract.get("read_expression") or ""),
                    " ".join(contract.get("contract") or []),
                ] if part
            )
            if kind in {"output_observable", "object_schema", "object_field", "state_mutation"}:
                wanted.append("table_schema")
            if (
                "sc" in table.lower()
                or "ikss" in field.lower()
                or "short circuit" in text.lower()
            ):
                wanted.append("short_circuit")
            if "res_cost" in text or "opf_cost" in matched_intents:
                wanted.append("opf_cost")
            if "time_series" in matched_intents:
                wanted.append("time_series")

        existing = {contract.id for contract in self.boundary_contracts}
        return _dedup(cid for cid in wanted if cid in existing)

    def intent_anchor_functions(self, intent_ids: Iterable[str]) -> List[str]:
        """Return generated workflow anchors for matched intent ids."""
        wanted = set(str(i) for i in intent_ids or [])
        funcs: List[str] = []
        for intent in self.api_intents:
            if not isinstance(intent, dict):
                continue
            if str(intent.get("id") or "") not in wanted:
                continue
            funcs.extend(intent.get("anchor_functions") or [])
        return _dedup(
            self.canonical_name(fn)
            for fn in funcs
            if self.known(self.canonical_name(fn))
        )

    # ── Derived contract-index helpers ──────────────────────────────────

    def contracts_by_kind(self, kind: str) -> List[Dict]:
        return [
            c for c in self.derived_contracts
            if str(c.get("kind") or "") == kind
        ]

    def contract_for_id(self, contract_id: str) -> Optional[Dict]:
        return self.derived_contract_map.get(str(contract_id or ""))

    def contract_ids_for_query(
        self,
        query: str,
        *,
        kinds: Optional[Iterable[str]] = None,
        limit: int = 3,
    ) -> List[str]:
        kind_set = {str(k) for k in kinds or []}
        text = query or ""
        scored: List[Tuple[float, str]] = []
        for contract in self.derived_contracts:
            kind = str(contract.get("kind") or "")
            if kind_set and kind not in kind_set:
                continue
            score = self._contract_match_score(contract, text, use_query=True)
            if score > 0:
                scored.append((score, str(contract.get("id"))))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return _dedup(cid for _, cid in scored)[: max(1, int(limit or 3))]

    def render_contract_ids(self, ids: Iterable[str], *, max_lines_per_contract: int = 3) -> List[str]:
        rendered: List[str] = []
        for cid in _dedup(str(i) for i in ids or []):
            contract = self.contract_for_id(cid)
            if not contract:
                continue
            title = str(contract.get("name") or cid)
            source = str(contract.get("source") or "derived")
            lines = [f"[Derived Contract: {title} | {source}]"]
            for line in list(contract.get("contract") or [])[: max(1, int(max_lines_per_contract))]:
                if line:
                    lines.append(f"- {line}")
            rendered.append("\n".join(lines))
        return rendered

    def _contract_match_score(self, contract: Dict, text: str, *, use_query: bool) -> float:
        if not text:
            return 0.0
        patterns = list(contract.get("query_patterns" if use_query else "code_patterns") or [])
        score = 0.0
        if _matches_any(patterns, text):
            score += 1.0
        name_tokens = set(_simple_tokens(str(contract.get("name") or "")))
        field = str(contract.get("field") or "")
        if field:
            name_tokens.update(_simple_tokens(field))
        text_tokens = set(_simple_tokens(text))
        overlap = len(name_tokens & text_tokens)
        if overlap:
            score += min(0.8, 0.2 * overlap)
        if use_query:
            score -= _context_penalty(contract, text)
        try:
            score *= float(contract.get("confidence", 0.7))
        except (TypeError, ValueError):
            score *= 0.7
        return max(0.0, score)


class PathLike(str):
    """Tiny path helper to avoid importing pathlib into older call sites."""

    def parent_join(self, child: str) -> str:
        import os

        return os.path.join(os.path.dirname(str(self)), child)


def _simple_tokens(text: str) -> List[str]:
    return [
        p for p in re.split(r"[^A-Za-z0-9]+", str(text or "").lower())
        if len(p) >= 2 and p not in {
            "all", "and", "for", "from", "in", "net", "of", "res", "the", "to", "with"
        }
    ]


def _context_penalty(contract: Dict, text: str) -> float:
    name = str(contract.get("name") or "").lower()
    query = str(text or "").lower()
    penalty = 0.0
    if ("3ph" in name or "_3ph" in name) and not re.search(
        r"\b(?:3[- ]?phase|three[- ]?phase|phase[- ]?[abc]|unbalanced|asymmetric)\b",
        query,
    ):
        penalty += 0.75
    if "asymmetric" in name and not re.search(r"\b(?:asymmetric|unbalanced|phase[- ]?[abc])\b", query):
        penalty += 0.55
    if ("_sc" in name or name.endswith(".ikss_ka")) and not re.search(
        r"\b(?:short[- ]?circuit|fault|ikss)\b",
        query,
    ):
        penalty += 0.75
    if "_est" in name and not re.search(
        r"\b(?:state\s+estimation|estimated|measurement|wls)\b",
        query,
    ):
        penalty += 0.55
    if "_dc" in name and not re.search(r"\b(?:dc|direct current)\b", query):
        penalty += 0.45
    return penalty

