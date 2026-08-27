# --------------------------------------------------------------------------
# Repository copy of knowledge_injection/proactive.py, with imports and
# data-path constants adapted to this repository's layout. Runs CPU-only
# against the archived corpus/spec files in this repository. See
# code/README.md for the module map.
# --------------------------------------------------------------------------
"""
Proactive Knowledge Injection
=============================
Builds first-pass benchmark prompts using demand x model-risk layer selection.

The algorithm is library-adaptable: function snippets come from a standardized
API-spec JSON, while optional aliases, anchors, workflow filters, and boundary
contracts are loaded through LibraryKnowledgeBase.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from intervention.library_spec import LibraryKnowledgeBase

# ============================================================
# Proactive Compensation Injector
# ============================================================

class ProactiveInjector:
    """
    Builds injection-augmented prompts for proactive compensation experiments.

    Conditions:
      A  – Baseline: no injection (same as standard benchmark)
      B  – Demand only: top-k function names listed, no doc content
      C  – Demand + layer-wise probe-risk injection (function, layer candidates)
      X  – Demand-only upper bound: all top-k functions with all available layers

    Usage:
        injector = ProactiveInjector.from_paths(
            demand_export_path="…/qwen3_480b_candidates.json",
            knowledge_profile_path="…/knowledge_profile.json",
            docs_path="dataset/pandapower_docs.json",
        )
        messages = injector.build_messages(item, condition="C")
    """

    LAYER_ORDER = ["L0", "L1", "L2", "L3"]
    RISK_WEIGHTS = {"L0": 0.20, "L1": 0.25, "L2": 0.20, "L3": 0.35}
    LAYER_LABELS = {
        "L0": "name and one-line purpose",
        "L1": "signature",
        "L2": "parameter, return, and semantic contract",
        "L3": "compact usage example",
    }
    TASK_LAYER_NEED_BASE = {"L0": 0.70, "L1": 0.95, "L2": 0.70, "L3": 0.75}
    TASK_LAYER_NEED_CAP = {"L0": 1.05, "L1": 1.15, "L2": 1.30, "L3": 1.40}
    LAYER_COST_ALPHA = 0.5
    MAX_L3_EXAMPLES_C = 2
    MAX_L3_EXAMPLES_X = 10
    COMPACT_L2_MAX_PARAMS = 6
    HIGH_DEMAND_INTERFACE_MIN = 0.85
    QUERY_ANCHOR_INTERFACE_FLOORS = {"L0": 0.45, "L1": 0.40}
    HIGH_DEMAND_INTERFACE_FLOORS = {"L0": 0.16, "L1": 0.14}
    BOUNDARY_INTERFACE_FLOORS = {"L1": 0.22, "L2": 0.18}
    TS_DEPTH_CAPS = {}

    def __init__(
        self,
        demand_by_id: Dict[str, dict],
        profile: Dict[str, Dict],
        snippets: Dict[str, Dict],
        top_k: int = 10,
        token_budget_C: int = 2000,
        token_budget_X: int = 4000,
        threshold: float = 0.5,
        token_budget_RsemB: int = 600,
        knowledge_base: LibraryKnowledgeBase = None,
        risk_weights: Optional[Dict[str, float]] = None,
        bm25_predictor=None,
        sbert_predictor=None,
        risk_uniform: bool = False,
        recal_name_layer_floor: float = 0.0,
    ):
        self.demand_by_id = demand_by_id
        self.profile = profile
        self.snippets = snippets
        self.top_k = top_k
        self.token_budget_C = token_budget_C
        self.token_budget_X = token_budget_X
        self.token_budget_RsemB = token_budget_RsemB
        self.threshold = threshold
        self.kb = knowledge_base or LibraryKnowledgeBase({"functions": []})
        self.library_name = self.kb.library_name
        self.system_prompt = self.kb.system_prompt_code()
        self._network_anchor_map = None
        # Per-instance L0--L3 risk weights override (C4 sensitivity ablation).
        # When None, fall back to the class-level RISK_WEIGHTS default.
        if risk_weights is not None:
            missing = set(self.LAYER_ORDER) - set(risk_weights)
            if missing:
                raise ValueError(
                    f"risk_weights must specify all of L0..L3, missing: {sorted(missing)}"
                )
            self.risk_weights = dict(risk_weights)
        else:
            self.risk_weights = dict(self.RISK_WEIGHTS)
        # Optional BM25 retriever for condition R (vanilla RAG baseline). When
        # None, condition R is disabled. Required for build_messages(condition="R").
        self.bm25_predictor = bm25_predictor
        # Optional SBERT retriever for conditions Rsem/RsemB (vanilla dense-RAG
        # baseline). When None, those conditions are disabled.
        self.sbert_predictor = sbert_predictor
        # When True, replace per-(model, function, layer) probe-derived risk
        # with a constant 1.0 throughout the intervention scoring path. This
        # is the C_risk_uniform isolation ablation (Round 5 Exp-A): keeps every
        # other component of condition C identical (demand model, anchors,
        # boundary cards, layer pruning, token budget) and removes only the
        # model-specific knowledge-probing signal.
        self.risk_uniform = bool(risk_uniform)
        # E2-R recalibration arm (E2 doc §8, prereg 2026-08-02): when > 0, the
        # L0/L1 layer deficit of every demand-hit function in condition C is
        # floored at this value, so the name/signature layers always enter the
        # packing competition regardless of probe-profile strength. This is the
        # single-axis fix for the probe→COM-accessor risk-transfer boundary
        # (probe recall decouples from accessor spelling → under-injection).
        # 0.0 (default) = exact frozen condition-C behaviour.
        self.recal_name_layer_floor = float(recal_name_layer_floor or 0.0)

    @classmethod
    def resolve_demand_export(
        cls,
        demand_suite_dir: str,
        demand_model: str = "hybrid_tfidf",
    ) -> str:
        """
        Resolve the candidates JSON path from a suite directory.

        Looks for: {demand_suite_dir}/{demand_model}/exports/*.json
        All suites store exactly one candidates file per exports/ directory,
        so this returns that file regardless of its name.

        The demand export is model-agnostic w.r.t. the target evaluation models
        (7B, 14B, 32B, etc.). It encodes which functions are likely needed for
        each query — per-target-model knowledge differences are handled separately
        via knowledge_profile.json loaded from probe results.
        """
        from pathlib import Path
        import glob as _glob
        exports_dir = Path(demand_suite_dir) / demand_model / "exports"
        candidates = _glob.glob(str(exports_dir / "*_candidates.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No *_candidates.json found in {exports_dir}. "
                f"Check that suite_dir and demand_model are correct."
            )
        if len(candidates) > 1:
            # Pick the one with the highest-capability model name as a heuristic
            candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
        return candidates[0]

    @classmethod
    def from_paths(
        cls,
        knowledge_profile_path: str,
        docs_path: str,
        demand_export_path: str = None,
        demand_suite_dir: str = None,
        demand_model: str = "hybrid_tfidf",
        top_k: int = 10,
        token_budget_C: int = 2000,
        token_budget_X: int = 4000,
        token_budget_RsemB: int = 600,
        threshold: float = 0.5,
        risk_weights: Optional[Dict[str, float]] = None,
        bm25_docs_path: Optional[str] = None,
        sbert_docs_path: Optional[str] = None,
        risk_uniform: bool = False,
        recal_name_layer_floor: float = 0.0,
    ) -> "ProactiveInjector":
        """
        Build a ProactiveInjector from file paths.

        Demand export can be specified either as:
          - demand_suite_dir + demand_model  (preferred, auto-resolves the candidates JSON)
          - demand_export_path               (direct path, legacy)
        """
        from probing.probe_framework import InjectionGenerator

        if demand_export_path is None:
            if demand_suite_dir is None:
                raise ValueError(
                    "Provide either demand_export_path or demand_suite_dir + demand_model"
                )
            demand_export_path = cls.resolve_demand_export(demand_suite_dir, demand_model)

        print(f"  [ProactiveInjector] Loading demand export: {demand_export_path}")
        with open(demand_export_path) as f:
            demand_list = json.load(f)
        demand_by_id = {item["id"]: item for item in demand_list}

        print(f"  [ProactiveInjector] Loading knowledge profile: {knowledge_profile_path}")
        with open(knowledge_profile_path) as f:
            profile = json.load(f)

        kb = LibraryKnowledgeBase.from_json(docs_path)

        print(f"  [ProactiveInjector] Generating doc snippets: {docs_path}")
        gen = InjectionGenerator(docs_path, library_name=kb.library_name)
        snippets = gen.generate_all_snippets()

        bm25_predictor = None
        if bm25_docs_path:
            from .rag_baseline import BM25DemandPredictor
            print(f"  [ProactiveInjector] Loading BM25 baseline corpus: {bm25_docs_path}")
            bm25_predictor = BM25DemandPredictor(
                bm25_docs_path,
                function_filter=list(snippets.get("L0", {}).keys()) or None,
            )

        sbert_predictor = None
        if sbert_docs_path:
            from .semantic_rag import SBERTDemandPredictor
            print(f"  [ProactiveInjector] Loading SBERT dense-RAG baseline corpus: {sbert_docs_path}")
            sbert_predictor = SBERTDemandPredictor(
                sbert_docs_path,
                function_filter=list(snippets.get("L0", {}).keys()) or None,
                # CPU keeps the encoder off the vLLM head GPU: dense 405B at
                # 0.90 utilisation has <1.3 GiB headroom on GPU 0, and the
                # CUDA-default encoder (~1.2 GiB) OOMs the engine there.
                device="cpu",
            )

        return cls(demand_by_id, profile, snippets,
                   top_k, token_budget_C, token_budget_X, threshold,
                   token_budget_RsemB=token_budget_RsemB,
                   knowledge_base=kb, risk_weights=risk_weights,
                   bm25_predictor=bm25_predictor,
                   sbert_predictor=sbert_predictor,
                   risk_uniform=risk_uniform,
                   recal_name_layer_floor=recal_name_layer_floor)

    # ── Tier logic ──────────────────────────────────────────────────────────

    def _get_tier(self, func_name: str) -> int:
        """Highest-fail tier: deepest failing layer drives injection depth."""
        if func_name not in self.profile:
            return -1  # unknown → conservative Tier 3 fallback
        layer_data = self.profile[func_name]
        highest_fail = None
        for layer in self.LAYER_ORDER:
            val = layer_data.get(layer)
            if val is None:
                continue
            score = val["score"] if isinstance(val, dict) else val
            if score is not None and score < self.threshold:
                highest_fail = layer
        if highest_fail is None:
            return 0
        return self.LAYER_ORDER.index(highest_fail) + 1

    def _has_example(self, func_name: str) -> bool:
        l3 = self.snippets.get("L3", {}).get(func_name)
        return l3 is not None and "No example available" not in l3.content

    def _assemble_func_block(self, func_name: str, tier: int, max_tier: int = 4) -> tuple:
        """
        Returns (text, token_estimate). Handles tier fallbacks.

        Args:
            func_name: target-library function name
            tier: model's knowledge tier (0=knows it, 4=deep failure)
            max_tier: cap the effective tier (e.g. max_tier=3 skips L3 examples)
        """
        effective_tier = min(tier, max_tier)
        if tier == 4 and effective_tier == 4 and not self._has_example(func_name):
            effective_tier = 3   # no example available → fall back to tier 3
        if tier == -1:
            effective_tier = min(3, max_tier)   # unknown function → conservative

        if effective_tier == 0:
            return f"- `{func_name}`", 5

        layers = self.LAYER_ORDER[:effective_tier]
        lines = []
        tokens = 0
        for layer in layers:
            snip = self.snippets.get(layer, {}).get(func_name)
            if snip:
                lines.append(snip.content)
                tokens += snip.token_estimate
        return "\n".join(lines), tokens

    # ── Candidate selection ──────────────────────────────────────────────────

    def _function_known(self, func_name: str) -> bool:
        return any(func_name in layer_snips for layer_snips in self.snippets.values())

    def _is_network_loader(self, func_name: str) -> bool:
        lname = func_name.lower()
        meta = self.kb.func_map.get(func_name, {})
        category = str(meta.get("category") or "").lower()
        return (
            lname.startswith("case")
            or "network" in lname
            or "network" in category
            or lname in {"mv_oberrhein", "example_simple", "simple_four_bus_system"}
        )

    def _layer_score(self, func_name: str, layer: str) -> Optional[float]:
        entry = self.profile.get(func_name) or {}
        data = entry.get(layer)
        if data is None:
            return None
        if layer == "L3" and isinstance(data, dict):
            diag = data.get("diagnostics") or {}
            execution = diag.get("execution_success")
            target_use = diag.get("target_function_used")
            if isinstance(execution, (int, float)) and isinstance(target_use, (int, float)):
                return max(0.0, min(1.0, float(execution) * float(target_use)))
        val = data.get("score") if isinstance(data, dict) else data
        return float(val) if isinstance(val, (int, float)) else None

    def _knowledge_risk(self, func_name: str) -> Optional[float]:
        """Model-specific risk in [0, 1], recomputed from this target profile."""
        if self.risk_uniform:
            return 1.0
        total_w = 0.0
        mastery = 0.0
        for layer, weight in self.risk_weights.items():
            score = self._layer_score(func_name, layer)
            if score is None:
                continue
            mastery += weight * max(0.0, min(1.0, score))
            total_w += weight
        if total_w == 0:
            return None
        return 1.0 - mastery / total_w

    def _layer_risk(self, func_name: str, layer: str) -> Optional[float]:
        """Layer-specific deficit in [0, 1]; 0 means the layer is above threshold."""
        if self.risk_uniform:
            # C_risk_uniform ablation: all functions/layers treated as fully
            # at risk; the model-specific probe profile is bypassed but every
            # other gating component (demand, layer_need, layer_weight,
            # interface floors, token budget) remains in effect.
            return 1.0
        score = self._layer_score(func_name, layer)
        if score is None:
            return None
        if self.threshold <= 0:
            return 0.0
        return max(0.0, min(1.0, (self.threshold - score) / self.threshold))

    def _boundary_related_functions(self, item: dict, funcs: List[str]) -> set:
        """Functions tied to task-level API contracts already triggered by this query."""
        query = item.get("natural_language_query", "")
        matched_ids = set(self.kb.boundary_contract_ids_for_query(query, funcs=funcs, task=""))
        if not matched_ids:
            return set()
        related = set()
        for contract in getattr(self.kb, "boundary_contracts", []) or []:
            if contract.id in matched_ids:
                related.update(self.kb.canonical_name(f) for f in contract.related_functions)
        return {fn for fn in related if self._function_known(fn)}

    def _regex_signal(self, pattern: str, text: str) -> bool:
        return re.search(pattern, text, flags=re.IGNORECASE) is not None

    def _task_layer_profile(self, item: dict, demand_entries: List[dict] = None) -> dict:
        """
        Query-derived layer need profile for Cond C.

        Uses only deployment-available evidence: the natural language query,
        API-spec intent matches, and demand candidate roles. It must not read
        benchmark D1-D4 labels or scenario.task.
        """
        query = item.get("natural_language_query", "") or ""
        q = query.lower()
        entries = demand_entries or []
        roles = {str(e.get("role") or "").lower() for e in entries}
        workflows = self.kb.matched_workflow_ids(query)

        modification_terms = re.findall(
            r"\b(?:set|change|modify|adjust|update|increase|decrease|scale|"
            r"multiply|add|remove|disconnect|reconnect|connect|install|"
            r"curtail|replace|switch|open|close)\b|out\s+of\s+service",
            q,
            flags=re.IGNORECASE,
        )
        signals = {
            "workflow_intent": bool(workflows),
            "sequential": self._regex_signal(
                r"\b(?:first|then|after|before|following|once|next|step|"
                r"subsequent|again|rerun|re-run)\b",
                q,
            ),
            "conditional": self._regex_signal(
                r"\b(?:if|whether|when|check|verify|ensure|exceed|exceeds|"
                r"violate|violation|threshold|above|below|greater|less)\b",
                q,
            ),
            "aggregation": self._regex_signal(
                r"\b(?:max(?:imum)?|min(?:imum)?|highest|lowest|largest|"
                r"smallest|count|total|sum|average|mean|which|index|argmax|"
                r"argmin|rank)\b",
                q,
            ),
            "multi_modification": len(modification_terms) >= 2,
            "constructor_role": any("construction" in r or "network" in r for r in roles),
            "executor_role": any("analysis" in r or "executor" in r for r in roles),
            "long_query": len(q.split()) >= 35,
        }

        need = dict(self.TASK_LAYER_NEED_BASE)
        if signals["constructor_role"] or signals["executor_role"]:
            need["L1"] += 0.10
        if signals["conditional"] or signals["aggregation"]:
            need["L2"] += 0.22
        if signals["workflow_intent"]:
            need["L2"] += 0.10
            need["L3"] += 0.22
        if signals["sequential"]:
            need["L3"] += 0.22
        if signals["multi_modification"]:
            need["L2"] += 0.10
            need["L3"] += 0.12
        if signals["long_query"]:
            need["L2"] += 0.05
            need["L3"] += 0.05

        need = {
            layer: round(min(self.TASK_LAYER_NEED_CAP[layer], value), 4)
            for layer, value in need.items()
        }
        pressure = sum(
            int(signals[name])
            for name in ("workflow_intent", "sequential", "conditional", "aggregation", "multi_modification")
        )
        return {
            "layer_need": need,
            "signals": signals,
            "matched_workflows": workflows,
            "procedural_pressure": pressure,
        }

    def _l3_cap_for_item(self, item: dict, demand_entries: List[dict] = None) -> int:
        profile = self._task_layer_profile(item, demand_entries)
        pressure = int(profile.get("procedural_pressure", 0))
        if pressure <= 1:
            return 1
        if pressure == 2:
            return 2
        return 3

    def _interface_risk_floor(
        self,
        entry: dict,
        layer: str,
        boundary_related_funcs: set,
        task_profile: dict = None,
    ) -> Tuple[float, str]:
        """
        Task-side interface risk independent of model self-confidence.

        This is the proactive counterpart of demand modelling: if the query
        explicitly names an API intent, demand is very high, or a boundary
        contract is active, C should keep the minimal relevant layers even when
        an old probe profile marks the model as broadly competent.
        """
        floors: List[Tuple[float, str]] = []
        fn = entry.get("function")
        demand = float(entry.get("demand_score", 0.0))
        role = str(entry.get("role") or "").lower()
        task_profile = task_profile or {}
        layer_need = float((task_profile.get("layer_need") or {}).get(layer, 1.0))
        procedural_pressure = int(task_profile.get("procedural_pressure", 0))
        is_core_api = (
            bool(entry.get("is_query_anchor"))
            or demand >= self.HIGH_DEMAND_INTERFACE_MIN
            or any(token in role for token in ("analysis", "executor", "construction", "network"))
        )

        if entry.get("is_query_anchor") and layer in self.QUERY_ANCHOR_INTERFACE_FLOORS:
            floors.append((self.QUERY_ANCHOR_INTERFACE_FLOORS[layer], "query_anchor"))

        if demand >= self.HIGH_DEMAND_INTERFACE_MIN and layer in self.HIGH_DEMAND_INTERFACE_FLOORS:
            floors.append((self.HIGH_DEMAND_INTERFACE_FLOORS[layer], "high_demand"))

        if fn in boundary_related_funcs and layer in self.BOUNDARY_INTERFACE_FLOORS:
            floors.append((self.BOUNDARY_INTERFACE_FLOORS[layer], "boundary_contract"))

        if is_core_api and layer == "L2" and layer_need >= 0.95:
            floors.append((min(0.22, 0.10 + 0.20 * (layer_need - 0.95)), "task_layer_need"))

        if is_core_api and layer == "L3" and layer_need >= 1.0 and procedural_pressure >= 2:
            floors.append((min(0.24, 0.10 + 0.18 * (layer_need - 1.0)), "task_layer_need"))

        if not floors:
            return 0.0, ""
        return max(floors, key=lambda item: item[0])

    def _first_sentence(self, text: str) -> str:
        clean = " ".join(str(text or "").split())
        if not clean:
            return ""
        match = re.search(r"(.{1,220}?[.!?])(?:\s|$)", clean)
        return match.group(1) if match else clean[:220].rstrip()

    def _shorten(self, text: str, limit: int = 180) -> str:
        clean = " ".join(str(text or "").split())
        if len(clean) <= limit:
            return clean
        return clean[: max(0, limit - 3)].rstrip() + "..."

    def _param_is_required(self, param: dict) -> bool:
        return bool(param.get("required"))

    def _param_matches_query(self, param: dict, query: str) -> bool:
        q = (query or "").lower()
        name = str(param.get("name") or "").lower()
        if not name or name.startswith("*"):
            return False
        pieces = [name] + [p for p in re.split(r"[_\W]+", name) if len(p) >= 3]
        return any(piece and piece in q for piece in pieces)

    def _compact_l2_snippet_block(self, func_name: str, item: dict = None) -> Tuple[str, int]:
        """Task-facing L2 renderer for Cond C: compact contract, not full docs."""
        func_info = self.kb.func_map.get(func_name) or {}
        if not func_info:
            return "", 0

        query = (item or {}).get("natural_language_query", "")
        desc = self._first_sentence(func_info.get("description", ""))
        full_path = func_info.get("full_path") or f"{self.library_name}.{func_name}"
        params = [
            p for p in list(func_info.get("parameters") or [])
            if (
                isinstance(p, dict)
                and not str(p.get("name", "")).startswith("*")
                and str(p.get("name", "")).lower() not in {"kwargs", "**kwargs"}
            )
        ]

        required = [p for p in params if self._param_is_required(p)]
        query_params = [p for p in params if self._param_matches_query(p, query)]
        selected_params = []
        for param in required + query_params:
            if param not in selected_params:
                selected_params.append(param)
        if len(selected_params) < min(self.COMPACT_L2_MAX_PARAMS, len(params)):
            for param in params:
                if param not in selected_params:
                    selected_params.append(param)
                if len(selected_params) >= self.COMPACT_L2_MAX_PARAMS:
                    break
        else:
            selected_params = selected_params[:self.COMPACT_L2_MAX_PARAMS]

        call_args = [str(p.get("name")) for p in selected_params if p.get("name")]
        omitted = len(params) > len(selected_params)
        call = f"{full_path}({', '.join(call_args)}{', ...' if omitted and call_args else '...' if omitted else ''})"

        lines = []
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

        text = "\n".join(lines)
        return text, len(text.split()) + 10

    def _layer_snippet_block(self, func_name: str, layer: str, item: dict = None, condition: str = "C") -> Tuple[str, int]:
        """Return a self-contained snippet for one function-layer candidate."""
        if layer == "L2" and condition == "C":
            content, tokens = self._compact_l2_snippet_block(func_name, item=item)
            if not content or tokens <= 0:
                return "", 0
            header = (
                f"[{self.library_name}.{func_name} | {layer}: "
                f"{self.LAYER_LABELS.get(layer, 'knowledge supplement')}]\n"
            )
            text = f"{header}{content}"
            return text, tokens + max(1, len(header) // 4)

        snip = self.snippets.get(layer, {}).get(func_name)
        if not snip:
            return "", 0
        if layer == "L3" and "No example available" in snip.content:
            return "", 0
        header = (
            f"[{self.library_name}.{func_name} | {layer}: "
            f"{self.LAYER_LABELS.get(layer, 'knowledge supplement')}]\n"
        )
        text = f"{header}{snip.content}"
        return text, snip.token_estimate + max(1, len(header) // 4)

    def _layer_candidates(self, item: dict, condition: str = "C") -> List[dict]:
        """
        Build ranked function-layer candidates.

        Cond C uses demand * max(model_layer_deficit, task_interface_risk)
        * query-derived task_layer_need * layer_weight. Cond X is the
        demand-only upper bound, so every available layer is eligible with
        risk=1 while preserving the same query anchors and time-series filters.
        Cond R is the vanilla-BM25-RAG baseline: BM25-ranked candidates with
        no query anchors / pruning / boundary contracts (those belong to our
        pipeline) and no model-side risk gating.
        """
        if condition == "R":
            demand_entries = self._bm25_candidate_entries(item)
            is_ts_task = False
            boundary_related_funcs = set()
            task_profile = self._task_layer_profile(item, demand_entries)
        elif condition in ("Rsem", "RsemB"):
            demand_entries = self._sbert_candidate_entries(item)
            is_ts_task = False
            boundary_related_funcs = set()
            task_profile = self._task_layer_profile(item, demand_entries)
        else:
            demand_entries = self._candidate_entries(item, mode="demand")
            funcs = [e["function"] for e in demand_entries]
            is_ts_task = self._is_time_series_item(item, funcs)
            boundary_related_funcs = self._boundary_related_functions(item, funcs)
            task_profile = self._task_layer_profile(item, demand_entries)
        candidates = []

        for entry in demand_entries:
            fn = entry["function"]
            max_depth = self.TS_DEPTH_CAPS.get(fn, 4) if is_ts_task else 4
            for layer in self.LAYER_ORDER:
                layer_depth = self.LAYER_ORDER.index(layer) + 1
                if layer_depth > max_depth:
                    continue
                text, tokens = self._layer_snippet_block(fn, layer, item=item, condition=condition)
                if not text or tokens <= 0:
                    continue

                score = self._layer_score(fn, layer)
                task_layer_need = float(task_profile["layer_need"].get(layer, 1.0))
                if condition == "X":
                    probe_layer_risk = 1.0
                    interface_floor = 0.0
                    layer_risk = 1.0
                    risk_reason = "demand_upper_bound"
                    value = float(entry.get("demand_score", 0.0)) * self.risk_weights.get(layer, 0.0)
                elif condition == "R":
                    probe_layer_risk = 1.0
                    interface_floor = 0.0
                    layer_risk = 1.0
                    risk_reason = "bm25_baseline"
                    value = float(entry.get("demand_score", 0.0)) * self.risk_weights.get(layer, 0.0)
                elif condition in ("Rsem", "RsemB"):
                    probe_layer_risk = 1.0
                    interface_floor = 0.0
                    layer_risk = 1.0
                    risk_reason = "sbert_baseline"
                    value = float(entry.get("demand_score", 0.0)) * self.risk_weights.get(layer, 0.0)
                else:
                    probe_layer_risk = self._layer_risk(fn, layer)
                    interface_floor, interface_reason = self._interface_risk_floor(
                        entry, layer, boundary_related_funcs, task_profile
                    )
                    if probe_layer_risk is None:
                        probe_layer_risk = 0.0
                    # E2-R recalibration arm: floor the L0/L1 deficit of every
                    # demand-hit function so name/signature layers always enter
                    # packing competition (0.0 default = frozen C behaviour).
                    recal_floor = (
                        self.recal_name_layer_floor if layer in ("L0", "L1") else 0.0
                    )
                    floored_probe_risk = max(probe_layer_risk, recal_floor)
                    layer_risk = max(floored_probe_risk, interface_floor)
                    if layer_risk <= 0:
                        continue
                    if floored_probe_risk >= interface_floor and floored_probe_risk > 0:
                        risk_reason = (
                            "probe_deficit"
                            if probe_layer_risk >= recal_floor and probe_layer_risk > 0
                            else "recal_name_layer_floor"
                        )
                    else:
                        risk_reason = interface_reason
                    value = (
                        float(entry.get("demand_score", 0.0))
                        * layer_risk
                        * task_layer_need
                        * self.risk_weights.get(layer, 0.0)
                    )

                rank_score = value / (max(tokens, 1) ** self.LAYER_COST_ALPHA)
                candidates.append({
                    "function": fn,
                    "layer": layer,
                    "layer_score": score,
                    "layer_risk": layer_risk,
                    "probe_layer_risk": probe_layer_risk,
                    "interface_risk_floor": interface_floor,
                    "risk_reason": risk_reason,
                    "demand_score": float(entry.get("demand_score", 0.0)),
                    "task_layer_need": task_layer_need,
                    "knowledge_risk": entry.get("knowledge_risk"),
                    "layer_value": value,
                    "rank_score": rank_score,
                    "tokens": tokens,
                    "text": text,
                    "is_query_anchor": bool(entry.get("is_query_anchor", False)),
                    "source_rank": entry.get("source_rank", 0),
                })

        candidates.sort(key=lambda c: (
            -c["rank_score"],
            -c["layer_value"],
            -c["demand_score"],
            self.LAYER_ORDER.index(c["layer"]),
            c["function"],
        ))
        return candidates

    def _dominates_layer(self, selected_layer: str, candidate_layer: str, condition: str) -> bool:
        if selected_layer == candidate_layer:
            return True
        if condition != "C":
            return False
        if selected_layer == "L1" and candidate_layer == "L0":
            return True
        if condition == "C" and selected_layer in {"L2", "L3"} and candidate_layer in {"L0", "L1"}:
            return True
        return False

    def _select_layer_snippets(self, item: dict, condition: str, budget: int) -> Tuple[List[dict], int]:
        """Greedily select function-layer snippets under a doc budget."""
        if condition == "R":
            demand_entries = self._bm25_candidate_entries(item)
        elif condition in ("Rsem", "RsemB"):
            demand_entries = self._sbert_candidate_entries(item)
        else:
            demand_entries = self._candidate_entries(item, mode="demand")
        l3_cap = (
            self.MAX_L3_EXAMPLES_X
            if condition in ("X", "R", "Rsem", "RsemB")
            else min(self.MAX_L3_EXAMPLES_X, self._l3_cap_for_item(item, demand_entries))
        )
        selected = []
        total_tokens = 0
        l3_count = 0
        seen = set()

        for cand in self._layer_candidates(item, condition=condition):
            key = (cand["function"], cand["layer"])
            if key in seen:
                continue
            if any(
                existing["function"] == cand["function"]
                and self._dominates_layer(existing["layer"], cand["layer"], condition)
                for existing in selected
            ):
                continue

            dominated_indices = [
                idx for idx, existing in enumerate(selected)
                if existing["function"] == cand["function"]
                and self._dominates_layer(cand["layer"], existing["layer"], condition)
            ]
            freed_tokens = sum(selected[idx]["tokens"] for idx in dominated_indices)
            if cand["layer"] == "L3":
                if l3_count >= l3_cap:
                    continue
            if total_tokens - freed_tokens + cand["tokens"] > budget:
                continue
            for idx in sorted(dominated_indices, reverse=True):
                removed = selected.pop(idx)
                total_tokens -= removed["tokens"]
                if removed["layer"] == "L3":
                    l3_count -= 1
            if cand["layer"] == "L3":
                l3_count += 1
            seen.add(key)
            selected.append(cand)
            total_tokens += cand["tokens"]

        return selected, total_tokens

    def _format_layer_blocks(self, selected: List[dict]) -> str:
        return "\n\n".join(c["text"] for c in selected if c.get("text"))

    def _network_anchor_functions(self, query: str) -> List[str]:
        """Map explicit benchmark network names in the query to loader APIs."""
        if self._network_anchor_map is None:
            anchors = []
            try:
                from benchmark_generator.benchmark_config import NETWORK_REGISTRY, SIMBENCH_NETWORKS
                for cfg in list(NETWORK_REGISTRY.values()) + list(SIMBENCH_NETWORKS.values()):
                    display = (cfg.get("display") or "").strip().lower()
                    loader = cfg.get("loader") or ""
                    m = re.search(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", loader)
                    if display and m:
                        anchors.append((display, m.group(1)))
            except Exception:
                anchors = []
            self._network_anchor_map = anchors

        q = query.lower()
        funcs = []
        for display, func_name in self._network_anchor_map:
            if display in q and self._function_known(func_name):
                funcs.append(func_name)
        return funcs

    def _query_anchor_functions(self, item: dict) -> List[str]:
        """High-precision query cues loaded from the library spec."""
        query = item.get("natural_language_query", "")
        return self.kb.query_anchor_functions(
            query,
            known_predicate=self._function_known,
            extra_functions=self._network_anchor_functions(query),
        )

    def _prune_intent_mismatches(self, item: dict, merged: Dict[str, dict]) -> None:
        """
        Remove specialized workflow APIs when the query lacks the matching intent.

        The demand model is recall-oriented and can put mutually exclusive
        executors in the top-k tail. Layer-wise risk should not amplify those
        false positives into examples for an unrelated task.
        """
        self.kb.prune_intent_mismatches(
            item.get("natural_language_query", ""),
            merged,
        )

    def _is_time_series_item(self, item: dict, funcs: List[str]) -> bool:
        return "time_series" in self.kb.matched_workflow_ids(
            item.get("natural_language_query", "")
        )

    def _boundary_card_entries(self, item: dict, funcs: List[str] = None) -> List[Tuple[str, str, int]]:
        """Task-level API contracts injected only when the query crosses a known boundary."""
        return self.kb.boundary_card_entries(
            item.get("natural_language_query", ""),
            funcs=funcs or [],
            task="",
        )

    def _candidate_entries(self, item: dict, mode: str = "demand") -> List[dict]:
        """
        Return top candidates after query anchoring and per-target risk scoring.

        mode="demand" ranks by demand score for B/X name/doc lists.
        mode="intervention" ranks by model-specific demand*risk for diagnostics.
        """
        item_id = item.get("id", "")
        demand_item = self.demand_by_id.get(item_id, {})
        base_entries = demand_item.get("top_demand", []) or demand_item.get("top_intervention", [])

        merged = {}
        for pos, entry in enumerate(base_entries):
            name = entry.get("function")
            if not name or not self._function_known(name):
                continue
            merged[name] = {
                "function": name,
                "demand_score": float(entry.get("demand_score", 0.0)),
                "role": entry.get("role"),
                "is_reference_label": bool(entry.get("is_reference_label", False)),
                "is_query_anchor": False,
                "source_rank": pos,
            }

        max_demand = max((e["demand_score"] for e in merged.values()), default=1.0)
        anchor_score = max(1.0, max_demand + 0.05)
        for pos, name in enumerate(self._query_anchor_functions(item)):
            entry = merged.setdefault(name, {
                "function": name,
                "demand_score": anchor_score - 0.001 * pos,
                "role": "query_anchor",
                "is_reference_label": False,
                "source_rank": 10_000 + pos,
            })
            entry["demand_score"] = max(entry.get("demand_score", 0.0), anchor_score - 0.001 * pos)
            entry["is_query_anchor"] = True

        self._prune_intent_mismatches(item, merged)

        for fn in self.kb.workflow_suppressed_functions(item.get("natural_language_query", "")):
            merged.pop(fn, None)

        network_anchors = set(self._network_anchor_functions(item.get("natural_language_query", "")))
        if network_anchors:
            for fn in list(merged):
                if fn not in network_anchors and self._is_network_loader(fn):
                    merged.pop(fn, None)

        for entry in merged.values():
            risk = self._knowledge_risk(entry["function"])
            entry["knowledge_risk"] = risk
            effective_risk = 1.0 if risk is None else risk
            entry["intervention_score"] = entry["demand_score"] * effective_risk
            entry["tier"] = self._get_tier(entry["function"])

        entries = list(merged.values())
        if mode == "intervention":
            entries.sort(key=lambda e: (
                -e["intervention_score"],
                -e["demand_score"],
                -e["tier"],
                e["function"],
            ))
        else:
            entries.sort(key=lambda e: (
                -e["demand_score"],
                e["source_rank"],
                e["function"],
            ))
        return entries[:self.top_k]

    def _top_funcs(self, item: dict, mode: str = "demand") -> List[str]:
        return [d["function"] for d in self._candidate_entries(item, mode=mode)]

    def _bm25_candidate_entries(self, item: dict) -> List[dict]:
        """Vanilla BM25 candidate construction for condition R.

        Deliberately omits query anchors, intent pruning, workflow suppression,
        network-loader anchoring, and boundary-card injection -- those are part
        of our pipeline's contribution and would inflate a baseline that is
        meant to represent off-the-shelf RAG over raw API documentation.
        Returns the top_k function entries ranked by raw BM25 score.
        """
        if self.bm25_predictor is None:
            raise RuntimeError(
                "Condition R requires a BM25 predictor; pass bm25_docs_path "
                "to ProactiveInjector.from_paths()."
            )
        query = item.get("natural_language_query", "")
        scores = self.bm25_predictor.score(query)
        scored = [(fn, s) for fn, s in scores.items() if self._function_known(fn)]
        scored.sort(key=lambda kv: -kv[1])
        entries = []
        for pos, (name, score) in enumerate(scored[: self.top_k]):
            entries.append({
                "function": name,
                "demand_score": float(score),
                "role": None,
                "is_reference_label": False,
                "is_query_anchor": False,
                "source_rank": pos,
                "knowledge_risk": None,
                "intervention_score": float(score),
                "tier": -1,
            })
        return entries

    def _sbert_candidate_entries(self, item: dict) -> List[dict]:
        """Vanilla SBERT dense-RAG candidate construction for conditions Rsem/RsemB.

        Mirror of _bm25_candidate_entries with the lexical scorer swapped for
        dense cosine retrieval. Deliberately omits query anchors, intent
        pruning, workflow suppression, network-loader anchoring, and
        boundary-card injection -- those are part of our pipeline's contribution
        and would inflate a baseline meant to represent off-the-shelf dense RAG
        over raw API documentation. Returns the top_k function entries ranked by
        SBERT cosine.
        """
        if self.sbert_predictor is None:
            raise RuntimeError(
                "Conditions Rsem/RsemB require an SBERT predictor; pass "
                "sbert_docs_path to ProactiveInjector.from_paths()."
            )
        query = item.get("natural_language_query", "")
        scores = self.sbert_predictor.score(query)
        scored = [(fn, s) for fn, s in scores.items() if self._function_known(fn)]
        scored.sort(key=lambda kv: -kv[1])
        entries = []
        for pos, (name, score) in enumerate(scored[: self.top_k]):
            entries.append({
                "function": name,
                "demand_score": float(score),
                "role": None,
                "is_reference_label": False,
                "is_query_anchor": False,
                "source_rank": pos,
                "knowledge_risk": None,
                "intervention_score": float(score),
                "tier": -1,
            })
        return entries

    def describe_plan(self, item: dict, condition: str = "C") -> Dict:
        """Return CPU-only diagnostics for the proactive injection plan."""
        query = item.get("natural_language_query", "")
        demand_item = self.demand_by_id.get(item.get("id", ""), {})
        refs = set(demand_item.get("reference_functions", []))
        demand_entries = self._candidate_entries(item, mode="demand")
        demand_funcs = [e["function"] for e in demand_entries]
        matched_workflows = self.kb.matched_workflow_ids(query)
        task_profile = self._task_layer_profile(item, demand_entries)
        is_ts_task = "time_series" in matched_workflows
        raw_demand = demand_item.get("top_demand", []) or []
        suppressed_functions = set(self.kb.workflow_suppressed_functions(query))
        suppressed = [
            e.get("function") for e in raw_demand[:self.top_k]
            if e.get("function") in suppressed_functions
        ]

        plan = {
            "id": item.get("id"),
            "task": (item.get("scenario") or {}).get("task"),
            "condition": condition,
            "is_time_series": is_ts_task,
            "matched_workflows": matched_workflows,
            "task_layer_need": task_profile.get("layer_need", {}),
            "task_layer_signals": task_profile.get("signals", {}),
            "task_procedural_pressure": task_profile.get("procedural_pressure", 0),
            "l3_cap": (
                self.MAX_L3_EXAMPLES_X
                if condition == "X"
                else self._l3_cap_for_item(item, demand_entries)
            ),
            "query": query,
            "reference_functions": sorted(refs),
            "name_functions": demand_funcs,
            "query_anchors": [e["function"] for e in demand_entries if e.get("is_query_anchor")],
            "suppressed_time_series_internal_runners": suppressed,
            "selected_docs": [],
            "doc_tokens": 0,
        }
        boundary_cards = self._boundary_card_entries(item, demand_funcs)
        plan["boundary_cards"] = [name for name, _, _ in boundary_cards]
        plan["boundary_card_tokens"] = sum(tokens for _, _, tokens in boundary_cards)

        if condition == "B":
            plan["name_ref_recall"] = len(set(demand_funcs) & refs) / max(1, len(refs))
            return plan

        if condition == "C":
            budget = self.token_budget_C
        elif condition == "X":
            budget = self.token_budget_X
        else:
            return plan

        doc_budget = max(0, budget - plan["boundary_card_tokens"])
        selected_raw, total_tokens = self._select_layer_snippets(item, condition, doc_budget)
        selected = []
        for entry in selected_raw:
            fn = entry["function"]
            selected.append({
                "function": fn,
                "layer": entry["layer"],
                "layer_score": (
                    None if entry.get("layer_score") is None
                    else round(float(entry["layer_score"]), 6)
                ),
                "layer_risk": (
                    None if entry.get("layer_risk") is None
                    else round(float(entry["layer_risk"]), 6)
                ),
                "probe_layer_risk": round(float(entry.get("probe_layer_risk", 0.0)), 6),
                "interface_risk_floor": round(float(entry.get("interface_risk_floor", 0.0)), 6),
                "risk_reason": entry.get("risk_reason", ""),
                "demand_score": round(float(entry.get("demand_score", 0.0)), 6),
                "task_layer_need": round(float(entry.get("task_layer_need", 1.0)), 6),
                "knowledge_risk": (
                    None if entry.get("knowledge_risk") is None
                    else round(float(entry["knowledge_risk"]), 6)
                ),
                "layer_value": round(float(entry.get("layer_value", 0.0)), 6),
                "rank_score": round(float(entry.get("rank_score", 0.0)), 6),
                "tokens": entry["tokens"],
                "is_query_anchor": bool(entry.get("is_query_anchor", False)),
                "is_reference_label": fn in refs,
            })

        plan["selected_docs"] = selected
        plan["doc_tokens"] = total_tokens
        plan["name_ref_recall"] = len(set(demand_funcs) & refs) / max(1, len(refs))
        plan["doc_ref_recall"] = len({e["function"] for e in selected} & refs) / max(1, len(refs))
        return plan

    def build_messages(self, item: dict, condition: str) -> List[Dict]:
        """
        Build the messages list for a benchmark item under the given condition.

        Condition logic:
          A – Baseline: standard benchmark prompt (no injection)
          B – Demand-only: top_demand function names listed (no doc content)
          C – Targeted injection (MAIN condition):
              • Selects function-layer candidates from top_demand, then
                recomputes demand × max(layer_deficit, interface_risk) using
                this target model's profile and query-derived layer need.
              • Adds high-precision query anchors for network loaders and
                construction APIs that demand ranking often under-selects.
              • Renders C's L2 as a compact call contract, and lets higher
                layers dominate redundant lower-layer snippets for the same API.
              • Allows compact L3 examples when layer risk and query-derived
                procedural need are high, with a small cap to control token cost.
              • Sorted by token-aware layer value under the C budget.
          X – Demand-only upper bound: all top_demand functions/layers are eligible,
              regardless of model knowledge. Uses a larger budget.

        item must have keys: 'id', 'natural_language_query'
        """
        # render_benchmark_task_text appends the item's given network setup code when it
        # carries one (E2 B1' self-contained items) and is a no-op otherwise. Only the
        # rendered PROMPT text gets the code block -- the demand / retrieval / layer-need
        # heuristics above keep reading the raw natural_language_query, so injection
        # selection is unchanged by the presentation-layer fix.
        from backend.utils import render_benchmark_task_text
        query = render_benchmark_task_text(item)

        if condition == "A":
            from backend.utils import build_benchmark_prompt
            return build_benchmark_prompt(query)

        if condition == "B":
            # top_demand: broadest set of predicted functions
            top_funcs = self._top_funcs(item, mode="demand")
            if not top_funcs:
                from backend.utils import build_benchmark_prompt
                return build_benchmark_prompt(query)
            func_list = "\n".join(f"  - {self.library_name}.{fn}" for fn in top_funcs)
            injection = (
                "[Predicted API Functions for This Task]\n"
                f"The following {self.library_name} functions are likely needed:\n"
                f"{func_list}"
            )
            user_msg = (f"Task: {query}\n\n{injection}\n\n"
                        "Directly provide fully executable code, including library imports.")
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ]

        if condition == "C":
            # Main proactive intervention: rank (function, layer) candidates by
            # demand * model-specific layer deficit * query-derived layer need.
            # This keeps B as a clean ablation and avoids injecting unrelated
            # documentation layers.
            demand_funcs = self._top_funcs(item, mode="demand")
            if not demand_funcs:
                from backend.utils import build_benchmark_prompt
                return build_benchmark_prompt(query)

            boundary_cards = self._boundary_card_entries(item, demand_funcs)
            doc_budget = max(0, self.token_budget_C - sum(t for _, _, t in boundary_cards))
            selected, _ = self._select_layer_snippets(item, "C", doc_budget)

            injection_parts = []
            if boundary_cards:
                injection_parts.append("\n\n".join(text for _, text, _ in boundary_cards))
            if selected:
                injection_body = self._format_layer_blocks(selected)
                injection_parts.append(
                    "[API Reference – Knowledge Supplement]\n"
                    "Layer-specific documentation selected because the task is predicted "
                    "to need these APIs and the selected layers carry model-side or "
                    "task-side interface risk:\n\n"
                    f"{injection_body}"
                )

            if not injection_parts:
                from backend.utils import build_benchmark_prompt
                return build_benchmark_prompt(query)

            injection = "\n\n".join(injection_parts)
            user_msg = (f"Task: {query}\n\n{injection}\n\n"
                        "Directly provide fully executable code, including library imports.")
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ]

        if condition == "X":
            # Demand-only upper bound: all top_demand function-layer snippets are
            # eligible regardless of the target model's probe profile.
            top_funcs = self._top_funcs(item, mode="demand")
            if not top_funcs:
                from backend.utils import build_benchmark_prompt
                return build_benchmark_prompt(query)

            boundary_cards = self._boundary_card_entries(item, top_funcs)
            doc_budget = max(0, self.token_budget_X - sum(t for _, _, t in boundary_cards))
            selected, _ = self._select_layer_snippets(item, "X", doc_budget)

            injection_parts = []
            if boundary_cards:
                injection_parts.append("\n\n".join(text for _, text, _ in boundary_cards))
            if selected:
                injection_body = self._format_layer_blocks(selected)
                injection_parts.append(
                    "[API Reference – Demand-Only Upper Bound]\n"
                    f"Documentation for demand-predicted {self.library_name} functions "
                    "without model-specific filtering:\n\n"
                    f"{injection_body}"
                )
            if not injection_parts:
                from backend.utils import build_benchmark_prompt
                return build_benchmark_prompt(query)
            injection = (
                "\n\n".join(injection_parts)
            )
            user_msg = f"Task: {query}\n\n{injection}\n\nDirectly provide fully executable code, including library imports."
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ]

        if condition == "R":
            # Vanilla BM25-RAG baseline: top-k functions retrieved by BM25 over
            # raw API documentation; renders standard L0--L3 snippets up to the
            # same token budget as condition C (B_C). No query anchors, no
            # boundary-card injection, no model-side risk gating -- those are
            # part of the proposed pipeline's contribution and would inflate
            # this baseline.
            if self.bm25_predictor is None:
                raise RuntimeError(
                    "Condition R requires a BM25 predictor; pass bm25_docs_path "
                    "to ProactiveInjector.from_paths()."
                )
            selected, _ = self._select_layer_snippets(item, "R", self.token_budget_C)
            if not selected:
                from backend.utils import build_benchmark_prompt
                return build_benchmark_prompt(query)
            injection_body = self._format_layer_blocks(selected)
            injection = (
                f"[API Reference -- BM25 RAG Baseline]\n"
                f"Documentation for BM25-retrieved {self.library_name} functions "
                f"over the raw API specification:\n\n{injection_body}"
            )
            user_msg = (f"Task: {query}\n\n{injection}\n\n"
                        "Directly provide fully executable code, including library imports.")
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ]

        if condition in ("Rsem", "RsemB"):
            # Vanilla dense (SBERT) RAG baseline: top-k functions retrieved by
            # cosine similarity over raw API documentation; renders standard
            # L0--L3 snippets. Rsem uses condition R's nominal budget; RsemB
            # tightens it to token_budget_RsemB (calibrated to C's measured
            # cost). No query anchors, no boundary-card injection, no model-side
            # risk gating -- those are part of the proposed pipeline's
            # contribution and would inflate this baseline.
            if self.sbert_predictor is None:
                raise RuntimeError(
                    "Conditions Rsem/RsemB require an SBERT predictor; pass "
                    "sbert_docs_path to ProactiveInjector.from_paths()."
                )
            budget = self.token_budget_C if condition == "Rsem" else self.token_budget_RsemB
            selected, _ = self._select_layer_snippets(item, condition, budget)
            if not selected:
                from backend.utils import build_benchmark_prompt
                return build_benchmark_prompt(query)
            injection_body = self._format_layer_blocks(selected)
            injection = (
                f"[API Reference -- SBERT Dense-RAG Baseline]\n"
                f"Documentation for dense-retrieved {self.library_name} functions "
                f"over the raw API specification:\n\n{injection_body}"
            )
            user_msg = (f"Task: {query}\n\n{injection}\n\n"
                        "Directly provide fully executable code, including library imports.")
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ]

        raise ValueError(
            f"Unknown proactive condition: {condition!r}. Use A/B/C/X/R/Rsem/RsemB."
        )
