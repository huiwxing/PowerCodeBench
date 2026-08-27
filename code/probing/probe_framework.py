# --------------------------------------------------------------------------
# Repository copy of the pipeline module probe_framework.py, with imports and
# data-path constants adapted to this repository's layout. Runs CPU-only
# against the archived corpus/spec files in this repository. See
# code/README.md for the module map.
# --------------------------------------------------------------------------
"""LLM Knowledge Boundary Probing Framework.

Theoretical basis: Bloom's Revised Taxonomy (Anderson & Krathwohl, 2001).
Each layer maps to a cognitive level AND a documentation granularity level.

  Layer    Bloom Level     Tests                         Injection on Failure
  ─────    ───────────     ─────                         ────────────────────
  L0       Remember/       Function existence            name + 1-line description
           Recognise       (MCQ: real vs fake)
  L1       Remember/       Signature recall              full signature string
           Recall          (path + required params)
  L2       Understand      Function & parameter          description + param descriptions
                           semantics (MCQ)               + return info
  L3       Apply           Single-function executable    example code snippet
                           code generation

Design principles:
  - Each layer depends ONLY on a standardised API-spec JSON (library-agnostic).
  - No domain-specific synonym tables, entity lists, or hardcoded knowledge.
  - Each layer's failure maps to a concrete documentation fragment to inject.
  - Broad coverage: every function gets L0/L1/L3 probes; L2 is instantiated
    from available semantic documentation fields (no n_probes cap).
  - ``build_knowledge_profile`` returns by_probe_type / diagnostics breakdowns.
"""

import json
import random
import re
import hashlib
import sys
import os
import signal
import traceback
import itertools
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from pathlib import Path
from io import StringIO
from unittest.mock import patch, MagicMock
from contextlib import contextmanager
from backend.utils import execute_code_safely, execute_code_safely_subprocess, parse_error

# ============================================================
# Section 1: Data Structures
# ============================================================

@dataclass
class Probe:
    """A single probe question targeting one cognitive layer."""
    probe_id:     str
    layer:        str          # "L0", "L1", "L2", "L3"
    probe_type:   str          # "recognition", "recall", "comprehension", "application"
    prompt:       str
    ground_truth: dict
    metadata:     dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Evaluation result for a single probe."""
    probe_id:   str
    layer:      str
    probe_type: str
    scores:     dict
    details:    dict = field(default_factory=dict)


@dataclass
class InjectionSnippet:
    """
    A documentation fragment to inject when a model fails at a given layer.
    
    The key idea: each layer's failure maps to a specific documentation granularity.
    Snippets are pre-generated from the API spec so that at injection time,
    we simply look up the function name + failed layer → snippet.
    """
    function_name: str
    layer:         str          # which layer failure this snippet addresses
    content:       str          # the actual text to inject into the prompt
    token_estimate: int = 0    # rough token count for budget planning


# ============================================================
# Section 2: Universal Fake Name Generator
# ============================================================

class FakeNameGenerator:
    """
    Generates plausible-but-nonexistent function names from a real function
    name list.  Fully library-agnostic — uses only structural patterns
    observed in the function names themselves.
    
    Three complementary strategies:
      1. Structural combination: cross-product of observed prefixes × entities
      2. Character-level perturbation: swap/add/remove word segments
      3. Cross-function grafting: prefix from func A + suffix from func B
    """

    def __init__(self, function_names: Set[str]):
        self.real_names = set(function_names)
        self.prefixes: Dict[str, List[str]] = defaultdict(list)   # prefix → [entity]
        self.entities: Dict[str, List[str]] = defaultdict(list)   # entity → [prefix]
        self._analyze()

    def _analyze(self):
        """Extract prefix-entity structure from real function names."""
        for name in self.real_names:
            parts = name.split("_")
            # Try split positions 1 and 2 (e.g., "create" | "bus", "create_empty" | "network")
            for split_pos in range(1, min(len(parts), 3)):
                prefix = "_".join(parts[:split_pos]) + "_"
                entity = "_".join(parts[split_pos:])
                if entity and len(entity) > 1:
                    self.prefixes[prefix].append(entity)
                    self.entities[entity].append(prefix)
        # Keep only prefixes that appear with multiple entities (productive patterns)
        self.prefixes = {k: v for k, v in self.prefixes.items() if len(v) >= 2}

    def generate(self, n: int = 100, seed: int = 42) -> List[str]:
        """Generate up to n fake function names using all three strategies."""
        rng = random.Random(seed)
        fakes: Set[str] = set()

        # ── Strategy 1: Structural combination ──────────────────────
        for prefix in self.prefixes:
            for entity in self.entities:
                candidate = prefix + entity
                if candidate not in self.real_names and len(candidate) < 50:
                    fakes.add(candidate)

        # ── Strategy 2: Character-level perturbation ────────────────
        for name in self.real_names:
            parts = name.split("_")
            if len(parts) < 2:
                continue

            # 2a. Swap two adjacent word segments
            for i in range(1, len(parts) - 1):
                swapped = parts[:i] + [parts[i+1], parts[i]] + parts[i+2:]
                candidate = "_".join(swapped)
                if candidate not in self.real_names:
                    fakes.add(candidate)

            # 2b. Append a common suffix
            for suffix in ["element", "data", "info", "table", "set", "list", "config"]:
                candidate = name + "_" + suffix
                if candidate not in self.real_names and len(candidate) < 50:
                    fakes.add(candidate)

            # 2c. Remove one word segment (if name has 3+ parts)
            if len(parts) >= 3:
                for i in range(1, len(parts)):
                    candidate = "_".join(parts[:i] + parts[i+1:])
                    if candidate not in self.real_names and len(candidate) > 2:
                        fakes.add(candidate)

            # 2d. Replace common suffix variants
            suffix_variants = {
                "3ph": ["3phase", "three_phase"],
                "dc":  ["ac"],
                "ac":  ["dc"],
            }
            for orig_suffix, replacements in suffix_variants.items():
                if name.endswith("_" + orig_suffix):
                    base = name[: -(len(orig_suffix))]
                    for repl in replacements:
                        candidate = base + repl
                        if candidate not in self.real_names:
                            fakes.add(candidate)

        # ── Strategy 3: Cross-function grafting ─────────────────────
        common_prefixes = [p for p, ents in self.prefixes.items() if len(ents) >= 3]
        entity_list = list(self.entities.keys())
        for prefix in common_prefixes[:8]:
            sample_size = min(10, len(entity_list))
            for e1, e2 in itertools.combinations(
                rng.sample(entity_list, sample_size), 2
            ):
                candidate = prefix + e1 + "_" + e2
                if candidate not in self.real_names and len(candidate) < 50:
                    fakes.add(candidate)

        # ── Sample down to requested size ───────────────────────────
        fakes_list = sorted(fakes)
        if len(fakes_list) > n:
            fakes_list = rng.sample(fakes_list, n)
        return fakes_list

    def difficulty_label(self, fake_name: str) -> str:
        """
        Label a fake name as 'hard' or 'easy' based on edit similarity
        to real names.  Hard fakes differ by ≤1 word segment from a real name.
        """
        fake_parts = set(fake_name.split("_"))
        for real_name in self.real_names:
            real_parts = set(real_name.split("_"))
            if len(fake_parts.symmetric_difference(real_parts)) <= 1:
                return "hard"
        return "easy"


# ============================================================
# Section 3: Probe Generator
# ============================================================

class ProbeGenerator:
    """
    Generates probes for all four cognitive layers from a library API spec JSON.
    
    Every function is guaranteed to appear in L0/L1/L3.
    L2 uses all available semantic documentation fields. No n_probes caps —
    coverage is prioritised over dataset size.
    """

    def __init__(self, json_path: str, library_name: str = None, seed: int = 42):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.library_name = (
            library_name
            or self.data.get("library")
            or self.data.get("library_name")
            or ("pandapower" if "pandapower" in str(json_path).lower() else "unknown")
        )
        # Deduplicate by function name (keep first occurrence)
        raw_functions = self.data.get("functions", [])
        seen = set()
        deduped = []
        for f in raw_functions:
            if f["name"] not in seen:
                seen.add(f["name"])
                deduped.append(f)
        if len(deduped) < len(raw_functions):
            print(f"  ⚠ Deduplicated: {len(raw_functions)} → {len(deduped)} functions "
                  f"(removed {len(raw_functions) - len(deduped)} duplicates)")
        self.functions    = deduped
        self.func_names   = {f["name"] for f in self.functions}
        self.func_map     = {f["name"]: f for f in self.functions}
        self.seed         = seed
        self.rng          = random.Random(seed)
        self.fake_gen     = FakeNameGenerator(self.func_names)

    # ==========================================================
    # L0: Recognition (MCQ — does this function exist?)
    # 2 probes per function (find_real + find_fake)
    # ==========================================================

    def generate_L0_recognition(self) -> List[Probe]:
        """
        For EVERY function, generate exactly 2 probes:
          - find_real:  1 real (target) + 3 fake → "which one EXISTS?"
          - find_fake:  3 real (including target) + 1 fake → "which one does NOT exist?"
          
        This guarantees each function is tested in both directions.
        """
        probes = []
        real_list = sorted(self.func_names)
        # Generate enough fakes for all probes
        fake_names = self.fake_gen.generate(
            n=len(real_list) * 4, seed=self.seed
        )
        fake_pool = list(fake_names)

        for real_func in real_list:
            available_fakes = [f for f in fake_pool if f != real_func]
            if len(available_fakes) < 3:
                continue

            # ── Probe A: find the REAL function (1 real + 3 fake) ──
            distractors_a = self.rng.sample(available_fakes, 3)
            options = [real_func] + distractors_a
            self.rng.shuffle(options)
            answer_idx   = options.index(real_func)
            answer_label = chr(ord('A') + answer_idx)
            question = (
                f"Which ONE of the following is a real function in the "
                f"`{self.library_name}` Python library?\n\n"
                + "\n".join(f"  {chr(ord('A')+j)}) `{opt}`"
                           for j, opt in enumerate(options))
                + f"\n\nRespond ONLY in JSON:\n"
                '{"answer": "A", "reasoning": "one sentence"}'
            )
            difficulty = self.fake_gen.difficulty_label(distractors_a[0])
            probes.append(Probe(
                probe_id     = f"L0_{hashlib.md5(f'{real_func}_find_real'.encode()).hexdigest()[:8]}",
                layer        = "L0",
                probe_type   = "recognition",
                prompt       = question,
                ground_truth = {
                    "question_type":  "find_real",
                    "correct_answer": answer_label,
                    "correct_func":   real_func,
                    "options": {chr(ord('A')+j): opt for j, opt in enumerate(options)},
                },
                metadata     = {
                    "library": self.library_name,
                    "function_name": real_func,
                    "question_type": "find_real",
                    "difficulty": difficulty,
                },
            ))

            # ── Probe B: find the FAKE function (3 real incl. target + 1 fake) ──
            other_reals = self.rng.sample(
                [f for f in real_list if f != real_func],
                min(2, len(real_list) - 1)
            )
            fake_one = self.rng.choice(available_fakes)
            options_b = other_reals + [real_func, fake_one]
            self.rng.shuffle(options_b)
            answer_idx_b   = options_b.index(fake_one)
            answer_label_b = chr(ord('A') + answer_idx_b)
            question_b = (
                f"Which ONE of the following does NOT exist as a function in the "
                f"`{self.library_name}` Python library?\n\n"
                + "\n".join(f"  {chr(ord('A')+j)}) `{opt}`"
                           for j, opt in enumerate(options_b))
                + f"\n\nRespond ONLY in JSON:\n"
                '{"answer": "A", "reasoning": "one sentence"}'
            )
            difficulty_b = self.fake_gen.difficulty_label(fake_one)
            probes.append(Probe(
                probe_id     = f"L0_{hashlib.md5(f'{real_func}_find_fake'.encode()).hexdigest()[:8]}",
                layer        = "L0",
                probe_type   = "recognition",
                prompt       = question_b,
                ground_truth = {
                    "question_type":  "find_fake",
                    "correct_answer": answer_label_b,
                    "fake_func":      fake_one,
                    "options": {chr(ord('A')+j): opt for j, opt in enumerate(options_b)},
                },
                metadata     = {
                    "library": self.library_name,
                    "function_name": real_func,
                    "question_type": "find_fake",
                    "difficulty": difficulty_b,
                },
            ))

        return probes

    # ==========================================================
    # L1: Recall (JSON fill — path + required parameters)
    # All functions, no sampling
    # ==========================================================

    def generate_L1_recall(self) -> List[Probe]:
        """
        For EVERY function, test whether the model can recall:
          1. The correct callable path (e.g. pandapower.create_bus)
          2. ALL required (mandatory) parameter names and types
        """
        probes = []

        for func_info in self.functions:
            params   = func_info.get("parameters", [])
            # Filter out **kwargs — not library-specific knowledge
            params   = [p for p in params if not p["name"].startswith("*")]
            required = [p for p in params if p.get("required")]

            prompt = (
                f"For the function `{func_info['name']}` in the `{self.library_name}` library:\n\n"
                f"1. What is its correct full callable path "
                f"(e.g., {self.library_name}.some_function)?\n"
                f"2. List ALL required (mandatory) parameters with their name and type.\n\n"
                f"Respond ONLY in JSON:\n"
                f'{{"full_path": "...", "required_parameters": [{{"name": "...", "type": "..."}}]}}'
            )

            ground_truth = {
                "full_path": func_info.get("full_path", f"{self.library_name}.{func_info['name']}"),
                "required_parameters": [
                    {"name": p["name"], "type": p.get("type", "unknown")} for p in required
                ],
                "function_name": func_info["name"],
            }

            probes.append(Probe(
                probe_id     = f"L1_{hashlib.md5(func_info['name'].encode()).hexdigest()[:8]}",
                layer        = "L1",
                probe_type   = "recall",
                prompt       = prompt,
                ground_truth = ground_truth,
                metadata     = {
                    "library": self.library_name,
                    "function_name": func_info["name"],
                    "category": func_info.get("category", ""),
                    "n_required_params": len(required),
                },
            ))

        return probes

    # ==========================================================
    # L2: Comprehension (MCQ — function & parameter semantics)
    # Exhaustive: all applicable sub-types for all functions
    # ==========================================================

    # Max number of params to test per function when no required params exist
    _PARAM_FALLBACK_N = 3

    def generate_L2_comprehension(self) -> List[Probe]:
        """
        Generate semantics-focused probes from all available documentation fields:
          func_purpose   — what does the function do? (up to 1 per function)
          param_meaning  — what does this parameter mean? (required params only;
                           if none are required, fall back to first N params)
          return_info    — what does the function return? (up to 1 per function)
          
        param_type and param_default are REMOVED — they test recall (L1-level),
        not comprehension (L2).  This keeps the layer boundary clean.
        """
        probes = []

        # Pre-collect distractor pools from the entire library
        all_func_descriptions  = self._collect_func_descriptions()
        all_param_descriptions = self._collect_param_descriptions()
        all_return_descriptions = self._collect_return_descriptions()

        for func_info in self.functions:
            fname  = func_info["name"]
            desc   = func_info.get("description", "").strip()
            params = [p for p in func_info.get("parameters", [])
                      if not p["name"].startswith("*")]
            returns = func_info.get("return_info", [])

            # ── Sub-type A: Function purpose ────────────────────────
            if desc and len(desc) > 10:
                probe = self._make_func_purpose_probe(
                    func_info, desc, all_func_descriptions
                )
                if probe:
                    probes.append(probe)

            # ── Sub-type B: Parameter meaning (required only, with fallback) ──
            required_params = [p for p in params if p.get("required")]
            if not required_params:
                # Fallback: use first N params that have descriptions
                required_params = params[:self._PARAM_FALLBACK_N]

            for param in required_params:
                pdesc = param.get("description", "").strip()
                if pdesc and len(pdesc) > 5:
                    probe = self._make_param_meaning_probe(
                        func_info, param, all_param_descriptions
                    )
                    if probe:
                        probes.append(probe)

            # ── Sub-type C: Return info ─────────────────────────────
            if returns:
                ret_desc = returns[0].get("description", "").strip()
                if ret_desc and len(ret_desc) > 5:
                    probe = self._make_return_probe(
                        func_info, returns[0], all_return_descriptions
                    )
                    if probe:
                        probes.append(probe)

        return probes

    # ── L2 helper: collect distractor pools ──────────────────────

    def _collect_func_descriptions(self) -> List[Tuple[str, str]]:
        result = []
        for f in self.functions:
            desc = f.get("description", "").strip()
            if desc and len(desc) > 10:
                result.append((f["name"], desc))
        return result

    def _collect_param_descriptions(self) -> List[Tuple[str, str, str]]:
        result = []
        for f in self.functions:
            for p in f.get("parameters", []):
                pdesc = p.get("description", "").strip()
                if pdesc and len(pdesc) > 5:
                    result.append((f["name"], p["name"], pdesc))
        return result


    def _collect_return_descriptions(self) -> List[Tuple[str, str]]:
        result = []
        for f in self.functions:
            for r in f.get("return_info", []):
                rdesc = r.get("description", "").strip()
                if rdesc and len(rdesc) > 5:
                    result.append((f["name"], rdesc))
        return result

    # ── L2 helper: make individual probe sub-types ───────────────

    def _make_func_purpose_probe(
        self, func_info: dict, correct_desc: str,
        all_descs: List[Tuple[str, str]]
    ) -> Optional[Probe]:
        fname = func_info["name"]
        prefix = fname.split("_")[0] + "_"

        same_prefix = [(n, d) for n, d in all_descs
                       if n != fname and n.startswith(prefix)]
        diff_prefix = [(n, d) for n, d in all_descs
                       if n != fname and not n.startswith(prefix)]

        distractor_pool = same_prefix + diff_prefix
        if len(distractor_pool) < 3:
            return None

        distractors = self.rng.sample(distractor_pool, 3)
        options_raw = [(correct_desc, True)] + [(d, False) for _, d in distractors]
        self.rng.shuffle(options_raw)

        options = {}
        correct_label = None
        for j, (text, is_correct) in enumerate(options_raw):
            label = chr(ord('A') + j)
            display = text[:200] + "..." if len(text) > 200 else text
            options[label] = display
            if is_correct:
                correct_label = label

        question = (
            f"What does the function `{fname}` do in `{self.library_name}`?\n\n"
            + "\n".join(f"  {lbl}) {txt}" for lbl, txt in options.items())
            + f"\n\nRespond ONLY in JSON:\n"
            '{"answer": "A", "reasoning": "one sentence"}'
        )

        return Probe(
            probe_id     = f"L2_fp_{hashlib.md5(fname.encode()).hexdigest()[:8]}",
            layer        = "L2",
            probe_type   = "comprehension",
            prompt       = question,
            ground_truth = {
                "sub_type":       "func_purpose",
                "correct_answer": correct_label,
                "correct_text":   correct_desc,
                "options":        options,
            },
            metadata     = {
                "library": self.library_name,
                "function_name": fname,
                "sub_type": "func_purpose",
            },
        )

    def _make_param_meaning_probe(
        self, func_info: dict, param: dict,
        all_param_descs: List[Tuple[str, str, str]]
    ) -> Optional[Probe]:
        fname = func_info["name"]
        pname = param["name"]
        correct_desc = param["description"]

        candidates = [(fn, pn, pd) for fn, pn, pd in all_param_descs
                      if not (fn == fname and pn == pname)]
        if len(candidates) < 3:
            return None

        distractors = self.rng.sample(candidates, 3)
        options_raw = [(correct_desc, True)] + [(pd, False) for _, _, pd in distractors]
        self.rng.shuffle(options_raw)

        options = {}
        correct_label = None
        for j, (text, is_correct) in enumerate(options_raw):
            label = chr(ord('A') + j)
            display = text[:200] + "..." if len(text) > 200 else text
            options[label] = display
            if is_correct:
                correct_label = label

        question = (
            f"In `{self.library_name}.{fname}`, what does the parameter "
            f"`{pname}` represent?\n\n"
            + "\n".join(f"  {lbl}) {txt}" for lbl, txt in options.items())
            + f"\n\nRespond ONLY in JSON:\n"
            '{"answer": "A", "reasoning": "one sentence"}'
        )

        return Probe(
            probe_id     = f"L2_pm_{hashlib.md5(f'{fname}_{pname}'.encode()).hexdigest()[:8]}",
            layer        = "L2",
            probe_type   = "comprehension",
            prompt       = question,
            ground_truth = {
                "sub_type":       "param_meaning",
                "correct_answer": correct_label,
                "correct_text":   correct_desc,
                "options":        options,
            },
            metadata     = {
                "library": self.library_name,
                "function_name": fname,
                "param_name": pname,
                "sub_type": "param_meaning",
            },
        )


    def _make_return_probe(
        self, func_info: dict, return_info: dict,
        all_returns: List[Tuple[str, str]]
    ) -> Optional[Probe]:
        fname = func_info["name"]
        correct_ret = return_info.get("description", "").strip()

        other_returns = [(fn, rd) for fn, rd in all_returns if fn != fname]
        if len(other_returns) < 3:
            return None

        distractors = self.rng.sample(other_returns, 3)
        options_raw = [(correct_ret, True)] + [(rd, False) for _, rd in distractors]
        self.rng.shuffle(options_raw)

        options = {}
        correct_label = None
        for j, (text, is_correct) in enumerate(options_raw):
            label = chr(ord('A') + j)
            display = text[:200] + "..." if len(text) > 200 else text
            options[label] = display
            if is_correct:
                correct_label = label

        question = (
            f"What does `{self.library_name}.{fname}` return?\n\n"
            + "\n".join(f"  {lbl}) {txt}" for lbl, txt in options.items())
            + f"\n\nRespond ONLY in JSON:\n"
            '{"answer": "A", "reasoning": "one sentence"}'
        )

        return Probe(
            probe_id     = f"L2_rt_{hashlib.md5(f'{fname}_return'.encode()).hexdigest()[:8]}",
            layer        = "L2",
            probe_type   = "comprehension",
            prompt       = question,
            ground_truth = {
                "sub_type":       "return_info",
                "correct_answer": correct_label,
                "correct_text":   correct_ret,
                "options":        options,
            },
            metadata     = {
                "library": self.library_name,
                "function_name": fname,
                "sub_type": "return_info",
            },
        )

    # ==========================================================
    # L3: Application (write + execute single-function code)
    # All functions, no filtering
    # ==========================================================

    def generate_L3_application(self) -> List[Probe]:
        """
        For EVERY function, generate 1 code-generation probe.
        No primary/secondary filtering — full coverage.
        """
        probes = []

        for func_info in self.functions:
            fname = func_info["name"]
            description = func_info.get("description", "")

            prompt = (
                f"Write a minimal, self-contained, executable Python script that "
                f"demonstrates the use of the `{fname}` function from the "
                f"`{self.library_name}` library.\n\n"
                f"Description: {description}\n\n"
                f"Requirements:\n"
                f"- Include all necessary imports.\n"
                f"- Create any prerequisite objects (network, buses, etc.) as needed.\n"
                f"- Use only valid parameters for this exact function.\n"
                f"- Keep it as short as possible while being complete and runnable.\n"
                f"- Print a brief confirmation at the end.\n\n"
                f"Respond with ONLY the Python code in a ```python``` block."
            )

            probes.append(Probe(
                probe_id     = f"L3_{hashlib.md5(fname.encode()).hexdigest()[:8]}",
                layer        = "L3",
                probe_type   = "application",
                prompt       = prompt,
                ground_truth = {
                    "function_name":  fname,
                    "full_path":      func_info.get("full_path", ""),
                    "required_params": [
                        p["name"] for p in func_info.get("parameters", [])
                        if p.get("required") and not p["name"].startswith("*")
                    ],
                },
                metadata     = {
                    "library": self.library_name,
                    "function_name": fname,
                    "category": func_info.get("category", ""),
                },
            ))

        return probes

    # ==========================================================
    # Generate All Probes
    # ==========================================================

    def generate_all(self) -> Dict[str, List[Probe]]:
        """Generate probes for all four layers using all available doc fields."""
        probes = {
            "L0_recognition":   self.generate_L0_recognition(),
            "L1_recall":        self.generate_L1_recall(),
            "L2_comprehension": self.generate_L2_comprehension(),
            "L3_application":   self.generate_L3_application(),
        }
        for key, plist in probes.items():
            funcs = {p.metadata.get("function_name") for p in plist}
            print(f"  {key}: {len(plist)} probes, {len(funcs)} unique functions, "
                  f"avg {len(plist)/len(funcs):.1f} probes/func" if funcs else
                  f"  {key}: {len(plist)} probes")
        return probes


# ============================================================
# Section 4: Injection Snippet Generator
# ============================================================

class InjectionGenerator:
    """
    Pre-generates documentation fragments to inject when a model fails
    at a specific layer for a specific function.
    
    Layer → Injection mapping:
      L0  →  function name + one-line description     (~20 tokens)
      L1  →  full signature string                     (~50 tokens)
      L2  →  parameter descriptions + return info      (~100-300 tokens)
      L3  →  example code snippet                      (~50-200 tokens)
    """

    def __init__(self, json_path: str, library_name: str = None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.library_name = (
            library_name
            or self.data.get("library")
            or self.data.get("library_name")
            or ("pandapower" if "pandapower" in str(json_path).lower() else "unknown")
        )
        # Deduplicate by function name (same logic as ProbeGenerator)
        raw = self.data.get("functions", [])
        seen = set()
        self.functions = []
        for f in raw:
            if f["name"] not in seen:
                seen.add(f["name"])
                self.functions.append(f)
        self.func_map  = {f["name"]: f for f in self.functions}

    def generate_all_snippets(self) -> Dict[str, Dict[str, InjectionSnippet]]:
        snippets = {"L0": {}, "L1": {}, "L2": {}, "L3": {}}
        for func_info in self.functions:
            fname = func_info["name"]
            snippets["L0"][fname] = self._make_L0_snippet(func_info)
            snippets["L1"][fname] = self._make_L1_snippet(func_info)
            snippets["L2"][fname] = self._make_L2_snippet(func_info)
            snippets["L3"][fname] = self._make_L3_snippet(func_info)
        return snippets

    def _make_L0_snippet(self, func_info: dict) -> InjectionSnippet:
        desc = func_info.get("description", "").strip()
        first_sentence = desc.split(".")[0] + "." if desc else "No description available."
        content = f"- `{func_info['name']}`: {first_sentence}"
        return InjectionSnippet(
            function_name=func_info["name"], layer="L0",
            content=content, token_estimate=len(content.split()) + 5,
        )

    def _make_L1_snippet(self, func_info: dict) -> InjectionSnippet:
        signature = func_info.get("signature", "")
        if not signature:
            params = func_info.get("parameters", [])
            param_strs = []
            for p in params:
                if p["name"].startswith("*"):
                    continue
                if p.get("required"):
                    param_strs.append(p["name"])
                else:
                    default = p.get("default", "None")
                    param_strs.append(f"{p['name']}={default}")
            signature = f"{func_info.get('full_path', func_info['name'])}({', '.join(param_strs)})"
        content = f"Signature: `{signature}`"
        return InjectionSnippet(
            function_name=func_info["name"], layer="L1",
            content=content, token_estimate=len(content.split()) + 5,
        )

    def _make_L2_snippet(self, func_info: dict) -> InjectionSnippet:
        lines = []
        desc = func_info.get("description", "").strip()
        if desc:
            lines.append(f"Description: {desc}")
        params = func_info.get("parameters", [])
        param_lines = []
        for p in params:
            if p["name"].startswith("*"):
                continue
            pdesc = p.get("description", "").strip()
            ptype = p.get("type", "").strip()
            pdefault = p.get("default", "")
            parts = [f"  - `{p['name']}`"]
            if ptype and ptype.lower() != "any":
                parts.append(f"({ptype})")
            if pdesc:
                parts.append(f": {pdesc}")
            if pdefault and str(pdefault).lower() not in ("none", "null", ""):
                parts.append(f"[default: {pdefault}]")
            if p.get("required"):
                parts.append("**required**")
            param_lines.append(" ".join(parts))
        if param_lines:
            lines.append("Parameters:")
            lines.extend(param_lines)
        returns = func_info.get("return_info", [])
        for r in returns:
            rdesc = r.get("description", "").strip()
            if rdesc:
                lines.append(f"Returns: {rdesc}")
                break
        content = "\n".join(lines)
        return InjectionSnippet(
            function_name=func_info["name"], layer="L2",
            content=content, token_estimate=len(content.split()) + 10,
        )

    def _make_L3_snippet(self, func_info: dict) -> InjectionSnippet:
        examples = func_info.get("examples", [])
        if examples:
            code = examples[0] if isinstance(examples[0], str) else str(examples[0])
            content = f"Example usage of `{func_info['name']}`:\n```python\n{code}\n```"
        else:
            content = f"No example available for `{func_info['name']}`."
        return InjectionSnippet(
            function_name=func_info["name"], layer="L3",
            content=content, token_estimate=len(content.split()) + 10,
        )

    def build_injection_prompt(
        self,
        profile: Dict[str, Dict[str, Any]],
        snippets: Dict[str, Dict[str, InjectionSnippet]],
        token_budget: int = 4000,
    ) -> str:
        """
        Given a model's knowledge profile and pre-generated snippets,
        build a compact injection prompt that targets the worst gaps first.
        
        Profile values may be dicts with a "score" key or plain floats.
        """
        injection_plan = []

        for fname, layer_data in profile.items():
            if fname not in snippets.get("L0", {}):
                continue

            THRESHOLDS = {"L0": 0.5, "L1": 0.5, "L2": 0.5, "L3": 0.5}
            lowest_fail = None
            for layer in ["L0", "L1", "L2", "L3"]:
                val = layer_data.get(layer)
                # Support both {"score": 0.5, ...} and plain float
                score = val["score"] if isinstance(val, dict) else val
                if score is not None and score < THRESHOLDS[layer]:
                    lowest_fail = layer
                    break

            if lowest_fail is None:
                continue

            layer_order = ["L0", "L1", "L2", "L3"]
            fail_idx = layer_order.index(lowest_fail)
            func_snippets = []
            total_tokens = 0
            for li in range(fail_idx + 1):
                lyr = layer_order[li]
                snip = snippets.get(lyr, {}).get(fname)
                if snip:
                    func_snippets.append(snip)
                    total_tokens += snip.token_estimate

            injection_plan.append({
                "function": fname,
                "lowest_fail": lowest_fail,
                "severity": fail_idx,
                "snippets": func_snippets,
                "total_tokens": total_tokens,
            })

        injection_plan.sort(key=lambda x: (x["severity"], x["function"]))

        selected = []
        remaining_budget = token_budget
        for item in injection_plan:
            if item["total_tokens"] <= remaining_budget:
                selected.append(item)
                remaining_budget -= item["total_tokens"]

        if not selected:
            return ""

        lines = [
            f"## {self.library_name} API Reference (knowledge supplement)\n",
            "The following functions may be relevant to your task:\n",
        ]
        for item in selected:
            lines.append(f"### `{item['function']}`")
            for snip in item["snippets"]:
                lines.append(snip.content)
            lines.append("")
        return "\n".join(lines)


# ============================================================
# Section 6: Probe Evaluator
# ============================================================

class ProbeEvaluator:
    """
    Evaluates model responses against probe ground truths.
    Each layer has its own evaluator method.
    """

    def __init__(self, json_path: str = None):
        self.func_map = {}
        self.func_names = set()
        self.library_name = "unknown"
        self.import_aliases = []
        if json_path:
            with open(json_path, "r") as f:
                data = json.load(f)
            self.func_map = {f["name"]: f for f in data.get("functions", [])}
            self.func_names = set(self.func_map.keys())
            self.library_name = (
                data.get("library")
                or data.get("library_name")
                or "unknown"
            )
            aliases = [self.library_name] + list(data.get("import_aliases") or [])
            self.import_aliases = list(dict.fromkeys(str(a) for a in aliases if a))

    # ── Shared utilities ─────────────────────────────────────────

    @staticmethod
    def parse_json_response(text: str) -> Optional[dict]:
        if not text:
            return None
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        for pat in [r'```json\s*\n?(.*?)\n?\s*```',
                    r'```\s*\n?(.*?)\n?\s*```',
                    r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}']:
            m = re.search(pat, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1) if '```' in pat else m.group(0))
                except (json.JSONDecodeError, IndexError):
                    pass
        return None

    @staticmethod
    def extract_code(text: str) -> Optional[str]:
        if not text:
            return None
        text = text.strip()
        for marker in ['assistantfinal```', 'assistant```', 'final```']:
            if marker in text:
                text = '```' + text.split(marker, 1)[1]
                break
        m = re.search(r'```python\s*\n(.*?)\n\s*```', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r'```\s*\n(.*?)\n\s*```', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        s = text.strip()
        if s.startswith(("import ", "from ", "#")):
            return s
        return None

    # ── L0: Recognition ──────────────────────────────────────────

    def evaluate_L0(self, probe: Probe, response: str) -> EvalResult:
        parsed = self.parse_json_response(response)
        gt = probe.ground_truth
        scores  = {"accuracy": 0.0, "parseable": 0.0}
        details = {"raw_response": response, "question_type": gt["question_type"]}

        if parsed is None:
            details["error_category"] = "invalid_response"
            return EvalResult(probe.probe_id, "L0", "recognition", scores, details)

        scores["parseable"] = 1.0
        predicted = str(parsed.get("answer", "")).strip().upper()
        correct   = gt["correct_answer"].upper()

        if predicted == correct:
            scores["accuracy"] = 1.0
            details["error_category"] = "correct"
        else:
            details["error_category"] = (
                "hallucination" if gt["question_type"] == "find_real"
                else "false_rejection"
            )

        details.update({
            "predicted": predicted,
            "correct":   correct,
            "reasoning": parsed.get("reasoning", ""),
            "options":   gt.get("options", {}),
        })
        return EvalResult(probe.probe_id, "L0", "recognition", scores, details)

    # ── L1: Recall ───────────────────────────────────────────────

    def evaluate_L1(self, probe: Probe, response: str) -> EvalResult:
        parsed = self.parse_json_response(response)
        gt = probe.ground_truth
        scores = {
            "parseable": 0.0, "path_correct": 0.0,
            "required_params_jaccard": 0.0,
            "required_params_precision": 0.0,
            "required_params_recall": 0.0,
            "overall": 0.0,
        }
        details = {"raw_response": response}

        if parsed is None:
            details["parse_error"] = "JSON parse failed"
            return EvalResult(probe.probe_id, "L1", "recall", scores, details)

        scores["parseable"] = 1.0

        pred_path = (parsed.get("full_path") or "").strip().rstrip("()")
        gt_path   = gt["full_path"].strip().rstrip("()")
        pred_func = pred_path.split(".")[-1] if "." in pred_path else pred_path
        gt_func   = gt_path.split(".")[-1] if "." in gt_path else gt_path
        scores["path_correct"] = 1.0 if (
            pred_path == gt_path or pred_func == gt_func
        ) else 0.0

        gt_names = {p["name"] for p in gt["required_parameters"]}
        pred_names = set()
        for p in parsed.get("required_parameters", []):
            pred_names.add(p.get("name", "") if isinstance(p, dict) else str(p))

        if gt_names or pred_names:
            inter = gt_names & pred_names
            union = gt_names | pred_names
            scores["required_params_jaccard"]   = len(inter) / len(union) if union else 0.0
            scores["required_params_precision"] = len(inter) / len(pred_names) if pred_names else 0.0
            scores["required_params_recall"]    = len(inter) / len(gt_names) if gt_names else 1.0
        else:
            scores["required_params_jaccard"] = 1.0
            scores["required_params_precision"] = 1.0
            scores["required_params_recall"] = 1.0

        scores["overall"] = 0.5 * scores["path_correct"] + 0.5 * scores["required_params_jaccard"]

        details.update({
            "predicted_path": pred_path,
            "gt_path": gt_path,
            "gt_required_params": sorted(gt_names),
            "predicted_required_params": sorted(pred_names),
            "hallucinated_params": sorted(pred_names - gt_names),
            "missed_params": sorted(gt_names - pred_names),
        })
        return EvalResult(probe.probe_id, "L1", "recall", scores, details)

    # ── L2: Comprehension ────────────────────────────────────────

    def evaluate_L2(self, probe: Probe, response: str) -> EvalResult:
        parsed = self.parse_json_response(response)
        gt = probe.ground_truth
        sub_type = gt.get("sub_type", "unknown")
        scores  = {"accuracy": 0.0, "parseable": 0.0}
        details = {"raw_response": response, "sub_type": sub_type}

        if parsed is None:
            details["parse_error"] = "JSON parse failed"
            return EvalResult(probe.probe_id, "L2", "comprehension", scores, details)

        scores["parseable"] = 1.0
        predicted = str(parsed.get("answer", "")).strip().upper()
        correct   = gt["correct_answer"].upper()
        scores["accuracy"] = 1.0 if predicted == correct else 0.0

        details.update({
            "predicted": predicted,
            "correct":   correct,
            "reasoning": parsed.get("reasoning", ""),
            "options":   gt.get("options", {}),
        })
        return EvalResult(probe.probe_id, "L2", "comprehension", scores, details)

    # ── L3: Application ──────────────────────────────────────────

    def evaluate_L3(self, probe: Probe, response: str) -> EvalResult:
        scores = {
            "code_extracted": 0.0,
            "api_validity_rate": 0.0,
            "execution_success": 0.0,
            "target_function_used": 0.0,
            "error_type": None,
        }
        details = {"raw_response": response}

        code = self.extract_code(response)
        if code is None:
            details["error"] = "No code extracted"
            return EvalResult(probe.probe_id, "L3", "application", scores, details)

        scores["code_extracted"] = 1.0
        details["extracted_code"] = code

        api_check = self._check_api_validity(code)
        scores["api_validity_rate"] = api_check["api_validity_rate"]
        details["api_check"] = api_check

        # Check if the target function was actually called
        target_func = probe.ground_truth.get("function_name", "")
        if target_func:
            used_funcs = set(api_check.get("all_used", []))
            used = target_func in used_funcs
            if not used and "_" in target_func:
                # Class_method-style entries called on instance variables
                # (e.g. Network_pf → n.pf()) are invisible to alias-based
                # extraction; accept a .method( call co-occurring with the
                # class being constructed or referenced.
                cls, meth = target_func.split("_", 1)
                if re.search(rf"\.{re.escape(meth)}\s*\(", code) and re.search(
                    rf"\b{re.escape(cls)}\s*\(", code
                ):
                    used = True
            if not used:
                # Instance-level entries documented under their bare member
                # name (e.g. pypsa "pf", "buses_t"): alias extraction never
                # sees `n.pf(...)` / `n.buses_t...`, so accept attribute-style
                # access on any receiver — parenless for attribute entries,
                # call-form otherwise.
                entry = self.func_map.get(target_func, {})
                entry_type = entry.get("type", "function")
                if entry_type == "attribute":
                    # Accept the documented name and the dotted member tail of
                    # full_path (e.g. pf_converged / pypsa.Network.pf.converged
                    # → ".pf_converged" or ".pf.converged").
                    pats = [rf"\.{re.escape(target_func)}\b"]
                    fp_parts = entry.get("full_path", "").split(".")
                    if len(fp_parts) > 2:
                        tail = ".".join(fp_parts[2:])
                        if tail and tail != target_func:
                            pats.append(rf"\.{re.escape(tail)}\b")
                    used = any(re.search(pat, code) for pat in pats)
                else:
                    used = bool(re.search(rf"\.{re.escape(target_func)}\s*\(", code))
            scores["target_function_used"] = 1.0 if used else 0.0

        exec_result = execute_code_safely_subprocess(code)
        error_msg = parse_error(exec_result, code)
        scores["execution_success"] = 1.0 if exec_result["success"] else 0.0
        scores["error_type"] = exec_result.get("error_type")
        details["error_msg"] = error_msg
        details["exec_result"] = exec_result

        return EvalResult(probe.probe_id, "L3", "application", scores, details)

    def _check_api_validity(self, code: str) -> dict:
        if not self.func_names:
            return {"api_validity_rate": -1.0, "error": "No API spec loaded"}

        lib_re = re.escape(self.library_name)
        aliases = set(self.import_aliases or [self.library_name])

        # import pandapower as pp
        # import pandapower.networks as pn
        # import pandapower.networks
        for match in re.finditer(
            rf'^\s*import\s+({lib_re}(?:\.[A-Za-z_]\w*)*)\s*(?:as\s+([A-Za-z_]\w*))?',
            code,
            flags=re.MULTILINE,
        ):
            module = match.group(1)
            alias = match.group(2)
            if alias:
                aliases.add(alias)
            else:
                aliases.add(module.split(".")[0])
                aliases.add(module)

        direct_alias_to_name = {}
        for match in re.finditer(
            rf'^\s*from\s+{lib_re}(?:\.[A-Za-z_]\w*)*\s+import\s+(.+)$',
            code,
            flags=re.MULTILINE,
        ):
            imp_line = match.group(1).split("#", 1)[0]
            for raw_name in imp_line.split(","):
                raw_name = raw_name.strip()
                if not raw_name or raw_name == "*":
                    continue
                parts = re.split(r'\s+as\s+', raw_name, maxsplit=1)
                canonical = parts[0].strip()
                alias = parts[1].strip() if len(parts) == 2 else canonical
                if canonical and alias:
                    direct_alias_to_name[alias] = canonical

        all_used = set()
        for alias in aliases:
            # Capture pp.runpp(...), pn.case5(...), and pandapower.networks.case5(...)
            pattern = rf'\b{re.escape(alias)}\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\('
            for dotted in re.findall(pattern, code):
                # Module_Method-style corpora (e.g. OpenDSSDirect) document
                # dotted accessors under underscore-joined names; prefer that
                # form when it is the documented one, else keep the last part.
                underscore = dotted.replace(".", "_")
                if underscore in self.func_names:
                    all_used.add(underscore)
                else:
                    all_used.add(dotted.split(".")[-1])

        called_names = set(re.findall(r'\b([A-Za-z_]\w*)\s*\(', code))
        for alias, canonical in direct_alias_to_name.items():
            if alias in called_names:
                all_used.add(canonical)

        if not all_used:
            return {
                "api_validity_rate": 0.0,
                "valid_functions": [], "invalid_functions": [], "all_used": [],
            }

        valid   = all_used & self.func_names
        invalid = all_used - self.func_names
        rate    = len(valid) / len(all_used)

        return {
            "api_validity_rate": round(rate, 4),
            "valid_functions":   sorted(valid),
            "invalid_functions": sorted(invalid),
            "all_used":          sorted(all_used),
        }

    # ── Dispatch ─────────────────────────────────────────────────

    def evaluate(self, probe: Probe, response: str) -> EvalResult:
        dispatch = {"L0": self.evaluate_L0, "L1": self.evaluate_L1,
                     "L2": self.evaluate_L2, "L3": self.evaluate_L3}
        fn = dispatch.get(probe.layer)
        if fn:
            result = fn(probe, response)
            result.details["function_name"] = probe.metadata.get("function_name", "")
            return result
        return EvalResult(probe.probe_id, probe.layer, probe.probe_type,
                        {"error": f"no evaluator for layer {probe.layer}"}, {})

    def evaluate_all(self, probes_with_responses: List[Tuple[Probe, str]]) -> List[EvalResult]:
        return [self.evaluate(p, r) for p, r in probes_with_responses]


# ============================================================
# Section 7: Summary & Profile Utilities
# ============================================================

def compute_L0_summary(results: List[dict]) -> dict:
    """Summarise L0 results by question type."""
    find_real_correct = find_real_total = 0
    find_fake_correct = find_fake_total = 0

    for r in results:
        if isinstance(r, EvalResult):
            qtype = r.details.get("question_type")
            acc   = r.scores.get("accuracy")
        else:
            qtype = r.get("details", {}).get("question_type")
            acc   = r.get("scores", {}).get("accuracy")

        if acc is None:
            continue
        if qtype == "find_real":
            find_real_total += 1
            find_real_correct += int(acc == 1.0)
        elif qtype == "find_fake":
            find_fake_total += 1
            find_fake_correct += int(acc == 1.0)

    overall_total   = find_real_total + find_fake_total
    overall_correct = find_real_correct + find_fake_correct

    return {
        "overall_accuracy":   round(overall_correct / overall_total, 4) if overall_total else 0.0,
        "find_real_accuracy": round(find_real_correct / find_real_total, 4) if find_real_total else 0.0,
        "find_fake_accuracy": round(find_fake_correct / find_fake_total, 4) if find_fake_total else 0.0,
        "n_find_real": find_real_total,
        "n_find_fake": find_fake_total,
    }


def compute_L2_subtype_summary(results: List[dict]) -> dict:
    """Summarise L2 results by sub-type (aggregate across all functions)."""
    subtype_correct = defaultdict(int)
    subtype_total   = defaultdict(int)

    for r in results:
        if isinstance(r, EvalResult):
            sub = r.details.get("sub_type", "unknown")
            acc = r.scores.get("accuracy")
        else:
            sub = r.get("details", {}).get("sub_type", "unknown")
            acc = r.get("scores", {}).get("accuracy")

        if acc is None:
            continue
        subtype_total[sub] += 1
        subtype_correct[sub] += int(acc == 1.0)

    summary = {}
    for sub in sorted(subtype_total.keys()):
        total = subtype_total[sub]
        correct = subtype_correct[sub]
        summary[sub] = {
            "accuracy": round(correct / total, 4) if total else 0.0,
            "n_probes": total,
        }
    return summary


def build_knowledge_profile(
    all_results: Dict[str, List[dict]],
    **kwargs,
) -> Dict[str, Dict[str, dict]]:
    """
    Build a per-function knowledge profile from probe evaluation results.
    
    Clean separation of two structural patterns:
    
      by_probe_type  (L0, L2) — multiple independent probes grouped by type.
                                n_probes counts actual questions, and they sum
                                to the layer total.
      diagnostics    (L1, L3) — one probe per function, multiple scoring
                                dimensions extracted from the same response.
                                No n_probes inside diagnostics (it's always
                                the same 1 probe).
    
    Returns:
        {
            "create_bus": {
                "L0": {
                    "score": 0.5,
                    "n_probes": 2,
                    "by_probe_type": {
                        "find_real": {"score": 1.0, "n_probes": 1},
                        "find_fake": {"score": 0.0, "n_probes": 1},
                    }
                },
                "L1": {
                    "score": 0.75,
                    "n_probes": 1,
                    "diagnostics": {
                        "path_correct": 1.0,
                        "required_params_jaccard": 0.5,
                        "required_params_precision": 0.67,
                        "required_params_recall": 0.5,
                    }
                },
                "L2": {
                    "score": 0.67,
                    "n_probes": 3,
                    "by_probe_type": {
                        "func_purpose":  {"score": 1.0, "n_probes": 1},
                        "param_meaning": {"score": 0.5, "n_probes": 2},
                    }
                },
                "L3": {
                    "score": 0.0,
                    "n_probes": 1,
                    "diagnostics": {
                        "execution_success": 0.0,
                        "api_validity_rate": 0.5,
                        "target_function_used": 0.0,
                    }
                },
            },
            ...
        }
    """
    # ── Stage 1: collect raw scores ──────────────────────────────
    # L0/L2: func → layer → probe_type → [accuracy_values]
    # L1:    func → "L1" → metric_name → [values]
    # L3:    func → "L3" → metric_name → [values]
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for layer_key, results in all_results.items():
        for r in results:
            if isinstance(r, dict):
                fname    = r.get("details", {}).get("function_name")
                scores_d = r.get("scores", {})
                details  = r.get("details", {})
            else:
                fname    = getattr(r, 'details', {}).get("function_name")
                scores_d = r.scores if hasattr(r, 'scores') else {}
                details  = r.details if hasattr(r, 'details') else {}

            if not fname:
                continue

            if "L0" in layer_key:
                qtype = details.get("question_type", "unknown")
                raw[fname]["L0"][qtype].append(scores_d.get("accuracy", 0.0))

            elif "L1" in layer_key:
                raw[fname]["L1"]["path_correct"].append(
                    scores_d.get("path_correct", 0.0))
                raw[fname]["L1"]["required_params_jaccard"].append(
                    scores_d.get("required_params_jaccard", 0.0))
                raw[fname]["L1"]["required_params_precision"].append(
                    scores_d.get("required_params_precision", 0.0))
                raw[fname]["L1"]["required_params_recall"].append(
                    scores_d.get("required_params_recall", 0.0))

            elif "L2" in layer_key:
                sub = details.get("sub_type", "unknown")
                raw[fname]["L2"][sub].append(scores_d.get("accuracy", 0.0))

            elif "L3" in layer_key:
                raw[fname]["L3"]["execution_success"].append(
                    scores_d.get("execution_success", 0.0))
                raw[fname]["L3"]["api_validity_rate"].append(
                    scores_d.get("api_validity_rate", 0.0))
                raw[fname]["L3"]["target_function_used"].append(
                    scores_d.get("target_function_used", 0.0))

    # ── Stage 2: aggregate into profile ──────────────────────────
    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    profile = {}
    for fname, layers in raw.items():
        profile[fname] = {}

        # ── L0: by_probe_type (independent probes) ──
        if "L0" in layers:
            groups = layers["L0"]
            by_pt = {}
            all_acc = []
            for ptype, vals in groups.items():
                by_pt[ptype] = {"score": _mean(vals), "n_probes": len(vals)}
                all_acc.extend(vals)
            profile[fname]["L0"] = {
                "score": _mean(all_acc),
                "n_probes": len(all_acc),
                "by_probe_type": by_pt,
            }
        else:
            profile[fname]["L0"] = None

        # ── L1: diagnostics (multi-metric from same probe) ──
        if "L1" in layers:
            d = layers["L1"]
            path  = _mean(d.get("path_correct", []))
            jacc  = _mean(d.get("required_params_jaccard", []))
            prec  = _mean(d.get("required_params_precision", []))
            rec   = _mean(d.get("required_params_recall", []))
            n_probes = len(d.get("path_correct", []))
            profile[fname]["L1"] = {
                "score": round(0.5 * path + 0.5 * jacc, 4),
                "n_probes": n_probes,
                "diagnostics": {
                    "path_correct": path,
                    "required_params_jaccard": jacc,
                    "required_params_precision": prec,
                    "required_params_recall": rec,
                },
            }
        else:
            profile[fname]["L1"] = None

        # ── L2: by_probe_type (independent probes) ──
        if "L2" in layers:
            groups = layers["L2"]
            by_pt = {}
            all_acc = []
            for ptype, vals in groups.items():
                by_pt[ptype] = {"score": _mean(vals), "n_probes": len(vals)}
                all_acc.extend(vals)
            profile[fname]["L2"] = {
                "score": _mean(all_acc),
                "n_probes": len(all_acc),
                "by_probe_type": by_pt,
            }
        else:
            profile[fname]["L2"] = None

        # ── L3: diagnostics (multi-metric from same probe) ──
        if "L3" in layers:
            d = layers["L3"]
            ex   = _mean(d.get("execution_success", []))
            api  = _mean(d.get("api_validity_rate", []))
            tgt  = _mean(d.get("target_function_used", []))
            application = round(ex * tgt, 4)
            n_probes = len(d.get("execution_success", []))
            profile[fname]["L3"] = {
                "score": application,  # executable use of the target API
                "n_probes": n_probes,
                "diagnostics": {
                    "execution_success": ex,
                    "api_validity_rate": api,
                    "target_function_used": tgt,
                    "application_score": application,
                },
            }
        else:
            profile[fname]["L3"] = None

    return profile


# ============================================================
# Section 8: Save / Load
# ============================================================

def save_probes(probes: Dict[str, List[Probe]], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: [asdict(p) for p in v] for k, v in probes.items()},
            f, indent=2, ensure_ascii=False,
        )
    total = sum(len(v) for v in probes.values())
    print(f"Saved {total} probes to {output_path}")


def load_probes(input_path: str) -> Dict[str, List[Probe]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: [Probe(**p) for p in v] for k, v in data.items()}


def save_results(results: List[EvalResult], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} results to {output_path}")


def save_injection_snippets(snippets: Dict[str, Dict[str, InjectionSnippet]], output_path: str):
    serializable = {}
    for layer, func_map in snippets.items():
        serializable[layer] = {
            fname: asdict(snip) for fname, snip in func_map.items()
        }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    total = sum(len(v) for v in snippets.values())
    print(f"Saved {total} injection snippets to {output_path}")


# ============================================================
# Section 9: Backward Compatibility
# ============================================================

def compute_l0_auc(results) -> dict:
    """Backward-compatible wrapper."""
    summary = compute_L0_summary(results)
    return {
        "auc":               summary["overall_accuracy"],
        "best_threshold":    None,
        "best_balanced_acc": summary["overall_accuracy"],
        "n_real":            summary["n_find_real"],
        "n_fake":            summary["n_find_fake"],
        **summary,
    }


# ============================================================
# Section 10: Main (Demo)
# ============================================================

def main():
    json_path  = "dataset/pandapower_docs.json"
    output_dir = Path("dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading API spec from {json_path}...")
    generator = ProbeGenerator(json_path)
    print(f"Library  : {generator.library_name}")
    print(f"Functions: {len(generator.functions)}")

    # ── Generate probes ───────────────────────────────────
    print("\n📋 Generating probes...")
    all_probes = generator.generate_all()
    save_probes(all_probes, str(output_dir / "all_probes.json"))

    # ── Generate injection snippets ──────────────────────────────
    print("\n💉 Generating injection snippets...")
    injector = InjectionGenerator(json_path)
    snippets = injector.generate_all_snippets()
    save_injection_snippets(snippets, str(output_dir / "injection_snippets.json"))

    # ── Show sample probes ───────────────────────────────────────
    print("\n📝 Sample probes:")
    for layer_key, probes in all_probes.items():
        if probes:
            print(f"\n  [{layer_key}] (showing 1 of {len(probes)})")
            print(f"  Prompt: {probes[0].prompt[:200]}...")

    # ── Show sample injection ────────────────────────────────────
    print("\n💉 Sample injection snippets for 'create_empty_network':")
    for layer in ["L0", "L1", "L2", "L3"]:
        snip = snippets.get(layer, {}).get("create_empty_network")
        if snip:
            print(f"\n  [{layer}] ({snip.token_estimate} tokens est.)")
            print(f"  {snip.content[:200]}...")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
