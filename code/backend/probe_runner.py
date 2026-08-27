# --------------------------------------------------------------------------
# Repository copy of the pipeline module probe_runner.py, with imports and
# data-path constants adapted to this repository's layout. Requires the GPU
# serving stack (vLLM/torch) and/or provider API keys supplied via environment
# variables; archived to document the frozen runs. See code/README.md for the
# module map.
# --------------------------------------------------------------------------
"""Probe Evaluation Runner with integrated Benchmark Evaluation.

A single ``MultiModelComparison`` instance loads each model once and runs
both the L0--L3 probe sweep and the PowerCodeBench benchmark, so a model
does not have to be re-loaded between phases.

Usage:
    comp = MultiModelComparison.from_model_names(
        model_names=[...],
        probes_path="dataset/all_probes.json",
        api_spec_path="dataset/pandapower_docs.json",
    )
    comp.compare(
        output_dir="comparison/",
        benchmark_path="benchmark.json",
    )
"""
 
import json
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import Counter
import torch
import numpy as np
import pandas as pd
import random
import os
import re
import time
torch.manual_seed(22)
random.seed(22)
np.random.seed(22)

from huggingface_hub import login
from backend.llm_backend import LLMBackend
from probing.probe_framework import (
    ProbeGenerator, ProbeEvaluator, Probe, EvalResult,
    load_probes, save_results, compute_l0_auc,
    compute_L0_summary, compute_L2_subtype_summary,
    build_knowledge_profile,
)

from benchmark_generator.benchmark_engine import diagnostic_checks, semantic_rewrite_checks
from intervention.proactive import ProactiveInjector

os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false') 


def _load_dotenv_once():
    """Load local API keys from .env without requiring python-dotenv."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
    ]
    seen = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv_once()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


BASELINE_A_FILENAME = "benchmark_results_condA.json"
BASELINE_FX_FILENAME = "benchmark_results_condA_FX.json"
LEGACY_BASELINE_FILENAME = "benchmark_results.json"


def _baseline_a_path(model_dir: Path) -> Path:
    """Canonical output file for the pure Round-0 condition A baseline."""
    return model_dir / BASELINE_A_FILENAME


def _baseline_fx_path(model_dir: Path) -> Path:
    """Canonical output file for the A+FX benchmark baseline."""
    return model_dir / BASELINE_FX_FILENAME


def _legacy_baseline_path(model_dir: Path) -> Path:
    """Older baseline filename, read-only compatibility fallback."""
    return model_dir / LEGACY_BASELINE_FILENAME


def _preferred_baseline_path(model_dir: Path) -> Path:
    """Prefer canonical pure-A, then A+FX, then the legacy filename."""
    pure_a = _baseline_a_path(model_dir)
    if pure_a.exists():
        return pure_a
    canonical = _baseline_fx_path(model_dir)
    return canonical if canonical.exists() else _legacy_baseline_path(model_dir)


def _token_value(record: Dict, primary: str, legacy: str = None):
    value = record.get(primary)
    if value is None and legacy:
        value = record.get(legacy)
    return value


def _env_float(env_key: str, default: float) -> float:
    """Read a float env override while keeping config defaults stable."""
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        print(f"  [env override] {env_key}={raw!r} is not a float, ignoring")
        return float(default)
    print(f"  [env override] {env_key.lower()} = {value:g}")
    return value


def _exact_text_tokens(backend: LLMBackend, text: str, label: str) -> int:
    """Count tokens with the active model tokenizer; fail rather than estimate."""
    if not text:
        return 0
    try:
        return int(backend.count_text_tokens(text))
    except Exception as e:
        raise RuntimeError(f"Exact token counting failed for {label}: {e}") from e


def _precount_chat_tokens(backend: LLMBackend, messages_list: List[List[Dict]], label: str):
    """Best-effort exact pre-count. GPT exact counts arrive from API usage later."""
    try:
        return [int(v) for v in backend.count_chat_prompt_tokens_batch(messages_list)]
    except Exception as e:
        print(f"    ℹ️  {label} prompt tokens will be read after generation ({e})")
        return None


def _runtime_prompt_counts(backend: LLMBackend, expected: int):
    counts = backend.last_prompt_token_counts
    if len(counts) == expected and all(v is not None for v in counts):
        return [int(v) for v in counts]
    return None
 
 
# ============================================================
# System Prompts
# ============================================================
 
SYSTEM_PROMPT_JSON = """You are a pandapower Python library expert.
Answer the question accurately based on your knowledge of pandapower Python library.
Respond ONLY in the exact JSON format requested. Do not add any explanations outside the JSON."""
 
SYSTEM_PROMPT_CODE = """You are a Power System Python code generator for the pandapower library.
Output ONLY executable Python code. Include all necessary imports."""
 
 
# ============================================================
# Probe Runner
# ============================================================
 
class ProbeRunner:
    """
    Sends probes to an LLM and collects responses.
    Inference goes through LLMBackend.generate_batch().
    """
 
    LAYER_MODE = {
        "L0": "json",
        "L1": "json",
        "L2": "json",
        "L3": "code",
    }
 
    def __init__(self, backend: LLMBackend, api_spec_path: str = None):
        self.backend = backend
        self.api_spec_path = api_spec_path
 
    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        backend_type: str = "vllm",
        base_config: Dict = None,
        api_spec_path: str = None,
    ) -> "ProbeRunner":
        patch = MODEL_PATCHES.get(model_name, {"gpus": 1, "max_seq_len": 8192})
        use_gpt = patch.get("_use_gpt", False)
        use_anthropic = patch.get("_use_anthropic", False)
        use_gemini = patch.get("_use_gemini", False)
 
        cfg = copy.deepcopy(base_config or {})
        cfg["model_name"] = model_name
        cfg["max_seq_len"] = patch.get("max_seq_len", 8192)
        cfg["required_gpus"] = patch.get("gpus", 1)
        cfg["quantization"] = patch.get("quantization")
        cfg["tp_size"] = patch.get("tp_size")
        cfg["pp_size"] = patch.get("pp_size")
        cfg["enable_expert_parallel"] = patch.get("enable_expert_parallel", False)
        cfg["gpt_model"] = patch.get("gpt_model", model_name)
        cfg["gpt_base_url"] = patch.get("gpt_base_url")
        if patch.get("gpt_api_key_name"):
            cfg["gpt_api_key"] = cfg.get(patch["gpt_api_key_name"], "")
        cfg["gpt_concurrency"] = patch.get("gpt_concurrency", 40)
        cfg["anthropic_model"] = patch.get("anthropic_model", model_name)
        cfg["anthropic_version"] = patch.get("anthropic_version", "2023-06-01")
        cfg["gemini_model"] = patch.get("gemini_model", model_name)

        if use_gpt:
            resolved_backend = "gpt"
        elif use_anthropic:
            resolved_backend = "anthropic"
        elif use_gemini:
            resolved_backend = "gemini"
        else:
            resolved_backend = backend_type
        backend = LLMBackend(resolved_backend, cfg)
        return cls(backend, api_spec_path=api_spec_path)
 
    def cleanup(self):
        self.backend.cleanup()
 
    # ----------------------------------------------------------
    # Main entry
    # ----------------------------------------------------------
 
    def run_all(self, probes_path, output_dir, layers=None,
                max_per_layer=None, batch_size=8, resume=True):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
 
        with open(probes_path) as f:
            raw = json.load(f)
        all_probes = {k: [_dict_to_probe(p) for p in v] for k, v in raw.items()}
        target_layers = layers or list(all_probes.keys())
 
        # Step 1: determine which layers need inference
        pending_layers = []
        for layer_key in target_layers:
            result_file = output_dir / f"results_{layer_key}.json"
            if resume and result_file.exists():
                print(f"  ✅ Skipping '{layer_key}' (already done)")
            else:
                pending_layers.append(layer_key)
 
        # Step 2: load existing results
        all_results = {}
        for layer_key in target_layers:
            result_file = output_dir / f"results_{layer_key}.json"
            if result_file.exists() and layer_key not in pending_layers:
                with open(result_file) as f:
                    all_results[layer_key] = json.load(f)
 
        if not pending_layers:
            print(f"  ⏭️  All layers done, skipping model load.")
            summary = self._generate_summary(all_results)
            with open(output_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            self._print_summary(summary)
            return summary
 
        # Step 3: load model and run pending layers
        print(f"  📋 Layers to run: {pending_layers}")
        if not self.backend.is_ready():
            self.backend.setup()
 
        evaluator = ProbeEvaluator(json_path=self.api_spec_path)
        probe_temperature = _env_float(
            "PROBE_TEMPERATURE",
            self.backend.config.get("probe_temperature", 0.0),
        )
 
        for layer_key in pending_layers:
            probes = all_probes.get(layer_key, [])
            if max_per_layer:
                probes = probes[:max_per_layer]
 
            print(f"\n{'='*60}")
            print(f"🔍 Layer: {layer_key}  ({len(probes)} probes)")
 
            messages_list = [self._build_messages(p) for p in probes]
            responses = self.backend.generate_batch(
                messages_list,
                max_new_tokens=self.backend.config.get("max_seq_len", 5000),
                temperature=probe_temperature,
                batch_size=batch_size,
            )
 
            results = evaluator.evaluate_all(list(zip(probes, responses)))
            serializable = [_eval_result_to_dict(r) for r in results]
 
            result_file = output_dir / f"results_{layer_key}.json"
            with open(result_file, "w") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
 
            all_results[layer_key] = serializable
 
        # Step 4: generate summary + knowledge profile
        summary = self._generate_summary(all_results)
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        profile = build_knowledge_profile(all_results)
        with open(output_dir / "knowledge_profile.json", "w") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        print(f"  💾 Knowledge profile: {len(profile)} functions profiled")
 
        self._print_summary(summary)
        return summary
 
    # ----------------------------------------------------------
    # Prompt building
    # ----------------------------------------------------------
 
    def _build_messages(self, probe) -> List[Dict]:
        layer = probe.layer
        mode = self.LAYER_MODE.get(layer, "json")
        system = SYSTEM_PROMPT_JSON if mode == "json" else SYSTEM_PROMPT_CODE
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": probe.prompt},
        ]
 
    # ----------------------------------------------------------
    # Summary generation
    # ----------------------------------------------------------
 
    def _generate_summary(self, all_results: Dict) -> Dict:
        summary = {"timestamp": datetime.now().isoformat(), "layers": {}}
 
        for layer_key, results in all_results.items():
            if not results:
                continue
 
            score_keys = set()
            for r in results:
                score_keys.update(r.get("scores", {}).keys())
 
            layer_summary = {"count": len(results), "scores": {}}
            for key in sorted(score_keys):
                vals = [
                    r["scores"][key]
                    for r in results
                    if key in r["scores"]
                    and isinstance(r["scores"][key], (int, float))
                    and r["scores"][key] >= 0
                ]
                if vals:
                    layer_summary["scores"][key] = {
                        "mean": round(sum(vals) / len(vals), 4),
                        "min":  round(min(vals), 4),
                        "max":  round(max(vals), 4),
                    }
 
            if "L0" in layer_key:
                hallucination = sum(
                    1 for r in results
                    if r.get("details", {}).get("error_category") == "hallucination"
                )
                false_reject = sum(
                    1 for r in results
                    if r.get("details", {}).get("error_category") == "false_rejection"
                )
                fake_total = sum(
                    1 for r in results
                    if r.get("details", {}).get("question_type") == "find_fake"
                )
                real_total = sum(
                    1 for r in results
                    if r.get("details", {}).get("question_type") == "find_real"
                )
 
                hallucination_rate   = round(hallucination / fake_total, 4) if fake_total else 0.0
                false_rejection_rate = round(false_reject / real_total, 4) if real_total else 0.0
 
                layer_summary["hallucination_rate"]   = hallucination_rate
                layer_summary["false_rejection_rate"] = false_rejection_rate
                layer_summary["balanced_accuracy"] = round(
                    ((1 - false_rejection_rate) + (1 - hallucination_rate)) / 2, 4
                )
 
                auc_info = compute_l0_auc(results)
                if auc_info["n_real"] > 0:
                    layer_summary["auc"]               = auc_info["auc"]
                    layer_summary["best_threshold"]    = auc_info["best_threshold"]
                    layer_summary["best_balanced_acc"] = auc_info["best_balanced_acc"]
 
            if "L2" in layer_key:
                subtype_summary = compute_L2_subtype_summary(results)
                layer_summary["sub_type_breakdown"] = subtype_summary
 
            if "L3" in layer_key:
                success_rate = sum(
                    1 for r in results
                    if r["scores"].get("execution_success", 0) == 1.0
                ) / len(results)
                layer_summary["execution_pass_rate"] = round(success_rate, 4)
 
                api_vals = [
                    r["scores"]["api_validity_rate"]
                    for r in results
                    if isinstance(r["scores"].get("api_validity_rate"), (int, float))
                    and r["scores"]["api_validity_rate"] >= 0
                ]
                if api_vals:
                    layer_summary["avg_api_validity"] = round(
                        sum(api_vals) / len(api_vals), 4
                    )
 
                error_dist = Counter(
                    r["scores"].get("error_type")
                    for r in results
                    if r["scores"].get("error_type")
                )
                layer_summary["error_type_distribution"] = dict(error_dist.most_common())
 
                total_errors  = sum(error_dist.values())
                import_errors = (error_dist.get("ImportError", 0) +
                                 error_dist.get("ModuleNotFoundError", 0))
                if total_errors > 0 and import_errors / total_errors > 0.5:
                    layer_summary["env_warning"] = (
                        f"⚠️  {import_errors}/{total_errors} failures are ImportError — "
                        "library may not be installed."
                    )
 
            summary["layers"][layer_key] = layer_summary
 
        return summary
 
    def _print_summary(self, summary: Dict):
        print("\n" + "=" * 60)
        print("📊 Evaluation Summary")
        print("=" * 60)
 
        for layer_key, info in summary.get("layers", {}).items():
            print(f"\n  [{layer_key}]  n={info['count']}")
 
            for metric, val in info.get("scores", {}).items():
                if isinstance(val, dict):
                    print(f"    {metric:35s}: mean={val['mean']:.4f}"
                          f"  [{val['min']:.4f}, {val['max']:.4f}]")
 
            if "hallucination_rate" in info:
                print(f"    {'hallucination_rate':35s}: {info['hallucination_rate']:.4f}")
                print(f"    {'false_rejection_rate':35s}: {info['false_rejection_rate']:.4f}")
                print(f"    {'balanced_accuracy':35s}: {info['balanced_accuracy']:.4f}"
                      f"  ← bias-free")
 
            if "auc" in info:
                print(f"    {'auc':35s}: {info['auc']:.4f}")
 
            if "sub_type_breakdown" in info:
                print(f"    sub-type breakdown:")
                for sub, sub_info in info["sub_type_breakdown"].items():
                    print(f"      {sub:30s}: acc={sub_info['accuracy']:.4f}  n={sub_info['n_probes']}")
 
            if "execution_pass_rate" in info:
                print(f"    {'execution_pass_rate':35s}: {info['execution_pass_rate']:.4f}")
            if "avg_api_validity" in info:
                print(f"    {'avg_api_validity':35s}: {info['avg_api_validity']:.4f}")
            if "error_type_distribution" in info:
                print(f"    error_type_distribution   : {info['error_type_distribution']}")
            if "env_warning" in info:
                print(f"    {info['env_warning']}")
 
 
# ============================================================
# Multi-Model Comparison (with Benchmark Integration)
# ============================================================
 
 
class MultiModelComparison:
    """Compare multiple models on probes and (optionally) the benchmark.

    When ``benchmark_path`` is supplied to ``compare()``, the benchmark
    evaluation runs against each model with the same loaded backend used
    for probing, so the model is only loaded once per run.
    """
 
    def __init__(self, probes_path: str, api_spec_path: str = None):
        self.probes_path = probes_path
        self.api_spec_path = api_spec_path
        self._model_list: List[Tuple[str, ProbeRunner, bool]] = []
 
    def add_model(self, name: str, runner: ProbeRunner, use_gpt: bool = False):
        self._model_list.append((name, runner, use_gpt))
 
    @classmethod
    def from_model_names(
        cls,
        model_names: List[str],
        probes_path: str,
        backend_type: str = "vllm",
        base_config: Dict = None,
        api_spec_path: str = None,
    ) -> "MultiModelComparison":
        instance = cls(probes_path, api_spec_path=api_spec_path)
 
        for name in model_names:
            runner = ProbeRunner.from_model_name(
                model_name=name,
                backend_type=backend_type,
                base_config=base_config,
                api_spec_path=api_spec_path,
            )
            safe_name = name.replace("/", "_")
            instance.add_model(safe_name, runner)
            patch = MODEL_PATCHES.get(name, {})
            print(f"  ✅ Registered: {safe_name}  "
                  f"(backend={backend_type}, gpus={patch.get('gpus', 1)})")
 
        return instance
 
    def compare(
        self,
        output_dir: str = "comparison/",
        layers: List[str] = None,
        max_per_layer: int = None,
        batch_size: int = 8,
        # ── Benchmark parameters ──
        benchmark_path: str = None,
        benchmark_max_items: int = None,
        benchmark_sample_mode: str = "head",
        benchmark_sample_seed: int = 22,
        benchmark_sample_strata: str = "difficulty",
        benchmark_rtol: float = 1e-2,
        benchmark_atol: float = 1e-3,
        benchmark_batch_size: int = None,
        benchmark_timeout: int = 60,
        benchmark_exec_hard_timeout: int = 75,
        benchmark_max_hard_timeout_fix_attempts: int = 1,
        benchmark_max_new_tokens: int = 4096,
        benchmark_fix_max_new_tokens: int = 2048,
        benchmark_exec_workers: int = 1,
        benchmark_fix_rounds: int = 0,
        search_engine = None,
        # ── Proactive Compensation parameters ──
        # Set proactive_conditions to run proactive compensation alongside the standard benchmark.
        # Condition "A" must NOT be listed — the standard A+FX benchmark
        # (benchmark_results_condA_FX.json) provides round-0 condition A for
        # proactive tables and plain-fix A+FX for reactive/cost tables.
        # Knowledge profiles are loaded per-model from {output_dir}/{model}/knowledge_profile.json.
        # Results are saved to benchmark_results_cond{X}.json (resume-capable).
        proactive_conditions: List[str] = None,   # e.g. ["B", "C", "X"]  (never include "A")
        # Demand export — specify EITHER suite_dir+model OR the direct file path:
        proactive_demand_suite: str = None,        # e.g. "results/demand_suites/suite_4428033"
        proactive_demand_model: str = "hybrid_tfidf",  # subdirectory under suite_dir
        proactive_demand_export: str = None,       # direct path (legacy / fallback)
        proactive_docs_path: str = None,           # defaults to self.api_spec_path
        proactive_top_k: int = 10,
        proactive_token_budget_C: int = 2000,
        proactive_token_budget_X: int = 4000,
        proactive_threshold: float = 0.5,
        # ── Reactive Injection parameters ──
        # reactive_combinations: list of "P_F" strings, e.g. ["C_FX", "C_FD", "C_FDR"]
        # Each entry runs a combination of (proactive_base P) + (fix_condition F).
        # Output: benchmark_results_cond{P}_{F}.json per model.
        # "A" base = no proactive injection (standard round-0).
        # FX/FR/FD/FDR/FS/FDRS/FE determine fix-round intervention.
        reactive_combinations: List[str] = None,
        reactive_fix_rounds: int = 3,
        reactive_docs_path: str = None,
        reactive_max_funcs: int = 3,
    ) -> Dict:
        """
        Run probe evaluation + optional benchmark evaluation for all models.

        Proactive Compensation (if proactive_conditions is set):
            For each model, after standard benchmark, loads that model's
            knowledge_profile.json and runs proactive conditions B/C/X.
            Results are saved to benchmark_results_cond{X}.json — existing
            files are reused (resume), so re-running compare() is safe.
            Call rerun_proactive() to force a fresh run of specific conditions.

        Reactive Injection (if reactive_combinations is set):
            For each combination "P_F" and each model, runs round-0 with proactive
            base P (or baseline A) then fix rounds using ReactiveInjector with fix
            condition F. Results saved to benchmark_results_cond{P}_{F}.json.
            Call rerun_reactive() to force a fresh run of specific combinations.

        Args:
            output_dir: base output directory
            layers: probe layers to evaluate (None = all)
            max_per_layer: max probes per layer (None = all)
            batch_size: probe inference batch size
            benchmark_path: path to benchmark JSON (None = skip)
            benchmark_max_items: limit benchmark items (for debugging)
            benchmark_sample_mode: subset mode when benchmark_max_items is set
                ("head", "random", or "stratified")
            benchmark_sample_seed: deterministic seed for random/stratified sampling
            benchmark_sample_strata: comma-separated strata for stratified sampling
            benchmark_rtol / benchmark_atol: matching tolerances
            benchmark_batch_size: benchmark batch size (defaults to batch_size)
            benchmark_timeout: per-item execution timeout in seconds
            benchmark_exec_hard_timeout: parent-side liveness fuse for code execution
            benchmark_max_hard_timeout_fix_attempts: max fix rounds to try after
                an item has started hitting parent-side hard timeouts
            benchmark_fix_rounds: number of fix attempts for failed items (0 = no fix)
            proactive_conditions: list of conditions to run, e.g. ["B", "C", "X"]
            proactive_demand_export: path to demand predictions JSON (required if proactive_conditions set)
            proactive_docs_path: path to docs JSON (default: api_spec_path)
            proactive_top_k: top-k demand functions per item
            proactive_token_budget_C: token budget for condition C
            proactive_token_budget_X: token budget for condition X
            proactive_threshold: knowledge score threshold for tier assignment
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        bench_bs = benchmark_batch_size or batch_size
        docs_path = proactive_docs_path or self.api_spec_path

        all_probe_summaries = {}
        all_bench_summaries = {}
        all_proactive_summaries = {}   # cond → {model → summary}
        all_reactive_summaries = {}    # combination → {model → summary}

        for model_name, runner, use_gpt in self._model_list:
            print(f"\n{'#'*60}")
            print(f"# Model: {model_name}")
            print(f"{'#'*60}")

            try:
                model_dir = output_dir / model_name

                # ── Phase 1: Probe evaluation ──
                probe_summary = runner.run_all(
                    probes_path=self.probes_path,
                    output_dir=str(model_dir),
                    layers=layers,
                    max_per_layer=max_per_layer,
                    batch_size=batch_size,
                )
                all_probe_summaries[model_name] = probe_summary

                # ── Phase 2: Standard benchmark (= condition A baseline) ──
                if benchmark_path:
                    print(f"\n  {'─'*50}")
                    print(f"  📐 Running benchmark eval (standard baseline)...")
                    print(f"  {'─'*50}")

                    a_path = _baseline_a_path(model_dir)
                    a_summary = _run_benchmark_with_backend(
                        backend=runner.backend,
                        benchmark_path=benchmark_path,
                        output_path=str(a_path),
                        max_items=benchmark_max_items,
                        sample_mode=benchmark_sample_mode,
                        sample_seed=benchmark_sample_seed,
                        sample_strata=benchmark_sample_strata,
                        batch_size=bench_bs,
                        rtol=benchmark_rtol,
                        atol=benchmark_atol,
                        exec_timeout=benchmark_timeout,
                        exec_hard_timeout=benchmark_exec_hard_timeout,
                        max_hard_timeout_fix_attempts=benchmark_max_hard_timeout_fix_attempts,
                        fix_rounds=0,
                        search_engine=None,
                        max_new_tokens=benchmark_max_new_tokens,
                        fix_max_new_tokens=benchmark_fix_max_new_tokens,
                        exec_workers=benchmark_exec_workers,
                    )
                    bench_summary = a_summary

                    if benchmark_fix_rounds > 0:
                        fx_path = _baseline_fx_path(model_dir)
                        if _has_resumeable_fix_checkpoint(str(fx_path), benchmark_fix_rounds):
                            print("    ⏩ Existing A_FX fix checkpoint found — preserving and resuming")
                            seeded_fx = True
                        else:
                            seeded_fx = _seed_round0_from_baseline(str(a_path), str(fx_path))
                        if not seeded_fx:
                            print("    ⚠️  Could not seed A_FX from A — A_FX will rerun Round 0")
                        bench_summary = _run_benchmark_with_backend(
                            backend=runner.backend,
                            benchmark_path=benchmark_path,
                            output_path=str(fx_path),
                            max_items=benchmark_max_items,
                            sample_mode=benchmark_sample_mode,
                            sample_seed=benchmark_sample_seed,
                            sample_strata=benchmark_sample_strata,
                            batch_size=bench_bs,
                            rtol=benchmark_rtol,
                            atol=benchmark_atol,
                            exec_timeout=benchmark_timeout,
                            exec_hard_timeout=benchmark_exec_hard_timeout,
                            max_hard_timeout_fix_attempts=benchmark_max_hard_timeout_fix_attempts,
                            resume=seeded_fx,
                            fix_rounds=benchmark_fix_rounds,
                            search_engine=None,
                            max_new_tokens=benchmark_max_new_tokens,
                            fix_max_new_tokens=benchmark_fix_max_new_tokens,
                            exec_workers=benchmark_exec_workers,
                        )
                    all_bench_summaries[model_name] = bench_summary

                # ── Phase 3: Proactive compensation conditions ──
                if proactive_conditions and benchmark_path:
                    _bad = [c for c in proactive_conditions if c == "A"]
                    if _bad:
                        raise ValueError(
                            "Condition 'A' must not be in proactive_conditions — "
                            "the standard A+FX benchmark is generated separately and "
                            "its round-0 snapshot is used as condition A automatically."
                        )
                    profile_path = model_dir / "knowledge_profile.json"
                    if not profile_path.exists():
                        print(f"\n  ⚠️  No knowledge_profile.json for {model_name} — "
                              f"skipping proactive conditions")
                    elif not proactive_demand_suite and not proactive_demand_export:
                        print(f"\n  ⚠️  Neither proactive_demand_suite nor proactive_demand_export "
                              f"is set — skipping proactive conditions")
                    else:
                        print(f"\n  {'─'*50}")
                        print(f"  🧠 Proactive compensation: conditions {proactive_conditions}")
                        print(f"  {'─'*50}")

                        injector = ProactiveInjector.from_paths(
                            knowledge_profile_path=str(profile_path),
                            docs_path=docs_path,
                            demand_suite_dir=proactive_demand_suite,
                            demand_model=proactive_demand_model,
                            demand_export_path=proactive_demand_export,
                            top_k=proactive_top_k,
                            token_budget_C=proactive_token_budget_C,
                            token_budget_X=proactive_token_budget_X,
                            threshold=proactive_threshold,
                        )

                        for cond in proactive_conditions:
                            out_path = model_dir / f"benchmark_results_cond{cond}.json"
                            cond_summary = _run_benchmark_with_backend(
                                backend=runner.backend,
                                benchmark_path=benchmark_path,
                                output_path=str(out_path),
                                max_items=benchmark_max_items,
                                sample_mode=benchmark_sample_mode,
                                sample_seed=benchmark_sample_seed,
                                sample_strata=benchmark_sample_strata,
                                batch_size=bench_bs,
                                rtol=benchmark_rtol,
                                atol=benchmark_atol,
                                exec_timeout=benchmark_timeout,
                                exec_hard_timeout=benchmark_exec_hard_timeout,
                                max_hard_timeout_fix_attempts=benchmark_max_hard_timeout_fix_attempts,
                                fix_rounds=0,   # no fix rounds for proactive conditions
                                injector=injector,
                                proactive_condition=cond,
                                max_new_tokens=benchmark_max_new_tokens,
                                exec_workers=benchmark_exec_workers,
                            )
                            if cond not in all_proactive_summaries:
                                all_proactive_summaries[cond] = {}
                            all_proactive_summaries[cond][model_name] = cond_summary

                # ── Phase 4: Reactive injection combination conditions ──
                if reactive_combinations and benchmark_path:
                    from intervention.reactive import ReactiveInjector

                    r_docs_path = reactive_docs_path or self.api_spec_path
                    pro_docs = proactive_docs_path or self.api_spec_path
                    fix_conds_needed = {
                        comb.split("_", 1)[1] for comb in reactive_combinations
                    }
                    pro_bases_needed = {
                        comb.split("_")[0] for comb in reactive_combinations
                        if comb.split("_")[0] not in ("A", "")
                    }

                    r_injectors = {}
                    for fix_cond in fix_conds_needed:
                        if fix_cond == "FX":
                            continue
                        r_injectors[fix_cond] = ReactiveInjector.from_paths(
                            docs_path=r_docs_path, fix_condition=fix_cond,
                            max_funcs=reactive_max_funcs,
                        )

                    p_injector_for_reactive = None
                    if pro_bases_needed:
                        profile_path = model_dir / "knowledge_profile.json"
                        if profile_path.exists() and (
                            proactive_demand_suite or proactive_demand_export
                        ):
                            p_injector_for_reactive = ProactiveInjector.from_paths(
                                knowledge_profile_path=str(profile_path),
                                docs_path=pro_docs,
                                demand_suite_dir=proactive_demand_suite,
                                demand_model=proactive_demand_model,
                                demand_export_path=proactive_demand_export,
                                top_k=proactive_top_k,
                                token_budget_C=proactive_token_budget_C,
                                token_budget_X=proactive_token_budget_X,
                                threshold=proactive_threshold,
                            )
                        else:
                            print(f"\n  ⚠️  Reactive combinations with proactive base "
                                  f"require knowledge_profile.json + demand suite. Skipping "
                                  f"non-A bases.")

                    for comb in reactive_combinations:
                        parts = comb.split("_", 1)
                        if len(parts) != 2:
                            continue
                        pro_cond, fix_cond = parts[0], parts[1]
                        if pro_cond not in ("A", "") and p_injector_for_reactive is None:
                            continue

                        p_inj = p_injector_for_reactive if pro_cond not in ("A", "") else None
                        r_inj = r_injectors.get(fix_cond) if fix_cond != "FX" else None

                        out_path = model_dir / f"benchmark_results_cond{pro_cond}_{fix_cond}.json"
                        print(f"\n  {'─'*50}")
                        print(f"  ⚡ Reactive combination: {comb}  →  {out_path.name}")
                        print(f"  {'─'*50}")

                        # Seed Round 0 from base/FX condition if output doesn't exist yet
                        if not out_path.exists():
                            if not _seed_reactive_round0(model_dir, pro_cond, fix_cond, out_path):
                                print("    ⚠️  No reusable Round 0 file found — "
                                      "Round 0 will be run from scratch")

                        comb_summary = _run_benchmark_with_backend(
                            backend=runner.backend,
                            benchmark_path=benchmark_path,
                            output_path=str(out_path),
                            max_items=benchmark_max_items,
                            sample_mode=benchmark_sample_mode,
                            sample_seed=benchmark_sample_seed,
                            sample_strata=benchmark_sample_strata,
                            batch_size=bench_bs,
                            rtol=benchmark_rtol,
                            atol=benchmark_atol,
                            exec_timeout=benchmark_timeout,
                            exec_hard_timeout=benchmark_exec_hard_timeout,
                            max_hard_timeout_fix_attempts=benchmark_max_hard_timeout_fix_attempts,
                            resume=True,
                            fix_rounds=reactive_fix_rounds,
                            injector=p_inj,
                            proactive_condition=pro_cond if pro_cond not in ("A", "") else None,
                            reactive_injector=r_inj,
                            max_new_tokens=benchmark_max_new_tokens,
                            fix_max_new_tokens=benchmark_fix_max_new_tokens,
                            exec_workers=benchmark_exec_workers,
                        )
                        if comb not in all_reactive_summaries:
                            all_reactive_summaries[comb] = {}
                        all_reactive_summaries[comb][model_name] = comb_summary

            except Exception as e:
                print(f"\n  ❌ Model '{model_name}' failed, skipping.")
                print(f"     {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()

            finally:
                print(f"  🧹 Releasing GPU memory for {model_name}...")
                runner.cleanup()

        # ── Build unified comparison table ──
        comparison = self._build_comparison_table(
            all_probe_summaries, all_bench_summaries
        )
        comp_path = output_dir / "comparison_table.json"
        with open(comp_path, "w") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        self._print_comparison_table(comparison)
        print(f"\n💾 Comparison table saved to {comp_path}")

        # ── Standard benchmark comparison ──
        if all_bench_summaries:
            bench_comp = self._build_bench_comparison(all_bench_summaries)
            bench_comp_path = output_dir / "benchmark_comparison.json"
            with open(bench_comp_path, "w") as f:
                json.dump(bench_comp, f, indent=2, ensure_ascii=False)
            self._print_bench_comparison(bench_comp)

        # ── Proactive compensation comparison ──
        if all_proactive_summaries:
            self._print_proactive_comparison(
                all_bench_summaries, all_proactive_summaries, output_dir
            )

        # ── Reactive injection comparison ──
        if all_reactive_summaries:
            print(f"\n{'═'*60}")
            print("⚡ Reactive Injection Results")
            print(f"{'═'*60}")
            for comb, model_results in sorted(all_reactive_summaries.items()):
                print(f"\n  Combination: {comb}")
                for m, s in sorted(model_results.items()):
                    acc = s.get("result_accuracy", s.get("accuracy", "?"))
                    exec_r = s.get("exec_rate", "?")
                    if isinstance(acc, float) and isinstance(exec_r, float):
                        print(f"    {m:50s}  acc={acc:.3f}  exec={exec_r:.3f}")
                    else:
                        print(f"    {m:50s}  acc={acc}  exec={exec_r}")
            reactive_path = output_dir / "reactive_comparison.json"
            with open(reactive_path, "w") as f:
                json.dump(all_reactive_summaries, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Reactive comparison saved to {reactive_path}")

        return comparison
 
    def rerun(
        self,
        layers: List[str],
        output_dir: str = "comparison/",
        max_per_layer: int = None,
        batch_size: int = 8,
    ):
        """Re-run specific probe layers (ignores existing results)."""
        output_dir = Path(output_dir)
        for model_name, runner, use_gpt in self._model_list:
            print(f"\n  🔁 Re-running {layers} for: {model_name}")
            try:
                runner.run_all(
                    probes_path=self.probes_path,
                    output_dir=str(output_dir / model_name),
                    layers=layers,
                    resume=False,
                    max_per_layer=max_per_layer,
                    batch_size=batch_size,
                )
            except Exception as e:
                print(f"  ❌ {model_name} failed: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
            finally:
                runner.cleanup()
 
    def rerun_benchmark(
        self,
        benchmark_path: str,
        output_dir: str = "comparison/",
        max_items: int = None,
        sample_mode: str = "head",
        sample_seed: int = 22,
        sample_strata: str = "difficulty",
        batch_size: int = 8,
        rtol: float = 1e-2,
        atol: float = 1e-3,
        exec_timeout: int = None,
        exec_hard_timeout: int = None,
        max_hard_timeout_fix_attempts: int = None,
        models: List[str] = None,
        fix_rounds: int = 0,
        search_engine = None,
        max_new_tokens: int = None,
        fix_max_new_tokens: int = None,
        exec_workers: int = None,
    ):
        """
        Force re-run standard benchmark evaluation (ignores existing results).

        Probe results are untouched. benchmark_results_condA.json is regenerated first;
        if fix_rounds > 0, benchmark_results_condA_FX.json is seeded from it and fixed.
        For proactive compensation conditions, use rerun_proactive() instead.

        Args:
            benchmark_path: path to benchmark.json
            output_dir: base output directory (same as compare())
            max_items: limit items (for debugging)
            sample_mode / sample_seed / sample_strata: smoke-test sampling controls
            batch_size: inference batch size
            rtol / atol: matching tolerances
            exec_timeout: per-item timeout in seconds
            exec_hard_timeout: parent-side liveness fuse for stuck executions
            max_hard_timeout_fix_attempts: max fix rounds after hard timeout
            models: model names to re-run (None = all); accepts "Qwen/..." or "Qwen_..." format
            fix_rounds: number of fix attempts for failed items (0 = no fix)

        Usage:
            comp.rerun_benchmark("benchmark.json", output_dir="comparison/")
        """
        exec_timeout      = exec_timeout      or CONFIG.get("benchmark_exec_timeout", 60)
        exec_hard_timeout = exec_hard_timeout if exec_hard_timeout is not None else CONFIG.get("benchmark_exec_hard_timeout", 75)
        max_hard_timeout_fix_attempts = (
            max_hard_timeout_fix_attempts
            if max_hard_timeout_fix_attempts is not None
            else CONFIG.get("benchmark_max_hard_timeout_fix_attempts", 1)
        )
        max_new_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else CONFIG.get("benchmark_max_new_tokens", 4096)
        )
        fix_max_new_tokens = (
            fix_max_new_tokens
            if fix_max_new_tokens is not None
            else CONFIG.get("benchmark_fix_max_new_tokens", 2048)
        )
        exec_workers = (
            exec_workers
            if exec_workers is not None
            else CONFIG.get("benchmark_exec_workers", 1)
        )

        output_dir = Path(output_dir)
        all_bench_summaries = {}

        if models:
            models = [m.replace("/", "_") for m in models]

        for model_name, runner, use_gpt in self._model_list:
            if models and model_name not in models:
                print(f"  ⏭️  Skipping {model_name} (not in models list)")
                continue

            print(f"\n{'#'*60}")
            print(f"# Benchmark rerun: {model_name}")
            print(f"{'#'*60}")

            try:
                model_dir = output_dir / model_name
                model_dir.mkdir(parents=True, exist_ok=True)

                if not runner.backend.is_ready():
                    runner.backend.setup()

                a_path = _baseline_a_path(model_dir)
                a_summary = _run_benchmark_with_backend(
                    backend=runner.backend,
                    benchmark_path=benchmark_path,
                    output_path=str(a_path),
                    max_items=max_items,
                    sample_mode=sample_mode,
                    sample_seed=sample_seed,
                    sample_strata=sample_strata,
                    batch_size=batch_size,
                    rtol=rtol,
                    atol=atol,
                    exec_timeout=exec_timeout,
                    exec_hard_timeout=exec_hard_timeout,
                    max_hard_timeout_fix_attempts=max_hard_timeout_fix_attempts,
                    resume=False,
                    fix_rounds=0,
                    search_engine=None,
                    max_new_tokens=max_new_tokens,
                    fix_max_new_tokens=fix_max_new_tokens,
                    exec_workers=exec_workers,
                )
                bench_summary = a_summary

                if fix_rounds > 0:
                    fx_path = _baseline_fx_path(model_dir)
                    if _has_resumeable_fix_checkpoint(str(fx_path), fix_rounds):
                        print("    ⏩ Existing A_FX fix checkpoint found — preserving and resuming")
                        resume_fx = True
                    elif not _seed_round0_from_baseline(str(a_path), str(fx_path)):
                        print("    ⚠️  Could not seed A_FX from A — A_FX will rerun Round 0")
                        resume_fx = False
                    else:
                        resume_fx = True
                    bench_summary = _run_benchmark_with_backend(
                        backend=runner.backend,
                        benchmark_path=benchmark_path,
                        output_path=str(fx_path),
                        max_items=max_items,
                        sample_mode=sample_mode,
                        sample_seed=sample_seed,
                        sample_strata=sample_strata,
                        batch_size=batch_size,
                        rtol=rtol,
                        atol=atol,
                        exec_timeout=exec_timeout,
                        exec_hard_timeout=exec_hard_timeout,
                        max_hard_timeout_fix_attempts=max_hard_timeout_fix_attempts,
                        resume=resume_fx,
                        fix_rounds=fix_rounds,
                        search_engine=None,
                        max_new_tokens=max_new_tokens,
                        fix_max_new_tokens=fix_max_new_tokens,
                        exec_workers=exec_workers,
                    )
                all_bench_summaries[model_name] = bench_summary

            except Exception as e:
                print(f"  ❌ {model_name} failed: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
            finally:
                print(f"  🧹 Releasing GPU memory for {model_name}...")
                runner.cleanup()

        if all_bench_summaries:
            bench_comp = self._build_bench_comparison(all_bench_summaries)
            bench_comp_path = output_dir / "benchmark_comparison.json"
            if models and bench_comp_path.exists():
                try:
                    with open(bench_comp_path) as f:
                        existing = json.load(f)
                    existing.update(bench_comp)
                    bench_comp = existing
                except Exception:
                    pass
            with open(bench_comp_path, "w") as f:
                json.dump(bench_comp, f, indent=2, ensure_ascii=False)
            self._print_bench_comparison(bench_comp)
            print(f"\n  💾 Updated {bench_comp_path}")

        return all_bench_summaries

    def resume_benchmark_fix(
        self,
        benchmark_path: str,
        output_dir: str = "comparison/",
        max_items: int = None,
        sample_mode: str = "head",
        sample_seed: int = 22,
        sample_strata: str = "difficulty",
        batch_size: int = 8,
        rtol: float = 1e-2,
        atol: float = 1e-3,
        exec_timeout: int = None,
        exec_hard_timeout: int = None,
        max_hard_timeout_fix_attempts: int = None,
        models: List[str] = None,
        fix_rounds: int = 3,
        max_new_tokens: int = None,
        fix_max_new_tokens: int = None,
        exec_workers: int = None,
    ):
        """
        Resume the standard A_FX benchmark fix checkpoint without re-running A.

        This is intended for interrupted jobs where benchmark_results_condA_FX.json
        already contains Round 0 and one or more completed fix rounds.
        """
        exec_timeout      = exec_timeout      or CONFIG.get("benchmark_exec_timeout", 60)
        exec_hard_timeout = exec_hard_timeout if exec_hard_timeout is not None else CONFIG.get("benchmark_exec_hard_timeout", 75)
        max_hard_timeout_fix_attempts = (
            max_hard_timeout_fix_attempts
            if max_hard_timeout_fix_attempts is not None
            else CONFIG.get("benchmark_max_hard_timeout_fix_attempts", 1)
        )
        max_new_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else CONFIG.get("benchmark_max_new_tokens", 4096)
        )
        fix_max_new_tokens = (
            fix_max_new_tokens
            if fix_max_new_tokens is not None
            else CONFIG.get("benchmark_fix_max_new_tokens", 2048)
        )
        exec_workers = (
            exec_workers
            if exec_workers is not None
            else CONFIG.get("benchmark_exec_workers", 1)
        )

        output_dir = Path(output_dir)
        all_bench_summaries = {}

        if models:
            models = [m.replace("/", "_") for m in models]

        for model_name, runner, use_gpt in self._model_list:
            if models and model_name not in models:
                print(f"  ⏭️  Skipping {model_name} (not in models list)")
                continue

            print(f"\n{'#'*60}")
            print(f"# Benchmark fix resume: {model_name}")
            print(f"{'#'*60}")

            try:
                model_dir = output_dir / model_name
                fx_path = _baseline_fx_path(model_dir)
                if not fx_path.exists():
                    print(f"    ⚠️  No A_FX checkpoint found at {fx_path}; skipping")
                    continue

                if not runner.backend.is_ready():
                    runner.backend.setup()

                bench_summary = _run_benchmark_with_backend(
                    backend=runner.backend,
                    benchmark_path=benchmark_path,
                    output_path=str(fx_path),
                    max_items=max_items,
                    sample_mode=sample_mode,
                    sample_seed=sample_seed,
                    sample_strata=sample_strata,
                    batch_size=batch_size,
                    rtol=rtol,
                    atol=atol,
                    exec_timeout=exec_timeout,
                    exec_hard_timeout=exec_hard_timeout,
                    max_hard_timeout_fix_attempts=max_hard_timeout_fix_attempts,
                    resume=True,
                    fix_rounds=fix_rounds,
                    search_engine=None,
                    max_new_tokens=max_new_tokens,
                    fix_max_new_tokens=fix_max_new_tokens,
                    exec_workers=exec_workers,
                )
                all_bench_summaries[model_name] = bench_summary

            except Exception as e:
                print(f"  ❌ {model_name} failed: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
            finally:
                print(f"  🧹 Releasing GPU memory for {model_name}...")
                runner.cleanup()

        if all_bench_summaries:
            bench_comp = self._build_bench_comparison(all_bench_summaries)
            bench_comp_path = output_dir / "benchmark_comparison.json"
            if models and bench_comp_path.exists():
                try:
                    with open(bench_comp_path) as f:
                        existing = json.load(f)
                    existing.update(bench_comp)
                    bench_comp = existing
                except Exception:
                    pass
            with open(bench_comp_path, "w") as f:
                json.dump(bench_comp, f, indent=2, ensure_ascii=False)
            self._print_bench_comparison(bench_comp)
            print(f"\n  💾 Updated {bench_comp_path}")

        return all_bench_summaries

    def rerun_proactive(
        self,
        benchmark_path: str,
        proactive_demand_suite: str = None,
        proactive_demand_model: str = "hybrid_tfidf",
        proactive_demand_export: str = None,   # legacy / direct path fallback
        conditions: List[str] = None,
        output_dir: str = "comparison/",
        max_items: int = None,
        sample_mode: str = "head",
        sample_seed: int = 22,
        sample_strata: str = "difficulty",
        batch_size: int = 8,
        rtol: float = 1e-2,
        atol: float = 1e-3,
        exec_timeout: int = None,
        exec_hard_timeout: int = None,
        max_hard_timeout_fix_attempts: int = None,
        models: List[str] = None,
        proactive_docs_path: str = None,
        proactive_top_k: int = 10,
        proactive_token_budget_C: int = 2000,
        proactive_token_budget_X: int = 4000,
        proactive_token_budget_RsemB: int = 800,
        proactive_threshold: float = 0.5,
        risk_weights: Dict[str, float] = None,
        bm25_docs_path: str = None,
        sbert_docs_path: str = None,
        risk_uniform: bool = False,
        recal_name_layer_floor: float = 0.0,
        profile_source_dir: str = None,
        max_new_tokens: int = None,
        exec_workers: int = None,
    ):
        """
        Force re-run proactive compensation conditions (ignores existing results).

        Loads each model's knowledge_profile.json from {output_dir}/{model}/
        to build the per-model injector. Use this to re-run specific conditions
        without redoing probes or the standard benchmark.

        The demand export is model-agnostic w.r.t. target evaluation models —
        one file serves all models. Per-model knowledge differences are handled
        by knowledge_profile.json loaded from probe results.

        Args:
            benchmark_path: path to benchmark.json
            proactive_demand_suite: suite directory (e.g. "results/demand_suites/suite_4428033")
            proactive_demand_model: demand model subdir (default: "hybrid_tfidf")
            proactive_demand_export: direct path to candidates JSON (legacy fallback)
            conditions: list of conditions to re-run, e.g. ["C", "X"]
                        (default: ["B", "C", "X"]); never include "A"
            output_dir: base output directory (same as compare())
            max_items: limit items (for debugging)
            sample_mode / sample_seed / sample_strata: smoke-test sampling controls
            batch_size: inference batch size
            rtol / atol: matching tolerances
            exec_timeout: per-item timeout
            exec_hard_timeout: parent-side liveness fuse for stuck executions
            max_hard_timeout_fix_attempts: max fix rounds after hard timeout
            models: model names to re-run (None = all)
            proactive_docs_path: path to docs JSON (default: self.api_spec_path)
            proactive_top_k / token_budget_C / token_budget_X / threshold: injector params

        Usage:
            # Re-run conditions C and X for one model:
            comp.rerun_proactive(
                benchmark_path="benchmark.json",
                proactive_demand_suite="results/demand_suites/suite_4428033",
                conditions=["C", "X"],
                models=["Qwen/Qwen2.5-Coder-7B-Instruct"],
            )
        """
        exec_timeout      = exec_timeout      or CONFIG.get("benchmark_exec_timeout", 60)
        exec_hard_timeout = exec_hard_timeout if exec_hard_timeout is not None else CONFIG.get("benchmark_exec_hard_timeout", 75)
        max_hard_timeout_fix_attempts = (
            max_hard_timeout_fix_attempts
            if max_hard_timeout_fix_attempts is not None
            else CONFIG.get("benchmark_max_hard_timeout_fix_attempts", 1)
        )
        max_new_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else CONFIG.get("benchmark_max_new_tokens", 4096)
        )
        exec_workers = (
            exec_workers
            if exec_workers is not None
            else CONFIG.get("benchmark_exec_workers", 1)
        )

        if conditions is None:
            conditions = ["B", "C", "X"]
        if "A" in conditions:
            raise ValueError(
                "Condition 'A' must not be in conditions — "
                "the A+FX benchmark file provides condition A via its round-0 snapshot."
            )

        output_dir = Path(output_dir)
        profile_base_dir = Path(profile_source_dir) if profile_source_dir else output_dir
        docs_path = proactive_docs_path or self.api_spec_path

        if models:
            models = [m.replace("/", "_") for m in models]

        all_proactive_summaries = {}

        for model_name, runner, use_gpt in self._model_list:
            if models and model_name not in models:
                print(f"  ⏭️  Skipping {model_name} (not in models list)")
                continue

            print(f"\n{'#'*60}")
            print(f"# Proactive rerun: {model_name}  conditions={conditions}")
            print(f"{'#'*60}")

            model_dir = output_dir / model_name
            profile_path = profile_base_dir / model_name / "knowledge_profile.json"

            if not profile_path.exists():
                print(f"  ⚠️  No knowledge_profile.json found at {profile_path} — skipping")
                continue

            try:
                if not runner.backend.is_ready():
                    runner.backend.setup()

                injector = ProactiveInjector.from_paths(
                    knowledge_profile_path=str(profile_path),
                    docs_path=docs_path,
                    demand_suite_dir=proactive_demand_suite,
                    demand_model=proactive_demand_model,
                    demand_export_path=proactive_demand_export,
                    top_k=proactive_top_k,
                    token_budget_C=proactive_token_budget_C,
                    token_budget_X=proactive_token_budget_X,
                    token_budget_RsemB=proactive_token_budget_RsemB,
                    threshold=proactive_threshold,
                    risk_weights=risk_weights,
                    bm25_docs_path=bm25_docs_path,
                    sbert_docs_path=sbert_docs_path,
                    risk_uniform=risk_uniform,
                    recal_name_layer_floor=recal_name_layer_floor,
                )

                for cond in conditions:
                    out_path = model_dir / f"benchmark_results_cond{cond}.json"
                    cond_summary = _run_benchmark_with_backend(
                        backend=runner.backend,
                        benchmark_path=benchmark_path,
                        output_path=str(out_path),
                        max_items=max_items,
                        sample_mode=sample_mode,
                        sample_seed=sample_seed,
                        sample_strata=sample_strata,
                        batch_size=batch_size,
                        rtol=rtol,
                        atol=atol,
                        exec_timeout=exec_timeout,
                        exec_hard_timeout=exec_hard_timeout,
                        max_hard_timeout_fix_attempts=max_hard_timeout_fix_attempts,
                        resume=False,
                        fix_rounds=0,
                        injector=injector,
                        proactive_condition=cond,
                        max_new_tokens=max_new_tokens,
                        exec_workers=exec_workers,
                        # fix_max_new_tokens not needed: fix_rounds=0
                    )
                    if cond not in all_proactive_summaries:
                        all_proactive_summaries[cond] = {}
                    all_proactive_summaries[cond][model_name] = cond_summary

            except Exception as e:
                print(f"  ❌ {model_name} failed: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
            finally:
                print(f"  🧹 Releasing GPU memory for {model_name}...")
                runner.cleanup()

        # Print comparison using standard benchmark as condition A baseline
        if all_proactive_summaries:
            # Load existing standard benchmark summaries for the delta table
            all_bench_summaries = {}
            for model_name, _, _ in self._model_list:
                bench_path = _preferred_baseline_path(output_dir / model_name)
                if bench_path.exists():
                    try:
                        with open(bench_path) as f:
                            data = json.load(f)
                        all_bench_summaries[model_name] = data.get("summary", data)
                    except Exception:
                        pass

            self._print_proactive_comparison(
                all_bench_summaries, all_proactive_summaries, output_dir
            )

        return all_proactive_summaries

    def rerun_reactive(
        self,
        benchmark_path: str,
        combinations: List[str] = None,
        output_dir: str = "comparison/",
        fix_rounds: int = 3,
        max_items: int = None,
        sample_mode: str = "head",
        sample_seed: int = 22,
        sample_strata: str = "difficulty",
        batch_size: int = 8,
        rtol: float = 1e-2,
        atol: float = 1e-3,
        exec_timeout: int = None,
        exec_hard_timeout: int = None,
        max_hard_timeout_fix_attempts: int = None,
        models: List[str] = None,
        reactive_docs_path: str = None,
        reactive_max_funcs: int = 3,
        # Proactive params (needed when combinations include C or X bases)
        proactive_demand_suite: str = None,
        proactive_demand_model: str = "hybrid_tfidf",
        proactive_demand_export: str = None,
        proactive_docs_path: str = None,
        proactive_top_k: int = 10,
        proactive_token_budget_C: int = 2000,
        proactive_token_budget_X: int = 4000,
        proactive_threshold: float = 0.5,
        risk_weights: Dict[str, float] = None,
        bm25_docs_path: str = None,
        risk_uniform: bool = False,
        profile_source_dir: str = None,
        max_new_tokens: int = None,
        fix_max_new_tokens: int = None,
        exec_workers: int = None,
    ):
        """
        Run reactive knowledge injection experiments (fix-round doc augmentation).

        Each combination string encodes a (proactive_base, fix_condition) pair:
          "A_FD"  – baseline round-0 + reactive FD fix   → benchmark_results_condA_FD.json
          "C_FD"  – proactive-C round-0 + reactive FD fix→ benchmark_results_condC_FD.json
          "C_FDR" – proactive-C round-0 + fix-demand routed fix
          "C_FDRS"– proactive-C round-0 + routed fix + semantic value route
          "X_FD"  – proactive-X round-0 + reactive FD fix→ benchmark_results_condX_FD.json
          "C_FX"  – proactive-C round-0 + plain fix      → benchmark_results_condC_FX.json
          "A_FX"  – baseline round-0 + plain fix (no reactive injection)
          Any "P_F" combination is supported.

        NOTE: For combinations using proactive bases (C, X), a proactive_demand_suite
        or proactive_demand_export must be provided AND knowledge_profile.json must exist
        for each model (populated by probe evaluation).

        Args:
            benchmark_path:   path to benchmark.json
            combinations:     list of "P_F" strings (default: C_FX,C_FD,C_FDR)
            output_dir:       base output directory (same as compare())
            fix_rounds:       number of fix rounds per combination (default: 3)
            max_items:        limit items for debugging
            sample_mode / sample_seed / sample_strata: smoke-test sampling controls
            batch_size:       inference batch size
            models:           model names to run (None = all)
            exec_hard_timeout: parent-side liveness fuse for stuck executions
            max_hard_timeout_fix_attempts: max fix rounds after hard timeout
            reactive_docs_path: path to docs JSON for ReactiveInjector (default: api_spec_path)
            reactive_max_funcs: maximum implicated functions to document per fix prompt
            proactive_*:      ProactiveInjector params (required if any combination uses C/X base)

        Usage:
            comp.rerun_reactive(
                benchmark_path="benchmark.json",
                combinations=["C_FX", "C_FD", "C_FDR"],
                proactive_demand_suite="results/demand_suites/suite_4428033",
            )
        """
        exec_timeout       = exec_timeout      or CONFIG.get("benchmark_exec_timeout", 60)
        exec_hard_timeout = (
            exec_hard_timeout
            if exec_hard_timeout is not None
            else CONFIG.get("benchmark_exec_hard_timeout", 75)
        )
        max_hard_timeout_fix_attempts = (
            max_hard_timeout_fix_attempts
            if max_hard_timeout_fix_attempts is not None
            else CONFIG.get("benchmark_max_hard_timeout_fix_attempts", 1)
        )
        max_new_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else CONFIG.get("benchmark_max_new_tokens", 4096)
        )
        fix_max_new_tokens = (
            fix_max_new_tokens
            if fix_max_new_tokens is not None
            else CONFIG.get("benchmark_fix_max_new_tokens", 2048)
        )
        exec_workers = (
            exec_workers
            if exec_workers is not None
            else CONFIG.get("benchmark_exec_workers", 1)
        )

        from intervention.reactive import ReactiveInjector

        if combinations is None:
            combinations = ["C_FX", "C_FD", "C_FDR"]

        output_dir = Path(output_dir)
        profile_base_dir = Path(profile_source_dir) if profile_source_dir else output_dir
        docs_path = reactive_docs_path or self.api_spec_path
        pro_docs_path = proactive_docs_path or self.api_spec_path

        if models:
            models = [m.replace("/", "_") for m in models]

        # Determine which proactive bases are needed
        proactive_bases_needed = {
            comb.split("_")[0] for comb in combinations
            if comb.split("_")[0] not in ("A", "")
        }
        fix_conditions_needed = {
            comb.split("_", 1)[1] for comb in combinations
        }

        all_reactive_summaries = {}  # combination → {model → summary}

        for model_name, runner, use_gpt in self._model_list:
            if models and model_name not in models:
                print(f"  ⏭️  Skipping {model_name} (not in models list)")
                continue

            print(f"\n{'#'*60}")
            print(f"# Reactive rerun: {model_name}  combinations={combinations}")
            print(f"{'#'*60}")

            model_dir = output_dir / model_name

            try:
                if not runner.backend.is_ready():
                    runner.backend.setup()

                # Build ReactiveInjectors per fix condition (shared across proactive bases)
                reactive_injectors = {}
                for fix_cond in fix_conditions_needed:
                    if fix_cond == "FX":
                        continue
                    reactive_injectors[fix_cond] = ReactiveInjector.from_paths(
                        docs_path=docs_path,
                        fix_condition=fix_cond,
                        max_funcs=reactive_max_funcs,
                    )

                # Build ProactiveInjector if any combination needs a proactive base
                proactive_injector = None
                if proactive_bases_needed:
                    profile_path = profile_base_dir / model_name / "knowledge_profile.json"
                    if not profile_path.exists():
                        print(f"  ⚠️  No knowledge_profile.json at {profile_path}")
                        print(f"       Combinations requiring proactive base will be skipped.")
                    elif not proactive_demand_suite and not proactive_demand_export:
                        print(f"  ⚠️  No proactive_demand_suite set — "
                              f"proactive-base combinations will be skipped.")
                    else:
                        proactive_injector = ProactiveInjector.from_paths(
                            knowledge_profile_path=str(profile_path),
                            docs_path=pro_docs_path,
                            demand_suite_dir=proactive_demand_suite,
                            demand_model=proactive_demand_model,
                            demand_export_path=proactive_demand_export,
                            top_k=proactive_top_k,
                            token_budget_C=proactive_token_budget_C,
                            token_budget_X=proactive_token_budget_X,
                            threshold=proactive_threshold,
                            risk_weights=risk_weights,
                            bm25_docs_path=bm25_docs_path,
                            risk_uniform=risk_uniform,
                        )

                # Run each combination
                for comb in combinations:
                    parts = comb.split("_", 1)
                    if len(parts) != 2:
                        print(f"  ⚠️  Invalid combination format {comb!r} — skipping")
                        continue
                    pro_cond, fix_cond = parts[0], parts[1]

                    # Check feasibility
                    if pro_cond not in ("A", "") and proactive_injector is None:
                        print(f"  ⏭️  Skipping {comb}: proactive injector not available")
                        continue

                    r_injector = reactive_injectors.get(fix_cond)
                    p_injector = proactive_injector if pro_cond not in ("A", "") else None

                    out_path = model_dir / f"benchmark_results_cond{pro_cond}_{fix_cond}.json"
                    print(f"\n  {'─'*50}")
                    print(f"  ⚡ Reactive: {comb}  →  {out_path.name}")
                    print(f"  {'─'*50}")

                    # For FX fix condition, reactive_injector = None (plain fix prompt).
                    # Re-run the fix rounds from a reusable Round 0 whenever possible;
                    # rerunning proactive Round 0 is only a fallback when no compatible
                    # seed exists.
                    actual_r_injector = r_injector if fix_cond != "FX" else None
                    seeded = _seed_reactive_round0(model_dir, pro_cond, fix_cond, out_path)
                    if not seeded:
                        print("    ⚠️  No reusable Round 0 file found — "
                              "Round 0 will be re-run from scratch")

                    cond_summary = _run_benchmark_with_backend(
                        backend=runner.backend,
                        benchmark_path=benchmark_path,
                        output_path=str(out_path),
                        max_items=max_items,
                        sample_mode=sample_mode,
                        sample_seed=sample_seed,
                        sample_strata=sample_strata,
                        batch_size=batch_size,
                        rtol=rtol,
                        atol=atol,
                        exec_timeout=exec_timeout,
                        exec_hard_timeout=exec_hard_timeout,
                        max_hard_timeout_fix_attempts=max_hard_timeout_fix_attempts,
                        resume=seeded,
                        fix_rounds=fix_rounds,
                        injector=p_injector,
                        proactive_condition=pro_cond if pro_cond not in ("A", "") else None,
                        reactive_injector=actual_r_injector,
                        max_new_tokens=max_new_tokens,
                        fix_max_new_tokens=fix_max_new_tokens,
                        exec_workers=exec_workers,
                    )
                    if comb not in all_reactive_summaries:
                        all_reactive_summaries[comb] = {}
                    all_reactive_summaries[comb][model_name] = cond_summary

            except Exception as e:
                print(f"  ❌ {model_name} failed: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
            finally:
                print(f"  🧹 Releasing GPU memory for {model_name}...")
                runner.cleanup()

        if all_reactive_summaries:
            reactive_path = output_dir / "reactive_comparison.json"
            merged = all_reactive_summaries
            if models and reactive_path.exists():
                try:
                    with open(reactive_path) as f:
                        existing = json.load(f)
                    for comb, model_results in all_reactive_summaries.items():
                        existing.setdefault(comb, {}).update(model_results)
                    merged = existing
                except Exception:
                    pass
            with open(reactive_path, "w") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            print(f"\n  💾 Updated {reactive_path}")

        return all_reactive_summaries

    # ----------------------------------------------------------
    # Comparison table (probes + benchmark merged)
    # ----------------------------------------------------------
 
    def _build_comparison_table(
        self,
        all_probe_summaries: Dict,
        all_bench_summaries: Dict,
    ) -> Dict:
        """Build merged comparison: probe metrics + benchmark metrics."""
        table = {}
 
        # Probe metrics (same as before)
        for model_name, summary in all_probe_summaries.items():
            for layer_key, layer_info in summary.get("layers", {}).items():
                if layer_key not in table:
                    table[layer_key] = {}
                model_row = {}
                for metric, val in layer_info.get("scores", {}).items():
                    if isinstance(val, dict):
                        model_row[metric] = val["mean"]
                for extra in ("hallucination_rate", "false_rejection_rate",
                              "balanced_accuracy", "auc",
                              "execution_pass_rate", "avg_api_validity"):
                    if extra in layer_info:
                        model_row[extra] = layer_info[extra]
 
                if "sub_type_breakdown" in layer_info:
                    for sub, sub_info in layer_info["sub_type_breakdown"].items():
                        model_row[f"L2_{sub}"] = sub_info["accuracy"]
 
                table[layer_key][model_name] = model_row
 
        # Benchmark metrics → new "Benchmark" section in table
        if all_bench_summaries:
            table["Benchmark"] = {}
            for model_name, bench in all_bench_summaries.items():
                row = {
                    "execution_rate": bench.get("execution_rate", 0),
                    "result_accuracy": bench.get("result_accuracy", 0),
                    "result_acc_of_exec": bench.get("result_accuracy_of_executed", 0),
                }
                for diff, ds in bench.get("per_difficulty", {}).items():
                    acc_of_exec = ds.get("accuracy_of_executed")
                    if acc_of_exec is None:
                        executed = ds.get("executed", 0)
                        acc_of_exec = ds.get("matched", 0) / executed if executed else 0
                    row[f"diff_{diff}_exec"] = ds.get("exec_rate", 0)
                    row[f"diff_{diff}_acc"] = ds.get("accuracy", 0)
                    row[f"diff_{diff}_acc_of_exec"] = acc_of_exec
                # Per-task accuracy
                for task, ts in bench.get("per_task", {}).items():
                    row[f"task_{task}"] = ts["accuracy"]
                # Per-GT-type accuracy
                for gt, gs in bench.get("per_gt_type", {}).items():
                    row[f"gt_{gt}"] = gs["accuracy"]
                table["Benchmark"][model_name] = row
 
        return table
 
    def _build_bench_comparison(self, all_bench_summaries: Dict) -> Dict:
        """Detailed benchmark-only comparison."""
        comp = {}
        for model_name, bench in all_bench_summaries.items():
            comp[model_name] = {
                "execution_rate": bench.get("execution_rate", 0),
                "result_accuracy": bench.get("result_accuracy", 0),
                "result_acc_of_exec": bench.get("result_accuracy_of_executed", 0),
                "n_total": bench.get("total", 0),
                "n_executed": bench.get("n_executed", 0),
                "n_matched": bench.get("n_matched", 0),
                "per_task": bench.get("per_task", {}),
                "per_gt_type": bench.get("per_gt_type", {}),
                "per_query_type": bench.get("per_query_type", {}),
                "per_n_modifications": bench.get("per_n_modifications", {}),
                "per_network": bench.get("per_network", {}),
                "per_difficulty": bench.get("per_difficulty", {}),
                "error_distribution": bench.get("error_distribution", {}),
                "match_failure_reasons": bench.get("match_failure_reasons", {}),
                "float_error_stats": bench.get("float_error_stats", {}),
                "semantic_rewrite_summary": bench.get("semantic_rewrite_summary", {}),
                "code_stats": bench.get("code_stats", {}),
                "round_snapshots": bench.get("round_snapshots", []),
            }
        return comp
 
    def _print_comparison_table(self, table: Dict):
        print("\n" + "=" * 80)
        print("📊 Model Comparison Table")
        print("=" * 80)
 
        def _shorten(name: str, max_len: int = 22) -> str:
            if "_" in name:
                name = name.split("_", 1)[1]
            for suffix in ("-Instruct", "-instruct", "Coder", "-Coder"):
                name = name.replace(suffix, "")
            return name[:max_len] if len(name) > max_len else name
 
        COL_W = 24
        MET_W = 32
 
        for layer_key, model_data in table.items():
            print(f"\n  [{layer_key}]")
            models      = list(model_data.keys())
            short_names = [_shorten(m) for m in models]
 
            metrics = sorted({k for m in models for k in model_data[m]})
 
            header = f"  {'Metric':<{MET_W}}" + "".join(
                f"{s:>{COL_W}}" for s in short_names
            )
            sep = "  " + "-" * (MET_W + COL_W * len(models))
            print(header)
            print(sep)
 
            for metric in metrics:
                row = f"  {metric:<{MET_W}}"
                for m in models:
                    val = model_data[m].get(metric)
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        row += f"{val:>{COL_W}.4f}"
                    else:
                        row += f"{'N/A':>{COL_W}}"
                print(row)
 
        print()
 
    def _print_bench_comparison(self, comp: Dict):
        """Print comprehensive multi-model benchmark comparison."""
        print(f"\n{'='*100}")
        print("📐 Benchmark Comparison (all models)")
        print(f"{'='*100}")
 
        models = list(comp.keys())
        if not models:
            return
 
        def _short(name, n=16):
            if "_" in name:
                name = name.split("_", 1)[1]
            for suffix in ("-Instruct", "-instruct"):
                name = name.replace(suffix, "")
            return name[:n] if len(name) > n else name
 
        COL_W = 24
        LBL_W = 30
 
        short_names = [_short(m) for m in models]
        header = f"  {'':>{LBL_W}}" + "".join(f"{s:>{COL_W}}" for s in short_names)
        sep = "  " + "─" * (LBL_W + COL_W * len(models))
 
        def _print_section(title, data_fn):
            """Print a comparison section. data_fn(model_comp) -> dict of {label: value}."""
            print(f"\n  {title}")
            print(header)
            print(sep)
            all_labels = []
            seen = set()
            for m in models:
                for lbl in data_fn(comp[m]).keys():
                    if lbl not in seen:
                        all_labels.append(lbl)
                        seen.add(lbl)
            for lbl in all_labels:
                row = f"  {lbl:>{LBL_W}}"
                for m in models:
                    v = data_fn(comp[m]).get(lbl)
                    if isinstance(v, float):
                        row += f"{v:>{COL_W}.4f}"
                    elif isinstance(v, int):
                        row += f"{v:>{COL_W}d}"
                    else:
                        row += f"{'—':>{COL_W}}"
                print(row)
 
        # ── Overall ──
        _print_section("Overall", lambda c: {
            "execution_rate": c["execution_rate"],
            "result_accuracy": c["result_accuracy"],
            "acc | executed": c["result_acc_of_exec"],
        })
 
        # ── Per GT type ──
        _print_section("Per GT Type (accuracy)", lambda c: {
            gt: gs["accuracy"]
            for gt, gs in sorted(c.get("per_gt_type", {}).items())
        })
 
        # ── Per task (exec% / acc%) ──
        ref_tasks = comp[models[0]].get("per_task", {})
        sorted_tasks = sorted(ref_tasks.keys(), key=lambda t: -ref_tasks[t].get("total", 0))
 
        print(f"\n  Per Task (exec% / acc%)")
        print(header)
        print(sep)
        for task in sorted_tasks:
            row = f"  {task:>{LBL_W}}"
            for m in models:
                ts = comp[m].get("per_task", {}).get(task, {})
                ex = ts.get("exec_rate", 0)
                ac = ts.get("accuracy", 0)
                n = ts.get("total", 0)
                cell = f"{ex:>3.0%}/{ac:>3.0%}({n:>3})"
                row += f"{cell:>{COL_W}}"
            print(row)
 
        # ── Per query type ──
        all_qt = sorted({q for m in models for q in comp[m].get("per_query_type", {})})
        if all_qt:
            _print_section("Per Query Type (accuracy)", lambda c: {
                q: qs["accuracy"]
                for q, qs in sorted(c.get("per_query_type", {}).items())
            })
 
        # ── Per # modifications ──
        all_nm = sorted({int(nm) for m in models for nm in comp[m].get("per_n_modifications", {})})
        if all_nm:
            _print_section("Per # Modifications (accuracy)", lambda c: {
                f"{nm} mods": ms["accuracy"]
                for nm, ms in sorted(c.get("per_n_modifications", {}).items(),
                                     key=lambda x: int(x[0]))
            })
 
        # ── Error distribution ──
        all_errs = sorted({e for m in models for e in comp[m].get("error_distribution", {})})
        if all_errs:
            _print_section("Error Distribution (count)", lambda c: {
                e: c.get("error_distribution", {}).get(e, 0)
                for e in all_errs
            })
 
        # ── Float error stats ──
        has_float = any(comp[m].get("float_error_stats") for m in models)
        if has_float:
            _print_section("Float Error (matched items)", lambda c: {
                k: v for k, v in c.get("float_error_stats", {}).items()
            })
# ── Fix-round accuracy curve ──
        has_fix = any(len(comp[m].get("round_snapshots", [])) > 1 for m in models)
        if has_fix:
            max_rounds = max(
                len(comp[m].get("round_snapshots", [])) for m in models
            )
            print(f"\n  Fix-Round Accuracy Curve")
            print(header)
            print(sep)
            for ri in range(max_rounds):
                label = f"round_{ri}" if ri > 0 else "round_0 (baseline)"
                row = f"  {label:>{LBL_W}}"
                for m in models:
                    snaps = comp[m].get("round_snapshots", [])
                    if ri < len(snaps):
                        acc = snaps[ri]["result_accuracy"]
                        n = snaps[ri]["n_matched"]
                        row += f"{acc:>{COL_W-4}.4f}({n:>3})"
                    else:
                        row += f"{'—':>{COL_W}}"
                print(row)

        print(f"\n{'='*100}")

    def _print_proactive_comparison(
        self,
        baseline_summaries: Dict,   # model → standard benchmark summary (condition A)
        proactive_summaries: Dict,  # cond → {model → summary}
        output_dir: Path,
    ):
        """
        Print and save a delta table comparing proactive conditions against baseline.

        Reads missing baseline/condition summaries from saved JSON files.
        """
        # Collect all models across all conditions.
        # Also scan output_dir on disk: per-model job launches otherwise rewrite
        # this comparison with only the last-finishing model's rows.
        all_models = set(baseline_summaries.keys())
        disk_conds: set = set()
        for cond_data in proactive_summaries.values():
            all_models.update(cond_data.keys())
        if output_dir.exists():
            for sub in output_dir.iterdir():
                if not sub.is_dir():
                    continue
                found = list(sub.glob("benchmark_results_cond*.json"))
                if found:
                    all_models.add(sub.name)
                    for fp in found:
                        c = fp.stem[len("benchmark_results_cond"):]
                        if c and c not in ("A", "A_FX"):
                            disk_conds.add(c)
        all_models = sorted(all_models)
        all_conds = ["A"] + sorted(set(proactive_summaries.keys()) | disk_conds)

        # ── Helper: load the full JSON for a model+condition ──────────────────
        _loaded: Dict = {}   # cache

        def _load_data(model, cond):
            key = (model, cond)
            if key in _loaded:
                return _loaded[key]
            if cond == "A":
                p = _preferred_baseline_path(output_dir / model)
            else:
                p = output_dir / model / f"benchmark_results_cond{cond}.json"
            if not p.exists():
                _loaded[key] = None
                return None
            with open(p) as f:
                data = json.load(f)
            _loaded[key] = data
            return data

        def _round0(model):
            """Return round_0 snapshot for condition A (no fix rounds, fair baseline)."""
            data = _load_data(model, "A")
            if data is None:
                return None
            snaps = data.get("round_snapshots") or data.get("summary", {}).get("round_snapshots")
            if snaps:
                return snaps[0]
            return data.get("summary", data)   # fallback: no fix rounds used

        def _summary(model, cond):
            """Return summary dict for a proactive condition."""
            data = _load_data(model, cond)
            if data is None:
                return None
            s = data.get("summary", data)
            return s if isinstance(s, dict) else None

        def _val(d, key, default=None):
            return d.get(key, default) if d else default

        def _short(name):
            if "_" in name:
                name = name.split("_", 1)[1]
            for s in ("-Instruct", "-instruct"):
                name = name.replace(s, "")
            return name[:16]

        def _fmt(v, d=4):
            return f"{v:.{d}f}" if v is not None else "—"

        def _fmtd(v, d=4):
            return f"{v:+.{d}f}" if v is not None else "—"

        def _delta(a, b):
            return (a - b) if (a is not None and b is not None) else None

        W = 8   # column width
        LW = 16  # label width
        conds_proactive = sorted(set(proactive_summaries.keys()) | disk_conds)   # e.g. ["B","C","X"]
        all_conds_display = ["A"] + conds_proactive
        delta_pairs = [("B","A"), ("C","A"), ("X","A"), ("C","X"), ("C","B")]
        delta_pairs = [(h,l) for h,l in delta_pairs
                       if h in all_conds_display and l in all_conds_display]

        SEP = "=" * 110

        def _print_table(title, note, value_fn, fmt_fn=_fmt, delta_fn=_fmtd, d=4):
            """Generic table printer: rows=models, cols=A/B/C/X + deltas."""
            print(f"\n{SEP}")
            print(f"  {title}")
            if note:
                print(f"  {note}")
            hdr = f"  {'Model':<{LW}}"
            for c in all_conds_display:
                hdr += f"  {'Cond'+c:>{W}}"
            for h, l in delta_pairs:
                hdr += f"  {f'Δ{h}-{l}':>{W}}"
            print(hdr)
            print("  " + "─" * (LW + (len(all_conds_display)+len(delta_pairs))*(W+2)))
            rows = []
            for model in all_models:
                vals = {c: value_fn(model, c) for c in all_conds_display}
                row = f"  {_short(model):<{LW}}"
                for c in all_conds_display:
                    row += f"  {fmt_fn(vals[c], d):>{W}}"
                for h, l in delta_pairs:
                    row += f"  {delta_fn(_delta(vals[h], vals[l]), d):>{W}}"
                print(row)
                rows.append({"model": model, **vals})
            print(SEP)
            return rows

        # ── Table 1: Execution Rate ────────────────────────────────────────────
        def exec_rate(model, cond):
            if cond == "A":
                r0 = _round0(model)
                return _val(r0, "execution_rate")
            return _val(_summary(model, cond), "execution_rate")

        _print_table(
            "1. EXECUTION RATE  (code ran without runtime error)",
            "⚠ Cond A = round_0 (no fix rounds) — same evaluation setup as B/C/X",
            exec_rate, d=3
        )

        # ── Table 2: Result Accuracy (main metric, A=round_0) ─────────────────
        def result_acc(model, cond):
            if cond == "A":
                r0 = _round0(model)
                return _val(r0, "result_accuracy")
            return _val(_summary(model, cond), "result_accuracy")

        rows_acc = _print_table(
            "2. RESULT ACCURACY  (matched / total, A = round_0 — no fix rounds)",
            "⚠ After system-prompt fix, format-bug parse failures should be eliminated",
            result_acc, d=4
        )

        # ── Table 3: Accuracy-of-Executed (isolates format/parse failures) ────
        def acc_of_exec(model, cond):
            if cond == "A":
                r0 = _round0(model)
                ne = _val(r0, "n_executed", 0)
                nm = _val(r0, "n_matched", 0)
                return nm / ne if ne else None
            s = _summary(model, cond)
            ne = _val(s, "n_executed", 0)
            nm = _val(s, "n_matched", 0)
            return nm / ne if ne else None

        _print_table(
            "3. ACCURACY-OF-EXECUTED  (matched / executed — isolates parser format failures)",
            "⚠ Low values here mean the model computed correctly but output format was wrong",
            acc_of_exec, d=4
        )

        # ── Table 4: Parse failure rate ───────────────────────────────────────
        def parse_fail_rate(model, cond):
            if cond == "A":
                r0 = _round0(model)
                ne = _val(r0, "n_executed", 0)
                # round_0 snapshot doesn't store parse_failures; approximate from error_dist
                errs = _val(r0, "error_distribution", {})
                # parse fails don't appear in error_distribution (those are exec errors)
                return None   # not available in round_0 snapshot
            s = _summary(model, cond)
            if s is None:
                return None
            rate = s.get("parse_failure_rate")
            if rate is None:
                ne = _val(s, "n_executed", 0)
                n_pf = s.get("n_parse_failures", 0)
                rate = n_pf / ne if ne else 0.0
            return rate

        print(f"\n{SEP}")
        print("  4. FORMAT PARSE FAILURE RATE  (cannot_parse_float — computed correctly, wrong output format)")
        print("  ⚠ Caused by missing 'print ONLY numeric value' instruction (fixed in SYSTEM_PROMPT)")
        print("  Note: % of *executed* items that failed due to output format, not logic errors")
        hdr4 = f"  {'Model':<{LW}}"
        for c in conds_proactive:
            hdr4 += f"  {'Cond'+c+' pfail':>{W+4}}"
        print(hdr4)
        print("  " + "─" * (LW + len(conds_proactive)*(W+6)))
        for model in all_models:
            row = f"  {_short(model):<{LW}}"
            for c in conds_proactive:
                s = _summary(model, c)
                ne = _val(s, "n_executed", 0)
                npf = _val(s, "n_parse_failures", 0) or 0
                if ne:
                    row += f"  {npf}/{ne} ({npf/ne*100:.0f}%):>{W+4}".replace(":>8", "")
                    row = row[:-len(f":>{W+4}")]   # remove format residue
                    row += f"  {f'{npf}/{ne}({npf/ne*100:.0f}%)':>{W+4}}"
                else:
                    row += f"  {'—':>{W+4}}"
            print(row)
        print(SEP)

        # ── Table 5: Token usage ──────────────────────────────────────────────
        print(f"\n{SEP}")
        print("  5. PROMPT TOKEN USAGE (approx, 4 chars/token) — cost comparison across conditions")
        print("  Note: Cond A uses standard benchmark prompt; B/C/X add function hints/docs")
        hdr5 = f"  {'Model':<{LW}}"
        hdr5 += f"  {'CondA avg_tok':>{W+3}}"
        for c in conds_proactive:
            hdr5 += f"  {f'Cond{c} avg_tok':>{W+3}}"
        hdr5 += f"  {'ratio C/A':>{W+2}}  {'ratio X/A':>{W+2}}"
        print(hdr5)
        print("  " + "─" * (LW + (len(conds_proactive)+1)*(W+5) + 2*(W+4)))
        for model in all_models:
            row = f"  {_short(model):<{LW}}"
            tok_vals = {}
            a_data = _load_data(model, "A")
            a_stats = (a_data or {}).get("summary", {}).get("round0_prompt_token_stats", {})
            if not a_stats:
                a_stats = (a_data or {}).get("summary", {}).get("prompt_token_stats", {})
            a_avg = a_stats.get("avg") if a_stats else None
            tok_vals["A"] = a_avg
            row += f"  {str(a_avg) if a_avg else '—':>{W+3}}"
            for c in conds_proactive:
                s = _summary(model, c)
                toks = _val(s, "round0_prompt_token_stats") or _val(s, "prompt_token_stats")
                avg = toks.get("avg") if toks else None
                tok_vals[c] = avg
                row += f"  {str(avg) if avg else '—':>{W+3}}"
            # Ratios
            c_tok = tok_vals.get("C"); x_tok = tok_vals.get("X")
            rc = f"{c_tok/a_avg:.1f}x" if (c_tok and a_avg) else "—"
            rx = f"{x_tok/a_avg:.1f}x" if (x_tok and a_avg) else "—"
            row += f"  {rc:>{W+2}}  {rx:>{W+2}}"
            print(row)
        print(SEP)

        # ── Table 6: Per-task execution rate ─────────────────────────────────
        # Aggregate: mean exec_rate across all models with data, per task per cond
        print(f"\n{SEP}")
        print("  6. PER-TASK EXECUTION RATE  (avg across all models with data)")
        print("  Sorted by ΔC-A descending — shows where Cond C injection helps most")

        all_tasks: set = set()
        for model in all_models:
            for cond in all_conds_display:
                if cond == "A":
                    r0 = _round0(model)
                    if r0:
                        all_tasks.update(r0.get("per_task", {}).keys())
                else:
                    s = _summary(model, cond)
                    if s:
                        all_tasks.update(s.get("per_task", {}).keys())
        all_tasks = sorted(all_tasks)

        def _task_exec(model, cond, task):
            if cond == "A":
                r0 = _round0(model)
                return _val(r0.get("per_task", {}).get(task) if r0 else None, "exec_rate")
            s = _summary(model, cond)
            return _val(s.get("per_task", {}).get(task) if s else None, "exec_rate")

        task_rows = []
        for task in all_tasks:
            avg = {}
            for c in all_conds_display:
                vals = [_task_exec(m, c, task) for m in all_models
                        if _task_exec(m, c, task) is not None]
                avg[c] = sum(vals)/len(vals) if vals else None
            dc_a = _delta(avg.get("C"), avg.get("A"))
            task_rows.append((task, avg, dc_a))
        task_rows.sort(key=lambda x: (x[2] or -99), reverse=True)

        TW = 22
        hdr6 = f"  {'Task':<{TW}}"
        for c in all_conds_display:
            hdr6 += f"  {'Cond'+c:>{W}}"
        for h, l in [("B","A"),("C","A"),("X","A"),("C","X")]:
            if h in all_conds_display:
                hdr6 += f"  {f'Δ{h}-{l}':>{W}}"
        print(hdr6)
        print("  " + "─" * (TW + (len(all_conds_display)+4)*(W+2)))
        for task, avg, _ in task_rows:
            row = f"  {task:<{TW}}"
            for c in all_conds_display:
                v = avg.get(c)
                row += f"  {_fmt(v, 3):>{W}}"
            for h, l in [("B","A"),("C","A"),("X","A"),("C","X")]:
                if h in all_conds_display:
                    row += f"  {_fmtd(_delta(avg.get(h), avg.get(l)), 3):>{W}}"
            print(row)
        print(SEP)

        # ── Save comprehensive JSON ────────────────────────────────────────────
        out = {
            "conditions": all_conds_display,
            "models": rows_acc,
            "delta_pairs": [f"{h}-{l}" for h, l in delta_pairs],
            "per_task_exec_rate": [
                {"task": t, **{c: avg.get(c) for c in all_conds_display}}
                for t, avg, _ in task_rows
            ],
        }
        p = output_dir / "proactive_comparison.json"
        with open(p, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  💾 Proactive comparison saved to {p}")


# ============================================================
# Proactive Compensation Injector
# ============================================================
# Implemented in intervention/proactive.py.  probe_runner keeps only the
# experiment orchestration path.


# ============================================================
# Benchmark runner using existing backend (KEY INTEGRATION)
# ============================================================

def _execute_code_worker(args):
    """Module-level worker for ProcessPoolExecutor — must be picklable.
    args = (cleaned_code: str, exec_timeout: int)
    Returns execute_code_safely result dict, or None if no code.
    """
    cleaned, exec_timeout = args
    if not cleaned:
        return None
    from backend.utils import execute_code_safely
    return execute_code_safely(cleaned, timeout=exec_timeout)


def _eval_one_item_worker(args):
    """Module-level wrapper around _eval_one_item for ProcessPoolExecutor.
    args = (bench_i, item, raw, rtol, atol, exec_timeout)
    """
    return _eval_one_item(*args)


def _timeout_exec_result(hard_timeout: int) -> Dict:
    return {
        "success": False,
        "error": f"Hard execution timeout after {hard_timeout} seconds",
        "output": "",
        "error_type": "TimeoutError",
        "traceback": "",
        "hard_timeout": True,
        "hard_timeout_seconds": hard_timeout,
    }


def _timeout_eval_result(args, hard_timeout: int) -> Dict:
    """Build a benchmark item result when the parent watchdog kills execution."""
    from backend.utils import clean_generated_code

    bench_i, item, raw, _rtol, _atol, _exec_timeout = args
    cleaned = clean_generated_code(raw)
    return {
        "bench_index": bench_i,
        "benchmark_original_index": item.get("_benchmark_original_index", bench_i),
        "item_id": item.get("id", str(bench_i)),
        "difficulty_level": item.get("difficulty_level", "D1_basic"),
        "task": item["scenario"]["task"],
        "network": item["scenario"]["network"],
        "query_type": item["scenario"]["query_target"]["qtype"],
        "n_modifications": len(item["scenario"].get("modifications", [])),
        "natural_language_query": item["natural_language_query"],
        "reference_code": item["reference_code"],
        "ground_truth": item["ground_truth"],
        "ground_truth_type": item["ground_truth_type"],
        "eval_criteria": item.get("eval_criteria", {}),
        "generated_code": cleaned,
        "code_lines": len(cleaned.split("\n")) if cleaned else 0,
        "raw_output": raw[:500] if raw else "",
        "executed": False,
        "error_type": "TimeoutError",
        "error_msg": f"Hard execution timeout after {hard_timeout} seconds",
        "exec_output": "",
        "match": False,
        "match_detail": None,
        "hard_timeout": True,
        "hard_timeout_seconds": hard_timeout,
    }


def _watchdog_timeout_result(func, args, hard_timeout: int):
    if func is _execute_code_worker:
        return _timeout_exec_result(hard_timeout)
    if func is _eval_one_item_worker:
        return _timeout_eval_result(args, hard_timeout)
    raise TimeoutError(f"Hard timeout after {hard_timeout} seconds")


def _watchdog_exception_result(func, args, exc: Exception):
    if func is _execute_code_worker:
        return {
            "success": False,
            "error": str(exc),
            "output": "",
            "error_type": type(exc).__name__,
            "traceback": "",
        }
    if func is _eval_one_item_worker:
        result = _timeout_eval_result(args, 0)
        result["error_type"] = type(exc).__name__
        result["error_msg"] = str(exc)
        result.pop("hard_timeout", None)
        result.pop("hard_timeout_seconds", None)
        return result
    raise exc


def _hard_timeout_fix_attempts(result: Dict) -> int:
    """Number of fix rounds for this item that hit the parent hard-timeout fuse."""
    return sum(
        1 for fc in (result.get("fix_chain") or [])
        if fc.get("hard_timeout") or fc.get("hard_timeout_seconds") is not None
    )


def _eligible_for_fix_after_timeout(result: Dict, max_attempts: int) -> bool:
    """Keep hard-timeout skipping narrow: only terminal repeated hard timeouts."""
    if max_attempts < 0:
        return True
    if result.get("error_type") != "TimeoutError":
        return True
    if not result.get("hard_timeout"):
        return True
    return _hard_timeout_fix_attempts(result) < max_attempts


def _format_fix_error(result: Dict) -> Tuple[str, str]:
    """Build a non-leaky fix-round error message and router error class."""
    match_detail = result.get("match_detail") or {}
    if result.get("executed") and not result.get("match"):
        match_error = match_detail.get("error")
        if match_error:
            return f"Code ran but wrong result: {match_error}", "default"
        if match_detail.get("parsed_value") is not None:
            return "Code ran but produced an incorrect numeric output.", "default"
        return "Code ran but produced incorrect output", "default"

    if result.get("error_msg") and result.get("error_type") != "code_extraction_failed":
        return str(result["error_msg"]), result.get("error_type", "")

    match_error = match_detail.get("error")
    if match_error:
        return f"Code ran but wrong result: {match_error}", "default"

    return "Code ran but produced incorrect output", "default"


def _execution_progress_step(total: int) -> int:
    if total <= 20:
        return 1
    return max(10, min(100, total // 20))


def _print_execution_progress(
    label: str,
    done: int,
    total: int,
    start_time: float,
    *,
    force: bool = False,
):
    if total <= 0:
        return
    step = _execution_progress_step(total)
    if not force and done not in (1, total) and done % step != 0:
        return
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    print(
        f"    ⏳ {label}: {done}/{total} ({done / total:.1%}) "
        f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
    )


def _parallel_map_with_watchdog(func, args_list, workers=1, hard_timeout=100, label="task"):
    """Run one-arg worker calls with a parent-side hard timeout.

    The inner execute_code_safely() timeout uses SIGALRM, which is not enough
    when generated code disables signals or gets stuck in long native calls.
    This watchdog kills and recreates the worker pool when any item exceeds the
    hard timeout, marks only that item as TimeoutError, and resumes the rest.
    """
    import multiprocessing as mp

    results = [None] * len(args_list)
    next_idx = 0
    n_workers = max(1, int(workers or 1))
    total = len(args_list)
    start_time = time.time()
    print(f"    ▶ {label}: executing {total} item(s) with {n_workers} worker(s)")

    while next_idx < total:
        pool = mp.Pool(processes=n_workers)
        async_results = [
            pool.apply_async(func, (args,))
            for args in args_list[next_idx:]
        ]
        timed_out = False
        for offset, async_result in enumerate(async_results):
            idx = next_idx + offset
            try:
                results[idx] = async_result.get(timeout=hard_timeout)
            except mp.TimeoutError:
                print(
                    f"    ⚠️  Hard timeout during {label}: "
                    f"item {idx + 1}/{len(args_list)} exceeded {hard_timeout}s; "
                    "restarting execution workers"
                )
                results[idx] = _watchdog_timeout_result(
                    func, args_list[idx], hard_timeout
                )
                _print_execution_progress(label, idx + 1, total, start_time, force=True)
                pool.terminate()
                timed_out = True
                next_idx = idx + 1
                break
            except Exception as exc:
                print(
                    f"    ⚠️  Worker exception during {label}: "
                    f"item {idx + 1}/{len(args_list)} {type(exc).__name__}: {exc}"
                )
                results[idx] = _watchdog_exception_result(
                    func, args_list[idx], exc
                )
            _print_execution_progress(label, idx + 1, total, start_time)
        if timed_out:
            pool.join()
            continue
        pool.close()
        pool.join()
        return results

    return results


def _parallel_map(func, args_list, workers=1, hard_timeout=None, label="task"):
    """Run func(args) for each args in args_list.
    workers=1  → serial (no subprocess overhead)
    workers>1  → ProcessPoolExecutor with automatic chunksize
    Preserves order. Module-level picklable functions only.
    """
    if not args_list:
        return []
    if hard_timeout is not None:
        return _parallel_map_with_watchdog(
            func, args_list, workers=workers,
            hard_timeout=hard_timeout, label=label,
        )
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        chunksize = max(1, len(args_list) // (workers * 4))
        results = [None] * len(args_list)
        total = len(args_list)
        start_time = time.time()
        print(f"    ▶ {label}: executing {total} item(s) with {workers} worker(s)")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(func, args): idx
                for idx, args in enumerate(args_list)
            }
            done = 0
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                done += 1
                _print_execution_progress(label, done, total, start_time)
        return results
    total = len(args_list)
    start_time = time.time()
    print(f"    ▶ {label}: executing {total} item(s) with 1 worker")
    results = []
    for idx, args in enumerate(args_list):
        results.append(func(args))
        _print_execution_progress(label, idx + 1, total, start_time)
    return results


def _benchmark_stratum_value(item: Dict, field: str) -> str:
    """Return a stable stratum value for benchmark smoke sampling."""
    field = (field or "").strip().lower()
    scenario = item.get("scenario") or {}
    if field in ("difficulty", "difficulty_level", "d"):
        return str(item.get("difficulty_level", "unknown"))
    if field in ("task", "task_family"):
        return str(scenario.get("task", "unknown"))
    if field in ("query", "query_type", "qtype"):
        return str((scenario.get("query_target") or {}).get("qtype", "unknown"))
    if field == "network":
        return str(scenario.get("network", "unknown"))
    if field in ("gt", "gt_type", "ground_truth_type"):
        return str(item.get("ground_truth_type", "unknown"))
    return str(item.get(field, "unknown"))


def _benchmark_stratum_key(item: Dict, strata: str) -> str:
    fields = [f.strip() for f in re.split(r"[,/+|]", strata or "difficulty") if f.strip()]
    if not fields:
        fields = ["difficulty"]
    return "|".join(_benchmark_stratum_value(item, f) for f in fields)


def _select_benchmark_items(
    benchmark: List[Dict],
    max_items: int = None,
    sample_mode: str = "head",
    sample_seed: int = 22,
    sample_strata: str = "difficulty",
) -> Tuple[List[Dict], Dict]:
    """
    Select benchmark items for smoke tests while preserving full-run behavior.

    Modes:
      head       : first N items (legacy behavior)
      random     : uniformly random N items
      stratified : proportional random sample by `sample_strata`

    Returned items carry `_benchmark_original_index` for traceability. Runtime
    `bench_index` remains the local sample index so fix rounds can index into
    the selected list safely.
    """
    n_total = len(benchmark)
    mode = (sample_mode or "head").strip().lower()
    if mode in ("first", "prefix"):
        mode = "head"
    if mode in ("stratified_random", "stratified-random"):
        mode = "stratified"

    if not max_items or max_items >= n_total:
        selected = list(range(n_total))
        effective_mode = "full"
    elif max_items <= 0:
        selected = []
        effective_mode = mode
    elif mode == "head":
        selected = list(range(max_items))
        effective_mode = "head"
    elif mode in ("random", "shuffle"):
        rng = random.Random(sample_seed)
        selected = sorted(rng.sample(range(n_total), max_items))
        effective_mode = "random"
    elif mode == "stratified":
        rng = random.Random(sample_seed)
        groups = {}
        for idx, item in enumerate(benchmark):
            groups.setdefault(_benchmark_stratum_key(item, sample_strata), []).append(idx)
        for indices in groups.values():
            rng.shuffle(indices)

        raw_quota = {
            key: max_items * (len(indices) / max(1, n_total))
            for key, indices in groups.items()
        }
        quota = {
            key: min(len(groups[key]), int(raw_quota[key]))
            for key in groups
        }
        remaining = max_items - sum(quota.values())
        order = sorted(
            groups,
            key=lambda k: (raw_quota[k] - quota[k], len(groups[k]), k),
            reverse=True,
        )
        while remaining > 0:
            progressed = False
            for key in order:
                if remaining <= 0:
                    break
                if quota[key] < len(groups[key]):
                    quota[key] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break

        selected = []
        for key in sorted(groups):
            selected.extend(groups[key][:quota[key]])
        selected = sorted(selected)
        effective_mode = "stratified"
    else:
        raise ValueError(
            f"Unknown BENCHMARK_SAMPLE_MODE={sample_mode!r}. "
            "Use head, random, or stratified."
        )

    items = []
    for original_index in selected:
        item = dict(benchmark[original_index])
        item["_benchmark_original_index"] = original_index
        items.append(item)

    stratum_counts = Counter(
        _benchmark_stratum_key(benchmark[i], sample_strata) for i in selected
    )
    metadata = {
        "mode": effective_mode,
        "seed": sample_seed,
        "strata": sample_strata,
        "max_items": max_items,
        "source_total": n_total,
        "selected_total": len(selected),
        "selected_indices": selected,
        "selected_ids": [benchmark[i].get("id", str(i)) for i in selected],
        "stratum_counts": dict(sorted(stratum_counts.items())),
    }
    return items, metadata


def _run_benchmark_with_backend(
    backend: LLMBackend,
    benchmark_path: str,
    output_path: str,
    max_items: int = None,
    sample_mode: str = "head",
    sample_seed: int = 22,
    sample_strata: str = "difficulty",
    batch_size: int = 8,
    rtol: float = 1e-2,
    atol: float = 1e-3,
    exec_timeout: int = 60,
    exec_hard_timeout: int = 100,
    max_hard_timeout_fix_attempts: int = 1,
    resume: bool = True,
    fix_rounds: int = 0,
    search_engine = None,
    injector: "ProactiveInjector" = None,
    proactive_condition: str = None,
    reactive_injector = None,
    max_new_tokens: int = 4096,
    fix_max_new_tokens: int = 2048,
    exec_workers: int = 1,
    generation_temperature: float = None,
    fix_temperature: float = None,
) -> Dict:
    """
    Run benchmark evaluation with optional multi-round fix.

    Flow:
        Round 0: generate code from NL query → execute → match GT
        Round 1..N (if fix_rounds > 0):
            For each failed item: build fix prompt (error + optional doc injection)
            → regenerate → re-execute → re-match GT
            Track per-round cumulative accuracy.

    Args:
        backend: already-loaded LLMBackend
        sample_mode: benchmark subset mode when max_items is set (head/random/stratified)
        sample_seed: deterministic seed for random or stratified smoke sampling
        sample_strata: comma-separated strata for stratified mode, e.g. difficulty or difficulty,task
        exec_timeout: normal in-code timeout passed into execute_code_safely()
        exec_hard_timeout: parent-process liveness fuse. This should be much
            larger than exec_timeout; it exists to keep a stuck generated code
            sample from hanging the whole job.
        max_hard_timeout_fix_attempts: after an item has hard-timeout failures
            in this many fix rounds, skip further fix prompts for that item.
        fix_rounds: number of fix attempts for failed items (0 = no fix)
        resume: if True, skip completed items; if False, redo all
        injector: ProactiveInjector instance for proactive compensation prompts
        proactive_condition: one of "A"/"B"/"C"/"X" (None = standard baseline)
            When set, injector must also be provided (except "A" which falls
            back to standard build_benchmark_prompt automatically).
        reactive_injector: ReactiveInjector instance for error-targeted doc injection
            in fix rounds.
            Fix condition (FX/FR/FD/FDR/FS/FDRS/FE) is baked into the ReactiveInjector at creation.
    """
    from backend.utils import (
        match_ground_truth, parse_error, build_fix_prompt,
        build_benchmark_prompt, clean_generated_code,
        render_benchmark_task_text,
    )
    import gc
    import os
    import time

    if exec_hard_timeout is not None and exec_hard_timeout <= 0:
        exec_hard_timeout = None
    if generation_temperature is None:
        generation_temperature = CONFIG.get("benchmark_generation_temperature", 0.0)
    if fix_temperature is None:
        fix_temperature = CONFIG.get("benchmark_fix_temperature", 0.0)
    generation_temperature = _env_float(
        "BENCHMARK_GENERATION_TEMPERATURE",
        generation_temperature,
    )
    fix_temperature = _env_float(
        "BENCHMARK_FIX_TEMPERATURE",
        fix_temperature,
    )
 
    with open(benchmark_path) as f:
        full_benchmark = json.load(f)
    benchmark, sample_metadata = _select_benchmark_items(
        full_benchmark,
        max_items=max_items,
        sample_mode=sample_mode,
        sample_seed=sample_seed,
        sample_strata=sample_strata,
    )
 
    total = len(benchmark)
    hard_timeout_label = exec_hard_timeout if exec_hard_timeout else "off"
    print(f"    📊 Benchmark: {total} items  rtol={rtol} atol={atol} "
          f"fix_rounds={fix_rounds} resume={resume} "
          f"exec_timeout={exec_timeout}s hard_timeout={hard_timeout_label}s "
          f"temp={generation_temperature:g}/{fix_temperature:g}")
    if sample_metadata["mode"] != "full":
        print(
            f"    🎲 Sample: mode={sample_metadata['mode']} "
            f"seed={sample_metadata['seed']} strata={sample_metadata['strata']} "
            f"from={sample_metadata['source_total']}  "
            f"counts={sample_metadata['stratum_counts']}"
        )
 
    # Warn about duplicate IDs
    n_unique = len(set(item.get("id", "") for item in benchmark))
    if n_unique < total:
        print(f"    ⚠️  {total - n_unique} duplicate IDs (using position index)")
 
    # Resume support — key by position index
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    start_fix_round = 1
    round_snapshots = []
    if resume and output_p.exists():
        try:
            with open(output_p) as f:
                prev = json.load(f)
            prev_token_mode = (
                (prev.get("config") or {})
                .get("token_accounting", {})
                .get("mode")
            )
            if prev_token_mode != "exact":
                print("    ⚠️  Existing result lacks exact token accounting; ignoring resume")
                prev = {}
                completed = {}
                start_fix_round = 1
                round_snapshots = []
                raise RuntimeError("__stale_token_accounting__")
            prev_sample = (prev.get("config") or {}).get("benchmark_sample") or {}
            prev_ids = prev_sample.get("selected_ids")
            sample_compatible = True
            if prev_ids is not None:
                sample_compatible = prev_ids == sample_metadata["selected_ids"]
            elif sample_metadata["mode"] not in ("full", "head"):
                sample_compatible = False
            if not sample_compatible:
                print("    ⚠️  Existing result uses a different/unknown benchmark sample; ignoring resume")
                prev = {}
                completed = {}
                start_fix_round = 1
                round_snapshots = []
                raise RuntimeError("__sample_mismatch__")
            for r in prev.get("item_results", []):
                completed[r["item_id"]] = r
            prev_fix_rounds = prev.get("config", {}).get("completed_rounds", 0)
            if len(completed) >= total and prev_fix_rounds >= fix_rounds:
                print(f"    ⏩ All {total} items done with {prev_fix_rounds} fix rounds, skipping")
                return prev.get("summary", {})
            elif len(completed) >= total:
                print(f"    ⏩ All items exist, but only {prev_fix_rounds}/{fix_rounds} fix rounds done")
                start_fix_round = prev_fix_rounds + 1
            else:
                print(f"    ⏩ Resuming: {len(completed)}/{total} done")
                start_fix_round = 1
            round_snapshots = prev.get("round_snapshots", [])
            print(f"    ⏩ Resuming: {len(completed)}/{total} done")
        except RuntimeError as e:
            if str(e) not in ("__sample_mismatch__", "__stale_token_accounting__"):
                pass
        except Exception:
            pass
    elif not resume and output_p.exists():
        print(f"    🗑️  Clearing previous results (resume=False)")
 
    if not backend.is_ready():
        backend.setup()
 
    error_retriever = None

    # ══════════════════════════════════════════════════════════
    # Round 0: Initial generation
    # ══════════════════════════════════════════════════════════
    pending = [(i, benchmark[i]) for i in range(total)
           if benchmark[i].get("id", str(i)) not in completed]
    if pending:
        cond_label = f" [condition={proactive_condition}]" if proactive_condition else ""
        print(f"\n    ── Round 0: Generate ({len(pending)} items){cond_label} ──")

        n_setup = sum(1 for _, item in pending if item.get("network_setup_code"))
        if n_setup:
            # Runtime verification point for the B1' self-contained-prompt wiring
            # (E2 cross-backend bench); 0 for every pandapower benchmark.
            print(f"    🧩 Given network setup code attached to {n_setup}/{len(pending)} prompts")

        if injector and proactive_condition:
            messages_list = [
                injector.build_messages(item, proactive_condition)
                for _, item in pending
            ]
        else:
            messages_list = [
                build_benchmark_prompt(render_benchmark_task_text(item))
                for _, item in pending
            ]

        prompt_token_counts = _precount_chat_tokens(
            backend, messages_list, "Round-0"
        )
        if prompt_token_counts:
            avg_prompt_tokens = int(sum(prompt_token_counts) / max(len(prompt_token_counts), 1))
            total_prompt_tokens = int(sum(prompt_token_counts))
            print(f"    📊 Prompt tokens: avg={avg_prompt_tokens}, total={total_prompt_tokens}")

        t0 = time.time()
        raw_outputs = backend.generate_batch(
            messages_list, max_new_tokens=max_new_tokens,
            temperature=generation_temperature, batch_size=batch_size,
        )
        print(f"    ⏱  Inference: {time.time()-t0:.1f}s")

        runtime_counts = _runtime_prompt_counts(backend, len(messages_list))
        if runtime_counts:
            prompt_token_counts = runtime_counts
            avg_prompt_tokens = int(sum(prompt_token_counts) / max(len(prompt_token_counts), 1))
            total_prompt_tokens = int(sum(prompt_token_counts))
            print(f"    📊 Prompt tokens (actual backend count): avg={avg_prompt_tokens}, total={total_prompt_tokens}")
        if not prompt_token_counts:
            raise RuntimeError(
                "Exact Round-0 prompt token counts are unavailable; refusing to write approximate costs."
            )
 
        t_exec = time.time()
        eval_args = [
            (bench_i, item, raw, rtol, atol, exec_timeout)
            for (bench_i, item), raw in zip(pending, raw_outputs)
        ]
        eval_results = _parallel_map(
            _eval_one_item_worker,
            eval_args,
            workers=exec_workers,
            hard_timeout=exec_hard_timeout,
            label="round-0 code execution",
        )
        print(f"    ⏱  Code execution: {time.time()-t_exec:.1f}s ({exec_workers} worker(s))")
        for idx, result in enumerate(eval_results):
            result["prompt_tokens"] = prompt_token_counts[idx]
            result["prompt_tokens_approx"] = prompt_token_counts[idx]  # legacy key; value is exact
            completed[result["item_id"]] = result
        gc.collect()

        _print_round_stats(completed, 0)
        round_snapshots = [_snapshot_round(completed, 0)]
        _save_checkpoint(output_path, completed, round_snapshots, rtol, atol,
                         exec_timeout, fix_rounds, completed_rounds=0,
                         exec_hard_timeout=exec_hard_timeout,
                         max_hard_timeout_fix_attempts=max_hard_timeout_fix_attempts,
                         sample_metadata=sample_metadata)
    elif start_fix_round > 1:
        resumed_round = start_fix_round - 1
        print(f"    ⏩ Continuing fix rounds from checkpoint round {resumed_round}/{fix_rounds}")
        _print_round_stats(completed, resumed_round)
        if not round_snapshots:
            round_snapshots = [_snapshot_round(completed, resumed_round)]
    else:
        _print_round_stats(completed, 0)
        if not round_snapshots:
            round_snapshots = [_snapshot_round(completed, 0)]
        _save_checkpoint(output_path, completed, round_snapshots, rtol, atol,
                         exec_timeout, fix_rounds, completed_rounds=0,
                         exec_hard_timeout=exec_hard_timeout,
                         max_hard_timeout_fix_attempts=max_hard_timeout_fix_attempts,
                         sample_metadata=sample_metadata)
    # ══════════════════════════════════════════════════════════
    # Rounds 1..N: Fix failed items
    # ══════════════════════════════════════════════════════════
    for round_i in range(start_fix_round, fix_rounds + 1):
        failed_keys = [
            k for k, r in completed.items()
            if not r["match"] and r.get("generated_code")  # has code but wrong/crashed
            and _eligible_for_fix_after_timeout(r, max_hard_timeout_fix_attempts)
        ]
        skipped_timeout_keys = [
            k for k, r in completed.items()
            if not r["match"] and r.get("generated_code")
            and not _eligible_for_fix_after_timeout(r, max_hard_timeout_fix_attempts)
        ]
        if not failed_keys:
            if skipped_timeout_keys:
                print(
                    f"\n    ── Round {round_i}: No eligible failures left; "
                    f"{len(skipped_timeout_keys)} repeated hard-timeout item(s) held out ──"
                )
            else:
                print(f"\n    ── Round {round_i}: No failures left, stopping ──")
            break
 
        extra_skip = (
            f" ({len(skipped_timeout_keys)} repeated hard-timeout item(s) held out)"
            if skipped_timeout_keys else ""
        )
        print(f"\n    ── Round {round_i}: Fix {len(failed_keys)} failed items{extra_skip} ──")
 
        # Build fix prompts
        fix_messages = []
        fix_items = []
        for key in failed_keys:
            r = completed[key]
            bench_i = r["bench_index"]
            item = benchmark[bench_i]
            buggy_code = r["generated_code"]
 
            # Build error description
            error_msg, route_error_type = _format_fix_error(r)

            # Error-targeted doc injection (reactive > RAG)
            docs = ""
            doc_meta = {
                "source": "none",
                "functions": [],
                "error_class": r.get("error_type") or "default",
                "layers": [],
                "n_layers": 0,
                "route": "none",
                "route_reason": "",
                "boundary_cards": [],
                "semantic_signals": [],
                "semantic_repair_actions": [],
                "semantic_contract_ids": [],
                "semantic_intents": [],
                "semantic_anchor_functions": [],
                "semantic_expected_contract": {},
                "semantic_observed_contract": {},
                "doc_tokens": 0,
                "doc_tokens_approx": 0,
            }
            if reactive_injector:
                try:
                    details = reactive_injector.get_injection_details(
                        error_type=route_error_type,
                        error_msg=error_msg,
                        generated_code=r.get("generated_code", ""),
                        original_query=item.get("natural_language_query", ""),
                    )
                    docs = details.get("docs", "")
                    doc_meta.update({
                        "source": details.get("source") or ("reactive" if docs else "reactive_empty"),
                        "functions": details.get("functions", []),
                        "error_class": details.get("error_class"),
                        "layers": details.get("layers", []),
                        "n_layers": details.get("n_layers", 0),
                        "route": details.get("route", "unknown"),
                        "route_reason": details.get("route_reason", ""),
                        "boundary_cards": details.get("boundary_cards", []),
                        "semantic_signals": details.get("semantic_signals", []),
                        "semantic_repair_actions": details.get("semantic_repair_actions", []),
                        "semantic_contract_ids": details.get("semantic_contract_ids", []),
                        "semantic_intents": details.get("semantic_intents", []),
                        "semantic_anchor_functions": details.get("semantic_anchor_functions", []),
                        "semantic_expected_contract": details.get("semantic_expected_contract", {}),
                        "semantic_observed_contract": details.get("semantic_observed_contract", {}),
                    })
                except Exception:
                    pass
            elif error_retriever:
                try:
                    doc_result = error_retriever.analyze_and_retrieve(error_msg, buggy_code)
                    docs = doc_result.get("retrieved_docs") or doc_result.get("fixed_docs", "")
                    doc_meta.update({
                        "source": "rag" if docs else "rag_empty",
                    })
                except Exception:
                    pass

            doc_tokens = _exact_text_tokens(backend, docs, "fix docs")
            doc_meta["doc_tokens"] = doc_tokens
            doc_meta["doc_tokens_approx"] = doc_tokens  # legacy key; value is exact
 
            fix_msg = build_fix_prompt(
                buggy_code=buggy_code,
                error_msg=error_msg,
                original_query=render_benchmark_task_text(item),
                retrieved_docs=docs,
            )
            fix_messages.append(fix_msg)

            fix_items.append((key, bench_i, item, {
                "previous_error": error_msg,
                "fix_docs_tokens": doc_meta["doc_tokens"],
                "fix_docs_tokens_approx": doc_meta["doc_tokens_approx"],
                "fix_docs_source": doc_meta["source"],
                "fix_docs_functions": doc_meta["functions"],
                "fix_docs_error_class": doc_meta["error_class"],
                "fix_docs_layers": doc_meta["layers"],
                "fix_docs_n_layers": doc_meta["n_layers"],
                "fix_route": doc_meta["route"],
                "fix_route_reason": doc_meta["route_reason"],
                "fix_boundary_cards": doc_meta["boundary_cards"],
                "fix_semantic_signals": doc_meta["semantic_signals"],
                "fix_semantic_repair_actions": doc_meta["semantic_repair_actions"],
                "fix_semantic_contract_ids": doc_meta["semantic_contract_ids"],
                "fix_semantic_intents": doc_meta["semantic_intents"],
                "fix_semantic_anchor_functions": doc_meta["semantic_anchor_functions"],
                "fix_semantic_expected_contract": doc_meta["semantic_expected_contract"],
                "fix_semantic_observed_contract": doc_meta["semantic_observed_contract"],
            }))

        if fix_items:
            fix_prompt_counts = _precount_chat_tokens(
                backend, fix_messages, f"Round-{round_i} fix"
            )
            if fix_prompt_counts:
                for idx, entry in enumerate(fix_items):
                    meta = entry[3]
                    meta["fix_prompt_tokens"] = fix_prompt_counts[idx]
                    meta["fix_prompt_tokens_approx"] = fix_prompt_counts[idx]  # legacy key; value is exact
            fix_prompt_tokens = [
                meta.get("fix_prompt_tokens") for *_, meta in fix_items
                if meta.get("fix_prompt_tokens") is not None
            ]
            fix_doc_tokens = [meta["fix_docs_tokens"] for *_, meta in fix_items]
            n_doc_nonempty = sum(1 for toks in fix_doc_tokens if toks > 0)
            if fix_prompt_tokens:
                print(
                    "    📊 Fix prompt tokens: "
                    f"avg={int(sum(fix_prompt_tokens) / len(fix_prompt_tokens))}, "
                    f"total={sum(fix_prompt_tokens)}, "
                    f"docs_avg={int(sum(fix_doc_tokens) / len(fix_doc_tokens))}, "
                    f"docs_nonempty={n_doc_nonempty}/{len(fix_doc_tokens)}"
                )
 
        # Generate fixes
        t0 = time.time()
        fix_outputs = backend.generate_batch(
            fix_messages, max_new_tokens=fix_max_new_tokens,
            temperature=fix_temperature, batch_size=batch_size,
        )
        print(f"    ⏱  Fix inference: {time.time()-t0:.1f}s")
        runtime_fix_counts = _runtime_prompt_counts(backend, len(fix_messages))
        if runtime_fix_counts:
            for idx, entry in enumerate(fix_items):
                meta = entry[3]
                meta["fix_prompt_tokens"] = runtime_fix_counts[idx]
                meta["fix_prompt_tokens_approx"] = runtime_fix_counts[idx]  # legacy key; value is exact
            print(
                "    📊 Fix prompt tokens (actual backend count): "
                f"avg={int(sum(runtime_fix_counts) / len(runtime_fix_counts))}, "
                f"total={sum(runtime_fix_counts)}"
            )
        if fix_items and any(entry[3].get("fix_prompt_tokens") is None for entry in fix_items):
            raise RuntimeError(
                "Exact fix prompt token counts are unavailable; refusing to write approximate costs."
            )

        # --- Parallel code execution ---
        cleaned_list = [clean_generated_code(raw) for raw in fix_outputs]
        t_exec = time.time()
        exec_result_list = _parallel_map(
            _execute_code_worker,
            [(c, exec_timeout) for c in cleaned_list],
            workers=exec_workers,
            hard_timeout=exec_hard_timeout,
            label=f"round-{round_i} fix execution",
        )
        print(f"    ⏱  Code execution: {time.time()-t_exec:.1f}s ({exec_workers} worker(s))")

        # Evaluate fixes
        n_fixed_this_round = 0
        for (key, bench_i, item, fix_meta), raw, cleaned, exec_result in zip(
            fix_items, fix_outputs, cleaned_list, exec_result_list
        ):
            old_result = completed[key]
 
            # Record in fix chain
            fix_entry = {
                "round": round_i,
                "previous_code": old_result["generated_code"],
                "previous_error": fix_meta.get("previous_error"),
                "previous_error_type": old_result.get("error_type"),
                "previous_executed": old_result.get("executed"),
                "previous_match": old_result.get("match"),
                "previous_match_detail": old_result.get("match_detail"),
                "fix_condition": reactive_injector.fix_condition if reactive_injector else "FX",
                **fix_meta,
            }
 
            if not cleaned or exec_result is None:
                fix_entry["success"] = False
                fix_entry["error_type"] = "code_extraction_failed"
            else:
                error_msg = parse_error(exec_result, cleaned)
                fix_entry["fixed_code"] = cleaned
                fix_entry["executed"] = exec_result["success"]
                fix_entry["error_type"] = exec_result.get("error_type")
                fix_entry["error_msg"] = error_msg
                if exec_result.get("hard_timeout"):
                    fix_entry["hard_timeout"] = True
                    fix_entry["hard_timeout_seconds"] = exec_result.get("hard_timeout_seconds")
 
                if exec_result["success"]:
                    match_result = match_ground_truth(
                        exec_result["output"], item["ground_truth"],
                        item["ground_truth_type"], rtol=rtol, atol=atol,
                    )
                    fix_entry["match"] = match_result["match"]
                    fix_entry["success"] = match_result["match"]
 
                    if match_result["match"]:
                        # Update the main result to reflect the fix
                        old_result["generated_code"] = cleaned
                        old_result["executed"] = True
                        old_result["error_type"] = None
                        old_result["error_msg"] = None
                        old_result["exec_output"] = (exec_result["output"] or "")[:200]
                        old_result["match"] = True
                        old_result["match_detail"] = match_result
                        old_result["fixed_in_round"] = round_i
                        old_result.pop("hard_timeout", None)
                        old_result.pop("hard_timeout_seconds", None)
                        n_fixed_this_round += 1
                    else:
                        # Update code for next round but don't mark as matched
                        old_result["generated_code"] = cleaned
                        old_result["executed"] = exec_result["success"]
                        old_result["error_type"] = exec_result.get("error_type")
                        old_result["error_msg"] = error_msg
                        old_result["exec_output"] = (exec_result["output"] or "")[:200]
                        old_result["match_detail"] = match_result
                        old_result.pop("hard_timeout", None)
                        old_result.pop("hard_timeout_seconds", None)
                else:
                    fix_entry["match"] = False
                    fix_entry["success"] = False
                    # Update code for next round
                    old_result["generated_code"] = cleaned
                    old_result["executed"] = False
                    old_result["error_type"] = exec_result.get("error_type")
                    old_result["error_msg"] = error_msg
                    old_result["exec_output"] = ""
                    if exec_result.get("hard_timeout"):
                        old_result["hard_timeout"] = True
                        old_result["hard_timeout_seconds"] = exec_result.get("hard_timeout_seconds")
                    else:
                        old_result.pop("hard_timeout", None)
                        old_result.pop("hard_timeout_seconds", None)
 
            # Append to fix chain (skip if this round already recorded)
            if "fix_chain" not in old_result:
                old_result["fix_chain"] = []
            existing_rounds = {e["round"] for e in old_result["fix_chain"]}
            if round_i not in existing_rounds:
                old_result["fix_chain"].append(fix_entry)
            gc.collect()
 
        _print_round_stats(completed, round_i, n_fixed_this_round)

        # Snapshot + checkpoint after each fix round
        round_snapshots.append(_snapshot_round(completed, round_i))
        _save_checkpoint(output_path, completed, round_snapshots, rtol, atol,
                         exec_timeout, fix_rounds, completed_rounds=round_i,
                         exec_hard_timeout=exec_hard_timeout,
                         max_hard_timeout_fix_attempts=max_hard_timeout_fix_attempts,
                         generation_temperature=generation_temperature,
                         fix_temperature=fix_temperature,
                         sample_metadata=sample_metadata)
        print(f"    💾 Checkpoint saved after round {round_i}")

    # ══════════════════════════════════════════════════════════
    # Build final summary & save
    # ══════════════════════════════════════════════════════════
    all_results = list(completed.values())
    summary = _build_bench_summary(all_results, rtol, atol, round_snapshots)

    _save_checkpoint(output_path, completed, round_snapshots, rtol, atol,
                     exec_timeout, fix_rounds,
                     completed_rounds=len(round_snapshots) - 1,
                     exec_hard_timeout=exec_hard_timeout,
                     max_hard_timeout_fix_attempts=max_hard_timeout_fix_attempts,
                     generation_temperature=generation_temperature,
                     fix_temperature=fix_temperature,
                     summary_override=summary,
                     sample_metadata=sample_metadata)

    _print_bench_summary(summary)
    return summary
 
 
# ── Helper: evaluate one benchmark item ──────────────────────
 
def _eval_one_item(bench_i, item, raw, rtol, atol, exec_timeout):
    """Evaluate a single benchmark item. Returns item_result dict."""
    from backend.utils import (
        match_ground_truth, execute_code_safely, clean_generated_code, parse_error,
    )
    cleaned = clean_generated_code(raw)
 
    result = {
        "bench_index": bench_i,
        "benchmark_original_index": item.get("_benchmark_original_index", bench_i),
        "item_id": item.get("id", str(bench_i)),
        "difficulty_level": item.get("difficulty_level", "D1_basic"),
        "task": item["scenario"]["task"],
        "network": item["scenario"]["network"],
        "query_type": item["scenario"]["query_target"]["qtype"],
        "n_modifications": len(item["scenario"].get("modifications", [])),
        "natural_language_query": item["natural_language_query"],
        "reference_code": item["reference_code"],
        "ground_truth": item["ground_truth"],
        "ground_truth_type": item["ground_truth_type"],
        "eval_criteria": item.get("eval_criteria", {}),
        "generated_code": cleaned,
        "code_lines": len(cleaned.split("\n")) if cleaned else 0,
        "raw_output": raw[:500] if raw else "",
    }
 
    if not cleaned:
        result.update({
            "executed": False, "error_type": "code_extraction_failed", "error_msg": None,
            "exec_output": "", "match": False, "match_detail": None,
        })
    else:
        exec_result = execute_code_safely(cleaned, timeout=exec_timeout)
        error_msg = parse_error(exec_result, cleaned)
        result["executed"] = exec_result["success"]
        result["error_type"] = exec_result.get("error_type")
        result["error_msg"] = error_msg
        result["exec_output"] = (exec_result["output"] or "")[:200]
 
        if exec_result["success"]:
            match_result = match_ground_truth(
                exec_result["output"], item["ground_truth"],
                item["ground_truth_type"], rtol=rtol, atol=atol,
            )
            result["match"] = match_result["match"]
            result["match_detail"] = match_result
        else:
            result["match"] = False
            result["match_detail"] = None
 
    cleaned = result.get("generated_code", "")
    difficulty = item.get("difficulty_level", "D1_basic")
 
    if cleaned and difficulty in ("D1_basic", "D2_multi_step"):
        diag = diagnostic_checks(
            code=cleaned,
            task=item["scenario"]["task"],
            network=item["scenario"]["network"],
            modifications=item["scenario"].get("modifications", []),
        )
        result["diagnostics"] = diag
 
    elif difficulty in ("D3_semantic", "D4_compound"):
        # Prompt-quality diagnostics for derived semantic items.
        result["rewrite_diagnostics"] = semantic_rewrite_checks(
            query=item.get("natural_language_query", ""),
            modifications=item["scenario"].get("modifications", []),
            eval_criteria=item.get("eval_criteria", {}),
        )

    return result

def _snapshot_round(completed: Dict, round_i: int) -> Dict:
    """Take a full stats snapshot of current completed results."""
    from collections import defaultdict

    results = list(completed.values())
    total = len(results)
    n_executed = sum(1 for r in results if r["executed"])
    n_matched = sum(1 for r in results if r["match"])

    snapshot = {
        "round": round_i,
        "total": total,
        "n_executed": n_executed,
        "n_matched": n_matched,
        "execution_rate": round(n_executed / total, 4) if total else 0,
        "result_accuracy": round(n_matched / total, 4) if total else 0,
    }

    # Per-task
    task_stats = defaultdict(lambda: {"total": 0, "executed": 0, "matched": 0})
    for r in results:
        t = r["task"]
        task_stats[t]["total"] += 1
        if r["executed"]: task_stats[t]["executed"] += 1
        if r["match"]: task_stats[t]["matched"] += 1
    for t, s in task_stats.items():
        s["exec_rate"] = round(s["executed"] / s["total"], 4) if s["total"] else 0
        s["accuracy"] = round(s["matched"] / s["total"], 4) if s["total"] else 0
    snapshot["per_task"] = dict(task_stats)

    # Per GT-type
    gt_stats = defaultdict(lambda: {"total": 0, "executed": 0, "matched": 0})
    for r in results:
        g = r["ground_truth_type"]
        gt_stats[g]["total"] += 1
        if r["executed"]: gt_stats[g]["executed"] += 1
        if r["match"]: gt_stats[g]["matched"] += 1
    for g, s in gt_stats.items():
        s["accuracy"] = round(s["matched"] / s["total"], 4) if s["total"] else 0
    snapshot["per_gt_type"] = dict(gt_stats)

    # Error distribution
    error_types = Counter(r["error_type"] for r in results if r.get("error_type"))
    snapshot["error_distribution"] = dict(error_types.most_common())

    return snapshot


def _extract_error_type(error_msg: str):
    """Extract error type name from a compact parse_error() string.

    parse_error() produces strings like:
        "Line 5 | pp.create_bus(net) | AttributeError: object has no attribute..."
        "TypeError: func() got an unexpected keyword argument..."
    Returns the bare exception class name, or None if not found.
    """
    if not error_msg:
        return None
    import re
    # Last pipe-delimited segment is "ErrorType: message"
    last = error_msg.rsplit(" | ", 1)[-1]
    m = re.match(r'([A-Za-z][A-Za-z0-9]*(?:Error|Exception|Warning))\s*:', last)
    return m.group(1) if m else None


def _seed_round0_from_baseline(baseline_path: str, output_path: str) -> bool:
    """Seed a reactive experiment file from an existing base condition's Round 0 results.

    Avoids re-running Round 0 inference when the base condition (A/C/X) has already
    been evaluated. Restores item states to their Round 0 form so that
    _run_benchmark_with_backend (resume=True, fix_rounds=N) will skip Round 0 and
    go directly to fix rounds.

    Transformation per item:
      - Items correct in R0 (empty fix_chain, match=True): kept unchanged
      - Items with fix_chain (were attempted for fix): Round 0 code/error restored
        from fix_chain[0].previous_{code,error}; match reset to False
      - Parse failures (no generated_code, empty fix_chain): kept unchanged

    Args:
        baseline_path: path to existing base condition JSON (e.g. benchmark_results_condC.json)
        output_path:   target path to write seeded file (e.g. benchmark_results_condC_FD.json)

    Returns:
        True if seeding succeeded, False if baseline not found or unreadable.
    """
    if not Path(baseline_path).exists():
        return False
    try:
        with open(baseline_path) as f:
            baseline = json.load(f)
    except Exception as e:
        print(f"    ⚠️  Could not read baseline {baseline_path}: {e}")
        return False

    token_mode = (
        (baseline.get("config") or {})
        .get("token_accounting", {})
        .get("mode")
    )
    if token_mode != "exact":
        print(
            f"    ⚠️  Not seeding from {Path(baseline_path).name}: "
            "token accounting is not marked exact"
        )
        return False

    item_results = baseline.get("item_results", [])
    if not item_results:
        return False

    seeded_items = []
    n_restored = 0
    for r in item_results:
        item = dict(r)  # shallow copy — safe since we only reassign scalar fields
        fix_chain = item.get("fix_chain", [])

        if fix_chain:
            # Item went through at least one fix attempt — restore Round 0 state
            fc0 = fix_chain[0]
            item["generated_code"] = fc0.get("previous_code")
            item["error_msg"]      = fc0.get("previous_error")
            item["error_type"]     = (
                fc0.get("previous_error_type")
                if "previous_error_type" in fc0
                else _extract_error_type(item["error_msg"] or "")
            )
            item["match"]          = bool(fc0.get("previous_match", False))
            previous_detail = fc0.get("previous_match_detail")
            item["match_detail"]   = previous_detail if isinstance(previous_detail, dict) else {}
            item["fix_chain"]      = []
            item.pop("fixed_in_round", None)
            # Older files did not store previous_executed; keep the old fallback.
            item["executed"] = (
                bool(fc0.get("previous_executed"))
                if "previous_executed" in fc0
                else bool(item["generated_code"])
            )
            n_restored += 1
        # else: match=True (R0 correct) or parse_failed — keep as-is

        seeded_items.append(item)

    # Write seeded file: completed_rounds=0 so runner skips R0 and goes to fix rounds
    seeded = {
        "summary": baseline.get("round_snapshots", [{}])[0] if baseline.get("round_snapshots") else {},
        "config": {
            **baseline.get("config", {}),
            "completed_rounds": 0,
            "seeded_from": str(baseline_path),
        },
        "round_snapshots": [],   # will be rebuilt by _run_benchmark_with_backend
        "item_results": seeded_items,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(output_path) + ".seed.tmp"
    with open(tmp, "w") as f:
        json.dump(seeded, f, indent=2, ensure_ascii=False)
    import os; os.replace(tmp, str(output_path))

    n_correct = sum(1 for it in seeded_items if it.get("match"))
    n_failing = sum(1 for it in seeded_items if not it.get("match") and it.get("generated_code"))
    print(f"    🌱 Seeded R0 from {Path(baseline_path).name}: "
          f"{len(seeded_items)} items  "
          f"(already correct={n_correct}, will fix={n_failing}, restored={n_restored})")
    return True


def _has_resumeable_fix_checkpoint(output_path: str, fix_rounds: int) -> bool:
    """Return True when an existing A_FX file should be continued, not reseeded."""
    path = Path(output_path)
    if not path.exists():
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return False

    config = data.get("config") or {}
    token_mode = (config.get("token_accounting") or {}).get("mode")
    if token_mode != "exact":
        return False

    try:
        completed_rounds = int(config.get("completed_rounds", 0))
    except (TypeError, ValueError):
        completed_rounds = 0

    if completed_rounds <= 0:
        return False
    if not data.get("item_results"):
        return False

    # completed_rounds >= fix_rounds is useful too: the benchmark runner will
    # load the final checkpoint and skip without rewriting it.
    return completed_rounds <= fix_rounds


def _base_condition_path(model_dir: Path, pro_cond: str) -> Path:
    """Return the expected Round-0 result file for a given proactive base condition."""
    if pro_cond in ("A", "", None):
        return _preferred_baseline_path(model_dir)
    return model_dir / f"benchmark_results_cond{pro_cond}.json"


def _reactive_seed_candidates(
    model_dir: Path,
    pro_cond: str,
    fix_cond: str,
    output_path: Path,
) -> List[Path]:
    """Candidate files that can restore the same proactive Round 0 state."""
    candidates = []

    def add(path: Path, *, allow_output: bool = False):
        if (path == output_path and not allow_output) or path in candidates:
            return
        candidates.append(path)

    # In smoke/cost runs we often request only C_FX,C_FD,C_FDR without
    # writing standalone benchmark_results_condC.json. The sibling FX file still
    # contains enough fix_chain history to restore its Round 0 state.
    if fix_cond != "FX":
        pro_label = pro_cond if pro_cond not in ("", None) else "A"
        add(model_dir / f"benchmark_results_cond{pro_label}_FX.json")

    add(_base_condition_path(model_dir, pro_cond))
    # Last-resort fallback for forced reruns: an existing P_F file can usually
    # restore its own Round 0 from fix_chain[0].previous_* without regenerating.
    add(output_path, allow_output=True)

    return candidates


def _seed_reactive_round0(
    model_dir: Path,
    pro_cond: str,
    fix_cond: str,
    output_path: Path,
) -> bool:
    """Seed a reactive run from the best available paired Round 0 source."""
    tried = []
    for base_path in _reactive_seed_candidates(model_dir, pro_cond, fix_cond, output_path):
        tried.append(base_path.name)
        if _seed_round0_from_baseline(str(base_path), str(output_path)):
            return True
    if tried:
        print(f"    Tried Round 0 seed files: {', '.join(tried)}")
    return False


def _save_checkpoint(output_path, completed, round_snapshots, rtol, atol,
                     exec_timeout, fix_rounds, completed_rounds,
                     exec_hard_timeout=None,
                     max_hard_timeout_fix_attempts=None,
                     generation_temperature=0.0,
                     fix_temperature=0.0,
                     summary_override=None, sample_metadata=None):
    """Atomic checkpoint save."""
    import os
    all_results = list(completed.values())
    data = {
        "summary": summary_override or {},
        "config": {
            "rtol": rtol, "atol": atol, "timeout": exec_timeout,
            "hard_timeout": exec_hard_timeout,
            "max_hard_timeout_fix_attempts": max_hard_timeout_fix_attempts,
            "fix_rounds": fix_rounds, "completed_rounds": completed_rounds,
            "generation_temperature": generation_temperature,
            "fix_temperature": fix_temperature,
            "token_accounting": {
                "mode": "exact",
                "source": "backend tokenizer for HF/vLLM; API usage for GPT prompt tokens",
                "legacy_approx_fields": "exact aliases retained for old analysis scripts",
            },
            "benchmark_sample": sample_metadata or {},
        },
        "round_snapshots": round_snapshots,
        "item_results": all_results,
    }
    tmp_path = str(output_path) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, str(output_path))
 
def _print_progress(completed, idx, total, tag):
    done = list(completed.values())
    n_exec = sum(1 for r in done if r["executed"])
    n_match = sum(1 for r in done if r["match"])
    print(f"      [{tag}] {idx}/{total}  exec={n_exec}/{len(done)}  match={n_match}/{len(done)}")
 
 
def _print_round_stats(completed, round_i, n_fixed_this_round=None):
    done = list(completed.values())
    n_exec = sum(1 for r in done if r["executed"])
    n_match = sum(1 for r in done if r["match"])
    total = len(done)
    extra = f"  (+{n_fixed_this_round} fixed)" if n_fixed_this_round else ""
    print(f"    📊 After round {round_i}: "
          f"exec={n_exec}/{total} ({n_exec/total:.1%})  "
          f"match={n_match}/{total} ({n_match/total:.1%}){extra}")
 
 
def _build_bench_summary(results: List[Dict], rtol: float, atol: float,
                         round_snapshots: List[Dict] = None) -> Dict:
    """Build comprehensive benchmark summary from item results."""
    from collections import defaultdict
 
    total = len(results)
    n_executed = sum(1 for r in results if r["executed"])
    n_matched = sum(1 for r in results if r["match"])
 
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "execution_rate": round(n_executed / total, 4) if total else 0,
        "result_accuracy": round(n_matched / total, 4) if total else 0,
        "result_accuracy_of_executed": round(n_matched / n_executed, 4) if n_executed else 0,
        "n_executed": n_executed,
        "n_matched": n_matched,
        "tolerance": {"rtol": rtol, "atol": atol},
    }
 
    # Error distribution
    error_types = Counter(r["error_type"] for r in results if r["error_type"])
    summary["error_distribution"] = dict(error_types.most_common())
 
    # Match failure reasons (executed but wrong result)
    match_errors = Counter(
        r["match_detail"]["error"]
        for r in results
        if r["executed"] and not r["match"]
        and r.get("match_detail") and r["match_detail"].get("error")
    )
    # Also count "no_error_but_wrong" — value parsed but outside tolerance
    n_value_wrong = sum(
        1 for r in results
        if r["executed"] and not r["match"]
        and r.get("match_detail") and not r["match_detail"].get("error")
    )
    if n_value_wrong:
        match_errors["value_outside_tolerance"] = n_value_wrong
    summary["match_failure_reasons"] = dict(match_errors.most_common())

    # ── Token usage (exact tokenizer/API usage; *_approx keys are legacy aliases) ──
    def _token_stat(values: List[int]) -> Dict:
        vals = [int(v) for v in values if v is not None]
        if not vals:
            return {}
        return {
            "avg": int(sum(vals) / len(vals)),
            "total": sum(vals),
            "min": min(vals),
            "max": max(vals),
        }

    token_counts = [
        _token_value(r, "prompt_tokens", "prompt_tokens_approx")
        for r in results
        if _token_value(r, "prompt_tokens", "prompt_tokens_approx") is not None
    ]
    if token_counts:
        round0_stats = _token_stat(token_counts)
        summary["prompt_token_stats"] = round0_stats  # backward-compatible alias
        summary["round0_prompt_token_stats"] = round0_stats

    fix_entries = [
        fc for r in results for fc in (r.get("fix_chain") or [])
        if _token_value(fc, "fix_prompt_tokens", "fix_prompt_tokens_approx") is not None
    ]
    if fix_entries:
        fix_prompt_tokens = [
            _token_value(fc, "fix_prompt_tokens", "fix_prompt_tokens_approx")
            for fc in fix_entries
        ]
        fix_doc_tokens = [
            _token_value(fc, "fix_docs_tokens", "fix_docs_tokens_approx") or 0
            for fc in fix_entries
        ]
        per_round = {}
        for round_i in sorted({fc.get("round") for fc in fix_entries}):
            round_entries = [fc for fc in fix_entries if fc.get("round") == round_i]
            round_prompt = [
                _token_value(fc, "fix_prompt_tokens", "fix_prompt_tokens_approx")
                for fc in round_entries
            ]
            round_docs = [
                _token_value(fc, "fix_docs_tokens", "fix_docs_tokens_approx") or 0
                for fc in round_entries
            ]
            per_round[str(round_i)] = {
                "attempts": len(round_entries),
                "prompt": _token_stat(round_prompt),
                "docs": _token_stat(round_docs),
                "docs_nonempty": sum(1 for v in round_docs if v > 0),
                "successful_fixes": sum(1 for fc in round_entries if fc.get("success")),
            }

        by_source = Counter(fc.get("fix_docs_source", "unknown") for fc in fix_entries)
        by_error = Counter(fc.get("fix_docs_error_class", "unknown") for fc in fix_entries)
        by_route = Counter(fc.get("fix_route", "unknown") for fc in fix_entries)
        by_boundary = Counter(
            card
            for fc in fix_entries
            for card in (fc.get("fix_boundary_cards") or [])
        )
        route_stats = {}
        for route in sorted(by_route):
            route_entries = [fc for fc in fix_entries if fc.get("fix_route", "unknown") == route]
            route_docs = [
                _token_value(fc, "fix_docs_tokens", "fix_docs_tokens_approx") or 0
                for fc in route_entries
            ]
            route_prompt = [
                _token_value(fc, "fix_prompt_tokens", "fix_prompt_tokens_approx") or 0
                for fc in route_entries
            ]
            route_stats[route] = {
                "attempts": len(route_entries),
                "successful_fixes": sum(1 for fc in route_entries if fc.get("success")),
                "docs": _token_stat(route_docs),
                "prompt": _token_stat(route_prompt),
                "docs_nonempty": sum(1 for v in route_docs if v > 0),
            }
        successful = sum(1 for fc in fix_entries if fc.get("success"))
        total_fix_tokens = sum(int(v or 0) for v in fix_prompt_tokens)
        total_doc_tokens = sum(int(v or 0) for v in fix_doc_tokens)
        summary["fix_prompt_token_stats"] = {
            "attempts": len(fix_entries),
            "prompt": _token_stat(fix_prompt_tokens),
            "docs": _token_stat(fix_doc_tokens),
            "docs_total": total_doc_tokens,
            "docs_nonempty": sum(1 for v in fix_doc_tokens if v > 0),
            "docs_nonempty_rate": round(
                sum(1 for v in fix_doc_tokens if v > 0) / len(fix_doc_tokens), 4
            ),
            "successful_fixes": successful,
            "prompt_tokens_per_successful_fix": (
                round(total_fix_tokens / successful, 1) if successful else None
            ),
            "doc_sources": dict(by_source.most_common()),
            "error_classes": dict(by_error.most_common()),
            "routes": dict(by_route.most_common()),
            "boundary_cards": dict(by_boundary.most_common()),
            "route_stats": route_stats,
            "per_round": per_round,
        }

    if token_counts or fix_entries:
        round0_total = sum(token_counts) if token_counts else 0
        fix_total = sum(
            int(_token_value(fc, "fix_prompt_tokens", "fix_prompt_tokens_approx") or 0)
            for fc in fix_entries
        ) if fix_entries else 0
        summary["total_prompt_token_stats"] = {
            "round0_total": round0_total,
            "fix_total": fix_total,
            "total": round0_total + fix_total,
            "avg_per_item_including_fixes": round((round0_total + fix_total) / max(total, 1), 1),
            "fix_to_round0_ratio": (
                round(fix_total / round0_total, 3) if round0_total else None
            ),
        }

    # ── Parse failure rate (format bug metric) ──
    n_parse_fail = sum(
        1 for r in results
        if r["executed"] and not r["match"]
        and r.get("match_detail") and "cannot_parse" in str(r["match_detail"].get("error",""))
    )
    if n_executed > 0:
        summary["parse_failure_rate"] = round(n_parse_fail / n_executed, 4)
    summary["n_parse_failures"] = n_parse_fail
 
    # ── Per-task ──
    task_stats = defaultdict(lambda: {"total": 0, "executed": 0, "matched": 0})
    for r in results:
        t = r["task"]
        task_stats[t]["total"] += 1
        if r["executed"]: task_stats[t]["executed"] += 1
        if r["match"]: task_stats[t]["matched"] += 1
    for t, s in task_stats.items():
        s["exec_rate"] = round(s["executed"] / s["total"], 4) if s["total"] else 0
        s["accuracy"] = round(s["matched"] / s["total"], 4) if s["total"] else 0
        s["accuracy_of_executed"] = round(s["matched"] / s["executed"], 4) if s["executed"] else 0
    summary["per_task"] = dict(task_stats)
 
    # ── Per GT-type ──
    gt_stats = defaultdict(lambda: {"total": 0, "executed": 0, "matched": 0})
    for r in results:
        g = r["ground_truth_type"]
        gt_stats[g]["total"] += 1
        if r["executed"]: gt_stats[g]["executed"] += 1
        if r["match"]: gt_stats[g]["matched"] += 1
    for g, s in gt_stats.items():
        s["exec_rate"] = round(s["executed"] / s["total"], 4) if s["total"] else 0
        s["accuracy"] = round(s["matched"] / s["total"], 4) if s["total"] else 0
    summary["per_gt_type"] = dict(gt_stats)
 
    # ── Per query-type ──
    qt_stats = defaultdict(lambda: {"total": 0, "executed": 0, "matched": 0})
    for r in results:
        q = r.get("query_type", "unknown")
        qt_stats[q]["total"] += 1
        if r["executed"]: qt_stats[q]["executed"] += 1
        if r["match"]: qt_stats[q]["matched"] += 1
    for q, s in qt_stats.items():
        s["exec_rate"] = round(s["executed"] / s["total"], 4) if s["total"] else 0
        s["accuracy"] = round(s["matched"] / s["total"], 4) if s["total"] else 0
    summary["per_query_type"] = dict(qt_stats)
 
    # ── Per # modifications ──
    mod_stats = defaultdict(lambda: {"total": 0, "executed": 0, "matched": 0})
    for r in results:
        nm = r.get("n_modifications", 0)
        mod_stats[nm]["total"] += 1
        if r["executed"]: mod_stats[nm]["executed"] += 1
        if r["match"]: mod_stats[nm]["matched"] += 1
    for nm, s in mod_stats.items():
        s["exec_rate"] = round(s["executed"] / s["total"], 4) if s["total"] else 0
        s["accuracy"] = round(s["matched"] / s["total"], 4) if s["total"] else 0
    summary["per_n_modifications"] = dict(sorted(mod_stats.items(), key=lambda x: int(x[0])))
 
    # ── Per-network ──
    net_stats = defaultdict(lambda: {"total": 0, "executed": 0, "matched": 0})
    for r in results:
        n = r["network"]
        net_stats[n]["total"] += 1
        if r["executed"]: net_stats[n]["executed"] += 1
        if r["match"]: net_stats[n]["matched"] += 1
    for n, s in net_stats.items():
        s["exec_rate"] = round(s["executed"] / s["total"], 4) if s["total"] else 0
        s["accuracy"] = round(s["matched"] / s["total"], 4) if s["total"] else 0
    summary["per_network"] = dict(net_stats)
 
    # ── Code length stats ──
    code_lines = [r.get("code_lines", 0) for r in results if r.get("code_lines", 0) > 0]
    if code_lines:
        summary["code_stats"] = {
            "avg_lines": round(sum(code_lines) / len(code_lines), 1),
            "min_lines": min(code_lines),
            "max_lines": max(code_lines),
        }
 
    # ── Float error distribution (for matched floats) ──
    float_errors = []
    for r in results:
        if r["match"] and r["ground_truth_type"] == "float" and r.get("match_detail"):
            rel = r["match_detail"].get("rel_error")
            if rel is not None:
                float_errors.append(rel)
    if float_errors:
        import numpy as np
        arr = np.array(float_errors)
        summary["float_error_stats"] = {
            "mean_rel_error": round(float(arr.mean()), 6),
            "median_rel_error": round(float(np.median(arr)), 6),
            "max_rel_error": round(float(arr.max()), 6),
            "p95_rel_error": round(float(np.percentile(arr, 95)), 6),
        }
 
    # ── Round snapshots (direct, not reverse-engineered) ──
    if round_snapshots and len(round_snapshots) > 1:
        summary["round_snapshots"] = round_snapshots
 

    diff_stats = defaultdict(lambda: {
        "total": 0, "executed": 0, "matched": 0,
    })
 
    for r in results:
        d = r.get("difficulty_level", "D1_basic")
        diff_stats[d]["total"] += 1
        if r.get("executed"):
            diff_stats[d]["executed"] += 1
        if r.get("match"):
            diff_stats[d]["matched"] += 1
 
    for d, s in diff_stats.items():
        s["exec_rate"] = round(s["executed"] / max(s["total"], 1), 4)
        s["accuracy"] = round(s["matched"] / max(s["total"], 1), 4)
        s["accuracy_of_executed"] = round(s["matched"] / s["executed"], 4) if s["executed"] else 0
 
    summary["per_difficulty"] = dict(diff_stats)

    # ── Semantic rewrite diagnostics (D3/D4 only) ──
    rewrite_rows = [
        r for r in results
        if r.get("difficulty_level") in ("D3_semantic", "D4_compound")
        and r.get("rewrite_diagnostics")
    ]
    if rewrite_rows:
        leak_counter = Counter()
        phrase_hit_dist = Counter()
        source_counter = Counter()
        rule_counter = Counter()
        per_diff = defaultdict(lambda: {
            "total": 0,
            "has_semantic_phrase": 0,
            "leaks_explicit_params": 0,
            "total_phrase_hits": 0,
            "max_phrase_hits": 0,
        })

        for r in rewrite_rows:
            diag = r["rewrite_diagnostics"]
            diff = r.get("difficulty_level", "unknown")
            per_diff[diff]["total"] += 1
            if diag.get("has_semantic_phrase"):
                per_diff[diff]["has_semantic_phrase"] += 1
            if diag.get("leaks_explicit_params"):
                per_diff[diff]["leaks_explicit_params"] += 1
            n_hits = int(diag.get("n_semantic_phrase_hits", 0) or 0)
            per_diff[diff]["total_phrase_hits"] += n_hits
            per_diff[diff]["max_phrase_hits"] = max(per_diff[diff]["max_phrase_hits"], n_hits)
            phrase_hit_dist[n_hits] += 1
            source_counter[diag.get("semantic_source", "unknown")] += 1
            if diag.get("semantic_rule"):
                rule_counter[diag["semantic_rule"]] += 1
            for tok in diag.get("leaked_tokens", []) or []:
                leak_counter[tok] += 1

        for _, stats in per_diff.items():
            total_d = max(stats["total"], 1)
            stats["semantic_phrase_rate"] = round(stats["has_semantic_phrase"] / total_d, 4)
            stats["explicit_leak_rate"] = round(stats["leaks_explicit_params"] / total_d, 4)
            stats["avg_phrase_hits"] = round(stats["total_phrase_hits"] / total_d, 3)
            del stats["total_phrase_hits"]

        total_rw = len(rewrite_rows)
        total_has_phrase = sum(1 for r in rewrite_rows if r["rewrite_diagnostics"].get("has_semantic_phrase"))
        total_leaks = sum(1 for r in rewrite_rows if r["rewrite_diagnostics"].get("leaks_explicit_params"))
        total_phrase_hits = sum(int((r["rewrite_diagnostics"].get("n_semantic_phrase_hits", 0) or 0)) for r in rewrite_rows)

        summary["semantic_rewrite_summary"] = {
            "total": total_rw,
            "semantic_phrase_rate": round(total_has_phrase / total_rw, 4),
            "explicit_leak_rate": round(total_leaks / total_rw, 4),
            "avg_phrase_hits": round(total_phrase_hits / total_rw, 3),
            "phrase_hit_distribution": dict(sorted(phrase_hit_dist.items())),
            "semantic_sources": dict(source_counter.most_common()),
            "top_semantic_rules": dict(rule_counter.most_common(10)),
            "top_leaked_tokens": dict(leak_counter.most_common(12)),
            "per_difficulty": dict(per_diff),
        }

    return summary
 
 
def _print_bench_summary(summary: Dict):
    """Print comprehensive benchmark statistics (mirrors BenchmarkEngine.print_stats style)."""
 
    W = 60  # section width
 
    print(f"\n    {'='*W}")
    print(f"    📐 BENCHMARK RESULTS ({summary['total']} items)")
    print(f"    {'='*W}")
    print(f"      Execution rate:       {summary['execution_rate']:>6.1%}  "
          f"({summary['n_executed']}/{summary['total']})")
    print(f"      Result accuracy:      {summary['result_accuracy']:>6.1%}  "
          f"({summary['n_matched']}/{summary['total']})")
    print(f"      Accuracy | executed:  {summary['result_accuracy_of_executed']:>6.1%}  "
          f"({summary['n_matched']}/{summary['n_executed']})")

    # ── Per difficulty ──
    diff_order = ["D1_basic", "D2_multi_step", "D3_semantic", "D4_compound"]
    diff_stats = summary.get("per_difficulty", {})
    if diff_stats:
        print(f"\n      Difficulty Levels:")
        ordered = [(d, diff_stats[d]) for d in diff_order if d in diff_stats]
        ordered += [(d, ds) for d, ds in sorted(diff_stats.items()) if d not in diff_order]
        for diff, ds in ordered:
            bar_len = max(1, round(ds.get("accuracy", 0) * 20))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"        {diff:18s}:  n={ds['total']:4d}  "
                  f"exec={ds['exec_rate']:5.1%}  "
                  f"acc={ds['accuracy']:5.1%}  "
                  f"acc|ex={ds.get('accuracy_of_executed', 0):5.1%}  "
                  f"{bar}")
 
    # ── Per GT type ──
    print(f"\n      GT Types:")
    for gt, gs in sorted(summary.get("per_gt_type", {}).items(), key=lambda x: -x[1]["total"]):
        print(f"        {gt:8s}:  n={gs['total']:4d}  "
              f"exec={gs['exec_rate']:5.1%}  "
              f"acc={gs['accuracy']:5.1%}  "
              f"({gs['matched']}/{gs['total']})")
 
    # ── Per task (ALL tasks) ──
    print(f"\n      Tasks ({len(summary.get('per_task', {}))} types):")
    for task, ts in sorted(summary.get("per_task", {}).items(), key=lambda x: -x[1]["total"]):
        bar_len = max(1, round(ts["accuracy"] * 20))
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"        {task:25s}:  n={ts['total']:3d}  "
              f"exec={ts['exec_rate']:5.1%}  "
              f"acc={ts['accuracy']:5.1%}  "
              f"acc|ex={ts['accuracy_of_executed']:5.1%}  "
              f"{bar}")
 
    # ── Per query type ──
    qt = summary.get("per_query_type", {})
    if qt:
        print(f"\n      Query Types ({len(qt)}):")
        for q, qs in sorted(qt.items(), key=lambda x: -x[1]["total"]):
            print(f"        {q:25s}:  n={qs['total']:3d}  "
                  f"exec={qs['exec_rate']:5.1%}  "
                  f"acc={qs['accuracy']:5.1%}")
 
    # ── Per # modifications ──
    mod = summary.get("per_n_modifications", {})
    if mod:
        print(f"\n      # Modifications → accuracy:")
        for nm, ms in sorted(mod.items(), key=lambda x: int(x[0])):
            print(f"        {nm} mods:  n={ms['total']:4d}  "
                  f"exec={ms['exec_rate']:5.1%}  "
                  f"acc={ms['accuracy']:5.1%}")
 
    # ── Per network ──
    nets = summary.get("per_network", {})
    if nets:
        print(f"\n      Networks ({len(nets)} unique):")
        for n, ns in sorted(nets.items(), key=lambda x: -x[1]["total"]):
            print(f"        {n:30s}:  n={ns['total']:3d}  "
                  f"exec={ns['exec_rate']:5.1%}  "
                  f"acc={ns['accuracy']:5.1%}")
 
    # ── Error distribution ──
    errors = summary.get("error_distribution", {})
    if errors:
        print(f"\n      Execution Errors ({sum(errors.values())} total):")
        for err, cnt in sorted(errors.items(), key=lambda x: -x[1]):
            pct = cnt / summary["total"] * 100
            print(f"        {err:30s}: {cnt:4d}  ({pct:4.1f}%)")
 
    # ── Match failure reasons ──
    mf = summary.get("match_failure_reasons", {})
    if mf:
        n_exec_wrong = summary["n_executed"] - summary["n_matched"]
        print(f"\n      Match Failures ({n_exec_wrong} executed but wrong):")
        for reason, cnt in sorted(mf.items(), key=lambda x: -x[1]):
            print(f"        {reason:40s}: {cnt:4d}")
 
    # ── Code stats ──
    cs = summary.get("code_stats", {})
    if cs:
        print(f"\n      Generated code: avg={cs['avg_lines']:.0f} lines  "
              f"min={cs['min_lines']}  max={cs['max_lines']}")

    # ── Prompt token accounting ──
    total_tok = summary.get("total_prompt_token_stats", {})
    if total_tok:
        print(f"\n      Prompt tokens:")
        r0 = summary.get("round0_prompt_token_stats", {})
        if r0:
            print(f"        round0 : avg={r0.get('avg', 0):5d}  total={r0.get('total', 0):8d}")
        fx = summary.get("fix_prompt_token_stats", {})
        if fx:
            prompt = fx.get("prompt", {})
            docs = fx.get("docs", {})
            print(
                f"        fix    : attempts={fx.get('attempts', 0):4d}  "
                f"avg={prompt.get('avg', 0):5d}  total={prompt.get('total', 0):8d}  "
                f"docs_avg={docs.get('avg', 0):5d}  "
                f"docs_nonempty={fx.get('docs_nonempty', 0)}/{fx.get('attempts', 0)}"
            )
            if fx.get("successful_fixes"):
                print(
                    f"        fix ROI: successful={fx['successful_fixes']:4d}  "
                    f"tokens/success={fx.get('prompt_tokens_per_successful_fix')}"
                )
        print(
            f"        total  : {total_tok.get('total', 0):8d}  "
            f"avg/item={total_tok.get('avg_per_item_including_fixes', 0):.1f}  "
            f"fix/round0={total_tok.get('fix_to_round0_ratio')}"
        )

    # ── Semantic rewrite diagnostics ──
    sr = summary.get("semantic_rewrite_summary", {})
    if sr:
        print(f"\n      Semantic rewrite quality ({sr['total']} D3/D4 items):")
        print(f"        phrase_present={sr['semantic_phrase_rate']:.1%}  "
              f"explicit_leak={sr['explicit_leak_rate']:.1%}  "
              f"avg_phrase_hits={sr['avg_phrase_hits']:.2f}")
        per_diff = sr.get("per_difficulty", {})
        for diff, ds in sorted(per_diff.items()):
            print(f"        {diff:14s}:  n={ds['total']:3d}  "
                  f"phrase={ds['semantic_phrase_rate']:.1%}  "
                  f"leak={ds['explicit_leak_rate']:.1%}  "
                  f"avg_hits={ds['avg_phrase_hits']:.2f}")
        top_rules = sr.get("top_semantic_rules", {})
        if top_rules:
            preview = ", ".join(f"{k}:{v}" for k, v in list(top_rules.items())[:6])
            print(f"        top rules: {preview}")
        top_leaks = sr.get("top_leaked_tokens", {})
        if top_leaks:
            preview = ", ".join(f"{k}:{v}" for k, v in list(top_leaks.items())[:6])
            print(f"        leaked tokens: {preview}")
 
    # ── Float error stats ──
    fs = summary.get("float_error_stats", {})
    if fs:
        print(f"      Float error (matched): "
              f"mean_rel={fs['mean_rel_error']:.6f}  "
              f"median={fs['median_rel_error']:.6f}  "
              f"p95={fs['p95_rel_error']:.6f}  "
              f"max={fs['max_rel_error']:.6f}")
 
# ── Fix-round stats (from snapshots) ──
    snapshots = summary.get("round_snapshots", [])
    if len(snapshots) > 1:
        r0 = snapshots[0]
        rf = snapshots[-1]
        print(f"\n      🔧 Fix Rounds ({len(snapshots)-1} rounds executed)")
        print(f"        Round 0 (baseline):  "
              f"exec={r0['execution_rate']:.1%}  "
              f"acc={r0['result_accuracy']:.1%}  "
              f"({r0['n_matched']}/{r0['total']})")
        print(f"        Final (round {rf['round']}):    "
              f"exec={rf['execution_rate']:.1%}  "
              f"acc={rf['result_accuracy']:.1%}  "
              f"({rf['n_matched']}/{rf['total']})")

        # Accuracy curve
        print(f"\n        Accuracy curve:")
        for snap in snapshots:
            ri = snap["round"]
            delta = ""
            if ri > 0:
                prev = snapshots[ri - 1] if ri - 1 < len(snapshots) else snapshots[0]
                d = snap["n_matched"] - prev["n_matched"]
                delta = f"  (+{d} fixed)"
            print(f"          round_{ri}:  "
                  f"exec={snap['execution_rate']:>5.1%}  "
                  f"acc={snap['result_accuracy']:>5.1%}{delta}")

        # Per-task: round 0 vs final
        r0_tasks = r0.get("per_task", {})
        rf_tasks = rf.get("per_task", {})
        if r0_tasks and rf_tasks:
            improved = {
                t: rf_tasks[t]["accuracy"] - r0_tasks.get(t, {}).get("accuracy", 0)
                for t in rf_tasks
                if rf_tasks[t]["accuracy"] > r0_tasks.get(t, {}).get("accuracy", 0)
            }
            if improved:
                print(f"\n        Per-task (round_0 → round_{rf['round']}):")
                for t, delta in sorted(improved.items(), key=lambda x: -x[1]):
                    a0 = r0_tasks.get(t, {}).get("accuracy", 0)
                    af = rf_tasks[t]["accuracy"]
                    print(f"          {t:25s}: {a0:5.1%} → {af:5.1%}  (+{delta:.1%})")

        # Error change
        r0_errs = r0.get("error_distribution", {})
        rf_errs = rf.get("error_distribution", {})
        if r0_errs != rf_errs:
            print(f"\n        Error change (round_0 → round_{rf['round']}):")
            all_errs = sorted(set(list(r0_errs.keys()) + list(rf_errs.keys())))
            for err in all_errs:
                c0 = r0_errs.get(err, 0)
                cf = rf_errs.get(err, 0)
                if c0 != cf:
                    print(f"          {err:25s}: {c0:4d} → {cf:4d}")
 
    print(f"    {'='*W}")
 
 
# ============================================================
# Helpers
# ============================================================
 
def _dict_to_probe(d: Dict):
    try:
        from probing.probe_framework import Probe
        return Probe(**d)
    except ImportError:
        from types import SimpleNamespace
        return SimpleNamespace(**d)
 
 
def _eval_result_to_dict(r) -> Dict:
    try:
        from dataclasses import asdict
        return asdict(r)
    except Exception:
        return r.__dict__
 

# ============================================================
# Entry point
# ============================================================

def run_probe_evaluation(
    config: Dict,
    probes_path: str = "dataset/all_probes.json",
    api_spec_path: str = "dataset/pandapower_docs.json",
    benchmark_path: str = None,
    benchmark_fix_rounds: int = 0,
    search_engine = None,
):
    """
    Call this from your main() to run the full evaluation.

    Proactive Compensation CONFIG keys (all optional; comment out to disable):
        "proactive_conditions"      : ["B", "C", "X"]   — conditions to run alongside standard benchmark.
                                      Do NOT include "A" — A+FX round 0 is condition A.
                                      Knowledge profiles are auto-loaded per model from probe results.
                                      Results saved to benchmark_results_cond{X}.json (resume-capable).
        "proactive_demand_suite"    : suite directory, e.g. "results/demand_suites/suite_4428033"
        "proactive_demand_model"    : demand model name (default: "hybrid_tfidf")
                                      The demand export is model-agnostic for target evaluation models
                                      (7B/14B/32B etc.) — one file serves all. Per-model knowledge
                                      differences are captured in knowledge_profile.json from probes.
        "proactive_docs_path"       : path to docs JSON (default: tool_library_path)
        "proactive_top_k"           : int, default 10
        "proactive_token_budget_C"  : int, default 2000
        "proactive_token_budget_X"  : int, default 4000
        "proactive_threshold"       : float, default 0.5

    Reactive Injection CONFIG keys (all optional; comment out to disable):
        "reactive_combinations"     : list of "P_F" strings, default ["C_FX", "C_FD", "C_FDR"]
                                      P = proactive base (A=none, C, X)
                                      F = fix condition (FX=no injection, FR, FD=always-doc,
                                      FDR=routed, FS=semantic-only, FDRS=routed+semantic, FE)
                                      Output: benchmark_results_cond{P}_{F}.json per model.
                                      C+FDR is the low-cost routed candidate; C+FD remains always-doc baseline.
        "reactive_fix_rounds"       : int, fix rounds for reactive experiments (default: 3)
        "reactive_docs_path"        : path to docs JSON for ReactiveInjector (default: tool_library_path)
        "reactive_max_funcs"        : max implicated functions documented per fix prompt (default: 3)

    Env var overrides (from run_multinode_job.sh):
        PROACTIVE_CONDITIONS        : comma-separated, e.g. "B,C,X" (never include "A")
        PROACTIVE_DEMAND_SUITE      : path to suite directory
        PROACTIVE_DEMAND_MODEL      : demand model name (default: hybrid_tfidf)
        PROACTIVE_DEMAND_EXPORT     : direct demand export JSON path
        PROACTIVE_DOCS_PATH         : docs JSON for proactive injection
        REACTIVE_DOCS_PATH          : docs JSON for reactive injection
        PROACTIVE_TOP_K             : integer, demand candidates per task
        PROACTIVE_TOKEN_BUDGET_C/X  : integer token budgets for C/X
        PROACTIVE_THRESHOLD         : float knowledge threshold
        REACTIVE_COMBINATIONS       : comma-separated, e.g. "C_FX,C_FD,C_FDR"
        REACTIVE_FIX_ROUNDS         : integer, e.g. "3"
        REACTIVE_MAX_FUNCS          : integer, e.g. "2" for lower-cost reactive injection
        BENCHMARK_BATCH_SIZE        : integer generation batch size
        BENCHMARK_MAX_NEW_TOKENS    : integer round-0 generation limit
        BENCHMARK_FIX_MAX_NEW_TOKENS: integer fix-round generation limit
        BENCHMARK_EXEC_WORKERS      : integer parallel code-execution workers
        BENCHMARK_MAX_ITEMS         : integer smoke-test limit
        BENCHMARK_RTOL / ATOL       : float matching tolerances
        BENCHMARK_SAMPLE_MODE       : head | random | stratified
        BENCHMARK_SAMPLE_SEED       : integer seed for random/stratified smoke sampling
        BENCHMARK_SAMPLE_STRATA     : comma-separated strata, e.g. difficulty or difficulty,task
        BENCHMARK_EXEC_HARD_TIMEOUT : parent-side liveness fuse in seconds; <=0 disables it
        BENCHMARK_MAX_HARD_TIMEOUT_FIX_ATTEMPTS
                                    : max reactive fix attempts after parent hard timeout
        BENCHMARK_GENERATION_TEMPERATURE
                                    : round-0 decoding temperature (default 0.0)
        BENCHMARK_FIX_TEMPERATURE   : fix-round decoding temperature (default 0.0)
        PROBE_TEMPERATURE           : probe decoding temperature (default 0.0)
        GENERATION_SEED             : vLLM sampling seed when supported (default 22)
        PROBE_LAYERS                : comma-separated probe layers for rerun_probe
        RUN_ACTION                  : rerun_probe | rerun_benchmark | resume_benchmark_fix | rerun_proactive | rerun_reactive | empty compare flow
        OUTPUT_DIR                  : output directory for smoke/full results
        PROFILE_SOURCE_DIR          : directory to read existing knowledge_profile.json files
        PROBES_PATH                 : probe question set JSON (default dataset/all_probes.json);
                                      swap for cross-backend transfer (E2)
        API_SPEC_PATH               : backend spec JSON for fake-name pool + probe evaluator
                                      (default dataset/pandapower_docs.json); swap for E2
        RUN_MODELS                  : comma-separated model names, e.g. "Qwen/...,meta-llama/..."
    """
    results_dir = "probe_eval_results"
    backend_type = config.get("llm_backend", "vllm")

    # RUN_MODELS: comma-separated model names passed at sbatch time, e.g.:
    #   RUN_MODELS="Qwen/Qwen3-Coder-480B-A35B-Instruct,meta-llama/Llama-3.1-405B-Instruct" sbatch run_multinode_job.sh
    # Falls back to the hardcoded list below if env var is not set.
    _run_models_env = os.environ.get("RUN_MODELS", "").strip()
    if _run_models_env:
        _model_names = [m.strip() for m in _run_models_env.split(",") if m.strip()]
    else:
        _model_names = [
            "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "Qwen/Qwen2.5-Coder-14B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct",
            "Qwen/Qwen3-Coder-Next",
            "Qwen/Qwen3-Coder-480B-A35B-Instruct",
            "meta-llama/Llama-3.1-405B-Instruct",
            "openai/gpt-oss-120b",
            # API robustness models; uncomment explicitly and configure .env keys.
            # "gpt-4.1",
            # "claude-sonnet-4.5",
            # "gemini-2.5-pro",
            # "deepseek-v4-flash",
            # "gpt-4o-mini",
        ]

    comp = MultiModelComparison.from_model_names(
        model_names=_model_names,
        probes_path=probes_path,
        base_config={
            "gpt_api_key": OPENAI_API_KEY,
            "deepseek_api_key": DEEPSEEK_API_KEY,
            "anthropic_api_key": ANTHROPIC_API_KEY,
            "gemini_api_key": GEMINI_API_KEY,
            "generation_seed": config.get("generation_seed", 22),
            "probe_temperature": config.get("probe_temperature", 0.0),
        },
        backend_type=backend_type,
        api_spec_path=api_spec_path,
    )

    output_dir = config.get("output_dir") or f"{results_dir}/comparison/"

    proactive_conditions  = config.get("proactive_conditions")
    proactive_docs_path   = config.get("proactive_docs_path") or config.get(
        "tool_library_path", "dataset/pandapower_docs.json"
    )
    reactive_combinations = config.get("reactive_combinations")
    reactive_docs_path    = config.get("reactive_docs_path") or config.get(
        "tool_library_path", "dataset/pandapower_docs.json"
    )

    # ── Option 1: Only re-run standard A+FX benchmark (force) ───────────────
    # comp.rerun_benchmark(benchmark_path, output_dir=output_dir, fix_rounds=benchmark_fix_rounds)

    # ── Option 2: Re-run specific probe layers ───────────────────────────────
    # comp.rerun(["L0_recognition", "L1_recall", "L2_comprehension", "L3_application"], output_dir=output_dir)

    # ── Option 3: Force re-run proactive conditions ──────────────────────────
    # comp.rerun_proactive(
    #     benchmark_path=benchmark_path,
    #     proactive_demand_suite=config.get("proactive_demand_suite"),
    #     proactive_demand_model=config.get("proactive_demand_model", "hybrid_tfidf"),
    #     conditions=["B", "C", "X"],
    #     output_dir=output_dir,
    #     # models=["Qwen/Qwen2.5-Coder-7B-Instruct"],
    # )

    # ── Option 4: Force re-run reactive combinations ─────────────────────────
    # comp.rerun_reactive(
    #     benchmark_path=benchmark_path,
    #     combinations=["C_FX", "C_FD", "C_FDR"],
    #     proactive_demand_suite=config.get("proactive_demand_suite"),
    #     output_dir=output_dir,
    #     # models=["Qwen/Qwen2.5-Coder-7B-Instruct"],
    # )

    run_action = os.environ.get("RUN_ACTION", "").strip().lower()
    probe_layers_env = os.environ.get("PROBE_LAYERS", "").strip()
    probe_layers = (
        [x.strip() for x in probe_layers_env.split(",") if x.strip()]
        if probe_layers_env
        else ["L0_recognition", "L1_recall", "L2_comprehension", "L3_application"]
    )

    if run_action == "rerun_probe":
        comp.rerun(
            layers=probe_layers,
            output_dir=output_dir,
            batch_size=8,
        )
        return

    if run_action == "rerun_benchmark":
        comp.rerun_benchmark(
            benchmark_path=benchmark_path,
            output_dir=output_dir,
            fix_rounds=benchmark_fix_rounds,
            max_items=config.get("benchmark_max_items"),
            sample_mode=config.get("benchmark_sample_mode", "head"),
            sample_seed=config.get("benchmark_sample_seed", 22),
            sample_strata=config.get("benchmark_sample_strata", "difficulty"),
            batch_size=config.get("benchmark_batch_size", 8),
            rtol=config.get("benchmark_rtol", 1e-2),
            atol=config.get("benchmark_atol", 1e-3),
            models=_model_names,
            exec_timeout=config.get("benchmark_exec_timeout", 60),
            exec_hard_timeout=config.get("benchmark_exec_hard_timeout", 75),
            max_hard_timeout_fix_attempts=config.get(
                "benchmark_max_hard_timeout_fix_attempts", 1
            ),
            search_engine=None,
            max_new_tokens=config.get("benchmark_max_new_tokens", 4096),
            fix_max_new_tokens=config.get("benchmark_fix_max_new_tokens", 2048),
            exec_workers=config.get("benchmark_exec_workers", 1),
        )
        return

    if run_action in ("resume_benchmark_fix", "resume_benchmark_fx"):
        comp.resume_benchmark_fix(
            benchmark_path=benchmark_path,
            output_dir=output_dir,
            fix_rounds=benchmark_fix_rounds,
            max_items=config.get("benchmark_max_items"),
            sample_mode=config.get("benchmark_sample_mode", "head"),
            sample_seed=config.get("benchmark_sample_seed", 22),
            sample_strata=config.get("benchmark_sample_strata", "difficulty"),
            batch_size=config.get("benchmark_batch_size", 8),
            rtol=config.get("benchmark_rtol", 1e-2),
            atol=config.get("benchmark_atol", 1e-3),
            models=_model_names,
            exec_timeout=config.get("benchmark_exec_timeout", 60),
            exec_hard_timeout=config.get("benchmark_exec_hard_timeout", 75),
            max_hard_timeout_fix_attempts=config.get(
                "benchmark_max_hard_timeout_fix_attempts", 1
            ),
            max_new_tokens=config.get("benchmark_max_new_tokens", 4096),
            fix_max_new_tokens=config.get("benchmark_fix_max_new_tokens", 2048),
            exec_workers=config.get("benchmark_exec_workers", 1),
        )
        return

    if run_action == "rerun_proactive":
        comp.rerun_proactive(
            benchmark_path=benchmark_path,
            proactive_demand_suite=config.get("proactive_demand_suite"),
            proactive_demand_model=config.get("proactive_demand_model", "hybrid_tfidf"),
            proactive_demand_export=config.get("proactive_demand_export"),
            conditions=proactive_conditions or ["C", "X"],
            output_dir=output_dir,
            max_items=config.get("benchmark_max_items"),
            sample_mode=config.get("benchmark_sample_mode", "head"),
            sample_seed=config.get("benchmark_sample_seed", 22),
            sample_strata=config.get("benchmark_sample_strata", "difficulty"),
            batch_size=config.get("benchmark_batch_size", 8),
            rtol=config.get("benchmark_rtol", 1e-2),
            atol=config.get("benchmark_atol", 1e-3),
            models=_model_names,
            proactive_docs_path=proactive_docs_path,
            proactive_top_k=config.get("proactive_top_k", 10),
            proactive_token_budget_C=config.get("proactive_token_budget_C", 2000),
            proactive_token_budget_X=config.get("proactive_token_budget_X", 4000),
            proactive_token_budget_RsemB=config.get("proactive_token_budget_RsemB", 800),
            proactive_threshold=config.get("proactive_threshold", 0.5),
            risk_weights=config.get("risk_weights"),
            bm25_docs_path=config.get("bm25_docs_path"),
            sbert_docs_path=config.get("sbert_docs_path"),
            risk_uniform=config.get("risk_uniform", False),
            recal_name_layer_floor=config.get("recal_name_layer_floor", 0.0),
            exec_timeout=config.get("benchmark_exec_timeout", 60),
            exec_hard_timeout=config.get("benchmark_exec_hard_timeout", 75),
            max_hard_timeout_fix_attempts=config.get(
                "benchmark_max_hard_timeout_fix_attempts", 1
            ),
            profile_source_dir=config.get("profile_source_dir"),
            max_new_tokens=config.get("benchmark_max_new_tokens", 4096),
            exec_workers=config.get("benchmark_exec_workers", 1),
        )
        return

    if run_action == "rerun_reactive":
        comp.rerun_reactive(
            benchmark_path=benchmark_path,
            combinations=reactive_combinations,
            output_dir=output_dir,
            fix_rounds=config.get("reactive_fix_rounds", 3),
            max_items=config.get("benchmark_max_items"),
            sample_mode=config.get("benchmark_sample_mode", "head"),
            sample_seed=config.get("benchmark_sample_seed", 22),
            sample_strata=config.get("benchmark_sample_strata", "difficulty"),
            batch_size=config.get("benchmark_batch_size", 8),
            rtol=config.get("benchmark_rtol", 1e-2),
            atol=config.get("benchmark_atol", 1e-3),
            models=_model_names,
            proactive_demand_suite=config.get("proactive_demand_suite"),
            proactive_demand_model=config.get("proactive_demand_model", "hybrid_tfidf"),
            proactive_demand_export=config.get("proactive_demand_export"),
            proactive_docs_path=proactive_docs_path,
            reactive_docs_path=reactive_docs_path,
            proactive_top_k=config.get("proactive_top_k", 10),
            proactive_token_budget_C=config.get("proactive_token_budget_C", 2000),
            proactive_token_budget_X=config.get("proactive_token_budget_X", 4000),
            proactive_threshold=config.get("proactive_threshold", 0.5),
            risk_weights=config.get("risk_weights"),
            bm25_docs_path=config.get("bm25_docs_path"),
            risk_uniform=config.get("risk_uniform", False),
            reactive_max_funcs=config.get("reactive_max_funcs", 3),
            exec_timeout=config.get("benchmark_exec_timeout", 60),
            exec_hard_timeout=config.get("benchmark_exec_hard_timeout", 75),
            max_hard_timeout_fix_attempts=config.get(
                "benchmark_max_hard_timeout_fix_attempts", 1
            ),
            profile_source_dir=config.get("profile_source_dir"),
            max_new_tokens=config.get("benchmark_max_new_tokens", 4096),
            fix_max_new_tokens=config.get("benchmark_fix_max_new_tokens", 2048),
            exec_workers=config.get("benchmark_exec_workers", 1),
        )
        return

    # ── Main: run probes + standard benchmark + proactive + reactive (resume-capable) ───
    comp.compare(
        output_dir=output_dir,
        benchmark_path=benchmark_path,
        benchmark_max_items=config.get("benchmark_max_items"),
        benchmark_batch_size=config.get("benchmark_batch_size", 8),
        benchmark_sample_mode=config.get("benchmark_sample_mode", "head"),
        benchmark_sample_seed=config.get("benchmark_sample_seed", 22),
        benchmark_sample_strata=config.get("benchmark_sample_strata", "difficulty"),
        benchmark_rtol=config.get("benchmark_rtol", 1e-2),
        benchmark_atol=config.get("benchmark_atol", 1e-3),
        benchmark_fix_rounds=benchmark_fix_rounds,
        benchmark_timeout=config.get("benchmark_exec_timeout", 60),
        benchmark_exec_hard_timeout=config.get("benchmark_exec_hard_timeout", 75),
        benchmark_max_hard_timeout_fix_attempts=config.get(
            "benchmark_max_hard_timeout_fix_attempts", 1
        ),
        benchmark_max_new_tokens=config.get("benchmark_max_new_tokens", 4096),
        benchmark_fix_max_new_tokens=config.get("benchmark_fix_max_new_tokens", 2048),
        benchmark_exec_workers=config.get("benchmark_exec_workers", 1),
        search_engine=search_engine,
        proactive_conditions=proactive_conditions,
        proactive_demand_suite=config.get("proactive_demand_suite"),
        proactive_demand_model=config.get("proactive_demand_model", "hybrid_tfidf"),
        proactive_demand_export=config.get("proactive_demand_export"),   # legacy fallback
        proactive_docs_path=proactive_docs_path,
        proactive_top_k=config.get("proactive_top_k", 10),
        proactive_token_budget_C=config.get("proactive_token_budget_C", 2000),
        proactive_token_budget_X=config.get("proactive_token_budget_X", 4000),
        proactive_threshold=config.get("proactive_threshold", 0.5),
            reactive_combinations=reactive_combinations,
        reactive_fix_rounds=config.get("reactive_fix_rounds", 3),
        reactive_docs_path=reactive_docs_path,
        reactive_max_funcs=config.get("reactive_max_funcs", 3),
    )


# ============================================================================
# CONFIG
# ============================================================================
CONFIG = {
    # LLM backend
    "llm_backend": "vllm",

    # HuggingFace
    "hf_token": os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACEHUB_API_TOKEN", ""),

    # Library docs (used as default for proactive/reactive injectors)
    "tool_library_path": "dataset/pandapower_docs.json",

    # Benchmark
    "benchmark_path": "benchmark.json",
    "benchmark_max_items": None,
    "benchmark_batch_size": 8,
    "benchmark_sample_mode": "head",
    "benchmark_sample_seed": 22,
    "benchmark_sample_strata": "difficulty",
    "benchmark_rtol": 1e-2,
    "benchmark_atol": 1e-3,
    "generation_seed": 22,
    "probe_temperature": 0.0,
    "benchmark_generation_temperature": 0.0,
    "benchmark_fix_temperature": 0.0,
    "benchmark_fix_rounds": 3,
    "benchmark_exec_timeout"      : 60,    # per-item execution timeout (seconds)
    "benchmark_exec_hard_timeout" : 75,   # parent-side liveness fuse; <=0 disables
    "benchmark_max_hard_timeout_fix_attempts": 1,  # avoid repeated stuck-item fix loops
    "benchmark_max_new_tokens"    : 4096,  # max tokens for round-0 generation
    "benchmark_fix_max_new_tokens": 2048,  # max tokens for fix-round generation (shorter code)
    "benchmark_exec_workers"      : 16,     # parallel code execution workers (ProcessPoolExecutor)
    "output_dir": None,
    "profile_source_dir": None,

    # Proactive Compensation (uncomment to enable; do NOT include "A" in conditions)
    # Demand export is model-agnostic — one suite file serves all target models.
    # Per-model knowledge differences come from probe results (knowledge_profile.json).
    "proactive_conditions"       : ["B", "C", "X"],
    "proactive_demand_suite"     : "results/demand_suites/suite_4428033",
    "proactive_demand_model"     : "hybrid_tfidf",            # subdir under suite
    "proactive_demand_export"    : None,
    "proactive_docs_path"        : "dataset/pandapower_docs.json",   # default: tool_library_path
    "proactive_top_k"            : 10,
    "proactive_token_budget_C"   : 2000,
    "proactive_token_budget_X"   : 4000,
    "proactive_threshold"        : 0.5,

    # Reactive Injection (uncomment to enable)
    # Each entry is "P_F": proactive base P (A=none, C, X) + fix condition F.
    # FX = plain fix; FD = legacy always-doc reactive; FDR = fix-demand routed reactive.
    # FS/FDRS are semantic value-bug ablations and must be requested explicitly.
    # C+FDR is the low-cost candidate; C+FD remains the always-doc baseline.
    # Main default is final-paper reactive path; pass A_* or X_* via env for ablations.
    "reactive_combinations"      : ["C_FX", "C_FD", "C_FDR"],
    "reactive_fix_rounds"        : 3,
    "reactive_docs_path"         : "dataset/pandapower_docs.json",
    "reactive_max_funcs"         : 3,

    # vLLM engine tuning
    "vllm_max_num_seqs"           : 256,   # max concurrent sequences in vLLM scheduler
    "vllm_gpu_memory_utilization" : 0.90,  # fraction of GPU memory for KV cache

    # Quantization (for HF backend)
    "load_in_4bit": False,
    "load_in_8bit": False,
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}

 
MODEL_PATCHES: Dict[str, Dict] = {
    "Qwen/Qwen2.5-Coder-0.5B-Instruct": {"gpus": 1, "max_seq_len": 16000},
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": {"gpus": 1, "max_seq_len": 16000},
    "Qwen/Qwen2.5-Coder-7B-Instruct":   {"gpus": 1, "max_seq_len": 16000},
    "Qwen/Qwen2.5-Coder-14B-Instruct":  {"gpus": 2, "max_seq_len": 16000},
    "Qwen/Qwen2.5-Coder-32B-Instruct":  {"gpus": 2, "max_seq_len": 16000},
    "Qwen/Qwen2.5-Coder-72B-Instruct":  {"gpus": 2, "max_seq_len": 16000},
    "Qwen/Qwen3-Coder-Next":            {"gpus": 2, "max_seq_len": 16000},
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": {
        "gpus": 16, "max_seq_len": 20000, "enable_expert_parallel": True,
    },
    "meta-llama/Llama-3.1-8B-Instruct":   {"gpus": 1, "max_seq_len": 16000},
    "meta-llama/Llama-3.1-70B-Instruct":   {"gpus": 2, "max_seq_len": 16000},
    "meta-llama/Llama-3.3-70B-Instruct":   {"gpus": 2, "max_seq_len": 16000},
    "meta-llama/Llama-3.1-405B-Instruct":  {"gpus": 16, "max_seq_len": 16000},
    "hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4": {
        "gpus": 8, "max_seq_len": 16000,
    },
    "deepseek-ai/deepseek-coder-6.7b-instruct": {"gpus": 1, "max_seq_len": 8192},
    "openai/gpt-oss-120b": {"gpus": 8, "max_seq_len": 20000},
    "deepseek-v4-flash": {
        "_use_gpt": True,
        "gpt_model": "deepseek-v4-flash",
        "gpt_base_url": "https://api.deepseek.com",
        "gpt_api_key_name": "deepseek_api_key",
        "gpt_concurrency": 20,
    },
    "deepseek-v4-pro": {
        "_use_gpt": True,
        "gpt_model": "deepseek-v4-pro",
        "gpt_base_url": "https://api.deepseek.com",
        "gpt_api_key_name": "deepseek_api_key",
        "gpt_concurrency": 20,
    },
    "gpt-4.1": {
        "_use_gpt": True,
        "gpt_model": "gpt-4.1-2025-04-14",
        "gpt_concurrency": 20,
    },
    "gpt-4.1-mini": {
        "_use_gpt": True,
        "gpt_model": "gpt-4.1-mini-2025-04-14",
        "gpt_concurrency": 20,
    },
    "gpt-4o": {"_use_gpt": True, "gpt_model": "gpt-4o", "gpt_concurrency": 20},
    "gpt-4o-mini": {"_use_gpt": True, "gpt_model": "gpt-4o-mini", "gpt_concurrency": 20},
    "claude-sonnet-4.5": {
        "_use_anthropic": True,
        "anthropic_model": "claude-sonnet-4-5-20250929",
        "max_seq_len": 8192,
    },
    "gemini-2.5-pro": {
        "_use_gemini": True,
        "gemini_model": "gemini-2.5-pro",
        "max_seq_len": 8192,
    },
}

def main(config: Dict = CONFIG):
    import os

    # ── Apply environment variable overrides (set by run_multinode_job.sh) ──
    config = dict(config)  # shallow copy to avoid mutating module-level CONFIG

    # PROACTIVE_CONDITIONS: comma-separated string → list, e.g. "B,C,X" → ["B","C","X"]
    conds_str = os.environ.get("PROACTIVE_CONDITIONS", "").strip()
    if conds_str:
        config["proactive_conditions"] = [c.strip() for c in conds_str.split(",") if c.strip()]
        print(f"  [env override] proactive_conditions = {config['proactive_conditions']}")

    demand_suite_env = os.environ.get("PROACTIVE_DEMAND_SUITE", "").strip()
    if demand_suite_env:
        config["proactive_demand_suite"] = demand_suite_env
        print(f"  [env override] proactive_demand_suite = {demand_suite_env}")

    demand_model_env = os.environ.get("PROACTIVE_DEMAND_MODEL", "").strip()
    if demand_model_env:
        config["proactive_demand_model"] = demand_model_env
        print(f"  [env override] proactive_demand_model = {demand_model_env}")

    demand_export_env = os.environ.get("PROACTIVE_DEMAND_EXPORT", "").strip()
    if demand_export_env:
        config["proactive_demand_export"] = demand_export_env
        print(f"  [env override] proactive_demand_export = {demand_export_env}")

    proactive_docs_env = os.environ.get("PROACTIVE_DOCS_PATH", "").strip()
    if proactive_docs_env:
        config["proactive_docs_path"] = proactive_docs_env
        print(f"  [env override] proactive_docs_path = {proactive_docs_env}")

    reactive_docs_env = os.environ.get("REACTIVE_DOCS_PATH", "").strip()
    if reactive_docs_env:
        config["reactive_docs_path"] = reactive_docs_env
        print(f"  [env override] reactive_docs_path = {reactive_docs_env}")

    benchmark_path_env = os.environ.get("BENCHMARK_PATH", "").strip()
    if benchmark_path_env:
        config["benchmark_path"] = benchmark_path_env
        print(f"  [env override] benchmark_path = {benchmark_path_env}")

    output_dir_env = os.environ.get("OUTPUT_DIR", "").strip()
    if output_dir_env:
        config["output_dir"] = output_dir_env
        print(f"  [env override] output_dir = {output_dir_env}")

    sample_mode_env = os.environ.get("BENCHMARK_SAMPLE_MODE", "").strip()
    if sample_mode_env:
        config["benchmark_sample_mode"] = sample_mode_env
        print(f"  [env override] benchmark_sample_mode = {sample_mode_env}")

    sample_strata_env = os.environ.get("BENCHMARK_SAMPLE_STRATA", "").strip()
    if sample_strata_env:
        config["benchmark_sample_strata"] = sample_strata_env
        print(f"  [env override] benchmark_sample_strata = {sample_strata_env}")

    profile_source_env = os.environ.get("PROFILE_SOURCE_DIR", "").strip()
    if profile_source_env:
        config["profile_source_dir"] = profile_source_env
        print(f"  [env override] profile_source_dir = {profile_source_env}")

    # PROBES_PATH / API_SPEC_PATH: swap the probe question set and the backend
    # spec (fake-name pool + probe evaluator) for cross-backend transfer (E2).
    # Unset → pandapower defaults, so the main-pipeline behaviour is unchanged.
    probes_path_env = os.environ.get("PROBES_PATH", "").strip()
    if probes_path_env:
        config["probes_path"] = probes_path_env
        print(f"  [env override] probes_path = {probes_path_env}")

    api_spec_env = os.environ.get("API_SPEC_PATH", "").strip()
    if api_spec_env:
        config["api_spec_path"] = api_spec_env
        print(f"  [env override] api_spec_path = {api_spec_env}")

    bm25_docs_env = os.environ.get("BM25_DOCS_PATH", "").strip()
    if bm25_docs_env:
        config["bm25_docs_path"] = bm25_docs_env
        print(f"  [env override] bm25_docs_path = {bm25_docs_env}")

    sbert_docs_env = os.environ.get("SBERT_DOCS_PATH", "").strip()
    if sbert_docs_env:
        config["sbert_docs_path"] = sbert_docs_env
        print(f"  [env override] sbert_docs_path = {sbert_docs_env}")

    # BENCHMARK_LIBRARY: library named in the benchmark system prompts (read directly by
    # utils.benchmark_library_name(); echoed here as the run-time verification point).
    # Unset = pandapower = unchanged main-line behaviour.
    benchmark_library_env = os.environ.get("BENCHMARK_LIBRARY", "").strip()
    if benchmark_library_env:
        print(f"  [env override] benchmark_library = {benchmark_library_env}")

    # PROACTIVE_RISK_UNIFORM: when set to "1"/"true"/"yes", replaces the
    # model-specific layer risk r_{M,l}(f) with a constant 1.0 throughout
    # the proactive scoring path (Round 5 Exp-A: C_risk_uniform isolation
    # ablation).  Demand model, layer pruning, token budget, anchors, and
    # boundary cards remain unchanged; only the model-side probe profile is
    # bypassed.
    risk_uniform_env = os.environ.get("PROACTIVE_RISK_UNIFORM", "").strip().lower()
    if risk_uniform_env in {"1", "true", "yes", "on"}:
        config["risk_uniform"] = True
        print(f"  [env override] risk_uniform = True")

    # RECAL_NAME_LAYER_FLOOR: float in (0, 1]. E2-R recalibration arm (E2 doc
    # §8 prereg 2026-08-02): floors condition C's L0/L1 layer deficit for
    # demand-hit functions so the name/signature layers always enter packing
    # competition. Unset (default) = exact frozen condition-C behaviour.
    recal_floor_env = os.environ.get("RECAL_NAME_LAYER_FLOOR", "").strip()
    if recal_floor_env:
        try:
            recal_floor = float(recal_floor_env)
            if not 0.0 < recal_floor <= 1.0:
                raise ValueError(f"need a value in (0, 1], got {recal_floor}")
            config["recal_name_layer_floor"] = recal_floor
            print(f"  [env override] recal_name_layer_floor = {recal_floor}")
        except ValueError as e:
            print(f"  [env override] RECAL_NAME_LAYER_FLOOR={recal_floor_env!r} invalid ({e}), ignoring")

    # RISK_WEIGHTS: 4 floats for L0,L1,L2,L3.
    # Accepts comma- OR underscore-separated values; the underscore form is
    # used when invoking via sbatch --export, which uses comma as its own
    # delimiter. Both "0.25,0.25,0.25,0.25" and "0.25_0.25_0.25_0.25" parse.
    risk_weights_env = os.environ.get("RISK_WEIGHTS", "").strip()
    if risk_weights_env:
        try:
            sep = "_" if "_" in risk_weights_env and "," not in risk_weights_env else ","
            parts = [float(p) for p in risk_weights_env.split(sep)]
            if len(parts) != 4:
                raise ValueError(f"need exactly 4 values, got {len(parts)}")
            config["risk_weights"] = dict(zip(["L0", "L1", "L2", "L3"], parts))
            print(f"  [env override] risk_weights = {config['risk_weights']}")
        except ValueError as e:
            print(f"  [env override] RISK_WEIGHTS={risk_weights_env!r} invalid ({e}), ignoring")

    # VLLM_MAX_NUM_SEQS: integer, max concurrent sequences in the vLLM
    # scheduler (long-prompt conditions on dense 405B need a lower cap).
    vms_env = os.environ.get("VLLM_MAX_NUM_SEQS", "").strip()
    if vms_env:
        try:
            config["vllm_max_num_seqs"] = int(vms_env)
            print(f"  [env override] vllm_max_num_seqs = {config['vllm_max_num_seqs']}")
        except ValueError:
            print(f"  [env override] VLLM_MAX_NUM_SEQS={vms_env!r} invalid, ignoring")

    # REACTIVE_COMBINATIONS: comma-separated, e.g. "C_FX,C_FD,C_FDR"
    reactive_combs_env = os.environ.get("REACTIVE_COMBINATIONS", "").strip()
    if reactive_combs_env:
        config["reactive_combinations"] = [c.strip() for c in reactive_combs_env.split(",") if c.strip()]
        print(f"  [env override] reactive_combinations = {config['reactive_combinations']}")

    # REACTIVE_FIX_ROUNDS: integer, e.g. "3"
    reactive_rounds_env = os.environ.get("REACTIVE_FIX_ROUNDS", "").strip()
    if reactive_rounds_env:
        try:
            config["reactive_fix_rounds"] = int(reactive_rounds_env)
            print(f"  [env override] reactive_fix_rounds = {config['reactive_fix_rounds']}")
        except ValueError:
            print(f"  [env override] REACTIVE_FIX_ROUNDS={reactive_rounds_env!r} is not an integer, ignoring")

    # BENCHMARK_MAX_NEW_TOKENS / BENCHMARK_FIX_MAX_NEW_TOKENS: integers
    for env_key, cfg_key in [
        ("BENCHMARK_MAX_NEW_TOKENS",     "benchmark_max_new_tokens"),
        ("BENCHMARK_FIX_MAX_NEW_TOKENS", "benchmark_fix_max_new_tokens"),
        ("BENCHMARK_FIX_ROUNDS",         "benchmark_fix_rounds"),
        ("BENCHMARK_EXEC_TIMEOUT",       "benchmark_exec_timeout"),
        ("BENCHMARK_EXEC_HARD_TIMEOUT",  "benchmark_exec_hard_timeout"),
        ("BENCHMARK_MAX_HARD_TIMEOUT_FIX_ATTEMPTS",
                                            "benchmark_max_hard_timeout_fix_attempts"),
        ("BENCHMARK_EXEC_WORKERS",       "benchmark_exec_workers"),
        ("BENCHMARK_BATCH_SIZE",         "benchmark_batch_size"),
        ("BENCHMARK_MAX_ITEMS",          "benchmark_max_items"),
        ("BENCHMARK_SAMPLE_SEED",        "benchmark_sample_seed"),
        ("GENERATION_SEED",              "generation_seed"),
        ("PROACTIVE_TOP_K",              "proactive_top_k"),
        ("PROACTIVE_TOKEN_BUDGET_C",      "proactive_token_budget_C"),
        ("PROACTIVE_TOKEN_BUDGET_X",      "proactive_token_budget_X"),
        ("PROACTIVE_TOKEN_BUDGET_RSEMB",  "proactive_token_budget_RsemB"),
        ("REACTIVE_MAX_FUNCS",           "reactive_max_funcs"),
    ]:
        env_val = os.environ.get(env_key, "").strip()
        if env_val:
            try:
                config[cfg_key] = int(env_val)
                print(f"  [env override] {cfg_key} = {config[cfg_key]}")
            except ValueError:
                print(f"  [env override] {env_key}={env_val!r} is not an integer, ignoring")

    for env_key, cfg_key in [
        ("PROBE_TEMPERATURE",                 "probe_temperature"),
        ("BENCHMARK_GENERATION_TEMPERATURE", "benchmark_generation_temperature"),
        ("BENCHMARK_FIX_TEMPERATURE",        "benchmark_fix_temperature"),
        ("BENCHMARK_RTOL",                   "benchmark_rtol"),
        ("BENCHMARK_ATOL",                   "benchmark_atol"),
        ("PROACTIVE_THRESHOLD",              "proactive_threshold"),
    ]:
        env_val = os.environ.get(env_key, "").strip()
        if env_val:
            try:
                config[cfg_key] = float(env_val)
                print(f"  [env override] {cfg_key} = {config[cfg_key]:g}")
            except ValueError:
                print(f"  [env override] {env_key}={env_val!r} is not a float, ignoring")

    if config.get('hf_token'):
        login(token=config['hf_token'])

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                  f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")

    print("\n" + "=" * 60)
    print("📊 Running Evaluation Pipeline")
    print("=" * 60)
    run_probe_evaluation(
        config,
        probes_path=config.get("probes_path", "dataset/all_probes.json"),
        api_spec_path=config.get("api_spec_path", "dataset/pandapower_docs.json"),
        benchmark_path=config.get("benchmark_path", "benchmark.json"),
        benchmark_fix_rounds=config.get("benchmark_fix_rounds", 0),
    )


if __name__ == "__main__":
    main(CONFIG)
