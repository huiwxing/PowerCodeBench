#!/usr/bin/env python3
"""Export and aggregate the S2/S3 per-function knowledge profiles.

The compact source preserves the complete model x function x layer object,
including probe counts, by-probe-type fields and diagnostics.  It is therefore
lossless for the uniform and mild-L3 risk calculations used by the original
``e6_probe_auroc.py``; only pretty-printing and unreported Qwen-0.5B data are
omitted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evidence_common import read_json, repo_descriptor, source_descriptor, write_json

HERE = Path(__file__).resolve()
PCB_ROOT = HERE.parents[2]
DEFAULT_SOURCE = PCB_ROOT.parent / "igpt"
DEFAULT_ARTIFACT = PCB_ROOT / "results/supplementary_evidence/probe_profiles.json"
DEFAULT_OUTPUT = PCB_ROOT / "results/aggregates/supplementary_probe_profiles.json"

MODELS = {
    "Qwen2.5-1.5B": ("comparison", "Qwen_Qwen2.5-Coder-1.5B-Instruct"),
    "Llama-8B": ("comparison", "meta-llama_Llama-3.1-8B-Instruct"),
    "Qwen2.5-7B": ("comparison", "Qwen_Qwen2.5-Coder-7B-Instruct"),
    "Qwen2.5-14B": ("comparison", "Qwen_Qwen2.5-Coder-14B-Instruct"),
    "Qwen2.5-32B": ("comparison", "Qwen_Qwen2.5-Coder-32B-Instruct"),
    "Qwen3-Next": ("comparison", "Qwen_Qwen3-Coder-Next"),
    "Llama-70B": ("comparison", "meta-llama_Llama-3.1-70B-Instruct"),
    "GPT-OSS-120B": ("comparison", "openai_gpt-oss-120b"),
    "Llama-405B": ("comparison", "meta-llama_Llama-3.1-405B-Instruct"),
    "Qwen3-480B": ("comparison", "Qwen_Qwen3-Coder-480B-A35B-Instruct"),
    "Gemini-2.5-Flash": ("api_comparison_2000", "gemini-2.5-flash"),
    "DeepSeek-V4-Flash": ("api_comparison_2000", "deepseek-v4-flash"),
    "Claude-Haiku-4-5": ("api_comparison_2000", "claude-haiku-4-5"),
    "GPT-5.4-mini": ("api_comparison_2000", "gpt-5.4-mini"),
}
LAYERS = ("L0", "L1", "L2", "L3")
WEIGHTS = {
    "uniform": {"L0": 0.25, "L1": 0.25, "L2": 0.25, "L3": 0.25},
    "mild_l3": {"L0": 0.20, "L1": 0.25, "L2": 0.20, "L3": 0.35},
}


def export(source_root: Path, artifact: Path) -> None:
    files = []
    models = {}
    for label, (panel, source_model) in MODELS.items():
        path = source_root / "probe_eval_results" / panel / source_model / "knowledge_profile.json"
        profile = read_json(path)
        if len(profile) != 275:
            raise ValueError(f"{path}: expected 275 functions, found {len(profile)}")
        missing = [f"{fn}:{layer}" for fn, data in profile.items()
                   for layer in LAYERS if layer not in data]
        if missing:
            raise ValueError(f"{path}: missing layer data, first={missing[0]}")
        files.append(source_descriptor(source_root, path))
        models[label] = {"source_model": source_model, "profile": profile}
    out = {
        "schema_version": 1,
        "evidence_unit": "model x API function; complete L0-L3 profile objects",
        "exclusions": [
            "Qwen2.5-Coder-0.5B is not in the manuscript S2 full panel",
        ],
        "provenance": {**repo_descriptor(source_root), "files": files},
        "models": models,
    }
    write_json(artifact, out)
    print(f"exported {len(models)} profiles ({artifact.stat().st_size} bytes) -> {artifact}")


def l3_score(layer: dict) -> float:
    diag = layer.get("diagnostics") or {}
    es = diag.get("execution_success")
    tu = diag.get("target_function_used")
    if isinstance(es, (int, float)) and isinstance(tu, (int, float)):
        return float(es) * float(tu)
    return float(layer["score"])


def aggregate(artifact: Path, output: Path) -> None:
    source = read_json(artifact)
    table = {}
    risk_summary = {}
    protocol_counts = None
    for label, model in source["models"].items():
        profile = model["profile"]
        this_protocol = {}
        for layer in LAYERS:
            layer_data = [value[layer] for value in profile.values()]
            by_type = {}
            for value in layer_data:
                for probe_type, entry in (value.get("by_probe_type") or {}).items():
                    by_type[probe_type] = by_type.get(probe_type, 0) + int(entry["n_probes"])
            this_protocol[layer] = {
                "n_probes": sum(int(value["n_probes"]) for value in layer_data),
                "by_probe_type": dict(sorted(by_type.items())),
            }
        if protocol_counts is None:
            protocol_counts = this_protocol
        elif this_protocol != protocol_counts:
            raise ValueError(f"probe protocol counts differ for {label}")
        means = {layer: sum(float(v[layer]["score"]) for v in profile.values()) / len(profile)
                 for layer in LAYERS}
        table[label] = {layer: round(means[layer], 6) for layer in LAYERS}
        variants = {}
        for variant, weights in WEIGHTS.items():
            per_function = {}
            for fn, layers in profile.items():
                scores = {layer: (l3_score(layers[layer]) if layer == "L3"
                                  else float(layers[layer]["score"])) for layer in LAYERS}
                per_function[fn] = sum(weights[layer] * (1.0 - scores[layer]) for layer in LAYERS)
            variants[variant] = {
                "mean_risk": round(sum(per_function.values()) / len(per_function), 6),
                "per_function": {fn: round(value, 9) for fn, value in per_function.items()},
            }
        risk_summary[label] = variants
    out = {
        "source_artifact": artifact.relative_to(PCB_ROOT).as_posix(),
        "definitions": {"weights": WEIGHTS, "L3_with_diagnostics": "execution_success * target_function_used"},
        "probe_protocol_counts": protocol_counts,
        "mean_probe_scores": table,
        "risk_profiles": risk_summary,
    }
    write_json(output, out)
    print(f"rebuilt S2 table and uniform/mild-L3 profiles -> {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.export:
        export(args.source_root.resolve(), args.artifact.resolve())
    aggregate(args.artifact.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
