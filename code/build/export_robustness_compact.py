#!/usr/bin/env python3
"""Export the minimal per-item inputs for robustness/reasoning aggregators.

This migration helper is intentionally separate from the public aggregators.  It
reads the tracked, full experiment payloads in the original ``igpt`` repository
and writes privacy-neutral compact records: item identity, outcome, and only the
token/sample metadata needed by the corresponding frozen analysis.  Every cell
records the SHA-256, byte count, relative path, and source commit of its full raw
input, so the lossy field projection remains auditable against the archive.

The main comparison tree is being exported by the main-panel migration route and
is deliberately not duplicated here.  This helper exports only:

* the 10-model x 10-seed C+FDRS generation-seed sweep;
* the three alternate risk-weight grids (uniform, L3-heavy, L0-heavy);
* all 24 E6 gate/budget cells; and
* the ten 200-item reasoning-on cells.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SEED_COMMIT = "3841e89ed03cef19d388134c6e2d76ebba4e425c"
REVISION_COMMIT = "6990ac025ad3b7eb26d9767c5cc470eaef02374f"

PANEL = [
    "Qwen_Qwen2.5-Coder-1.5B-Instruct",
    "Qwen_Qwen2.5-Coder-7B-Instruct",
    "meta-llama_Llama-3.1-8B-Instruct",
    "Qwen_Qwen2.5-Coder-14B-Instruct",
    "Qwen_Qwen2.5-Coder-32B-Instruct",
    "meta-llama_Llama-3.1-70B-Instruct",
    "openai_gpt-oss-120b",
    "Qwen_Qwen3-Coder-Next",
    "meta-llama_Llama-3.1-405B-Instruct",
    "Qwen_Qwen3-Coder-480B-A35B-Instruct",
]
SEEDS = [22, 33, 44, 55, 66, 77, 88, 99, 111, 222]
RISK_MODELS = [
    "Qwen_Qwen2.5-Coder-14B-Instruct",
    "Qwen_Qwen2.5-Coder-32B-Instruct",
    "meta-llama_Llama-3.1-70B-Instruct",
    "openai_gpt-oss-120b",
    "Qwen_Qwen3-Coder-480B-A35B-Instruct",
]
E6_MODELS = [
    "Qwen_Qwen2.5-Coder-14B-Instruct",
    "meta-llama_Llama-3.1-70B-Instruct",
    "Qwen_Qwen3-Coder-480B-A35B-Instruct",
]
REASONING_MODELS = ["gpt-5.4-mini", "deepseek-reasoner"]
REASONING_CONDITIONS = ["A", "C", "C_FX", "C_FDR", "C_FDRS"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def provenance(source_root: Path, path: Path, commit: str) -> dict:
    return {
        "source_repository": "frozen experimental pipeline",
        "source_commit": commit,
        "source_path": path.relative_to(source_root).as_posix(),
        "source_bytes": path.stat().st_size,
        "source_sha256": sha256(path),
        "projection": "lossy minimal-field export from the byte-hashed full raw JSON",
    }


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def checked_summary(data: dict, path: Path) -> dict:
    items = data.get("item_results") or []
    summary = data.get("summary") or {}
    n = len(items)
    matched = sum(bool(item.get("match")) for item in items)
    if summary.get("total") != n or summary.get("n_matched") != matched:
        raise ValueError(f"summary/item mismatch in {path}")
    return {
        "total": n,
        "n_matched": matched,
        "result_accuracy": matched / n if n else None,
    }


def write_compact(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}


def export_seed_sweep(source_root: Path, output_root: Path) -> list[dict]:
    records = []
    for seed in SEEDS:
        for model in PANEL:
            rel = Path("probe_eval_results/seed_sweep_400") / f"seed{seed}" / model / "benchmark_results_condC_FDRS.json"
            src = source_root / rel
            data = load(src)
            summary = checked_summary(data, src)
            payload = {
                "schema_version": "1.0",
                "provenance": provenance(source_root, src, SEED_COMMIT),
                "summary": summary,
                "item_results": [
                    {"item_id": item["item_id"], "match": bool(item.get("match"))}
                    for item in data["item_results"]
                ],
            }
            dst = output_root / rel.relative_to("probe_eval_results")
            rec = write_compact(dst, payload)
            rec.update({"experiment": "generation_seed_sweep", "model": model, "seed": seed})
            records.append(rec)
    return records


def export_risk_grid(source_root: Path, output_root: Path) -> list[dict]:
    records = []
    for weighting in ("uniform", "l3heavy", "l0heavy"):
        tree = f"comparison_riskw_{weighting}"
        for model in RISK_MODELS:
            rel = Path("probe_eval_results") / tree / model / "benchmark_results_condC.json"
            src = source_root / rel
            data = load(src)
            summary = checked_summary(data, src)
            seen = set()
            items = []
            for item in data["item_results"]:
                idx = item["bench_index"]
                if idx in seen:
                    raise ValueError(f"duplicate bench_index {idx} in {src}")
                seen.add(idx)
                items.append({
                    "bench_index": idx,
                    "item_id": item["item_id"],
                    "match": bool(item.get("match")),
                })
            payload = {
                "schema_version": "1.0",
                "provenance": provenance(source_root, src, REVISION_COMMIT),
                "summary": summary,
                "item_results": items,
            }
            dst = output_root / rel.relative_to("probe_eval_results")
            rec = write_compact(dst, payload)
            rec.update({"experiment": "risk_weight_grid", "model": model, "weighting": weighting})
            records.append(rec)
    return records


def export_e6(source_root: Path, output_root: Path) -> list[dict]:
    records = []
    for budget in (600, 800, 1000, 2000):
        for arm in ("on", "off"):
            for model in E6_MODELS:
                rel = (Path("probe_eval_results/e6_budget") / f"{budget}_gate-{arm}" /
                       model / "benchmark_results_condC.json")
                src = source_root / rel
                data = load(src)
                summary = checked_summary(data, src)
                summary["prompt_token_stats"] = data["summary"].get("prompt_token_stats")
                bs = data["config"]["benchmark_sample"]
                sample = {
                    key: bs.get(key)
                    for key in ("mode", "seed", "strata", "max_items", "selected_indices")
                }
                payload = {
                    "schema_version": "1.0",
                    "provenance": provenance(source_root, src, REVISION_COMMIT),
                    "summary": summary,
                    "config": {"benchmark_sample": sample},
                    "item_results": [
                        {
                            "bench_index": item["bench_index"],
                            "item_id": item["item_id"],
                            "match": bool(item.get("match")),
                            "prompt_tokens": item.get("prompt_tokens"),
                        }
                        for item in data["item_results"]
                    ],
                }
                dst = output_root / rel.relative_to("probe_eval_results")
                rec = write_compact(dst, payload)
                rec.update({"experiment": "e6_budget_scan", "model": model,
                            "budget": budget, "arm": arm})
                records.append(rec)
    return records


def export_reasoning(source_root: Path, output_root: Path) -> list[dict]:
    records = []
    for model in REASONING_MODELS:
        for condition in REASONING_CONDITIONS:
            rel = (Path("probe_eval_results/api_comparison_reasoning_200") / model /
                   f"benchmark_results_cond{condition}.json")
            src = source_root / rel
            data = load(src)
            summary = checked_summary(data, src)
            payload = {
                "schema_version": "1.0",
                "provenance": provenance(source_root, src, SEED_COMMIT),
                "summary": summary,
                "item_results": [
                    {
                        "bench_index": item["bench_index"],
                        "item_id": item["item_id"],
                        "match": bool(item.get("match")),
                    }
                    for item in data["item_results"]
                ],
            }
            dst = output_root / rel.relative_to("probe_eval_results")
            rec = write_compact(dst, payload)
            rec.update({"experiment": "reasoning_200", "model": model,
                        "condition": condition})
            records.append(rec)
    return records


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=repo.parent / "igpt")
    parser.add_argument("--output-root", type=Path,
                        default=repo / "results/raw/robustness_compact")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    records = []
    records.extend(export_seed_sweep(source_root, output_root))
    records.extend(export_risk_grid(source_root, output_root))
    records.extend(export_e6(source_root, output_root))
    records.extend(export_reasoning(source_root, output_root))

    # Store release-relative paths in the manifest, never machine-specific paths.
    for record in records:
        record["path"] = Path(record["path"]).relative_to(repo).as_posix()
    manifest = {
        "schema_version": "1.0",
        "purpose": "Index and integrity ledger for minimal per-item robustness/reasoning exports.",
        "source_repository": "frozen experimental pipeline",
        "source_commits": {
            "generation_seed_sweep_and_reasoning": SEED_COMMIT,
            "risk_weight_grid_and_e6_budget_scan": REVISION_COMMIT,
        },
        "scope_note": (
            "The deployed comparison cells and vanilla reasoning baselines are excluded "
            "because the main-panel compact export is their canonical public copy."
        ),
        "shared_dependencies": {
            "primary_outcomes": "results/raw/main_experiment/primary_outcomes_compact.json",
            "reasoning_subset": "results/aggregates/api_comparison_reasoning_200/subset_items.json",
            "probe_profiles_for_e6_auroc": "results/supplementary_evidence/probe_profiles.json",
        },
        "projection_note": (
            "Each compact cell embeds its full-source path, byte count, SHA-256, commit, "
            "and only the per-item fields consumed by the frozen aggregation."
        ),
        "counts": {
            "generation_seed_sweep": 100,
            "risk_weight_grid_alternate_arms": 15,
            "e6_budget_scan": 24,
            "reasoning_200": 10,
            "total": len(records),
        },
        "files": records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(records)} compact cells and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
