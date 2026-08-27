#!/usr/bin/env python3
"""Export and rebuild the supplementary per-model reactive-router shares.

The public artifact contains one compact record per *observed fix attempt*.
Route counts are therefore rebuilt from events, rather than copied from the
source run summaries.  ``--export`` is a maintainer-only convenience for
projecting the frozen, verbose run logs into that compact representation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evidence_common import (
    read_json,
    repo_descriptor,
    sha256_file,
    source_descriptor,
    write_json,
)

HERE = Path(__file__).resolve()
PCB_ROOT = HERE.parents[2]
DEFAULT_SOURCE = PCB_ROOT.parent / "igpt"
DEFAULT_ARTIFACT = PCB_ROOT / "results/supplementary_evidence/reactive_router_counts.json"
DEFAULT_OUTPUT = PCB_ROOT / "results/aggregates/supplementary_reactive_router_shares.json"

MODELS = {
    "Qwen2.5-1.5B": "Qwen_Qwen2.5-Coder-1.5B-Instruct",
    "Qwen2.5-7B": "Qwen_Qwen2.5-Coder-7B-Instruct",
    "Qwen2.5-14B": "Qwen_Qwen2.5-Coder-14B-Instruct",
    "Qwen2.5-32B": "Qwen_Qwen2.5-Coder-32B-Instruct",
    "Llama-8B": "meta-llama_Llama-3.1-8B-Instruct",
    "Llama-70B": "meta-llama_Llama-3.1-70B-Instruct",
    "Llama-405B": "meta-llama_Llama-3.1-405B-Instruct",
    "Qwen3-Next": "Qwen_Qwen3-Coder-Next",
    "Qwen3-480B": "Qwen_Qwen3-Coder-480B-A35B-Instruct",
    "GPT-OSS-120B": "openai_gpt-oss-120b",
}
CONDITIONS = {"C+FDR": "condC_FDR", "C+FDRS": "condC_FDRS"}
ROUTES = ["api_doc", "basic_fix", "boundary_contract", "semantic_fix"]


def _stats(values: list[int]) -> dict:
    return {
        "avg": int(sum(values) / len(values)) if values else 0,
        "total": sum(values),
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
    }


def _route_stats(events: list[list]) -> dict:
    out = {}
    for route_index, route in enumerate(ROUTES):
        selected = [event for event in events if event[2] == route_index]
        if not selected:
            continue
        docs = [int(event[3]) for event in selected]
        prompts = [int(event[4]) for event in selected]
        out[route] = {
            "attempts": len(selected),
            "successful_fixes": sum(bool(event[5]) for event in selected),
            "docs": _stats(docs),
            "prompt": _stats(prompts),
            "docs_nonempty": sum(value > 0 for value in docs),
        }
    return out


def export(source_root: Path, artifact: Path) -> None:
    comparison = source_root / "probe_eval_results/comparison"
    files = []
    models = {}
    for label, source_model in MODELS.items():
        models[label] = {"source_model": source_model, "conditions": {}}
        for display, condition in CONDITIONS.items():
            path = comparison / source_model / f"benchmark_results_{condition}.json"
            raw = read_json(path)
            stats = raw["summary"]["fix_prompt_token_stats"]
            item_ids = sorted(item["item_id"] for item in raw["item_results"])
            if len(item_ids) != len(set(item_ids)):
                raise ValueError(f"{path}: duplicate item IDs")
            item_index = {item_id: index for index, item_id in enumerate(item_ids)}
            events = []
            for item in raw["item_results"]:
                for attempt in item.get("fix_chain", []):
                    route = attempt.get("fix_route")
                    if route not in ROUTES:
                        raise ValueError(f"{path}: unknown/missing route {route!r}")
                    events.append([
                        item_index[item["item_id"]],
                        int(attempt["round"]),
                        ROUTES.index(route),
                        int(attempt.get("fix_docs_tokens") or 0),
                        int(attempt.get("fix_prompt_tokens") or 0),
                        1 if attempt.get("success") else 0,
                    ])
            events.sort(key=lambda event: (event[0], event[1]))
            route_stats = _route_stats(events)
            if route_stats != stats["route_stats"]:
                raise ValueError(f"{path}: event reconstruction != source route_stats")
            if len(events) != int(stats["attempts"]):
                raise ValueError(
                    f"{path}: event attempts {len(events)} != total {stats['attempts']}"
                )
            models[label]["conditions"][display] = {
                "item_ids": item_ids,
                "events": events,
                "event_fields": [
                    "item_index", "round", "route_index", "docs_tokens",
                    "prompt_tokens", "success",
                ],
                # Retain the independently written source summary as a public
                # cross-check; default aggregation below refuses any event /
                # source-summary disagreement.
                "source_summary_route_stats": stats["route_stats"],
                "source_run_sha256": sha256_file(path),
            }
            files.append(source_descriptor(source_root, path))
    out = {
        "schema_version": 2,
        "evidence_unit": "one compact record per observed item-round fix attempt",
        "route_dictionary": ROUTES,
        "event_encoding": (
            "Each event uses event_fields; item_index addresses the condition-local "
            "sorted item_ids array and route_index addresses route_dictionary."
        ),
        "provenance": {**repo_descriptor(source_root), "files": files},
        "models": models,
    }
    write_json(artifact, out)
    print(f"exported 20 per-attempt router logs ({artifact.stat().st_size} bytes) -> {artifact}")


def aggregate(artifact: Path, output: Path) -> None:
    source = read_json(artifact)
    if source.get("schema_version") != 2:
        raise ValueError("router evidence must use the per-attempt schema v2")
    if source.get("route_dictionary") != ROUTES:
        raise ValueError("router evidence route dictionary changed")
    if set(source.get("models", {})) != set(MODELS):
        raise ValueError("router evidence does not contain the exact 10-model panel")
    provenance_files = source.get("provenance", {}).get("files", [])
    provenance_by_path = {row.get("path"): row for row in provenance_files}
    if (len(provenance_files) != 20
            or len(provenance_by_path) != len(provenance_files)):
        raise ValueError("router provenance must contain 20 unique source runs")
    rows = {}
    total_events = 0
    used_provenance_paths = set()
    for label, model in source["models"].items():
        if model.get("source_model") != MODELS[label]:
            raise ValueError(f"{label}: source-model identity changed")
        if set(model.get("conditions", {})) != set(CONDITIONS):
            raise ValueError(f"{label}: expected exactly C+FDR and C+FDRS")
        rows[label] = {}
        for condition, data in model["conditions"].items():
            item_ids = data.get("item_ids") or []
            events = data.get("events") or []
            if (len(item_ids) != 2000 or item_ids != sorted(item_ids)
                    or len(item_ids) != len(set(item_ids))):
                raise ValueError(f"{label}/{condition}: invalid item axis")
            if data.get("event_fields") != [
                    "item_index", "round", "route_index", "docs_tokens",
                    "prompt_tokens", "success"]:
                raise ValueError(f"{label}/{condition}: event-field schema changed")
            source_path = (
                "probe_eval_results/comparison/"
                f"{MODELS[label]}/benchmark_results_{CONDITIONS[condition]}.json"
            )
            provenance = provenance_by_path.get(source_path)
            if (not provenance
                    or provenance.get("sha256") != data.get("source_run_sha256")):
                raise ValueError(
                    f"{label}/{condition}: source hash/provenance disagreement")
            used_provenance_paths.add(source_path)
            seen = set()
            for event in events:
                if not (isinstance(event, list) and len(event) == 6):
                    raise ValueError(f"{label}/{condition}: malformed event")
                item_index, round_index, route_index, docs, prompt, success = event
                if not (isinstance(item_index, int) and 0 <= item_index < len(item_ids)):
                    raise ValueError(f"{label}/{condition}: invalid event item index")
                if not (isinstance(round_index, int) and 1 <= round_index <= 3):
                    raise ValueError(f"{label}/{condition}: invalid fix round")
                if not (isinstance(route_index, int) and 0 <= route_index < len(ROUTES)):
                    raise ValueError(f"{label}/{condition}: invalid route index")
                if not (isinstance(docs, int) and docs >= 0
                        and isinstance(prompt, int) and prompt >= 0
                        and success in (0, 1)):
                    raise ValueError(f"{label}/{condition}: invalid event payload")
                event_key = (item_index, round_index)
                if event_key in seen:
                    raise ValueError(f"{label}/{condition}: duplicate item-round event")
                seen.add(event_key)
            route_stats = _route_stats(events)
            if route_stats != data.get("source_summary_route_stats"):
                raise ValueError(
                    f"{label}/{condition}: events disagree with frozen source summary"
                )
            total = len(events)
            total_events += total
            counts = {route: int(stats["attempts"])
                      for route, stats in route_stats.items()}
            if sum(counts.values()) != total:
                raise ValueError(f"{label}/{condition}: counts do not sum to attempts")
            rows[label][condition] = {
                "attempts": total,
                "counts": counts,
                "shares_pct": {route: round(100.0 * count / total, 1)
                               for route, count in counts.items()},
                "route_stats": route_stats,
            }
    if used_provenance_paths != set(provenance_by_path):
        raise ValueError("router provenance contains an unused source run")
    if total_events != 68107:
        raise ValueError(
            f"router evidence event census changed: {total_events} != 68107")
    out = {
        "source_artifact": artifact.relative_to(PCB_ROOT).as_posix(),
        "definition": "route share = route attempt count / all fix attempts in model-condition run",
        "rows": rows,
    }
    write_json(output, out)
    print(f"rebuilt supplementary router table -> {output}")


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
