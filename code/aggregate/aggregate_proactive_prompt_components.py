#!/usr/bin/env python3
"""Export and rebuild condition-C's approximate prompt decomposition.

The frozen run logs record exact, backend-specific chat-prompt token counts for
conditions A and C.  The proactive injection log independently records the
construction-time token estimate for every selected L0--L3 snippet and boundary
card.  Joining those two *item-aligned* sources yields the manuscript's rounded
``base + documentation + framing`` description without solving backwards from
the displayed means.

``--export`` projects the verbose private injection log to a public compact
vector.  Default operation uses only repository-local evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evidence_common import (
    read_json,
    repo_descriptor,
    sha256_file,
    source_descriptor,
    write_json,
)


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
DEFAULT_SOURCE = ROOT.parent / "igpt"
DEFAULT_PRIMARY = ROOT / "results/raw/main_experiment/primary_outcomes_compact.json"
DEFAULT_ARTIFACT = ROOT / "results/supplementary_evidence/proactive_prompt_components.json"
DEFAULT_OUTPUT = ROOT / "results/aggregates/proactive_prompt_components.json"

MODELS = [
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


def canonical(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def run(primary: dict, model: str, condition: str) -> dict:
    return primary["runs"][f"comparison|{model}|{condition}"]


def export(source_root: Path, primary_path: Path, artifact_path: Path) -> None:
    primary = read_json(primary_path)
    axis = primary["datasets"]["powercodebench_2000"]["item_ids"]
    injection_path = source_root / "probe_eval_results/injection_log.json"
    injection = read_json(injection_path)

    expected_source = primary["injection_log_compact"]["source"]
    if sha256_file(injection_path) != expected_source["sha256"]:
        raise ValueError("injection log does not match the source pinned by primary compact")

    models = {}
    for model in MODELS:
        model_log = injection.get(model)
        if not isinstance(model_log, dict) or set(model_log) != set(axis):
            raise ValueError(f"{model}: injection items do not equal the public item axis")
        snippets = []
        boundaries = []
        for item_id in axis:
            item = model_log[item_id]
            snippet_tokens = int(item.get("doc_tokens") or 0)
            selected_sum = sum(
                int(entry.get("tokens") or 0)
                for entry in (item.get("selected_layers") or [])
            )
            if selected_sum != snippet_tokens:
                raise ValueError(f"{model}/{item_id}: selected snippet token mismatch")
            snippets.append(snippet_tokens)
            boundaries.append(int(item.get("boundary_card_tokens") or 0))
        payload = {
            "snippet_tokens": snippets,
            "boundary_card_tokens": boundaries,
        }
        models[model] = {**payload, "vector_sha256": digest(payload)}

    out = {
        "schema_version": 1,
        "evidence_unit": "item-aligned construction-time injection token estimates",
        "item_axis": {
            "n_items": len(axis),
            "item_ids_sha256": digest(axis),
            "primary_compact": primary_path.relative_to(ROOT).as_posix(),
            "primary_compact_sha256": sha256_file(primary_path),
        },
        "measurement_note": (
            "snippet_tokens and boundary_card_tokens are the frozen injection "
            "planner's construction-time estimates. A/C total prompt vectors in "
            "the primary compact are exact backend chat-template counts."
        ),
        "provenance": {
            **repo_descriptor(source_root),
            "source": source_descriptor(source_root, injection_path),
        },
        "models": models,
    }
    write_json(artifact_path, out)
    print(
        f"exported {len(MODELS) * len(axis):,} item-level component records "
        f"({artifact_path.stat().st_size} bytes) -> {artifact_path}"
    )


def aggregate(primary_path: Path, artifact_path: Path, output_path: Path) -> None:
    primary = read_json(primary_path)
    artifact = read_json(artifact_path)
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported prompt-component evidence schema")
    if artifact["item_axis"]["primary_compact_sha256"] != sha256_file(primary_path):
        raise ValueError("primary compact changed after component export")
    axis = primary["datasets"]["powercodebench_2000"]["item_ids"]
    if artifact["item_axis"].get("n_items") != len(axis):
        raise ValueError("prompt-component item count differs from primary compact")
    if artifact["item_axis"].get("primary_compact") \
            != primary_path.relative_to(ROOT).as_posix():
        raise ValueError("prompt-component primary path changed")
    if artifact["item_axis"]["item_ids_sha256"] != digest(axis):
        raise ValueError("prompt-component item axis differs from primary compact")
    if set(artifact.get("models", {})) != set(MODELS):
        raise ValueError("prompt-component artifact does not contain the panel")
    injection_source = primary["injection_log_compact"]["source"]
    artifact_source = artifact.get("provenance", {}).get("source", {})
    if (
        artifact.get("provenance", {}).get("commit")
            != injection_source.get("commit")
        or artifact_source.get("path") != injection_source.get("path")
        or artifact_source.get("sha256") != injection_source.get("sha256")
        or artifact_source.get("bytes") != injection_source.get("size_bytes")
    ):
        raise ValueError("prompt-component injection provenance changed")

    rows = []
    panel = {"n_items": 0, "base": 0, "snippets": 0, "boundary": 0,
             "documentation": 0, "framing_residual": 0, "total": 0}
    for model in MODELS:
        evidence = artifact["models"][model]
        payload = {
            "snippet_tokens": evidence["snippet_tokens"],
            "boundary_card_tokens": evidence["boundary_card_tokens"],
        }
        if digest(payload) != evidence.get("vector_sha256"):
            raise ValueError(f"{model}: prompt-component vector hash mismatch")
        base = run(primary, model, "A")["prompt_tokens"]
        total = run(primary, model, "C")["prompt_tokens"]
        snippets = payload["snippet_tokens"]
        boundaries = payload["boundary_card_tokens"]
        if not all(len(values) == len(axis)
                   for values in (base, total, snippets, boundaries)):
            raise ValueError(f"{model}: component vectors are not item-aligned")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in snippets + boundaries
        ):
            raise ValueError(f"{model}: invalid construction-time token estimate")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in base + total
        ):
            raise ValueError(f"{model}: invalid exact A/C prompt token count")
        documentation = [a + b for a, b in zip(snippets, boundaries)]
        framing = [c - a - d for a, c, d in zip(base, total, documentation)]
        if any(value < 0 for value in framing):
            raise ValueError(f"{model}: negative framing residual")
        totals = {
            "n_items": len(axis),
            "base": sum(base),
            "snippets": sum(snippets),
            "boundary": sum(boundaries),
            "documentation": sum(documentation),
            "framing_residual": sum(framing),
            "total": sum(total),
        }
        if totals["base"] + totals["documentation"] \
                + totals["framing_residual"] != totals["total"]:
            raise AssertionError("component accounting identity failed")
        for key in panel:
            panel[key] += totals[key]
        rows.append({
            "model": model,
            "totals": totals,
            "averages": {
                key: totals[key] / totals["n_items"]
                for key in ("base", "snippets", "boundary", "documentation",
                            "framing_residual", "total")
            },
            "joined_item_vector_sha256": digest({
                "item_ids": axis,
                "base": base,
                "snippets": snippets,
                "boundary": boundaries,
                "framing_residual": framing,
                "total": total,
            }),
        })

    averages = {
        key: panel[key] / panel["n_items"]
        for key in ("base", "snippets", "boundary", "documentation",
                    "framing_residual", "total")
    }
    out = {
        "schema_version": 1,
        "source": {
            "primary_compact": primary_path.relative_to(ROOT).as_posix(),
            "primary_compact_sha256": sha256_file(primary_path),
            "component_artifact": artifact_path.relative_to(ROOT).as_posix(),
            "component_artifact_sha256": sha256_file(artifact_path),
        },
        "definition": {
            "base": "exact condition-A chat-prompt tokens for the same model/item",
            "documentation": (
                "construction-time estimate: selected L0-L3 snippet tokens plus "
                "boundary-card tokens"
            ),
            "framing_residual": (
                "exact C total minus exact A base minus documentation estimate; "
                "includes injection headings/instructions and tokenizer/template "
                "non-additivity, so the manuscript reports this decomposition with ~"
            ),
        },
        "per_model": rows,
        "panel_pooled_exact_from_item_vectors": {
            "totals": panel,
            "averages": averages,
            "rounded_manuscript_components": {
                "base_tokens": round(averages["base"] / 10) * 10,
                "documentation_tokens": round(averages["documentation"] / 10) * 10,
                "framing_tokens": round(averages["framing_residual"] / 10) * 10,
            },
            "accounting_identity_holds": (
                panel["base"] + panel["documentation"]
                + panel["framing_residual"] == panel["total"]
            ),
        },
    }
    write_json(output_path, out)
    print(
        "rebuilt prompt components: "
        f"base={averages['base']:.3f}, docs={averages['documentation']:.3f}, "
        f"framing={averages['framing_residual']:.3f}, "
        f"C total={averages['total']:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.export:
        export(args.source_root.resolve(), args.primary.resolve(), args.artifact.resolve())
    aggregate(args.primary.resolve(), args.artifact.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
