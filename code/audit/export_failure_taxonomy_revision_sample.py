#!/usr/bin/env python3
"""Export a fresh, balanced 100-item Failure Anatomy revision sample.

The exporter only draws the sample: it verifies and streams the 55 frozen
comparison result files, then applies a fully specified deterministic
reservoir sampler independently within each of the 4 model-tier x 5 condition
cells.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import resource
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from recompute_failure_taxonomy import MODEL_TIERS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPACT = ROOT / "results/raw/main_experiment/primary_outcomes_compact.json"
DEFAULT_OUTPUT = ROOT / "audit/failure_taxonomy/revision_sample"

GLOBAL_SEED = 20260809
SAMPLES_PER_CELL = 5
TIERS = ("tier1", "tier2", "tier3", "tier4")
SOURCE_CONDITIONS = ("A_FD", "A_FDR", "A_FDRS", "A_FX", "C_FDRS")

MODEL_ALIASES = {
    "Qwen_Qwen2.5-Coder-0.5B-Instruct": "Qwen2.5-Coder-0.5B",
    "Qwen_Qwen2.5-Coder-1.5B-Instruct": "Qwen2.5-Coder-1.5B",
    "Qwen_Qwen2.5-Coder-7B-Instruct": "Qwen2.5-Coder-7B",
    "Qwen_Qwen2.5-Coder-14B-Instruct": "Qwen2.5-Coder-14B",
    "Qwen_Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder-32B",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama_Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "meta-llama_Llama-3.1-405B-Instruct": "Llama-3.1-405B",
    "openai_gpt-oss-120b": "gpt-oss-120b",
    "Qwen_Qwen3-Coder-480B-A35B-Instruct": "Qwen3-Coder-480B",
    "Qwen_Qwen3-Coder-Next": "Qwen3-Coder-Next",
}

PRIVATE_ENV_RE = re.compile(r"/projects/[^/\s\"')]+/[^/\s\"')]+/igpt_venv")
ITEM_RESULTS_RE = re.compile(r'"item_results"\s*:\s*\[')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sanitize(value: Any) -> Any:
    """Remove the only machine-specific prefix present in execution traces."""
    if isinstance(value, str):
        return PRIVATE_ENV_RE.sub("<EXECUTION_ENV>", value).replace("\x00", "")
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def iter_item_results(path: Path) -> Iterator[dict[str, Any]]:
    """Stream the top-level item_results array using the standard library."""
    decoder = json.JSONDecoder()
    chunk_size = 1024 * 1024
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError(f"item_results array not found: {path}")
            buffer += chunk
            marker = ITEM_RESULTS_RE.search(buffer)
            if marker:
                buffer = buffer[marker.end():]
                break
            buffer = buffer[-128:]

        eof = False
        while True:
            buffer = buffer.lstrip(" \t\r\n,")
            if buffer.startswith("]"):
                return
            try:
                item, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError(f"truncated item_results array: {path}")
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue
            if not isinstance(item, dict):
                raise ValueError(f"non-object item_results entry: {path}")
            yield item
            buffer = buffer[end:]


class CellRng:
    """Version-independent SHA-256 counter RNG with unbiased randbelow()."""

    def __init__(self, tier: str, condition: str) -> None:
        label = f"failure-taxonomy-revision|{GLOBAL_SEED}|{tier}|{condition}"
        self.key = hashlib.sha256(label.encode("utf-8")).digest()
        self.counter = 0

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("randbelow upper bound must be positive")
        modulus = 1 << 256
        limit = modulus - modulus % upper
        while True:
            block = hashlib.sha256(
                self.key + self.counter.to_bytes(16, "big")
            ).digest()
            self.counter += 1
            value = int.from_bytes(block, "big")
            if value < limit:
                return value % upper


def sample_record(
    item: dict[str, Any],
    *,
    model: str,
    source_model: str,
    tier: str,
    condition: str,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Retain the evidence needed for a new classification, without executing it."""
    record = {
        "model": model,
        "source_model": source_model,
        "model_tier": tier,
        "condition": f"cond{condition}",
        "item_id": item.get("item_id"),
        "bench_index": item.get("bench_index"),
        "benchmark_original_index": item.get("benchmark_original_index"),
        "difficulty_level": item.get("difficulty_level"),
        "task": item.get("task"),
        "network": item.get("network"),
        "query_type": item.get("query_type"),
        "n_modifications": item.get("n_modifications"),
        "query": item.get("natural_language_query"),
        "reference_code": item.get("reference_code"),
        "ground_truth": item.get("ground_truth"),
        "ground_truth_type": item.get("ground_truth_type"),
        "generated_code": item.get("generated_code"),
        "raw_output": item.get("raw_output"),
        "executed": item.get("executed"),
        "error_type": item.get("error_type"),
        "error_msg": item.get("error_msg"),
        "exec_output": item.get("exec_output"),
        "match": item.get("match"),
        "match_detail": item.get("match_detail"),
        "diagnostics": item.get("diagnostics"),
        "source": {
            "path": source_path,
            "sha256": source_sha256,
        },
    }
    return sanitize(record)


def source_descriptors(compact: dict[str, Any]) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for run in compact["runs"].values():
        if run.get("group") != "comparison" or run.get("condition") not in SOURCE_CONDITIONS:
            continue
        source_model = str(run["model"])
        model = MODEL_ALIASES.get(source_model)
        if model is None:
            raise ValueError(f"unmapped source model: {source_model}")
        tier = MODEL_TIERS.get(model)
        if tier not in TIERS:
            raise ValueError(f"missing frozen tier for model: {model}")
        source = run["source"]
        descriptors.append({
            "model": model,
            "source_model": source_model,
            "tier": tier,
            "condition": str(run["condition"]),
            "path": str(source["path"]),
            "sha256": str(source["sha256"]),
            "size_bytes": int(source["size_bytes"]),
        })
    if len(descriptors) != 55:
        raise ValueError(f"expected 55 source descriptors, found {len(descriptors)}")
    return sorted(
        descriptors,
        key=lambda row: (row["tier"], row["condition"], row["model"], row["path"]),
    )


def verify_rebuild(output_dir: Path, frozen_dir: Path) -> None:
    for name in (
        "sampling_frame_keys.json",
        "sample_records.json",
        "summary.json",
        "manifest.json",
    ):
        rebuilt = json.loads((output_dir / name).read_text(encoding="utf-8"))
        frozen = json.loads((frozen_dir / name).read_text(encoding="utf-8"))
        if rebuilt != frozen:
            raise ValueError(f"field-for-field rebuild mismatch: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-against", type=Path)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    compact_sha = sha256(args.compact)
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    descriptors = source_descriptors(compact)
    del compact
    gc.collect()

    cells = [(tier, condition) for tier in TIERS for condition in SOURCE_CONDITIONS]
    rngs = {cell: CellRng(*cell) for cell in cells}
    pools = {cell: 0 for cell in cells}
    reservoirs: dict[tuple[str, str], list[dict[str, Any]]] = {cell: [] for cell in cells}
    previous_key: dict[tuple[str, str], tuple[str, int, str]] = {}
    source_audit: list[dict[str, Any]] = []
    public_frame_sources: list[dict[str, Any]] = []

    for descriptor in descriptors:
        source_rel = descriptor["path"]
        source_path = (source_root / source_rel).resolve()
        if source_path != source_root and source_root not in source_path.parents:
            raise ValueError(f"source path escapes source root: {source_rel}")
        if source_path.stat().st_size != descriptor["size_bytes"]:
            raise ValueError(f"source size mismatch: {source_rel}")
        actual_sha = sha256(source_path)
        if actual_sha != descriptor["sha256"]:
            raise ValueError(f"source hash mismatch: {source_rel}")

        cell = (descriptor["tier"], descriptor["condition"])
        n_items = 0
        n_failed = 0
        match_false_keys: list[list[Any]] = []
        for item in iter_item_results(source_path):
            n_items += 1
            bench_index = item.get("bench_index")
            if not isinstance(bench_index, int):
                raise ValueError(f"invalid bench_index in {source_rel}")
            stable_key = (descriptor["model"], bench_index, str(item.get("item_id")))
            if cell in previous_key and stable_key <= previous_key[cell]:
                raise ValueError(f"source stream is not strictly stable-key sorted: {source_rel}")
            previous_key[cell] = stable_key
            if item.get("match") is not False:
                continue

            n_failed += 1
            match_false_keys.append([bench_index, str(item.get("item_id"))])
            pools[cell] += 1
            record = sample_record(
                item,
                model=descriptor["model"],
                source_model=descriptor["source_model"],
                tier=descriptor["tier"],
                condition=descriptor["condition"],
                source_path=source_rel,
                source_sha256=actual_sha,
            )
            reservoir = reservoirs[cell]
            seen = pools[cell]
            if len(reservoir) < SAMPLES_PER_CELL:
                reservoir.append(record)
            else:
                replacement = rngs[cell].randbelow(seen)
                if replacement < SAMPLES_PER_CELL:
                    reservoir[replacement] = record

        if n_items != 2000:
            raise ValueError(f"expected 2,000 item_results, found {n_items}: {source_rel}")
        source_audit.append({
            "path": source_rel,
            "sha256": actual_sha,
            "size_bytes": descriptor["size_bytes"],
            "n_item_results": n_items,
            "n_match_false": n_failed,
        })
        public_frame_sources.append({
            "model": descriptor["model"],
            "source_model": descriptor["source_model"],
            "model_tier": descriptor["tier"],
            "condition": f"cond{descriptor['condition']}",
            "source": {
                "path": source_rel,
                "sha256": actual_sha,
                "size_bytes": descriptor["size_bytes"],
            },
            "match_false_keys": match_false_keys,
        })
        gc.collect()

    if sum(pools.values()) != 89420:
        raise ValueError(f"expected 89,420 match-false records, found {sum(pools.values())}")
    if any(len(reservoirs[cell]) != SAMPLES_PER_CELL for cell in cells):
        raise ValueError("one or more cells did not produce exactly five samples")

    records: list[dict[str, Any]] = []
    for tier, condition in cells:
        selected = sorted(
            reservoirs[(tier, condition)],
            key=lambda row: (row["model"], row["bench_index"], row["item_id"]),
        )
        for ordinal, record in enumerate(selected, start=1):
            record = {"revision_sample_id": f"{tier}_{condition}_{ordinal:02d}", **record}
            records.append(record)

    condition_counts = Counter(record["condition"] for record in records)
    tier_counts = Counter(record["model_tier"] for record in records)
    cell_counts = Counter(
        (record["model_tier"], record["condition"].removeprefix("cond"))
        for record in records
    )
    if set(cell_counts.values()) != {SAMPLES_PER_CELL} or len(cell_counts) != 20:
        raise ValueError("revision sample is not a balanced 4 x 5 x 5 design")

    args.output.mkdir(parents=True, exist_ok=True)
    frame_path = args.output / "sampling_frame_keys.json"
    sample_path = args.output / "sample_records.json"
    summary_path = args.output / "summary.json"
    manifest_path = args.output / "manifest.json"

    frame_document = {
        "schema_version": 1,
        "description": (
            "Public minimum-sufficient 89,420-event sampling frame: ordered "
            "(bench_index, item_id) keys for every match-false record. It replays "
            "the deterministic draw without redistributing the 2.98 GB source logs."
        ),
        "sampling": {
            "global_seed": GLOBAL_SEED,
            "stable_key": ["model", "bench_index", "item_id"],
            "source_order": ["model_tier", "condition", "model", "source.path"],
            "cell_rng_key": (
                "SHA256('failure-taxonomy-revision|20260809|<tier>|<condition>')"
            ),
        },
        "sources": public_frame_sources,
    }
    write_json(frame_path, frame_document)

    sample_document = {
        "schema_version": 1,
        "description": (
            "Fresh deterministic 100-item Failure Anatomy revision sample; "
            "no qualitative classifications are included."
        ),
        "sampling": {
            "global_seed": GLOBAL_SEED,
            "design": "4 model tiers x 5 conditions x 5 match-false records",
            "stable_key": ["model", "bench_index", "item_id"],
            "algorithm": (
                "Sources are ordered by tier/condition/model/path and each source is "
                "verified to be ordered by bench_index/item_id. Each cell uses Algorithm-R "
                "reservoir sampling with an independent SHA-256 counter RNG; randbelow "
                "uses 256-bit rejection sampling."
            ),
            "cell_rng_key": (
                "SHA256('failure-taxonomy-revision|20260809|<tier>|<condition>')"
            ),
            "pool_counts": {
                f"{tier}|cond{condition}": pools[(tier, condition)]
                for tier, condition in cells
            },
        },
        "records": records,
    }
    write_json(sample_path, sample_document)

    summary = {
        "schema_version": 1,
        "status": "ok",
        "n_records": len(records),
        "n_source_files_verified": len(source_audit),
        "n_sampling_frame_match_false": sum(pools.values()),
        "tier_counts": dict(sorted(tier_counts.items())),
        "condition_counts": dict(sorted(condition_counts.items())),
        "cell_counts": {
            f"{tier}|cond{condition}": cell_counts[(tier, condition)]
            for tier, condition in cells
        },
        "model_counts": dict(sorted(Counter(record["model"] for record in records).items())),
        "task_counts": dict(sorted(Counter(record["task"] for record in records).items())),
        "checks": {
            "all_55_source_hashes_verified": len(source_audit) == 55,
            "sampling_frame_matches_frozen_count": sum(pools.values()) == 89420,
            "balanced_4_x_5_x_5": len(records) == 100 and set(cell_counts.values()) == {5},
            "all_sampled_records_are_match_false": all(record["match"] is False for record in records),
            "no_classification_fields_exported": all(
                "category_id" not in record and "justification" not in record
                for record in records
            ),
            "public_sampling_frame_keys_exported": (
                sum(len(source["match_false_keys"])
                    for source in public_frame_sources) == 89420
            ),
        },
    }
    if not all(summary["checks"].values()):
        raise ValueError("revision sample validation failed")
    write_json(summary_path, summary)

    manifest = {
        "schema_version": 1,
        "description": "Frozen provenance for the Failure Anatomy revision sample.",
        "inputs": {
            "compact_path": str(args.compact.resolve().relative_to(ROOT)),
            "compact_sha256": compact_sha,
            "source_files": source_audit,
        },
        "generator": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
            "execution": "single process, sequential streaming JSON, no code execution or model loading",
        },
        "outputs": {
            "sampling_frame_keys.json": {
                "sha256": sha256(frame_path),
                "size_bytes": frame_path.stat().st_size,
            },
            "sample_records.json": {
                "sha256": sha256(sample_path),
                "size_bytes": sample_path.stat().st_size,
            },
            "summary.json": {
                "sha256": sha256(summary_path),
                "size_bytes": summary_path.stat().st_size,
            },
        },
    }
    write_json(manifest_path, manifest)

    if args.verify_against:
        verify_rebuild(args.output, args.verify_against)

    print(json.dumps({
        "status": "ok",
        "n_records": len(records),
        "n_source_files_verified": len(source_audit),
        "sampling_frame_match_false": sum(pools.values()),
        "sample_size_bytes": sample_path.stat().st_size,
        "sampling_frame_size_bytes": frame_path.stat().st_size,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "field_for_field_verified": bool(args.verify_against),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
