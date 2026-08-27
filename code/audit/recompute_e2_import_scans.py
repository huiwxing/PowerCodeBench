#!/usr/bin/env python3
"""Recompute the frozen E2 D-6 import scans from public evidence.

Default usage needs only this release and Python's standard library::

    python3 code/audit/recompute_e2_import_scans.py

The pre-fix raw generations are represented by a compact public audit file:
item identity, complete generated code, empty/non-empty state, relevant import
lines, import flags, and a code SHA-256.  The post-fix scan reads all 24
byte-exact full raw result files already distributed in PowerCodeBench.

Maintainers with the frozen source repository may recreate the compact
pre-fix evidence before running the same checks::

    python3 code/audit/recompute_e2_import_scans.py \
        --export-pre-source /path/to/igpt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "audit/e2_import_scan"
DEFAULT_PRE_EVIDENCE = AUDIT_DIR / "pre_fix_generations.json"
DEFAULT_PRE_ARCHIVE = AUDIT_DIR / "frozen/import_scan_prefix_B2.json"
DEFAULT_POST_ARCHIVE = AUDIT_DIR / "frozen/import_scan_v2_postfix.json"
DEFAULT_POST_ROOT = ROOT / "results/raw/e2_transfer"
DEFAULT_POST_MANIFEST = DEFAULT_POST_ROOT / "manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "results/aggregates/e2_import_scans"

SOURCE_COMMIT = "6990ac025ad3b7eb26d9767c5cc470eaef02374f"
MODELS_PRE = (
    "Qwen_Qwen2.5-Coder-32B-Instruct",
    "meta-llama_Llama-3.1-70B-Instruct",
)
MODELS_POST = (
    "Qwen_Qwen2.5-Coder-32B-Instruct",
    "deepseek-v4-flash",
    "meta-llama_Llama-3.1-70B-Instruct",
)
CONDITIONS_PRE = ("A", "C", "Rsem")
CONDITIONS_POST = ("A", "C", "Rsem", "RsemB")
BACKENDS = ("opendss", "pypsa")

PANDAPOWER_RE = re.compile(
    r"^(?:import pandapower|from pandapower)",
    re.MULTILINE,
)
TARGET_RES = {
    "opendss": re.compile(
        r"^(?:import opendssdirect|from opendssdirect|import dss|from dss)",
        re.MULTILINE,
    ),
    "pypsa": re.compile(
        r"^(?:import pypsa|from pypsa)",
        re.MULTILINE,
    ),
}

PRE_META = {
    "purpose": (
        "Pre-fix (B2-freeze) evidence for instrument defect D-6: the bare-condition "
        "(A) system prompt hard-coded \"pandapower\" as the library name, so every "
        "condition-A generation on both non-pandapower backends imported pandapower "
        "instead of the target library. Fixed via the BENCHMARK_LIBRARY template slot "
        "(utils.py benchmark_library_name); see transfer/DEFECT_LEDGER.md."
    ),
    "method": (
        "line-anchored regex over generated_code of every frozen B2-round result file "
        "(read-only): pandapower = ^import pandapower|^from pandapower; target = "
        "opendssdirect/dss (OpenDSS) or pypsa (PyPSA) equivalents."
    ),
    "scan_scope": (
        "probe_eval_results/e2_{opendss,pypsa}_bench "
        "(pre-B1-prime frozen round)"
    ),
    "derived_from_frozen_data": True,
}

POST_META = {
    "purpose": (
        "Post-fix (v2, B1-prime re-freeze) verification that defect D-6 is fully "
        "reversed: zero pandapower imports, target-library imports in every non-empty "
        "generation."
    ),
    "method": (
        "same regex scan as transfer/import_scan_prefix_B2.json, read-only over the "
        "frozen v2 result files."
    ),
    "derived_from_frozen_data": True,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 (Git identity)


def load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def exact_json_bytes(value) -> bytes:
    # The two historical scan files were frozen with ``indent=1`` and no
    # terminal newline.
    return json.dumps(value, indent=1).encode("utf-8")


def write_json(path: Path, value, *, exact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exact:
        path.write_bytes(exact_json_bytes(value))
        return
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def relevant_import_lines(code: str, backend: str) -> tuple[list[str], list[str]]:
    pandapower = []
    target = []
    for line in code.splitlines():
        if PANDAPOWER_RE.search(line):
            pandapower.append(line)
        if TARGET_RES[backend].search(line):
            target.append(line)
    return pandapower, target


def generation_record(
    *,
    source_file: str,
    release_file: str | None,
    backend: str,
    model: str,
    condition: str,
    item_index: int,
    item: dict,
    include_generated_code: bool = False,
) -> dict:
    code = item.get("generated_code") or ""
    pp_lines, target_lines = relevant_import_lines(code, backend)
    record = {
        "source_file": source_file,
        "backend": backend,
        "model": model,
        "condition": condition,
        "item_index": item_index,
        "bench_index": item.get("bench_index"),
        "item_id": item.get("item_id"),
        "empty": not bool(code.strip()),
        "pandapower_import": bool(pp_lines),
        "target_library_import": bool(target_lines),
        "pandapower_import_lines": pp_lines,
        "target_library_import_lines": target_lines,
        "generated_code_bytes": len(code.encode("utf-8")),
        "generated_code_sha256": sha256_bytes(code.encode("utf-8")),
    }
    if release_file is not None:
        record["release_file"] = release_file
    if include_generated_code:
        record["generated_code"] = code
    return record


def compact_file_descriptor(path: Path, source_path: str) -> dict:
    data = path.read_bytes()
    digest = sha256_bytes(data)
    return {
        "source_path": source_path,
        "bytes": len(data),
        "sha256": digest,
        "storage": "git-lfs",
        "lfs_oid_sha256": digest,
        "n_generations": len(load_json(path)["item_results"]),
    }


def export_pre_fix(source_root: Path, output: Path) -> None:
    source_files = []
    records = []
    for backend in BACKENDS:
        directory = source_root / f"probe_eval_results/e2_{backend}_bench"
        for model in MODELS_PRE:
            for condition in CONDITIONS_PRE:
                path = directory / model / f"benchmark_results_cond{condition}.json"
                source_path = path.relative_to(source_root).as_posix()
                source_files.append(compact_file_descriptor(path, source_path))
                data = load_json(path)
                for item_index, item in enumerate(data["item_results"]):
                    records.append(generation_record(
                        source_file=source_path,
                        release_file=None,
                        backend=backend,
                        model=model,
                        condition=condition,
                        item_index=item_index,
                        item=item,
                        include_generated_code=True,
                    ))

    frozen_scan = source_root / "transfer/import_scan_prefix_B2.json"
    evidence = {
        "schema_version": 1,
        "evidence_unit": "one pre-fix frozen generation",
        "provenance": {
            "source_repository": "frozen experimental pipeline",
            "source_commit": SOURCE_COMMIT,
            "source_files": source_files,
            "frozen_scan": {
                "source_path": "transfer/import_scan_prefix_B2.json",
                "bytes": frozen_scan.stat().st_size,
                "sha256": sha256_bytes(frozen_scan.read_bytes()),
                "git_blob": git_blob_sha1(frozen_scan.read_bytes()),
            },
            "content_policy": (
                "Complete generated_code is retained so the public-default audit can "
                "rerun the line-anchored regex. Execution outputs, prompts, references, "
                "and other fields irrelevant to the import scan are omitted."
            ),
            "exporter": "code/audit/recompute_e2_import_scans.py",
        },
        "records": records,
    }
    if len(records) != 1080:
        raise RuntimeError(f"pre-fix export has {len(records)} records, expected 1080")
    write_json(output, evidence)
    print(f"exported {len(records)} pre-fix generation records -> {output}")


def validate_compact_records(
    evidence: dict,
    *,
    require_generated_code: bool = False,
) -> list[dict]:
    seen = set()
    rescanned = []
    for record in evidence["records"]:
        key = (record["source_file"], record["item_index"])
        if key in seen:
            raise RuntimeError(f"duplicate compact record: {key}")
        seen.add(key)

        if require_generated_code:
            if "generated_code" not in record:
                raise RuntimeError(f"compact record lacks complete generated_code: {key}")
            code = record["generated_code"]
            encoded = code.encode("utf-8")
            pp_lines, target_lines = relevant_import_lines(code, record["backend"])
            empty = not bool(code.strip())
            if len(encoded) != record["generated_code_bytes"]:
                raise RuntimeError(f"generated-code byte-size mismatch: {key}")
            if sha256_bytes(encoded) != record["generated_code_sha256"]:
                raise RuntimeError(f"generated-code SHA-256 mismatch: {key}")
            if empty != record["empty"]:
                raise RuntimeError(f"generated-code empty flag mismatch: {key}")
            if pp_lines != record["pandapower_import_lines"]:
                raise RuntimeError(f"pandapower import-line extraction mismatch: {key}")
            if target_lines != record["target_library_import_lines"]:
                raise RuntimeError(f"target import-line extraction mismatch: {key}")
        else:
            pp_lines = record["pandapower_import_lines"]
            target_lines = record["target_library_import_lines"]
            empty = record["empty"]

        pp_flag = bool(pp_lines)
        target_flag = bool(target_lines)
        if pp_flag != record["pandapower_import"]:
            raise RuntimeError(f"pandapower flag/import-line mismatch: {key}")
        if target_flag != record["target_library_import"]:
            raise RuntimeError(f"target flag/import-line mismatch: {key}")
        if empty and record["generated_code_bytes"] != 0:
            # Whitespace-only generations would be empty but nonzero; there are
            # none in this freeze, so fail if the evidence semantics ever drift.
            raise RuntimeError(f"unexpected whitespace-only compact generation: {key}")
        derived = dict(record)
        derived["empty"] = empty
        derived["pandapower_import"] = pp_flag
        derived["target_library_import"] = target_flag
        derived["pandapower_import_lines"] = pp_lines
        derived["target_library_import_lines"] = target_lines
        rescanned.append(derived)
    return rescanned


def per_file_from_records(records: list[dict], file_order: list[str]) -> list[dict]:
    by_file = {path: [] for path in file_order}
    for record in records:
        by_file[record["source_file"]].append(record)
    result = []
    for path in file_order:
        rows = by_file[path]
        result.append({
            "file": path,
            "n_generations": len(rows),
            "pandapower_imports": sum(r["pandapower_import"] for r in rows),
            "target_lib_imports": sum(r["target_library_import"] for r in rows),
            "non_target_items": [
                {"id": None, "empty": r["empty"]}
                for r in rows
                if not r["target_library_import"]
            ],
        })
    return result


def rebuild_pre_scan(evidence: dict) -> dict:
    # The flags stored by the maintainer export are not aggregation inputs:
    # rescan every complete generated_code and require the stored fields to
    # match only as a second, independent consistency check.
    records = validate_compact_records(evidence, require_generated_code=True)
    file_order = [entry["source_path"] for entry in evidence["provenance"]["source_files"]]
    per_file = per_file_from_records(records, file_order)
    cond_a = [record for record in records if record["condition"] == "A"]
    injected = [record for record in records if record["condition"] != "A"]
    return {
        "meta": PRE_META,
        "headline": {
            "condA_generations": len(cond_a),
            "condA_pandapower_imports": sum(r["pandapower_import"] for r in cond_a),
            "condA_target_lib_imports": sum(r["target_library_import"] for r in cond_a),
            "note": "SM S10 quotes this as \"360/360 pp imports pre-fix\".",
        },
        "injected_arms": {
            "generations": len(injected),
            "pandapower_imports": sum(r["pandapower_import"] for r in injected),
            "note": (
                "Injected arms name the target library via LibrarySpec; the single "
                "Rsem pandapower co-import (PyPSA/32B) also imports pypsa and is "
                "counted in target_lib_imports of its file."
            ),
        },
        "per_file": per_file,
    }


def verify_public_file(path: Path, expected: dict) -> None:
    data = path.read_bytes()
    if len(data) != expected["bytes"]:
        raise RuntimeError(f"byte-size mismatch for {path}: {len(data)} != {expected['bytes']}")
    digest = sha256_bytes(data)
    if digest != expected["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {digest} != {expected['sha256']}")


def scan_post_fix(post_root: Path, manifest_path: Path) -> tuple[list[dict], list[dict], dict]:
    manifest = load_json(manifest_path)
    inventory = manifest["cold_start_cells"]
    source_pattern = inventory["source_path_pattern"]
    release_pattern = inventory["release_path_pattern"]
    records = []
    source_files = []
    file_order = []

    for backend in BACKENDS:
        for model in MODELS_POST:
            for condition in CONDITIONS_POST:
                source_file = source_pattern.format(
                    backend=backend,
                    model=model,
                    condition=condition,
                )
                release_file = release_pattern.format(
                    backend=backend,
                    model=model,
                    condition=condition,
                )
                path = (
                    post_root
                    / backend
                    / "base"
                    / model
                    / f"benchmark_results_cond{condition}.json"
                )
                expected = inventory["files"][backend][model][condition]
                verify_public_file(path, expected)
                data = load_json(path)
                if len(data["item_results"]) != 90:
                    raise RuntimeError(
                        f"{release_file} has {len(data['item_results'])} generations, expected 90"
                    )
                file_order.append(source_file)
                source_files.append({
                    "source_path": source_file,
                    "release_path": release_file,
                    "source_commit": inventory["source_commit"],
                    "bytes": expected["bytes"],
                    "sha256": expected["sha256"],
                    "n_generations": len(data["item_results"]),
                })
                for item_index, item in enumerate(data["item_results"]):
                    records.append(generation_record(
                        source_file=source_file,
                        release_file=release_file,
                        backend=backend,
                        model=model,
                        condition=condition,
                        item_index=item_index,
                        item=item,
                    ))

    post_evidence = {
        "schema_version": 1,
        "evidence_unit": "one post-fix frozen generation",
        "provenance": {
            "source_repository": manifest["source_repository"]["local_path_at_migration"],
            "source_commit": inventory["source_commit"],
            "release_manifest": {
                "path": manifest_path.relative_to(ROOT).as_posix(),
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256_bytes(manifest_path.read_bytes()),
            },
            "source_files": source_files,
            "scan_source": (
                "Full generated_code fields in the byte-exact public raw result JSONs."
            ),
            "recomputer": "code/audit/recompute_e2_import_scans.py",
        },
        "records": records,
    }
    if len(records) != 2160:
        raise RuntimeError(f"post-fix scan has {len(records)} records, expected 2160")
    validate_compact_records(post_evidence)
    return records, per_file_from_records(records, file_order), post_evidence


def totals(files: list[dict]) -> dict:
    return {
        "n_generations": sum(row["n_generations"] for row in files),
        "pandapower_imports": sum(row["pandapower_imports"] for row in files),
        "target_lib_imports": sum(row["target_lib_imports"] for row in files),
    }


def non_target_groups(files: list[dict]) -> list[dict]:
    return [
        {"file": row["file"], "items": row["non_target_items"]}
        for row in files
        if row["non_target_items"]
    ]


def backend_from_source_path(path: str) -> str:
    if "e2_opendss_bench" in path:
        return "opendss"
    if "e2_pypsa_bench" in path:
        return "pypsa"
    raise RuntimeError(f"cannot infer backend from {path}")


def is_frozen_13_file(row: dict) -> bool:
    path = row["file"]
    if "benchmark_results_condRsemB.json" in path:
        return False
    if "/deepseek-v4-flash/" in path:
        return (
            "e2_opendss_bench_v2" in path
            and path.endswith("benchmark_results_condA.json")
        )
    return True


def rebuild_post_scan(per_file: list[dict]) -> dict:
    frozen = [row for row in per_file if is_frozen_13_file(row)]
    frozen_open = [row for row in frozen if backend_from_source_path(row["file"]) == "opendss"]
    frozen_pypsa = [row for row in frozen if backend_from_source_path(row["file"]) == "pypsa"]
    full_open = [row for row in per_file if backend_from_source_path(row["file"]) == "opendss"]
    full_pypsa = [row for row in per_file if backend_from_source_path(row["file"]) == "pypsa"]
    return {
        "meta": POST_META,
        "frozen_scope_13_files": {
            "description": (
                "The 13 v2 result files that existed at the 2026-07-27 frozen "
                "full-corpus scan (local models A/C/Rsem on both backends + DeepSeek "
                "OpenDSS condA); this is the scope quoted in SM S10."
            ),
            "total": totals(frozen),
            "opendss": totals(frozen_open),
            "pypsa": totals(frozen_pypsa),
            "non_target_items": non_target_groups(frozen),
            "note": (
                "SM S10 quotes: zero pp imports; 629/630 (OpenDSS) / 540/540 "
                "(PyPSA) target imports; the single OpenDSS miss is a DeepSeek empty "
                "generation (code_extraction_failed), not a wrong-library generation."
            ),
        },
        "full_corpus_24_files": {
            "description": (
                "Superset scan over all 24 v2 result files including the "
                "later-finishing DeepSeek cells and the budget-matched RsemB arms."
            ),
            "total": totals(per_file),
            "opendss": totals(full_open),
            "pypsa": totals(full_pypsa),
            "non_target_items": non_target_groups(per_file),
        },
        "per_file": per_file,
    }


def exact_crosscheck(label: str, rebuilt: dict, archive_path: Path) -> dict:
    rebuilt_bytes = exact_json_bytes(rebuilt)
    archive_bytes = archive_path.read_bytes()
    result = {
        "label": label,
        "archive_path": archive_path.relative_to(ROOT).as_posix(),
        "archive_bytes": len(archive_bytes),
        "archive_sha256": sha256_bytes(archive_bytes),
        "recomputed_bytes": len(rebuilt_bytes),
        "recomputed_sha256": sha256_bytes(rebuilt_bytes),
        "byte_exact": rebuilt_bytes == archive_bytes,
    }
    if not result["byte_exact"]:
        raise RuntimeError(
            f"{label} reconstruction differs from frozen archive: "
            f"{result['recomputed_sha256']} != {result['archive_sha256']}"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-evidence", type=Path, default=DEFAULT_PRE_EVIDENCE)
    parser.add_argument("--pre-archive", type=Path, default=DEFAULT_PRE_ARCHIVE)
    parser.add_argument("--post-archive", type=Path, default=DEFAULT_POST_ARCHIVE)
    parser.add_argument("--post-root", type=Path, default=DEFAULT_POST_ROOT)
    parser.add_argument("--post-manifest", type=Path, default=DEFAULT_POST_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--export-pre-source",
        type=Path,
        help="optional maintainer-only source repository used to rebuild pre-fix compact evidence",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pre_evidence_path = args.pre_evidence.resolve()
    if args.export_pre_source:
        export_pre_fix(args.export_pre_source.resolve(), pre_evidence_path)

    pre_evidence = load_json(pre_evidence_path)
    rebuilt_pre = rebuild_pre_scan(pre_evidence)
    post_records, post_per_file, post_evidence = scan_post_fix(
        args.post_root.resolve(),
        args.post_manifest.resolve(),
    )
    rebuilt_post = rebuild_post_scan(post_per_file)

    pre_check = exact_crosscheck("pre_fix", rebuilt_pre, args.pre_archive.resolve())
    post_check = exact_crosscheck("post_fix", rebuilt_post, args.post_archive.resolve())
    output_dir = args.output_dir.resolve()
    write_json(output_dir / "import_scan_prefix_B2.json", rebuilt_pre, exact=True)
    write_json(output_dir / "import_scan_v2_postfix.json", rebuilt_post, exact=True)
    write_json(output_dir / "post_fix_generations.json", post_evidence)

    summary = {
        "schema_version": 1,
        "method": {
            "pandapower_regex": PANDAPOWER_RE.pattern,
            "target_regex": {backend: regex.pattern for backend, regex in TARGET_RES.items()},
            "empty_definition": "not bool(generated_code.strip())",
        },
        "pre_fix": {
            "generation_records": len(pre_evidence["records"]),
            "headline": rebuilt_pre["headline"],
            "injected_arms": rebuilt_pre["injected_arms"],
            "frozen_archive_crosscheck": pre_check,
        },
        "post_fix": {
            "generation_records": len(post_records),
            "frozen_scope_13_files": rebuilt_post["frozen_scope_13_files"],
            "full_corpus_24_files": rebuilt_post["full_corpus_24_files"],
            "frozen_archive_crosscheck": post_check,
        },
        "all_frozen_scans_byte_exact": pre_check["byte_exact"] and post_check["byte_exact"],
    }
    write_json(output_dir / "crosscheck.json", summary)
    print(
        f"pre-fix records={len(pre_evidence['records'])}; "
        f"post-fix records={len(post_records)}; frozen scans byte-exact=True -> {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
