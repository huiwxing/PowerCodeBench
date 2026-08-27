#!/usr/bin/env python3
"""Recover conservative item links for the failure-taxonomy ledger.

The historical qualitative ledger retained audit sample IDs (s001--s100),
but not the original benchmark item IDs.  This maintainer-side forensic tool
uses the frozen source paths and hashes in ``primary_outcomes_compact.json``
to search the original result files.  Taxonomy labels are left as they stand,
and a candidate becomes a link only where the match is unique.

Resource discipline is deliberate: the 55 source JSON files are hashed and
loaded one at a time, candidates are reduced to short release records before
the next file is opened, and no model or external package is loaded.
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
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPACT = ROOT / "results/raw/main_experiment/primary_outcomes_compact.json"
DEFAULT_LEDGER = ROOT / "audit/failure_taxonomy/taxonomy_results.json"
DEFAULT_EXEMPLARS = ROOT / "audit/failure_taxonomy/taxonomy_exemplars.md"
DEFAULT_OUTPUT = ROOT / "audit/failure_taxonomy/item_linkage_candidates.json"
DEFAULT_REPORT = ROOT / "audit/failure_taxonomy/item_linkage_report.json"

POOL_CONDITIONS = {"A_FD", "A_FDR", "A_FDRS", "A_FX", "C_FDRS"}

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

QUALIFIED_RE = re.compile(
    r"\b(?:pp|pn|nw|net|network|pts|ts|pe|sc)\.[A-Za-z_]\w*", re.I
)
QUOTED_RE = re.compile(r"(?:['`])([^'`\n]{2,60})(?:['`])")
TECHNICAL_RE = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+|"
    r"AttributeError|KeyError|IndexError|TypeError|ValueError|SyntaxError|"
    r"ImportError|LoadflowNotConverged|cannot_parse_float|timeout)\b",
    re.I,
)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{3,}")

STOPWORDS = {
    "also", "attribute", "because", "before", "calls", "code", "correct",
    "does", "doesn", "error", "executed", "expected", "fails", "from",
    "function", "instead", "into", "loaded", "match", "method", "module",
    "network", "pandapower", "passed", "real", "reference", "result",
    "same", "table", "that", "their", "then", "this", "uses", "used",
    "with", "wrong",
}
PRIVATE_ENV_RE = re.compile(r"/projects/[^/\s\"')]+/[^/\s\"')]+/igpt_venv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_item_results(path: Path):
    """Yield the top-level item_results array without materializing the file.

    The files are ordinary JSON objects rather than JSONL.  A small incremental
    decoder keeps memory proportional to the largest single item, not the full
    20--40 MB source document.
    """
    decoder = json.JSONDecoder()
    marker_re = re.compile(r'"item_results"\s*:\s*\[')
    chunk_size = 1024 * 1024
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError(f"item_results array not found: {path}")
            buffer += chunk
            marker = marker_re.search(buffer)
            if marker:
                buffer = buffer[marker.end():]
                break
            # The marker is short; no reason to retain preceding top-level data.
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
                raise ValueError(f"non-object entry in item_results: {path}")
            yield item
            buffer = buffer[end:]


def normalize_condition(condition: str) -> str:
    return condition.removeprefix("cond")


def short_text(value: Any, limit: int) -> str:
    text = str(value or "").replace("\x00", "")
    text = PRIVATE_ENV_RE.sub("<EXECUTION_ENV>", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def code_excerpt(value: Any, max_lines: int = 8, limit: int = 700) -> str:
    lines = [line.rstrip() for line in str(value or "").splitlines()]
    lines = [line for line in lines if line.strip()][:max_lines]
    return short_text("\n".join(lines), limit)


def normalized_code_prefix(value: str) -> str:
    lines = [re.sub(r"\s+", " ", line.strip()) for line in value.splitlines()]
    return "\n".join(line for line in lines if line)


def parse_exemplars(path: Path) -> dict[str, str]:
    """Return the archived first-five-line code excerpt keyed by sample ID."""
    text = path.read_text(encoding="utf-8")
    headings = list(re.finditer(r"^###\s+(s\d{3})\b.*$", text, re.M))
    excerpts: dict[str, str] = {}
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        section = text[heading.end():end]
        block = re.search(r"```(?:python)?\s*\n(.*?)```", section, re.S)
        if block:
            excerpts[heading.group(1)] = normalized_code_prefix(block.group(1))
    return excerpts


def evidence_terms(justification: str) -> dict[str, list[str]]:
    qualified = sorted({term.lower() for term in QUALIFIED_RE.findall(justification)})
    quoted = sorted({term.lower().strip() for term in QUOTED_RE.findall(justification)})
    technical = sorted({term.lower() for term in TECHNICAL_RE.findall(justification)})
    lexical = sorted({
        term.lower() for term in WORD_RE.findall(justification)
        if term.lower() not in STOPWORDS
    })
    # More structured groups already carry the corresponding lexical evidence.
    claimed = set(qualified) | set(technical)
    lexical = [term for term in lexical if term not in claimed]
    return {
        "qualified_identifiers": qualified,
        "quoted_literals": quoted,
        "technical_tokens": technical,
        "lexical_tokens": lexical,
    }


def candidate_blob(item: dict[str, Any]) -> str:
    fields = (
        item.get("generated_code"), item.get("raw_output"), item.get("error_type"),
        item.get("error_msg"), item.get("exec_output"), item.get("match_detail"),
        item.get("natural_language_query"), item.get("reference_code"),
    )
    return "\n".join(str(field) for field in fields if field).lower()


def score_candidate(
    row: dict[str, Any],
    item: dict[str, Any],
    exemplar: str | None,
) -> tuple[int, list[str], dict[str, list[str]], bool]:
    blob = candidate_blob(item)
    terms = evidence_terms(str(row["justification"]))
    matched: dict[str, list[str]] = {}
    weights = {
        "qualified_identifiers": 12,
        "quoted_literals": 8,
        "technical_tokens": 5,
        "lexical_tokens": 1,
    }
    score = 0
    reasons: list[str] = []
    for group, group_terms in terms.items():
        present = [term for term in group_terms if term in blob]
        if present:
            matched[group] = present
            contribution = weights[group] * len(present)
            if group == "lexical_tokens":
                contribution = min(contribution, 12)
            score += contribution
            reasons.append(f"{group}: " + ", ".join(present[:8]))

    exact_excerpt = False
    if exemplar:
        generated = normalized_code_prefix(str(item.get("generated_code") or ""))
        exact_excerpt = bool(exemplar and generated.startswith(exemplar))
        if exact_excerpt:
            score += 100
            reasons.insert(0, "archived exemplar code prefix matches exactly")

    # Execution state is weak corroboration, not uniqueness evidence.
    error_name = str(item.get("error_type") or "").lower()
    if error_name and error_name in str(row["justification"]).lower():
        score += 3
        reasons.append(f"error_type matches: {item.get('error_type')}")

    return score, reasons, matched, exact_excerpt


def release_candidate(
    item: dict[str, Any],
    source_rel: str,
    source_sha: str,
    score: int,
    reasons: list[str],
    matched: dict[str, list[str]],
    exact_excerpt: bool,
) -> dict[str, Any]:
    return {
        "item_id": item.get("item_id"),
        "bench_index": item.get("bench_index"),
        "benchmark_original_index": item.get("benchmark_original_index"),
        "source_path": source_rel,
        "source_sha256": source_sha,
        "evidence_score": score,
        "exact_archived_exemplar_prefix": exact_excerpt,
        "matching_reasons": reasons,
        "matched_evidence": matched,
        "generated_code_excerpt": code_excerpt(item.get("generated_code")),
        "error_type": item.get("error_type"),
        "error_excerpt": short_text(item.get("error_msg"), 420),
        "execution_output_excerpt": short_text(item.get("exec_output"), 320),
    }


def decide_status(candidates: list[dict[str, Any]]) -> tuple[str, str]:
    if not candidates:
        return "zero_candidates", "no match-false item in the exact frozen stratum"
    if len(candidates) == 1:
        return "unique_forensic_link", "only match-false item in exact model/condition/task/network stratum"

    exact = [candidate for candidate in candidates if candidate["exact_archived_exemplar_prefix"]]
    if len(exact) == 1:
        return "unique_forensic_link", "one candidate exactly matches the archived exemplar code prefix"

    ranked = sorted(candidates, key=lambda item: (-item["evidence_score"], item["bench_index"]))
    top, runner = ranked[0], ranked[1]
    structured_matches = (
        len(top["matched_evidence"].get("qualified_identifiers", []))
        + len(top["matched_evidence"].get("quoted_literals", []))
        + len(top["matched_evidence"].get("technical_tokens", []))
    )
    if (
        structured_matches >= 2
        and top["evidence_score"] >= 30
        and top["evidence_score"] - runner["evidence_score"] >= 12
    ):
        return (
            "unique_forensic_link",
            "one candidate has a separated multi-token technical signature",
        )
    return "ambiguous_candidates", "exact stratum and text evidence do not establish a unique item"


def validate_inputs(ledger: dict[str, Any], compact: dict[str, Any]) -> list[dict[str, Any]]:
    rows = ledger["classifications"]
    expected_ids = {f"s{number:03d}" for number in range(1, 101)}
    if len(rows) != 100 or {row["sample_id"] for row in rows} != expected_ids:
        raise ValueError("expected complete s001--s100 taxonomy ledger")
    selected = [
        run for run in compact["runs"].values()
        if run.get("group") == "comparison" and run.get("condition") in POOL_CONDITIONS
    ]
    if len(selected) != 55:
        raise ValueError(f"expected 55 source runs, found {len(selected)}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="root of the original result repository")
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--exemplars", type=Path, default=DEFAULT_EXEMPLARS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-candidates", type=int, default=20)
    args = parser.parse_args()
    if args.max_candidates < 2:
        raise ValueError("--max-candidates must be at least 2")

    source_root = args.source_root.resolve()
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    selected = validate_inputs(ledger, compact)
    # The full compact archive is no longer needed after its 55 lightweight
    # source descriptors have been selected.
    del compact
    gc.collect()
    exemplars = parse_exemplars(args.exemplars)

    rows = ledger["classifications"]
    rows_by_source: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_source[(str(row["model"]), normalize_condition(str(row["condition"])))].append(row)

    candidates_by_id: dict[str, list[dict[str, Any]]] = {row["sample_id"]: [] for row in rows}
    source_audit: list[dict[str, Any]] = []
    total_source_bytes = 0

    for run in sorted(selected, key=lambda value: (value["model"], value["condition"])):
        source = run["source"]
        source_rel = str(source["path"])
        source_path = (source_root / source_rel).resolve()
        if source_path != source_root and source_root not in source_path.parents:
            raise ValueError(f"source path escapes source root: {source_rel}")
        actual_sha = sha256(source_path)
        if actual_sha != source["sha256"]:
            raise ValueError(f"source hash mismatch: {source_rel}")
        if source_path.stat().st_size != source["size_bytes"]:
            raise ValueError(f"source size mismatch: {source_rel}")

        model = MODEL_ALIASES.get(str(run["model"]))
        if model is None:
            raise ValueError(f"unmapped source model: {run['model']}")
        source_rows = rows_by_source[(model, str(run["condition"]))]

        target_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in source_rows:
            target_rows[(str(row["task"]), str(row["network"]))].append(row)

        n_items = 0
        n_match_false = 0
        for item in iter_item_results(source_path):
            n_items += 1
            if item.get("match") is not False:
                continue
            n_match_false += 1
            key = (str(item.get("task")), str(item.get("network")))
            for row in target_rows.get(key, []):
                sample_id = str(row["sample_id"])
                score, reasons, matched, exact = score_candidate(
                    row, item, exemplars.get(sample_id)
                )
                candidates_by_id[sample_id].append(release_candidate(
                    item, source_rel, actual_sha, score, reasons, matched, exact
                ))
        if n_items != 2000:
            raise ValueError(f"expected 2,000 item_results, found {n_items}: {source_rel}")

        total_source_bytes += source_path.stat().st_size
        source_audit.append({
            "source_path": source_rel,
            "sha256": actual_sha,
            "size_bytes": source_path.stat().st_size,
            "n_item_results": n_items,
            "n_match_false": n_match_false,
            "n_taxonomy_rows_searched": len(source_rows),
        })
        gc.collect()

    linkage_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    truncation_count = 0
    for row in sorted(rows, key=lambda item: int(item["sample_id"][1:])):
        all_candidates = sorted(
            candidates_by_id[row["sample_id"]],
            key=lambda item: (-item["evidence_score"], item["bench_index"]),
        )
        status, basis = decide_status(all_candidates)
        status_counts[status] += 1
        retained = all_candidates[: args.max_candidates]
        if len(retained) < len(all_candidates):
            truncation_count += 1
        unique_link = None
        if status == "unique_forensic_link":
            exact_links = [
                candidate for candidate in all_candidates
                if candidate["exact_archived_exemplar_prefix"]
            ]
            unique_link = exact_links[0] if len(exact_links) == 1 else all_candidates[0]
        linkage_rows.append({
            "sample_id": row["sample_id"],
            "taxonomy_stratum": {
                "model": row["model"],
                "condition": row["condition"],
                "task": row["task"],
                "network": row["network"],
            },
            "archived_category_id": row["category_id"],
            "archived_justification": row["justification"],
            "linkage_status": status,
            "linkage_basis": basis,
            "candidate_count": len(all_candidates),
            "retained_candidate_count": len(retained),
            "unique_item_id": unique_link["item_id"] if unique_link else None,
            "unique_bench_index": unique_link["bench_index"] if unique_link else None,
            "candidates": retained,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": 1,
        "description": (
            "Conservative forensic linkage candidates for the historical 100-item "
            "failure taxonomy. Archived labels are copied unchanged."
        ),
        "method": {
            "candidate_frame": "match=false within exact model/condition/task/network",
            "evidence": (
                "archived exemplar code prefix plus identifiers, literals, error classes, "
                "and lexical evidence in the archived justification"
            ),
            "unique_rule": (
                "single exact-stratum candidate, one exact exemplar-prefix match, or a "
                "separated multi-token technical signature; otherwise ambiguous"
            ),
            "non_claim": (
                "This is a forensic reconstruction, not a replay of the unretained "
                "historical round-robin/stride-skip sampler."
            ),
            "max_candidates_retained_per_sample": args.max_candidates,
        },
        "source_verification": {
            "n_files": len(source_audit),
            "total_bytes": total_source_bytes,
            "all_sha256_verified": True,
            "files": source_audit,
        },
        "summary": dict(sorted(status_counts.items())),
        "samples_with_truncated_candidate_lists": truncation_count,
        "links": linkage_rows,
    }
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    max_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "schema_version": 1,
        "status": "ok",
        "output_path": str(args.out.resolve().relative_to(ROOT)),
        "output_sha256": sha256(args.out),
        "counts": {
            "taxonomy_rows": len(linkage_rows),
            "source_files_verified": len(source_audit),
            "source_bytes_read_for_hash_and_parse": total_source_bytes * 2,
            **dict(sorted(status_counts.items())),
            "candidate_lists_truncated": truncation_count,
        },
        "resource_discipline": {
            "execution": "single process; source files loaded sequentially; no model loaded",
            "peak_rss_kib": max_rss_kib,
        },
        "checks": {
            "all_source_hashes_verified": True,
            "all_100_rows_reported": len(linkage_rows) == 100,
            "statuses_exhaustive": sum(status_counts.values()) == 100,
            "archived_labels_unchanged": all(
                linked["archived_category_id"] == row["category_id"]
                for linked, row in zip(linkage_rows, sorted(rows, key=lambda item: int(item["sample_id"][1:])))
            ),
        },
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "counts": report["counts"],
        "peak_rss_kib": max_rss_kib,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
