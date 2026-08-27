#!/usr/bin/env python3
"""Export and aggregate the item-level evidence behind the primary results.

Default operation reads the repository-local compact archive and rebuilds
the numeric views needed for main-text Tables 3--6, headline lifts, paired
bootstrap intervals, lexical/dense RAG comparisons, and the naturalistic
holdout.  ``--export-source-root`` rebuilds the archive from the original
Git-LFS run tree, which is that export's only input.

The archive uses shared, lexicographically sorted item axes.  Boolean fields
are bit strings and token fields are integer arrays.  Every run records the
source file SHA-256/size and a canonical item-vector SHA-256.  Condition-A
diagnostic text is losslessly dictionary encoded for the E6 AUROC analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import statistics as st
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = ROOT / "results/raw/main_experiment/primary_outcomes_compact.json"
DEFAULT_AUDIT_COPY = ROOT / "results/raw/main_experiment/bench_function_audit.json"
DEFAULT_OUTPUT = ROOT / "results/aggregates/primary_results_from_compact.json"

OPEN_MODELS = [
    "Qwen_Qwen2.5-Coder-0.5B-Instruct",
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
PANEL10 = OPEN_MODELS[1:]
HEADLINE_OPEN = OPEN_MODELS[2:]
API_MODELS = [
    "gpt-5.4-mini", "claude-haiku-4-5",
    "deepseek-v4-flash", "gemini-2.5-flash",
]
MAIN_CONDITIONS = [
    "A", "B", "C", "X",
    "A_FX", "A_FD", "A_FDR", "A_FDRS",
    "C_FX", "C_FD", "C_FDR", "C_FDRS",
    "X_FX", "X_FD", "X_FDR", "X_FDRS",
]
API_CONDITIONS = ["A", "C", "C_FX", "C_FDR", "C_FDRS"]
NATURALISTIC_CONDITIONS = ["A_FX", "C_FX", "C_FDR", "C_FDRS"]

T3_OPEN = [
    "meta-llama_Llama-3.1-405B-Instruct",
    "Qwen_Qwen3-Coder-480B-A35B-Instruct",
    "openai_gpt-oss-120b",
    "Qwen_Qwen3-Coder-Next",
    "meta-llama_Llama-3.1-70B-Instruct",
    "Qwen_Qwen2.5-Coder-32B-Instruct",
]
MW2_DELTAS = [
    ("C+FDRS", "C+FDR", "C_FDRS", "C_FDR"),
    ("C+FDR", "C+FD", "C_FDR", "C_FD"),
    ("C+FX", "A+FX", "C_FX", "A_FX"),
    ("C", "A", "C", "A"),
    ("C+FDRS", "C+FX", "C_FDRS", "C_FX"),
]

# Execution exceptions can embed the maintainer's virtual-environment prefix,
# either absolute or with ``../../`` traceback notation.  Only the package-
# relative suffix is analytically relevant (E6 looks for API names in the
# message).  Redact the prefix ending at the standard site-packages boundary
# while preserving that suffix byte-for-byte.
ABSOLUTE_SITE_PACKAGES = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:(?:\.\./)+|/)(?:[^/\s()'\"<>]+/)+site-packages/"
)
DIAGNOSTIC_ENV_PLACEHOLDER = "<PYTHON_ENV>/site-packages/"


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def source_specs(source_root: Path) -> list[dict]:
    base = source_root / "probe_eval_results"
    specs = []
    for model in OPEN_MODELS:
        for condition in MAIN_CONDITIONS:
            specs.append({
                "group": "comparison", "dataset": "powercodebench_2000",
                "model": model, "condition": condition,
                "path": base / "comparison" / model
                        / f"benchmark_results_cond{condition}.json",
            })
    for model in API_MODELS:
        for condition in API_CONDITIONS:
            specs.append({
                "group": "api", "dataset": "powercodebench_2000",
                "model": model, "condition": condition,
                "path": base / "api_comparison_2000" / model
                        / f"benchmark_results_cond{condition}.json",
            })
    for model in PANEL10:
        specs.append({
            "group": "bm25", "dataset": "powercodebench_2000",
            "model": model, "condition": "R",
            "path": base / "comparison_bm25rag" / model
                    / "benchmark_results_condR.json",
        })
        for condition in ("Rsem", "RsemB"):
            specs.append({
                "group": "semantic", "dataset": "powercodebench_2000",
                "model": model, "condition": condition,
                "path": base / "comparison_e5_semantic_rag" / model
                        / f"benchmark_results_cond{condition}.json",
            })
    for model in OPEN_MODELS:
        for condition in NATURALISTIC_CONDITIONS:
            specs.append({
                "group": "naturalistic", "dataset": "naturalistic_holdout_80",
                "model": model, "condition": condition,
                "path": base / "naturalistic_holdout" / model
                        / f"benchmark_results_cond{condition}.json",
            })
    return specs


def run_key(group: str, model: str, condition: str) -> str:
    return f"{group}|{model}|{condition}"


def file_payload(path: Path) -> tuple[bytes, str]:
    payload = path.read_bytes()
    return payload, digest_bytes(payload)


def compact_fix_stats(summary: dict) -> dict:
    fix = summary.get("fix_prompt_token_stats") or {}
    keep = {
        key: fix.get(key)
        for key in (
            "attempts", "prompt", "docs", "docs_total", "docs_nonempty",
            "docs_nonempty_rate", "successful_fixes",
            "prompt_tokens_per_successful_fix", "per_round",
        )
        if key in fix
    }
    return {
        "round0_prompt_token_stats": summary.get("round0_prompt_token_stats"),
        "fix_prompt_token_stats": keep,
        "total_prompt_token_stats": summary.get("total_prompt_token_stats"),
    }


def vector_hash(
    item_ids: list[str], executed: str, matched: str,
    prompt_tokens: list, prompt_tokens_approx: list,
) -> str:
    rows = (
        f"{iid}\t{exe}\t{mat}\t{tok}\t{approx}\n"
        for iid, exe, mat, tok, approx in zip(
            item_ids, executed, matched, prompt_tokens, prompt_tokens_approx
        )
    )
    return digest_bytes("".join(rows).encode("utf-8"))


def diagnostics_hash(
    item_ids: list[str], messages: list, correct_task_fn: list,
) -> str:
    rows = (
        canonical_json([iid, msg, value]) + "\n"
        for iid, msg, value in zip(item_ids, messages, correct_task_fn)
    )
    return digest_bytes("".join(rows).encode("utf-8"))


def normalize_diagnostic_message(message):
    """Return (portable message, number of redacted environment prefixes)."""
    if not isinstance(message, str):
        return message, 0
    return ABSOLUTE_SITE_PACKAGES.subn(DIAGNOSTIC_ENV_PLACEHOLDER, message)


def git_source_state(source_root: Path, paths: list[Path]) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    relative = [str(path.relative_to(source_root)) for path in paths]
    for cached in (False, True):
        cmd = ["git", "diff", "--quiet"]
        if cached:
            cmd.append("--cached")
        cmd.extend(["--", *relative])
        if subprocess.run(cmd, cwd=source_root).returncode != 0:
            raise ValueError("selected source paths have uncommitted changes")
    return commit


def build_injection_compact(
    injection: dict, item_ids: list[str], source: dict,
) -> dict:
    functions = sorted({
        entry.get("function")
        for model_data in injection.values()
        for item in model_data.values()
        for entry in (item.get("selected_layers") or [])
        if isinstance(entry.get("function"), str)
    })
    layers = sorted({
        entry.get("layer")
        for model_data in injection.values()
        for item in model_data.values()
        for entry in (item.get("selected_layers") or [])
        if isinstance(entry.get("layer"), str)
    })
    f_index = {value: i for i, value in enumerate(functions)}
    l_index = {value: i for i, value in enumerate(layers)}
    models = {}
    for model in PANEL10:
        model_data = injection.get(model)
        if not isinstance(model_data, dict):
            raise ValueError(f"injection log lacks model {model}")
        selected, error_bits = [], []
        for item_id in item_ids:
            item = model_data.get(item_id)
            if not isinstance(item, dict):
                raise ValueError(f"injection log lacks {model}/{item_id}")
            pairs = sorted({
                (f_index[row["function"]], l_index[row["layer"]])
                for row in (item.get("selected_layers") or [])
                if row.get("function") in f_index and row.get("layer") in l_index
            })
            selected.append([list(pair) for pair in pairs])
            error_bits.append("1" if "error" in item else "0")
        payload = canonical_json([
            [item_id, pairs, error]
            for item_id, pairs, error in zip(item_ids, selected, error_bits)
        ]).encode("utf-8")
        models[model] = {
            "selected_function_layer_pairs": selected,
            "error_bits": "".join(error_bits),
            "vector_sha256": digest_bytes(payload),
        }
    return {
        "description": (
            "Lossless function/layer projection of selected_layers, aligned to "
            "datasets.powercodebench_2000.item_ids; sufficient for Exp-D Stat 2."
        ),
        "function_dictionary": functions,
        "layer_dictionary": layers,
        "models": models,
        "source": source,
    }


def export_archive(source_root: Path, audit_copy: Path) -> dict:
    specs = source_specs(source_root)
    auxiliary_paths = [
        source_root / "probe_eval_results/bench_function_audit.json",
        source_root / "probe_eval_results/injection_log.json",
    ]
    all_paths = [spec["path"] for spec in specs] + auxiliary_paths
    missing = [str(path) for path in all_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing source files:\n" + "\n".join(missing))
    source_commit = git_source_state(source_root, all_paths)

    datasets: dict[str, dict] = {}
    runs = {}
    for number, spec in enumerate(specs, start=1):
        payload, raw_sha256 = file_payload(spec["path"])
        data = json.loads(payload)
        items = data.get("item_results")
        if not isinstance(items, list) or not items:
            raise ValueError(f"no item_results in {spec['path']}")
        by_id = {}
        for item in items:
            iid = item.get("item_id")
            if not isinstance(iid, str) or iid in by_id:
                raise ValueError(f"invalid/duplicate item_id in {spec['path']}")
            if not isinstance(item.get("executed"), bool):
                raise ValueError(f"non-boolean executed in {spec['path']}")
            if not isinstance(item.get("match"), bool):
                raise ValueError(f"non-boolean match in {spec['path']}")
            by_id[iid] = item
        dataset = spec["dataset"]
        ids = sorted(by_id)
        if dataset not in datasets:
            tasks = [by_id[iid].get("task") for iid in ids]
            difficulties = [by_id[iid].get("difficulty_level") for iid in ids]
            metadata = [[iid, task, difficulty]
                        for iid, task, difficulty in zip(ids, tasks, difficulties)]
            datasets[dataset] = {
                "n_items": len(ids), "item_order": "lexicographic item_id",
                "item_ids": ids, "task": tasks, "difficulty": difficulties,
                "metadata_sha256": digest_bytes(canonical_json(metadata).encode("utf-8")),
            }
        axis = datasets[dataset]
        if ids != axis["item_ids"]:
            raise ValueError(f"item axis mismatch in {spec['path']}")
        if any(
            by_id[iid].get("task") != task
            or by_id[iid].get("difficulty_level") != difficulty
            for iid, task, difficulty in zip(
                ids, axis["task"], axis["difficulty"]
            )
        ):
            raise ValueError(f"task/difficulty mismatch in {spec['path']}")

        executed = "".join("1" if by_id[iid]["executed"] else "0" for iid in ids)
        matched = "".join("1" if by_id[iid]["match"] else "0" for iid in ids)
        tokens = [by_id[iid].get("prompt_tokens") for iid in ids]
        approximate = [by_id[iid].get("prompt_tokens_approx") for iid in ids]
        if any(value is not None and not isinstance(value, int)
               for value in tokens + approximate):
            raise ValueError(f"non-integral prompt token field in {spec['path']}")
        summary = data.get("summary") or {}
        if summary.get("n_executed") != executed.count("1"):
            raise ValueError(f"source execution summary mismatch in {spec['path']}")
        if summary.get("n_matched") != matched.count("1"):
            raise ValueError(f"source match summary mismatch in {spec['path']}")
        snapshots = [
            {"round": row.get("round"), "total": row.get("total"),
             "n_executed": row.get("n_executed"), "n_matched": row.get("n_matched")}
            for row in (data.get("round_snapshots") or [])
        ]
        row = {
            "dataset": dataset, "group": spec["group"],
            "model": spec["model"], "condition": spec["condition"],
            "executed_bits": executed, "match_bits": matched,
            "prompt_tokens": tokens, "prompt_tokens_approx": approximate,
            "round_snapshots": snapshots,
            "token_summary": compact_fix_stats(summary),
            "vector_sha256": vector_hash(ids, executed, matched, tokens, approximate),
            "source": {
                "path": str(spec["path"].relative_to(source_root)),
                "commit": source_commit, "sha256": raw_sha256,
                "size_bytes": len(payload), "git_lfs_object": True,
            },
        }
        if spec["group"] == "comparison" and spec["condition"] == "A":
            original_messages = [by_id[iid].get("error_msg") for iid in ids]
            normalized = [normalize_diagnostic_message(message)
                          for message in original_messages]
            messages = [message for message, _ in normalized]
            substitution_count = sum(count for _, count in normalized)
            changed_item_occurrences = sum(
                original != portable
                for original, portable in zip(original_messages, messages)
            )
            correct = [
                (by_id[iid].get("diagnostics") or {}).get("correct_task_fn")
                for iid in ids
            ]
            dictionary = []
            dictionary_index = {}
            indices = []
            for message in messages:
                key = canonical_json(message)
                if key not in dictionary_index:
                    dictionary_index[key] = len(dictionary)
                    dictionary.append(message)
                indices.append(dictionary_index[key])
            row["condition_a_diagnostics"] = {
                "error_msg_dictionary": dictionary,
                "error_msg_index": indices,
                "correct_task_fn": correct,
                "diagnostics_sha256": diagnostics_hash(ids, messages, correct),
                "path_normalization": {
                    "scheme": "python-env-site-packages-prefix-v2",
                    "placeholder": DIAGNOSTIC_ENV_PLACEHOLDER,
                    "changed_dictionary_entries": sum(
                        isinstance(message, str)
                        and DIAGNOSTIC_ENV_PLACEHOLDER in message
                        for message in dictionary
                    ),
                    "changed_item_occurrences": changed_item_occurrences,
                    "prefix_substitutions_in_item_messages": substitution_count,
                    "pre_normalization_diagnostics_sha256": diagnostics_hash(
                        ids, original_messages, correct
                    ),
                },
            }
        key = run_key(spec["group"], spec["model"], spec["condition"])
        runs[key] = row
        if number % 10 == 0 or number == len(specs):
            print(f"exported {number}/{len(specs)} runs", flush=True)

    audit_payload, audit_sha256 = file_payload(auxiliary_paths[0])
    audit = json.loads(audit_payload)
    main_ids = datasets["powercodebench_2000"]["item_ids"]
    if set(audit) != set(main_ids):
        raise ValueError("bench_function_audit item set does not match main axis")
    compact_audit = {
        "used_functions": [audit[iid].get("used_functions") or [] for iid in main_ids],
        "used_attributes": [audit[iid].get("used_attributes") or [] for iid in main_ids],
        "source": {
            "path": str(auxiliary_paths[0].relative_to(source_root)),
            "commit": source_commit, "sha256": audit_sha256,
            "size_bytes": len(audit_payload),
            # This is provenance for the archive's published companion file,
            # not the caller's staging destination.  Keeping the logical
            # repository path stable also makes out-of-tree re-exports
            # byte-comparable with the checked-in compact archive.
            "repository_copy": str(DEFAULT_AUDIT_COPY.relative_to(ROOT)),
        },
    }
    audit_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(auxiliary_paths[0], audit_copy)

    injection_payload, injection_sha256 = file_payload(auxiliary_paths[1])
    injection = json.loads(injection_payload)
    compact_injection = build_injection_compact(
        injection, main_ids,
        {"path": str(auxiliary_paths[1].relative_to(source_root)),
         "commit": source_commit, "sha256": injection_sha256,
         "size_bytes": len(injection_payload)},
    )
    return {
        "schema_version": 1,
        "description": (
            "Minimum sufficient item-level archive for the primary experimental "
            "tables, paired resampling, token economy, RAG controls, holdout, "
            "E6 diagnostics, and Exp-D injection analysis."
        ),
        "provenance": {
            "source_repository": "frozen experimental pipeline",
            "source_commit": source_commit,
            "selected_paths_clean_at_export": True,
            "source_values": "raw item_results and raw-run summary blocks only",
            "not_used_as_export_inputs": ["paper", "results/aggregates"],
            "round_snapshot_granularity": (
                "R0--R3 aggregate cumulative counts; original run files do not "
                "archive item-level intermediate-round match vectors"
            ),
        },
        "datasets": datasets, "runs": runs,
        "bench_function_audit_compact": compact_audit,
        "injection_log_compact": compact_injection,
    }


def validate_archive(data: dict) -> None:
    if data.get("schema_version") != 1:
        raise ValueError("unsupported archive schema")
    datasets = data.get("datasets") or {}
    expected_n = {"powercodebench_2000": 2000, "naturalistic_holdout_80": 80}
    for name, n_expected in expected_n.items():
        axis = datasets.get(name) or {}
        ids = axis.get("item_ids")
        tasks = axis.get("task")
        difficulties = axis.get("difficulty")
        if not (isinstance(ids, list) and ids == sorted(ids)
                and len(ids) == len(set(ids)) == n_expected):
            raise ValueError(f"invalid dataset axis {name}")
        if len(tasks) != n_expected or len(difficulties) != n_expected:
            raise ValueError(f"invalid metadata axis {name}")
        metadata = [[iid, task, difficulty]
                    for iid, task, difficulty in zip(ids, tasks, difficulties)]
        if digest_bytes(canonical_json(metadata).encode("utf-8")) \
                != axis.get("metadata_sha256"):
            raise ValueError(f"metadata hash mismatch for {name}")
    expected_keys = {
        run_key(spec["group"], spec["model"], spec["condition"])
        for spec in source_specs(Path("SOURCE_ROOT_PLACEHOLDER"))
    }
    if set(data.get("runs", {})) != expected_keys:
        raise ValueError("archive run set is incomplete or has unexpected runs")
    for key, row in data["runs"].items():
        ids = datasets[row["dataset"]]["item_ids"]
        executed, matched = row["executed_bits"], row["match_bits"]
        tokens, approximate = row["prompt_tokens"], row["prompt_tokens_approx"]
        if (len(executed) != len(ids) or len(matched) != len(ids)
                or set(executed) - {"0", "1"} or set(matched) - {"0", "1"}
                or len(tokens) != len(ids) or len(approximate) != len(ids)):
            raise ValueError(f"invalid vector lengths in {key}")
        if vector_hash(ids, executed, matched, tokens, approximate) \
                != row.get("vector_sha256"):
            raise ValueError(f"vector hash mismatch in {key}")
        diagnostic = row.get("condition_a_diagnostics")
        if diagnostic is not None:
            dictionary = diagnostic["error_msg_dictionary"]
            if (len(diagnostic["error_msg_index"]) != len(ids)
                    or len(diagnostic["correct_task_fn"]) != len(ids)):
                raise ValueError(f"condition-A diagnostic length mismatch in {key}")
            messages = [dictionary[index] for index in diagnostic["error_msg_index"]]
            correct = diagnostic["correct_task_fn"]
            if diagnostics_hash(ids, messages, correct) \
                    != diagnostic.get("diagnostics_sha256"):
                raise ValueError(f"condition-A diagnostic hash mismatch in {key}")
            if any(
                count
                for message in dictionary
                for _, count in [normalize_diagnostic_message(message)]
            ):
                raise ValueError(f"condition-A diagnostic contains an absolute env path: {key}")

    main_ids = datasets["powercodebench_2000"]["item_ids"]
    audit = data.get("bench_function_audit_compact") or {}
    if (len(audit.get("used_functions", [])) != len(main_ids)
            or len(audit.get("used_attributes", [])) != len(main_ids)):
        raise ValueError("bench-function compact view has the wrong item count")
    audit_source = audit.get("source") or {}
    audit_copy = ROOT / audit_source.get("repository_copy", "MISSING")
    if audit_copy.is_file() and digest_bytes(audit_copy.read_bytes()) \
            != audit_source.get("sha256"):
        raise ValueError("byte-exact bench_function_audit copy has the wrong hash")

    injection = data.get("injection_log_compact") or {}
    functions = injection.get("function_dictionary") or []
    layers = injection.get("layer_dictionary") or []
    if set(injection.get("models", {})) != set(PANEL10):
        raise ValueError("injection compact view does not contain the panel")
    for model, model_row in injection["models"].items():
        selected = model_row.get("selected_function_layer_pairs") or []
        errors = model_row.get("error_bits")
        if (len(selected) != len(main_ids) or not isinstance(errors, str)
                or len(errors) != len(main_ids) or set(errors) - {"0", "1"}):
            raise ValueError(f"injection compact vector length mismatch for {model}")
        for pairs in selected:
            if any(
                not (isinstance(pair, list) and len(pair) == 2
                     and isinstance(pair[0], int) and 0 <= pair[0] < len(functions)
                     and isinstance(pair[1], int) and 0 <= pair[1] < len(layers))
                for pair in pairs
            ):
                raise ValueError(f"invalid injection function/layer index for {model}")
        payload = canonical_json([
            [item_id, pairs, error]
            for item_id, pairs, error in zip(main_ids, selected, errors)
        ]).encode("utf-8")
        if digest_bytes(payload) != model_row.get("vector_sha256"):
            raise ValueError(f"injection compact vector hash mismatch for {model}")


def get_run(data: dict, group: str, model: str, condition: str) -> dict:
    return data["runs"][run_key(group, model, condition)]


def endpoint(row: dict) -> dict:
    n = len(row["match_bits"])
    n_executed = row["executed_bits"].count("1")
    n_matched = row["match_bits"].count("1")
    return {
        "n": n, "n_executed": n_executed, "n_matched": n_matched,
        "execution_rate": n_executed / n, "accuracy": n_matched / n,
        "accuracy_of_executed": n_matched / n_executed if n_executed else None,
    }


def stratified_endpoint(row: dict, axis: dict, field: str) -> dict:
    groups = defaultdict(list)
    for index, value in enumerate(axis[field]):
        groups[value].append(index)
    result = {}
    for value, indices in sorted(groups.items()):
        n_executed = sum(row["executed_bits"][index] == "1" for index in indices)
        n_matched = sum(row["match_bits"][index] == "1" for index in indices)
        result[value] = {
            "n": len(indices), "n_executed": n_executed, "n_matched": n_matched,
            "execution_rate": n_executed / len(indices),
            "accuracy": n_matched / len(indices),
        }
    return result


def run_summaries(data: dict) -> dict:
    out = {}
    for key, row in data["runs"].items():
        axis = data["datasets"][row["dataset"]]
        out[key] = {
            **endpoint(row), "round_snapshots": row["round_snapshots"],
            "token_summary": row["token_summary"],
            "per_task": stratified_endpoint(row, axis, "task"),
            "per_difficulty": stratified_endpoint(row, axis, "difficulty"),
        }
    return out


def panel_mean_accuracy(data: dict, base: str, suffix: str | None) -> float:
    condition = base if suffix is None else f"{base}_{suffix}"
    return st.mean(endpoint(get_run(data, "comparison", model, condition))["accuracy"]
                   for model in PANEL10)


def bootstrap_hero(data: dict) -> dict:
    llama = get_run(data, "comparison", T3_OPEN[0], "C_FDRS")["match_bits"]
    claude = get_run(data, "api", "claude-haiku-4-5", "C_FDRS")["match_bits"]
    n, b, seed = len(llama), 1000, 22

    def distribution(bits: str) -> list[float]:
        rng = random.Random(seed)
        values = [int(bit) for bit in bits]
        return sorted(
            100 * sum(values[rng.randrange(n)] for _ in range(n)) / n
            for _ in range(b)
        )

    llama_bs, claude_bs = distribution(llama), distribution(claude)
    rng = random.Random(seed)
    deltas = [int(a) - int(c) for a, c in zip(llama, claude)]
    gap_bs = sorted(
        100 * sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(b)
    )
    return {
        "B": b, "seed": seed, "n_items": n,
        "llama_405b_acc_pct": 100 * llama.count("1") / n,
        "claude_haiku_acc_pct": 100 * claude.count("1") / n,
        "llama_ci95_pct": [llama_bs[int(.025*b)], llama_bs[int(.975*b)]],
        "claude_ci95_pct": [claude_bs[int(.025*b)], claude_bs[int(.975*b)]],
        "paired_delta_bootstrap_mean_pp": st.mean(gap_bs),
        "paired_delta_ci95_pp": [gap_bs[int(.025*b)], gap_bs[int(.975*b)]],
    }


def bootstrap_mw2(data: dict) -> dict:
    b, seed, n = 2000, 22, 2000
    rng = random.Random(seed)
    rows = []
    for label_a, label_b, condition_a, condition_b in MW2_DELTAS:
        item_values = []
        for index in range(n):
            value = st.mean(
                (int(get_run(data, "comparison", model, condition_a)["match_bits"][index])
                 - int(get_run(data, "comparison", model, condition_b)["match_bits"][index]))
                * 100
                for model in PANEL10
            )
            item_values.append(value)
        point = st.mean(item_values)
        values = sorted(
            sum(item_values[rng.randrange(n)] for _ in range(n)) / n
            for _ in range(b)
        )
        lo, hi = values[int(.025*b)], values[int(.975*b)-1]
        rows.append({
            "delta_label": f"{label_a} - {label_b}",
            "condition_a": condition_a, "condition_b": condition_b,
            "point_pp": point, "boot_median_pp": values[b//2],
            "ci95_pp": [lo, hi], "strict_positive": lo > 0,
            "strict_negative": hi < 0,
        })
    return {"B": b, "seed": seed, "n_items": n, "n_panel": 10, "rows": rows}


def token_economy(data: dict) -> dict:
    proactive = {}
    for condition in ("A", "B", "C", "X", "R"):
        per_model = []
        for model in PANEL10:
            group = "bm25" if condition == "R" else "comparison"
            row = get_run(data, group, model, condition)
            stats = row["token_summary"]["round0_prompt_token_stats"]
            prompt_total = sum(value for value in row["prompt_tokens"]
                               if value is not None)
            per_model.append({
                "model": model, "prompt_avg": stats["avg"],
                "prompt_total": stats["total"],
                "prompt_avg_exact_from_items": prompt_total / len(row["prompt_tokens"]),
                "n_matched": row["match_bits"].count("1"),
            })
        prompt_avg = st.mean(item["prompt_avg"] for item in per_model)
        prompt_avg_exact = st.mean(
            item["prompt_avg_exact_from_items"] for item in per_model)
        matched_avg = st.mean(item["n_matched"] for item in per_model)
        pooled_prompt_total = sum(item["prompt_total"] for item in per_model)
        pooled_matched = sum(item["n_matched"] for item in per_model)
        a_avg = st.mean(
            get_run(data, "comparison", model, "A")["token_summary"]
                ["round0_prompt_token_stats"]["avg"]
            for model in PANEL10
        )
        proactive[condition] = {
            "per_model": per_model, "prompt_avg": prompt_avg,
            "prompt_avg_exact_from_items": prompt_avg_exact,
            "docs_avg": 0 if condition == "A" else prompt_avg - a_avg,
            "successes_avg": matched_avg,
            "prompt_tokens_per_match": prompt_avg * 2000 / matched_avg,
            "prompt_tokens_per_match_exact_from_items":
                prompt_avg_exact * 2000 / matched_avg,
            "pooled_exact": {
                "n_items": 2000 * len(per_model),
                "prompt_total": pooled_prompt_total,
                "n_successes": pooled_matched,
                "prompt_avg": pooled_prompt_total / (2000 * len(per_model)),
                "prompt_tokens_per_success":
                    pooled_prompt_total / pooled_matched,
            },
        }
    x_prompt = proactive["X"]["prompt_avg"]
    for row in proactive.values():
        row["vs_x_prompt_ratio"] = row["prompt_avg"] / x_prompt

    reactive = {}
    for condition in ("C_FX", "C_FD", "C_FDR", "C_FDRS"):
        per_model = []
        for model in PANEL10:
            fix = get_run(data, "comparison", model, condition)["token_summary"] \
                ["fix_prompt_token_stats"]
            per_model.append({
                "model": model, "docs_avg": fix["docs"]["avg"],
                "prompt_avg": fix["prompt"]["avg"],
                "attempts": fix["attempts"],
                "docs_total": fix["docs"]["total"],
                "prompt_total": fix["prompt"]["total"],
                "docs_avg_exact": fix["docs"]["total"] / fix["attempts"],
                "prompt_avg_exact": fix["prompt"]["total"] / fix["attempts"],
                "successful_fixes": fix["successful_fixes"],
                "prompt_tokens_per_successful_fix":
                    fix["prompt_tokens_per_successful_fix"],
            })
        reactive[condition] = {
            "per_model": per_model,
            "docs_avg": st.mean(item["docs_avg"] for item in per_model),
            "prompt_avg": st.mean(item["prompt_avg"] for item in per_model),
            "docs_avg_exact": st.mean(
                item["docs_avg_exact"] for item in per_model),
            "prompt_avg_exact": st.mean(
                item["prompt_avg_exact"] for item in per_model),
            "successful_fixes_avg": st.mean(
                item["successful_fixes"] for item in per_model),
            "prompt_tokens_per_successful_fix": st.mean(
                item["prompt_tokens_per_successful_fix"] for item in per_model),
            "pooled_exact": {
                "n_attempts": sum(item["attempts"] for item in per_model),
                "docs_total": sum(item["docs_total"] for item in per_model),
                "prompt_total": sum(item["prompt_total"] for item in per_model),
                "n_successes": sum(
                    item["successful_fixes"] for item in per_model),
            },
        }
        pooled = reactive[condition]["pooled_exact"]
        pooled["docs_avg"] = pooled["docs_total"] / pooled["n_attempts"]
        pooled["prompt_avg"] = pooled["prompt_total"] / pooled["n_attempts"]
        pooled["prompt_tokens_per_success"] = \
            pooled["prompt_total"] / pooled["n_successes"]
        reactive[condition]["vs_x_prompt_ratio"] = \
            reactive[condition]["prompt_avg"] / x_prompt
    c_x = [
        get_run(data, "comparison", model, "C")["token_summary"]
            ["round0_prompt_token_stats"]["avg"]
        / get_run(data, "comparison", model, "X")["token_summary"]
            ["round0_prompt_token_stats"]["avg"]
        for model in PANEL10
    ]
    return {
        "proactive": proactive, "reactive": reactive,
        "panel_mean_of_per_model_C_over_X": st.mean(c_x),
        "ratio_of_panel_mean_prompts_C_over_X":
            proactive["C"]["prompt_avg"] / proactive["X"]["prompt_avg"],
        "pooled_exact_display": {
            "proactive": {
                condition: {
                    "docs_avg": 0.0 if condition == "A" else
                        proactive[condition]["pooled_exact"]["prompt_avg"]
                        - proactive["A"]["pooled_exact"]["prompt_avg"],
                    "prompt_avg": proactive[condition]["pooled_exact"]
                                             ["prompt_avg"],
                    "successes_avg_per_model":
                        proactive[condition]["pooled_exact"]["n_successes"]
                        / len(PANEL10),
                    "prompt_tokens_per_success": proactive[condition]
                        ["pooled_exact"]["prompt_tokens_per_success"],
                    "vs_x_prompt_ratio": proactive[condition]["pooled_exact"]
                        ["prompt_avg"] / proactive["X"]["pooled_exact"]
                        ["prompt_avg"],
                }
                for condition in ("A", "B", "C", "X", "R")
            },
            "reactive": {
                condition: {
                    "docs_avg": reactive[condition]["pooled_exact"]["docs_avg"],
                    "prompt_avg": reactive[condition]["pooled_exact"]["prompt_avg"],
                    "successes_avg_per_model": reactive[condition]
                        ["pooled_exact"]["n_successes"] / len(PANEL10),
                    "prompt_tokens_per_success": reactive[condition]
                        ["pooled_exact"]["prompt_tokens_per_success"],
                    "vs_x_prompt_ratio": reactive[condition]["pooled_exact"]
                        ["prompt_avg"] / proactive["X"]["pooled_exact"]
                        ["prompt_avg"],
                }
                for condition in ("C_FX", "C_FD", "C_FDR", "C_FDRS")
            },
        },
    }


def aggregate(data: dict, archive_path: Path) -> dict:
    validate_archive(data)
    t3 = []
    for model in API_MODELS + T3_OPEN:
        group = "api" if model in API_MODELS else "comparison"
        conditions = API_CONDITIONS if group == "api" else API_CONDITIONS
        t3.append({
            "model": model, "group": group,
            **{condition: endpoint(get_run(data, group, model, condition))["accuracy"]
               for condition in conditions},
        })
    t4 = []
    for model in PANEL10:
        row = {"model": model}
        for condition in ("A", "B", "C", "X"):
            row[condition] = endpoint(
                get_run(data, "comparison", model, condition))["accuracy"]
        row["R"] = endpoint(get_run(data, "bm25", model, "R"))["accuracy"]
        for condition in ("Rsem", "RsemB"):
            row[condition] = endpoint(
                get_run(data, "semantic", model, condition))["accuracy"]
        t4.append(row)
    t5 = {
        base: {
            "R0": panel_mean_accuracy(data, base, None),
            **{suffix: panel_mean_accuracy(data, base, suffix)
               for suffix in ("FX", "FD", "FDR", "FDRS")},
        }
        for base in ("A", "C", "X")
    }
    headline = []
    for model in HEADLINE_OPEN + API_MODELS:
        group = "api" if model in API_MODELS else "comparison"
        a = endpoint(get_run(data, group, model, "A"))["accuracy"]
        fdrs = endpoint(get_run(data, group, model, "C_FDRS"))["accuracy"]
        headline.append({"model": model, "group": group, "A": a,
                         "C_FDRS": fdrs, "lift_pp": (fdrs-a)*100})
    naturalistic = {
        model: {
            condition: endpoint(get_run(data, "naturalistic", model, condition))
            for condition in NATURALISTIC_CONDITIONS
        }
        for model in OPEN_MODELS
    }
    return {
        "schema_version": 1,
        "source": {
            # Report the published logical archive path even when aggregation
            # is run against a byte-identical staging export outside the repo.
            "compact_archive": str(DEFAULT_ARCHIVE.relative_to(ROOT)),
            "compact_archive_sha256": digest_bytes(archive_path.read_bytes()),
            "original_repository_commit": data["provenance"]["source_commit"],
        },
        "coverage": {
            "n_runs": len(data["runs"]),
            "groups": dict(sorted(defaultdict(int, {
                group: sum(row["group"] == group for row in data["runs"].values())
                for group in {row["group"] for row in data["runs"].values()}
            }).items())),
            "round_snapshot_limit": data["provenance"]["round_snapshot_granularity"],
            "covered_claim_families": [
                "T3 cross-vendor endpoints and hero paired bootstrap",
                "T4 A/B/C/X and R/Rsem/RsemB item-level results",
                "T5 full A/C/X by FX/FD/FDR/FDRS matrix and endpoint pairing",
                "T6 proactive/reactive token sufficient statistics",
                "headline A-to-C_FDRS lifts",
                "naturalistic holdout endpoints",
                "condition-A E6 diagnostics and Exp-D injection functions",
            ],
            "outside_this_archive": [
                "probe profile raw responses (separate S2/S3 evidence package)",
                "reasoning-tier, backend-transfer, serving, and acceptance-audit studies",
                "generated code and verbose error distributions",
            ],
        },
        "run_summaries": run_summaries(data),
        "T3_cross_vendor": t3,
        "T4_proactive_and_rag": t4,
        "T5_reactive_matrix_panel_mean": t5,
        "T6_token_economy": token_economy(data),
        "headline_lifts": headline,
        "bootstrap": {"hero": bootstrap_hero(data), "panel_mw2": bootstrap_mw2(data)},
        "naturalistic_holdout": naturalistic,
        "auxiliary_evidence": {
            "condition_A_diagnostics_models": len(OPEN_MODELS),
            "bench_function_audit_items": len(
                data["bench_function_audit_compact"]["used_functions"]),
            "injection_log_models": len(data["injection_log_compact"]["models"]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--audit-copy", type=Path, default=DEFAULT_AUDIT_COPY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--export-source-root", type=Path,
        help="original igpt repository root; rebuilds the compact archive first",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_path = args.archive.resolve()
    if args.export_source_root is not None:
        data = export_archive(args.export_source_root.resolve(), args.audit_copy.resolve())
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_text(
            json.dumps(data, separators=(",", ":"), ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {archive_path}", flush=True)
    else:
        data = json.loads(archive_path.read_text(encoding="utf-8"))
    result = aggregate(data, archive_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
