#!/usr/bin/env python3
"""Reconstruct the two S4 cross-encoder rankings under strict frozen gates.

This is a maintainer-side recovery tool, not part of the lightweight public
reaggregation path.  It loads the historical task-demand implementation and
its three data inputs directly from one fixed Git revision, loads one pinned
CrossEncoder snapshot exactly once, and reconstructs unfiltered top-20 item
rankings for Zero-shot Cross-Encoder and Hybrid Cross-Encoder.

On the historical CUDA path, output is accepted as an exact reconstruction
only after all seven evaluation slices match the frozen recall/precision/
hit-rate cells at k={1,3,5,10,20}; the Hybrid arm must also reproduce the
complete alpha-selection curve, selected alpha, and role weights.  A CPU run
is accepted only with ``--revision-rerun`` and remains labelled as such even
if every aggregate cell matches.  Neither form is represented as a recovered
historical raw item log.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from export_s4_item_evidence import metrics, sha256_file


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
DEFAULT_SOURCE = ROOT.parent / "igpt"
SOURCE_REVISION = "7916953d4f289bb472224eae958935d1afb24ce5"
SUITE_RELATIVE = Path("task_demand/results/suites/suite_4428033")
LOG_RELATIVE = Path("task_demand/logs/task_demand_suite_4428033.out")
MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-12-v2"
MODEL_REVISION = "7b0235231ca2674cb8ca8f022859a6eba2b1c968"
EXPECTED_MODEL_FILES = {
    "config.json": "69a745055b0307584d2903ea5cf4e1899254e511834f358be80d049918b144d8",
    "model.safetensors": "1ed84b90cdf3518f76ec9bb93a16f97887eea7c4e7ee5dfb03cc297e394fbbc3",
    "special_tokens_map.json": "3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "tokenizer_config.json": "a5c2e5a7b1a29a0702cd28c08a399b5ecc110c263009d17f7e3b415f25905fd8",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
KS = (1, 3, 5, 10, 20)
EXPECTED_ENVIRONMENT = {
    "python": "3.11.15",
    "numpy": "2.2.6",
    "scikit_learn": "1.6.0",
    "torch": "2.9.1+cu129",
    "transformers": "4.57.6",
    "sentence_transformers": "5.2.3",
}

SOURCE_PATHS = (
    "task_demand/task_demand_model.py",
    "task_demand/run_task_demand.sh",
    "dataset/augmented_dataset.json",
    "dataset/pandapower_docs.json",
    "benchmark/benchmark6.json",
)

METHODS = {
    "Zero-shot Cross-Encoder": {
        "directory": "zero_shot_cross_encoder",
        "model": "zero_shot_tfidf",
        "role_reweight": False,
        "auto_hybrid_alpha": False,
        "max_positive_per_function": None,
    },
    "Hybrid Cross-Encoder": {
        "directory": "hybrid_cross_encoder",
        "model": "hybrid_tfidf",
        "role_reweight": True,
        "auto_hybrid_alpha": True,
        "max_positive_per_function": 500,
    },
}

ARCHIVED_SPLITS = {
    "augmented_dev": "Aug-dev",
    "augmented_dev_original_only": "Aug-dev-orig",
    "augmented_dev_variant_only": "Aug-dev-variant",
    "augmented_test": "Aug-test",
    "augmented_test_original_only": "Aug-orig",
    "augmented_test_variant_only": "Aug-test-variant",
    "benchmark_reference_eval_only": "Bench-ref",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def git_bytes(repository: Path, revision: str, relative_path: str) -> tuple[bytes, dict]:
    spec = f"{revision}:{relative_path}"
    blob = subprocess.run(
        ["git", "-C", str(repository), "show", spec],
        check=True,
        capture_output=True,
    ).stdout
    blob_id = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", spec],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return blob, {
        "path": relative_path,
        "revision": revision,
        "git_blob": blob_id,
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def local_descriptor(root: Path, path: Path) -> dict:
    result = {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    try:
        result["git_blob"] = subprocess.run(
            ["git", "-C", str(root), "rev-parse", f"HEAD:{result['path']}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        result["git_blob"] = None
    return result


def model_snapshot_descriptor(snapshot: Path) -> dict:
    require(snapshot.is_dir(), f"model snapshot is not a directory: {snapshot}")
    files = []
    for relative, expected_sha256 in sorted(EXPECTED_MODEL_FILES.items()):
        path = snapshot / relative
        require(path.is_file(), f"pinned model file is missing: {relative}")
        actual_sha256 = sha256_file(path)
        require(actual_sha256 == expected_sha256, (
            f"{relative} differs from pinned {MODEL_ID}@{MODEL_REVISION} snapshot"
        ))
        files.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": actual_sha256,
        })
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": files,
    }


def import_historical_module(code_path: Path):
    name = "s4_historical_task_demand_7916953"
    spec = importlib.util.spec_from_file_location(name, code_path)
    require(spec is not None and spec.loader is not None,
            "could not create historical module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metric_check(recomputed: dict, archived: dict) -> bool:
    return (
        recomputed["n_eval"] == archived["n_examples"]
        and all(
            recomputed["top_k"][str(k)] == archived["top_k"][str(k)]
            for k in KS
        )
    )


def summary_check(recomputed: dict, archived: dict) -> bool:
    """Compare summaries while tolerating only a tied top-k boundary.

    The historical summarizer stores ``Counter.most_common(30)`` over labels
    held in Python sets.  The underlying split is stable, but hash-randomized
    set iteration can choose a different label when several labels tie at the
    30th position.  All non-ranking fields and every common frequency remain
    strict; omitted/added labels must be a one-for-one tie at the boundary.
    """
    stable_keys = (
        "n_examples",
        "n_groups",
        "n_variants",
        "n_non_variants",
        "n_unique_functions",
        "avg_labels_per_example",
    )
    if any(recomputed.get(key) != archived.get(key) for key in stable_keys):
        return False

    current = recomputed.get("top_functions", {})
    frozen = archived.get("top_functions", {})
    common = current.keys() & frozen.keys()
    if any(current[key] != frozen[key] for key in common):
        return False
    current_only = current.keys() - frozen.keys()
    frozen_only = frozen.keys() - current.keys()
    if not current_only and not frozen_only:
        return True
    if len(current_only) != len(frozen_only):
        return False
    boundary = min(min(current.values()), min(frozen.values()))
    return all(current[key] == boundary for key in current_only) and all(
        frozen[key] == boundary for key in frozen_only
    )


def data_summary_check(recomputed: dict, archived: dict) -> bool:
    return recomputed.keys() == archived.keys() and all(
        summary_check(recomputed[key], archived[key]) for key in recomputed
    )


def score_split(ranker, examples, cards, reranker, tdm, *, top_n: int, batch_size: int):
    valid = []
    names = set(cards)
    for example in examples:
        truth = sorted(example.functions & names)
        if truth:
            valid.append((example, truth, ranker.score(example.question)))
    merged = tdm.batch_rerank_with_cross_encoder(
        [(example.question, scores) for example, _, scores in valid],
        cards,
        reranker=reranker,
        top_n=top_n,
        batch_size=batch_size,
    )
    return [
        {
            "sample_id": example.sample_id,
            "gold_functions": truth,
            "ranking": tdm.rank_names(scores, 20),
        }
        for (example, truth, _), scores in zip(valid, merged)
    ]


def validate_frozen_config(label: str, config: dict, expected: dict) -> None:
    checks = {
        "model": config.get("model") == expected["model"],
        "include_variants": config.get("include_variants") is True,
        "seed": config.get("seed") == 22,
        "negative_per_positive": config.get("negative_per_positive") == 8,
        "max_train_pairs": config.get("max_train_pairs") is None,
        "max_positive_per_function": (
            config.get("max_positive_per_function")
            == expected["max_positive_per_function"]
        ),
        "hybrid_alpha": config.get("hybrid_alpha") == 0.5,
        "auto_hybrid_alpha": (
            config.get("auto_hybrid_alpha") is expected["auto_hybrid_alpha"]
        ),
        "hybrid_alpha_target_k": config.get("hybrid_alpha_target_k") == 10,
        "top_k": config.get("top_k") == list(KS),
        "cross_encoder_model": config.get("cross_encoder_model") == MODEL_ID,
        "cross_encoder_top_n": config.get("cross_encoder_top_n") == 30,
        "ce_batch_size": config.get("ce_batch_size") == 512,
        "ce_fp16": config.get("ce_fp16") is False,
        "role_reweight": config.get("role_reweight") is expected["role_reweight"],
        "max_role_weight": config.get("max_role_weight") == 3.0,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    require(not failed, f"{label}: frozen configuration differs for {failed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE,
                        help="Historical igpt repository containing the frozen suite.")
    parser.add_argument("--source-revision", default=SOURCE_REVISION,
                        help="Must equal the fixed historical code/data revision.")
    parser.add_argument("--model-snapshot", type=Path, required=True,
                        help="Pinned local CrossEncoder snapshot directory.")
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON path; existing files and historical suite paths are refused.")
    parser.add_argument("--device", default="cuda",
                        help="SentenceTransformers device (historical run used cuda).")
    parser.add_argument("--ce-batch-size", type=int, default=512,
                        help="CrossEncoder inference batch size; historical value is 512.")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHODS),
        default=list(METHODS),
        help=(
            "Methods to reconstruct. Selecting only Zero-shot Cross-Encoder "
            "avoids retraining the historically unsaved Hybrid first stage."
        ),
    )
    parser.add_argument(
        "--revision-rerun",
        action="store_true",
        help=(
            "Retain a deterministic revision rerun when historical metrics do "
            "not match. Source/data/model gates remain strict and every "
            "metric difference is recorded."
        ),
    )
    args = parser.parse_args()

    source = args.source_root.resolve()
    snapshot = args.model_snapshot.resolve()
    output = args.output.resolve()
    suite = (source / SUITE_RELATIVE).resolve()
    require(args.source_revision == SOURCE_REVISION,
            f"source revision must be {SOURCE_REVISION}")
    require(source.is_dir(), f"source repository does not exist: {source}")
    require(not output.exists(), f"refusing to overwrite existing output: {output}")
    require(suite != output and suite not in output.parents,
            "output must not be inside the historical suite")
    require(args.ce_batch_size == 512,
            "strict reconstruction requires historical ce_batch_size=512")
    require(args.device in {"cuda", "cpu"}, "device must be cuda or cpu")
    require(args.device == "cuda" or args.revision_rerun,
            "the non-historical CPU path must be labelled --revision-rerun")
    require(os.environ.get("PYTHONHASHSEED") is not None,
            "set PYTHONHASHSEED explicitly before launch; the historical code iterates sets")

    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    environment_versions = {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "sentence_transformers": importlib.metadata.version(
            "sentence-transformers"
        ),
    }
    require(environment_versions == EXPECTED_ENVIRONMENT, (
        "recovery environment differs from the historical pinned environment: "
        f"expected {EXPECTED_ENVIRONMENT}, found {environment_versions}"
    ))

    snapshot_provenance = model_snapshot_descriptor(snapshot)
    source_blobs = {}
    source_descriptors = []
    for relative in SOURCE_PATHS:
        blob, descriptor = git_bytes(source, SOURCE_REVISION, relative)
        source_blobs[relative] = blob
        source_descriptors.append(descriptor)

    selected_methods = {
        label: METHODS[label] for label in args.methods
    }
    frozen = {}
    frozen_descriptors = []
    for label, method in selected_methods.items():
        path = suite / method["directory"] / "metrics.json"
        require(path.is_file(), f"missing frozen metrics: {path}")
        frozen[label] = json.loads(path.read_text(encoding="utf-8"))
        frozen_descriptors.append(local_descriptor(source, path))
        validate_frozen_config(label, frozen[label]["config"], method)
    log_path = source / LOG_RELATIVE
    require(log_path.is_file(), f"missing historical job log: {log_path}")
    log_descriptor = local_descriptor(source, log_path)

    with tempfile.TemporaryDirectory(prefix="s4_ce_reconstruction_") as temp_name:
        temp = Path(temp_name)
        for relative, blob in source_blobs.items():
            path = temp / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        tdm = import_historical_module(temp / "task_demand/task_demand_model.py")

        docs_path = temp / "dataset/pandapower_docs.json"
        augmented_path = temp / "dataset/augmented_dataset.json"
        benchmark_path = temp / "benchmark/benchmark6.json"
        cards = tdm.load_function_cards(docs_path)
        function_names = set(cards)
        augmented = tdm.load_augmented_examples(
            augmented_path, function_names, include_variants=True
        )
        train, dev, test = tdm.split_by_group(augmented, seed=22)
        bench = tdm.load_benchmark_reference_examples(
            benchmark_path, function_names, max_items=None
        )
        slices = {
            "augmented_dev": dev,
            "augmented_dev_original_only": [row for row in dev if not row.is_variant],
            "augmented_dev_variant_only": [row for row in dev if row.is_variant],
            "augmented_test": test,
            "augmented_test_original_only": [row for row in test if not row.is_variant],
            "augmented_test_variant_only": [row for row in test if row.is_variant],
            "benchmark_reference_eval_only": bench,
        }
        data_summary = {
            "all": tdm.summarize_examples(augmented),
            "train": tdm.summarize_examples(train),
            "dev": tdm.summarize_examples(dev),
            "test": tdm.summarize_examples(test),
            "benchmark_reference_eval_only": tdm.summarize_examples(bench),
        }
        for label, archived in frozen.items():
            require(data_summary_check(data_summary, archived["data_summary"]),
                    f"{label}: historical source blobs produce a different data summary")

        # Recreate each first-stage ranker exactly as its historical process did.
        random.seed(22)
        tdm.np.random.seed(22)
        zero_ranker = tdm.ZeroShotTfidfDemandRanker().fit(cards)

        role_weights = None
        alpha_tuning = None
        hybrid_ranker = None
        if "Hybrid Cross-Encoder" in selected_methods:
            random.seed(22)
            tdm.np.random.seed(22)
            role_weights = tdm.compute_role_weights(
                train, bench, cards, max_weight=3.0
            )
            hybrid_ranker = tdm.HybridTfidfDemandRanker(
                alpha=0.5,
                negative_per_positive=8,
                max_train_pairs=None,
                max_positive_per_function=500,
                seed=22,
            ).fit(train, cards, role_weights=role_weights)
            alpha_tuning = tdm.tune_hybrid_alpha(
                hybrid_ranker,
                dev,
                cards,
                candidate_alphas=[round(value * 0.1, 1) for value in range(11)],
                target_k=10,
            )
            hybrid_frozen = frozen["Hybrid Cross-Encoder"]
            role_weights_match = role_weights == hybrid_frozen["role_weights"]
            alpha_tuning_match = alpha_tuning == hybrid_frozen["alpha_tuning"]
            if not args.revision_rerun:
                require(role_weights_match,
                        "Hybrid Cross-Encoder role weights differ from the frozen run")
                require(alpha_tuning_match,
                        "Hybrid Cross-Encoder alpha curve or selected alpha differs")
        else:
            role_weights_match = None
            alpha_tuning_match = None

        # The only neural model load in this program.  Reuse it for both methods
        # and every split; do not instantiate per evaluate_ranker call.
        reranker = tdm._load_cross_encoder(str(snapshot), args.device, fp16=False)

        rankers = {}
        if "Zero-shot Cross-Encoder" in selected_methods:
            rankers["Zero-shot Cross-Encoder"] = zero_ranker
        if "Hybrid Cross-Encoder" in selected_methods:
            rankers["Hybrid Cross-Encoder"] = hybrid_ranker
        reconstructions = {}
        for label, ranker in rankers.items():
            archived = frozen[label]["metrics"]
            rows_by_slice = {}
            metric_checks = {}
            recomputed_metrics = {}
            for archived_name, examples in slices.items():
                rows = score_split(
                    ranker,
                    examples,
                    cards,
                    reranker,
                    tdm,
                    top_n=30,
                    batch_size=args.ce_batch_size,
                )
                result = metrics(rows, max_k=20)
                rows_by_slice[archived_name] = rows
                recomputed_metrics[archived_name] = result
                metric_checks[archived_name] = metric_check(
                    result, archived[archived_name]
                )
            failed = sorted(key for key, passed in metric_checks.items() if not passed)
            if not args.revision_rerun:
                require(not failed, f"{label}: frozen top-k metrics differ for {failed}")

            report_splits = {
                "Aug-test": "augmented_test",
                "Aug-orig": "augmented_test_original_only",
                "Bench-ref": "benchmark_reference_eval_only",
            }
            reconstructions[label] = {
                "origin": (
                    "pinned-snapshot deterministic reconstruction from historical "
                    f"source revision {SOURCE_REVISION}; not a historical raw item log"
                ),
                "coverage": "Aug-test, Aug-orig, and Bench-ref unfiltered top-20",
                "splits": {
                    display: {
                        "predictions": [
                            {"sample_id": row["sample_id"], "top20": row["ranking"]}
                            for row in rows_by_slice[archived_name]
                        ],
                        "metrics": recomputed_metrics[archived_name],
                    }
                    for display, archived_name in report_splits.items()
                },
                "all_seven_frozen_slice_metrics": recomputed_metrics,
                "frozen_metrics": archived,
                "all_seven_frozen_slice_checks": metric_checks,
            }

        evaluation_sets = {
            "Aug-test": {
                "items": [
                    {
                        "sample_id": row.sample_id,
                        "query": row.question,
                        "gold_functions": sorted(row.functions & function_names),
                        "is_variant": row.is_variant,
                    }
                    for row in test if row.functions & function_names
                ]
            },
            "Bench-ref": {
                "items": [
                    {
                        "sample_id": row.sample_id,
                        "query": row.question,
                        "gold_functions": sorted(row.functions & function_names),
                    }
                    for row in bench if row.functions & function_names
                ]
            },
        }

    import torch
    if args.device == "cuda":
        require(torch.cuda.is_available(),
                "historical cuda path requested but CUDA is unavailable")
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = platform.processor() or platform.machine() or "CPU"
    historical_execution_path = args.device == "cuda"
    environment = environment_versions | {
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "device_name": device_name,
        "device": args.device,
        "ce_batch_size": args.ce_batch_size,
        "ce_fp16": False,
        "historical_execution_path": historical_execution_path,
    }
    all_frozen_metric_checks_pass = all(
        all(method["all_seven_frozen_slice_checks"].values())
        for method in reconstructions.values()
    )
    exact_reconstruction = (
        all_frozen_metric_checks_pass
        and (alpha_tuning_match in (None, True))
        and (role_weights_match in (None, True))
        and historical_execution_path
    )
    artifact = {
        "schema_version": 1,
        "status": (
            "all strict historical-metric gates passed"
            if exact_reconstruction
            else "deterministic revision rerun; historical metric differences retained"
        ),
        "evidence_type": (
            "pinned deterministic reconstruction; not historical raw item predictions"
            if exact_reconstruction else
            "separately labelled deterministic revision rerun; not historical raw item predictions"
        ),
        "source_revision": SOURCE_REVISION,
        "model_snapshot": snapshot_provenance,
        "environment": environment,
        "provenance": {
            "source_repository": "frozen experimental pipeline",
            "historical_source_blobs": source_descriptors,
            "frozen_metrics": frozen_descriptors,
            "historical_job_log": log_descriptor,
            "recovery_code": {
                "path": HERE.relative_to(ROOT).as_posix(),
                "bytes": HERE.stat().st_size,
                "sha256": sha256_file(HERE),
            },
        },
        "historical_execution": {
            "job_id": 4428033,
            "gpu": "NVIDIA GH200 120GB",
            "cross_encoder_top_n": 30,
            "ce_batch_size": 512,
            "ce_fp16": False,
        },
        "hybrid_first_stage_checks": (
            {
                "role_weights": role_weights,
                "role_weights_match": role_weights_match,
                "alpha_tuning": alpha_tuning,
                "alpha_curve_and_selected_alpha_match": alpha_tuning_match,
            }
            if "Hybrid Cross-Encoder" in selected_methods else None
        ),
        "evaluation_sets": evaluation_sets,
        "reconstructions": reconstructions,
        "exact_historical_metric_reconstruction": exact_reconstruction,
        "all_frozen_metric_checks_pass": all_frozen_metric_checks_pass,
        "all_integrity_checks_pass": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "methods": sorted(reconstructions),
        "all_integrity_checks_pass": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
