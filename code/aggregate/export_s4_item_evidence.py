#!/usr/bin/env python3
"""Recover the strongest honest item-level evidence available for S4.

The frozen suite retained benchmark-reference top-10 exports for all six
estimators, but not augmented-split exports and not ranks 11--20.  Historical
pickle models were retained for Pairwise TF-IDF, adapted Hybrid TF-IDF, and
unadapted Hybrid TF-IDF.  Together with the deterministic Zero-shot TF-IDF
algorithm, those files permit exact top-20 reconstructions which are accepted
only if every required frozen metric matches.  Zero-shot SBERT is likewise
recoverable from the immutable all-MiniLM-L6-v2 snapshot recorded below; its
CPU reconstruction is accepted only when all seven frozen evaluation slices
match at every reported k.  The unadapted and adapted Hybrid models also
recover S4(b)'s before/after tradeoff on all three reported splits.  No exact
item-level claim is made for the two cross-encoder estimators whose weight
revisions were not frozen.

The output is public compact evidence.  The maintainer-side recovery step needs
the adjacent historical ``igpt`` repository and scikit-learn; the public
``aggregate_s4_item_evidence.py`` verifier is standard-library only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import subprocess
import sys
import warnings
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
DEFAULT_SOURCE = ROOT.parent / "igpt"
DEFAULT_OUTPUT = ROOT / "results/supplementary_evidence/s4_item_evidence.json"
KS = (1, 3, 5, 10, 20)
SBERT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
SBERT_WEIGHT_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"

ESTIMATORS = {
    "Zero-shot TF-IDF": "zero_shot_tfidf",
    "Pairwise TF-IDF+LogReg": "pairwise_tfidf_logreg",
    "Hybrid TF-IDF": "hybrid_tfidf",
    "Zero-shot SBERT": "zero_shot_sbert",
    "Zero-shot Cross-Encoder": "zero_shot_cross_encoder",
    "Hybrid Cross-Encoder": "hybrid_cross_encoder",
}

RECONSTRUCTABLE = {
    "Zero-shot TF-IDF": "algorithm",
    "Pairwise TF-IDF+LogReg": "saved_model",
    "Hybrid TF-IDF": "saved_model",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(source: Path, path: Path) -> dict:
    return {
        "path": path.resolve().relative_to(source.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def snapshot_descriptor(snapshot: Path) -> dict:
    files = []
    for path in sorted(candidate for candidate in snapshot.rglob("*") if candidate.is_file()):
        files.append({
            "path": path.relative_to(snapshot).as_posix(),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256_file(path),
        })
    weights = next((item for item in files if item["path"] == "model.safetensors"), None)
    if weights is None or weights["sha256"] != SBERT_WEIGHT_SHA256:
        raise AssertionError("SBERT snapshot weight hash differs from the pinned revision")
    if snapshot.name != SBERT_REVISION:
        raise AssertionError(
            f"SBERT snapshot directory must be the pinned revision {SBERT_REVISION}"
        )
    return {
        "model_id": SBERT_MODEL_ID,
        "revision": SBERT_REVISION,
        "files": files,
    }


def metrics(rows: list[dict], max_k: int) -> dict:
    output = {"n_eval": len(rows), "top_k": {}}
    for k in (value for value in KS if value <= max_k):
        recall_sum = precision_sum = 0.0
        hit_count = 0
        for row in rows:
            truth = set(row["gold_functions"])
            predicted = set(row["ranking"][:k])
            overlap = truth & predicted
            recall_sum += len(overlap) / len(truth)
            precision_sum += len(overlap) / k
            hit_count += bool(overlap)
        output["top_k"][str(k)] = {
            "recall": round(recall_sum / len(rows), 4),
            "precision": round(precision_sum / len(rows), 4),
            "hit_rate": round(hit_count / len(rows), 4),
        }
    return output


class HistoricalModelUnpickler(pickle.Unpickler):
    """Map classes pickled from the script's ``__main__`` module."""

    def __init__(self, stream, task_demand_model):
        super().__init__(stream)
        self.task_demand_model = task_demand_model

    def find_class(self, module: str, name: str):
        if module == "__main__" and hasattr(self.task_demand_model, name):
            return getattr(self.task_demand_model, name)
        return super().find_class(module, name)


def load_historical_model(path: Path, task_demand_model):
    with path.open("rb") as stream:
        payload = HistoricalModelUnpickler(stream, task_demand_model).load()
    return payload["ranker"]


def compact_predictions(ranker, examples, cards, task_demand_model) -> list[dict]:
    names = set(cards)
    output = []
    for example in examples:
        truth = sorted(example.functions & names)
        if not truth:
            continue
        output.append({
            "sample_id": example.sample_id,
            "gold_functions": truth,
            "ranking": task_demand_model.rank_names(ranker.score(example.question), 20),
        })
    return output


def load_git_json(source: Path, revision: str, relative_path: str) -> tuple[object, dict]:
    spec = f"{revision}:{relative_path}"
    blob = subprocess.run(
        ["git", "-C", str(source), "show", spec],
        check=True,
        capture_output=True,
    ).stdout
    blob_id = subprocess.run(
        ["git", "-C", str(source), "rev-parse", spec],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return json.loads(blob), {
        "path": relative_path,
        "revision": revision,
        "git_blob": blob_id,
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def benchmark_examples_from_items(items, function_names, task_demand_model):
    examples = []
    for index, item in enumerate(items):
        funcs = task_demand_model.extract_reference_functions(
            item.get("reference_code", ""), function_names
        )
        if not funcs:
            continue
        sample_id = str(item.get("id", index))
        examples.append(task_demand_model.DemandExample(
            sample_id=sample_id,
            base_id=sample_id,
            question=str(item.get("natural_language_query", "")).strip(),
            functions=funcs,
            attributes=set(),
            is_variant=False,
            source="benchmark_reference_eval_only",
        ))
    return examples


def exact_metric_match(recomputed: dict, archived: dict) -> bool:
    return all(
        recomputed["top_k"][str(k)] == archived["top_k"][str(k)]
        for k in KS
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sbert-snapshot",
        type=Path,
        required=True,
        help=(
            "Local Hugging Face snapshot directory for the pinned "
            f"{SBERT_MODEL_ID}@{SBERT_REVISION} reconstruction."
        ),
    )
    args = parser.parse_args()
    source = args.source_root.resolve()
    sbert_snapshot = args.sbert_snapshot.resolve()
    if not sbert_snapshot.is_dir():
        raise FileNotFoundError(sbert_snapshot)

    task_dir = source / "task_demand"
    sys.path.insert(0, str(task_dir))
    import task_demand_model as tdm
    import numpy
    import sklearn

    docs_path = source / "dataset/pandapower_docs.json"
    augmented_path = source / "dataset/augmented_dataset.json"
    # The frozen S4 suite predates the release rename and records benchmark6.json
    # in every run configuration.  Using benchmark.json here changes labels for
    # some IDs and therefore is not an exact reconstruction.
    cards = tdm.load_function_cards(docs_path)
    function_names = set(cards)
    augmented = tdm.load_augmented_examples(
        augmented_path, function_names, include_variants=True
    )
    _, augmented_dev, augmented_test = tdm.split_by_group(augmented, seed=22)
    augmented_original = [item for item in augmented_test if not item.is_variant]
    benchmark_items, benchmark_descriptor = load_git_json(
        source, "0ca5647", "benchmark/benchmark6.json"
    )
    benchmark = benchmark_examples_from_items(benchmark_items, function_names, tdm)

    evaluation_sets = {
        "Aug-test": {
            "items": [
                {
                    "sample_id": item.sample_id,
                    "query": item.question,
                    "gold_functions": sorted(item.functions & function_names),
                    "is_variant": item.is_variant,
                }
                for item in augmented_test
            ]
        },
        "Bench-ref": {
            "items": [
                {
                    "sample_id": item.sample_id,
                    "query": item.question,
                    "gold_functions": sorted(item.functions & function_names),
                }
                for item in benchmark
            ]
        },
    }

    suite = task_dir / "results/suites/suite_4428033"
    provenance_files = [
        descriptor(source, task_dir / "task_demand_model.py"),
        descriptor(source, source / "requirements.lock.txt"),
        descriptor(source, docs_path),
        descriptor(source, augmented_path),
        benchmark_descriptor,
    ]

    # Historical raw benchmark top-10 exports exist for all six estimators.
    historical_raw = {}
    bench_ids = [item.sample_id for item in benchmark]
    for label, directory in ESTIMATORS.items():
        run_dir = suite / directory
        export_path = run_dir / "exports/qwen3_480b_candidates.json"
        metrics_path = run_dir / "metrics.json"
        export = json.loads(export_path.read_text(encoding="utf-8"))
        archived = json.loads(metrics_path.read_text(encoding="utf-8"))
        if [row["id"] for row in export] != bench_ids:
            raise AssertionError(f"{label}: raw export ids differ from frozen benchmark")
        rows = [
            {
                "sample_id": row["id"],
                "gold_functions": row["reference_functions"],
                "ranking": [candidate["function"] for candidate in row["top_demand"]],
            }
            for row in export
        ]
        raw_metrics = metrics(rows, max_k=10)
        raw_check = all(len(row["ranking"]) == 10 for row in rows)
        if not raw_check:
            raise AssertionError(f"{label}: malformed retained deployment top-10")
        historical_raw[label] = {
            "coverage": "role-filtered Bench-ref deployment top-10 only",
            "origin": "historical raw prediction export with prediction_roles whitelist",
            "prediction_roles": archived["config"]["prediction_roles"],
            "metric_scope_note": (
                "These role-filtered deployment candidates are not the unfiltered "
                "rankings used for the S4 table and recall@k figure."
            ),
            "predictions": [
                {"sample_id": row["sample_id"], "top10": row["ranking"]}
                for row in rows
            ],
            "filtered_export_metrics": raw_metrics,
            "raw_structure_check": raw_check,
        }
        provenance_files.extend([
            descriptor(source, export_path),
            descriptor(source, metrics_path),
        ])

    # Exact reconstructions are limited to methods with a frozen model or a
    # fully deterministic local algorithm.  Every reconstructed curve must
    # match all archived k values and the retained benchmark top-10 prefix.
    reconstructions = {}
    for label, origin in RECONSTRUCTABLE.items():
        directory = ESTIMATORS[label]
        run_dir = suite / directory
        archived = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if origin == "algorithm":
            ranker = tdm.ZeroShotTfidfDemandRanker().fit(cards)
        else:
            model_path = run_dir / "artifacts/model.pkl"
            provenance_files.append(descriptor(source, model_path))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ranker = load_historical_model(model_path, tdm)

        split_predictions = {}
        split_checks = {}
        for display, examples, archived_key in (
            ("Aug-test", augmented_test, "augmented_test"),
            ("Aug-orig", augmented_original, "augmented_test_original_only"),
            ("Bench-ref", benchmark, "benchmark_reference_eval_only"),
        ):
            rows = compact_predictions(ranker, examples, cards, tdm)
            recomputed = metrics(rows, max_k=20)
            check = exact_metric_match(recomputed, archived["metrics"][archived_key])
            if not check:
                raise AssertionError(f"{label}/{display}: top-20 metrics differ from frozen run")
            if display == "Bench-ref":
                # The retained export applies a role whitelist before taking
                # top-10, whereas the S4 curves use the unfiltered universe.
                # It is preserved as separate raw deployment evidence, not as
                # a prefix cross-check on this unfiltered reconstruction.
                prefix_match = "not_applicable_role_filtered_export"
            else:
                prefix_match = None
            split_predictions[display] = {
                "predictions": [
                    {"sample_id": row["sample_id"], "top20": row["ranking"]}
                    for row in rows
                ],
                "metrics": recomputed,
            }
            split_checks[display] = {
                "all_frozen_k_metrics_match": check,
                "historical_raw_top10_prefix_matches": prefix_match,
            }
        reconstructions[label] = {
            "origin": (
                "exact reconstruction from frozen historical model.pkl"
                if origin == "saved_model"
                else "exact deterministic reconstruction from frozen local TF-IDF algorithm"
            ),
            "coverage": "Aug-test, Aug-orig, and Bench-ref top-20",
            "splits": split_predictions,
            "integrity_checks": split_checks,
        }

    # The named all-MiniLM snapshot was still present in the experiment cache.
    # Reconstruct its item rankings on CPU, one query at a time, mirroring the
    # historical evaluator.  The reconstruction is retained only when all
    # seven archived slices and all fifteen top-k cells per slice match.
    sbert_provenance = snapshot_descriptor(sbert_snapshot)
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    sbert_ranker = tdm.SbertDemandRanker(
        str(sbert_snapshot), batch_size=16, device="cpu"
    ).fit(cards)
    sbert_dev = compact_predictions(sbert_ranker, augmented_dev, cards, tdm)
    sbert_test = compact_predictions(sbert_ranker, augmented_test, cards, tdm)
    sbert_bench = compact_predictions(sbert_ranker, benchmark, cards, tdm)
    dev_original_ids = {item.sample_id for item in augmented_dev if not item.is_variant}
    dev_variant_ids = {item.sample_id for item in augmented_dev if item.is_variant}
    test_original_ids = {item.sample_id for item in augmented_original}
    test_variant_ids = {item.sample_id for item in augmented_test if item.is_variant}
    sbert_frozen_rows = {
        "augmented_dev": sbert_dev,
        "augmented_dev_original_only": [
            row for row in sbert_dev if row["sample_id"] in dev_original_ids
        ],
        "augmented_dev_variant_only": [
            row for row in sbert_dev if row["sample_id"] in dev_variant_ids
        ],
        "augmented_test": sbert_test,
        "augmented_test_original_only": [
            row for row in sbert_test if row["sample_id"] in test_original_ids
        ],
        "augmented_test_variant_only": [
            row for row in sbert_test if row["sample_id"] in test_variant_ids
        ],
        "benchmark_reference_eval_only": sbert_bench,
    }
    sbert_archived = json.loads(
        (suite / "zero_shot_sbert/metrics.json").read_text(encoding="utf-8")
    )
    sbert_frozen_checks = {
        split: exact_metric_match(metrics(rows, max_k=20), sbert_archived["metrics"][split])
        for split, rows in sbert_frozen_rows.items()
    }
    if not all(sbert_frozen_checks.values()):
        failed = sorted(key for key, value in sbert_frozen_checks.items() if not value)
        raise AssertionError(f"Zero-shot SBERT frozen metrics differ for: {failed}")
    sbert_reported_rows = {
        "Aug-test": sbert_test,
        "Aug-orig": sbert_frozen_rows["augmented_test_original_only"],
        "Bench-ref": sbert_bench,
    }
    sbert_archived_keys = {
        "Aug-test": "augmented_test",
        "Aug-orig": "augmented_test_original_only",
        "Bench-ref": "benchmark_reference_eval_only",
    }
    reconstructions["Zero-shot SBERT"] = {
        "origin": (
            "pinned-snapshot deterministic CPU reconstruction from "
            f"{SBERT_MODEL_ID}@{SBERT_REVISION}"
        ),
        "coverage": "Aug-test, Aug-orig, and Bench-ref top-20",
        "splits": {
            display: {
                "predictions": [
                    {"sample_id": row["sample_id"], "top20": row["ranking"]}
                    for row in rows
                ],
                "metrics": metrics(rows, max_k=20),
            }
            for display, rows in sbert_reported_rows.items()
        },
        "integrity_checks": {
            display: {
                "all_frozen_k_metrics_match": exact_metric_match(
                    metrics(rows, max_k=20),
                    sbert_archived["metrics"][sbert_archived_keys[display]],
                ),
                "historical_raw_top10_prefix_matches": (
                    "not_applicable_role_filtered_export" if display == "Bench-ref" else None
                ),
            }
            for display, rows in sbert_reported_rows.items()
        },
        "all_seven_frozen_slice_checks": sbert_frozen_checks,
    }

    # The before arm in S4(b) is a separate frozen unadapted Hybrid model.
    before_dir = task_dir / "results/suites/suite_unadapted/hybrid_tfidf"
    before_model = before_dir / "artifacts/model.pkl"
    before_metrics_path = before_dir / "metrics.json"
    before_export_path = before_dir / "exports/qwen3_480b_candidates.json"
    provenance_files.extend([
        descriptor(source, before_model),
        descriptor(source, before_metrics_path),
        descriptor(source, before_export_path),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before_ranker = load_historical_model(before_model, tdm)
    before_archived = json.loads(before_metrics_path.read_text(encoding="utf-8"))
    before_predictions = {}
    before_metrics = {}
    before_checks = {}
    archived_split_keys = {
        "Aug-test": "augmented_test",
        "Aug-orig": "augmented_test_original_only",
        "Bench-ref": "benchmark_reference_eval_only",
    }
    for display, examples in (
        ("Aug-test", augmented_test),
        ("Aug-orig", augmented_original),
        ("Bench-ref", benchmark),
    ):
        rows = compact_predictions(before_ranker, examples, cards, tdm)
        recomputed = metrics(rows, max_k=20)
        check = exact_metric_match(
            recomputed, before_archived["metrics"][archived_split_keys[display]]
        )
        if not check:
            raise AssertionError(
                f"S4(b) unadapted {display} reconstruction differs from frozen metrics"
            )
        before_predictions[display] = [
            {"sample_id": row["sample_id"], "top20": row["ranking"]}
            for row in rows
        ]
        before_metrics[display] = recomputed
        before_checks[display] = check

    after_splits = reconstructions["Hybrid TF-IDF"]["splits"]
    split_results = {}
    for split in ("Aug-test", "Aug-orig", "Bench-ref"):
        after = after_splits[split]
        delta = round(
            (
                after["metrics"]["top_k"]["10"]["recall"]
                - before_metrics[split]["top_k"]["10"]["recall"]
            )
            * 100,
            2,
        )
        split_results[split] = {
            "before": {
                "predictions": before_predictions[split],
                "metrics": before_metrics[split],
            },
            "after": {
                "prediction_reference": f"reconstructions/Hybrid TF-IDF/splits/{split}",
                "metrics": after["metrics"],
            },
            "recall_at_10_delta_pp": delta,
            "before_all_frozen_k_metrics_match": before_checks[split],
            "after_all_frozen_k_metrics_match": reconstructions["Hybrid TF-IDF"][
                "integrity_checks"
            ][split]["all_frozen_k_metrics_match"],
        }

    adapted_metrics = json.loads(
        (suite / "hybrid_tfidf/metrics.json").read_text(encoding="utf-8")
    )
    s4b = {
        "scope": "frozen Hybrid role-reweighting tradeoff on all reported evaluation splits",
        "before_origin": "exact reconstruction from frozen suite_unadapted model.pkl",
        "after_origin": "exact adapted Hybrid reconstruction shared with S4(a)",
        "selected_alpha": {
            "before": before_archived["alpha_tuning"]["selected"]["alpha"],
            "after": adapted_metrics["alpha_tuning"]["selected"]["alpha"],
        },
        "splits": split_results,
        "tradeoff_summary": {
            "Aug-test_delta_pp": split_results["Aug-test"]["recall_at_10_delta_pp"],
            "Bench-ref_delta_pp": split_results["Bench-ref"]["recall_at_10_delta_pp"],
            "interpretation": (
                "Role reweighting trades a small Aug-test decrease for a large "
                "deployment-aligned Bench-ref increase."
            ),
        },
    }

    unavailable = {
        label: {
            "available": "historical role-filtered Bench-ref deployment top-10 export",
            "missing": (
                "unfiltered Aug-test/Aug-orig item predictions and unfiltered "
                "Bench-ref item rankings used by the S4 curves"
            ),
            "reason": (
                "No frozen item export or model weights/revision sufficient for an exact "
                "reconstruction were retained; aggregate metrics remain available."
            ),
        }
        for label in ("Zero-shot Cross-Encoder", "Hybrid Cross-Encoder")
    }

    output = {
        "schema_version": 3,
        "evidence_unit": "query x gold functions x ranked API functions",
        "status": "mixed historical raw evidence and strictly validated exact reconstructions",
        "coverage_note": (
            "Full top-20 curves are recovered for the three TF-IDF estimators, "
            "Zero-shot SBERT from its immutable model snapshot, and "
            "the S4(b) unadapted Hybrid arm on all three evaluation splits. "
            "The two cross-encoder estimators "
            "retain only historical role-filtered Bench-ref deployment top-10 "
            "evidence, which is not the unfiltered S4 curve ranking."
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scikit_learn_used_for_recovery": sklearn.__version__,
            "historical_scikit_learn_pin": "1.6.0",
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "sbert_recovery": {
                "torch": importlib.metadata.version("torch"),
                "transformers": importlib.metadata.version("transformers"),
                "sentence_transformers": importlib.metadata.version(
                    "sentence-transformers"
                ),
                "device": "cpu",
                "batch_size_for_card_embeddings": 16,
                "query_scoring": "one query at a time",
            },
        },
        "provenance": {
            "historical_repository": "frozen experimental pipeline",
            "recovery_code": {
                "path": HERE.relative_to(ROOT).as_posix(),
                "bytes": HERE.stat().st_size,
                "sha256": sha256_file(HERE),
            },
            "files": provenance_files,
            "sbert_model_snapshot": sbert_provenance,
        },
        "evaluation_sets": evaluation_sets,
        "historical_benchmark_top10": historical_raw,
        "reconstructions": reconstructions,
        "s4b_before_after_by_split": s4b,
        "unavailable_exact_item_coverage": unavailable,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)")
    print(json.dumps({
        "full_top20_reconstructions": sorted(reconstructions),
        "raw_benchmark_top10": sorted(historical_raw),
        "s4b_recall_at_10_delta_pp": {
            split: data["recall_at_10_delta_pp"]
            for split, data in s4b["splits"].items()
        },
        "unavailable_full_top20": sorted(unavailable),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
