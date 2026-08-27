#!/usr/bin/env python3
"""Rebuild S4 and E4 demand-model results from compact sufficient statistics.

Release-mode (the default) uses only PowerCodeBench files.  Maintainers can
use ``--export-estimators`` to freeze the six S4 metrics files and
``--export-e4`` to train the frozen deterministic E4 arms once and retain, per
query, the gold functions/roles, selector decision and each arm's top-20.
Those item records are sufficient to recompute recall@k, hit@k, per-role
recall, and the query-conditional selector without the original repository.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections import defaultdict
from pathlib import Path

from evidence_common import read_json, repo_descriptor, source_descriptor, write_json

HERE = Path(__file__).resolve()
PCB_ROOT = HERE.parents[2]
DEFAULT_SOURCE = PCB_ROOT.parent / "igpt"
EVIDENCE_DIR = PCB_ROOT / "results/supplementary_evidence"
DEFAULT_ESTIMATORS = EVIDENCE_DIR / "demand_estimators.json"
DEFAULT_E4 = EVIDENCE_DIR / "e4_item_predictions.json"
DEFAULT_UNADAPTED_HYBRID = (
    PCB_ROOT / "results/demand_suites/suite_unadapted/hybrid_tfidf/metrics.json"
)
DEFAULT_ADAPTED_HYBRID = (
    PCB_ROOT / "results/demand_suites/suite_4428033/hybrid_tfidf/metrics.json"
)
DEFAULT_OUTPUT = PCB_ROOT / "results/aggregates/supplementary_demand_evidence.json"
KS = (1, 3, 5, 10, 20)

ESTIMATORS = {
    "Zero-shot TF-IDF": "zero_shot_tfidf",
    "Pairwise TF-IDF+LogReg": "pairwise_tfidf_logreg",
    "Hybrid TF-IDF": "hybrid_tfidf",
    "Zero-shot SBERT": "zero_shot_sbert",
    "Zero-shot Cross-Encoder": "zero_shot_cross_encoder",
    "Hybrid Cross-Encoder": "hybrid_cross_encoder",
}


def export_estimators(source_root: Path, artifact: Path) -> None:
    files = []
    runs = {}
    for label, directory in ESTIMATORS.items():
        path = source_root / "task_demand/results/suites/suite_4428033" / directory / "metrics.json"
        data = read_json(path)
        files.append(source_descriptor(source_root, path))
        runs[label] = {
            "source_directory": directory,
            "metrics": data["metrics"],
            # Historical metrics store this under ``alpha_tuning``.  Preserve
            # the release key for schema compatibility, but export the actual
            # curve rather than a silent null.
            "hybrid_alpha_tuning": data.get("alpha_tuning"),
            "role_weights": data.get("role_weights"),
        }
    out = {
        "schema_version": 1,
        "evidence_unit": "estimator x evaluation split x k",
        "suite": "suite_4428033",
        "provenance": {**repo_descriptor(source_root), "files": files},
        "runs": runs,
    }
    write_json(artifact, out)
    print(f"exported six-estimator evidence ({artifact.stat().st_size} bytes) -> {artifact}")


def export_e4(source_root: Path, artifact: Path) -> None:
    demand_dir = source_root / "task_demand"
    sys.path.insert(0, str(demand_dir))
    ce = importlib.import_module("e4_conditional_eval")
    tdm = importlib.import_module("task_demand_model")

    cards = tdm.load_function_cards(tdm.DEFAULT_DOCS_PATH)
    function_names = set(cards)
    set_examples = {}
    all_questions = []
    for name, relative in ce.EVAL_SETS.items():
        examples = tdm.load_benchmark_reference_examples(source_root / relative, function_names)
        set_examples[name] = examples
        all_questions.extend(example.question for example in examples)

    ua_alpha, _, ua_scores = ce.train_arm_and_score("unadapted", all_questions, seed=22)
    ad_alpha, role_weights, ad_scores = ce.train_arm_and_score("adapted", all_questions, seed=22)
    sets = {}
    for name, examples in set_examples.items():
        items = []
        for example in examples:
            truth = sorted(example.functions & function_names)
            tops = {
                "unadapted": tdm.rank_names(ua_scores[example.question], 20),
                "adapted": tdm.rank_names(ad_scores[example.question], 20),
            }
            decision = ce.selector_intent(example.question)
            tops["conditional"] = list(tops["adapted" if decision == "analysis" else "unadapted"])
            items.append({
                "sample_id": example.sample_id,
                "query": example.question,
                "gold_functions": truth,
                "gold_function_roles": {fn: tdm.infer_function_role(fn, cards.get(fn)) for fn in truth},
                "selector_intent": decision,
                "top20": tops,
            })
        sets[name] = {"source_path": ce.EVAL_SETS[name], "items": items}

    paths = [
        demand_dir / "e4_conditional_eval.py",
        demand_dir / "task_demand_model.py",
        Path(tdm.DEFAULT_AUGMENTED_PATH),
        Path(tdm.DEFAULT_DOCS_PATH),
        demand_dir / "results/e4_conditional/metrics_by_set_v2.json",
        demand_dir / "results/e4_conditional/per_role_v2.json",
    ] + [source_root / relative for relative in ce.EVAL_SETS.values()]
    out = {
        "schema_version": 1,
        "evidence_unit": "query x gold function labels/roles x arm top-20",
        "pipeline": {
            "seed": 22,
            "unadapted_alpha": ua_alpha,
            "adapted_alpha": ad_alpha,
            "adapted_role_weights": role_weights,
            "selector_patterns": ce.NAMED_NET_PATTERNS,
            "top_k_support": list(KS),
        },
        "provenance": {
            **repo_descriptor(source_root),
            "files": [source_descriptor(source_root, path) for path in paths],
        },
        "archived_cross_checks": {
            "metrics_by_set_v2": read_json(demand_dir / "results/e4_conditional/metrics_by_set_v2.json"),
            "per_role_v2": read_json(demand_dir / "results/e4_conditional/per_role_v2.json"),
        },
        "sets": sets,
    }
    write_json(artifact, out)
    print(f"exported {sum(len(v['items']) for v in sets.values())} E4 items "
          f"({artifact.stat().st_size} bytes) -> {artifact}")


def aggregate_items(items: list[dict], arm: str) -> dict:
    result = {"n_eval": len(items), "top_k": {}, "per_role_recall": {}}
    for k in KS:
        recall_sum = 0.0
        hit_sum = 0
        role_counts = defaultdict(lambda: [0, 0])
        for item in items:
            truth = set(item["gold_functions"])
            pred = set(item["top20"][arm][:k])
            overlap = truth & pred
            recall_sum += len(overlap) / len(truth)
            hit_sum += bool(overlap)
            for fn in truth:
                role = item["gold_function_roles"][fn]
                role_counts[role][0] += int(fn in pred)
                role_counts[role][1] += 1
        result["top_k"][str(k)] = {
            "recall": round(recall_sum / len(items), 4),
            "hit_rate": round(hit_sum / len(items), 4),
        }
        for role, (hits, total) in role_counts.items():
            result["per_role_recall"].setdefault(role, {})[str(k)] = round(hits / total, 4)
    result["per_role_counts_at_10"] = {}
    for role in result["per_role_recall"]:
        hits = total = 0
        for item in items:
            pred = set(item["top20"][arm][:10])
            for fn in item["gold_functions"]:
                if item["gold_function_roles"][fn] == role:
                    hits += int(fn in pred)
                    total += 1
        result["per_role_counts_at_10"][role] = {"hits": hits, "total": total}
    return result


def aggregate_role_reweighting(unadapted_path: Path, adapted_path: Path) -> dict:
    """Rebuild S4(b) from the two Hybrid suites deployed in the main study.

    Each arm tuned its mixing coefficient on its own development split, so the
    selected alpha is part of the frozen arm rather than a shared constant.
    """
    arms = {
        "before": (unadapted_path, read_json(unadapted_path)),
        "after": (adapted_path, read_json(adapted_path)),
    }
    split_map = {
        "augmented_test": "Aug-test",
        "benchmark_reference_eval_only": "Bench-ref",
    }
    recall = {}
    for source_split, display in split_map.items():
        before = arms["before"][1]["metrics"][source_split]["top_k"]["10"]["recall"]
        after = arms["after"][1]["metrics"][source_split]["top_k"]["10"]["recall"]
        recall[display] = {
            "before": before,
            "after": after,
            "delta_percentage_points": round((after - before) * 100, 2),
        }

    arm_details = {}
    for label, (metrics_path, data) in arms.items():
        run_dir = metrics_path.parent
        export_path = run_dir / "exports/qwen3_480b_candidates.json"
        # Runtime descriptors must be identical in a source archive without
        # .git and in a maintainer checkout.  Source provenance is already
        # pinned in the exported evidence; these release-local inputs are
        # therefore identified by path/bytes/SHA rather than checkout state.
        metrics_descriptor = source_descriptor(PCB_ROOT, metrics_path)
        export_descriptor = source_descriptor(PCB_ROOT, export_path)
        metrics_descriptor.pop("git_blob", None)
        export_descriptor.pop("git_blob", None)
        arm_details[label] = {
            "suite": run_dir.parent.name,
            "run_name": data["run_name"],
            "role_reweight": data["config"]["role_reweight"],
            "selected_alpha": data["alpha_tuning"]["selected"]["alpha"],
            "role_weights": data.get("role_weights"),
            "metrics_artifact": metrics_descriptor,
            "candidate_export": export_descriptor,
        }

    checks = {
        "before_role_reweight_is_false": arm_details["before"]["role_reweight"] is False,
        "after_role_reweight_is_true": arm_details["after"]["role_reweight"] is True,
        "before_has_no_role_weights": arm_details["before"]["role_weights"] is None,
        "after_has_role_weights": bool(arm_details["after"]["role_weights"]),
    }
    return {
        "scope": (
            "Frozen Hybrid demand suites deployed for the C-unadapted and "
            "C-adapted main-study arms. No standalone Pairwise-before run is claimed."
        ),
        "alpha_policy": "Each frozen arm independently selected alpha on its development split.",
        "arms": arm_details,
        "recall_at_10": recall,
        "integrity_checks": checks,
        "all_integrity_checks_pass": all(checks.values()),
    }


def aggregate(
    estimators_path: Path,
    e4_path: Path,
    unadapted_hybrid_path: Path,
    adapted_hybrid_path: Path,
    output: Path,
) -> None:
    estimators = read_json(estimators_path)
    panel = {}
    curves = {}
    split_map = {
        "augmented_test": "Aug-test",
        "augmented_test_original_only": "Aug-orig",
        "benchmark_reference_eval_only": "Bench-ref",
    }
    for label, run in estimators["runs"].items():
        panel[label] = {}
        curves[label] = {}
        for source_split, display in split_map.items():
            top_k = run["metrics"][source_split]["top_k"]
            panel[label][display] = {
                "recall@10": top_k["10"]["recall"],
                "hit@10": top_k["10"]["hit_rate"],
            }
            curves[label][display] = {
                k: {"recall": top_k[k]["recall"], "hit_rate": top_k[k]["hit_rate"]}
                for k in map(str, KS)
            }

    e4 = read_json(e4_path)
    e4_results = {}
    all_checks = True
    archived_metrics = e4["archived_cross_checks"]["metrics_by_set_v2"]
    archived = archived_metrics["comparison_recall_at_10"]
    archived_roles = e4["archived_cross_checks"]["per_role_v2"]["sets"]
    for name, data in e4["sets"].items():
        arms = {arm: aggregate_items(data["items"], arm)
                for arm in ("unadapted", "adapted", "conditional")}
        reference = archived[name]
        check = {
            "unadapted_recall@10": arms["unadapted"]["top_k"]["10"]["recall"] == reference["unadapted"],
            "adapted_recall@10": arms["adapted"]["top_k"]["10"]["recall"] == reference["adapted"],
            "conditional_recall@10": arms["conditional"]["top_k"]["10"]["recall"] == reference["conditional"],
            "unadapted_hit@10": arms["unadapted"]["top_k"]["10"]["hit_rate"] == reference["hit@10_unadapted"],
            "adapted_hit@10": arms["adapted"]["top_k"]["10"]["hit_rate"] == reference["hit@10_adapted"],
            "conditional_hit@10": arms["conditional"]["top_k"]["10"]["hit_rate"] == reference["hit@10_conditional"],
        }
        selector_distribution = {
            "construction": sum(i["selector_intent"] == "construction" for i in data["items"]),
            "analysis": sum(i["selector_intent"] == "analysis" for i in data["items"]),
        }
        check["selector_distribution"] = (
            selector_distribution
            == archived_metrics["sets"][name]["selector_distribution"]
        )
        for arm in ("unadapted", "adapted"):
            role_reference = archived_roles[name]["arms"][arm]
            check[f"{arm}_per_role@10"] = all(
                arms[arm]["per_role_recall"][role]["10"] == value
                for role, value in role_reference["per_role@10"].items()
            )
            check[f"{arm}_role_counts@10"] = (
                arms[arm]["per_role_counts_at_10"] == role_reference["role_counts"]
            )
        all_checks = all_checks and all(check.values())
        e4_results[name] = {
            "selector_distribution": selector_distribution,
            "arms": arms,
            "archived_aggregate_check": check,
        }

    role_reweighting = aggregate_role_reweighting(
        unadapted_hybrid_path,
        adapted_hybrid_path,
    )
    out = {
        "source_artifacts": [
            estimators_path.relative_to(PCB_ROOT).as_posix(),
            e4_path.relative_to(PCB_ROOT).as_posix(),
            unadapted_hybrid_path.relative_to(PCB_ROOT).as_posix(),
            adapted_hybrid_path.relative_to(PCB_ROOT).as_posix(),
        ],
        "s4_six_estimator_table": panel,
        "s4_recall_hit_curves": curves,
        "s4_role_reweighting_deployed_hybrid": role_reweighting,
        "e4_item_reaggregation": e4_results,
        "all_s4_role_reweighting_checks_pass": role_reweighting["all_integrity_checks_pass"],
        "all_e4_archived_checks_pass": all_checks,
    }
    write_json(output, out)
    print(
        "rebuilt S4/E4 evidence; "
        f"S4 checks pass={role_reweighting['all_integrity_checks_pass']}; "
        f"E4 archived checks pass={all_checks} -> {output}"
    )
    if not role_reweighting["all_integrity_checks_pass"]:
        raise SystemExit("S4 role-reweighting suites do not have the expected arm semantics")
    if not all_checks:
        raise SystemExit("E4 item-level reaggregation does not match archived aggregate")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="export both source artifacts")
    parser.add_argument("--export-estimators", action="store_true")
    parser.add_argument("--export-e4", action="store_true")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--estimators", type=Path, default=DEFAULT_ESTIMATORS)
    parser.add_argument("--e4", type=Path, default=DEFAULT_E4)
    parser.add_argument("--unadapted-hybrid", type=Path, default=DEFAULT_UNADAPTED_HYBRID)
    parser.add_argument("--adapted-hybrid", type=Path, default=DEFAULT_ADAPTED_HYBRID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source_root.resolve()
    if args.export or args.export_estimators:
        export_estimators(source, args.estimators.resolve())
    if args.export or args.export_e4:
        export_e4(source, args.e4.resolve())
    aggregate(
        args.estimators.resolve(),
        args.e4.resolve(),
        args.unadapted_hybrid.resolve(),
        args.adapted_hybrid.resolve(),
        args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
