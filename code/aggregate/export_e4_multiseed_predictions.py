#!/usr/bin/env python3
"""Deterministically reconstruct the unarchived E4 five-seed item rankings.

The contemporaneous seed-42--46 files retain aggregate metrics and selector
decisions, but not per-query rankings.  This maintainer exporter reruns the
frozen lightweight Hybrid TF-IDF pipeline from the public PowerCodeBench
inputs and stores the two independent arms' top-20 rankings.  The conditional
arm is not stored separately: it is the frozen selector's per-query choice
between those two rankings.

Run with CPython 3.11.15 and the exact packages in
``e4_reconstruction_requirements.txt``.  The script refuses to export if an
input, package version, or archived aggregate differs from the frozen record.
The resulting JSON is timestamp-free and byte-deterministic.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Fix numerical-library concurrency before importing NumPy/scikit-learn.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

HERE = Path(__file__).resolve()
PCB_ROOT = HERE.parents[2]
TASK_DEMAND_DIR = PCB_ROOT / "code/task_demand"
DEFAULT_OUTPUT = PCB_ROOT / "results/supplementary_evidence/e4_multiseed_predictions.json"
DEFAULT_BASE = PCB_ROOT / "results/supplementary_evidence/e4_item_predictions.json"
SEEDS = (42, 43, 44, 45, 46)
KS = (1, 3, 5, 10, 20)
SET_ORDER = (
    "layer1_holdout_n80",
    "layer2a_n84",
    "layer2a_ext_n10",
    "layer2b_n20",
)
EVAL_SETS = {
    "layer1_holdout_n80": "benchmark/naturalistic_holdout/naturalistic_holdout_seed22_n80.json",
    "layer2a_n84": "benchmark/e4_layer2a/e4_layer2a_set.json",
    "layer2a_ext_n10": "benchmark/e4_layer2a/e4_layer2a_ext.json",
    "layer2b_n20": "benchmark/e4_layer2b/e4_layer2b_set.json",
}

EXPECTED_PYTHON = "3.11.15"
EXPECTED_PACKAGES = {
    "numpy": "2.2.6",
    "scipy": "1.16.3",
    "scikit-learn": "1.6.0",
    "joblib": "1.5.3",
    "threadpoolctl": "3.6.0",
}
EXPECTED_INPUT_HASHES = {
    "code/task_demand/e4_conditional_eval.py": "10df0630f8d4b44cf66b20e8a0b377e4913a5c8e93d8041ce14faeb07816ddf4",
    "code/task_demand/task_demand_model.py": "3c66d7ad948a2fa9cf03bfe9e892627fb8819d93c38d42b03c89d593c107efda",
    "dataset/augmented_dataset.json": "95ba0ab44abffacd8d1fecb81a4d9e8f60b78afc6ace5909702f7271fa4abad8",
    "dataset/pandapower_docs.json": "f2fd02dda4816bced0f61bd2377a302d4e0e2877d2845358698869a448b76ed3",
    "benchmark.json": "dfcea3b3408d58ca676f01112b7ae61ef1bf077e14b440e0126a745bcad7130d",
    "benchmark/naturalistic_holdout/naturalistic_holdout_seed22_n80.json": "3c22226db28101232d0524bd6183475295cad167bfb75bf891e408f22969daad",
    "benchmark/e4_layer2a/e4_layer2a_set.json": "30e8431899ded9ccf6d00f52493f0a43582ed42852b264b417f5f2974a8cade7",
    "benchmark/e4_layer2a/e4_layer2a_ext.json": "77fad23f3d2db64e8c9c8a8f865be6c2968b7a0438fa444aeaea781702824a57",
    "benchmark/e4_layer2b/e4_layer2b_set.json": "c166973801b3a204a6864329d910d69302759f5c7349b0048526c19bc7b7c4e6",
    "results/supplementary_evidence/e4_item_predictions.json": "b6188f979b298fa77ccf70ff07bae7d4cec32c15ca21410dba94fd14d9f8d269",
    "external_queries/conditional_selector/seed_sensitivity/metrics_by_set_seed42.json": "be27159f69c958e3d184ee49aa8ef7ca5704979eaaeff42f0bff9df4af16790b",
    "external_queries/conditional_selector/seed_sensitivity/metrics_by_set_seed43.json": "25518c0b5b6ad0e80a54da58983ce947ac0bd95c0391c5a27484156da54198ab",
    "external_queries/conditional_selector/seed_sensitivity/metrics_by_set_seed44.json": "68cbeb44c95ab69f23fb0a0c84dc654f2846c41c8405c6af6e7573883bb7c67a",
    "external_queries/conditional_selector/seed_sensitivity/metrics_by_set_seed45.json": "cb78091e7081def5967fc9f4a59c45c6b9919eca3be72bf6dc7ecabe98d5a4b2",
    "external_queries/conditional_selector/seed_sensitivity/metrics_by_set_seed46.json": "fb8c2e7f149983bbe859a30bd49dbdbed3c286641d7d3af6c0ab41082cf8c484",
}

# Historical source pins retained to distinguish reconstruction from raw data.
HISTORICAL_PROVENANCE = {
    "repository": "frozen experimental pipeline",
    "snapshot_commit": "425e82847b5ff4b9c4cd4c6ee09e9c79f6950e3e",
    "determinism_fix_commit": "eb5f3464452b2d586c1220faa0b0682173c933b3",
    "original_e4_code_sha256": "090471815dc7f075e707313daf5863b8bd9beb37086cf0840475bdd4805c7c92",
    "original_model_code_sha256": "b7755afd56eb0de0ce1cc730367aed742e7d45840676849b6c3cecc516b04658",
    "original_augmented_corpus_sha256": "19ddccf566f0cf1938abd94ebbc7025e07d4d2fc62e55fe52235ffbc1b080d73",
    "training_projection_sha256": "8022a8b67e342e14eab8917e075a3913a67442ce93f2364be901f6825c966158",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        stream.write("\n")


def git_head(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def descriptor(relative: str) -> dict:
    path = PCB_ROOT / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_frozen_inputs() -> list[dict]:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("set PYTHONHASHSEED=0 before launching this exporter")
    if platform.python_version() != EXPECTED_PYTHON:
        raise SystemExit(
            f"Python mismatch: expected {EXPECTED_PYTHON}, got {platform.python_version()}"
        )
    for package, expected in EXPECTED_PACKAGES.items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise SystemExit(f"package mismatch for {package}: expected {expected}, got {actual}")

    files = []
    for relative, expected in EXPECTED_INPUT_HASHES.items():
        path = PCB_ROOT / relative
        if not path.is_file():
            raise SystemExit(f"missing frozen input: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(f"input hash mismatch for {relative}: expected {expected}, got {actual}")
        files.append(descriptor(relative))
    return files


def training_projection_sha256(path: Path) -> str:
    rows = read_json(path)["samples"]
    projected = [
        {key: row.get(key) for key in ("id", "question", "functions", "attributes")}
        for row in rows
    ]
    payload = json.dumps(
        projected, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_args(tdm, role_reweight: bool, seed: int):
    return argparse.Namespace(
        augmented_path=tdm.DEFAULT_AUGMENTED_PATH,
        docs_path=tdm.DEFAULT_DOCS_PATH,
        model="hybrid_tfidf",
        include_variants=True,
        seed=seed,
        negative_per_positive=8,
        max_train_pairs=None,
        max_positive_per_function=None,
        hybrid_alpha=0.5,
        auto_hybrid_alpha=True,
        hybrid_alpha_target_k=10,
        sbert_model="sentence-transformers/all-MiniLM-L6-v2",
        sbert_batch_size=64,
        sentence_transformers_device=None,
        role_reweight=role_reweight,
        max_role_weight=3.0,
    )


def train_arm_and_score(tdm, arm: str, questions: list[str], seed: int):
    """Public-layout equivalent of the frozen E4 arm reproduction helper."""
    args = make_args(tdm, role_reweight=(arm == "adapted"), seed=seed)
    random.seed(seed)
    tdm.np.random.seed(seed)
    cards = tdm.load_function_cards(args.docs_path)
    function_names = set(cards)
    examples = tdm.load_augmented_examples(
        args.augmented_path, function_names, include_variants=True
    )
    train, dev, _test = tdm.split_by_group(examples, seed=seed)
    ranker = tdm.build_ranker(args)

    role_weights = None
    if arm == "adapted":
        benchmark = tdm.load_benchmark_reference_examples(
            PCB_ROOT / "benchmark.json", function_names, max_items=None
        )
        role_weights = tdm.compute_role_weights(
            train, benchmark, cards, max_weight=args.max_role_weight
        )

    tdm.train_or_fit_ranker(ranker, args, train, cards, role_weights=role_weights)
    tdm.tune_hybrid_alpha(
        ranker,
        dev,
        cards,
        candidate_alphas=[round(value * 0.1, 1) for value in range(11)],
        target_k=10,
    )
    scores = {question: ranker.score(question) for question in questions}
    alpha = ranker.alpha
    del ranker
    gc.collect()
    return alpha, role_weights, scores


def aggregate_top20(items: list[dict], arm: str) -> dict:
    metrics = {}
    for k in KS:
        sums = defaultdict(float)
        for item in items:
            truth = set(item["gold_functions"])
            predicted = set(item["top20"][arm][:k])
            overlap = truth & predicted
            sums["recall"] += len(overlap) / len(truth)
            sums["precision"] += len(overlap) / k
            sums["hit_rate"] += bool(overlap)
        metrics[str(k)] = {
            name: round(total / len(items), 4) for name, total in sums.items()
        }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-items", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()

    input_descriptors = verify_frozen_inputs()
    projection_hash = training_projection_sha256(PCB_ROOT / "dataset/augmented_dataset.json")
    if projection_hash != HISTORICAL_PROVENANCE["training_projection_sha256"]:
        raise SystemExit(
            "training-field projection differs from the frozen original: " + projection_hash
        )

    sys.path.insert(0, str(TASK_DEMAND_DIR))
    ce = importlib.import_module("e4_conditional_eval")
    tdm = importlib.import_module("task_demand_model")

    cards = tdm.load_function_cards(tdm.DEFAULT_DOCS_PATH)
    function_names = set(cards)
    set_examples = {}
    all_questions = []
    for set_name in SET_ORDER:
        examples = tdm.load_benchmark_reference_examples(
            PCB_ROOT / EVAL_SETS[set_name], function_names
        )
        set_examples[set_name] = examples
        all_questions.extend(example.question for example in examples)

    base = read_json(args.base_items)
    base_items = {
        (set_name, item["sample_id"]): item
        for set_name, set_data in base["sets"].items()
        for item in set_data["items"]
    }
    axis = []
    item_metadata = []
    for set_name in SET_ORDER:
        for example in set_examples[set_name]:
            key = (set_name, example.sample_id)
            if key not in base_items:
                raise SystemExit(f"seed-22 item evidence missing {key}")
            archived = base_items[key]
            gold = sorted(example.functions & function_names)
            selector = ce.selector_intent(example.question)
            if archived["query"] != example.question:
                raise SystemExit(f"query mismatch for {key}")
            if archived["gold_functions"] != gold:
                raise SystemExit(f"gold-label mismatch for {key}")
            if archived["selector_intent"] != selector:
                raise SystemExit(f"selector mismatch for {key}")
            axis.append([set_name, example.sample_id])
            item_metadata.append({
                "set": set_name,
                "sample_id": example.sample_id,
                "question": example.question,
                "gold_functions": gold,
                "selector_intent": selector,
            })
    if len(axis) != 194 or len(base_items) != 194:
        raise SystemExit(f"expected 194 aligned E4 items, got {len(axis)} / {len(base_items)}")

    raw_predictions = {}
    pipeline = {}
    all_ranked_names = set()
    for seed in SEEDS:
        print(f"reconstructing E4 seed {seed} (unadapted + adapted)", flush=True)
        ua_alpha, _, ua_scores = train_arm_and_score(tdm, "unadapted", all_questions, seed)
        ad_alpha, role_weights, ad_scores = train_arm_and_score(tdm, "adapted", all_questions, seed)
        per_arm = {"unadapted": [], "adapted": []}
        for item in item_metadata:
            for arm, score_map in (("unadapted", ua_scores), ("adapted", ad_scores)):
                ranking = tdm.rank_names(score_map[item["question"]], 20)
                if len(ranking) != 20 or len(set(ranking)) != 20:
                    raise SystemExit(f"invalid top-20 for seed={seed}, item={item['sample_id']}, arm={arm}")
                per_arm[arm].append(ranking)
                all_ranked_names.update(ranking)
        raw_predictions[str(seed)] = per_arm
        pipeline[str(seed)] = {
            "unadapted_alpha": ua_alpha,
            "adapted_alpha": ad_alpha,
            "adapted_role_weights": role_weights,
        }

        # Before retaining reconstructed rankings, require every top-k metric
        # to reproduce the independently archived contemporaneous summary.
        summary = read_json(
            PCB_ROOT
            / f"external_queries/conditional_selector/seed_sensitivity/metrics_by_set_seed{seed}.json"
        )
        if summary["arms"]["unadapted"]["alpha"] != ua_alpha:
            raise SystemExit(f"seed {seed} unadapted alpha mismatch")
        if summary["arms"]["adapted"]["alpha"] != ad_alpha:
            raise SystemExit(f"seed {seed} adapted alpha mismatch")
        if summary["arms"]["adapted"]["role_weights"] != role_weights:
            raise SystemExit(f"seed {seed} role-weight mismatch")
        cursor = 0
        for set_name in SET_ORDER:
            n_items = len(set_examples[set_name])
            rows = []
            for offset in range(n_items):
                metadata = dict(item_metadata[cursor + offset])
                metadata["top20"] = {
                    arm: per_arm[arm][cursor + offset] for arm in ("unadapted", "adapted")
                }
                selected = (
                    "adapted" if metadata["selector_intent"] == "analysis" else "unadapted"
                )
                metadata["top20"]["conditional"] = metadata["top20"][selected]
                rows.append(metadata)
            cursor += n_items
            for arm in ("unadapted", "adapted", "conditional"):
                recomputed = aggregate_top20(rows, arm)
                archived = summary["sets"][set_name]["metrics"][arm]
                if recomputed != archived:
                    raise SystemExit(
                        f"seed {seed} {set_name} {arm} aggregate mismatch: "
                        f"recomputed={recomputed}, archived={archived}"
                    )

    function_dictionary = sorted(all_ranked_names)
    function_ids = {name: index for index, name in enumerate(function_dictionary)}
    encoded = {
        seed: {
            arm: [[function_ids[name] for name in ranking] for ranking in rankings]
            for arm, rankings in per_arm.items()
        }
        for seed, per_arm in raw_predictions.items()
    }

    out = {
        "schema_version": 1,
        "evidence_kind": "deterministic reconstruction of previously unarchived item rankings",
        "evidence_unit": "training seed x E4 query x independent arm top-20 rank",
        "seeds": list(SEEDS),
        "top_k_support": list(KS),
        "set_order": list(SET_ORDER),
        "item_axis": axis,
        "function_dictionary": function_dictionary,
        "top20_ids_by_seed": encoded,
        "pipeline_by_seed": pipeline,
        "conditional_derivation": (
            "Use the adapted ranking iff the frozen selector labels the query analysis; "
            "otherwise use the unadapted ranking. Gold labels and selector decisions are "
            "read from e4_item_predictions.json by the release recomputer."
        ),
        "provenance": {
            "release_repository": "PowerCodeBench",
            "release_commit_at_export": git_head(PCB_ROOT),
            "release_inputs": input_descriptors,
            "historical_source": HISTORICAL_PROVENANCE,
            "historical_raw_item_status": (
                "No seed-42--46 per-query rankings were archived in Git history, reflog, "
                "or Git LFS; only aggregate metrics and selector decisions survived."
            ),
            "reconstruction_validation": (
                "All 900 archived metric cells (5 seeds x 4 sets x 3 arms x 5 k values "
                "x recall/precision/hit-rate), all alphas, and all adapted role weights "
                "matched before export."
            ),
            "training_projection_sha256": projection_hash,
            "environment": {
                "python": platform.python_version(),
                "packages": EXPECTED_PACKAGES,
                "numeric_threads": 1,
                "pythonhashseed": os.environ["PYTHONHASHSEED"],
            },
        },
    }
    write_json(args.output, out)
    print(f"wrote {len(axis)} items x {len(SEEDS)} seeds x 2 arms -> {args.output}")
    print(f"sha256={sha256_file(args.output)} bytes={args.output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
