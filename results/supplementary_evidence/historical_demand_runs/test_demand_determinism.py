"""Regression test: demand-model training sequences are PYTHONHASHSEED-independent.

Guards the 2026-08-02 P0 fix in ``task_demand/task_demand_model.py``
(``split_by_group``): train/dev/test group ids were previously placed in a
``set`` and iterated directly, so with an identical training seed the
train/dev/test example sequences — and therefore pairwise training order and
negative sampling — varied across processes with different ``PYTHONHASHSEED``.

The test spawns one subprocess per PYTHONHASHSEED in {0, 1, 2}; each builds a
small training-pair sequence (subset of the train split, reduced negatives so
the test stays light) and reports SHA-256 digests of the split sequence and the
pair sequence. All three runs must agree bit-for-bit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_DEMAND_DIR = PROJECT_ROOT / "task_demand"

_WORKER_SNIPPET = r"""
import hashlib, json, os, sys
sys.path.insert(0, sys.argv[1])
import task_demand_model as tdm

SEED = 22
cards = tdm.load_function_cards(tdm.DEFAULT_DOCS_PATH)
examples = tdm.load_augmented_examples(
    tdm.DEFAULT_AUGMENTED_PATH, set(cards), include_variants=True)
train, dev, test = tdm.split_by_group(examples, seed=SEED)

# Small subset + reduced negatives: keeps the test light while still exercising
# the split-order and negative-sampling paths end to end.
train_sub = train[:300]
ranker = tdm.PairwiseTfidfLogRegDemandRanker(negative_per_positive=4, seed=SEED)
rows, labels, weights = ranker._make_pairs(train_sub, cards)

split_blob = json.dumps({
    "train": [ex.sample_id for ex in train],
    "dev": [ex.sample_id for ex in dev],
    "test": [ex.sample_id for ex in test],
}).encode()
pairs_blob = json.dumps(
    {"rows": rows, "labels": labels, "weights": weights}).encode()
print(json.dumps({
    "hashseed": os.environ.get("PYTHONHASHSEED"),
    "n_pairs": len(rows),
    "split_sha": hashlib.sha256(split_blob).hexdigest(),
    "pairs_sha": hashlib.sha256(pairs_blob).hexdigest(),
}))
"""


class DemandDeterminismTest(unittest.TestCase):
    def test_training_sequence_sha_identical_across_hashseeds(self):
        results = []
        for hashseed in ("0", "1", "2"):
            env = dict(os.environ)
            env.update({
                "PYTHONHASHSEED": hashseed,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            })
            proc = subprocess.run(
                [sys.executable, "-c", _WORKER_SNIPPET, str(TASK_DEMAND_DIR)],
                env=env, capture_output=True, text=True, timeout=110,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            results.append(json.loads(proc.stdout.strip().splitlines()[-1]))

        split_shas = {r["split_sha"] for r in results}
        pairs_shas = {r["pairs_sha"] for r in results}
        n_pairs = {r["n_pairs"] for r in results}
        self.assertEqual(len(n_pairs), 1, msg=f"pair counts diverged: {results}")
        self.assertEqual(
            len(split_shas), 1,
            msg=f"split sequence depends on PYTHONHASHSEED: {results}")
        self.assertEqual(
            len(pairs_shas), 1,
            msg=f"training-pair sequence depends on PYTHONHASHSEED: {results}")


if __name__ == "__main__":
    unittest.main()
