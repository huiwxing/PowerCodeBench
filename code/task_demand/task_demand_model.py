# --------------------------------------------------------------------------
# PowerCodeBench repository copy of the pipeline module task_demand/task_demand_model.py, with
# import statements and data-path constants adapted to this repository's
# layout (the original is untouched in the frozen experimental pipeline).
# Runs CPU-only against the archived corpus/spec files in this repository.
# See code/README.md for the module map.
# --------------------------------------------------------------------------
"""
Task Knowledge Demand Modeling for the PowerCodeBench pipeline.

This module is intentionally independent from benchmark generation/evaluation.
It turns question-function supervision from ``augmented_dataset.json`` into a
query-only API primitive demand predictor, then optionally evaluates the same
predictor on benchmark reference labels as an oracle-free downstream diagnostic.

The main design goals are:
  - do not train on the benchmark release when evaluating its demand recall;
  - split augmented samples by base id so LLM-generated variants do not leak;
  - keep a zero-shot retrieval baseline comparable to the earlier RAG attempt;
  - provide a supervised pairwise ranker that can be swapped for stronger
    encoders/rerankers later.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


DEFAULT_RANDOM_SEED = 22
TASK_DEMAND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DEMAND_DIR.parent.parent  # repository root (code/task_demand/ -> repo)
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
DEFAULT_AUGMENTED_PATH = PROJECT_ROOT / "dataset" / "augmented_dataset.json"
DEFAULT_DOCS_PATH = PROJECT_ROOT / "dataset" / "pandapower_docs.json"
DEFAULT_OUTPUT_DIR = TASK_DEMAND_DIR / "results"


@dataclass(frozen=True)
class FunctionCard:
    name: str
    full_path: str
    category: str
    signature: str
    description: str
    parameters: Tuple[str, ...]
    examples: Tuple[str, ...]

    @classmethod
    def from_doc(cls, doc: Dict) -> "FunctionCard":
        params = []
        for param in doc.get("parameters", []) or []:
            pname = str(param.get("name", "")).strip()
            ptype = str(param.get("type", "")).strip()
            pdesc = str(param.get("description", "")).strip()
            required = "required" if param.get("required") else "optional"
            pieces = [pname]
            if ptype:
                pieces.append(f"type={ptype}")
            if required:
                pieces.append(required)
            if pdesc:
                pieces.append(pdesc[:160])
            params.append(" | ".join(p for p in pieces if p))

        return cls(
            name=str(doc.get("name", "")).strip(),
            full_path=str(doc.get("full_path", "")).strip(),
            category=str(doc.get("category", "")).strip(),
            signature=str(doc.get("signature", "")).strip(),
            description=str(doc.get("description", "")).strip(),
            parameters=tuple(params),
            examples=tuple(str(x).strip() for x in (doc.get("examples") or []) if str(x).strip()),
        )

    def text(self, include_examples: bool = True) -> str:
        parts = [
            f"Function: {self.name}",
            f"Path: {self.full_path}",
            f"Category: {self.category}",
            f"Signature: {self.signature}",
            f"Description: {self.description}",
        ]
        if self.parameters:
            parts.append("Parameters: " + " ; ".join(self.parameters[:12]))
        if include_examples and self.examples:
            parts.append("Examples: " + " ; ".join(self.examples[:4]))
        return "\n".join(p for p in parts if p and not p.endswith(": "))


@dataclass
class DemandExample:
    sample_id: str
    base_id: str
    question: str
    functions: Set[str]
    attributes: Set[str]
    is_variant: bool
    source: str


def _base_id(sample_id: str) -> str:
    return sample_id.split("_variant", 1)[0]


def load_function_cards(docs_path: Path) -> Dict[str, FunctionCard]:
    with docs_path.open() as f:
        raw = json.load(f)

    cards = {}
    for doc in raw.get("functions", []):
        card = FunctionCard.from_doc(doc)
        if card.name:
            cards[card.name] = card
    return cards


def load_augmented_examples(
    dataset_path: Path,
    function_names: Set[str],
    *,
    include_variants: bool = True,
    min_positive_labels: int = 1,
) -> List[DemandExample]:
    with dataset_path.open() as f:
        raw = json.load(f)
    samples = raw.get("samples", [])

    examples: List[DemandExample] = []
    dropped_empty = 0
    for sample in samples:
        sid = str(sample.get("id", ""))
        is_variant = "variant" in sid.lower()
        if is_variant and not include_variants:
            continue

        funcs = {
            str(fn).strip()
            for fn in (sample.get("functions") or [])
            if str(fn).strip() in function_names
        }
        attrs = {str(a).strip() for a in (sample.get("attributes") or []) if str(a).strip()}
        question = str(sample.get("question", "")).strip()
        if not question or len(funcs) < min_positive_labels:
            dropped_empty += 1
            continue

        examples.append(
            DemandExample(
                sample_id=sid,
                base_id=_base_id(sid),
                question=question,
                functions=funcs,
                attributes=attrs,
                is_variant=is_variant,
                source="augmented_dataset",
            )
        )

    if dropped_empty:
        print(f"  Dropped {dropped_empty} augmented samples with no usable labels/question")
    return examples


def split_by_group(
    examples: Sequence[DemandExample],
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
) -> Tuple[List[DemandExample], List[DemandExample], List[DemandExample]]:
    groups: Dict[str, List[DemandExample]] = defaultdict(list)
    for ex in examples:
        groups[ex.base_id].append(ex)

    rng = random.Random(seed)
    group_ids = list(groups)
    rng.shuffle(group_ids)

    n_groups = len(group_ids)
    n_train = int(round(n_groups * train_ratio))
    n_dev = int(round(n_groups * dev_ratio))
    # Deterministic iteration order: sets iterate in PYTHONHASHSEED-dependent
    # order, which previously made the train/dev/test example sequences (and
    # hence pairwise training order + negative sampling) vary across processes
    # with identical seeds. sorted() fixes the traversal order.
    train_ids = sorted(group_ids[:n_train])
    dev_ids = sorted(group_ids[n_train:n_train + n_dev])
    test_ids = sorted(group_ids[n_train + n_dev:])

    train = [ex for gid in train_ids for ex in groups[gid]]
    dev = [ex for gid in dev_ids for ex in groups[gid]]
    test = [ex for gid in test_ids for ex in groups[gid]]
    return train, dev, test


def _pair_text(question: str, card: FunctionCard) -> str:
    return f"Query:\n{question}\n\nCandidate API:\n{card.text(include_examples=True)}"


class ZeroShotTfidfDemandRanker:
    """Unsupervised query-to-function-card retrieval baseline."""

    def __init__(self):
        self.vectorizer = None
        self.card_matrix = None
        self.function_names: List[str] = []

    def fit(self, cards: Dict[str, FunctionCard]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        self.function_names = sorted(cards)
        texts = [cards[name].text(include_examples=True) for name in self.function_names]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.card_matrix = normalize(self.vectorizer.fit_transform(texts), norm="l2")
        return self

    def score(self, query: str) -> Dict[str, float]:
        from sklearn.preprocessing import normalize

        q_vec = normalize(self.vectorizer.transform([query]), norm="l2")
        scores = (q_vec @ self.card_matrix.T).toarray()[0]
        return {name: float(score) for name, score in zip(self.function_names, scores)}


class PairwiseTfidfLogRegDemandRanker:
    """
    Supervised pairwise demand ranker.

    Each training row is a (query, function_card) pair. Positives come from
    augmented_dataset["functions"]; negatives are sampled from the remaining API
    candidates. The model remains lightweight and library-portable because the
    candidate API is represented by text, not by fixed label ids alone.
    """

    def __init__(
        self,
        *,
        negative_per_positive: int = 8,
        max_train_pairs: Optional[int] = None,
        max_positive_per_function: Optional[int] = None,
        seed: int = DEFAULT_RANDOM_SEED,
    ):
        self.negative_per_positive = negative_per_positive
        self.max_train_pairs = max_train_pairs
        self.max_positive_per_function = max_positive_per_function
        self.seed = seed
        self.pipeline = None
        self.cards: Dict[str, FunctionCard] = {}
        self.function_names: List[str] = []

    def _make_pairs(
        self,
        examples: Sequence[DemandExample],
        cards: Dict[str, FunctionCard],
        role_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], List[int], List[float]]:
        """Build (text, label, weight) triples for pairwise training.

        When *role_weights* is provided each positive pair (query, function)
        receives the weight corresponding to the function's role; its sampled
        negatives receive the same weight so the local positive/negative ratio
        is preserved within each role's contribution.
        """
        rng = random.Random(self.seed)
        names = sorted(cards)
        rows: List[str] = []
        labels: List[int] = []
        weights: List[float] = []
        positive_counts = Counter()
        shuffled_examples = list(examples)
        rng.shuffle(shuffled_examples)

        for ex in shuffled_examples:
            positives = sorted(ex.functions & set(cards))
            if not positives:
                continue
            negatives = [n for n in names if n not in ex.functions]
            for fn in positives:
                if (
                    self.max_positive_per_function is not None
                    and positive_counts[fn] >= self.max_positive_per_function
                ):
                    continue
                positive_counts[fn] += 1
                role = infer_function_role(fn, cards.get(fn))
                w = role_weights.get(role, 1.0) if role_weights else 1.0
                rows.append(_pair_text(ex.question, cards[fn]))
                labels.append(1)
                weights.append(w)
                if negatives and self.negative_per_positive > 0:
                    k = min(self.negative_per_positive, len(negatives))
                    for neg in rng.sample(negatives, k=k):
                        rows.append(_pair_text(ex.question, cards[neg]))
                        labels.append(0)
                        weights.append(w)   # same weight as its positive

        if self.max_train_pairs and len(rows) > self.max_train_pairs:
            idxs = list(range(len(rows)))
            rng.shuffle(idxs)
            idxs = idxs[:self.max_train_pairs]
            rows = [rows[i] for i in idxs]
            labels = [labels[i] for i in idxs]
            weights = [weights[i] for i in idxs]

        return rows, labels, weights

    def fit(
        self,
        examples: Sequence[DemandExample],
        cards: Dict[str, FunctionCard],
        role_weights: Optional[Dict[str, float]] = None,
    ):
        """Fit the pairwise ranker.

        Args:
            role_weights: optional per-role sample weights computed by
                ``compute_role_weights()``.  When provided, each training pair
                is weighted by the role of the positive function so that
                under-represented target roles receive higher gradient.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        self.cards = dict(cards)
        self.function_names = sorted(cards)
        rows, labels, weights = self._make_pairs(examples, cards, role_weights=role_weights)
        if not rows:
            raise ValueError("No training pairs were created.")

        n_pos = sum(labels)
        print(f"  Training pairwise TF-IDF+LogReg on {len(rows)} pairs "
              f"({n_pos} positives, {len(labels) - n_pos} negatives)"
              + (f" [role-reweighted]" if role_weights else ""))
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=self.seed,
            )),
        ])
        # Pass sample weights via Pipeline's fit_params mechanism.
        # TfidfVectorizer ignores unknown fit_params; LogisticRegression uses them.
        fit_params = {"clf__sample_weight": weights} if role_weights else {}
        self.pipeline.fit(rows, labels, **fit_params)
        return self

    def score(self, query: str) -> Dict[str, float]:
        rows = [_pair_text(query, self.cards[name]) for name in self.function_names]
        probs = self.pipeline.predict_proba(rows)[:, 1]
        return {name: float(score) for name, score in zip(self.function_names, probs)}


def _minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if math.isclose(lo, hi):
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridTfidfDemandRanker:
    """
    Weighted fusion of zero-shot API-card retrieval and supervised pairwise TF-IDF.

    This is useful when augmented supervision is valuable but distribution-shifted:
    the zero-shot branch preserves lexical/API-doc matching, while the pairwise
    branch learns task-function associations from question-function labels.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        negative_per_positive: int = 8,
        max_train_pairs: Optional[int] = None,
        max_positive_per_function: Optional[int] = None,
        seed: int = DEFAULT_RANDOM_SEED,
    ):
        self.alpha = alpha
        self.zero_shot = ZeroShotTfidfDemandRanker()
        self.pairwise = PairwiseTfidfLogRegDemandRanker(
            negative_per_positive=negative_per_positive,
            max_train_pairs=max_train_pairs,
            max_positive_per_function=max_positive_per_function,
            seed=seed,
        )

    def fit(
        self,
        examples: Sequence[DemandExample],
        cards: Dict[str, FunctionCard],
        role_weights: Optional[Dict[str, float]] = None,
    ):
        self.zero_shot.fit(cards)
        self.pairwise.fit(examples, cards, role_weights=role_weights)
        return self

    def score(self, query: str) -> Dict[str, float]:
        zs = _minmax_normalize(self.zero_shot.score(query))
        pw = _minmax_normalize(self.pairwise.score(query))
        names = set(zs) | set(pw)
        return {
            name: (1.0 - self.alpha) * zs.get(name, 0.0) + self.alpha * pw.get(name, 0.0)
            for name in names
        }


class SbertDemandRanker:
    """Optional zero-shot bi-encoder ranker using sentence-transformers."""

    def __init__(self, model_name: str, batch_size: int = 64, device: Optional[str] = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.function_names: List[str] = []
        self.card_embeddings = None

    def fit(self, cards: Dict[str, FunctionCard]):
        from sentence_transformers import SentenceTransformer

        model_kwargs = {"trust_remote_code": True}
        if self.device:
            model_kwargs["device"] = self.device
        self.model = SentenceTransformer(self.model_name, **model_kwargs)
        self.function_names = sorted(cards)
        texts = [cards[name].text(include_examples=True) for name in self.function_names]
        self.card_embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return self

    def score(self, query: str) -> Dict[str, float]:
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = np.dot(self.card_embeddings, q)
        return {name: float(score) for name, score in zip(self.function_names, scores)}


def rerank_with_cross_encoder(
    query: str,
    cards: Dict[str, FunctionCard],
    initial_scores: Dict[str, float],
    *,
    reranker,
    top_n: int = 50,
    batch_size: int = 512,
) -> Dict[str, float]:
    """Single-query CrossEncoder rerank of top-N candidates.

    Prefer ``batch_rerank_with_cross_encoder`` when evaluating many queries at
    once — it merges all (query, doc) pairs into one predict() call for much
    better GPU utilisation.
    """
    names = [name for name, _ in sorted(initial_scores.items(), key=lambda x: -x[1])[:top_n]]
    pairs = [[query, cards[name].text(include_examples=True)] for name in names]
    rerank_scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    merged = dict(initial_scores)
    if len(rerank_scores):
        for name, score in zip(names, rerank_scores):
            merged[name] = float(score)
        tail_floor = min(float(x) for x in rerank_scores) - 1.0
        min_base = min(initial_scores.values()) if initial_scores else 0.0
        for name in merged:
            if name not in names:
                merged[name] = min(tail_floor, min_base)
    return merged


def batch_rerank_with_cross_encoder(
    examples_scores: List[Tuple[str, Dict[str, float]]],
    cards: Dict[str, FunctionCard],
    *,
    reranker,
    top_n: int = 50,
    batch_size: int = 512,
) -> List[Dict[str, float]]:
    """Batch CrossEncoder rerank across *all* queries in one predict() call.

    Instead of calling ``reranker.predict(50_pairs)`` once per example (which
    launches a separate GPU kernel each time), this function collects every
    (query, doc) pair from all examples, runs a single large predict() with
    ``batch_size`` pairs per GPU batch, then distributes the scores back.

    Args:
        examples_scores: list of (query_text, first_stage_score_dict) tuples.
        cards: function card dict.
        reranker: loaded CrossEncoder instance.
        top_n: number of first-stage candidates to rerank per query.
        batch_size: pairs per GPU forward pass.  512 works well on A100/V100.

    Returns:
        List of merged score dicts, one per input example, in the same order.
    """
    if not examples_scores:
        return []

    # Build flat lists of (pair, example_idx, candidate_name) across all queries.
    all_pairs: List[List[str]] = []
    pair_index: List[Tuple[int, str]] = []   # (example_idx, function_name)

    for idx, (query, scores) in enumerate(examples_scores):
        top_names = [n for n, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]
        for name in top_names:
            all_pairs.append([query, cards[name].text(include_examples=True)])
            pair_index.append((idx, name))

    # Single batched inference call — GPU runs at full throughput.
    all_ce_scores = reranker.predict(
        all_pairs, batch_size=batch_size, show_progress_bar=len(all_pairs) > 500
    )

    # Scatter CE scores back to per-example dicts.
    ce_per_example: List[Optional[Dict[str, float]]] = [None] * len(examples_scores)
    for (idx, name), ce_score in zip(pair_index, all_ce_scores):
        if ce_per_example[idx] is None:
            ce_per_example[idx] = {}
        ce_per_example[idx][name] = float(ce_score)

    # Merge CE scores with base first-stage scores (untouched tail stays below).
    merged_list: List[Dict[str, float]] = []
    for idx, (query, base_scores) in enumerate(examples_scores):
        ce_dict = ce_per_example[idx]
        if not ce_dict:
            merged_list.append(dict(base_scores))
            continue
        tail_floor = min(ce_dict.values()) - 1.0
        min_base = min(base_scores.values()) if base_scores else 0.0
        merged = {
            name: (ce_dict[name] if name in ce_dict else min(tail_floor, min_base))
            for name in base_scores
        }
        merged_list.append(merged)

    return merged_list


def rank_names(scores: Dict[str, float], top_k: int) -> List[str]:
    return [name for name, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]]


def _load_cross_encoder(
    cross_encoder_model: str,
    sentence_transformers_device: Optional[str],
    fp16: bool = False,
):
    """Load a CrossEncoder, optionally in FP16 for faster GPU inference."""
    from sentence_transformers import CrossEncoder

    reranker_kwargs: Dict = {"max_length": 512}
    if sentence_transformers_device:
        reranker_kwargs["device"] = sentence_transformers_device
    if fp16:
        import torch
        reranker_kwargs["automodel_args"] = {"torch_dtype": torch.float16}
    return CrossEncoder(cross_encoder_model, **reranker_kwargs)


def evaluate_ranker(
    ranker,
    examples: Sequence[DemandExample],
    *,
    cards: Dict[str, FunctionCard],
    top_ks: Sequence[int] = (1, 3, 5, 10, 20),
    cross_encoder_model: Optional[str] = None,
    cross_encoder_top_n: int = 50,
    ce_batch_size: int = 512,
    ce_fp16: bool = False,
    sentence_transformers_device: Optional[str] = None,
) -> Dict:
    """Evaluate a demand ranker on a set of examples.

    When a cross-encoder model is provided, all (query, candidate) pairs across
    every example are batched into a **single** ``reranker.predict()`` call
    (``batch_rerank_with_cross_encoder``).  This eliminates the per-example GPU
    kernel launch overhead and typically yields 5-10× speedup over the old
    one-example-at-a-time approach.
    """
    reranker = None
    if cross_encoder_model:
        reranker = _load_cross_encoder(cross_encoder_model, sentence_transformers_device, fp16=ce_fp16)

    # ── Phase 1: first-stage scoring (CPU, fast) ─────────────────────────────
    valid: List[Tuple[DemandExample, Set[str], Dict[str, float]]] = []
    for ex in examples:
        truth = ex.functions & set(cards)
        if not truth:
            continue
        valid.append((ex, truth, ranker.score(ex.question)))

    # ── Phase 2: batched cross-encoder (all examples in one predict() call) ──
    if reranker and valid:
        print(f"  [CE] batch-reranking {len(valid)} examples × top_{cross_encoder_top_n} "
              f"= {len(valid) * cross_encoder_top_n:,} pairs (batch_size={ce_batch_size})")
        queries_scores = [(ex.question, scores) for ex, _, scores in valid]
        merged_scores = batch_rerank_with_cross_encoder(
            queries_scores, cards, reranker=reranker,
            top_n=cross_encoder_top_n, batch_size=ce_batch_size,
        )
        valid = [(ex, truth, ms) for (ex, truth, _), ms in zip(valid, merged_scores)]

    # ── Phase 3: compute metrics ──────────────────────────────────────────────
    totals = {k: {"recall": 0.0, "precision": 0.0, "hit": 0.0} for k in top_ks}
    per_role = defaultdict(lambda: {k: {"hits": 0, "total": 0} for k in top_ks})
    label_freq: Counter = Counter()

    for ex, truth, scores in valid:
        label_freq.update(truth)
        for k in top_ks:
            pred = set(rank_names(scores, k))
            overlap = pred & truth
            totals[k]["recall"] += len(overlap) / len(truth)
            totals[k]["precision"] += len(overlap) / max(k, 1)
            totals[k]["hit"] += 1.0 if overlap else 0.0
            for fn in truth:
                role = infer_function_role(fn, cards.get(fn))
                per_role[role][k]["total"] += 1
                if fn in pred:
                    per_role[role][k]["hits"] += 1

    n = len(valid)
    metrics = {"n_examples": n, "top_k": {}, "per_role_recall": {},
               "label_frequency": dict(label_freq.most_common())}
    if not n:
        return metrics

    for k, vals in totals.items():
        metrics["top_k"][str(k)] = {
            "recall": round(vals["recall"] / n, 4),
            "precision": round(vals["precision"] / n, 4),
            "hit_rate": round(vals["hit"] / n, 4),
        }
    for role, role_vals in per_role.items():
        metrics["per_role_recall"][role] = {}
        for k, vals in role_vals.items():
            total = vals["total"]
            metrics["per_role_recall"][role][str(k)] = round(vals["hits"] / total, 4) if total else 0.0

    return metrics


def infer_function_role(name: str, card: Optional[FunctionCard]) -> str:
    category = (card.category if card else "").lower()
    lname = name.lower()
    if lname.startswith("case") or "network" in lname or lname in {"mv_oberrhein", "example_simple"}:
        return "network_loader"
    if lname.startswith("run") or lname in {"calc_sc", "estimate", "diagnostic"}:
        return "analysis_executor"
    if lname.startswith("create_"):
        return "network_or_element_construction"
    if "control" in lname or lname in {"dfdata", "outputwriter", "constcontrol"}:
        return "controller_or_timeseries"
    if "plot" in category or "plot" in lname or "draw" in lname:
        return "plotting"
    return "other"


def compute_role_distribution(
    examples: Sequence["DemandExample"],
    cards: Dict[str, "FunctionCard"],
) -> Dict[str, float]:
    """Return share of each role across all function labels in *examples*.

    Counts are over (example, function) label occurrences, not unique functions,
    so high-frequency labels contribute more weight.
    """
    counts: Counter = Counter()
    for ex in examples:
        for fn in ex.functions & set(cards):
            counts[infer_function_role(fn, cards.get(fn))] += 1
    total = sum(counts.values())
    if not total:
        return {}
    return {role: count / total for role, count in counts.items()}


def compute_role_weights(
    train_examples: Sequence["DemandExample"],
    target_examples: Sequence["DemandExample"],
    cards: Dict[str, "FunctionCard"],
    *,
    max_weight: float = 10.0,
) -> Dict[str, float]:
    """Compute per-role importance weights for training.

    For each role r:
        raw_weight(r) = target_share(r) / train_share(r)

    Weights are clipped at *max_weight* (to avoid extreme values for roles
    that are nearly absent in training) and then normalised so the maximum
    weight equals 1.0.

    In pairwise training, a positive pair (query, function_r) receives the
    weight for role r; its associated negatives receive the same weight so
    the local positive-vs-negative ratio is preserved.

    Returns:
        Dict mapping role name -> weight in (0, 1].
    """
    train_dist = compute_role_distribution(train_examples, cards)
    target_dist = compute_role_distribution(target_examples, cards)

    all_roles = set(train_dist) | set(target_dist)
    raw: Dict[str, float] = {}
    for role in all_roles:
        train_share = train_dist.get(role, 1e-6)   # avoid div-by-zero
        target_share = target_dist.get(role, 0.0)
        raw[role] = min(target_share / train_share, max_weight)

    max_w = max(raw.values()) if raw else 1.0
    return {r: w / max_w for r, w in raw.items()}


PP_CALL_RE = re.compile(r"\bpp\.([A-Za-z_]\w*)\s*\(")
PN_CALL_RE = re.compile(r"\bpn\.([A-Za-z_]\w*)\s*\(")
SC_CALL_RE = re.compile(r"\b(?:sc|shortcircuit)\.([A-Za-z_]\w*)\s*\(")
IMPORT_FROM_RE = re.compile(r"\bfrom\s+pandapower(?:\.[\w.]+)?\s+import\s+([^\n]+)")


def extract_reference_functions(code: str, function_names: Set[str]) -> Set[str]:
    funcs = set(PP_CALL_RE.findall(code))
    funcs.update(PN_CALL_RE.findall(code))
    funcs.update(SC_CALL_RE.findall(code))

    for import_blob in IMPORT_FROM_RE.findall(code):
        for part in import_blob.split(","):
            name = part.strip().split(" as ", 1)[0].strip()
            if name in function_names:
                funcs.add(name)

    return {fn for fn in funcs if fn in function_names}


def load_benchmark_reference_examples(
    benchmark_path: Path,
    function_names: Set[str],
    *,
    max_items: Optional[int] = None,
) -> List[DemandExample]:
    with benchmark_path.open() as f:
        items = json.load(f)
    if max_items:
        items = items[:max_items]

    examples = []
    for idx, item in enumerate(items):
        funcs = extract_reference_functions(item.get("reference_code", ""), function_names)
        if not funcs:
            continue
        sid = str(item.get("id", idx))
        examples.append(
            DemandExample(
                sample_id=sid,
                base_id=sid,
                question=str(item.get("natural_language_query", "")).strip(),
                functions=funcs,
                attributes=set(),
                is_variant=False,
                source="benchmark_reference_eval_only",
            )
        )
    return examples


def load_knowledge_profile(profile_path: Optional[Path]) -> Optional[Dict]:
    if not profile_path:
        return None
    with profile_path.open() as f:
        return json.load(f)


def profile_knowledge_risk(function_name: str, profile: Optional[Dict]) -> Optional[float]:
    """Return model-specific risk in [0, 1], or None if no profile exists."""
    if not profile or function_name not in profile:
        return None
    entry = profile.get(function_name) or {}

    def layer_score(layer: str) -> Optional[float]:
        data = entry.get(layer)
        if not data:
            return None
        val = data.get("score")
        return float(val) if isinstance(val, (int, float)) else None

    weights = {"L0": 0.20, "L1": 0.25, "L2": 0.20, "L3": 0.35}
    total_w = 0.0
    mastery = 0.0
    for layer, weight in weights.items():
        score = layer_score(layer)
        if score is None:
            continue
        mastery += weight * max(0.0, min(1.0, score))
        total_w += weight
    if total_w == 0:
        return None
    return round(1.0 - mastery / total_w, 4)


def _parse_role_filter(raw: Optional[str]) -> Optional[Set[str]]:
    if not raw:
        return None
    roles = {part.strip() for part in raw.split(",") if part.strip()}
    return roles or None


def export_predictions(
    ranker,
    examples: Sequence[DemandExample],
    cards: Dict[str, FunctionCard],
    output_path: Path,
    *,
    top_k: int = 20,
    knowledge_profile: Optional[Dict] = None,
    allowed_roles: Optional[Set[str]] = None,
    cross_encoder_model: Optional[str] = None,
    cross_encoder_top_n: int = 50,
    ce_batch_size: int = 512,
    ce_fp16: bool = False,
    sentence_transformers_device: Optional[str] = None,
) -> None:
    """Export top-k demand candidates per benchmark item.

    Candidates are selected by **rank** (top_k by demand score), not by
    absolute threshold.  This ensures the exported count is always exactly
    top_k (minus role-filtered items) regardless of the score scale of the
    underlying ranker, making exports from different methods directly
    comparable.

    Args:
        top_k: number of candidates to export per item, selected by demand-score
               rank before reordering by intervention score.
        allowed_roles: optional role whitelist; functions outside this set are
                       skipped.  This is a semantic filter, not a score filter.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reranker = None
    if cross_encoder_model:
        reranker = _load_cross_encoder(cross_encoder_model, sentence_transformers_device, fp16=ce_fp16)

    # First-stage scores for all examples (CPU).
    all_first_stage = [(ex, ranker.score(ex.question)) for ex in examples]

    # Batch cross-encoder reranking across all examples at once.
    if reranker:
        print(f"  [CE export] batch-reranking {len(all_first_stage)} examples × "
              f"top_{cross_encoder_top_n} pairs (batch_size={ce_batch_size})")
        qs = [(ex.question, sc) for ex, sc in all_first_stage]
        merged = batch_rerank_with_cross_encoder(
            qs, cards, reranker=reranker,
            top_n=cross_encoder_top_n,
            batch_size=ce_batch_size,
        )
        all_first_stage = [(ex, ms) for (ex, _), ms in zip(all_first_stage, merged)]

    rows = []
    for ex, demand_scores in all_first_stage:
        ranked = []
        # Sort by demand score descending; collect top_k by rank.
        # No absolute-score threshold is applied so that score scales are
        # irrelevant and all methods export the same number of candidates.
        sorted_names = sorted(demand_scores.items(), key=lambda x: (-x[1], x[0]))
        for name, demand in sorted_names:
            role = infer_function_role(name, cards.get(name))
            if allowed_roles and role not in allowed_roles:
                continue
            risk = profile_knowledge_risk(name, knowledge_profile)
            intervention_score = demand if risk is None else demand * risk
            ranked.append({
                "function": name,
                "demand_score": round(float(demand), 6),
                "knowledge_risk": risk,
                "intervention_score": round(float(intervention_score), 6),
                "role": role,
                "is_reference_label": name in ex.functions,
            })
            if len(ranked) >= top_k:
                break

        intervention_ranked = sorted(
            ranked,
            key=lambda x: (-x["intervention_score"], x["function"]),
        )
        rows.append({
            "id": ex.sample_id,
            "source": ex.source,
            "question": ex.question,
            "reference_functions": sorted(ex.functions),
            "top_demand": ranked,
            "top_intervention": intervention_ranked,
        })

    with output_path.open("w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def summarize_examples(examples: Sequence[DemandExample]) -> Dict:
    freq = Counter()
    n_variant = 0
    group_ids = set()
    for ex in examples:
        freq.update(ex.functions)
        n_variant += int(ex.is_variant)
        group_ids.add(ex.base_id)
    label_counts = [len(ex.functions) for ex in examples]
    return {
        "n_examples": len(examples),
        "n_groups": len(group_ids),
        "n_variants": n_variant,
        "n_non_variants": len(examples) - n_variant,
        "n_unique_functions": len(freq),
        "avg_labels_per_example": round(float(np.mean(label_counts)), 3) if label_counts else 0.0,
        "top_functions": dict(freq.most_common(30)),
    }


def select_examples(
    examples: Sequence[DemandExample],
    *,
    variants: Optional[bool] = None,
) -> List[DemandExample]:
    if variants is None:
        return list(examples)
    return [ex for ex in examples if ex.is_variant == variants]


def build_ranker(args):
    if args.model == "zero_shot_tfidf":
        return ZeroShotTfidfDemandRanker()
    if args.model == "pairwise_tfidf_logreg":
        return PairwiseTfidfLogRegDemandRanker(
            negative_per_positive=args.negative_per_positive,
            max_train_pairs=args.max_train_pairs,
            max_positive_per_function=args.max_positive_per_function,
            seed=args.seed,
        )
    if args.model == "hybrid_tfidf":
        return HybridTfidfDemandRanker(
            alpha=args.hybrid_alpha,
            negative_per_positive=args.negative_per_positive,
            max_train_pairs=args.max_train_pairs,
            max_positive_per_function=args.max_positive_per_function,
            seed=args.seed,
        )
    if args.model == "zero_shot_sbert":
        return SbertDemandRanker(
            args.sbert_model,
            batch_size=args.sbert_batch_size,
            device=args.sentence_transformers_device,
        )
    raise ValueError(f"Unknown model: {args.model}")


def train_or_fit_ranker(ranker, args, train_examples, cards, role_weights=None):
    if isinstance(ranker, (PairwiseTfidfLogRegDemandRanker, HybridTfidfDemandRanker)):
        return ranker.fit(train_examples, cards, role_weights=role_weights)
    return ranker.fit(cards)


def tune_hybrid_alpha(
    ranker: HybridTfidfDemandRanker,
    dev_examples: Sequence[DemandExample],
    cards: Dict[str, FunctionCard],
    *,
    candidate_alphas: Sequence[float],
    target_k: int = 10,
) -> Dict:
    cached_rows = []
    for ex in dev_examples:
        truth = ex.functions & set(cards)
        if not truth:
            continue
        cached_rows.append((
            truth,
            _minmax_normalize(ranker.zero_shot.score(ex.question)),
            _minmax_normalize(ranker.pairwise.score(ex.question)),
        ))

    best = {"alpha": ranker.alpha, "recall": -1.0}
    curve = []
    for alpha in candidate_alphas:
        total_recall = 0.0
        total_hit = 0.0
        for truth, zs, pw in cached_rows:
            names = set(zs) | set(pw)
            scores = {
                name: (1.0 - alpha) * zs.get(name, 0.0) + alpha * pw.get(name, 0.0)
                for name in names
            }
            pred = set(rank_names(scores, target_k))
            overlap = pred & truth
            total_recall += len(overlap) / len(truth)
            total_hit += 1.0 if overlap else 0.0
        denom = max(len(cached_rows), 1)
        recall = round(total_recall / denom, 4)
        hit_rate = round(total_hit / denom, 4)
        row = {"alpha": alpha, f"recall@{target_k}": recall, f"hit@{target_k}": hit_rate}
        curve.append(row)
        if recall > best["recall"]:
            best = {"alpha": alpha, "recall": recall, "hit_rate": hit_rate}
    ranker.alpha = best["alpha"]
    return {"selected": best, "curve": curve}


def save_model(path: Path, ranker, args, cards: Dict[str, FunctionCard], metadata: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.model == "zero_shot_sbert":
        raise ValueError("Saving SBERT rankers is not supported by this lightweight pickle path.")
    payload = {
        "model_type": args.model,
        "ranker": ranker,
        "function_names": sorted(cards),
        "metadata": metadata,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def resolve_run_name(args) -> str:
    if args.run_name:
        return args.run_name
    if args.cross_encoder_model:
        return f"{args.model}_cross_encoder"
    return args.model


def build_compact_summary(results: Dict) -> Dict:
    summary = {
        "run_name": results.get("run_name"),
        "model": results.get("config", {}).get("model"),
        "cross_encoder_model": results.get("config", {}).get("cross_encoder_model"),
        "hybrid_alpha_selected": None,
        "splits": {},
    }
    alpha_tuning = results.get("alpha_tuning")
    if alpha_tuning and alpha_tuning.get("selected"):
        summary["hybrid_alpha_selected"] = alpha_tuning["selected"].get("alpha")

    for split_name, split_metrics in (results.get("metrics") or {}).items():
        top_k = split_metrics.get("top_k") or {}
        split_summary = {}
        for k in ("1", "3", "5", "10", "20"):
            if k not in top_k:
                continue
            split_summary[f"recall@{k}"] = top_k[k].get("recall")
            split_summary[f"hit@{k}"] = top_k[k].get("hit_rate")
        summary["splits"][split_name] = split_summary
    return summary


def write_standard_outputs(
    output_dir: Path,
    run_name: str,
    results: Dict,
    *,
    exported_predictions_path: Optional[Path] = None,
    saved_model_path: Optional[Path] = None,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    config_path = output_dir / "run_config.json"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    with metrics_path.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    with config_path.open("w") as f:
        json.dump(results.get("config", {}), f, indent=2, ensure_ascii=False, default=str)

    summary = build_compact_summary(results)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    manifest = {
        "run_name": run_name,
        "artifacts": {
            "metrics": str(metrics_path),
            "run_config": str(config_path),
            "summary": str(summary_path),
            "saved_model": str(saved_model_path) if saved_model_path else None,
            "prediction_export": str(exported_predictions_path) if exported_predictions_path else None,
        },
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest["artifacts"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate task demand predictors.")
    parser.add_argument("--augmented-path", type=Path, default=DEFAULT_AUGMENTED_PATH)
    parser.add_argument("--docs-path", type=Path, default=DEFAULT_DOCS_PATH)
    parser.add_argument("--benchmark-path", type=Path, default=None,
                        help="Optional benchmark JSON for eval-only reference-demand diagnostics.")
    parser.add_argument("--reweight-benchmark-path", type=Path, default=None,
                        help="Optional separate benchmark JSON used ONLY as the unlabelled "
                             "target distribution for role-frequency reweighting; if unset, "
                             "falls back to --benchmark-path. Use this to evaluate held-out "
                             "transfer (reweight on one split, eval on another).")
    parser.add_argument("--knowledge-profile", type=Path, default=None,
                        help="Optional model knowledge_profile.json for demand*risk intervention export.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", choices=["zero_shot_tfidf", "pairwise_tfidf_logreg", "hybrid_tfidf", "zero_shot_sbert"],
                        default="pairwise_tfidf_logreg")
    parser.add_argument("--include-variants", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--negative-per-positive", type=int, default=8)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-positive-per-function", type=int, default=None,
                        help="Optional cap for positives per function to reduce boilerplate-function skew.")
    parser.add_argument("--hybrid-alpha", type=float, default=0.5,
                        help="Weight of supervised pairwise branch in hybrid_tfidf; 0 means zero-shot only.")
    parser.add_argument("--auto-hybrid-alpha", action="store_true",
                        help="Tune hybrid alpha on augmented dev recall@target-k.")
    parser.add_argument("--hybrid-alpha-target-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--sbert-batch-size", type=int, default=64)
    parser.add_argument("--cross-encoder-model", default=None,
                        help="Optional CrossEncoder reranker model for top-N candidates.")
    parser.add_argument("--cross-encoder-top-n", type=int, default=50)
    parser.add_argument("--ce-batch-size", type=int, default=512,
                        help="Pairs per GPU forward pass for CrossEncoder. "
                             "Larger values improve GPU utilisation; reduce if OOM. "
                             "Default: 512 (works well on A100/V100).")
    parser.add_argument("--ce-fp16", action="store_true", default=False,
                        help="Run CrossEncoder in FP16 for ~2x faster GPU inference.")
    parser.add_argument("--sentence-transformers-device", default=None,
                        help="Optional device override for SentenceTransformer/CrossEncoder, e.g. cpu or cuda.")
    parser.add_argument("--save-model", type=Path, default=None)
    parser.add_argument("--benchmark-max-items", type=int, default=None)
    parser.add_argument("--export-predictions", type=Path, default=None,
                        help="Optional JSON path for benchmark prediction/intervention candidates.")
    parser.add_argument("--run-name", default=None,
                        help="Optional logical run name used for normalized output files.")
    parser.add_argument("--prediction-top-k", type=int, default=10,
                        help="Number of candidates to export per item, selected by demand-score rank.")
    parser.add_argument("--prediction-roles", default=None,
                        help="Optional comma-separated role whitelist for exported candidates.")
    parser.add_argument("--role-reweight", action="store_true", default=False,
                        help="Enable role-aware sample reweighting during pairwise/hybrid training. "
                             "Requires --benchmark-path. Computes target role distribution from "
                             "benchmark reference labels and reweights training pairs so that "
                             "roles underrepresented in training but common in the target domain "
                             "receive higher gradient.")
    parser.add_argument("--max-role-weight", type=float, default=3.0,
                        help="Maximum per-role weight multiplier for role-aware reweighting "
                             "(before normalisation). Lower values reduce saturation risk. "
                             "Default: 3.0 (was 10.0 in earlier runs; 10.0 caused analysis_executor "
                             "saturation and pairwise pathology).")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    cards = load_function_cards(args.docs_path)
    function_names = set(cards)
    print(f"Loaded {len(cards)} function cards from {args.docs_path}")

    examples = load_augmented_examples(
        args.augmented_path,
        function_names,
        include_variants=args.include_variants,
    )
    train, dev, test = split_by_group(examples, seed=args.seed)
    print("Augmented data summary:")
    print(json.dumps({
        "all": summarize_examples(examples),
        "train": summarize_examples(train),
        "dev": summarize_examples(dev),
        "test": summarize_examples(test),
    }, indent=2, ensure_ascii=False)[:6000])

    ranker = build_ranker(args)

    # ── Role-aware reweighting ───────────────────────────────────────────────
    role_weights = None
    if args.role_reweight:
        reweight_path = args.reweight_benchmark_path or args.benchmark_path
        if not reweight_path:
            print("WARNING: --role-reweight requires --benchmark-path or --reweight-benchmark-path; skipping reweighting.")
        else:
            bench_for_weights = load_benchmark_reference_examples(
                reweight_path, function_names, max_items=None
            )
            role_weights = compute_role_weights(train, bench_for_weights, cards,
                                                max_weight=args.max_role_weight)
            print("Role-aware reweighting enabled.")
            print(f"  Reweight source    : {reweight_path}")
            print(f"  Training role dist : {compute_role_distribution(train, cards)}")
            print(f"  Target role dist   : {compute_role_distribution(bench_for_weights, cards)}")
            print(f"  Computed weights   : {role_weights}")

    train_or_fit_ranker(ranker, args, train, cards, role_weights=role_weights)
    alpha_tuning = None
    if isinstance(ranker, HybridTfidfDemandRanker) and args.auto_hybrid_alpha:
        alpha_tuning = tune_hybrid_alpha(
            ranker,
            dev,
            cards,
            candidate_alphas=[round(x * 0.1, 1) for x in range(0, 11)],
            target_k=args.hybrid_alpha_target_k,
        )
        print(f"  Auto-selected hybrid alpha={ranker.alpha} "
              f"using dev recall@{args.hybrid_alpha_target_k}")

    results = {
        "run_name": resolve_run_name(args),
        "config": vars(args) | {
            "augmented_path": str(args.augmented_path),
            "docs_path": str(args.docs_path),
            "benchmark_path": str(args.benchmark_path) if args.benchmark_path else None,
            "output_dir": str(args.output_dir),
            "save_model": str(args.save_model) if args.save_model else None,
            "export_predictions": str(args.export_predictions) if args.export_predictions else None,
        },
        "data_summary": {
            "all": summarize_examples(examples),
            "train": summarize_examples(train),
            "dev": summarize_examples(dev),
            "test": summarize_examples(test),
        },
        "alpha_tuning": alpha_tuning,
        "role_weights": role_weights,
        "metrics": {},
    }

    # Shared kwargs forwarded to every evaluate_ranker / export_predictions call.
    eval_ce_kwargs = dict(
        cross_encoder_model=args.cross_encoder_model,
        cross_encoder_top_n=args.cross_encoder_top_n,
        ce_batch_size=args.ce_batch_size,
        ce_fp16=args.ce_fp16,
        sentence_transformers_device=args.sentence_transformers_device,
    )

    print("\nEvaluating on augmented dev split...")
    results["metrics"]["augmented_dev"] = evaluate_ranker(
        ranker, dev, cards=cards, top_ks=args.top_k, **eval_ce_kwargs)
    dev_original = select_examples(dev, variants=False)
    dev_variant = select_examples(dev, variants=True)
    if dev_original:
        print("Evaluating on augmented dev original-only slice...")
        results["metrics"]["augmented_dev_original_only"] = evaluate_ranker(
            ranker, dev_original, cards=cards, top_ks=args.top_k, **eval_ce_kwargs)
    if dev_variant:
        print("Evaluating on augmented dev variant-only slice...")
        results["metrics"]["augmented_dev_variant_only"] = evaluate_ranker(
            ranker, dev_variant, cards=cards, top_ks=args.top_k, **eval_ce_kwargs)

    print("Evaluating on augmented test split...")
    results["metrics"]["augmented_test"] = evaluate_ranker(
        ranker, test, cards=cards, top_ks=args.top_k, **eval_ce_kwargs)
    test_original = select_examples(test, variants=False)
    test_variant = select_examples(test, variants=True)
    if test_original:
        print("Evaluating on augmented test original-only slice...")
        results["metrics"]["augmented_test_original_only"] = evaluate_ranker(
            ranker, test_original, cards=cards, top_ks=args.top_k, **eval_ce_kwargs)
    if test_variant:
        print("Evaluating on augmented test variant-only slice...")
        results["metrics"]["augmented_test_variant_only"] = evaluate_ranker(
            ranker, test_variant, cards=cards, top_ks=args.top_k, **eval_ce_kwargs)

    if args.benchmark_path:
        print("\nEvaluating on benchmark reference labels (eval-only, not trained)...")
        bench_examples = load_benchmark_reference_examples(
            args.benchmark_path, function_names, max_items=args.benchmark_max_items)
        results["data_summary"]["benchmark_reference_eval_only"] = summarize_examples(bench_examples)
        results["metrics"]["benchmark_reference_eval_only"] = evaluate_ranker(
            ranker, bench_examples, cards=cards, top_ks=args.top_k, **eval_ce_kwargs)
        if args.export_predictions:
            profile = load_knowledge_profile(args.knowledge_profile)
            export_predictions(
                ranker,
                bench_examples,
                cards,
                args.export_predictions,
                top_k=args.prediction_top_k,
                knowledge_profile=profile,
                allowed_roles=_parse_role_filter(args.prediction_roles),
                cross_encoder_model=args.cross_encoder_model,
                cross_encoder_top_n=args.cross_encoder_top_n,
                ce_batch_size=args.ce_batch_size,
                ce_fp16=args.ce_fp16,
                sentence_transformers_device=args.sentence_transformers_device,
            )
            print(f"Exported prediction candidates to {args.export_predictions}")

    if args.save_model:
        save_model(args.save_model, ranker, args, cards, metadata=results["data_summary"])
        print(f"Saved model to {args.save_model}")

    artifacts = write_standard_outputs(
        args.output_dir,
        results["run_name"],
        results,
        exported_predictions_path=args.export_predictions,
        saved_model_path=args.save_model,
    )
    print(f"\nSaved normalized outputs under {args.output_dir}")
    print(json.dumps(artifacts, indent=2, ensure_ascii=False))

    print("\nMetrics summary:")
    compact = {
        split: metrics.get("top_k", {})
        for split, metrics in results["metrics"].items()
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
