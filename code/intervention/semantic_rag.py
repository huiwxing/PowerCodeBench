# --------------------------------------------------------------------------
# Repository copy of knowledge_injection/semantic_rag.py, with imports and
# data-path constants adapted to this repository's layout. Runs CPU-only
# against the archived corpus/spec files in this repository. See
# code/README.md for the module map.
# --------------------------------------------------------------------------
"""Vanilla dense (SBERT) retrieval over raw API documentation.

Dense semantic counterpart to ``rag_baseline.BM25DemandPredictor``. It supplies
the off-the-shelf embedding-retrieval baseline for proactive conditions Rsem /
RsemB: documentation retrieved by semantic similarity, without model-specific
boundary profiles.

Design choices, kept parallel to the BM25 baseline for a fair comparison:
  * Bi-encoder cosine only, no cross-encoder reranking. Adding a reranker would
    bake in supervised machinery the proposed pipeline already uses and dilute
    the vanilla dense-RAG comparison.
  * Indexes the same raw API spec JSON as condition R, flattened with the same
    ``_function_text`` (examples excluded), at the same per-API-entry granularity.
  * Reuses the project-default general-domain model all-MiniLM-L6-v2 (the same
    model task_demand's zero_shot_sbert uses). Being out-of-domain for
    power/API terms is the honest vanilla setting, not a defect.
  * The score(query) interface mirrors ``BM25DemandPredictor.score`` (cosine +
    min-max normalisation) so downstream code paths swap predictors without any
    other changes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from intervention.rag_baseline import _function_text

DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SBERTDemandPredictor:
    """Vanilla dense bi-encoder retriever with the demand-predictor interface.

    Construct once per evaluation run (embeds the whole corpus up front), then
    call score(query) per item.

    Args:
        docs_path: path to the raw API spec JSON (must contain a top-level
            "functions" list with name/description/signature/parameters fields).
        function_filter: optional iterable of function names to restrict the
            scoring vocabulary (rest get score 0.0). When omitted, all
            documented functions are indexed. Useful when downstream rendering
            only knows certain function names (e.g. those present in the snippet
            corpus).
        model_name: sentence-transformers model id (default: project-internal
            all-MiniLM-L6-v2, shared with task_demand's zero_shot_sbert).
        device: optional device override for the encoder (CPU is sufficient).
        batch_size: card-encoding batch size.
    """

    def __init__(
        self,
        docs_path: str,
        function_filter: Optional[Sequence[str]] = None,
        model_name: str = DEFAULT_SBERT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 64,
    ):
        from sentence_transformers import SentenceTransformer

        path = Path(docs_path)
        if not path.exists():
            raise FileNotFoundError(f"SBERT docs source not found: {docs_path}")
        with path.open() as f:
            raw = json.load(f)
        self.library_name = raw.get("library_name") or raw.get("library") or ""

        functions = raw.get("functions") or []
        if not functions:
            raise ValueError(f"No 'functions' entries in {docs_path}")

        allowed = set(function_filter) if function_filter else None

        names: List[str] = []
        texts: List[str] = []
        for entry in functions:
            name = entry.get("name") or ""
            if not name:
                continue
            if allowed is not None and name not in allowed:
                continue
            names.append(name)
            texts.append(_function_text(entry))

        if not names:
            raise ValueError("SBERT corpus is empty after filtering")

        model_kwargs = {"trust_remote_code": True}
        if device:
            model_kwargs["device"] = device
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, **model_kwargs)
        self.function_names: List[str] = names
        self._card_embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._docs_path = str(path)

    def __repr__(self) -> str:
        return (
            f"SBERTDemandPredictor(library={self.library_name!r}, "
            f"n_functions={len(self.function_names)}, model={self.model_name!r}, "
            f"docs={self._docs_path})"
        )

    def score(self, query: str) -> Dict[str, float]:
        """Return {function_name: score in [0, 1]} via min-max normalisation.

        Similarity is cosine over L2-normalised embeddings (== normalised dot
        product, the same measure as task_demand's ZeroShotSBERT). Min-max keeps
        the dynamic range comparable to the BM25 baseline, so downstream
        rank-based selection (top-k) is unchanged. An empty query yields all
        zeros, matching BM25DemandPredictor.
        """
        if not query:
            return {name: 0.0 for name in self.function_names}
        q = self._model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        raw_scores = np.dot(self._card_embeddings, q)
        lo = float(raw_scores.min())
        hi = float(raw_scores.max())
        if hi <= lo:
            return {name: 0.0 for name in self.function_names}
        scale = hi - lo
        return {
            name: float((s - lo) / scale)
            for name, s in zip(self.function_names, raw_scores)
        }
