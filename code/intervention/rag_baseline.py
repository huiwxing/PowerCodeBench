# --------------------------------------------------------------------------
# Repository copy of knowledge_injection/rag_baseline.py, unmodified apart
# from this header. Runs CPU-only against the archived corpus/spec files in
# this repository. See code/README.md for the module map.
# --------------------------------------------------------------------------
"""Vanilla BM25 retrieval over raw API documentation.

This module supplies a deliberately naive lexical retrieval baseline for
proactive condition R (RAG-only). It is the comparison point for the question
"how does the proposed pipeline compare against an off-the-shelf RAG that
indexes the same out-of-the-box documentation?"

Design choices made for an honest baseline:
  * BM25 only, no embedding/cross-encoder reranking. Adding learned components
    here would partly bake in techniques the proposed pipeline already uses
    (TF-IDF + supervised reranking) and dilute the comparison.
  * Indexes the raw API spec JSON (e.g. dataset/pandapower_docs.json), not the
    derived library_knowledge artifact. The derived contracts are part of our
    pipeline's contribution; running BM25 over them would conflate retrieval
    quality with our derived facts.
  * The score(query) interface mirrors the supervised demand rankers so that
    downstream code paths can swap in this predictor without other changes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _tokenize(text: str) -> List[str]:
    """Lower-cased word/identifier tokens. Preserves snake_case as a single token."""
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _function_text(entry: Dict) -> str:
    """Concatenate the searchable fields of a documented function entry.

    The raw spec contains: name, full_path, signature, description, parameters,
    return_info, examples. Examples are excluded because they are typically
    noisy and biased toward construction patterns; including them would inflate
    BM25 hits on construction queries unrealistically.
    """
    parts: List[str] = []
    parts.append(str(entry.get("name", "")))
    parts.append(str(entry.get("full_path", "")))
    parts.append(str(entry.get("category", "")))
    parts.append(str(entry.get("signature", "")))
    parts.append(str(entry.get("description", "")))
    for p in entry.get("parameters") or []:
        parts.append(str(p.get("name", "")))
        parts.append(str(p.get("description", "")))
    for r in entry.get("return_info") or []:
        parts.append(str(r.get("description", "")))
    return " ".join(parts)


class BM25DemandPredictor:
    """Vanilla BM25 retriever with the demand-predictor interface.

    Construct once per evaluation run, then call score(query) per item.

    Args:
        docs_path: path to the raw API spec JSON (must contain a top-level
            "functions" list with name/description/signature/parameters fields).
        function_filter: optional iterable of function names to restrict the
            scoring vocabulary (rest get score 0.0). When omitted, all
            documented functions are indexed. Useful when downstream rendering
            only knows certain function names (e.g. those present in the snippet
            corpus).
    """

    def __init__(
        self,
        docs_path: str,
        function_filter: Optional[Sequence[str]] = None,
    ):
        from rank_bm25 import BM25Okapi

        path = Path(docs_path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 docs source not found: {docs_path}")
        with path.open() as f:
            raw = json.load(f)
        self.library_name = raw.get("library_name") or raw.get("library") or ""

        functions = raw.get("functions") or []
        if not functions:
            raise ValueError(f"No 'functions' entries in {docs_path}")

        allowed = set(function_filter) if function_filter else None

        names: List[str] = []
        corpus_tokens: List[List[str]] = []
        for entry in functions:
            name = entry.get("name") or ""
            if not name:
                continue
            if allowed is not None and name not in allowed:
                continue
            names.append(name)
            corpus_tokens.append(_tokenize(_function_text(entry)))

        if not names:
            raise ValueError("BM25 corpus is empty after filtering")

        self.function_names: List[str] = names
        self._bm25 = BM25Okapi(corpus_tokens)
        self._docs_path = str(path)

    def __repr__(self) -> str:
        return (
            f"BM25DemandPredictor(library={self.library_name!r}, "
            f"n_functions={len(self.function_names)}, docs={self._docs_path})"
        )

    def score(self, query: str) -> Dict[str, float]:
        """Return {function_name: score in [0, 1]} via min-max normalisation.

        Min-max keeps the dynamic range comparable to other rankers used in
        this codebase, so downstream rank-based selection (top-k) is the
        same. Functions with raw BM25 score 0 receive normalised score 0.
        """
        q_tokens = _tokenize(query)
        if not q_tokens:
            return {name: 0.0 for name in self.function_names}
        raw_scores = self._bm25.get_scores(q_tokens)
        lo = float(raw_scores.min())
        hi = float(raw_scores.max())
        if hi <= lo:
            return {name: 0.0 for name in self.function_names}
        scale = hi - lo
        return {
            name: float((s - lo) / scale)
            for name, s in zip(self.function_names, raw_scores)
        }
