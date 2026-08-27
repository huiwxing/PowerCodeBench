# --------------------------------------------------------------------------
# Repository copy of audit/records.py from the frozen experimental pipeline.
# Its inputs are raw per-item run trees too large to ship here, so it does not
# run from this checkout; see ARTIFACT_INDEX.md for the outputs it produced.
# --------------------------------------------------------------------------
"""Streaming reader + frozen strata constants for the E1 validity audit.

Why a custom reader instead of ``ijson``?  The result files embed bare
``NaN`` / ``Infinity`` tokens (produced by ``json.dump`` with ``allow_nan``
on non-matched items' ``match_detail``).  Neither ``ijson`` backend (yajl2_c
nor pure-python) accepts those tokens, whereas the standard-library
``json`` scanner does.  So we stream the file in chunks and peel one array
element at a time with ``json.JSONDecoder.raw_decode``, which:

  * never materialises the whole file (memory bounded by one record + one
    read chunk -- honours the 4 GiB login-node cgroup, E1 doc s7 "R-OOM");
  * tolerates NaN/Infinity exactly like ``json.loads`` does.

``item_results`` is the last top-level key in every file (order observed:
summary, config, round_snapshots, item_results) and the header before it is
small (~130 KB), so locating the array start is cheap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional

# --------------------------------------------------------------------------
# Paths.  audit/ lives at the project root, so parents[1] is the project root.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_ROOT = PROJECT_ROOT / "probe_eval_results" / "comparison"
API_ROOT = PROJECT_ROOT / "probe_eval_results" / "api_comparison_2000"

# --------------------------------------------------------------------------
# Frozen panel strata (E1 doc s4.1 tier table).  Model dir names as on disk.
# --------------------------------------------------------------------------
TIER_MODELS: Dict[str, List[str]] = {
    "T1": [  # sub-7B
        "Qwen_Qwen2.5-Coder-0.5B-Instruct",
        "Qwen_Qwen2.5-Coder-1.5B-Instruct",
    ],
    "T2": [  # 7B-32B
        "Qwen_Qwen2.5-Coder-7B-Instruct",
        "meta-llama_Llama-3.1-8B-Instruct",
        "Qwen_Qwen2.5-Coder-14B-Instruct",
        "Qwen_Qwen2.5-Coder-32B-Instruct",
    ],
    "T3": [  # 70B-120B
        "meta-llama_Llama-3.1-70B-Instruct",
        "openai_gpt-oss-120b",
    ],
    "T4": [  # 405B+
        "meta-llama_Llama-3.1-405B-Instruct",
        "Qwen_Qwen3-Coder-480B-A35B-Instruct",
        "Qwen_Qwen3-Coder-Next",
    ],
}

# API panel (E1 doc s4.2 "T5-API" layer), served from api_comparison_2000/.
API_MODELS: List[str] = [
    "claude-haiku-4-5",
    "deepseek-v4-flash",
    "gemini-2.5-flash",
    "gpt-5.4-mini",
]

# Condition file-name suffixes we sample from (E1 doc s2.1 / s4.2).
CONDITION_HEADLINE = "C_FDRS"   # complete method endpoint  -> main frame P
CONDITION_CONTRAST = "A_FX"     # plain baseline + plain-fix -> contrast frame S


def cond_path(model_dir: str, condition: str, *, api: bool = False) -> Path:
    """Path to a model's per-item result file for one condition.

    ``condition`` is the suffix after ``cond`` in the file name, e.g.
    ``"C_FDRS"`` -> ``benchmark_results_condC_FDRS.json``.
    """
    root = API_ROOT if api else COMPARISON_ROOT
    return root / model_dir / f"benchmark_results_cond{condition}.json"


# --------------------------------------------------------------------------
# Streaming record reader.
# --------------------------------------------------------------------------
_ARRAY_MARKER = '"item_results"'
_DECODER = json.JSONDecoder()


def iter_item_records(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict]:
    """Yield each element of the ``item_results`` array, one at a time.

    Memory is bounded by the file header (~130 KB, read once while locating
    the array) plus at most one record and one read chunk.  Never loads the
    whole file.  Tolerates NaN/Infinity via the stdlib scanner.
    """
    with open(path, "r", encoding="utf-8") as fh:
        buf = ""
        # -- Phase 1: locate the opening '[' of the item_results array. --
        while True:
            idx = buf.find(_ARRAY_MARKER)
            if idx != -1:
                after_key = buf[idx + len(_ARRAY_MARKER):]
                colon = after_key.find(":")
                if colon != -1:
                    tail = after_key[colon + 1:]
                    lb = tail.find("[")
                    if lb != -1:
                        buf = tail[lb + 1:]
                        break
            chunk = fh.read(chunk_size)
            if not chunk:
                raise ValueError(f"item_results array not found in {path}")
            buf += chunk

        # -- Phase 2: peel one object per iteration. --
        while True:
            buf = buf.lstrip()
            while buf[:1] == ",":
                buf = buf[1:].lstrip()
            if buf[:1] == "]":
                return
            if not buf:
                chunk = fh.read(chunk_size)
                if not chunk:
                    return
                buf += chunk
                continue
            try:
                obj, end = _DECODER.raw_decode(buf)
            except json.JSONDecodeError:
                chunk = fh.read(chunk_size)
                if not chunk:
                    # Buffer holds an incomplete tail at EOF: nothing valid left.
                    return
                buf += chunk
                continue
            yield obj
            buf = buf[end:]


def iter_matched_records(path: Path) -> Iterator[dict]:
    """Yield only records with ``match is True``."""
    for rec in iter_item_records(path):
        if rec.get("match") is True:
            yield rec


def fetch_record(
    path: Path,
    *,
    bench_index: Optional[int] = None,
    item_id: Optional[str] = None,
) -> Optional[dict]:
    """Stream ``path`` and return the first record matching the given
    pointer keys, or ``None`` if not found.  At least one of ``bench_index``
    / ``item_id`` must be given; both are checked when both are supplied.
    """
    if bench_index is None and item_id is None:
        raise ValueError("fetch_record needs bench_index and/or item_id")
    for rec in iter_item_records(path):
        if bench_index is not None and rec.get("bench_index") != bench_index:
            continue
        if item_id is not None and rec.get("item_id") != item_id:
            continue
        return rec
    return None
