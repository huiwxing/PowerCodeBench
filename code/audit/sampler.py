# --------------------------------------------------------------------------
# Repository copy of audit/sampler.py from the frozen experimental pipeline.
# Its inputs are raw per-item run trees too large to ship here, so it does not
# run from this checkout; see ARTIFACT_INDEX.md for the outputs it produced.
# --------------------------------------------------------------------------
"""T3 -- frozen stratified sampling-frame drawer for the E1 validity audit.

Streams the per-item result files (never ``json.load`` on a whole file --
see ``audit.records``), builds the matched-record population for each
(condition, tier[, model]) stratum, and draws the frozen sample defined in
E1 doc s4.2:

  main frame P  (condC_FDRS, deployment accept-set)
      T4=120  T3=110  T2=110  T1=60                             (open-source)
  API layer T5 (condC_FDRS)   4 API models x 15 = 60
  contrast S    (condA_FX)    T2=50  T3=50  T4=50 = 150
  pilot         (independent, 20)  drawn first and *excluded* from P/S/T5

Within each tier the budget is split across task families in proportion to
their matched availability, with a floor of >=2 per non-empty (tier, family)
cell and largest-remainder rounding; leftover from thin/empty families
backfills the densest families in the same tier (E1 doc s4.2).

Output: a manifest of lightweight pointers (model, condition, bench_index,
item_id + strata labels) -- NOT the bulky record bodies.  The sandbox/checker
re-fetch each record on demand via ``records.fetch_record``.

Everything is deterministic given ``SEED``: pools are read in file/tier order,
and every draw uses a Random seeded by a stable per-cell string, so the
manifest regenerates bit-for-bit.

Run:  python -m audit.sampler            (writes audit/e1_sample_manifest.json)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from audit import records
from audit.records import (API_MODELS, CONDITION_CONTRAST, CONDITION_HEADLINE,
                           PROJECT_ROOT, TIER_MODELS)

# --------------------------------------------------------------------------
# Frozen parameters (E1 doc s4.2; freeze date 2026-07-23).
# --------------------------------------------------------------------------
SEED = 20260723

MAIN_P_TIER_TARGETS = {"T4": 120, "T3": 110, "T2": 110, "T1": 60}   # condC_FDRS
CONTRAST_S_TIER_TARGETS = {"T2": 50, "T3": 50, "T4": 50}            # condA_FX
API_PER_MODEL = 15                                                  # condC_FDRS
PILOT_TOTAL = 20

MAIN_MIN_FLOOR = 2      # min per non-empty (tier, family) cell, frames P and S
API_MIN_FLOOR = 1       # smaller n per API model

# pilot allocation across (tier, condition) cells (sums to 20)
PILOT_CELLS = {
    ("T1", CONDITION_HEADLINE): 2,
    ("T2", CONDITION_HEADLINE): 3,
    ("T3", CONDITION_HEADLINE): 3,
    ("T4", CONDITION_HEADLINE): 3,
    ("T2", CONDITION_CONTRAST): 3,
    ("T3", CONDITION_CONTRAST): 3,
    ("T4", CONDITION_CONTRAST): 3,
}

_POINTER_FIELDS = ("bench_index", "item_id", "task", "difficulty_level",
                   "network", "query_type", "n_modifications")


# --------------------------------------------------------------------------
# Pool building (streaming).
# --------------------------------------------------------------------------
def _pointer(rec: dict, *, tier: str, model: str, condition: str) -> dict:
    p = {"tier": tier, "model": model, "condition": condition}
    for f in _POINTER_FIELDS:
        p[f] = rec.get(f)
    return p


def build_pools(verbose: bool = True):
    """Return (pools, api_pool, pop_counts).

    pools[condition][tier]  -> list[pointer]     (open-source, pooled by tier)
    api_pool[model]         -> list[pointer]     (condC_FDRS, per API model)
    pop_counts[condition][tier or 'API'][model] -> n_matched
    """
    pools: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    api_pool: Dict[str, List[dict]] = {}
    pop_counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))

    # open-source, both conditions
    for condition, tiers in (
        (CONDITION_HEADLINE, ("T1", "T2", "T3", "T4")),
        (CONDITION_CONTRAST, ("T2", "T3", "T4")),
    ):
        for tier in tiers:
            for model in TIER_MODELS[tier]:
                path = records.cond_path(model, condition)
                if not path.exists():
                    pop_counts[condition][tier][model] = 0
                    if verbose:
                        print(f"  [skip] {condition} {tier} {model}: file absent")
                    continue
                n = 0
                for rec in records.iter_matched_records(path):
                    pools[condition][tier].append(
                        _pointer(rec, tier=tier, model=model, condition=condition))
                    n += 1
                pop_counts[condition][tier][model] = n
                if verbose:
                    print(f"  [pool] {condition} {tier} {model}: {n} matched")

    # API layer, headline condition
    for model in API_MODELS:
        path = records.cond_path(model, CONDITION_HEADLINE, api=True)
        pts = []
        if path.exists():
            for rec in records.iter_matched_records(path):
                pts.append(_pointer(rec, tier="API", model=model,
                                    condition=CONDITION_HEADLINE))
        api_pool[model] = pts
        pop_counts[CONDITION_HEADLINE]["API"][model] = len(pts)
        if verbose:
            print(f"  [pool] {CONDITION_HEADLINE} API {model}: {len(pts)} matched")

    return pools, api_pool, pop_counts


# --------------------------------------------------------------------------
# Allocation: split a tier budget across families (largest remainder + floor).
# --------------------------------------------------------------------------
def allocate(avail: Dict[str, int], total: int, min_floor: int) -> Dict[str, int]:
    avail = {k: int(v) for k, v in avail.items() if v > 0}
    if not avail:
        return {}
    keys = sorted(avail)
    if sum(avail.values()) <= total:
        return {k: avail[k] for k in keys}
    alloc = {k: 0 for k in keys}
    remaining = total
    # 1) floors, densest families first (deterministic tie-break by name)
    for k in sorted(keys, key=lambda x: (-avail[x], x)):
        if remaining <= 0:
            break
        f = min(min_floor, avail[k], remaining)
        alloc[k] = f
        remaining -= f
    # 2) largest-remainder proportional to residual capacity, iterated
    while remaining > 0:
        cap = {k: avail[k] - alloc[k] for k in keys if avail[k] - alloc[k] > 0}
        if not cap:
            break
        tot = sum(cap.values())
        quota = {k: remaining * cap[k] / tot for k in cap}
        base = {k: min(int(quota[k]), cap[k]) for k in cap}
        for k in cap:
            alloc[k] += base[k]
        remaining -= sum(base.values())
        if remaining <= 0:
            break
        order = sorted(cap, key=lambda k: (-(quota[k] - int(quota[k])), -cap[k], k))
        for k in order:
            if remaining <= 0:
                break
            if alloc[k] < avail[k]:
                alloc[k] += 1
                remaining -= 1
    return alloc


def _key(p: dict) -> tuple:
    """Unique record identity for cross-frame exclusion."""
    return (p["model"], p["condition"], p["bench_index"])


def _draw(pool: List[dict], count: int, seed_str: str,
          used: set, frame: str) -> List[dict]:
    """Deterministically sample ``count`` unused pointers from ``pool``."""
    candidates = sorted((p for p in pool if _key(p) not in used),
                        key=lambda p: (p["model"], p["bench_index"]))
    count = min(count, len(candidates))
    if count <= 0:
        return []
    rng = random.Random(seed_str)
    picked = rng.sample(candidates, count)
    out = []
    for p in picked:
        used.add(_key(p))
        q = dict(p)
        q["frame"] = frame
        out.append(q)
    return out


def _by_family(pool: List[dict], used: set) -> Dict[str, List[dict]]:
    fam = defaultdict(list)
    for p in pool:
        if _key(p) not in used:
            fam[p["task"]].append(p)
    return fam


def draw_stratified_tier(pool, tier, total, min_floor, seed_tag, used, frame):
    """Allocate ``total`` across families in ``pool`` and draw each cell."""
    fam = _by_family(pool, used)
    avail = {f: len(v) for f, v in fam.items()}
    alloc = allocate(avail, total, min_floor)
    picks, cell_alloc = [], {}
    for family in sorted(alloc):
        n = alloc[family]
        if n <= 0:
            continue
        cell_alloc[family] = n
        picks += _draw(fam[family], n,
                       f"{SEED}|{seed_tag}|{tier}|{family}", used, frame)
    return picks, cell_alloc


# --------------------------------------------------------------------------
# Pilot draw (independent; drawn first; spread across families per cell).
# --------------------------------------------------------------------------
def draw_pilot(pools, used) -> List[dict]:
    picks = []
    for cell_i, ((tier, condition), n) in enumerate(PILOT_CELLS.items()):
        pool = pools[condition][tier]
        fam = _by_family(pool, used)
        families = sorted(fam, key=lambda f: (-len(fam[f]), f))
        # rotate the family start per cell so calibration spans more families
        if families:
            off = cell_i % len(families)
            families = families[off:] + families[:off]
        # round-robin one item per distinct family until n reached
        taken, fi = 0, 0
        # pre-shuffle each family list deterministically
        shuffled = {}
        for f in families:
            lst = sorted(fam[f], key=lambda p: (p["model"], p["bench_index"]))
            rng2 = random.Random(f"{SEED}|pilot|{tier}|{condition}|{f}")
            rng2.shuffle(lst)
            shuffled[f] = lst
        cursor = {f: 0 for f in families}
        while taken < n and families:
            f = families[fi % len(families)]
            lst = shuffled[f]
            if cursor[f] < len(lst):
                p = lst[cursor[f]]
                cursor[f] += 1
                if _key(p) not in used:
                    used.add(_key(p))
                    q = dict(p)
                    q["frame"] = "pilot"
                    picks.append(q)
                    taken += 1
            fi += 1
            # stop if all families exhausted
            if all(cursor[f] >= len(shuffled[f]) for f in families):
                break
    return picks


# --------------------------------------------------------------------------
# Orchestration.
# --------------------------------------------------------------------------
def draw_manifest(verbose: bool = True) -> dict:
    pools, api_pool, pop_counts = build_pools(verbose=verbose)
    used: set = set()

    pilot = draw_pilot(pools, used)  # first -> excluded from all other frames

    main_p, main_alloc = [], {}
    for tier, total in MAIN_P_TIER_TARGETS.items():
        picks, cell = draw_stratified_tier(
            pools[CONDITION_HEADLINE][tier], tier, total, MAIN_MIN_FLOOR,
            "P", used, "main_P")
        main_p += picks
        main_alloc[tier] = cell

    api_t5, api_alloc = [], {}
    for model in API_MODELS:
        picks, cell = draw_stratified_tier(
            api_pool[model], model, API_PER_MODEL, API_MIN_FLOOR,
            "T5", used, "api_T5")
        # tag frame already set; keep per-model allocation
        api_t5 += picks
        api_alloc[model] = cell

    contrast_s, contrast_alloc = [], {}
    for tier, total in CONTRAST_S_TIER_TARGETS.items():
        picks, cell = draw_stratified_tier(
            pools[CONDITION_CONTRAST][tier], tier, total, MAIN_MIN_FLOOR,
            "S", used, "contrast_S")
        contrast_s += picks
        contrast_alloc[tier] = cell

    manifest = {
        "meta": {
            "protocol": "E1_validity_audit.md (s4.2, frozen 2026-07-23)",
            "seed": SEED,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "pandapower": "3.4.0",
            "condition_headline": f"cond{CONDITION_HEADLINE}",
            "condition_contrast": f"cond{CONDITION_CONTRAST}",
            "note": ("Lightweight pointers only (no record bodies). Re-fetch a "
                     "record via audit.records.fetch_record(cond_path(model, "
                     "condition[, api=...]), bench_index=..., item_id=...). "
                     "Pilot is drawn first and excluded from P/T5/S; the three "
                     "sample frames are disjoint by construction (distinct files)."),
        },
        "population_counts": {k: {t: dict(m) for t, m in v.items()}
                              for k, v in pop_counts.items()},
        "allocations": {"main_P": main_alloc, "api_T5": api_alloc,
                        "contrast_S": contrast_alloc,
                        "pilot_cells": {f"{t}|{c}": n for (t, c), n in PILOT_CELLS.items()}},
        "samples": {"pilot": pilot, "main_P": main_p,
                    "api_T5": api_t5, "contrast_S": contrast_s},
        "summary": {
            "pilot": len(pilot),
            "main_P": len(main_p),
            "api_T5": len(api_t5),
            "contrast_S": len(contrast_s),
            "total": len(pilot) + len(main_p) + len(api_t5) + len(contrast_s),
            "main_P_by_tier": {t: sum(1 for p in main_p if p["tier"] == t)
                               for t in MAIN_P_TIER_TARGETS},
            "contrast_S_by_tier": {t: sum(1 for p in contrast_s if p["tier"] == t)
                                   for t in CONTRAST_S_TIER_TARGETS},
            "api_T5_by_model": {m: sum(1 for p in api_t5 if p["model"] == m)
                                for m in API_MODELS},
        },
    }
    return manifest


def _print_summary(man: dict) -> None:
    s = man["summary"]
    print("\n=== E1 SAMPLE MANIFEST ===")
    print(f"seed={man['meta']['seed']}  total send-to-audit={s['total']}")
    print(f"  pilot      = {s['pilot']}")
    print(f"  main_P     = {s['main_P']}   by tier {s['main_P_by_tier']}")
    print(f"  api_T5     = {s['api_T5']}   by model {s['api_T5_by_model']}")
    print(f"  contrast_S = {s['contrast_S']}   by tier {s['contrast_S_by_tier']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw the frozen E1 sample manifest.")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "audit" / "e1_sample_manifest.json"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    man = draw_manifest(verbose=not args.quiet)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, default=str)
    _print_summary(man)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
