# --------------------------------------------------------------------------
# Repository copy of audit/pair_sampler.py from the frozen experimental
# pipeline. Its inputs are raw per-item run trees too large to ship here, so
# it does not run from this checkout; see ARTIFACT_INDEX.md for the outputs it
# produced.
# --------------------------------------------------------------------------
"""E1-P -- common-support pair population + frozen stratified pair draw.

Declared supplement round to the frozen 610-pointer main audit (E1 doc s8,
2026-07-30 entry).  The main audit drew ``main_P`` (condC_FDRS) and
``contrast_S`` (condA_FX) *independently*, so its strictly paired subset is
2 items and McNemar is not computable (E1 doc s8.5).  This module draws the
paired sample the doc's s8.5 closing paragraph asks for:

  population : every (model, bench_index) whose record is ``match: True`` in
               **both** ``condA_FX`` and ``condC_FDRS`` -- the common support
               already counted by the A5-1 pass (2106 pairs; T1 0 / T2 288 /
               T3 696 / T4 1122).  Open-source panel only: the API layer has
               no ``condA_FX`` counterpart on disk, so it cannot pair.
  sample     : 120 pairs = T2 30 / T3 40 / T4 50, split inside each tier
               across task families **proportionally** (largest remainder, no
               min-per-cell floor -- unlike the main audit's ``min_floor=2``,
               which at these budgets would spend the whole tier quota on
               floors and destroy proportionality), seed 20260730.
  overlap    : pairs whose C and/or A side was already ruled in the frozen 610
               are **kept** (pair completeness wins) and marked; the ruled side
               reuses its final label from ``t4_final_labels_v2.json`` instead
               of being re-audited, which saves judge budget and guarantees the
               supplement round cannot contradict the frozen round on a shared
               item.

Output: ``audit/e1p_pair_manifest.json`` -- pair records (both sides) plus the
two run-frames ``pair_C`` / ``pair_A`` in exactly the pointer shape
``audit.run_t4`` consumes, holding only the sides that still need the automatic
layer.  Pointers only (E1 doc s7 memory discipline).

Run:  python -m audit.pair_sampler        (writes audit/e1p_pair_manifest.json)

Extension mode (E1 doc sP.13, preregistered 2026-08-02): the T4 stratum of the
frozen 120-pair draw is under-powered (Delta=12.0pp, p=0.070 at n=50), and the
prereg declares a single +50-pair T4 extension from the *same* frozen
population, same per-tier proportional protocol, extension seed 20260802,
excluding every already-drawn pair (both sides).  The base manifest is
read-only; the extension writes its own manifest with an ``extension_of``
pointer and an exclusion ledger.

Run:  python -m audit.pair_sampler --extend-tier T4 --extend-n 50 \
          --seed 20260802 --exclude-manifest audit/e1p_pair_manifest.json
      (writes audit/e1p_ext_manifest.json)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from audit import records
from audit.records import (CONDITION_CONTRAST, CONDITION_HEADLINE,
                           PROJECT_ROOT, TIER_MODELS)
from audit.sampler import allocate

# --------------------------------------------------------------------------
# Frozen parameters (E1 doc s8 2026-07-30 "E1-P 配对审计轮立项"; freeze 2026-07-31).
# --------------------------------------------------------------------------
SEED = 20260730

PAIR_TIER_TARGETS = {"T2": 30, "T3": 40, "T4": 50}   # 120 pairs = 240 sides
PAIR_TIERS = ("T2", "T3", "T4")          # tiers with a sampling budget
SCAN_TIERS = ("T1", "T2", "T3", "T4")    # tiers streamed for the population
FAMILY_MIN_FLOOR = 0            # pure proportional split inside a tier

# A5-1 common-support counts to re-verify against (E1 doc s8.5 / t4_far_weighted.json).
# T1 is scanned rather than assumed: its zero is the doc's "condA_FX has no T1"
# claim, and it must be re-derived, not inherited.
EXPECTED_POPULATION = {"total": 2106, "T1": 0, "T2": 288, "T3": 696, "T4": 1122}

FRAME_OF_CONDITION = {CONDITION_HEADLINE: "pair_C", CONDITION_CONTRAST: "pair_A"}

_SIDE_FIELDS = ("item_id", "task", "difficulty_level", "network",
                "query_type", "n_modifications")


# --------------------------------------------------------------------------
# Population: stream both conditions, intersect on (model, bench_index).
# --------------------------------------------------------------------------
def _matched_index(model: str, condition: str) -> Tuple[Dict[int, dict], int, int]:
    """(bench_index -> light side-metadata, n_matched_records, n_duplicate_bi).

    Streaming (never ``json.load``); the returned dict holds only the small
    pointer fields, so memory stays bounded by one record at a time.  A
    bench_index seen twice keeps its first occurrence (deterministic) and is
    counted, so the manifest reports the duplicate count next to the pair
    population it was drawn from.
    """
    path = records.cond_path(model, condition)
    out: Dict[int, dict] = {}
    n_matched = n_dup = 0
    if not path.exists():
        return out, 0, 0
    for rec in records.iter_matched_records(path):
        bi = rec.get("bench_index")
        if bi is None:
            continue
        n_matched += 1
        if bi in out:
            n_dup += 1
            continue
        out[bi] = {f: rec.get(f) for f in _SIDE_FIELDS}
    return out, n_matched, n_dup


def build_pair_population(verbose: bool = True) -> Tuple[List[dict], dict]:
    """Return (pairs, population_stats) over the open-source panel."""
    pairs: List[dict] = []
    by_model: Dict[str, int] = {}
    a_pool: Dict[str, int] = {}
    c_pool: Dict[str, int] = {}
    dup_bi: Dict[str, int] = {}
    task_mismatch: List[dict] = []

    for tier in SCAN_TIERS:
        for model in TIER_MODELS[tier]:
            c_idx, c_n, c_dup = _matched_index(model, CONDITION_HEADLINE)
            a_idx, a_n, a_dup = _matched_index(model, CONDITION_CONTRAST)
            c_pool[model], a_pool[model] = c_n, a_n
            if c_dup or a_dup:
                dup_bi[model] = c_dup + a_dup
            common = sorted(set(c_idx) & set(a_idx))
            for bi in common:
                c_side, a_side = c_idx[bi], a_idx[bi]
                if c_side["task"] != a_side["task"]:
                    task_mismatch.append({"model": model, "bench_index": bi,
                                          "task_C": c_side["task"],
                                          "task_A": a_side["task"]})
                pairs.append({
                    "tier": tier, "model": model, "bench_index": bi,
                    "task": c_side["task"], "network": c_side["network"],
                    "query_type": c_side["query_type"],
                    "n_modifications": c_side["n_modifications"],
                    "difficulty_level": c_side["difficulty_level"],
                    "sides": {CONDITION_HEADLINE: c_side,
                              CONDITION_CONTRAST: a_side},
                })
            by_model[model] = len(common)
            if verbose:
                print(f"  [pair] {tier} {model}: C={len(c_idx)} A={len(a_idx)} "
                      f"-> common {len(common)}", flush=True)

    by_tier = {t: sum(1 for p in pairs if p["tier"] == t) for t in SCAN_TIERS}
    by_tier_family: Dict[str, Dict[str, int]] = {t: defaultdict(int) for t in SCAN_TIERS}
    for p in pairs:
        by_tier_family[p["tier"]][p["task"]] += 1

    a_by_tier = {t: sum(a_pool[m] for m in TIER_MODELS[t]) for t in SCAN_TIERS}
    c_by_tier = {t: sum(c_pool[m] for m in TIER_MODELS[t]) for t in SCAN_TIERS}
    n_a, n_c = sum(a_by_tier.values()), sum(c_by_tier.values())

    recount_ok = (len(pairs) == EXPECTED_POPULATION["total"]
                  and all(by_tier[t] == EXPECTED_POPULATION[t] for t in SCAN_TIERS))
    stats = {
        "n_pairs": len(pairs),
        "by_tier": by_tier,
        "by_model": by_model,
        "by_tier_family": {t: dict(sorted(d.items())) for t, d in by_tier_family.items()},
        "A_FX_matched_by_tier": a_by_tier,
        "C_FDRS_matched_by_tier": c_by_tier,
        "A_FX_matched_by_model": a_pool,
        "C_FDRS_matched_by_model": c_pool,
        "duplicate_bench_index_by_model": dup_bi,
        "share_of_A_FX_pool_pct": round(100 * len(pairs) / n_a, 2) if n_a else None,
        "share_of_C_FDRS_pool_pct": round(100 * len(pairs) / n_c, 2) if n_c else None,
        "note_T1": "T1 is streamed, not assumed: its condA_FX matched pool is "
                   "re-derived here, so the frozen round's 'contrast frame has "
                   "no T1' fact and the resulting zero T1 pairs are reproduced "
                   "rather than inherited; pool shares therefore use the full "
                   "four-tier pools, matching the A5-1 figures",
        "recount_vs_A5_1": {"expected": EXPECTED_POPULATION,
                            "observed": {"total": len(pairs), **by_tier},
                            "reproduced": recount_ok},
        "task_family_mismatch_across_sides": task_mismatch,
    }
    return pairs, stats


# --------------------------------------------------------------------------
# Draw: proportional-by-family inside each tier, frozen seed.
# --------------------------------------------------------------------------
def draw_pairs(pairs: List[dict], verbose: bool = True, *,
               targets: Dict[str, int] = PAIR_TIER_TARGETS, seed: int = SEED,
               id_start: int = 1) -> Tuple[List[dict], dict]:
    by_tier_family: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        by_tier_family[p["tier"]][p["task"]].append(p)

    drawn: List[dict] = []
    allocation: Dict[str, Dict[str, int]] = {}
    for tier in (t for t in PAIR_TIERS if t in targets):
        fam = by_tier_family[tier]
        avail = {f: len(v) for f, v in fam.items()}
        alloc = allocate(avail, targets[tier], FAMILY_MIN_FLOOR)
        allocation[tier] = {f: alloc[f] for f in sorted(alloc) if alloc[f] > 0}
        for family in sorted(allocation[tier]):
            n = allocation[tier][family]
            pool = sorted(fam[family], key=lambda p: (p["model"], p["bench_index"]))
            rng = random.Random(f"{seed}|E1P|{tier}|{family}")
            drawn.extend(rng.sample(pool, n))
        if verbose:
            print(f"  [draw] {tier}: target {targets[tier]} over "
                  f"{len(avail)} families -> {allocation[tier]}", flush=True)

    drawn.sort(key=lambda p: (p["tier"], p["model"], p["bench_index"]))
    for i, p in enumerate(drawn, start=id_start):
        p["pair_id"] = f"E1P-{i:03d}"
    return drawn, allocation


# --------------------------------------------------------------------------
# Overlap with the frozen 610 (and with the pilot 20, informational).
# --------------------------------------------------------------------------
def _label_key(model: str, condition: str, bench_index: int) -> str:
    return f"{model}|{condition}|{bench_index}"


def verify_against_frozen_pools(pop: dict, manifest_path: Path) -> dict:
    """Cross-check the re-streamed matched pools against the frozen manifest.

    ``e1_sample_manifest.json`` recorded the per-model matched counts at freeze
    time (2026-07-23); if this round's stream reproduces them model-by-model,
    the pair population is built on the same data the frozen round used.
    """
    frozen = json.load(open(manifest_path, "r", encoding="utf-8"))["population_counts"]
    mismatches = []
    for condition, key in ((CONDITION_HEADLINE, "C_FDRS_matched_by_model"),
                           (CONDITION_CONTRAST, "A_FX_matched_by_model")):
        for tier in SCAN_TIERS:
            for model, n_frozen in frozen.get(condition, {}).get(tier, {}).items():
                n_now = pop[key].get(model)
                if n_now != n_frozen:
                    mismatches.append({"condition": condition, "model": model,
                                       "frozen": n_frozen, "recounted": n_now})
    return {"source": str(manifest_path.name), "mismatches": mismatches,
            "reproduced": not mismatches}


def load_frozen_context(labels_path: Path, manifest_path: Path) -> Tuple[dict, dict, set]:
    """Return (final_labels, frame_of_pointer, pilot_keys) from frozen artefacts."""
    labels = json.load(open(labels_path, "r", encoding="utf-8"))
    frame_of = {k: v.get("frame") for k, v in labels.items()}
    pilot = set()
    man = json.load(open(manifest_path, "r", encoding="utf-8"))
    for ptr in man["samples"].get("pilot", []):
        pilot.add(_label_key(ptr["model"], ptr["condition"], ptr["bench_index"]))
    return labels, frame_of, pilot


def mark_overlap(drawn: List[dict], labels: dict, frame_of: dict,
                 pilot_keys: set) -> dict:
    """Annotate each drawn pair's sides with frozen-round membership."""
    n_sides = {CONDITION_HEADLINE: 0, CONDITION_CONTRAST: 0}
    n_pilot = 0
    reused: List[dict] = []
    for pair in drawn:
        overlap = {}
        for condition in (CONDITION_HEADLINE, CONDITION_CONTRAST):
            key = _label_key(pair["model"], condition, pair["bench_index"])
            entry = labels.get(key)
            info = {
                "in_frozen_610": entry is not None,
                "frozen_frame": frame_of.get(key),
                "final_label": (entry or {}).get("label"),
                "in_pilot_20": key in pilot_keys,
                "label_key": key,
            }
            if entry is not None:
                n_sides[condition] += 1
                reused.append({"pair_id": pair["pair_id"], "model": pair["model"],
                               "condition": condition,
                               "bench_index": pair["bench_index"],
                               "frozen_frame": frame_of.get(key),
                               "final_label": entry.get("label")})
            if key in pilot_keys:
                n_pilot += 1
            overlap[condition] = info
        pair["overlap"] = overlap

    n_pairs_any = sum(1 for p in drawn
                      if any(p["overlap"][c]["in_frozen_610"]
                             for c in (CONDITION_HEADLINE, CONDITION_CONTRAST)))
    n_pairs_both = sum(1 for p in drawn
                       if all(p["overlap"][c]["in_frozen_610"]
                              for c in (CONDITION_HEADLINE, CONDITION_CONTRAST)))
    n_total_sides = 2 * len(drawn)
    return {
        "n_pairs": len(drawn),
        "n_sides_total": n_total_sides,
        "n_sides_in_frozen_610": sum(n_sides.values()),
        "n_sides_in_frozen_610_by_condition": {f"cond{c}": n for c, n in n_sides.items()},
        "n_pairs_with_any_overlapping_side": n_pairs_any,
        "n_pairs_with_both_sides_overlapping": n_pairs_both,
        "overlap_rate_sides_pct": round(100 * sum(n_sides.values()) / n_total_sides, 2),
        "overlap_rate_pairs_pct": round(100 * n_pairs_any / len(drawn), 2) if drawn else None,
        "n_sides_in_pilot_20": n_pilot,
        "reused_rulings": reused,
        "policy": "overlapping sides are NOT excluded from the pair draw (pair "
                  "completeness first); they skip the automatic layer and reuse "
                  "their frozen final ruling from t4_final_labels_v2.json, so "
                  "the supplement round stays consistent with the frozen round "
                  "and spends judge budget only on unruled sides",
    }


# --------------------------------------------------------------------------
# Run-frames in audit.run_t4 pointer shape.
# --------------------------------------------------------------------------
def build_run_frames(drawn: List[dict]) -> Tuple[Dict[str, List[dict]], List[dict]]:
    samples: Dict[str, List[dict]] = {"pair_C": [], "pair_A": []}
    skipped: List[dict] = []
    for pair in drawn:
        for condition in (CONDITION_HEADLINE, CONDITION_CONTRAST):
            side = pair["sides"][condition]
            ptr = {
                "frame": FRAME_OF_CONDITION[condition],
                "pair_id": pair["pair_id"],
                "tier": pair["tier"],
                "model": pair["model"],
                "condition": condition,
                "bench_index": pair["bench_index"],
                "item_id": side["item_id"],
                "task": side["task"],
                "difficulty_level": side["difficulty_level"],
                "network": side["network"],
                "query_type": side["query_type"],
                "n_modifications": side["n_modifications"],
            }
            if pair["overlap"][condition]["in_frozen_610"]:
                skipped.append({**ptr,
                                "reason": "ruled in frozen 610",
                                "final_label": pair["overlap"][condition]["final_label"],
                                "frozen_frame": pair["overlap"][condition]["frozen_frame"]})
            else:
                samples[FRAME_OF_CONDITION[condition]].append(ptr)
    return samples, skipped


# --------------------------------------------------------------------------
# Orchestration.
# --------------------------------------------------------------------------
def _summary(drawn: List[dict], samples: Dict[str, List[dict]],
             skipped: List[dict], tiers: Tuple[str, ...] = PAIR_TIERS) -> dict:
    return {
        "n_pairs_drawn": len(drawn),
        "by_tier": {t: sum(1 for p in drawn if p["tier"] == t) for t in tiers},
        "by_model": {m: sum(1 for p in drawn if p["model"] == m)
                     for t in tiers for m in TIER_MODELS[t]},
        "by_tier_family": {t: {f: sum(1 for p in drawn if p["tier"] == t
                                      and p["task"] == f)
                               for f in sorted({p["task"] for p in drawn
                                                if p["tier"] == t})}
                           for t in tiers},
        "n_sides_total": 2 * len(drawn),
        "n_sides_to_audit": sum(len(v) for v in samples.values()),
        "n_sides_to_audit_by_frame": {k: len(v) for k, v in samples.items()},
        "n_sides_reusing_frozen_ruling": len(skipped),
    }


def draw_manifest(labels_path: Path, frozen_manifest_path: Path,
                  verbose: bool = True) -> dict:
    pairs, pop = build_pair_population(verbose=verbose)
    pop["recount_vs_frozen_manifest"] = verify_against_frozen_pools(
        pop, frozen_manifest_path)
    drawn, allocation = draw_pairs(pairs, verbose=verbose)
    labels, frame_of, pilot_keys = load_frozen_context(labels_path,
                                                       frozen_manifest_path)
    overlap = mark_overlap(drawn, labels, frame_of, pilot_keys)
    samples, skipped = build_run_frames(drawn)

    return {
        "meta": {
            "round": "E1-P (declared supplement to the frozen 610-pointer main audit)",
            "protocol": "E1_validity_audit.md "
                        "s8 (2026-07-30 entry, E1-P 配对审计轮立项) on the s8.5 "
                        "common-support population",
            "seed": SEED,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "condition_headline": f"cond{CONDITION_HEADLINE}",
            "condition_contrast": f"cond{CONDITION_CONTRAST}",
            "pair_definition": "(model, bench_index) with match=True in BOTH "
                               "conditions; open-source panel only (the API "
                               "layer has no condA_FX counterpart)",
            "allocation_rule": "tier budgets T2/T3/T4 = 30/40/50, split across "
                               "task families proportionally (largest remainder, "
                               "no min-per-cell floor); per-cell RNG seeded "
                               f"'{SEED}|E1P|<tier>|<family>'",
            "frozen_round_untouched": "reads t4_final_labels_v2.json and "
                                      "e1_sample_manifest.json read-only; no "
                                      "number of the frozen 610 is altered",
            "note": "Lightweight pointers only. samples.pair_C / samples.pair_A "
                    "are the sides still needing the automatic layer and are in "
                    "audit.run_t4 pointer shape; overlapping sides are listed in "
                    "skipped_overlap_sides with their frozen final label.",
        },
        "population": pop,
        "allocation": allocation,
        "overlap": overlap,
        "pairs": drawn,
        "samples": samples,
        "skipped_overlap_sides": skipped,
        "summary": _summary(drawn, samples, skipped),
    }


def extend_manifest(labels_path: Path, frozen_manifest_path: Path,
                    exclude_manifest_path: Path, tier: str, n: int, seed: int,
                    verbose: bool = True) -> dict:
    """Preregistered single-tier incremental extension (E1 doc sP.13).

    Draws ``n`` additional pairs in ``tier`` from the *same* frozen
    common-support population, with the same per-family proportional protocol,
    over the residual pool: every pair already drawn in
    ``exclude_manifest_path`` is excluded (which excludes both of its sides).
    The base manifest is read-only; pair_ids continue its numbering.
    """
    base = json.load(open(exclude_manifest_path, "r", encoding="utf-8"))
    base_pairs = base["pairs"]
    excluded = {(p["model"], p["bench_index"]) for p in base_pairs}
    base_sides = {(p["model"], c, p["bench_index"]) for p in base_pairs
                  for c in (CONDITION_HEADLINE, CONDITION_CONTRAST)}

    pairs, pop = build_pair_population(verbose=verbose)
    pop["recount_vs_frozen_manifest"] = verify_against_frozen_pools(
        pop, frozen_manifest_path)
    pop["recount_vs_base_manifest"] = {
        "source": Path(exclude_manifest_path).name,
        "base": {"n_pairs": base["population"]["n_pairs"],
                 "by_tier": base["population"]["by_tier"]},
        "observed": {"n_pairs": pop["n_pairs"], "by_tier": pop["by_tier"]},
        "reproduced": (pop["n_pairs"] == base["population"]["n_pairs"]
                       and pop["by_tier"] == base["population"]["by_tier"]),
    }

    tier_total = sum(1 for p in pairs if p["tier"] == tier)
    eligible = [p for p in pairs if p["tier"] == tier
                and (p["model"], p["bench_index"]) not in excluded]
    drawn, allocation = draw_pairs(eligible, verbose=verbose, targets={tier: n},
                                   seed=seed, id_start=len(base_pairs) + 1)

    drawn_keys = {(p["model"], p["bench_index"]) for p in drawn}
    drawn_sides = {(p["model"], c, p["bench_index"]) for p in drawn
                   for c in (CONDITION_HEADLINE, CONDITION_CONTRAST)}
    checks = {
        "n_drawn": len(drawn),
        "all_in_extend_tier": all(p["tier"] == tier for p in drawn),
        "n_pair_overlap_with_base": len(drawn_keys & excluded),
        "n_side_overlap_with_base_sides": len(drawn_sides & base_sides),
        "population_reproduced_vs_base": pop["recount_vs_base_manifest"]["reproduced"],
        "population_reproduced_vs_A5_1": pop["recount_vs_A5_1"]["reproduced"],
        "population_reproduced_vs_frozen_manifest":
            pop["recount_vs_frozen_manifest"]["reproduced"],
        "tier_pool": tier_total,
        "tier_pool_after_exclusion": len(eligible),
    }
    assert checks["n_drawn"] == n and checks["all_in_extend_tier"], checks
    assert checks["n_pair_overlap_with_base"] == 0, checks
    assert checks["n_side_overlap_with_base_sides"] == 0, checks
    assert checks["population_reproduced_vs_base"], checks

    labels, frame_of, pilot_keys = load_frozen_context(labels_path,
                                                       frozen_manifest_path)
    overlap = mark_overlap(drawn, labels, frame_of, pilot_keys)
    samples, skipped = build_run_frames(drawn)

    return {
        "meta": {
            "round": "E1-P T4 power extension (preregistered continuation of "
                     "the E1-P paired round, not a new experiment)",
            "protocol": "E1_validity_audit.md "
                        "sP.13 (frozen 2026-08-02, prior to data) via the "
                        "sP.7-5 pre-declared extension path",
            "seed": seed,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "condition_headline": f"cond{CONDITION_HEADLINE}",
            "condition_contrast": f"cond{CONDITION_CONTRAST}",
            "pair_definition": "(model, bench_index) with match=True in BOTH "
                               "conditions; open-source panel only (the API "
                               "layer has no condA_FX counterpart)",
            "allocation_rule": f"single-tier extension budget {tier}={n}, "
                               "split across task families proportionally "
                               "(largest remainder, no min-per-cell floor) "
                               "over the residual pool after excluding the "
                               "base draw; per-cell RNG seeded "
                               f"'{seed}|E1P|<tier>|<family>'",
            "extension_of": {
                "manifest": Path(exclude_manifest_path).name,
                "seed": base["meta"]["seed"],
                "n_pairs": len(base_pairs),
                "pair_ids": f"E1P-001..E1P-{len(base_pairs):03d} (this file "
                            f"continues at E1P-{len(base_pairs) + 1:03d})",
            },
            "frozen_round_untouched": "reads e1p_pair_manifest.json, "
                                      "t4_final_labels_v2.json and "
                                      "e1_sample_manifest.json read-only; no "
                                      "number of the frozen 610 or of the "
                                      "base 120-pair round is altered",
            "note": "Lightweight pointers only. samples.pair_C / samples.pair_A "
                    "are the sides still needing the automatic layer and are in "
                    "audit.run_t4 pointer shape; overlapping sides are listed in "
                    "skipped_overlap_sides with their frozen final label.",
        },
        "population": pop,
        "allocation": allocation,
        "exclusion": {
            "source_manifest": Path(exclude_manifest_path).name,
            "n_excluded_pairs": len(excluded),
            "n_excluded_sides": len(base_sides),
            "n_excluded_pairs_in_extend_tier": tier_total - len(eligible),
            "excluded_pair_keys": sorted(f"{m}|{bi}" for m, bi in excluded),
        },
        "extension_checks": checks,
        "overlap": overlap,
        "pairs": drawn,
        "samples": samples,
        "skipped_overlap_sides": skipped,
        "summary": _summary(drawn, samples, skipped, tiers=(tier,)),
    }


def _print_summary(man: dict) -> None:
    pop, s, ov = man["population"], man["summary"], man["overlap"]
    print("\n=== E1-P PAIR MANIFEST ===")
    print(f"population: {pop['n_pairs']} pairs  by tier {pop['by_tier']}  "
          f"(A5-1 reproduced: {pop['recount_vs_A5_1']['reproduced']}; "
          f"frozen pools reproduced: "
          f"{pop['recount_vs_frozen_manifest']['reproduced']})")
    print(f"drawn:      {s['n_pairs_drawn']} pairs  by tier {s['by_tier']}")
    print(f"overlap:    {ov['n_sides_in_frozen_610']}/{ov['n_sides_total']} sides "
          f"({ov['overlap_rate_sides_pct']}%)  "
          f"{ov['n_sides_in_frozen_610_by_condition']}  "
          f"pairs touched {ov['n_pairs_with_any_overlapping_side']}")
    print(f"to audit:   {s['n_sides_to_audit']} sides {s['n_sides_to_audit_by_frame']}"
          f"  (reusing frozen rulings: {s['n_sides_reusing_frozen_ruling']})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Draw the frozen E1-P common-support pair manifest, or a "
                    "preregistered single-tier incremental extension of it "
                    "(--extend-tier, E1 doc sP.13).")
    ap.add_argument("--out", default=None,
                    help="output manifest (default: e1p_pair_manifest.json, "
                         "or e1p_ext_manifest.json in extension mode)")
    ap.add_argument("--final-labels",
                    default=str(PROJECT_ROOT / "audit" / "t4_final_labels_v2.json"))
    ap.add_argument("--frozen-manifest",
                    default=str(PROJECT_ROOT / "audit" / "e1_sample_manifest.json"))
    ap.add_argument("--extend-tier", default=None, choices=PAIR_TIERS,
                    help="extension mode: draw --extend-n extra pairs in this "
                         "tier only, excluding every pair in --exclude-manifest")
    ap.add_argument("--extend-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None,
                    help="extension seed (required with --extend-tier; the "
                         "base draw's seed is frozen in code)")
    ap.add_argument("--exclude-manifest",
                    default=str(PROJECT_ROOT / "audit" / "e1p_pair_manifest.json"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.extend_tier:
        if args.seed is None:
            ap.error("--extend-tier requires an explicit --seed (prereg sP.13)")
        out = Path(args.out or PROJECT_ROOT / "audit" / "e1p_ext_manifest.json")
        if out.resolve() == Path(args.exclude_manifest).resolve():
            ap.error("refusing to overwrite the base manifest")
        man = extend_manifest(Path(args.final_labels), Path(args.frozen_manifest),
                              Path(args.exclude_manifest), args.extend_tier,
                              args.extend_n, args.seed, verbose=not args.quiet)
    else:
        if args.seed is not None:
            ap.error("--seed is only for extension mode; the base draw's seed "
                     "is frozen in code")
        out = Path(args.out or PROJECT_ROOT / "audit" / "e1p_pair_manifest.json")
        man = draw_manifest(Path(args.final_labels), Path(args.frozen_manifest),
                            verbose=not args.quiet)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, default=str)
    _print_summary(man)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
