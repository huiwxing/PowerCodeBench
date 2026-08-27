#!/usr/bin/env python3
"""Rebuild the public engineering-validity table statistics.

This is the analysis-only counterpart to ``run_t4.py``.  It does not rerun
the simulator or revise any adjudication; it recomputes the published FAR,
automatic-screen, and tolerance-sensitivity point estimates from the frozen
per-item labels, sample manifest, automatic-check records, and design cells
archived under ``audit/``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "audit"
PRIMARY_COMPACT = (
    ROOT / "results" / "raw" / "main_experiment"
    / "primary_outcomes_compact.json"
)
BOOTSTRAP_B = 2_000
BOOTSTRAP_SEED = 20_260_731
OPEN_MODEL_TIERS = {
    "Qwen_Qwen2.5-Coder-0.5B-Instruct": "T1",
    "Qwen_Qwen2.5-Coder-1.5B-Instruct": "T1",
    "Qwen_Qwen2.5-Coder-7B-Instruct": "T2",
    "meta-llama_Llama-3.1-8B-Instruct": "T2",
    "Qwen_Qwen2.5-Coder-14B-Instruct": "T2",
    "Qwen_Qwen2.5-Coder-32B-Instruct": "T2",
    "meta-llama_Llama-3.1-70B-Instruct": "T3",
    "openai_gpt-oss-120b": "T3",
    "meta-llama_Llama-3.1-405B-Instruct": "T4",
    "Qwen_Qwen3-Coder-480B-A35B-Instruct": "T4",
    "Qwen_Qwen3-Coder-Next": "T4",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    if not n:
        return None
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(100 * (center - half), 2), round(100 * (center + half), 2)]


def sample_key(row: dict) -> str:
    return f"{row['model']}|{row['condition']}|{row['bench_index']}"


def design_cell(row: dict) -> str:
    frame = row["frame"]
    if frame == "api_T5":
        return f"{frame}|{row['model']}|{row['task']}"
    return f"{frame}|{row['tier']}|{row['task']}"


def pct(num: float, den: float) -> float | None:
    return round(100 * num / den, 2) if den else None


def bootstrap_stratum(row: dict, mode: str) -> str:
    """Return the frozen A5-1 resampling stratum for one sampled item.

    ``cell`` is the sampler allocation cell.  ``collapsed`` pools cells at
    frame x tier (open-weight frames) or frame x model (the API frame).
    ``hybrid`` retains cells with at least two original draws and pools only
    singleton cells at that collapsed level.  The singleton test deliberately
    uses the frozen *design* n, not the post-defect or post-tolerance count.
    """
    if mode not in ("cell", "collapsed", "hybrid"):
        raise ValueError(f"unknown bootstrap mode: {mode}")
    if row["frame"] == "api_T5":
        collapsed = f"{row['frame']}|{row['model']}"
    else:
        collapsed = f"{row['frame']}|{row['tier']}"
    if mode == "collapsed":
        return collapsed
    if mode == "cell" or row["design_n_sample"] >= 2:
        return row["design_cell"]
    return collapsed


def bootstrap_far_ci(
    rows: list[dict], *, mode: str, weighted: bool,
    tightened: bool = False,
) -> list[float]:
    """Recompute the frozen percentile bootstrap interval.

    This mirrors the archived A5-1 implementation exactly: ordinary FAR draws
    resample the full design and then exclude sampled item-defect copies,
    whereas the tightened-tolerance sensitivity first restricts the accept
    set; each stratum is sampled with replacement at its retained size using
    ``random.Random(seed).choices``; and the sorted order statistics at indices
    ``int(.025*B)`` and ``int(.975*B)`` are used.
    The latter convention is recorded explicitly because replacing it with a
    library's interpolated percentile changes a few endpoints by 0.01--0.09pp.
    """
    retained = rows
    if tightened:
        retained = [
            row for row in rows
            if row["label"] != "item-defect" and not row["would_fail_tight"]
        ]
    strata: dict[str, list[dict]] = defaultdict(list)
    for row in retained:
        strata[bootstrap_stratum(row, mode)].append(row)

    rng = random.Random(BOOTSTRAP_SEED)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_B):
        draw = [
            row
            for stratum in strata.values()
            for row in rng.choices(stratum, k=len(stratum))
        ]
        if not tightened:
            draw = [row for row in draw if row["label"] != "item-defect"]
        denominator = sum(row["weight"] if weighted else 1.0 for row in draw)
        numerator = sum(
            row["weight"] if weighted else 1.0
            for row in draw if row["label"] == "contaminated"
        )
        estimates.append(100 * numerator / denominator)

    estimates.sort()
    lo = estimates[int(0.025 * BOOTSTRAP_B)]
    hi = estimates[int(0.975 * BOOTSTRAP_B)]
    return [round(lo, 2), round(hi, 2)]


def bootstrap_difference_ci(
    treatment: list[dict], reference: list[dict], *, weighted: bool,
) -> list[float]:
    """Joint bootstrap for two separately sampled frame estimates.

    The treatment frame is drawn first and the reference frame second from one
    frozen RNG stream on every replicate.  The samples remain independent in
    design terms; sharing an RNG object only defines the deterministic Monte
    Carlo sequence and does not turn this into a paired item comparison.
    """
    def groups(rows: list[dict]) -> list[list[dict]]:
        strata: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            strata[bootstrap_stratum(row, "hybrid")].append(row)
        return list(strata.values())

    def draw_far(strata: list[list[dict]], rng: random.Random) -> float:
        draw = [
            row
            for stratum in strata
            for row in rng.choices(stratum, k=len(stratum))
            if row["label"] != "item-defect"
        ]
        denominator = sum(row["weight"] if weighted else 1.0 for row in draw)
        numerator = sum(
            row["weight"] if weighted else 1.0
            for row in draw if row["label"] == "contaminated"
        )
        return 100 * numerator / denominator

    treatment_strata = groups(treatment)
    reference_strata = groups(reference)
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = [
        draw_far(treatment_strata, rng) - draw_far(reference_strata, rng)
        for _ in range(BOOTSTRAP_B)
    ]
    estimates.sort()
    return [
        round(estimates[int(0.025 * BOOTSTRAP_B)], 2),
        round(estimates[int(0.975 * BOOTSTRAP_B)], 2),
    ]


def standardised_far(
    rows: list[dict], *, tiers: tuple[str, ...], tier_population: dict[str, int],
    weighted: bool,
) -> float:
    """Directly standardise frame FARs to a frozen population tier mix."""
    population = sum(tier_population[tier] for tier in tiers)
    estimate = 0.0
    for tier in tiers:
        tier_rows = [row for row in rows if row["tier"] == tier]
        denominator = sum(
            row["weight"] if weighted else 1.0
            for row in tier_rows if row["label"] != "item-defect"
        )
        numerator = sum(
            row["weight"] if weighted else 1.0
            for row in tier_rows if row["label"] == "contaminated"
        )
        if not denominator:
            raise ValueError(f"empty effective common-support tier: {tier}")
        estimate += (
            tier_population[tier] / population * 100 * numerator / denominator
        )
    return estimate


def bootstrap_standardised_difference_ci(
    treatment: list[dict], reference: list[dict], *, tiers: tuple[str, ...],
    tier_population: dict[str, int], weighted: bool,
) -> list[float]:
    """Joint common-support bootstrap for a tier-standardised contrast.

    This recovers the frozen A5-1 routine: restrict both independently drawn
    audit frames to items matched under both arms, retain the original hybrid
    sampling strata, draw the baseline/contrast frame first and the full-method
    frame second from one RNG stream, and directly standardise every replicate
    to the common-support population's tier distribution.
    """
    def groups(rows: list[dict]) -> list[list[dict]]:
        strata: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row["tier"] in tiers:
                strata[bootstrap_stratum(row, "hybrid")].append(row)
        return list(strata.values())

    def draw(strata: list[list[dict]], rng: random.Random) -> list[dict]:
        return [
            row
            for stratum in strata
            for row in rng.choices(stratum, k=len(stratum))
            if row["label"] != "item-defect"
        ]

    treatment_strata = groups(treatment)
    reference_strata = groups(reference)
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = []
    for _ in range(BOOTSTRAP_B):
        treatment_draw = draw(treatment_strata, rng)
        reference_draw = draw(reference_strata, rng)
        estimates.append(
            standardised_far(
                treatment_draw, tiers=tiers, tier_population=tier_population,
                weighted=weighted)
            - standardised_far(
                reference_draw, tiers=tiers, tier_population=tier_population,
                weighted=weighted)
        )
    estimates.sort()
    return [
        round(estimates[int(0.025 * BOOTSTRAP_B)], 2),
        round(estimates[int(0.975 * BOOTSTRAP_B)], 2),
    ]


def far_stats(rows: list[dict]) -> dict:
    effective = [row for row in rows if row["label"] != "item-defect"]
    contaminated = [row for row in effective if row["label"] == "contaminated"]
    weighted_den = sum(row["weight"] for row in effective)
    weighted_num = sum(row["weight"] for row in contaminated)
    return {
        "n_sampled": len(rows),
        "n_defect": len(rows) - len(effective),
        "n_eff": len(effective),
        "n_contaminated": len(contaminated),
        "far_unweighted_pct": pct(len(contaminated), len(effective)),
        "wilson95_pct": wilson(len(contaminated), len(effective)),
        "far_weighted_pct": pct(weighted_num, weighted_den),
        "est_population_nondefect": round(weighted_den, 1),
        "est_population_contaminated": round(weighted_num, 1),
    }


def frame_far_stats(rows: list[dict], *, full_sensitivity: bool = True) -> dict:
    """FAR estimates plus the frame-level hybrid bootstrap."""
    out = far_stats(rows)
    out["boot95_unweighted_pct"] = bootstrap_far_ci(
        rows, mode="hybrid", weighted=False)
    out["boot95_weighted_pct"] = bootstrap_far_ci(
        rows, mode="hybrid", weighted=True)
    if full_sensitivity:
        out["boot95_weighted_cell_pct"] = bootstrap_far_ci(
            rows, mode="cell", weighted=True)
        out["boot95_weighted_collapsed_pct"] = bootstrap_far_ci(
            rows, mode="collapsed", weighted=True)
    out["delta_w_minus_u_pp"] = round(
        out["far_weighted_pct"] - out["far_unweighted_pct"], 2)
    return out


def tier_far_stats(rows: list[dict]) -> dict:
    """Tier table used the strict allocation-cell bootstrap."""
    out = far_stats(rows)
    out["boot95_weighted_pct"] = bootstrap_far_ci(
        rows, mode="cell", weighted=True)
    return out


def difference_stats(treatment: list[dict], reference: list[dict]) -> dict:
    treatment_stats = far_stats(treatment)
    reference_stats = far_stats(reference)
    return {
        "delta_unweighted_pp": round(
            treatment_stats["far_unweighted_pct"]
            - reference_stats["far_unweighted_pct"], 2),
        "boot95_unweighted_pp": bootstrap_difference_ci(
            treatment, reference, weighted=False),
        "delta_weighted_pp": round(
            treatment_stats["far_weighted_pct"]
            - reference_stats["far_weighted_pct"], 2),
        "boot95_weighted_pp": bootstrap_difference_ci(
            treatment, reference, weighted=True),
        "design": (
            "separately drawn frames (different item populations) -- not a "
            "paired/common-support contrast"),
    }


def gate_stats(rows: list[dict]) -> dict:
    effective = [row for row in rows if row["label"] != "item-defect"]
    tp = sum(row["flag"] and row["label"] == "contaminated" for row in effective)
    fp = sum(row["flag"] and row["label"] != "contaminated" for row in effective)
    fn = sum(not row["flag"] and row["label"] == "contaminated" for row in effective)
    tn = sum(not row["flag"] and row["label"] != "contaminated" for row in effective)
    n = len(effective)
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "prevalence_pct": pct(tp + fn, n),
        "flag_rate_pct": pct(tp + fp, n),
        "sensitivity_pct": pct(tp, tp + fn),
        "specificity_pct": pct(tn, tn + fp),
        "precision_ppv_pct": pct(tp, tp + fp),
        "npv_pct": pct(tn, tn + fn),
        "sens_wilson95_pct": wilson(tp, tp + fn),
        "spec_wilson95_pct": wilson(tn, tn + fp),
        "ppv_wilson95_pct": wilson(tp, tp + fp),
        "npv_wilson95_pct": wilson(tn, tn + fn),
        "balanced_accuracy_pct": round(
            (100 * tp / (tp + fn) + 100 * tn / (tn + fp)) / 2, 2),
    }


def tightened_stats(rows: list[dict]) -> dict:
    effective = [row for row in rows if row["label"] != "item-defect"]
    kept = [row for row in effective if not row["would_fail_tight"]]
    before_cont = [row for row in effective if row["label"] == "contaminated"]
    after_cont = [row for row in kept if row["label"] == "contaminated"]
    before_legit = [row for row in effective if row["label"] != "contaminated"]
    rejected_legit = [row for row in before_legit if row["would_fail_tight"]]
    out = {
        "n_eff_before": len(effective),
        "n_eff_after": len(kept),
        "contaminated_before": len(before_cont),
        "contaminated_after": len(after_cont),
        "contaminated_rejected_pct": pct(
            len(before_cont) - len(after_cont), len(before_cont)),
        "legitimate_rejected_pct": pct(len(rejected_legit), len(before_legit)),
        "far_unweighted_pct": pct(len(after_cont), len(kept)),
        "far_weighted_pct": pct(
            sum(row["weight"] for row in after_cont),
            sum(row["weight"] for row in kept)),
    }
    out["boot95_weighted_pct"] = bootstrap_far_ci(
        rows, mode="hybrid", weighted=True, tightened=True)
    return out


def select(rows: list[dict], *, frames: tuple[str, ...] | None = None,
           tier: str | None = None, model: str | None = None) -> list[dict]:
    return [
        row for row in rows
        if (frames is None or row["frame"] in frames)
        and (tier is None or row["tier"] == tier)
        and (model is None or row["model"] == model)
    ]


def recover_common_support(
    rows: list[dict], manifest: dict, compact: dict,
) -> tuple[list[dict], dict]:
    """Recover the two-arm matched domain from the public compact bit vectors."""
    dataset = compact["datasets"]["powercodebench_2000"]
    item_ids = dataset["item_ids"]
    if dataset["item_order"] != "lexicographic item_id":
        raise ValueError("unexpected compact item ordering")
    if len(item_ids) != dataset["n_items"] or len(set(item_ids)) != len(item_ids):
        raise ValueError("invalid compact item axis")
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}

    model_tier = {}
    for tier, models in manifest["population_counts"]["C_FDRS"].items():
        if tier == "API":
            continue
        for model in models:
            if model in model_tier:
                raise ValueError(f"model appears in multiple tiers: {model}")
            model_tier[model] = tier

    support: set[tuple[str, str]] = set()
    by_model = {}
    by_tier = {tier: 0 for tier in ("T1", "T2", "T3", "T4")}
    for model, tier in model_tier.items():
        arm_bits = []
        for condition in ("A_FX", "C_FDRS"):
            run = compact["runs"][f"comparison|{model}|{condition}"]
            bits = run["match_bits"]
            if (
                run["dataset"] != "powercodebench_2000"
                or run["model"] != model
                or run["condition"] != condition
                or len(bits) != len(item_ids)
                or set(bits) - {"0", "1"}
            ):
                raise ValueError(f"invalid compact match vector: {model}|{condition}")
            arm_bits.append(bits)
        count = 0
        for index, item_id in enumerate(item_ids):
            if arm_bits[0][index] == "1" and arm_bits[1][index] == "1":
                support.add((model, item_id))
                count += 1
        by_model[model] = count
        by_tier[tier] += count

    common_rows = []
    for row in rows:
        if row["frame"] not in ("main_P", "contrast_S"):
            continue
        index = item_index.get(row["item_id"])
        if index is None:
            raise ValueError(f"sample item absent from compact axis: {row['item_id']}")
        own_condition = "C_FDRS" if row["frame"] == "main_P" else "A_FX"
        own_run = compact["runs"][
            f"comparison|{row['model']}|{own_condition}"]
        if own_run["match_bits"][index] != "1":
            raise ValueError(f"sample was not matched in its own arm: {row['key']}")
        if (row["model"], row["item_id"]) in support:
            common_rows.append(row)

    a_population = sum(
        n
        for models in manifest["population_counts"]["A_FX"].values()
        for n in models.values()
    )
    c_population = sum(
        n
        for tier, models in manifest["population_counts"]["C_FDRS"].items()
        if tier != "API"
        for n in models.values()
    )
    population = {
        "n_pairs_matched_both_conditions": len(support),
        "by_tier": by_tier,
        "by_model": by_model,
        "A_FX_matched_population": a_population,
        "C_FDRS_matched_population": c_population,
        "share_of_A_FX_pool_pct": round(100 * len(support) / a_population, 2),
        "share_of_C_FDRS_pool_pct": round(100 * len(support) / c_population, 2),
    }
    return common_rows, population


def common_support_stats(common_rows: list[dict], population: dict) -> dict:
    """Rebuild the sample-domain and population-standardised S14 contrasts."""
    reference = select(common_rows, frames=("main_P",))
    treatment = select(common_rows, frames=("contrast_S",))

    sample_domain = {}
    for key, frame_rows in (
        ("main_P_on_common_support", reference),
        ("contrast_S_on_common_support", treatment),
    ):
        stats = far_stats(frame_rows)
        stats["boot95_weighted_pct"] = bootstrap_far_ci(
            frame_rows, mode="hybrid", weighted=True)
        sample_domain[key] = stats
    sample_delta = difference_stats(treatment, reference)
    for weighted, point_field in (
        (False, "delta_unweighted_pp"), (True, "delta_weighted_pp")):
        def raw_far(frame_rows: list[dict]) -> float:
            denominator = sum(
                row["weight"] if weighted else 1.0
                for row in frame_rows if row["label"] != "item-defect")
            numerator = sum(
                row["weight"] if weighted else 1.0
                for row in frame_rows if row["label"] == "contaminated")
            return 100 * numerator / denominator
        sample_delta[point_field] = round(
            raw_far(treatment) - raw_far(reference), 2)
    sample_domain["delta_S_minus_P"] = sample_delta

    tiers = ("T2", "T3", "T4")
    tier_population = population["by_tier"]
    standardised = {}
    for weighted in (False, True):
        treatment_far = standardised_far(
            treatment, tiers=tiers, tier_population=tier_population,
            weighted=weighted)
        reference_far = standardised_far(
            reference, tiers=tiers, tier_population=tier_population,
            weighted=weighted)
        key = f"T2_T3_T4|{'weighted' if weighted else 'unweighted'}"
        standardised[key] = {
            "standardisation_weights": {
                tier: round(
                    tier_population[tier]
                    / sum(tier_population[value] for value in tiers), 4)
                for tier in tiers
            },
            "far_contrast_S_pct": round(treatment_far, 2),
            "far_main_P_pct": round(reference_far, 2),
            "delta_S_minus_P_pp": round(treatment_far - reference_far, 2),
            "boot95_delta_pp": bootstrap_standardised_difference_ci(
                treatment, reference, tiers=tiers,
                tier_population=tier_population, weighted=weighted),
        }
    return {
        "population": population,
        "audit_sample_domain": sample_domain,
        "tier_standardised": standardised,
    }


def compare_fields(got: dict, expected: dict, fields: tuple[str, ...],
                   where: str, mismatches: list[dict]) -> None:
    for field in fields:
        if got.get(field) != expected.get(field):
            mismatches.append({
                "where": where,
                "field": field,
                "expected": expected.get(field),
                "got": got.get(field),
            })


def recompute() -> dict:
    labels = load(AUDIT / "t4_final_labels_v2.json")
    manifest = load(AUDIT / "e1_sample_manifest.json")
    automatic = load(AUDIT / "t4_auto_results.json")
    design = load(AUDIT / "t4_far_weighted.json")
    tolerance = load(AUDIT / "t4_tolerance_sensitivity.json")
    compact = load(PRIMARY_COMPACT)

    manifest_model_tiers = {
        model: tier
        for tier, models in manifest["population_counts"]["C_FDRS"].items()
        if tier != "API"
        for model in models
    }
    model_tier_mapping_valid = (
        manifest_model_tiers == OPEN_MODEL_TIERS
        and all(
            pointer["tier"] == OPEN_MODEL_TIERS.get(pointer["model"])
            for frame, samples in manifest["samples"].items()
            if frame != "api_T5"
            for pointer in samples
        )
    )
    if not model_tier_mapping_valid:
        raise ValueError("sample-manifest model-to-tier mapping changed")

    flag_by_key = {
        sample_key(item["pointer"]): bool(item["any_flag"])
        for item in automatic["items"]
    }
    tight_by_key = {
        item["key"]: bool(item["would_fail_tight"])
        for group in ("contaminated", "noncontaminated")
        for item in tolerance[group]
    }

    rows = []
    missing = []
    for frame, samples in manifest["samples"].items():
        if frame not in ("main_P", "api_T5", "contrast_S"):
            continue
        for pointer in samples:
            pointer = dict(pointer)
            pointer["frame"] = frame
            key = sample_key(pointer)
            label = labels.get(key)
            cell = design["cells"].get(design_cell(pointer))
            if (
                label is None
                or key not in flag_by_key
                or cell is None
                or (label["label"] != "item-defect" and key not in tight_by_key)
            ):
                missing.append(key)
                continue
            rows.append({
                **pointer,
                "key": key,
                "label": label["label"],
                "flag": flag_by_key[key],
                "would_fail_tight": tight_by_key.get(key, False),
                # Reconstruct the design weight from its integer components;
                # the ``weight`` field in the frozen table is display-rounded.
                "weight": cell["N_pop"] / cell["n_sample"],
                "design_n_sample": cell["n_sample"],
                "design_cell": design_cell(pointer),
            })

    frames = {
        "main_P": frame_far_stats(select(rows, frames=("main_P",))),
        "api_T5": frame_far_stats(select(rows, frames=("api_T5",))),
        "C_combined": frame_far_stats(
            select(rows, frames=("main_P", "api_T5"))),
        "contrast_S": frame_far_stats(select(rows, frames=("contrast_S",))),
        "main_P_T2toT4": frame_far_stats([
            row for row in select(rows, frames=("main_P",))
            if row["tier"] in ("T2", "T3", "T4")],
            full_sensitivity=False),
    }
    tiers = {}
    for frame, tier_values in (("main_P", ("T1", "T2", "T3", "T4")),
                               ("contrast_S", ("T2", "T3", "T4"))):
        for tier in tier_values:
            tiers[f"{frame}|{tier}"] = tier_far_stats(
                select(rows, frames=(frame,), tier=tier))
    tiers["api_T5|API"] = tier_far_stats(
        select(rows, frames=("api_T5",)))

    contrast_rows = select(rows, frames=("contrast_S",))
    main_rows = select(rows, frames=("main_P",))
    combined_rows = select(rows, frames=("main_P", "api_T5"))
    aligned_main_rows = [
        row for row in main_rows if row["tier"] in ("T2", "T3", "T4")]
    contrasts = {
        "S_minus_P": difference_stats(contrast_rows, main_rows),
        "S_minus_Ccombined": difference_stats(contrast_rows, combined_rows),
        "S_minus_P_tier_aligned_T2toT4": difference_stats(
            contrast_rows, aligned_main_rows),
    }

    api_models = {
        model: far_stats(select(rows, frames=("api_T5",), model=model))
        for model in sorted({row["model"] for row in rows if row["frame"] == "api_T5"})
    }

    gate = {
        "main_P": gate_stats(select(rows, frames=("main_P",))),
        "api_T5": gate_stats(select(rows, frames=("api_T5",))),
        "C_combined": gate_stats(select(rows, frames=("main_P", "api_T5"))),
        "contrast_S": gate_stats(select(rows, frames=("contrast_S",))),
        "ALL_610": gate_stats(rows),
    }
    tightened = {
        "main_P": tightened_stats(select(rows, frames=("main_P",))),
        "api_T5": tightened_stats(select(rows, frames=("api_T5",))),
        "C_combined": tightened_stats(select(rows, frames=("main_P", "api_T5"))),
        "contrast_S": tightened_stats(select(rows, frames=("contrast_S",))),
    }
    common_rows, common_population = recover_common_support(
        rows, manifest, compact)
    common_support = common_support_stats(common_rows, common_population)

    mismatches: list[dict] = []
    far_fields = (
        "n_sampled", "n_eff", "n_contaminated", "far_unweighted_pct",
        "wilson95_pct", "far_weighted_pct", "boot95_weighted_pct")
    for key, got in frames.items():
        fields = list(far_fields)
        expected = design["far_by_frame"][key]
        for field in ("boot95_unweighted_pct", "boot95_weighted_cell_pct",
                      "boot95_weighted_collapsed_pct", "delta_w_minus_u_pp"):
            if field in expected:
                fields.append(field)
        compare_fields(got, expected, tuple(fields),
                       f"far_by_frame.{key}", mismatches)
    for key, got in tiers.items():
        compare_fields(got, design["far_by_tier"][key], far_fields,
                       f"far_by_tier.{key}", mismatches)
    for key, got in api_models.items():
        compare_fields(got, design["far_api_by_model"][key],
                       ("n_eff", "n_contaminated", "far_unweighted_pct",
                        "far_weighted_pct"),
                       f"far_api_by_model.{key}", mismatches)
    for key, got in contrasts.items():
        compare_fields(
            got, design["separately_sampled_contrast"][key],
            ("delta_unweighted_pp", "boot95_unweighted_pp",
             "delta_weighted_pp", "boot95_weighted_pp", "design"),
            f"separately_sampled_contrast.{key}", mismatches)
    for key, got in gate.items():
        expected_key = "ALL_610" if key == "ALL_610" else key
        compare_fields(got,
                       design["gate_operating_characteristics"]["defect_excluded"]
                       [expected_key],
                       ("n", "tp", "fp", "fn", "tn", "prevalence_pct",
                        "flag_rate_pct", "sensitivity_pct", "specificity_pct",
                        "precision_ppv_pct", "npv_pct", "sens_wilson95_pct",
                        "spec_wilson95_pct", "ppv_wilson95_pct",
                        "npv_wilson95_pct", "balanced_accuracy_pct"),
                       f"gate.{key}", mismatches)
    for key, got in tightened.items():
        compare_fields(got,
                       design["tolerance_tightened_far"]["by_frame"][key],
                       ("n_eff_before", "n_eff_after", "contaminated_before",
                        "contaminated_after", "contaminated_rejected_pct",
                        "legitimate_rejected_pct", "far_unweighted_pct",
                        "far_weighted_pct", "boot95_weighted_pct"),
                       f"tolerance.{key}", mismatches)

    common_expected = design["common_support"]
    compare_fields(
        common_support["population"], common_expected["population"],
        ("n_pairs_matched_both_conditions", "by_tier", "by_model",
         "share_of_A_FX_pool_pct", "share_of_C_FDRS_pool_pct"),
        "common_support.population", mismatches)
    sample_fields = (
        "n_sampled", "n_eff", "n_contaminated", "far_unweighted_pct",
        "wilson95_pct", "far_weighted_pct", "boot95_weighted_pct",
        "est_population_nondefect")
    for key in ("main_P_on_common_support", "contrast_S_on_common_support"):
        compare_fields(
            common_support["audit_sample_domain"][key],
            common_expected["audit_sample_domain"][key], sample_fields,
            f"common_support.audit_sample_domain.{key}", mismatches)
    compare_fields(
        common_support["audit_sample_domain"]["delta_S_minus_P"],
        common_expected["audit_sample_domain"]["delta_S_minus_P"],
        ("delta_unweighted_pp", "boot95_unweighted_pp",
         "delta_weighted_pp", "boot95_weighted_pp"),
        "common_support.audit_sample_domain.delta_S_minus_P", mismatches)
    for key in ("T2_T3_T4|unweighted", "T2_T3_T4|weighted"):
        compare_fields(
            common_support["tier_standardised"][key],
            common_expected["tier_standardised"][key],
            ("standardisation_weights", "far_contrast_S_pct",
             "far_main_P_pct", "delta_S_minus_P_pp", "boot95_delta_pp"),
            f"common_support.tier_standardised.{key}", mismatches)

    return {
        "schema_version": "1.2",
        "inputs": {
            "final_labels": "audit/t4_final_labels_v2.json",
            "sample_manifest": "audit/e1_sample_manifest.json",
            "automatic_checks": "audit/t4_auto_results.json",
            "design_cells": "audit/t4_far_weighted.json#cells",
            "tolerance_replay": "audit/t4_tolerance_sensitivity.json",
            "primary_item_outcomes": (
                "results/raw/main_experiment/primary_outcomes_compact.json"),
        },
        "scope": (
            "Point estimates, Wilson intervals, and deterministic stratified "
            "bootstrap intervals for SM S13--S14 and main Table 10, including "
            "the common-support population-standardised contrast. Paired SM "
            "S15 is independently rebuilt by pair_mcnemar.py."),
        "bootstrap": {
            "B": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "rng": "Python random.Random(seed).choices",
            "ci_order_statistics": [
                int(0.025 * BOOTSTRAP_B), int(0.975 * BOOTSTRAP_B)],
            "primary_frame_mode": (
                "hybrid: design cells with n_sample>=2 retained; singleton "
                "cells collapsed by frame x tier/model"),
            "tier_mode": "strict design-cell resampling",
            "collapsed_sensitivity_mode": "frame x tier/model resampling",
            "common_support_standardisation": (
                "independent audit frames restricted to two-arm match support; "
                "hybrid-stratum joint bootstrap, contrast_S drawn before main_P; "
                "direct standardisation to the two-arm population tier mix"),
        },
        "n_rows": len(rows),
        "missing_inputs": missing,
        "far_by_frame": frames,
        "far_by_tier": tiers,
        "far_api_by_model": api_models,
        "separately_sampled_contrast": contrasts,
        "gate_operating_characteristics": gate,
        "tolerance_tightened_far": tightened,
        "common_support": common_support,
        "validation": {
            "status": "pass" if not mismatches and len(rows) == 610 else "fail",
            "model_tier_mapping_matches_frozen_design": model_tier_mapping_valid,
            "mismatches": mismatches,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=AUDIT / "validity_tables_recomputed.json")
    args = parser.parse_args()
    report = recompute()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["validation"], indent=2))
    print(f"[saved] {args.out}")
    return 0 if report["validation"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
