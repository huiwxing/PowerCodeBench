# --------------------------------------------------------------------------
# Repository copy of audit/pair_mcnemar.py, with path constants adapted to
# this repository's layout. It runs here against the archived artifacts; see
# ARTIFACT_INDEX.md.
# --------------------------------------------------------------------------
"""E1-P paired-contrast endgame: pair-level 2x2 + exact McNemar.

Consumes the frozen E1-P chain artifacts
  e1p_pair_manifest.json   120 pairs + 8 overlap sides carrying frozen-610 labels
  e1p_auto_results.json    232 fresh sides (i-indexed, pointer identity)
  e1p_adjudication.json    final labels, flagged queue (137)
  e1p_census.json          final labels, all-pass queue (95)
  e1p_refute.json          optional refutation patch over physical-error labels

and, when the T4 extension round artifacts are present (e1p_ext_*),
the merged 170-pair analysis mandated by E1 doc s P.13 (extension totals
primary, original 120 bracketed).  Emits e1p_mcnemar.json plus a stdout
table.  Protocol: E1 doc s P.7-4/5 --
contamination = final label 'physical-error'; suspected-item-defect handled in
two prespecified versions (included-as-clean / whole-pair excluded); exact
two-sided McNemar on discordant pairs; pair bootstrap CI (seed 20260801).
"""

import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "audit"  # frozen audit-chain artifacts
SEED = 20260801
N_BOOT = 10_000

FROZEN_LABEL_MAP = {
    "contaminated": "physical-error",
    "equivalent": "equivalent-path",
    "clean-autopass": "clean",
}
OPEN_MODEL_TIERS = {
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


def _load(name, required=True):
    p = HERE / name
    if not p.exists():
        if required:
            raise SystemExit(f"missing required artifact: {p}")
        return None
    return json.loads(p.read_text())


def _side_key(pair_id, condition):
    """Canonical fresh-side key.  Conditions appear as C_FDRS / condC_FDRS
    etc. across artifacts; collapse to the C/A arm."""
    arm = "C" if condition.lstrip("cond").startswith("C") else "A"
    return (pair_id, arm)


def side_labels(auto_name="e1p_auto_results.json",
                adj_name="e1p_adjudication.json",
                census_name="e1p_census.json",
                refute_name="e1p_refute.json"):
    """Final label per fresh side, keyed by (pair_id, arm).

    The adjudication and census files carry queue-local `i` indices, so the
    only safe join is the (pair_id, condition) identity, which is unique per
    fresh side (120 pairs x 2 arms - 8 frozen-610 overlap sides = 232).
    """
    auto = _load(auto_name)
    fresh = {_side_key(it["pointer"]["pair_id"], it["pointer"]["condition"])
             for it in auto["items"]}
    if len(fresh) != len(auto["items"]):
        raise SystemExit("duplicate (pair_id, arm) among fresh sides")

    labels, source = {}, {}
    for block, queue in ((adj_name, "flagged"), (census_name, "census")):
        data = _load(block)
        rows = data.get("final") or data.get("judgments")
        for r in rows:
            key = _side_key(r["pair_id"], r["condition"])
            if key not in fresh:
                raise SystemExit(f"{queue} labels unknown side {key}")
            if key in labels:
                raise SystemExit(f"side {key} labelled by both queues")
            labels[key] = r["label"]
            source[key] = queue
    missing = fresh - set(labels)
    if missing:
        raise SystemExit(f"{len(missing)} sides unlabelled, e.g. {sorted(missing)[:5]}")

    refute = _load(refute_name, required=False)
    n_overturned = 0
    if refute:
        for v in refute["verdicts"]:
            key = _side_key(v["pair_id"], v["condition"])
            if labels.get(key) != "physical-error":
                raise SystemExit(f"refute verdict targets non-PE side {key}")
            if v["verdict"] == "OVERTURNED_EQUIVALENT":
                labels[key] = "equivalent-path"
                n_overturned += 1
            elif v["verdict"] == "OVERTURNED_DEFECT":
                labels[key] = "suspected-item-defect"
                n_overturned += 1
            elif v["verdict"] != "UPHELD":
                raise SystemExit(f"unknown refute verdict {v['verdict']!r}")
    return labels, source, refute is not None, n_overturned


def pair_table(manifest_name="e1p_pair_manifest.json", **label_kw):
    """(pair_id, tier) -> {'C': label, 'A': label} for one round's pairs."""
    manifest = _load(manifest_name)
    manifest_pairs = manifest["pairs"]
    if any(
        OPEN_MODEL_TIERS.get(pair.get("model")) != pair.get("tier")
        for pair in manifest_pairs
    ):
        raise SystemExit(f"{manifest_name}: model-to-tier mapping changed")
    observed_models = {pair["model"] for pair in manifest_pairs}
    expected_models = (
        {model for model, tier in OPEN_MODEL_TIERS.items() if tier == "T4"}
        if manifest_name == "e1p_ext_manifest.json"
        else set(OPEN_MODEL_TIERS)
    )
    if observed_models != expected_models:
        raise SystemExit(f"{manifest_name}: model coverage changed")
    labels, _, refuted, n_overturned = side_labels(**label_kw)

    table = {}
    for (pair_id, arm), lbl in labels.items():
        table.setdefault(pair_id, {})[arm] = lbl
    for s in manifest["skipped_overlap_sides"]:
        pair_id, arm = _side_key(s["pair_id"], s["condition"])
        table.setdefault(pair_id, {})[arm] = FROZEN_LABEL_MAP[s["final_label"]]

    tiers = {p["pair_id"]: p["tier"] for p in manifest_pairs}
    bad = [pid for pid, sides in table.items() if set(sides) != {"C", "A"}]
    if bad or len(table) != len(manifest_pairs):
        raise SystemExit(f"pair completeness violated: n={len(table)}, incomplete={bad[:5]}")
    return table, tiers, refuted, n_overturned


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, j) for j in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def analyse(pairs_ca, tag):
    """pairs_ca: list of (contamC: bool, contamA: bool)."""
    a = sum(1 for x, y in pairs_ca if x and y)
    b = sum(1 for x, y in pairs_ca if x and not y)   # C contaminated only
    c = sum(1 for x, y in pairs_ca if not x and y)   # A contaminated only
    d = sum(1 for x, y in pairs_ca if not x and not y)
    n = len(pairs_ca)
    delta = (c - b) / n * 100  # A-minus-C contamination rate, pp

    rng = random.Random(SEED)
    boot = []
    for _ in range(N_BOOT):
        sample = [pairs_ca[rng.randrange(n)] for _ in range(n)]
        bb = sum(1 for x, y in sample if x and not y)
        cc = sum(1 for x, y in sample if not x and y)
        boot.append((cc - bb) / n * 100)
    boot.sort()
    lo, hi = boot[int(0.025 * N_BOOT)], boot[int(0.975 * N_BOOT)]

    return {
        "version": tag,
        "n_pairs": n,
        "table": {"both": a, "C_only": b, "A_only": c, "neither": d},
        "rate_C_pct": round((a + b) / n * 100, 2),
        "rate_A_pct": round((a + c) / n * 100, 2),
        "delta_A_minus_C_pp": round(delta, 2),
        "delta_ci95_pp": [round(lo, 2), round(hi, 2)],
        "discordant": b + c,
        "mcnemar_exact_p": mcnemar_exact(b, c),
        "power_gate_discordant_ge_15": (b + c) >= 15,
    }


def merged_tables():
    table, tiers, refuted, n_overturned = pair_table()
    ext_present = (HERE / "e1p_ext_adjudication.json").exists()
    if ext_present:
        et, etiers, erefuted, eover = pair_table(
            manifest_name="e1p_ext_manifest.json",
            auto_name="e1p_ext_auto_results.json",
            adj_name="e1p_ext_adjudication.json",
            census_name="e1p_ext_census.json",
            refute_name="e1p_ext_refute.json")
        if set(et) & set(table):
            raise SystemExit("extension pair ids collide with main round")
        table.update(et); tiers.update(etiers)
        refuted = refuted and erefuted
        n_overturned += eover
    return table, tiers, refuted, n_overturned, ext_present


def main():
    table, tiers, refuted, n_overturned, ext_present = merged_tables()
    is_pe = lambda lbl: lbl == "physical-error"
    has_sid = lambda sides: "suspected-item-defect" in sides.values()

    all_pairs = [(is_pe(s["C"]), is_pe(s["A"])) for s in table.values()]
    kept = [(is_pe(s["C"]), is_pe(s["A"])) for s in table.values() if not has_sid(s)]

    results = {
        "meta": {
            "protocol": "E1_validity_audit.md sP.7-4/5",
            "refute_applied": refuted,
            "n_refute_overturned": n_overturned,
            "n_pairs_with_item_defect_side": sum(1 for s in table.values() if has_sid(s)),
            "extension_included": ext_present,
            "seed": SEED,
        },
        "primary_sid_included_as_clean": analyse(all_pairs, "sid_included"),
        "secondary_sid_pairs_excluded": analyse(kept, "sid_excluded"),
        "by_tier": {},
    }
    for tier in ("T2", "T3", "T4"):
        sub = [(is_pe(s["C"]), is_pe(s["A"]))
               for pid, s in table.items() if tiers[pid] == tier]
        results["by_tier"][tier] = analyse(sub, f"{tier}_sid_included")

    # Write next to the caller, never over the frozen audit/e1p_mcnemar.json;
    # reproduce.py diffs this recomputation against the frozen artifact.
    out = Path.cwd() / "e1p_mcnemar.recomputed.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")
    for key in ("primary_sid_included_as_clean", "secondary_sid_pairs_excluded"):
        r = results[key]
        t = r["table"]
        print(f"\n[{key}]  n={r['n_pairs']}")
        print(f"  2x2  both={t['both']}  C_only={t['C_only']}  A_only={t['A_only']}  neither={t['neither']}")
        print(f"  contamination  C={r['rate_C_pct']}%  A={r['rate_A_pct']}%  "
              f"delta(A-C)={r['delta_A_minus_C_pp']}pp  CI95={r['delta_ci95_pp']}")
        print(f"  discordant={r['discordant']}  exact-McNemar p={r['mcnemar_exact_p']:.3g}  "
              f"power-gate(>=15)={'PASS' if r['power_gate_discordant_ge_15'] else 'FAIL'}")


if __name__ == "__main__":
    main()
