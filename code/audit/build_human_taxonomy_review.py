#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Builds the human review layer of the Failure Anatomy taxonomy from the two
# engineers' completed booklets.
#
# Two practising power-systems engineers labelled the same frozen 100-record
# sample independently, under the rubric the preliminary expert pass used. This
# script emits their two label sets in the judge schema, an agreement record
# (Cohen's kappa, the disagreement list), and a sensitivity analysis showing
# how every reported quantity moves across all 2^k resolutions of the
# disagreements.
#
# Disagreements are reported rather than adjudicated into a single "final
# label", and the sensitivity table is what supports the numbers quoted in the
# manuscript.
#
# It is the recovery half of a reusable pair: build_taxonomy_booklet.py renders
# a sample into blind booklets, this reads the completed ones back. Point
# --pattern at a different booklet naming to run the same round elsewhere.
#
# Usage:  python3 code/audit/build_human_taxonomy_review.py --booklets DIR
#         python3 code/audit/build_human_taxonomy_review.py --booklets DIR \
#             --pattern 'booklet_*.md'
# --------------------------------------------------------------------------
"""Emit the human review layer + agreement + sensitivity for the taxonomy."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "audit/failure_taxonomy/revision_sample"
OUT = ROOT / "audit/failure_taxonomy/human_review"

CANON = {"C1": "C1_api_contract", "C2": "C2_param_misuse",
         "C3": "C3_workflow_logic", "C4": "C4_numerical_extraction",
         "C5": "C5_format_env", "ambiguous": "ambiguous_item_defect"}
CONF = {"high": 0.95, "medium": 0.75, "low": 0.5}
CATS = list(CANON.values())

REC_RE = re.compile(r"^##\s+(\d+)\s*/\s*(\d+)\s*·\s*`([^`]+)`", re.M)


def parse_booklet(path: Path) -> tuple[str, list[dict]]:
    text = path.read_text(encoding="utf-8")
    who = re.search(r"\*\*Annotator\*\*:\s*([^\s　]+)", text)
    parts = REC_RE.split(text)
    recs = []
    for i in range(1, len(parts), 4):
        sid, body = parts[i + 2], parts[i + 3]

        def g(pat):
            m = re.search(pat, body, re.M)
            return m.group(1).strip() if m else ""

        raw_label = g(r"^-\s*\*\*Category\*\*:\s*`([^`]+)`")
        raw_conf = g(r"^-\s*\*\*Confidence\*\*:\s*`([^`]+)`")
        if raw_label not in CANON or raw_conf not in CONF:
            raise SystemExit(f"{path.name} item {sid}: unreadable judgement "
                             f"(category {raw_label!r}, confidence {raw_conf!r})")
        recs.append({
            "sample_id": sid,
            "label": CANON[raw_label],
            "confidence": CONF[raw_conf],
            "rationale": g(r"^-\s*\*\*Rationale\*\*:\s*(.*)$"),
            "primary_evidence": g(r"^-\s*\*\*Evidence\*\*:\s*(.*)$"),
        })
    return (who.group(1) if who else path.stem), recs


def kappa(a: dict, b: dict) -> tuple[float, float]:
    ids = sorted(set(a) & set(b))
    n = len(ids)
    po = sum(1 for i in ids if a[i] == b[i]) / n
    pe = sum((sum(1 for i in ids if a[i] == c) / n) *
             (sum(1 for i in ids if b[i] == c) / n) for c in CATS)
    return (po - pe) / (1 - pe), po


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--booklets", required=True,
                    help="directory holding the two completed booklets")
    ap.add_argument("--pattern", default="annotation_reviewer_*_en.md",
                    help="booklet filename glob within that directory")
    args = ap.parse_args()

    books = sorted(Path(args.booklets).glob(args.pattern))
    if len(books) != 2:
        raise SystemExit(f"expected 2 booklets matching {args.pattern!r}, "
                         f"found {len(books)}")

    meta = {r["sample_id"]: r for r in
            json.loads((SAMPLE / "adjudication_v2.json").read_text())["records"]}
    rubric = json.loads((SAMPLE / "judge_a.json").read_text())["rubric"]

    OUT.mkdir(parents=True, exist_ok=True)
    labels, reviewers = {}, []
    for p in books:
        who, recs = parse_booklet(p)
        if len(recs) != len(meta):
            raise SystemExit(f"{p.name}: {len(recs)} labelled items, "
                             f"sample holds {len(meta)}")
        reviewers.append(who)
        labels[who] = {r["sample_id"]: r["label"] for r in recs}
        (OUT / f"reviewer_{who}.json").write_text(json.dumps({
            "reviewer_type": "Practising power-systems engineer; independent, "
                             "rubric-constrained review of the frozen sample",
            "reviewer_id": who,
            "rubric": rubric,
            "records": sorted(recs, key=lambda r: r["sample_id"]),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote human_review/reviewer_{who}.json ({len(recs)} records)")

    a, b = reviewers
    A, B = labels[a], labels[b]
    k, po = kappa(A, B)
    dis = sorted(i for i in A if A[i] != B[i])

    # --- sensitivity: every resolution of the disagreements -----------------
    base = {i: A[i] for i in A if A[i] == B[i]}
    rows = []
    for combo in itertools.product(*[[A[i], B[i]] for i in dis]):
        D = dict(base) | dict(zip(dis, combo))
        c = Counter(D.values())
        nd = sum(v for kk, v in c.items() if kk != "ambiguous_item_defect")
        c12 = c["C1_api_contract"] + c["C2_param_misuse"]
        t4 = Counter(D[i] for i in D if meta[i]["model_tier"] == "tier4")
        rows.append({
            "resolution": dict(zip(dis, combo)),
            "C1": c["C1_api_contract"], "C3": c["C3_workflow_logic"],
            "C1_plus_C2": c12, "classifiable": nd,
            "C1_plus_C2_pct": round(100 * c12 / nd, 1),
            "tier4_C1": t4["C1_api_contract"], "tier4_C3": t4["C3_workflow_logic"],
            "C1_is_largest_overall": c.most_common(1)[0][0] == "C1_api_contract",
        })

    def span(key):
        vs = [r[key] for r in rows]
        return [min(vs), max(vs)]

    per_tier = {}
    for who, D in ((a, A), (b, B)):
        per_tier[who] = {}
        for t in sorted({m["model_tier"] for m in meta.values()}):
            per_tier[who][t] = dict(Counter(D[i] for i in D
                                            if meta[i]["model_tier"] == t))
    per_cond = {}
    for who, D in ((a, A), (b, B)):
        per_cond[who] = {}
        for cnd in sorted({m["condition"] for m in meta.values()}):
            per_cond[who][cnd] = dict(Counter(D[i] for i in D
                                              if meta[i]["condition"] == cnd))

    preliminary = {i: m["final_label"] for i, m in meta.items()}
    pk_a, ppo_a = kappa(A, preliminary)
    pk_b, ppo_b = kappa(B, preliminary)

    agreement = {
        "schema_version": 1,
        "scope": "Two practising power-systems engineers independently labelled "
                 f"the frozen {len(meta)}-record Failure Anatomy sample under "
                 "the rubric of the preliminary expert pass. Disagreements are "
                 "reported, not adjudicated; the sensitivity block shows every "
                 "reported quantity across all resolutions of them.",
        "reviewers": reviewers,
        "n_items": len(meta),
        "agreement_pct": round(100 * po, 1),
        "cohens_kappa": round(k, 3),
        "n_disagreements": len(dis),
        "disagreements": [
            {"sample_id": i, reviewers[0]: A[i], reviewers[1]: B[i],
             "preliminary_expert_final_label": preliminary[i],
             "model_tier": meta[i]["model_tier"], "condition": meta[i]["condition"]}
            for i in dis
        ],
        "per_reviewer_marginals": {w: dict(Counter(labels[w].values())) for w in reviewers},
        "per_reviewer_tier_tables": per_tier,
        "per_reviewer_condition_tables": per_cond,
        "vs_preliminary_expert_pass": {
            reviewers[0]: {"agreement_pct": round(100 * ppo_a, 1), "cohens_kappa": round(pk_a, 3)},
            reviewers[1]: {"agreement_pct": round(100 * ppo_b, 1), "cohens_kappa": round(pk_b, 3)},
            "note": "The preliminary expert pass is retained for provenance "
                    "only; the formal expert labels supersede it for every "
                    "reported quantity.",
        },
        "sensitivity_over_disagreements": {
            "n_resolutions": len(rows),
            "invariant": {
                "C1_is_largest_class_overall": all(r["C1_is_largest_overall"] for r in rows),
                "tier4_C3_exceeds_C1": all(r["tier4_C3"] > r["tier4_C1"] for r in rows),
            },
            "ranges": {
                "C1": span("C1"), "C3": span("C3"),
                "C1_plus_C2": span("C1_plus_C2"),
                "C1_plus_C2_pct_of_classifiable": span("C1_plus_C2_pct"),
                "tier4_C1": span("tier4_C1"), "tier4_C3": span("tier4_C3"),
            },
            "resolutions": rows,
        },
    }
    (OUT / "agreement.json").write_text(
        json.dumps(agreement, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote human_review/agreement.json")
    print(f"   kappa={k:.3f}  agreement={po:.1%}  disagreements={len(dis)}")
    print(f"   C1 {span('C1')}  C1+C2% {span('C1_plus_C2_pct')}  "
          f"tier4 C1 {span('tier4_C1')} vs C3 {span('tier4_C3')}")


if __name__ == "__main__":
    main()
