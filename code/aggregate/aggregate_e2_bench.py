#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Repository copy of the historical pipeline module
# scripts/aggregate/aggregate_e2_bench.py. The minimal raw E2 run set is now
# archived under results/raw/e2_transfer/. For a repository-local rebuild with
# hash validation and the C-recal arm included, use the release wrapper
# code/aggregate/aggregate_e2_transfer.py.
# --------------------------------------------------------------------------
"""Aggregate the E2 cross-backend transfer mini-bench (OpenDSS + PyPSA).

Disk-scanning aggregator (idempotent, model/condition discovery from files, not
from in-process state) for ``probe_eval_results/e2_<backend>_bench[_<tag>]/
<model>/benchmark_results_cond<COND>.json``.

Produces, per (backend, model, condition):
  * overall exec_rate / accuracy (with 95% Wilson CI) / accuracy_of_executed
    (Acc|Exec)
  * the same triple stratified by task family and by difficulty level
and, per (backend, model), paired bootstrap CIs over the 90 shared items for
the E2 §1.2 criteria (seed recorded in the output):
  * criterion 2: accuracy(C) - accuracy(A)
  * criterion 3: accuracy(C) - accuracy(R_sem), plus the equal-cost axis
    accuracy(C) - accuracy(RsemB)

A difference CI covering 0 is reported as "no significant difference detected
at n=90" only; it is never evidence of non-inferiority/equivalence (that would
require a prespecified margin + TOST, which E2 does not have).

An optional baseline run tag (e.g. the pre-B1' round) is reported side by side.

Usage:
    python scripts/aggregate/aggregate_e2_bench.py \
        --run-tag _v2 --baseline-tag "" \
        --out probe_eval_results/e2_bench_v2_aggregate.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import OrderedDict

BACKENDS = ("opendss", "pypsa")
CONDITIONS = ("A", "Rsem", "RsemB", "C")


def wilson_ci(k: int, n: int, z: float = 1.959964) -> "list[float] | None":
    """95% Wilson score interval for a binomial proportion k/n."""
    if not n:
        return None
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [round(center - half, 4), round(center + half, 4)]


def run_dir(root: str, backend: str, tag: str) -> str:
    return os.path.join(root, "probe_eval_results", f"e2_{backend}_bench{tag}")


def scan_run(root: str, backend: str, tag: str) -> "OrderedDict[str, dict]":
    """Discover {model: {cond: parsed_result}} by scanning the run directory."""
    base = run_dir(root, backend, tag)
    found: "OrderedDict[str, dict]" = OrderedDict()
    if not os.path.isdir(base):
        return found
    for model in sorted(os.listdir(base)):
        mdir = os.path.join(base, model)
        if not os.path.isdir(mdir):
            continue
        for cond in CONDITIONS:
            path = os.path.join(mdir, f"benchmark_results_cond{cond}.json")
            if os.path.isfile(path):
                with open(path) as fh:
                    found.setdefault(model, {})[cond] = json.load(fh)
    return found


def triple(block: dict) -> dict:
    """exec_rate / accuracy / Acc|Exec from a summary block."""
    total = block.get("total", 0)
    executed = block.get("executed", block.get("n_executed", 0))
    matched = block.get("matched", block.get("n_matched", 0))
    return {
        "n": total,
        "n_exec": executed,
        "n_match": matched,
        "exec_rate": round(executed / total, 4) if total else None,
        "accuracy": round(matched / total, 4) if total else None,
        "accuracy_wilson_ci95": wilson_ci(matched, total),
        "acc_given_exec": round(matched / executed, 4) if executed else None,
    }


def summarize(result: dict) -> dict:
    s = result["summary"]
    overall = triple(
        {"total": s["total"], "executed": s["n_executed"], "matched": s["n_matched"]}
    )
    return {
        "overall": overall,
        "per_task": {k: triple(v) for k, v in s.get("per_task", {}).items()},
        "per_difficulty": {
            k: triple(v) for k, v in s.get("per_difficulty", {}).items()
        },
        "error_distribution": s.get("error_distribution", {}),
        "timestamp": s.get("timestamp"),
    }


def item_vectors(result: dict) -> "OrderedDict[str, dict]":
    out: "OrderedDict[str, dict]" = OrderedDict()
    for it in result["item_results"]:
        out[it["item_id"]] = {
            "executed": bool(it.get("executed")),
            "match": bool(it.get("match")),
            "task": it.get("task"),
            "difficulty": it.get("difficulty_level"),
        }
    return out


def paired_bootstrap(
    ids: list, a_vec: dict, b_vec: dict, key: str, n_boot: int, seed: int
) -> dict:
    """CI for mean(a[key]) - mean(b[key]) over paired items (a = treatment)."""
    diffs = [float(a_vec[i][key]) - float(b_vec[i][key]) for i in ids]
    n = len(diffs)
    point = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)
    return {
        "n_pairs": n,
        "delta": round(point, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_discordant_pos": n_pos,
        "n_discordant_neg": n_neg,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--run-tag", default="_v2")
    ap.add_argument("--baseline-tag", default="")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = {
        "run_tag": args.run_tag,
        "baseline_tag": args.baseline_tag,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "backends": {},
    }

    for backend in BACKENDS:
        run = scan_run(args.root, backend, args.run_tag)
        base = scan_run(args.root, backend, args.baseline_tag)
        b_out = {"run_dir": run_dir(args.root, backend, args.run_tag), "models": {}}
        for model, conds in run.items():
            m_out = {"conditions": {}, "baseline_conditions": {}, "paired": {}}
            for cond, res in conds.items():
                m_out["conditions"][cond] = summarize(res)
            for cond, res in base.get(model, {}).items():
                m_out["baseline_conditions"][cond] = summarize(res)["overall"]

            # realized injection cost = prompt tokens above the bare-A prompt
            a_tok = (
                conds.get("A", {})
                .get("summary", {})
                .get("prompt_token_stats", {})
                .get("avg")
            )
            m_out["prompt_tokens_avg"] = {
                c: r["summary"].get("prompt_token_stats", {}).get("avg")
                for c, r in conds.items()
            }
            if a_tok:
                m_out["injected_tokens_avg"] = {
                    c: (v - a_tok) if v is not None else None
                    for c, v in m_out["prompt_tokens_avg"].items()
                }

            vecs = {c: item_vectors(r) for c, r in conds.items()}
            if "C" in vecs:
                for ref in ("A", "Rsem", "RsemB"):
                    if ref not in vecs:
                        continue
                    ids = [i for i in vecs["C"] if i in vecs[ref]]
                    for key, label in (("match", "accuracy"), ("executed", "exec")):
                        m_out["paired"][f"C-{ref}::{label}"] = paired_bootstrap(
                            ids, vecs["C"], vecs[ref], key, args.n_boot, args.seed
                        )
                    for diff in ("D1_basic", "D3_semantic"):
                        sub = [i for i in ids if vecs["C"][i]["difficulty"] == diff]
                        if not sub:
                            continue
                        m_out["paired"][f"C-{ref}::accuracy@{diff}"] = paired_bootstrap(
                            sub, vecs["C"], vecs[ref], "match", args.n_boot, args.seed
                        )
            b_out["models"][model] = m_out
        report["backends"][backend] = b_out

    # ---- console tables -------------------------------------------------
    print("\n=== E2 mini-bench aggregate (run tag %r) ===" % args.run_tag)
    print("exec_rate / accuracy / Acc|Exec   (N=90 per cell)\n")
    hdr = f"{'backend':<8} {'model':<34} " + " ".join(f"{c:<24}" for c in CONDITIONS)
    print(hdr)
    for backend, b in report["backends"].items():
        for model, m in b["models"].items():
            cells = []
            for cond in CONDITIONS:
                blk = m["conditions"].get(cond)
                if not blk:
                    cells.append(f"{'(missing)':<24}")
                    continue
                o = blk["overall"]
                cells.append(
                    f"{o['exec_rate']:.3f} / {o['accuracy']:.3f} / {o['acc_given_exec'] if o['acc_given_exec'] is not None else float('nan'):.3f}".ljust(
                        24
                    )
                )
            print(f"{backend:<8} {model:<34} " + " ".join(cells))

    if args.baseline_tag is not None:
        print("\n--- baseline (tag %r) accuracy for reference ---" % args.baseline_tag)
        for backend, b in report["backends"].items():
            for model, m in b["models"].items():
                if not m["baseline_conditions"]:
                    continue
                cells = []
                for cond in CONDITIONS:
                    o = m["baseline_conditions"].get(cond)
                    cells.append(
                        (
                            f"{o['exec_rate']:.3f} / {o['accuracy']:.3f} / "
                            f"{(o['acc_given_exec'] if o['acc_given_exec'] is not None else float('nan')):.3f}"
                        ).ljust(24)
                        if o
                        else f"{'(missing)':<24}"
                    )
                print(f"{backend:<8} {model:<34} " + " ".join(cells))

    print("\n--- realized injection cost (avg prompt tokens; delta vs bare A) ---")
    for backend, b in report["backends"].items():
        for model, m in b["models"].items():
            tok = m.get("prompt_tokens_avg", {})
            inj = m.get("injected_tokens_avg", {})
            cells = " ".join(
                f"{c}={tok.get(c)}(+{inj.get(c)})" if tok.get(c) is not None else f"{c}=--"
                for c in CONDITIONS
            )
            print(f"{backend:<8} {model:<34} {cells}")

    print("\n--- paired bootstrap (95%% CI, %d reps) ---" % args.n_boot)
    for backend, b in report["backends"].items():
        for model, m in b["models"].items():
            for name, st in m["paired"].items():
                print(
                    f"{backend:<8} {model:<34} {name:<20} "
                    f"delta={st['delta']:+.4f} CI=[{st['ci95'][0]:+.4f},{st['ci95'][1]:+.4f}] "
                    f"excl0={st['excludes_zero']} (+{st['n_discordant_pos']}/-{st['n_discordant_neg']})"
                )

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
