# --------------------------------------------------------------------------
# Repository copy of audit/run_t4.py from the frozen experimental pipeline.
# Its inputs are raw per-item run trees too large to ship here, so it does not
# run from this checkout; see ARTIFACT_INDEX.md for the outputs it produced.
# --------------------------------------------------------------------------
"""T4 -- run the five-check automatic layer over the full E1 audit sample.

Thin batch driver for the compute-node run (E1 doc s5 T4 / s8 T4 release).  It
streams the ``main_P`` (400) + ``api_T5`` (60) + ``contrast_S`` (150) = 610
frozen pointers from ``e1_sample_manifest.json``, re-executes and audits each one
with :func:`audit.checkers.audit_item` (the same five checks used for the pilot),
and writes per-item ``{pass/fail/ambiguous}`` verdicts plus evidence to
``t4_auto_results.json``.  It computes *no* FAR and makes *no* final ruling --
its output feeds the dual-judge + refute pass (E1 doc s2.4).

Discipline (E1 doc s7):
  * single-threaded, serial -- one record fetched, audited, released at a time;
    thread caps set below (before numpy/pandapower import) so a compute node does
    not oversubscribe.
  * per-item hard timeout -- inherited from ``audit_item`` (SIGALRM in the T1
    sandbox), so one runaway re-execution cannot stall the batch.
  * checkpointed -- the full results file is written atomically every
    ``--checkpoint-every`` items (default 50), so an interrupted job never loses
    completed work.  Re-running resumes from the checkpoint by default (pointers
    already present in the output are skipped).
"""

from __future__ import annotations

import os

# Thread caps must be set before numpy / pandapower import (single-thread, s7).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import datetime as _dt
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from audit import checkers, records
from audit.checkers import (AMBIG, DEFAULT_TIMEOUT, FAIL, PASS, audit_item)
from audit.records import PROJECT_ROOT, cond_path

# The three formal (non-pilot) sample frames, in run order (E1 doc s4.2).
# Frames are a *parameter* everywhere below (``run_t4(frames=...)`` / ``--frames``)
# so the same driver serves the E1-P supplement round's ``pair_C`` / ``pair_A``
# frames; the constant is only the default, so default behaviour is unchanged.
FRAMES = ("main_P", "api_T5", "contrast_S")
_CHECKS = ("check1_network_mods", "check2_solver_identity",
           "check3_physical_sanity", "check4_extraction", "check5_provenance")


def _atomic_write(out_path: Path, payload: dict) -> None:
    """Write ``payload`` to ``out_path`` via a temp file + rename (crash-safe)."""
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, out_path)


def _ptr_key(ptr: dict) -> tuple:
    """Identity of a sample pointer (frame + model + condition + bench_index)."""
    return (ptr.get("frame"), ptr["model"], ptr["condition"], ptr["bench_index"])


def _fresh_accumulators(frames: tuple = FRAMES) -> Dict[str, Any]:
    dist = {c: {PASS: 0, FAIL: 0, AMBIG: 0} for c in _CHECKS}
    dist_by_frame = {f: {c: {PASS: 0, FAIL: 0, AMBIG: 0} for c in _CHECKS}
                     for f in frames}
    degr = {"n": 0, "gen_net_recovered": 0, "ast_fallback": 0,
            "perturb_constructible": 0, "perturb_skipped": 0,
            "ref_reproduces_gt": 0, "gen_exec_failed": 0}
    return {"dist": dist, "dist_by_frame": dist_by_frame, "degr": degr}


def _tally(acc: Dict[str, Any], res: dict) -> None:
    frame = res["pointer"].get("frame")
    for c in _CHECKS:
        st = res["check_status"][c]
        acc["dist"][c][st] += 1
        if frame in acc["dist_by_frame"]:
            acc["dist_by_frame"][frame][c][st] += 1
    d, ex = acc["degr"], res["exec"]
    d["n"] += 1
    d["gen_net_recovered"] += int(bool(ex["gen_net_recovered"]))
    d["ast_fallback"] += int(res["checks"]["check1_network_mods"]
                             ["evidence"].get("degraded", False))
    d["ref_reproduces_gt"] += int(bool(ex["ref_reproduces_gt"]))
    d["gen_exec_failed"] += int(not ex["gen_success"])
    v2 = res["checks"]["check5_provenance"]["evidence"]["v2"]
    d["perturb_constructible"] += int(bool(v2.get("constructible")))
    d["perturb_skipped"] += int(v2.get("skip_reason") is not None)


def _rates(degr: Dict[str, int]) -> Dict[str, Optional[float]]:
    n = degr["n"]
    r = lambda x: round(x, 3)
    return {
        "net_recovery_rate": r(degr["gen_net_recovered"] / n) if n else None,
        "ast_fallback_rate": r(degr["ast_fallback"] / n) if n else None,
        "perturb_unconstructible_rate":
            r(1 - degr["perturb_constructible"] / n) if n else None,
        "ref_reproduces_gt_rate": r(degr["ref_reproduces_gt"] / n) if n else None,
        "gen_exec_fail_rate": r(degr["gen_exec_failed"] / n) if n else None,
    }


def _build_payload(pointers: List[dict], items: List[dict],
                   acc: Dict[str, Any], timeout: int,
                   missing: List[dict], done: bool,
                   frames: tuple = FRAMES) -> dict:
    return {
        "meta": {
            "protocol": "E1_validity_audit.md "
                        "s2.2/s4.2/s8 (T4 full-sample automatic layer)",
            "generated": _dt.datetime.now().isoformat(timespec="seconds"),
            "pandapower": "3.4.0", "perturb_factor": checkers.PERTURB_FACTOR,
            "timeout_s": timeout,
            "frames": list(frames),
            "n_pointers": len(pointers), "n_audited": len(items),
            "n_missing": len(missing), "complete": done,
            "note": "Automatic layer only: per-check {pass/fail/ambiguous} + "
                    "evidence. NOT a final clean/pollution ruling; no FAR. "
                    "Feeds dual-judge + refute (E1 doc s2.4).",
        },
        "distribution": acc["dist"],
        "distribution_by_frame": acc["dist_by_frame"],
        "degradation": {**acc["degr"], "rates": _rates(acc["degr"])},
        "missing": missing,
        "items": items,
    }


def run_t4(manifest_path: Path, out_path: Path, *, timeout: int = DEFAULT_TIMEOUT,
           checkpoint_every: int = 50, resume: bool = True,
           frames: tuple = FRAMES, verbose: bool = True) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    pointers: List[dict] = []
    for frame in frames:
        pointers.extend(manifest["samples"].get(frame, []))

    # Resume: reload any pointers already audited in a prior (interrupted) run.
    items: List[dict] = []
    missing: List[dict] = []
    done_keys = set()
    acc = _fresh_accumulators(frames)
    if resume and out_path.exists():
        try:
            prev = json.load(open(out_path, "r", encoding="utf-8"))
            for res in prev.get("items", []):
                items.append(res)
                done_keys.add(_ptr_key(res["pointer"]))
                _tally(acc, res)
            missing = list(prev.get("missing", []))
            for mp in missing:
                done_keys.add(_ptr_key(mp))
            if verbose and done_keys:
                print(f"[resume] {len(done_keys)} pointers already done in "
                      f"{out_path.name}", flush=True)
        except (ValueError, KeyError):
            items, missing, done_keys, acc = [], [], set(), _fresh_accumulators(frames)

    todo = [p for p in pointers if _ptr_key(p) not in done_keys]
    if verbose:
        print(f"[t4] {len(pointers)} pointers total; {len(todo)} to audit "
              f"(timeout={timeout}s, checkpoint every {checkpoint_every})",
              flush=True)

    since_ckpt = 0
    for i, ptr in enumerate(todo):
        api = ptr["tier"] == "API"
        path = cond_path(ptr["model"], ptr["condition"], api=api)
        rec = records.fetch_record(path, bench_index=ptr["bench_index"],
                                   item_id=ptr.get("item_id"))
        if rec is None:
            missing.append(ptr)
            done_keys.add(_ptr_key(ptr))
            if verbose:
                print(f"[{i + 1:3d}/{len(todo)}] MISSING {ptr['frame']} "
                      f"{ptr['model']} bi={ptr['bench_index']}", flush=True)
            since_ckpt += 1
        else:
            if verbose:
                print(f"[{i + 1:3d}/{len(todo)}] {ptr['frame']:10s} "
                      f"{ptr['tier']}/{ptr['condition']} {rec['task']:16s} "
                      f"bi={ptr['bench_index']} net={rec.get('network')}",
                      flush=True)
            res = audit_item(rec, timeout=timeout)
            res["pointer"] = ptr
            res["meta"] = {k: rec.get(k) for k in
                           ("bench_index", "item_id", "task", "network",
                            "query_type", "n_modifications", "difficulty_level",
                            "ground_truth", "ground_truth_type")}
            items.append(res)
            done_keys.add(_ptr_key(ptr))
            _tally(acc, res)
            if verbose:
                print("       " + "  ".join(
                    f"{c.split('_')[0]}={res['check_status'][c]}"
                    for c in _CHECKS), flush=True)
            since_ckpt += 1
        del rec
        gc.collect()

        if since_ckpt >= checkpoint_every:
            _atomic_write(out_path, _build_payload(
                pointers, items, acc, timeout, missing, done=False, frames=frames))
            since_ckpt = 0
            if verbose:
                print(f"       [checkpoint] {len(items)} audited written to "
                      f"{out_path.name}", flush=True)

    payload = _build_payload(pointers, items, acc, timeout, missing, done=True,
                             frames=frames)
    _atomic_write(out_path, payload)
    if verbose:
        print(f"\n[t4] complete: {len(items)} audited, {len(missing)} missing "
              f"-> {out_path}")
        print("distribution:", json.dumps(acc["dist"]))
        print("degradation rates:", json.dumps(_rates(acc["degr"])))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the E1 five-check automatic layer over the full sample "
                    "(main_P + api_T5 + contrast_S = 610). CPU compute node.")
    ap.add_argument("--manifest",
                    default=str(PROJECT_ROOT / "audit" / "e1_sample_manifest.json"))
    ap.add_argument("--out",
                    default=str(PROJECT_ROOT / "audit" / "t4_auto_results.json"))
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--frames", nargs="+", default=list(FRAMES),
                    help="manifest sample frames to audit, in run order "
                         "(default: the three frozen 610 frames; the E1-P "
                         "supplement round passes 'pair_C pair_A')")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore any existing output and re-audit from scratch")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run_t4(Path(args.manifest), Path(args.out), timeout=args.timeout,
           checkpoint_every=args.checkpoint_every, resume=not args.no_resume,
           frames=tuple(args.frames), verbose=not args.quiet)


if __name__ == "__main__":
    main()
