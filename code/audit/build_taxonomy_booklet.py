#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Renders the frozen 100-record Failure Anatomy sample into annotation
# booklets: the material a reviewer works from.
#
# This is how the released human labelling in audit/failure_taxonomy/
# human_review/ was collected, and it is reusable: point it at any sample in
# the same schema to run the equivalent labelling elsewhere.
#
# The booklet shows, per record: the operator query, the expected answer, the
# execution status and error, the automatic diagnostics, the line-numbered
# generated program, and a collapsed reference solution - then a slot for the
# category, the confidence, and two free-text fields. No existing label is
# included, so the reviewer works blind.
#
# Usage:
#   python3 code/audit/build_taxonomy_booklet.py                 # one booklet
#   python3 code/audit/build_taxonomy_booklet.py --batches 4     # split it
#
# Reads  audit/failure_taxonomy/revision_sample/sample_records.json
# Writes booklet_all.md (and booklet_batchN.md) into --out-dir.
# --------------------------------------------------------------------------
"""Render the frozen taxonomy sample into blind annotation booklets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "audit/failure_taxonomy/revision_sample/sample_records.json"

# The rubric as used by every pass over this sample. Do not reword: passes are
# only comparable to one another if the standard is identical.
RUBRIC = {
    "C1": "hallucinated, missing, or wrong callable or API contract",
    "C2": "wrong parameter, index, table, column, or result access",
    "C3": "wrong domain sequence, missing analysis, wrong network, intervention, or logic",
    "C4": "correct workflow but wrong numeric transformation or scalar extraction",
    "C5": "syntax, truncation, import, environment, timeout, or output-format execution failure",
    "ambiguous": "evidence is insufficient or the query and reference conflict",
}

FIELDS = ("revision_sample_id", "model", "model_tier", "condition", "task",
          "difficulty_level", "network", "query", "reference_code",
          "ground_truth", "ground_truth_type", "generated_code", "executed",
          "error_type", "error_msg", "exec_output", "match", "diagnostics")

DIAG_LABEL = {"correct_task_fn": "task function", "correct_network": "network",
              "modifications_applied": "modifications applied"}


def load_records() -> list[dict]:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    out = []
    for r in data["records"]:
        row = {k: r.get(k, "") for k in FIELDS}
        d = row["diagnostics"]
        if isinstance(d, str) and d.strip().startswith("{"):
            try:
                d = json.loads(d.replace("'", '"').replace("True", "true")
                               .replace("False", "false"))
            except ValueError:
                pass
        row["diagnostics"] = d
        out.append(row)
    return out


def numbered(src: str) -> str:
    lines = str(src or "").rstrip().split("\n")
    w = len(str(len(lines)))
    return "\n".join(f"{i+1:>{w}}  {line}" for i, line in enumerate(lines))


def record_md(r: dict, n: int, total: int) -> str:
    executed = str(r["executed"]) == "True"
    matched = str(r["match"]) == "True"
    status = ("execution succeeded" if executed else "execution failed")
    if r.get("error_type"):
        status += f" · `{r['error_type']}`"
    status += " · " + ("result matches" if matched else "result does not match")

    d = r.get("diagnostics")
    diag = (" · ".join(f"{DIAG_LABEL.get(k, k)} {'yes' if v else 'no'}"
                       for k, v in d.items())
            if isinstance(d, dict) and d else "(none)")

    parts = [
        f"## {n} / {total} · `{r['revision_sample_id']}`", "",
        "| Model | Tier | Condition | Task family | Difficulty | Network |",
        "|---|---|---|---|---|---|",
        f"| {r['model']} | {r['model_tier']} | `{r['condition']}` | "
        f"{r['task']} | {r['difficulty_level']} | {r['network']} |", "",
        "**Operator query**", "",
        "> " + str(r["query"]).replace("\n", "\n> "), "",
        f"**Expected answer**: `{r['ground_truth']}` ({r['ground_truth_type']})", "",
        f"**Execution result**: {status}", "",
    ]
    if str(r.get("error_msg") or "").strip():
        parts += ["**Error**", "", "```text", str(r["error_msg"]).rstrip(), "```", ""]
    if str(r.get("exec_output") or "").strip():
        parts += ["**Program output**", "", "```text",
                  str(r["exec_output"]).rstrip(), "```", ""]
    parts += [
        f"**Automatic diagnostics**: {diag}", "",
        "**Model-generated code**", "", "```python", numbered(r["generated_code"]), "```", "",
        "<details>", "<summary>Reference solution (correct approach) — expand if needed</summary>",
        "", "```python", numbered(r["reference_code"]), "```", "", "</details>", "",
        "### ▸ Judgement", "",
        "- **Category**: `____`　(C1 / C2 / C3 / C4 / C5 / ambiguous)",
        "- **Confidence**: `____`　(high / medium / low)",
        "- **Evidence**: ",
        "- **Rationale**: ", "", "---", "",
    ]
    return "\n".join(parts)


def booklet(recs: list[dict], start: int, total: int,
            part: str | None = None) -> str:
    rubric = "\n".join(f"| `{c}` | {d} |" for c, d in RUBRIC.items())
    head = [
        "# PowerCodeBench — Failure-Classification Annotation"
        + (f" ({part})" if part else ""), "",
        "**Annotator**: ________________　**Date**: ____________", "",
        "## 1. Task", "",
        "Each item below is a *failed* code generation: the model received a "
        "power-system analysis request, produced a Python program, and the "
        "program's result was not correct. Decide **the dominant cause of that "
        "failure**.", "",
        "## 2. Classification standard", "",
        "| Code | Definition |", "|---|---|", rubric, "",
        "> **Decision rule**: the category is the cause that **actually occurred "
        "first and produced this failure**. Where several problems coexist, the "
        "one that actually stopped execution governs — if the program aborts at "
        "line 6 on a nonexistent API, the later logic problem has not yet "
        "happened, so the item is `C1`, not `C3`.", ">",
        "> Where evidence is genuinely insufficient, or the query and the "
        "reference solution conflict, record `ambiguous` rather than forcing a "
        "class.", "",
        "## 3. How to fill this in", "",
        "Replace each `____` in the **▸ Judgement** block at the end of every "
        "item:", "", "```",
        "### ▸ Judgement", "",
        "- **Category**: `C1`　(C1 / C2 / C3 / C4 / C5 / ambiguous)",
        "- **Confidence**: `high`　(high / medium / low)",
        "- **Evidence**: `pp.import_network` does not exist; the reference uses "
        "`pn.create_cigre_network_hv()`",
        "- **Rationale**: execution aborts at line 6; the later steps never run",
        "```", "",
        "- **Category** and **Confidence** are required; **Evidence** and "
        "**Rationale** are optional but valuable on hard calls.",
        "- The numbers down the left of each program are line numbers; the "
        "`Line N` in an error message refers to them.",
        "- The reference solution is collapsed by default; expand it when you "
        "need to compare.",
        "- Order does not matter and the work can be interrupted; save and "
        "return the file when done.", "", "---", "",
    ]
    body = [record_md(r, start + i + 1, total) for i, r in enumerate(recs)]
    return "\n".join(head) + "\n".join(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".",
                    help="where to write the booklets (default: cwd)")
    ap.add_argument("--batches", type=int, default=1,
                    help="split the sample into N booklets (default: 1)")
    args = ap.parse_args()

    recs = load_records()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = len(recs)

    p = out / "booklet_all.md"
    p.write_text(booklet(recs, 0, total), encoding="utf-8")
    print(f"wrote {p}  ({total} records, {p.stat().st_size/1024:.0f} KB)")

    if args.batches > 1:
        size = -(-total // args.batches)
        for b in range(args.batches):
            chunk = recs[b * size:(b + 1) * size]
            if not chunk:
                continue
            q = out / f"booklet_batch{b+1}.md"
            q.write_text(booklet(chunk, b * size, total,
                                 f"part {b+1} of {args.batches}"), encoding="utf-8")
            print(f"wrote {q}  ({len(chunk)} records)")


if __name__ == "__main__":
    main()
