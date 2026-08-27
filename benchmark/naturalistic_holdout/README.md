# Naturalistic-Query Holdout

This directory holds an independent natural-language query holdout:
benchmark tasks restated in operator-style language, so they can be
evaluated by the existing benchmark pipeline without changing the runner.
Shipped here are `naturalistic_holdout_seed22_n80.json` with its manifest,
the source answer key, and the authoring instructions. The numbered recipe
below is the construction procedure. Commands run from this repository's
root unless marked as pipeline-only.

1. Generate an authoring packet. The authoring-packet script
   `scripts/build/prepare_naturalistic_holdout_authoring.py` lives in the
   frozen experimental pipeline and is not part of this release:

   ```bash
   python scripts/build/prepare_naturalistic_holdout_authoring.py --n 80 --seed 22
   ```

2. Give `authoring_tasks_seed22_n80.csv` to a human query author. They fill
   `naturalistic_query` in operator-style language and avoid copying the
   source wording in `task_brief`.

3. Have a second person review the queries. Accepted rows are marked
   `status=accepted`, `status=reviewed`, or `status=final`.

4. Build the benchmark JSON:

   ```bash
   python code/build/build_naturalistic_holdout.py \
     --authoring-csv benchmark/naturalistic_holdout/authoring_tasks_seed22_n80.csv \
     --answer-key benchmark/naturalistic_holdout/source_answer_key_seed22_n80.json \
     --output benchmark/naturalistic_holdout/naturalistic_holdout_seed22_n80.json
   ```

5. Run the usual benchmark pipeline against the new JSON. For API models,
   `api_eval.runner` and the `probe_eval_results/` run trees live in the
   frozen experimental pipeline; the frozen aggregate archived here is
   `results/aggregates/naturalistic_holdout/reactive_comparison.json`:

   ```bash
   python -m api_eval.runner \
     --benchmark benchmark/naturalistic_holdout/naturalistic_holdout_seed22_n80.json \
     --subset-size 80 \
     --conditions A,C,C_FX,C_FDR \
     --output-dir probe_eval_results/naturalistic_holdout_api
   ```

## Naming

A holdout whose queries were written and reviewed by the paper authors is an
`author-curated naturalistic-query holdout` or a `human-authored naturalistic
query holdout`. The labels `expert-curated` and `expert-reviewed
naturalistic-query holdout` apply when a domain expert in power or energy
systems authored or reviewed the query sheet. Anyone rebuilding a holdout with
this recipe should pick the label that matches who filled and reviewed it.
