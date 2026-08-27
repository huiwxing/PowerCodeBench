# E4 external-source supplement

This directory is a broad, reproducible supplement to the frozen E4
Layer-2a collection: a wider sweep of community sources, collected so the
demand behind Layer-2a can be checked against a larger pool. It is a
separate exploratory pool, and `e4_layer2a_set.json` (n=84) and the
evaluations built on it stay as they are.

## What is collected

- Stack Overflow questions tagged with or mentioning `pandapower`
- Electrical Engineering Stack Exchange questions mentioning `pandapower`
- Open Energy Modelling Forum topics tagged with or mentioning `pandapower`
- All pandapower GitHub Discussions, with existing E4-pool matches flagged
- Public issues in other GitHub repositories that mention `pandapower` in the
  title or body

The merged exploratory pool is `all_external_candidates.json`. For a manageable
first pass, `community_priority_shortlist.json` contains only high-priority
questions from Stack Exchange, OpenMod, and previously uncollected pandapower
Discussions. Source-specific files retain the full inventory needed to audit
counts and overlap.

## Screening labels

- `PRIORITY_REVIEW`: likely to contain a direct API, modeling, analysis, or
  interoperability demand
- `BROAD_REVIEW`: useful ecosystem or workflow signal that needs human review
- `LOW_PRIORITY`: mostly environment, maintenance, release, or weakly related
- `POSSIBLE_DUPLICATE`: likely overlap with an existing E4 record
- `ALREADY_PRESENT`: exact or normalized match already in the original pool;
  retained only in the source inventory

These are triage labels; inclusion is decided later by hand. Plotting,
conversion, private-data, and non-scalar requests stay visible in the pool
for exploration.

## Privacy and provenance

Each record contains a title, a body excerpt capped at 22 words, a source URL,
and non-identifying metadata. The collector stores no usernames, avatars,
email addresses, comments, answers, or full post bodies. The source URL is
retained for attribution and manual verification. Check the originating
platform's terms before redistributing source text.

## Reproduce

From the repository root:

```bash
python benchmark/e4_layer2a/external_supplement/collect_external_supplement.py
```

The script uses only the Python standard library. GitHub's unauthenticated search
rate limit makes the cross-repository pass take roughly 30 seconds.

## Relation to the frozen n=84 set

This pool serves as a source-diversity and demand-discovery audit of
Layer-2a, and its records are candidates rather than benchmark items. To
extend E4 from it, review `PRIORITY_REVIEW` first, remove cross-source
duplicates, then rewrite the selected requests into executable benchmark
tasks. The n=84 result remains the preregistered frozen main analysis until
a new evaluation round is declared.
