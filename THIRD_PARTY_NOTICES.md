# Third-party notices

This repository's own licences (CC BY 4.0 for data/documentation,
Apache-2.0 for code; see the "Licences" section of `README.md`) apply
only to content authored for PowerCodeBench. Two bodies of third-party
material travel with the repository and keep their own terms:
the texts referenced by the external-query construction pool under
`external_queries/` (and its archived copies under
`benchmark/e4_layer2a/`), and the upstream pandapower docstrings
reproduced in the API-spec corpus `dataset/pandapower_docs.json`.
**Nothing in this repository relicenses them**, and this repository's
CC BY 4.0 grant does not extend to them.

## What this repository actually contains

- **Candidate pools and screening ledgers** (`candidates_pool.json`,
  `all_external_candidates.json`, `source_lut*.json`,
  `collection_manifest.json`): titles, short excerpts, URLs, and dates
  recorded for provenance and screening only. Full third-party post
  bodies are not redistributed.
- **Frozen query sets** (`e4_layer2a_set.json`, `e4_layer2a_ext.json`):
  anonymised rewrites of the underlying questions, produced under the
  logged rewrite protocol (rewrite-log fields are preserved per item);
  the original wording is not redistributed, and each item keeps its
  source URL for attribution.
- **Set 3** (`set3_engineer_written_n20/`): queries written for this
  project by an engineer; no third-party text involved. The
  `ieee69_network.py` feeder implementation is authored for this project
  from published IEEE 69-bus test-system parameters.

## Source platforms and their terms

| Platform | Content referenced | Original terms |
|---|---|---|
| GitHub (`github.com` — chiefly `e2nIEE/pandapower` issues and discussions) | Issue/discussion titles and short excerpts; URLs | GitHub Terms of Service; the underlying posts remain © their authors. Original text is not redistributed here — only anonymised rewrites plus the source URL. |
| Stack Exchange network (`stackoverflow.com`, `electronics.stackexchange.com`) | Question titles and short excerpts; URLs | User contributions are licensed by Stack Exchange under CC BY-SA (3.0 or 4.0 depending on the post date). Attribution is preserved via the recorded source URLs. The CC BY-SA licence continues to govern the original posts; this repository's CC BY 4.0 does not replace it. |
| openmod forum (`forum.openmod.org`) | Thread titles and short excerpts; URLs | The forum's published terms; the underlying posts remain © their authors. Original text is not redistributed here. |
| pandapower (`e2nIEE/pandapower`, `pandapower.readthedocs.io`), nbviewer | **Docstring content is redistributed**: `dataset/pandapower_docs.json` holds 275 API entries whose `description`, `parameters` and `return_info` fields are extracted from the docstrings of `pandapower==3.4.0` (most verbatim, some lightly trimmed). Candidate-triage rows additionally record documentation URLs only. | The docstrings are part of the pandapower distribution and are reproduced here under pandapower's own **BSD 3-Clause** licence (© 2016-2026 University of Kassel and Fraunhofer IEE Kassel and individual contributors; the full text ships with the pinned package, `pandapower==3.4.0` in `environment/requirements.lock.txt`). This repository's CC BY 4.0 grant covers only the authors' own additions to that corpus — the L0--L3 layer renderings, the executable minimal-use examples, the role/category taxonomy, and the derived boundary contracts in `dataset/pandapower_library_knowledge.json`. |

If you reuse the external-query materials, follow the original platform
terms for anything you trace back through the recorded URLs. The
anonymised rewrites and screening annotations authored for this project
are covered by this repository's CC BY 4.0. If you reuse
`dataset/pandapower_docs.json`, keep pandapower's BSD 3-Clause notice
with the docstring-derived fields.

The two other backend API-spec corpora do **not** reproduce upstream
text: the 76 entries of `dataset/opendss_docs.json` and the 67 of
`dataset/pypsa_docs.json` are descriptions written for this project
against the public APIs of `opendssdirect.py==0.9.4` (© 2017 Alliance
for Sustainable Energy, LLC; BSD 3-Clause with a US Government rights
notice) and `pypsa==1.2.3` (MIT). Only the API names, signatures and
parameter names — facts about the interfaces — come from those packages.
