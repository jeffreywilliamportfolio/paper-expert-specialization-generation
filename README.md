# Specialization Is Not a Myth: Subject-Specific Expert Sets Recur During MoE Generation

Paper source, figures, data, audit trail, and reproducibility materials. (Version 1.0 was
titled "The Generation Half: Why Prompt-Routing Studies Understate Domain Specialization in
Mixture-of-Experts Models"; version 1.1, "Generation-Time Routing Reveals Expert
Specialization That Prefill Measurements Miss," restructured the paper around its single
claim; version 1.2 is a replicate-then-extend revision that adds the cross-design subject
identification result, its controls, and four new checkpoints of prefill replication, and
retains every version 1.1 result.)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20779604.svg)](https://doi.org/10.5281/zenodo.20779604)

**Author:** Jeffrey W. Shorthill (independent researcher) · `jws299792@icloud.com`
**Version:** 1.2 (September 2026; v1.1 August 2026, v1.0 June 2026) · preprint, not peer reviewed
**DOI:** [10.5281/zenodo.20779604](https://doi.org/10.5281/zenodo.20779604) (concept DOI — resolves to the latest version)
**License:** [CC BY 4.0](LICENSE)

## What this is

A mixture-of-experts (MoE) transformer routes tokens in two regimes: a parallel **prefill**
pass that reads the prompt, and a step-by-step **generation** pass that writes the answer.
Prior work correctly finds that pooled prefill routing converges on one shared set of
experts across domains; we replicate that result on five checkpoints across two model
families and two gating mechanisms. Generation gives a different answer: expert sets
measured on generated tokens follow the local subject of the answer, identifying the
subject across independent prompt designs at 95 to 96% (register-matched; 0.70 over all
design pairs) against 6% by chance, staying stable within a subject, turning over at
subject boundaries, and surviving position, drift, register, dialect, and checkpoint
controls. For short mixed prompts, prefill carries the wording of the question; generation
carries the subject of the answer.

## Repository layout

| Path | Contents |
|---|---|
| `main.tex` | Paper source (LaTeX), version 1.2. |
| `refs.bib` | Bibliography (every entry verified against an authoritative record). |
| `figures/` | The figures as `.pdf` (v1.2 uses fig1, fig3, and fig4). |
| `make_figures.py` | Regenerates the v1.1 figures and concentration summaries from the winner lists. Runs no model. |
| `SOURCES.md` | Claim-by-claim source-to-value index for the v1.1 values (all retained in v1.2). |
| `ledger/` | v1.2 audit trail: findings reports, data ledger, frozen predictions, number audit, hostile review, related-work reviews. |
| `curated-data-index/` | Provenance records, manifests, build script, and complete analysis scripts of the curated data archive (see Data). |
| `data/` | v1.1 supporting data (see `data/README.md`), retained: 60-prompt primary probe, token-balanced control, 122B replication. |
| `JOURNAL_v12.md`, `FIXLIST_v12_20260830.md`, `DRAFT_v12_*` | Working journal, consolidated fix list, and drafting record for v1.2. |
| `main.pdf` | Built PDF of the paper. |

## Data

The v1.2 results are computed from the curated archive `expert-specialization-data`
(commit `f06cd3e`, about 6 GB of router tensors and per-cell compactions). This repository
includes that archive's `PROVENANCE.md` (run id, model file, capture binary, decode flags,
and source path for every set), `MANIFEST.sha256` (file-by-file checksums),
`raw_manifest.tsv`, the build script, and the complete analysis scripts
(`curated-data-index/audit_20260828/`: the identification test, the four nulls, the
per-pair table, the GLM control sets, and the checkpoint-shift analysis). The full archive
is available from the author. No script re-runs a model.

`data/` retains the three v1.1 runs, each mapped to the claims it supports in
`data/README.md`:

- `35b-60prompt-primary/` — 35B 60-prompt primary probe (prefill generalist wins 18/20 → 20 distinct winners over generation).
- `35b-3chunk-token-balanced/` — token-balanced control (prefill/generation route to near-disjoint experts at matched token counts).
- `122b-replication/` — 122B replication (E233 wins 13/20 prefill; generation exploratory, see paper).

## Build

```bash
latexmk -pdf main.tex      # produces main.pdf
python make_figures.py     # regenerates the v1.1 figures from the winner lists (no model is run)
```

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the preprint (version 1.2, 2026); the
concept DOI resolves to the latest version.

## AI-use disclosure

Generative AI (Anthropic's Claude) was used for drafting and revising prose (including
first drafts of the v1.2 Results, Discussion, Limitations, and Conclusion sections,
written to the author's analyses), number auditing against source captures, organization,
and bibliography formatting. The author verified every reported value and reference and
takes full responsibility for the content.
