# The Generation Half

Why Prompt-Routing Studies Understate Domain Specialization in Mixture-of-Experts Models — paper source, figures, data, and reproducibility materials.

**Author:** Jeffrey W. Shorthill (independent researcher) · `jws299792@icloud.com`
**Version:** 1.0 (June 2026) · preprint, not peer reviewed
**License:** [CC BY 4.0](LICENSE)

## What this is

A mixture-of-experts (MoE) transformer routes tokens in two regimes: a parallel
**prefill** pass that reads the prompt, and a step-by-step **generation** pass that
writes the answer. Routing analyses (and the widely cited negative result on expert
specialization) typically measure supplied text — the prefill half. This paper shows
that over prefill a single generalist expert wins almost every domain and routing looks
domain-blind, while over generation the winners disperse into distinct per-domain
experts. The shift survives a length-matched control and reproduces across two model
sizes with unrelated expert indices. Domain specialization is largely a property of the
generation pass; prefill/supplied-text routing understates it.

## Repository layout

| Path | Contents |
|---|---|
| `main.tex` | Paper source (LaTeX). |
| `refs.bib` | Bibliography (every entry verified against an authoritative record). |
| `figures/` | The three figures, as `.pdf`. |
| `make_figures.py` | Regenerates the figures and concentration summaries from the winner lists. Runs no model. |
| `SOURCES.md` | Claim-by-claim source-to-value index. |
| `data/` | Supporting data for all reported numbers (see `data/README.md`). |
| `main.pdf` | Built PDF of the paper. |

## Data

`data/` bundles the three runs the paper rests on, each mapped to the claims it supports
in `data/README.md`:

- `35b-60prompt-primary/` — 35B 60-prompt primary probe (prefill generalist wins 18/20 → 20 distinct over generation).
- `35b-3chunk-token-balanced/` — length-matched control (prefill/generation route to near-disjoint experts at matched token counts).
- `122b-replication/` — 122B replication (E233 wins 13/20 prefill → 18 distinct over generation).

Per-domain winner tables carry every number in the paper; raw router tensors (`.npz`,
~26 MB total) are included as non-load-bearing provenance.

## Build

```bash
latexmk -pdf main.tex      # produces main.pdf
python make_figures.py     # regenerates figures/ from the winner lists (no model is run)
```

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the preprint (version 1.0, 2026).

## AI-use disclosure

Generative AI (Anthropic's Claude) was used for drafting, organization, and
bibliography formatting. The author verified every reported value and reference and
takes full responsibility for the content.
