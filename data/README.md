# Data for "Where Experts Specialize: Domain Identity Appears in Generation, Not Prefill"

This folder is the reproducibility archive for the paper. It contains the three
independent runs that establish the prefill-concentrated / generation-dispersed
result, organized one folder per run. Every headline number in the paper is
traceable to a file here; the claim-by-claim index is in the repo-root
[`../SOURCES.md`](../SOURCES.md).

All captures use greedy decoding and a top-8-of-256 routing reconstruction. The
`W = S · Q` identity holds **per token set** (one prompt, one pass, one layer) to
machine precision (residual ≤ 5.6×10⁻¹⁷), since `W`, `S`, and `Q` are three
reductions of the same renormalized weights. Note that the per-domain and overall
`W`/`S`/`Q` columns in the results tables are each averaged **independently** across
a domain's prompts and the model's layers, so `S × Q` in those tables does **not**
equal `W` (the gap is the across-prompt `S`–`Q` covariance). Domain winners are
ranked by `W`, the mean routed weight per token.

## Runs

### 1. `35b-60prompt-primary/` — primary result (35B)
Run `20260408T235839Z`. Model **Qwen3.5-35B-A3B** (HauhauCS Q8_0), no-think
greedy, `-n 2056`. 60 prompts, 20 domains × 3.
- **Shows:** prefill generalist **expert 224 wins 18/20 domains** (3 distinct
  winners, normalized entropy **0.13**); generation disperses to **20 distinct
  winners** (entropy **1.00**). Philosophy's generation winner (E114) sat at
  prefill rank 5.
- Files: `results_domain_specialists_20260408T235839Z.md` (full per-domain winner
  tables, prefill + generation + trimmed), `.json` (machine-readable),
  `RESULTS-summary.md` (run summary). `raw/…​.npz` = raw router tensors
  (not load-bearing for the winner-table claims; included for full re-derivation).
- Paper claims: SOURCES.md rows 1–8, 20.

### 2. `35b-3chunk-token-balanced/` — length control (35B)
Run `20260410T173400Z`. 3 packed prompts of **446 prompt tokens** each, 2048
generated. Holds token count fixed across prefill and generation.
- **Shows:** with length matched, prefill and generation select **near-disjoint
  expert sets** — top-W Jaccard **0.000 / 0.231 / 0.067**, top-S Jaccard
  **0.067 / 0.231 / 0.000** — while routing entropy is ~unchanged (prefill
  **0.958**, generation **0.953**). Length does not drive the shift.
- Key files: `DOCS/RESULTS.md` (Jaccard + entropy summary, the cited values),
  `results/…​3chunk_20260410T173400Z.md`/`.json` (winner tables), `results/…​
  token_audit_summary.*` (token-balance audit), `METHOD/`, `PROMPTS/`. Per-token
  `.npz` under `results/per_token_*/` are raw (not load-bearing).
- Paper claims: SOURCES.md rows 9–13.

### 3. `122b-replication/` — independent replication (122B)
Prefill run `20260412T160341Z` (`routing-only/`) + generation run
`20260412T161833Z` (`generation/`). Model **Qwen3.5-122B-A10B** (HauhauCS
Q8_K_P). Expert indices are unrelated to the 35B model's.
- **Shows:** a *different* generalist (**E233 wins 13/20** in prefill; 7 distinct,
  entropy **0.42**) and the same dispersion in generation (**18 distinct
  winners**, entropy **0.95**). The prefill-concentrated / generation-dispersed
  *shape* reproduces though no expert index transfers.
- `routing-only/results.md` + `RESULTS/…​routing_only.md`/`.json`; `generation/
  results.md` + `RESULTS/…​gen_n2048.md`/`.json`, `text-results.md` (token-cap /
  spill behavior: 49/60 cells hit the 2048 cap, mean generated ≈ 1868),
  `results-generated.txt`. `generation/raw/…​/` holds the per-prompt generated
  text + tokens for all 60 cells. `JOURNAL-122B.md` documents the 122B campaign.
- Paper claims: SOURCES.md rows 14–19.

## Figures
The three paper figures and the Table 1 concentration summaries are produced by
[`../make_figures.py`](../make_figures.py) from the winner lists in these files
alone — it re-runs no model and performs only deterministic counting (distinct
count, max wins, Herfindahl, normalized Shannon entropy).

## Notes on scope and provenance
- **Excluded:** run `20260415T214918Z` (a 35B re-analysis that *shares*
  `20260408T235839Z`'s deterministic prefill capture) is **not** included — it is
  not an independent replication and the paper does not cite it as one.
- The 122B source notes originally framed the generation analysis around locating
  an analog of one expert (E114); that framing is set aside. The paper reports
  only the pass-dependent concentration the same domain-winner tables support.
- **`.npz` files (~40 MB)** are raw router/per-token tensors. They are kept under
  `raw/` (35B) and `results/per_token_*/` (3-chunk) and are **not required** to
  reproduce any paper number. To keep the Git history lean, track them with
  Git LFS (`*.npz`) or exclude them and host on Zenodo.
