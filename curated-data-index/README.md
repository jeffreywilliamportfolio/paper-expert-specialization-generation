# Expert specialization in MoE routing: curated data archive

Curated from the ExternalSSD research archive on 2026-08-28. This repo holds every capture that
bears on **expert specialization by domain, with prefill and generation kept separate**, plus the
controls that scope it (token balance, register, dialect, base vs fine-tune, causal bias).
It is the data companion to *Generation-Time Routing Reveals Expert Specialization That Prefill
Measurements Miss* (Shorthill; concept DOI 10.5281/zenodo.20779604) and to the 2026-08-28 audit
in `audit_20260828/`.

Nothing here is new capture. Every file was copied or derived from an existing run; `PROVENANCE.md`
gives the run id, model file, binary, decode flags and source path for each set, and
`MANIFEST.sha256` covers every file in the repo. Raw router tensors that were too large to copy were
**compacted** (see below) and their sha256 + source path recorded in `raw_manifest.tsv`, so any
number can be re-derived from the originals on the drive.

## Layout

```
qwen35b/                          Qwen3.5-35B-A3B (HauhauCS Q8_0 unless noted)
  domain_probe_60prompt/          Run 1: 20 domains x 3 prompts, prefill + generation (both captures)
  token_balanced_3chunk/          Run 2: 3 packed prompts, 446 prompt / 2048 generated tokens
  philosophy_experts_bias_causal/ 9-level bias sweep on experts 114/87/170/68 (the only causal data)
  base_vs_hauhau_prefill/         base vs fine-tune, 150 self-reference prompts, prefill only
  controls/
    hvac_6cond_results/           HVAC register-control results tables (E114-centred)
    hvac_6cond_l1l3_distinct18_raw/  raw tensors for the 18 distinct HVAC prompts (of 180 cells)
    l1l3_register_hauhau_raw/     raw: technical / recursive / experience register, fine-tune
    l1l3_register_base_raw/       raw: same, base Qwen3.5-35B-A3B Q8_0
    five_cond_experience_probe_raw/  raw: experience probe x 5 deictics
    aave_routing_tables/          existing routing tables for the AAVE / medical arms
    compact/                      per-cell per-layer W/S/Q (see "Compact format")
      aave_5-5_register_run/      200 cells: 50 AAVE/AE pairs x base+fine-tune
      aave_5-15_medical/          medical triage arms (dialect-only, dialect+clinical register)
      hvac_6cond_l1l3_all180/     all 180 HVAC cells (18 distinct x 10 repeats)
qwen122b/                         Qwen3.5-122B-A10B (HauhauCS Q8_K_P)
  domain_probe_60prompt/          Run 3: prefill-only run + generation run (results JSON, text; no tensors)
  five_cond_experience_probe/     per-token npz + JSON, deictic axis only
  single_prompt_hum/              per-token npz, one prompt
  six_cond_hvac_E114only/         E114-only per-layer tables (no other expert recoverable)
  baseline_followups_per_token/   per-token npz for the follow-up runs
glm47flash/                       GLM-4.7-Flash (64 experts, top-4, sigmoid + correction bias)
  domain_exploratory_raw_prefill/ 20 domains x 1 prompt, raw router logits, PREFILL ONLY
  domain_powered_raw_prefill/     20 domains x 15 prompts, raw router logits, PREFILL ONLY
  analysis/                       expert_domain_map, layer separation, summaries
  register_run_meta/              batteries, scripts, router gate matrices (base + fine-tune), bias
  compact/register_run/           140 cells: FIRE/NOFIRE register banks with generation
audit_20260828/                   findings PDF, data ledger, frozen predictions, five agent reports
raw_manifest.tsv                  sha256 + source path of every raw tensor that was compacted, not copied
MANIFEST.sha256                   sha256 of every file in this repo
build_repo.py                     the script that built this repo (re-runnable, read-only on sources)
```

## Routing reconstruction

Qwen3.5: softmax over the 256 router logits, keep the top 8, renormalize to sum to 1.
GLM-4.7-Flash: sigmoid of the 64 logits; select the top 4 by sigmoid **plus** the per-layer
`e_score_correction_bias`; weight = plain sigmoid of the selected 4, renormalized (the 1.8 routed
scaling factor is omitted; it changes no ranking). Per expert on a token set: `W` = mean weight over
all tokens (0 where unselected), `S` = fraction of tokens selecting it, `Q` = mean weight when
selected; `W = S * Q` per token set.

Prefill block = prompt positions; generation block = produced tokens; never pooled.
`generation_trimmed` cuts at the first end-of-turn token id (Qwen 248044/248045/248046; GLM
154827/154828/154820/154829) read from `generated_tokens.json`. This is the correct trim; several
archived analysis scripts scanned for a 6-token BPE spelling of `<|im_end|>` instead, which never
matches (see `audit_20260828/FINDINGS_expert_specialization_20260828.pdf`, Section 4).

**Last-layer quirk.** llama.cpp evaluates the final layer for the last prompt token only, so the last
router file has `1` row in a prefill-only run and `n_gen + 1` rows in a generation run. The compaction
handles this (prefill left NaN at that layer; generation taken from row 1 on) and records it in `notes`.

## Compact format

One `.npz` per cell, keys `prefill_W/S/Q`, `gen_W/S/Q`, `gentrim_W/S/Q`, each `[n_layers, n_experts]`
(NaN where a block has no rows at that layer), plus `n_prompt`, `n_gen`, `trim_index` (-1 = no
end-of-turn token before the cap), `rows_per_layer`, `layers`, `model`, `reconstruction`, `source`
(path relative to `/Volumes/ExternalSSD`), `notes`, `prompt_id`. `INDEX.jsonl` in each compact
directory lists the cells. Pooled-over-layers values are the plain mean over layers of these arrays;
the archived Qwen analyses pool the same way (equal weight per layer, then per cell).

## What was deliberately left out, and why

- Duplicate copies of the same runs (`llama-eeg-tests/experiments/35B`, `git-updates-moe`,
  `sae-tests`, `aave-registers/runs`, the second HVAC copy): byte-identical, verified.
- `e114-construction-reframe/`, `qwen-huahua-expert-routing-data-injection/`, `e114-tensors-zenodo/`,
  the 5-cond consciousness suites on other models, `llama-eeg-backups/selfref-paired-1`: E114
  program data with no domain axis; belongs to the E114 papers.
- 122B 150-prompt baseline and its `Archive.zip`: no routing tensors survive, text only.
- The 2.4 GB philosophy-bias tar: its contents are already in `captures/` and `generated-text-merged/`.
- Everything under `cc-lens/outputs`, `papers/`, `black-holes/`: lens work, not routing.

## Known defects carried in from the source runs

Listed in full in the audit findings, Section 4. In brief: the 122B "trimmed" track is untrimmed;
the 122B prefill-only and generation runs differ at the last layer (routing-only run is the clean
prefill); E48 on 122B is a post-spill expert; the HVAC "180 prompts" are 18 x 10 repeats; the
philosophy-bias README misdescribes the design (all four experts received the same bias); the
token-balanced Jaccard uses top-8 sets.
