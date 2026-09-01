# Provenance

All paths are relative to `/Volumes/ExternalSSD/` (the source drive). "Binary" is the custom
llama.cpp `capture_activations` example that dumps router logits (`--routing-only`). Decode is greedy
(`--temp 0 --top-k 1`) everywhere unless stated. Where a hash was never recorded by the original run
it is marked NOT RECORDED rather than guessed.

## Qwen3.5-35B-A3B, HauhauCS "Uncensored Aggressive" Q8_0

Model file: `Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf`, sha256 `f3235db7…64cb17`
(full hash in `qwen35b/domain_probe_60prompt/results/20260415T214918Z_*_run_metadata.json`).
40 MoE layers (10 softmax attention at 3, 7, …, 39; 30 gated-DeltaNet), 256 experts, top-8.

| Set | Run id(s) | Design | Decode | Binary | Source |
|---|---|---|---|---|---|
| domain_probe_60prompt | 20260408T235839Z; re-capture 20260415T214918Z (bit-identical) | 20 domains x 3 prompts, no-think | `-n 2056`, seed 42, ctx 16384, KV q8_0 | build 8493 (1772701f), 2x RTX 5090; hash NOT RECORDED for the first run, recorded in the re-capture metadata | `moe-routing-organized/qwen3.5-35b-a3b-and-huahua/35B/qwen-huahua-expert-identification` |
| token_balanced_3chunk | 20260410T173400Z | 3 packed prompts (mechanism / historical figure / synthesis), 446 prompt tokens each (padding 0/73/44), 2048 generated | seed 42, ctx 32768, flash-attn, KV q8_0 | `/workspace/consciousness-experiment/capture_activations`, hash NOT RECORDED | `paper-expert-specialization-generation/data/35b-3chunk-token-balanced` |
| philosophy_experts_bias_causal | 20260409T19xxxx–20xxxxZ, 9 runs | `--expert-bias 114:b,87:b,170:b,68:b`, b in {-8,-5,-3,-2,0,+2,+3,+5,+8}, all 40 layers, prefill + generation, 60 prompts, `-n 1024` | greedy | `METHOD/capture_activations.cpp` in the set (bias applied before top-k, written back) | `git-updates-moe/qwen3.5-35b-a3b-huahua-philosophy-experts-bias` (tars excluded) |
| base_vs_hauhau_prefill | 2026-03-19 | 150 self-reference prompts, prefill only, base `Qwen3.5-35B-A3B` Q8_0 vs HauhauCS | `-n 0` | see `PLAN.md` in set | `moe-routing-organized/.../35B/qwen35b-a3b-vs-hauhaucs-uncensored-run1` |
| controls/hvac_6cond_* | 2026-04 | 1 HVAC + water-treatment paragraph x 3 registers (L1 technical / L2 recursive / L3 experience) x 6 deictics; **18 distinct prompts, each repeated 10x** | `-n 1024`, no-think | NOT RECORDED | `hvac_cal_water_treatment_6cond_l1l3_hauhau` (raw), `moe-routing-organized/.../35B/qwen-huahua-6cond-hvac` (results) |
| controls/l1l3_register_{hauhau,base}_raw | 2026-04 | transformer-description paragraph x 3 registers, 10 prompts each, deictic A; base = `Qwen3.5-35B-A3B-Q8_0.gguf` | greedy | NOT RECORDED (see `experiment.log`) | `moe-routing-organized/.../35B/l1l3_a_only_{hauhau,vanilla}` |
| controls/five_cond_experience_probe_raw | 20260410T045738Z | 3 experience-probe prompts x 5 deictics, no-think, `-n 1024` | greedy | NOT RECORDED | `moe-routing-organized/.../35B/qwen3.5-35b-a3b-huahua-five-cond-experience-probe` |
| controls/compact/aave_5-5_register_run | 20260505T205437Z | 50 AAVE / academic-English pairs x 8 prompt types x 7 speaker roles; base `ggml-org/Qwen3.5-35B-A3B-GGUF` Q8_0 and HauhauCS; no-think, `-n 2048` | greedy | see `docs/PLAN.md` in source | `aave-registers-cleaned/5-5-26_initial_50_pair_register_run/runs` (identical to `aave-registers/runs`) |
| controls/compact/aave_5-15_medical | 2026-05-15 | 5 triage scenarios x 2 dialects x {base, HauhauCS} x {think, no-think}; `chest_pain_plus` = 6 scenarios with AE rewritten clinical and AAVE vernacular; `financial_stress_pair` 1 pair | greedy | see `provenance/` in source | `aave-registers-cleaned/5-15-26/aave-register-medical` |
| controls/aave_routing_tables | derived | existing per-run expert profiles, JSD tables | — | — | `aave-registers-cleaned/analysis_all_models_expert_routing` |

## Qwen3.5-122B-A10B, HauhauCS "Uncensored Aggressive" Q8_K_P

Model file: `Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf`, 144,514,683,488 bytes,
sha256 NOT RECORDED. 48 MoE layers (12 softmax attention at 3, 7, …, 47; 36 gated-DeltaNet),
256 experts, top-8. Capture binary sha256
`c3e205b3a6324f6ce11e775b8ab1c9115904e348ec118fcb62d5e31ccae62e70` for the domain runs.
Flags: `-ngl 999 -c 16384 -t 16 -fa on --cache-type-k q8_0 --cache-type-v q8_0 --seed 42 --routing-only`.

| Set | Run id(s) | Design | Notes | Source |
|---|---|---|---|---|
| domain_probe_60prompt | prefill-only 20260412T160341Z (`-n 0`); generation 20260412T161833Z (`-n 2048`) | same 60 prompts as the 35B probe | **No router tensors survive** (deleted on the box); results JSON, per-cell tokens and text only. Prefill-only run is the clean prefill (the generation run's prefill block has last-layer contamination). The JSON "trimmed" track is untrimmed. | `paper-expert-specialization-generation/data/122b-replication` |
| five_cond_experience_probe | 20260412T172428Z | 3 prompt pairs x 5 deictics, `-n 2048` | per-token npz, layer-averaged, softmax/DeltaNet split | `moe-routing-organized/qwen3.5-122b-a10b-huahua/qwen3.5-122B-A10B-huahua-five-cond-experience-probe` |
| single_prompt_hum | 20260412T184544Z | 1 prompt, 119 prompt tokens, `-n 2048`, clean turn 458 | per-token npz | `.../qwen3.5-122B-A10B-huahua-single-prompt-processing-hum` |
| six_cond_hvac_E114only | 20260412T194000Z | 10 prompts x 3 registers x 6 deictics, `-n 2048` | JSON stores expert 114 only | `.../qwen3.5-122B-A10B-huahua-six-cond-hvac` |
| baseline_followups_per_token | 2026-04-12 | per-token npz for the follow-up runs above | `Archive.zip` (0 tensors) excluded | `.../qwen3.5-122B-A10B-huahua-baseline/followups` |

## GLM-4.7-Flash (zai-org), base and HauhauCS Uncensored

30B-A3B, 47 layers (layer 0 dense, 46 MoE), 64 routed experts + 1 shared, top-4, sigmoid gating with
per-layer `e_score_correction_bias` (values 8.91–9.05; identical across base and fine-tune).
Router gate matrices for base and fine-tune are in `glm47flash/register_run_meta/{base,hauhau}_gates.npz`
and are bit-identical (max abs diff 0).

| Set | Rig | Design | Notes | Source |
|---|---|---|---|---|
| domain_exploratory_raw_prefill | BF16 transformers hook | 20 domains x 1 prompt, 31–43 tokens, **prefill only** | raw `ffn_moe_logits-{1..46}.npy` `[n_tok, 64]` | `glm47-flash-domain-routing/raw` |
| domain_powered_raw_prefill | BF16 transformers hook | 20 domains x 15 prompts, 18–21 tokens, **prefill only**; analysis truncates to T=16 | raw as above | `glm47-flash-domain-routing/raw_powered` |
| analysis | derived | expert_domain_map, layer separation, summaries | | `glm47-flash-domain-routing/analysis` |
| register_run_meta | — | FIRE/NOFIRE batteries (heldout introspective, blockD worldview, blockF relational), scripts, outputs, gates, bias | | `glm47-flash-domain-routing/register_run` minus capture dirs |
| compact/register_run | llama.cpp Q8 GGUF, commit 6658925, greedy, `-n 4096`, thinking on, no stop-at-eog | 7 sets x 20 prompts (base boxA, base boxB, hauhau boxB; blockD base/hauhau; blockF base/hauhau) | layer 46 stores the last prompt position only in prefill (handled as the last-layer quirk) | `glm47-flash-domain-routing/register_run/explore_*` |

No GLM domain prompt was ever run through generation. That is the missing cross-family experiment.

## Audit documents

`audit_20260828/` holds the 2026-08-28 five-agent recompute: `FINDINGS_expert_specialization_20260828.pdf`
(plain-language findings), `DATA_LEDGER_20260828.pdf` (every paper datum with a significance label),
`FROZEN_PREDICTIONS_20260828T2115.md` (predictions written before the sweep, scored in the findings),
and `agents/{A_primary35b,B_controls35b,C_122b,D_glm,E_register}.md` (per-subset recomputes with
file paths for every number).
