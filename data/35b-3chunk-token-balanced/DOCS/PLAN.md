# Plan — Domain Expert Probe 3-Chunk Long-Prompt Cramming

## Goal

Test how routing changes when multiple domain prompts are crammed into a single long prompt, versus the per-prompt analysis in `../qwen3.5-35b-a3b-huahua-expert-identification/`. Specifically: does E114 behave differently during generation of a densely multi-domain prompt vs. a single-domain prompt?

Source: 60 short no-think prompts from `../qwen3.5-35b-a3b-huahua-6cond-moe-manips/PROMPTS/domain_expert_probe_60_no_think.tsv`

## Model and Hardware

- Model: HauhauCS Qwen3.5-35B-A3B Q8_0
- Binary: llama.cpp capture build 8493 (1772701f)
- Runtime: no-think, greedy, seed 42, temp 0, top-k 1, `-n 2048`
- Capture mode: router-only

## Prompt Construction

Collapse 60 source prompts into 3 long prompts:
- `20A`: all `_01` prompts across 20 domains in domain order
- `20B`: all `_02` prompts across 20 domains in domain order
- `20C`: all `_03` prompts across 20 domains in domain order

Within each long prompt, source questions are separated by literal `\n`.

## Token Balancing

Balance all 3 long prompts to exactly the same token count using the live HauhauCS tokenizer, to make cross-chunk comparisons valid. Method: append repeated `" ."` padding before `<|im_end|>`. Final balanced count: **446 tokens per chunk** for all 3.

Verify balance immediately before the actual generation run.

## Measurements

- Routing entropy (normalized, `log2(256)`)
- Per-expert W, S, Q pooled across all layers
- Prefill vs. generation-trimmed comparison
- Top-expert Jaccard overlap between prefill and generation for top-W and top-S
- Explicit E114 tracking: W, S, Q, and rank in each track

Per-token analysis via `METHOD/analyze_domain_expert_probe_3chunk_per_token.py`.
