# 122B Domain Specialist Probe — Generation

This sub-bundle mirrors the domain-specialist probe on the same 122B model, but with full prefill plus generation capture.

## Intended Run

- Model: `Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive Q8_K_P`
- Mode: `prefill + generation`
- Prompt suite: `domain_specialist_probe_60_no_think.tsv`
- Default generation cap: `-n 2048`

## Purpose

- compare the same 60-domain specialist map across prefill and generation on 122B
- test whether the domain-leading experts from routing-only persist into generation
- measure any softmax-vs-DeltaNet shifts under long generation

## Layout

- `PROMPTS/`: localized prompt JSON and runtime TSV
- `scripts/`: run-local launcher and analyzers
- `RESULTS/`: summary reports and compact outputs
- `raw/`: remote raw capture artifacts

