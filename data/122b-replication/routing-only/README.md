# 122B Domain Specialist Probe — Routing Only

This sub-bundle isolates the completed 122B domain-specialist probe run in routing-only mode.

## Run

- Model: `Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive Q8_K_P`
- Mode: `prefill-only / routing-only`
- Prompt suite: `domain_specialist_probe_60_no_think.tsv`
- Run id: `20260412T160341Z_qwen122_domain_specialist_probe_60_routing_only`

## Purpose

- reuse the established 60-domain specialist suite on the 122B model
- measure prefill routing only
- identify domain-leading experts and compare them against `E114`
- keep the artifact set separate from the diectic baseline

## Layout

- `PROMPTS/`: localized prompt JSON and runtime TSV
- `scripts/`: run-local launcher and analyzer
- `RESULTS/`: summary reports and compact outputs
- `raw/`: remote raw capture artifacts

