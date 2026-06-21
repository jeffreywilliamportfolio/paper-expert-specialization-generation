# Qwen3.5-35B-A3B HauhauCS — Domain Expert Probe 3-Chunk

Long-prompt domain-cramming experiment on HauhauCS Qwen3.5-35B-A3B Q8_0.

## Scope

Takes the original 60 short no-think domain prompts from `qwen3.5-35b-a3b-huahua-6cond-moe-manips/` and collapses them into 3 long prompts (20 questions each), token-balanced to exactly 446 tokens using the live HauhauCS tokenizer. Tests how routing behavior changes when multiple domains are packed into a single long prompt.

Run: `20260410T173400Z` | 3 prompts × 2048 generated tokens | no-think, greedy (seed 42, temp 0)

**Headline results:**
- Generation uses a materially different expert set than prefill (top-W Jaccard as low as 0.0)
- E114 strengthens during generation vs. prefill — especially in the B and C chunks (rank ~7 in generation vs. rank 89 in prefill for chunk C)
- Routing entropy slightly *decreases* in generation relative to prefill (contrary to the simpler expectation)

- `METHOD/`: analysis scripts, prompt builder, C++ capture binary, utility scripts
- `PROMPTS/`: balanced TSV and prompt JSON
- `DOCS/`: experiment plan and full results with per-chunk W/S/Q and Jaccard tables
- `results/`: JSON results, per-token NPZ/TSV files, token audit summaries, generated text
- `raw/`: timestamped capture directory

## Reproducibility

- Yes: reanalysis of included `results/*.json` and per-token TSV/NPZ files
- No: end-to-end rerun (requires model artifact and instance)
- Note: `.npy` files in `raw/` are excluded from git

## Reading Order

1. [DOCS/PLAN.md](DOCS/PLAN.md)
2. [DOCS/RESULTS.md](DOCS/RESULTS.md)
3. [METHOD/analyze_domain_expert_probe_3chunk.py](METHOD/analyze_domain_expert_probe_3chunk.py)
