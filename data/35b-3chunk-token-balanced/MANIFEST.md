# Manifest

Included:

- `METHOD/`: `analyze_domain_expert_probe_3chunk.py`, `analyze_domain_expert_probe_3chunk_per_token.py`, `build_domain_expert_probe_3chunk_no_think.py`, `qwen_router.py`, `capture_activations.cpp`, `bootstrap_remote_instance.sh`
- `PROMPTS/`: `domain_expert_probe_3chunk_no_think.tsv` (balanced, 446 tokens each), `domain_expert_probe_3chunk_prompts.json`
- `DOCS/`: `PLAN.md`, `RESULTS.md`
- `results/`: `results_domain_expert_probe_3chunk_20260410T173400Z.json`, `.md`, per-token NPZ/TSV files, token audit summaries, `generated-text.md`
- `raw/20260410T173400Z_domain_expert_probe_3chunk_balanced_gen_n2048/`: per-chunk subdirs with `generated_text.txt`, `metadata.txt` (`.npy` excluded from git)

Raw data:

- `.npy` router tensors: excluded from git (large binary files in `raw/` subdirs)

Reproducibility:

- Yes: reanalysis of included JSON and per-token TSV/NPZ files
- No: end-to-end rerun from scratch
