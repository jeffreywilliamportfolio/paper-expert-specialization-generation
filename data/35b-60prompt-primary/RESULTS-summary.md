# RESULTS

## Scope

This experiment runs the 60-prompt domain specialist set in [prompts/domain_specialist_probe_60_no_think.tsv](/Users/jeffreyshorthill/llama-eeg-tests/experiments/qwen-huahua-expert-identification/prompts/domain_specialist_probe_60_no_think.tsv) against HauhauCS Qwen3.5-35B-A3B Q8_0 and analyzes all 256 experts across prefill and generation.

Primary run:

- Run id: `20260408T235839Z`
- Model: HauhauCS `Qwen3.5-35B-A3B` Q8_0
- Runtime: no-think, greedy, generation `-n 2056`
- Prompt set: `60` prompts, `20` domains, `3` prompts per domain
- Capture dir: remote `/workspace/consciousness-experiment/experiments/qwen-huahua-expert-identification/captures/20260408T235839Z_domain_specialist_no_think_hauhau`

Primary result artifacts:

- [results_domain_specialists_20260408T235839Z.md](/Users/jeffreyshorthill/llama-eeg-tests/experiments/qwen-huahua-expert-identification/results/results_domain_specialists_20260408T235839Z.md)
- [results_domain_specialists_20260408T235839Z.json](/Users/jeffreyshorthill/llama-eeg-tests/experiments/qwen-huahua-expert-identification/results/results_domain_specialists_20260408T235839Z.json)
- [results_domain_specialists_20260408T235839Z.npz](/Users/jeffreyshorthill/llama-eeg-tests/experiments/qwen-huahua-expert-identification/results/results_domain_specialists_20260408T235839Z.npz)

## Capture

- Cells processed: `60/60`
- Prompt tokens: `2404`
- Generated tokens: `110412`
- Total tokens: `112816`
- Total runtime: `971685 ms` (`~16.2 min`)
- Layer-39 trim events: `60`
- Missing prefill-layer events: `0`
- Missing generation-layer events: `0`
- Identity residual:
  - prefill: `2.78e-17`
  - generation all: `5.55e-17`
  - generation trimmed: `5.55e-17`

This run used one generation capture for both tracks:
- prefill from `arr[:n_tokens_prompt]`
- generation from `arr[n_tokens_prompt:]`

## Overall

Overall top experts by routed weight:

| Track | Rank 1 | Rank 2 | Rank 3 | E114 |
|---|---|---|---|---|
| Prefill | `224` (`W=0.012708`, `S=0.078610`) | `243` (`0.010609`, `0.067567`) | `56` (`0.009227`, `0.052108`) | `W=0.003482`, `S=0.028423`, `Q=0.109399` |
| Generation all | `210` (`W=0.006228`, `S=0.048215`) | `146` (`0.006170`, `0.047122`) | `114` (`0.005761`, `0.043537`) | `W=0.005761`, `S=0.043537`, `Q=0.108469` |
| Generation trimmed | `146` (`W=0.006356`, `S=0.047994`) | `210` (`0.006054`, `0.047556`) | `114` (`0.005882`, `0.044590`) | `W=0.005882`, `S=0.044590`, `Q=0.108259` |

Winner concentration by domain:
pul
- Prefill winner-by-`W` is highly concentrated: expert `224` wins `18/20` domains, with expert `103` winning `linguistics` and expert `130` winning `chemistry`.
- Generation is fully dispersed: winner-by-`W` spans `20` distinct experts across `20` domains in both `generation_all` and `generation_trimmed`.

Expert 114 summary:

- Prefill:
  - no domain wins
  - top-10 by `W` only in `philosophy` at rank `5`
- Generation all:
  - wins `philosophy`
  - top-10 by `W` in `archaeology`, `comparative_religion`, `linguistics`, `philosophy`, `physics`, and `political_science`
- Generation trimmed:
  - wins `philosophy`
  - top-10 by `W` in `archaeology`, `comparative_religion`, `linguistics`, `philosophy`, `physics`, `political_science`, and `psychology`

## Domain Highlights

These are compact examples from the analyzed domain tables. They are not exhaustive; the full per-domain tables are in the main report JSON/Markdown.

- `philosophy`
  - prefill winner-by-`W`: `224`
  - generation winner-by-`W`: `114`
  - generation winner-by-`S`: `114`
  - generation candidate-by-`W` rank 1: `114` with `W=0.018755`
  - generation candidate-by-`S` rank 1: `114` with `S=0.097268`

- `computer_science`
  - prefill winner-by-`W`: `224`
  - generation winner-by-`W`: `206`
  - generation winner-by-`S`: `206`
  - generation candidate-by-`W` rank 1: `206` with `W=0.010256`
  - generation candidate-by-`S` rank 1: `206` with `S=0.068895`

- `mathematics`
  - prefill winner-by-`W`: `224`
  - prefill winner-by-`S`: `130`
  - generation winner-by-`W`: `100`
  - generation winner-by-`S`: `100` in `generation_all`, `116` in `generation_trimmed`
  - generation candidate-by-`W` rank 1: `100`
  - generation candidate-by-`S` rank 1: `100`

- `medicine`
  - prefill winner-by-`W`: `224`
  - generation winner-by-`W`: `152`
  - generation winner-by-`S`: `82`
  - generation candidate-by-`W` rank 1: `152`
  - generation candidate-by-`S` rank 1: `154`
  - this is one of the clearest domains where `W` and `S` do not select the same top candidate expert

- `history`
  - prefill winner-by-`W`: `224`
  - generation winner-by-`W`: `158`
  - generation winner-by-`S`: `224`
  - generation candidate-by-`W` rank 1: `110`
  - generation candidate-by-`S` rank 1: `224`

- `comparative_religion`
  - prefill winner-by-`W`: `224`
  - generation winner-by-`W`: `170`
  - generation winner-by-`S`: `170`
  - generation candidate-by-`W` rank 1: `198`
  - generation candidate-by-`S` rank 1: `170`
  - E114 is still high here in generation: rank `2` by `W`

## Notes

- The specialist analyzer now mirrors the candidate-specialist ranking for both `W` and `S`, not just `W`.
- The current run is no-think only. The think-mode TSV exists at [domain_specialist_probe_60_think.tsv](/Users/jeffreyshorthill/llama-eeg-tests/experiments/qwen-huahua-expert-identification/prompts/domain_specialist_probe_60_think.tsv), but it has not been run yet.
- For full per-domain tables, E114 comparison sections, subtype summaries, and domain-overlap Jaccard tables, use the main report and JSON rather than this summary file.
