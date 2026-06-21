# Results: Domain Expert Probe 3-Chunk Huahua Run

This note summarizes the Apr 10 long-prompt domain-cramming experiment in the active Strangeloop bundle.

Goal:
- take the original 60 short no-think domain prompts from `qwen-huahua-6cond-moe-manips`
- collapse them into 3 long prompts
- keep the 3 long prompts token-balanced under the exact Hauhau/Qwen tokenizer
- compare prefill versus generation routing behavior on those long prompts

Model/setup:
- model: `Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive Q8_0`
- binary: pinned `llama.cpp` capture build `8493 (1772701f)`
- capture mode: router-only
- generation run: `-n 2048`, `--seed 42`, `--temp 0`, `--top-k 1`

## Prompt construction

Source prompt surface:
- `qwen3.5-35b-a3b-huahua-6cond-moe-manips/PROMPTS/domain_expert_probe_60_no_think.tsv`

Local builder:
- [build_domain_expert_probe_3chunk_no_think.py](../METHOD/build_domain_expert_probe_3chunk_no_think.py)

Generated prompt artifacts:
- [domain_expert_probe_3chunk_prompts.json](../PROMPTS/domain_expert_probe_3chunk_prompts.json)
- [domain_expert_probe_3chunk_no_think.tsv](../PROMPTS/domain_expert_probe_3chunk_no_think.tsv)

Construction rule:
- `20A`: all `_01` prompts in original domain order
- `20B`: all `_02` prompts in original domain order
- `20C`: all `_03` prompts in original domain order
- within each long prompt, source questions are separated by literal `\n`

## Token balancing

The first unbalanced long prompts landed at:
- `A`: `446`
- `B`: `372`
- `C`: `401`

To make the long prompts directly comparable, I balanced them against the exact live Hauhau/Qwen tokenizer using the remote `capture_activations` binary.

Balanced token audit:
- [20260410T171900Z_domain_expert_probe_3chunk_balanced_token_audit_summary.md](20260410T171900Z_domain_expert_probe_3chunk_balanced_token_audit_summary.md)
- [20260410T171900Z_domain_expert_probe_3chunk_balanced_token_audit_summary.json](20260410T171900Z_domain_expert_probe_3chunk_balanced_token_audit_summary.json)

Final balancing method:
- append repeated `" ."` at the end of the concatenated prompt before `<|im_end|>`
- `A`: padding count `0`
- `B`: padding count `73`
- `C`: padding count `44`

Final verified prompt-token counts:
- `domain_expert_probe_20A_chunk`: `446`
- `domain_expert_probe_20B_chunk`: `446`
- `domain_expert_probe_20C_chunk`: `446`

That balance was verified again immediately before the actual generation run.

## Main run

Primary capture:
- remote run dir: `/workspace/consciousness-experiment/experiments/qwen3.5-35b-a3b-huahua-strangeloop/results/20260410T173400Z_domain_expert_probe_3chunk_balanced_gen_n2048`

Run metadata:
- `n_predict`: `2048`
- `seed`: `42`
- `temp`: `0`
- `top_k`: `1`
- mode: generation + prefill in one capture

Per-prompt counts:
- `domain_expert_probe_20A_chunk`: `446` prompt, `2048` generated
- `domain_expert_probe_20B_chunk`: `446` prompt, `2048` generated
- `domain_expert_probe_20C_chunk`: `446` prompt, `1838` generated

## Analysis artifacts

Bundle-local analyzer:
- [analyze_domain_expert_probe_3chunk.py](../METHOD/analyze_domain_expert_probe_3chunk.py)

Results:
- [results_domain_expert_probe_3chunk_20260410T173400Z.md](results_domain_expert_probe_3chunk_20260410T173400Z.md)
- [results_domain_expert_probe_3chunk_20260410T173400Z.json](results_domain_expert_probe_3chunk_20260410T173400Z.json)

What the analyzer reports:
- mean normalized routing entropy
- pooled expert routed weight `W`
- pooled selection rate `S`
- pooled conditional weight `Q`
- prefill versus generation-trimmed comparisons
- top-expert overlap via Jaccard on top-`W` and top-`S`
- explicit `E114` tracking

## Main findings

### 1. Generation entropy is slightly lower than prefill

Overall:
- mean prefill entropy: `0.957522`
- mean generation-trimmed entropy: `0.952677`

Per chunk:
- `A`: `0.957452 -> 0.952893`, delta `-0.004559`
- `B`: `0.956576 -> 0.950346`, delta `-0.006230`
- `C`: `0.958537 -> 0.954792`, delta `-0.003745`

So on these long crammed prompts, generation is modestly less entropic than prefill, not more.

### 2. Prefill and generation do not preserve the same leading experts

Top-expert overlap is low.

Top-`W` Jaccard, prefill vs generation-trimmed:
- `A`: `0.000000`
- `B`: `0.230769`
- `C`: `0.066667`

Top-`S` Jaccard, prefill vs generation-trimmed:
- `A`: `0.066667`
- `B`: `0.230769`
- `C`: `0.000000`

That is the clearest routing result from the run. Even with prompt tokens exactly matched across the 3 long prompts, the dominant expert set during generation is mostly different from the dominant expert set during prefill.

### 3. E114 strengthens during generation, especially in `B` and `C`

`A` chunk:
- prefill `W/S/Q`: `0.005756 / 0.046076 / 0.103020`
- generation-trimmed `W/S/Q`: `0.005652 / 0.046484 / 0.108754`
- prefill rank by `W/S`: `30 / 28`
- generation rank by `W/S`: `21 / 13`

`B` chunk:
- prefill `W/S/Q`: `0.004381 / 0.034585 / 0.113229`
- generation-trimmed `W/S/Q`: `0.005765 / 0.046411 / 0.107316`
- prefill rank by `W/S`: `82 / 82`
- generation rank by `W/S`: `25 / 21`

`C` chunk:
- prefill `W/S/Q`: `0.004358 / 0.033913 / 0.107973`
- generation-trimmed `W/S/Q`: `0.006598 / 0.050571 / 0.106468`
- prefill rank by `W/S`: `89 / 91`
- generation rank by `W/S`: `7 / 6`

Interpretation:
- `A` shows only a modest E114 lift
- `B` shows a large E114 lift from weak prefill ranks to mid-generation ranks
- `C` shows the strongest effect, where E114 becomes a top-generation expert despite weak prefill presence

### 4. The dominant generation experts differ by chunk

Generation-trimmed leaders by `W`:
- `A`: `E169`, `E252`, `E94`, `E248`
- `B`: `E248`, `E210`, `E34`, `E100`
- `C`: `E146`, `E116`, `E28`, `E110`, with `E114` reaching rank `7`

This suggests the long-prompt compression does not collapse everything into one single stable specialist set. The chunk contents still matter.

## Short interpretation

The long-prompt cramming manipulation does change routing behavior in a meaningful way:
- the three prompts were exactly token-matched at `446`
- generation uses a meaningfully different expert set than prefill
- routing entropy decreases slightly in generation
- E114 becomes stronger during generation than it appears during prefill, especially for the `B` and `C` long prompts

The cleanest takeaway is not just “entropy changes.” It is that once multiple domains are packed into one long prompt, the expert-selection landscape during generation shifts away from the prefill landscape, and that shift is large enough to materially change E114’s standing.

## Primary references

- balanced TSV: [domain_expert_probe_3chunk_no_think.tsv](../PROMPTS/domain_expert_probe_3chunk_no_think.tsv)
- prompt metadata: [domain_expert_probe_3chunk_prompts.json](../PROMPTS/domain_expert_probe_3chunk_prompts.json)
- balanced token audit: [20260410T171900Z_domain_expert_probe_3chunk_balanced_token_audit_summary.md](20260410T171900Z_domain_expert_probe_3chunk_balanced_token_audit_summary.md)
- main analysis: [results_domain_expert_probe_3chunk_20260410T173400Z.md](results_domain_expert_probe_3chunk_20260410T173400Z.md)
