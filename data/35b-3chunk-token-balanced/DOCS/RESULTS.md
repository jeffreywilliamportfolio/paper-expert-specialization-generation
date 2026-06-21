# Results — Domain Expert Probe 3-Chunk

Run: `20260410T173400Z` | 3 prompts × 446 prompt tokens × 2048 generated tokens  
Full analysis: `results/results_domain_expert_probe_3chunk_20260410T173400Z.md`

## TL;DR

Long-prompt cramming (20 domain questions per prompt) produces a routing landscape where generation and prefill diverge substantially. E114 becomes a stronger generation expert than prefill predicts, especially for chunks B and C.

## Routing Entropy

| Phase | Mean Entropy |
|---|---|
| Prefill | 0.957522 |
| Generation trimmed | 0.952677 |

Entropy slightly *decreases* in generation (contrary to naive expectation). Per-chunk deltas all negative: A −0.0046, B −0.0062, C −0.0037.

## Prefill vs. Generation Expert Overlap (Jaccard)

| Chunk | Top-W Jaccard | Top-S Jaccard |
|---|---|---|
| A | 0.000000 | 0.066667 |
| B | 0.230769 | 0.230769 |
| C | 0.066667 | 0.000000 |

The expert set that dominates generation is mostly different from the expert set that dominates prefill, even with exactly matched token counts across chunks.

## Expert 114 — Prefill vs. Generation

| Chunk | Prefill W | Prefill rank | Gen W | Gen rank |
|---|---|---|---|---|
| A | 0.005756 | 30 | 0.005652 | 21 |
| B | 0.004381 | 82 | 0.005765 | 25 |
| C | 0.004358 | 89 | 0.006598 | 7 |

E114 consistently improves its rank from prefill to generation. Chunk C shows the strongest effect: rank 89 in prefill → rank 7 in generation.

## Generation Leaders (by W)

| Chunk | Top 4 experts |
|---|---|
| A | E169, E252, E94, E248 |
| B | E248, E210, E34, E100 |
| C | E146, E116, E28, E110 (E114 at rank 7) |

Chunk content still determines the dominant generation experts despite identical token lengths.

## Interpretation

When multiple domains are packed into one long prompt, routing during generation diverges from routing during prefill. E114's rise from weak prefill ranks to strong generation ranks (especially chunk C) suggests it tracks something about the generation phase itself — possibly introspective or recursive content in the B and C domain questions — rather than simply reflecting the prefill distribution.
