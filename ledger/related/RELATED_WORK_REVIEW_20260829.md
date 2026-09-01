# Related work review: five 2026 papers on expert specialization, read against our result

2026-08-29. Five papers read in full (arXiv HTML/PDF, code where public) by one reader each against a
shared brief (`BRIEF.md`); per-paper reports with verbatim quotes are alongside this file. This document
is the compilation. Every quote below is verbatim from the paper named; anything else is our reading.

## 1. The one-table summary

| Paper | Models (closest to ours) | Unit of "expert usage" | Prefill vs generation | Domain tested in generation? | Causal test |
|---|---|---|---|---|---|
| **Myth of Expert Specialization** (Wang, Hayou, Nalisnick; 2604.09780; 10 Apr 2026) | gpt-oss-20b, Qwen3-30B-A3B, ERNIE-4.5-21B, DeepSeek-V2-Lite, Moonlight, Trinity-Mini, Ling-mini | visit-count frequency vector per sequence, cosine / top-P Jaccard; mostly middle layer | **Separates them; central claim.** Real rollouts on Ling-mini (3 hand-picked pairs) and gpt-oss-20b (1 pair). Rollout curves are cumulative prompt+rollout, never a generation-only block | **No.** Domain only on prefilled static text (OpenWebText / Math / Code / Logic). One generation pair differs in concept (GAN vs VAE); no pooled statistic | Prune to 12 of 32 experts in last k layers: <10% NLL rise on prompts, "substantially" more on completions |
| **Illusion of Specialization / Standing Committee** (Wang, Xu, Shen, Su, Huang, Zhu; 2601.03425; 6 Jan 2026) | Qwen3-30B-A3B, DeepSeek-V2-Lite, OLMoE | full-softmax expected weight (ECI) at the **last prompt token** of each MMLU question, pooled by 9 domains, per layer | **Prefill only**, narrowly: one routing vector per prompt at its final token. "generation" occurs 0 times | No | None ("We do not directly intervene in routing") |
| **Do Domain-specific Experts exist?** (Do, Le, Tran; 2604.05267; 7 Apr 2026) | 10 models, 16 to 512 experts; Qwen3-30B-A3B Instruct + Thinking, Qwen3-Next-80B, gpt-oss-20b/120b | binary top-k membership counts over question tokens, saliency-weighted; all layers jointly | **Prefill only** for identification: "we identify domain-specific tokens using only the question tokens, excluding answer tokens"; repo has no `generate` call in the analysis path. Steering then applies at generation | No routing statistic on generation; evidence is downstream accuracy under steering (Math, Bio, Phys, Chem) | Upweight top-K scored experts (alpha 3 to 5): +1.5 to +14.5 pts MMLU-Pro; never suppression |
| **Polysemantic Experts, Monosemantic Paths** (Ye, Yuan, Sharkey; 2604.17837; 20 Apr 2026; ICLR Re-Align workshop) | Qwen3-30B-A3B, gpt-oss-20b, GLM-4.5-Air, OLMoE, Granite-4-Tiny, DeepSeek-V2-Lite | per-token **top-1** expert path across a band of layers; no token-set pooling, no weights | **Prefill only**: forward passes over 2M / 10M tokens of C4 + HPLT | Domain is not a variable; they probe language, token id, position (all router-blind) | None; "causal" = SVD orthogonality guarantee plus probes |
| **The Expert Strikes Back** (Herbst, Wermter, Lee; 2604.02178; v2 15 May 2026; ICML 2026) | 12 models incl. Qwen3-30B-A3B, GLM-4.7-Flash, gpt-oss-20b, Mixtral | k-sparse probes on expert neurons; JSD of an expert's token-cluster distribution vs layer base rate; activity = g·‖E(x)‖, explicitly not router weight | **Prefill only**: pile / Wikipedia / LLM-written sentences fed as input; DLA to a target word already in the prompt | Domains = unsupervised unembedding clusters on corpus text; finding: fine-grained (k=5000) beats broad (k=10) | DLA rank attribution only; their Impact Statement asks for ablations |

**The pattern.** Four of five never measure routing on a token the model produced. The fifth (Myth) does,
on four hand-picked pairs, pooled cumulatively with the prompt, and draws exactly our premise from it:
"prefill-phase expert usage is not a reliable proxy for full generation usage pattern ... characterizing
specialization requires observing complete trajectories." None of the five tests whether generation
routing is organized by subject. That is the gap the paper sits in, and it is narrower and more
defensible than "experts specialize."

## 2. Verdicts on our six findings

| Finding | Myth | Standing Committee | Do et al. | Poly Paths | Strikes Back |
|---|---|---|---|---|---|
| 1. Prefill default expert; generation 20 winners | AGREES ("nearly identical experts during prefilling, but diverge during generation") | AGREES prefill half; SILENT generation | SILENT; implicitly acknowledges flat frequent experts ("active frequently but only process common, non-informative tokens") | SILENT; prefill "collapses many tokens onto the same expert" | SILENT |
| 2. Same subject, same crew; 95-96% identification | SILENT, consistent in spirit | SILENT | SILENT | SILENT ("Paths cluster by meaning, not surface form" is the nearest) | SILENT on generation; DISAGREES in spirit on corpus text ("do not represent broad semantic domains") |
| 3. Register moves crews; dialect a no-op | PARTLY AGREES (plain vs math language pair diverges in rollout) | SILENT | SILENT | AGREES on dialect (language is router-blind) | SILENT |
| 4. Long coherent prompt shares the crew | SILENT | SILENT | SILENT (their success is consistent with it) | SILENT | SILENT, but their whole corpus is this regime |
| 5. Leaders removable, not load-bearing | DISAGREES in emphasis (whole-layer pruning raises completion NLL) | SILENT | PARTLY DISAGREES in spirit (upweighting one expert moves accuracy 3 to 45%) | SILENT | SILENT |
| 6. Fine-tunes never touched routers | SILENT (chat template changes hidden states, not routers) | SILENT | SILENT | SILENT | SILENT |

No paper contradicts findings 1 to 4 and 6 with data. Finding 5 draws two "disagree in emphasis"
marks, both from different operations (prune 20 of 32 experts and read NLL; upweight a prefill-scored
expert and read accuracy) on different experts. Our sentence should say: the four generation leaders
are removable for the answer; this does not speak to whole-layer pruning or to gains from upweighting.

## 3. What each paper hands a reviewer, and the reply

1. **"Wang et al. already showed prompt routing does not predict rollout routing."** Yes, on three
   Ling-mini pairs and one gpt-oss pair, with similarity pooled over prompt plus rollout and no
   statistic. We separate the blocks, run 20 subjects in two designs, and test identification with
   four nulls. Cite them as the premise.
2. **"Router collapse is a gpt-oss reasoning-preamble artifact; ERNIE-PT shows none."** Our flat
   default appears on a non-reasoning fine-tune and on the base Qwen3.5-35B-A3B; say so and cite their
   ERNIE-PT null as the contrast.
3. **"Standing Committee reports cross-domain Jaccard 0.87 on Qwen3-30B; your 0.51 and 0.03 look like
   another regime."** Their 0.87 is nine pooled MMLU domains, full-softmax ECI, last prompt token; our
   0.51 is 20 subjects, routed weight, all prompt tokens. Both are prefill; both find one dominant set;
   their Qwen minimum (0.53) is in our range. Nothing they report bears on produced tokens.
4. **"Domain terminology rarely stabilizes on any expert (their Fig. 7)."** That panel is prompt tokens
   of an MMLU question; it tests whether content words join the committee, not whether generated content
   has stable crews.
5. **"Do et al. find domain experts from prefill and steering works, so prefill misses nothing."**
   Their identification is question text with a correction that discounts frequent experts on common
   tokens; gains show a prefill-scored expert can help, not that prefill routing is organized by
   subject. Their own cost analysis: identification is O(L) forward passes over questions, contrasted
   with RICE which reads generated thinking tokens.
6. **"The committee-of-specialists model has been largely invalidated (Ye et al.)."** Said of single
   experts, top-1 routing, prefill over web text. Our unit is a top-8 crew over a generation token set.
   Their own line that a single routing decision "collapses many tokens onto the same expert" matches
   our prefill default.
7. **"Herbst et al. show with a multinomial null that specialization is fine-grained operation, not
   domain."** On corpus text presented as input, which is our prefill condition, where our numbers agree
   with theirs. Our scoping result says long coherent input text shares the crew across blocks, which is
   all they use. And "subject" in our test is crew identity across independent designs (95-96% vs 6%),
   not a human label imposed on tokens.
8. **"Router weight is a poor activity measure" (Herbst).** Fair. Our crew is a routing-side object
   (their "Routing Specialization (Input)"); say so, do not claim it measures what the expert writes. A
   residual-write-norm crew is a cheap follow-up if we ever have the tensors.
9. **"Top-1 vs top-8; visit counts vs mean routed weight."** State once that overlap numbers across
   papers are not directly comparable.
10. **"Standing committees are per-layer; you pool over 40 layers."** State the pooling; per-layer
    capture exists (F6 in the findings: 35B flat across layers) and one per-layer panel would answer it.

## 4. Vocabulary

Reuse, with attribution: **router collapse** (Myth, for the prefill same-expert behaviour); **standing
committee**, **generalist core**, **core-periphery** (Standing Committee); **domain-agnostic experts**
(Do et al.); **Routing Specialization (Input)** vs **Functional Specialization (Output)** (Herbst);
**rollout** for generated tokens (Myth; we say "generation block", either is fine).

Avoid: **polysemantic expert** (Ye; a concept-purity claim about one expert, not ours); **path /
trajectory** except in Ye's per-token sense; **Illusion of Specialization** as a frame; **expert
homogenization**, **representation collapse**, **specialization collapse** (training-time failures,
different things); **thinking experts** (RICE); **generalist** as a bare noun (none of the five use it;
"default expert" or "generalist core" reads cleaner).

## 5. What none of them did, and we did

Generation-only block with the prompt removed; subject as an explicit 20-way axis at generation time;
within-answer crew turnover at subject boundaries; cross-design identification with chance,
position-matched, drift and label-shuffle nulls; register and dialect on matched text; removal (not
upweighting) of generation-identified leaders with the answer checked; base vs fine-tune router
identity; a 256-expert top-8 model. What they did and we did not: theory tying routing to hidden-state
geometry (Myth); many families with dense controls (Myth, Herbst, Ye); per-neuron probes and
auto-interp labels (Herbst); a formal committee extraction with a top-k sweep (Standing Committee);
downstream accuracy under steering on ten models (Do).

## 6. Draft related-work paragraph (for Jeffrey to rewrite; plain prose, no em-dashes)

Recent work on expert specialization has mostly measured routing on text supplied as input. Wang,
Xu and colleagues audit routing at the last prompt token of MMLU questions on three models and find a
small standing committee of experts holding most of the routing mass in every domain, with
cross-domain Jaccard near 0.87. Herbst, Wermter and Lee probe expert neurons on corpus text across
twelve models and conclude that experts are fine-grained task experts rather than broad domain
specialists. Ye, Yuan and Sharkey show that single experts are polysemantic on web text while
per-token paths across layers cluster by function. Do, Le and Tran score experts from question tokens
alone and show that upweighting the top-scored experts raises accuracy on ten models. Each of these
characterizes what we call the prefill block, and our prefill numbers agree with them. Wang, Hayou and
Nalisnick are the exception: they track routing from prompt into rollout on paired queries and report
that prompt-level routing does not predict rollout-level routing, concluding that specialization
requires observing complete trajectories. They do not ask whether the rollout divergence is organized
by subject. That is the question here.

## 7. Two things worth adding to the paper because of this read

- Myth's rollout curves pool the prompt with the rollout. Our generation block excludes the prompt.
  One sentence saying so explains why their divergence appears gradual and ours is present in the
  first 64 generated tokens.
- Myth attributes prefill collapse to reasoning preambles and sees none on ERNIE-PT. Our F11 (base
  checkpoint, no reasoning) and the flat default on a non-reasoning fine-tune are the direct answer;
  they belong in the base paragraph.

Per-paper reports: `myth_2604.09780.md`, `standing_committee_2601.03425.md`, `domain_experts_2604.05267.md`,
`polysemantic_paths_2604.17837.md`, `expert_strikes_back_2604.02178.md`.
