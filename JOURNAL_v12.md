# Working journal: expert-specialization paper, v1.2

Shared working notes, Jeffrey + Claude. Newest entries at the bottom.

## 2026-08-28 (Thu night into Fri)

- Audit of v1.1 against the archive: every number recomputes. Findings PDF, data ledger, frozen
  predictions and five agent reports in `ledger/`. Curated data repo built at
  `/Volumes/ExternalSSD/expert-specialization-data/` (commit f06cd3e).
- Found Wang, Hayou and Nalisnick, "The Myth of Expert Specialization in MoEs" (arXiv 2604.09780,
  April 2026). Verbatim: "semantically different prompts activate nearly identical experts during
  prefilling, but diverge during generation"; "prefill-phase expert usage is not a reliable proxy for
  full generation usage pattern." They never test whether generation diverges *by subject*. That is
  our question.
- New second result, from tensors already on disk (`ledger/CREW_REPRODUCIBILITY_20260828.md`):
  a subject's generation expert set identifies the subject across independent prompt designs at
  95-96% (chance 6%); prefill rows of the same questions 15-17%; not position (0.02 vs 0.19), not
  drift (83% against neighbours alone). Misses concentrate in the biography chunk, where the
  answer's register changed.
- Base Qwen (official ggml-org Q8_0) replicates the Table 1 shape on the 50-pair run and shares
  the same expert sets as HauhauCS across dialect and model (legal 122/109/85, medical 206/247).
- Scoping result: on a long coherent technical prompt (l1l3 base), prefill and generation share
  the set (Jaccard 0.6-0.78). The regime effect is a content/register effect that short-prompt
  prefill exposes. Abstract now says "for short mixed prompts."
- Archive bugs to disclose in v1.2: 122B last-layer prefill artifact (routing-only 13/20 is clean);
  122B trimmer no-op (~45% spill in the generation block); E48 is a spill expert; HVAC 180 = 18 x 10.
- Not necessary for v1.2: the 20-domain probe on base. One box-hour, next week or later.

### Abstract, v1.2 draft 1 (Jeffrey, 2026-08-28 23:10)

Sparse Mixture-of-Experts models contain a fixed pool of experts at each MoE layer. During both
prompt prefill and autoregressive generation, a learned router selects a small subset of experts
for each token, based on its current hidden representation.

Prior work finds that semantically different prompts can converge on nearly identical expert usage
during prefill and diverge during generation, making complete trajectories necessary to
characterize specialization. We ask whether that rollout divergence is organized by the local
subject of the answer.

We analyze five checkpoints across the Qwen3.5 and GLM-4.7-Flash families. In a Qwen3.5-35B-A3B
model, we test whether the selected expert sets follow the local subject of an answer by packing
multiple subjects into one generation, reordering them across prompts, controlling for position,
and comparing them with the same questions asked independently.

Position-resolved prefill routing primarily follows the wording of the question, while pooling
across the prompt produces the shared usage pattern reported in prior work. Across packed and
independently asked versions of identical questions, generation-based expert sets identify the
subject at 95–96% in Qwen3.5-35B, compared with 15–17% for prefill and 6% by chance.
Generation-time expert sets remain stable within a subject, change sharply at subject boundaries,
and recur for the same subject across positions and prompt designs. Together, these results show
that for short mixed prompts, prefill carries the wording of the question, while generation
carries the subject of the answer.

Open on the abstract: whether to name HauhauCS in it (Methods must either way); "expert sets" is
the term, "crews" stays out of the paper.

### Next
- v1.2 body: Results 3.3 = the identification table + four nulls; Limitations = winner fragility
  at n=3, the 122B disclosures, the long-prompt scoping; Discussion = register paragraph, base
  paragraph, Wang et al.
- Small corrections list in `ledger/FINDINGS_expert_specialization_20260828.pdf` §6.

## 2026-08-29 10:40 — reruns from raw tensors (Claude)

Asked: is anything worth rerunning from the raw tensors before v1.2? Two candidates; one ran, one can't.

**1. Token-balanced Jaccard, set size.** The paper's Section 3 text says "top-expert sets" without a
size; the audit flagged the archived numbers as top-8 while the rest of the paper uses top-15.
Recomputed from `expert-specialization-data/qwen35b/token_balanced_3chunk/results/per_token_20260410T173400Z/*.npz`
(mean W/S over positions, generation trimmed at first EOG id; C trims at 1812 of 1838, A and B hit the cap):

| chunk | top-8 W | top-8 S | top-15 W | top-15 S | top-30 W | top-30 S |
|---|---|---|---|---|---|---|
| A | 0.000 | 0.067 | 0.071 | 0.071 | 0.176 | 0.111 |
| B | 0.231 | 0.231 | 0.200 | 0.154 | 0.277 | 0.304 |
| C | 0.067 | 0.000 | 0.111 | 0.154 | 0.224 | 0.224 |

Top-8 reproduces the paper's 0.00/0.23/0.07 (W) and 0.07/0.23/0.00 (S) exactly. The conclusion is the
same at every set size. Smallest fix: say "top-eight" in Section 2.4 and the Figure 3 caption; or
switch the paper to the top-15 row so it matches the HVAC/register overlaps. Jeffrey's call.

**2. Leave-one-prompt-out on the 60-prompt generation winners.** Not possible from local data: no
per-cell router tensors for that run survive anywhere on the drive (agent A confirmed; only the
20x256 domain-pooled W/S/Q arrays exist). Winner fragility stays a qualitative statement (F3) unless
the base 20-domain probe is re-captured with per-cell tensors kept (the one-box-hour run, next week).

Nothing else is worth rerunning: 122B has no tensors, GLM was never generated, and the HVAC / l1l3 /
AAVE base controls were recomputed 8-28 with paths logged.

## 2026-08-29 10:55 — base-model (non-HauhauCS) tensors: a domain probe we never scored (Claude)

Jeffrey pointed out there are base tensors on the drive. Swept every capture dir with router files:
the base sets are l1l3 (done 8-28), AAVE 5-5 and 5-15 medical (done 8-28), and four E114/self-reference
sets with no domain axis. One set had never been scored for this paper:

`sae-tests/runs/base_qwen35_a3b_base_safety_smoke_20260429T1925Z/all_router_gen/..._financial_vs_consequence_theory_..._greedy_all_router_gen`
Base Qwen3.5-35B-A3B Q8_0, 2026-04-29, greedy, 12 prompts = 5 domains (finance x4, medical, physical,
professional, legal x2 each) x 2 registers (consequence-laden vs neutral), 45–51 prompt tokens,
256 generated (two professional cells end at 212/230), all 40 router layers. Script and output:
`expert-specialization-data/audit_20260828/base_safety_smoke_crews.py` + `_OUT.txt`.

Top-8 W crews, layer-pooled, generation trimmed at EOG:

| block | within-domain Jaccard (10 pairs) | between-domain (56 pairs) | nearest-neighbour same domain |
|---|---|---|---|
| prefill | 0.515 | 0.473 | 4/12 |
| generation | 0.457 | 0.035 | **11/12** (chance 1/11) |

Same shape as the fine-tune: prefill overlaps everything with everything (E95 wins 5 of 12 cells,
E224 wins 3), generation separates by subject. Two extras this set gives for free:
- **Register crossed inside domain, on base.** Each domain pairs a consequence-framed prompt with a
  neutral one; generation still pairs by domain (11/12), so subject beats this register contrast.
  (Not a contradiction of F4: the HVAC register axis was technical vs first-person experience, a
  much larger shift than consequence vs neutral.)
- **Same legal crew as the AAVE base run.** Legal generation winner is E122 in both cells here;
  the 8-28 AAVE base analysis had the legal crew as 122/109/85. Same subject, same expert, across
  two unrelated prompt sets on the base checkpoint.

Caveats: n=2 per domain, 256-token answers, and finance has 4 cells so it dominates the within mean
(finance-only within = 0.492; the other four pairs = 0.405). Good for the base paragraph in
Discussion, not for a table of its own.

- 11:00 Findings v2 rendered: `ledger/FINDINGS_expert_specialization_v2_20260829.pdf` (+html; copy in expert-specialization-data/audit_20260828/). Adds F2a identification table + nulls, F11 base replication incl. safety smoke, top-15 Jaccard, changelog. v1 left untouched.

- 11:30 Related-work read of five 2026 papers (Myth, Standing Committee, Do et al., Poly Paths, Expert Strikes Back): `ledger/related/RELATED_WORK_REVIEW_20260829.md` + per-paper reports. Four of five are prefill-only; Myth has rollouts but never tests subject; none tests domain in generation.

### Abstract, v1.2 draft 2 (Jeffrey 2026-08-29 10:50, base runs folded in by Claude at his request)

Title: Specialization is Not A Myth: Measuring Expert Sets inside Mixture-of-Experts Model Families

Sparse Mixture-of-Experts models contain a fixed pool of experts at each MoE layer. During both
prompt prefill and autoregressive generation, a learned router selects a small subset of experts
for each token, based on its current hidden representation.

Prior work correctly finds that pooled prefill routing does not reveal stable domain specialists.
We show that this negative result does not generalize to generation.

We analyze five checkpoints across the Qwen3.5 and GLM-4.7-Flash families. In a Qwen3.5-35B-A3B
model, we test whether the selected expert sets follow the local subject of an answer by packing
multiple subjects into one generation, reordering them across prompts, controlling for position,
and comparing them with the same questions asked independently. We repeat the subject test on the
base checkpoint of the same model.

Position-resolved prefill routing primarily follows the wording of the question, while pooling
across the prompt produces the shared usage pattern reported in prior work. Across packed and
independently asked versions of identical questions, generation-based expert sets identify the
subject at 95 to 96% in Qwen3.5-35B, compared with 15 to 17% for prefill and 6% by chance.
Generation-time expert sets remain stable within a subject, change sharply at subject boundaries,
and recur for the same subject across positions, prompt design, and the base and fine-tuned
checkpoints. On the base checkpoint, generation-time sets pair prompts by subject in 11 of 12 cases
where prefill pairs 4 of 12. For short mixed prompts, prefill carries the wording of the question,
while generation carries the subject of the answer.

Notes: 11/12 = base safety smoke (5 subjects x 2 registers, chance 1/11); alternative is the 50-pair
run, 8/8 by prompt type. Title subtitle open: "Model Families" promises cross-family generation
that GLM does not have; alternatives offered "Generation-Time Expert Sets Follow the Subject of the
Answer" / "Measuring Expert Sets Where the Model Writes".

### Abstract, v1.2 draft 3 (2026-08-29 10:52): body made affirmative
Paragraph 2 now reads: "Prior work correctly finds that pooled prefill routing converges on one
shared set of experts across domains. We show that generation gives a different answer."
Everything else as draft 2. Title unchanged.

### Intro, opening, draft 1 (Jeffrey 2026-08-29 10:52; citation added by Claude)

("I. Introduce the internal object" was Jeffrey's outline note, not a heading.)

Expert specialization in sparse Mixture-of-Experts models is commonly inferred from which expert
sets are selected for an input. That inference assumes that routing patterns recur when the
underlying computation recurs. However, Wang, Hayou and Nalisnick (2026) find that semantically
different prompts can converge on nearly identical expert usage during prompt prefill and diverge
during text generation, and that which regime applies is not predictable from the prompt alone.
What organizes that rollout divergence remains unknown. We ask whether it is organized by the
local subject of the answer.

Recurrence is the signature of specialization. If the same expert sets return for the same subject
after its wording, position and surrounding context change, then the router is organizing
computation by local subject rather than simply following tokens in the prompt.

Note: wording / position / surrounding context map to the prefill-rows control (15-17%), the
position-matched null (0.02 vs 0.19), and packed vs independent (95-96%). Make the mapping explicit
in Methods.

### Intro, part II (prior work), draft 1 (Jeffrey 2026-08-29 10:59; factual fixes by Claude 11:03)

Prior work established that on prompt tokens, a small fixed set of experts carry most of the
routing mass, across nearly all domains. This is referred to as the Standing Committee (^citation),
measured at the last prompt token of MMLU questions. We reproduce it: in prefill, one expert wins
18 of 20 subjects. Herbst et al found that experts probe as fine-grained task experts, closing
brackets, function definitions, across 12 models on corpus text. Ye et al. found that single
experts are polysemantic, and that per-token paths across layers are monosemantic. Do et al found
that experts scored from question tokens alone, when upweighted, raise accuracy across ten models.
Wang, Hayou and Nalisnick find that prompt-level routing does not predict rollout-level routing,
tracked from prompt through rollout on paired queries. They observed that routing follows hidden
state geometry and that complete trajectories are required (^citation).

Fixes applied: Do et al. separated from the Wang claim (they never compare prompt vs rollout);
"execution" dropped from Herbst's term; measurement clauses added; reproduce sentence added.
Citations: Standing Committee = Wang, Xu et al. 2601.03425; Herbst = 2604.02178; Ye = 2604.17837;
Do = 2604.05267; Wang/Hayou/Nalisnick = 2604.09780.

### Intro, part III (alternatives), draft 1 (Jeffrey 2026-08-29 11:07; revised by Claude 11:08)

Several alternatives that our data can decide. Divergence during rollout is the prompt's wording
playing out. We test this by using the same identification on prompt prefill rows of identical
questions against generation rows.

Position or drift can also account partly for the effect. Prior work by Shorthill (^pos confound
paper cite) shows that routing statistics move with token position, so we test whether the same
subject at different positions reproduces the effect, against different subjects at the same
position. We also test whether sets simply drift along the answer, by restricting candidates to
neighbouring subjects.

The organization could be something the fine-tune put there. We test this by comparing router
weights across base and fine-tuned checkpoints and by repeating the subject test on the base
checkpoint.

We also test whether the register of the answer is correlated with a specific expert set. The
packed prompts hold register constant while subject varies; one prompt design lets register move.

Tests behind each: prefill rows 15-17% vs gen 95-96%; position null 0.02 vs 0.19; adjacent-pool
0.83 vs 0.33; GLM routers bit-identical, Qwen prefill sets identical, base 11/12; register F4
(HVAC 0.50 -> 0.00), chunk B biographies. Pos-confound cite = entropy paper 2, Zenodo 22151499.

### Intro, part IV (why the design separates them), draft 1 (Jeffrey 2026-08-29 11:13; fixed by Claude 11:14)

We separate our experiments six ways; wording, position, drift, fine-tune, register and
prompt-design. The same 20 questions appear as prompt tokens and as answer tokens. If wording
drives it, the sets should come back on the prompt rows just as well as the answer rows. They come
back 15 to 17% of the time on the prompt rows and 95 to 96% on the answer rows. Packed prompts put
each subject at a different position in Chunk A than Chunk C. If position drives the expert set,
two different subjects at the same position should match, and one subject at two positions should
differ. The subject wins 9 of 10 comparisons.

If sets slide gradually along an answer, a subject's nearest match should often be its neighbor.
Restricting candidates to the previous, same and next subject still picks the right one 83% of the
time against 33% by chance. If the fine-tune put the organization there, the base checkpoint should
lack it. The routers are unchanged and the base checkpoint picks by subject 11 of 12 times. The
packed prompts also hold the voice constant while the local topic changes, so anything that recurs
there is organized by subject. One chunk lets the voice change, and there identification drops
while the subject expert stays. Finally, we ask the same subject packed with 19 other subjects and
asked alone, and the same expert set is selected. The context changes, the subject stays and the
expert set stays.

### Intro, part V (principal findings, three anchors), draft 1 (Jeffrey 2026-08-29 11:17; third sentence corrected by Claude)

The same expert sets recur 95 to 96% of the time across independent prompt designs, against 15 to
17% on prefill rows and 6% chance. Inside a single answer, overlap stays at 0.45 to 0.55 within a
single subject and drops to 0.05 at the boundary. The same shape holds across the base checkpoint,
a fine-tuned variant of it, and a second, larger model: in prefill one expert wins 18 of 20
subjects on the 35B and 13 of 20 on the 122B, while generation yields 20 and 18 distinct winners,
and the base checkpoint picks by subject 11 of 12 times.

Correction: 18/20 and 20 distinct are the 35B fine-tune only; 122B is 13/20 and 18 distinct
(routing-only prefill is the clean one); base never ran the 20-subject probe (8-type run: 6/8, 7
distinct; safety smoke 11/12). 122B caveat for Limitations: its generation block is ~45% spill.

### Intro, part VI (contributions and scope), draft 1 (Jeffrey 2026-08-29 11:23; fixed by Claude)

Our contributions include expert specialization defined as recurrence, measured by cross design
identification with four nulls (wording, position, drift, shuffle). Any MoE with a router hook can
run it.

On generated tokens the router organizes by the local subject of the answer, 95 to 96% against 6%,
while the same questions on prompt tokens reproduce the prior shared-set result. Register moves
generation sets as much as subject does, and the design separates the two.

The claim holds across short mixed prompts, where the prompt is a few tokens of one voice and the
answer is hundreds of tokens on the subject. We show that routers sort by subject. Causal tests
reveal that the leaders are removable. The knowledge question stays open but an expert set at a
local subject can be a set of task experts that co-occur. We capture full generation on one family
(Qwen3.5): the base model, its uncensored fine-tune (HauhauCS), and its larger A10B 122B variant.
We also validate the prefill half on GLM-4.7-Flash and its uncensored fine-tune. Winner names are
stable for four subjects at three prompts each.

Fixes: Causal; its; fine-tune; verb added; "MoE with a router hook"; register contribution added;
HauhauCS named. Intro I-VI now complete in draft.

### Title + abstract + full introduction, draft 2 (Jeffrey 2026-08-29 11:33)

Title: Specialization Is Not a Myth: Subject-Specific Expert Sets Recur During MoE Generation

Abstract as draft 3 (above). Introduction rewritten by Jeffrey as eight paragraphs: definition
(subject-conditioned recurrence), prior work at three units, the prefill/generation divide (Wang,
Hayou, Nalisnick), design, four alternatives, prefill reproduction + generation result, controls +
base, two regimes with scoping, three contributions. Technically cohesive; every number traces.

Open before compile:
1. Para 5 "precedes instruction fine-tuning" -> "precedes the uncensoring fine-tune" (official
   Qwen3.5-35B-A3B is the instruct model; HauhauCS is the uncensoring fine-tune; pretrained -Base
   never ran the subject test).
2. Stray "E\section{Introduction}" line.
3. Closing sentence duplicated.
Contribution 3 now reads "...and the instruction-trained–fine-tune boundary, and separates from
register." (register is a co-organizer, F4).
Full text lives in Jeffrey's draft; not copied here.

### Methods, blocks 1-2, draft 1 (2026-08-29 11:41-11:58)

Outline agreed (900 words): 1 models+captures ~150; 2 expert-set construction ~120; 3 prompt
designs ~150; 4 identification metric ~100; 5 controls ~330 (wording, position, drift, register,
shuffle); 6 official-checkpoint and cross-checkpoint checks ~50. "Base" = official
instruction-trained Qwen3.5-35B-A3B throughout; pretrained -Base never ran the subject test.

Block 1 (Claude draft for Jeffrey): five checkpoints named with quant/source; llama.cpp router
hook, greedy, routing only; blocks separate, cut at first EOG; generation for 35B x2 and 122B, GLM
prefill only (transformers hook, BF16).

Block 2 (Jeffrey draft, fixed by Claude): softmax -> top-8 -> rescale; W/S/Q defined, W=S.Q;
layers pooled by unweighted mean (per-layer in appendix); blocks never pooled + reason; set = top-8
by W; Jaccard, chance 0.016; GLM substitution (top-4 by sigmoid+bias, plain sigmoid weights;
64 experts, sets of 4). 165 words.

### Methods, blocks 3-7, draft 1 (Claude 2026-08-29 12:15, for Jeffrey to rewrite; ~680 words)

Block 2 corrections: layer pooling defined as unweighted mean over 40 per-layer W vectors indexed
by expert number -> "layer-pooled expert-index sets" (E114@L10 and E114@L20 are different modules);
per-layer sets in appendix as robustness. Chance Jaccard: exact hypergeometric 0.017; k/(2E-k)=0.016
is the approximation.

Blocks written: Prompt designs (independent 20x3 @2056; packed A/B/C 446 prompt / 2048 gen, C
reordered, segments at model headers A12/B15/C19, C trims at 1812; A,C hold register, B varies it;
HF 392 vs llama.cpp 446 rescale caveat; official-checkpoint 5x2 @256; l1l3 = scoping). Cross-design
identification (top-8 by W; Jaccard argmax; ties = miss; both directions; denominators 24/38/168,
prefill 40/240; accuracy vs matched-subject Jaccard). Controls (shuffle, position, adjacent, prefill,
official-checkpoint, set-size). Statistical analysis (formula; counts vs shuffle null; no CIs / no
multiple-comparison correction; segment-balanced; equal weights). Provenance + four exceptions.

Shuffle seeds now recorded: generation null added to
expert-specialization-data/audit_20260828/crew_reproducibility_nulls.py, default_rng(11), 200 draws:
mean 0.062, 95th 0.095, max 0.119 (first-pass unarchived run gave 0.063). Prefill null: seed 7,
100 draws, mean 0.050, 95th 0.067. Whole script rerun 2026-08-29 12:20 from repo root; all
identification numbers reproduce (117/168, 23/24, 36/38, 14/24; prefill 73/240, 7/40, 6/40, 27/40).
Full draft text is in the chat transcript / Jeffrey's working file.

## 2026-08-29 12:59 — DECISION (Jeffrey): reframe v1.2 as replicate-then-extend

The paper leads by replicating the prior prefill result (standing committee / router collapse)
on five checkpoints across two families and two gating mechanisms (Qwen3.5-35B official +
HauhauCS, 122B, GLM-4.7-Flash official + HauhauCS; softmax top-8 and sigmoid top-4 with bias),
none of which the five 2026 papers used. Then the same protocol shows the generation half.

What moves:
1. Results: Table 1 becomes Result 1 (prefill replication), generation dispersion Result 2,
   cross-design identification Result 3.
2. 122B and GLM promoted from "supporting evidence" to the replication itself; GLM is the
   cross-family, cross-gating leg of Result 1.
3. Contributions: new first line, "we replicate the prefill standing-committee result on three
   new checkpoints and a second gating family," then definition, generation result, controls.
4. Title unchanged ("Specialization Is Not a Myth: ..."); it now reads as scope correction.
5. Related work: "we reproduce" language throughout; each paper credited by unit and by
   measurement (RELATED_WORK_REVIEW_20260829.md §1, §4).

Indisputable datum for the pivot: same 20 questions, same model, same construction, same test;
prompt tokens 7/40 and 6/40, answer tokens 23/24 and 36/38.

Priority check (searches 2026-08-29, five full reads + three abstracts): no published prefill
replication on Qwen3.5 / 122B / GLM-4.7-Flash, no published generation-time subject test. Write
"to our knowledge"; re-run the search the week of submission.

Already consistent with the reframe: abstract para 2 ("correctly finds"), intro para 2 (credits by
unit), the prefill control as pivot. Next edit: contributions list, then intro para 5 (checkpoints'
role), then Results order.

### Result 1 section, working title "Prefill expert sets reproduce across five checkpoints" (2026-08-29 13:05)

Table rows (prefill only): 35B HauhauCS E224 18/20 (20x3); 35B official E224 6/8 (50-pair types),
E95 5 + E224 3 of 12 (safety smoke); 122B HauhauCS E233 13/20 (routing-only); GLM official E10
12/20 (20x15). Overlap for the Standing Committee comparison: 35B 0.51 vs their Qwen3-30B 0.87
(min 0.53).

Fifth checkpoint = GLM HauhauCS, used as a CONTROL (Jeffrey's call, checked 13:05 from
compact/register_run, 140 cells): register batteries have no subject axis. Pooled prefill top-4
main battery = {10,20,28,51} for base AND HauhauCS (Jaccard 1.0); blockD/F = {16,20,28,51}, also
identical across checkpoints. E10 (domain-battery winner) rank 2 on the register prompts; per-cell
winner E28. Reading: same standing set on a subject battery and a register battery, same set on
official and fine-tune; winner identity is a near-tie inside the set. Two controls from one leg:
subject-independence of the default, and fine-tune-independence (routers bit-identical).
Generation pooled top-4 on GLM register runs = {17,21,28,53} / {4,17,28,51}: differs from prefill,
no subject axis to test, so it stays out of Result 3.

## 2026-08-29 16:15 — official vs HauhauCS routing shift, measured on matched prompts (Claude)

Q (Jeffrey): did the HauhauCS fine-tune shift routing? Data: compact 50-pair (100 matched prompts,
same prompt on both checkpoints, generation) + medical nothink (24 matched). Top-8 by pooled W.

| | same prompt, official vs HauhauCS | different prompt, same checkpoint (floor) |
|---|---|---|
| prefill Jaccard (50-pair) | 0.853 (min 0.45) | 0.581 |
| generation Jaccard (50-pair) | 0.702 (min 0.23) | 0.428 |
| prefill (medical) | 0.826 | 0.553 |
| generation (medical) | 0.719 | 0.591 |

Per-layer generation agreement lowest at L0-2 (0.46-0.60), highest mid-stack (0.76-0.82):
consistent with a shifted residual stream read by unchanged routers. E114 gen W 0.0059 -> 0.0055
(rank 19 -> 23) on 50-pair; medical rank 98 -> 109. Largest per-expert shifts ~1e-3.
Reading: fine-tune moves ~15% of the reading set and ~30% of the writing set on an identical prompt,
less than the prompt-to-prompt variation. Router matrices: bit-identical on GLM (verified); Qwen
matrices never diffed (no GGUFs local) -> needs a box; bet is unchanged.
For the paper: Methods/Limitations sentence "the fine-tune shifts routing behaviour less than a
change of prompt does (0.70 vs 0.43 generation overlap on identical prompts)"; Result 1 control.

## 2026-08-30 10:20 — boundary timing: does the crew change before or after the header? (Claude, per-token 3-chunk data)

Method: per-token layer-averaged W (chunks A/B/C), segments from segs_raw.json (start = the '###'
token). Crew = top-8 of segment interior (15-token margins excluded). For each of the 43 boundaries,
count how many of each token's top-8 experts are in the incoming vs outgoing crew, offset from '###'.

Offset -12..-3: in 0.6 / out 2.4 (previous body). -2,-1 (blank line): 0.2 / 0.2. 0 ('###'): 0.35 /
0.33. +1 (first title word): 1.3 / 0.5. +3..+8 (title, mean 8.6 tokens): 1.7-2.1 / 0.7. +13..+30
(body): 2.5-3.4 / 0.7 (plateau).
Reading: outgoing crew releases on the blank line BEFORE the header; the header punctuation is a
neutral state; the incoming crew is on the router from the first title word (before the subject
has been named in full); ramps to plateau over ~15 body tokens.

Seam (prompt -> generation, n=3 chunks): first generated token has 0/8 on the pooled prompt set and
0/8 on subject-1 crew; tokens 0..24 = preamble ("Here are technical explanations..."), both counts
0.3-1; subject crew arrives at +26 = first '###'. Caveat: last 8 prompt tokens are " ." padding.
Reading: no ramp between question-set and subject-set; the reading set is gone at token 0, a
preamble state follows, and the subject crew arrives with its header, same as every later boundary.
Paper use: one sentence in Result 3 ("the set turns over on the blank line before the header and
reaches its plateau within ~15 tokens"), and it answers the Desktop question about the seam.

- 2026-08-30 11:25 Full-draft pass: `DRAFT_v12_sections_20260830.md` resolves the four {AUTHOR} holes (per-layer not computable on packed side; per-pair table summing to 168; adjacency exact chance 0.354 with 21 edge queries; probe = equal-weight mean over answers and layers), two provenance fixes (build 8493 = HauhauCS probe; commit 6658925 = GLM register runs), and drafts Results, Discussion, Limitations, Conclusion, references. Wilson CIs computed. Figure 2 (within-answer overlap trace) not yet drawn.
- 2026-08-30 11:35 Figure 4 drawn: `figures/fig4_within_answer.{pdf,png}` (chunk A, 48-token windows step 8, top-8 Jaccard vs own subject probe set 0.362 mean vs other 19 subjects 0.031; same>other in 250/250 windows; dips at boundaries; history is the weak segment, matching its 6/12 identification). Four agents launched 11:10: number audit -> ledger/AUDIT_v12_numbers_20260830.md; LaTeX assembly -> main_v12_DRAFT.tex + refs_v12.bib; hostile review -> ledger/REVIEW_v12_hostile_20260830.md; MATS confab lens measure 7 -> cc-lens/outputs/dsv4_confab_20260824/RESULTS_lens_measure7_20260830.md.
- 2026-08-30 11:45 Span-matched identification (fixlist item 17; reviewer 2 blocking objection): using only the first n generated tokens of each packed segment against the full probe sets: n=16 A<->P 22/24, C<->P 31/38, all 92/168; n=32 21/24, 33/38, 94/168; n=64 23/24, 35/38, excl-B 73/86, all 122/168; n=128 23/24, 36/38, 120/168; full 23/24, 36/38, 117/168. Prefill rows of the ~20-token questions: 7/40, 6/40. The subject signal is present in a span shorter than the question. Corrected checkpoint-shift floors (checkpoint_shift_50pair.py): same prompt across checkpoints 0.70 gen / 0.85 prefill; same content other dialect 0.75 / 0.75; all pairs 0.08 / 0.41. Fix list: FIXLIST_v12_20260830.md.
- 2026-08-30 12:30 Full paper assembled: main_v12_DRAFT.tex (Jeffrey abstract/intro/methods with fixlist corrections applied and logged in DRAFT_v12_jeffrey_CORRECTED_20260830.tex changelog; Claude Results/Discussion/Limitations/Conclusion with audit corrections; Tables 1, 2, per-layer appendix table; Figures 1, 3, 4; causal-bias Methods paragraph; data availability rewritten). Not compiled. Open: title decision (fixlist 15), 122B placement (20), Table 1 width, version/DOI line.
- 2026-08-30 14:55 COMPILED: main_v12_DRAFT.pdf, 10 pp, twocolumn, 0 undefined citations. Jeffrey decisions applied: title kept with bridging sentence in Discussion (scope not mechanism); version/DOI line blank until deposit; Table 1+2 resizebox to textwidth; collapsed-prefill-rows note in Table 2 caption; fig4 promoted to figure*; appendix table tightened. Remaining for Jeffrey: rewrite pass over Claude sections, version/DOI at deposit.

## 2026-09-01 10:55 — finishing pass (Claude, arxiv-preprint skill loaded)

- **Lost merge found and fixed:** the 08-30 changelog claimed fixlist 16 (abstract range) was
  applied, but main_v12_DRAFT.tex's abstract did not have it; the sentence existed only in
  DRAFT_v12_jeffrey_CORRECTED_20260830.tex. Ported: 95 to 96% now reads "when the answer's
  register is held constant (0.70 over all design pairs, 0.54 when the register changes)".
- **Fixlist 20 completed:** 122B DeltaNet/softmax disjoint-vocabulary disclosure (hostile review
  F6) added to Results 3.1 and Limitations; Methods models paragraph now introduces the hybrid
  stack (one sentence). Fixlist 13 propagated to Limitations (391 to 392 versus 446).
- **AI statement rewritten** (TODO closed): discloses Claude first drafts of Results through
  Conclusion written to the author's analyses, and the number audit. Table 1 width TODO removed
  (resizebox was already in). One voice-pass conversion in Results 3.2 (token-balanced closer).
- **Bibliography was silently broken:** the five 2026 arXiv entries used eprint/archivePrefix,
  which plainnat does not render (no arXiv ids in the PDF); converted to howpublished+doi.
  Herbst "ICML 2026" verified against icml.cc (poster 61679) — note stays. Hayashi "Shun-ichiro"
  was parsed by BibTeX as a von particle ("ichiro Hayashi et al."), latent since v1.1; braced.
- **Citation anchors restored:** the v1.2 rewrite had orphaned every pre-2026 bib entry — only
  7 references rendered. Restored from v1.1 (claims mirrored from main.tex L76-155, logged in
  the tex header): foundational MoE cites in intro p1; Mixtral echo + Mynampati/Hayashi
  input-dependence sentence in intro p2; serving-phases cite in intro p3; Qwen3 lineage cite in
  Methods. Now 16 rendered references; zoph2022stmoe left in .bib uncited.
- **Table 1 row 3 verified** from base_safety_smoke_crews_OUT.txt: prefill winners
  {95 x5, 224 x3, 47, 189, 36 x2} = 5 distinct ✓.
- **Priority re-search re-run (week of submission):** no published prefill replication on these
  checkpoints, no generation-time subject test. Closest new work, neither competing:
  arXiv 2604.14434 (rank-1 geometric routing, custom architecture, causal expert steering) and
  arXiv 2605.12476 (router-expert gradient coupling, 1B from-scratch SMoE). Optional cites,
  Jeffrey's call.
- Compiled clean: 11 pages, 0 undefined citations, 0 overfull boxes, 1 benign float warning.
- **Remaining for deposit (Jeffrey):** final read of the full PDF (the AI statement's "author
  reviewed the entire text" must be true before it ships); version/DOI footnote line; Zenodo
  new-version deposit under the existing concept DOI + CITATION.cff/.zenodo.json update for the
  new title; decide whether to cite 2604.14434/2605.12476.

## 2026-09-01 12:29 — Jeffrey's unsaved notes (pasted in chat, not previously on disk)

Pasted alongside an older snapshot of the 08-30 draft; the draft itself is superseded, these
notes are new:

> Where you measure routing determines what you see.
>
> What led to this:
> - looked for one register expert (E114); found it was four modules and a register expert;
> - ran a subject probe to see if any single expert owned a subject; saw one expert own all of
>   prefill and twenty own generation; packed subjects into one answer and saw the sets turn
>   over at boundaries and come back for the same subject
>
> What do people already believe:
> - Single experts are polysemantic during prefill
>
> What we contribute:
> - Effect replicates across GLM-4.7 Instruction-Trained+HauhauCS/Uncensored,
>   Qwen3.5-35B-A3B+HauhauCS/Uncensored+Qwen3.5-122B-A10B. [scope note: in the paper this
>   replication claim is prefill-only for GLM/122B; generation is the two 35B checkpoints]

- 12:35 Jeffrey's line "Where you measure routing determines what you see." added as the
  Conclusion opener (his call, from the pasted notes). Full review pass: Table 2 row sums
  re-verified by hand (gen 117/168, prefill 73/240, B-involving 44/82, excl-B 73/86; two Wilson
  CIs recomputed), token-balanced triples match the 08-29 recompute at all three set sizes, no
  live [citation]/{AUTHOR}/TODO in the body, no em/en-dashes in prose, all three figure files
  present, citations render inline. FINAL COMPILE: 11 pages, 0 undefined citations, 0 overfull
  boxes. Awaiting Jeffrey's read + deposit (version/DOI footnote, Zenodo new version).

## 2026-09-01 12:50 — v1.2 PUBLISHED

Jeffrey approved the deposit ("I think its good to be pushed to my zenodo"). Flow (same rails
as v1.0/v1.1): footnote version line filled (concept DOI); Data Availability now points to
curated-data-index/ (provenance, manifests, build script, audit_20260828/ scripts copied in,
10 MB; full ~6 GB archive from the author); v1.2 promoted to main.tex/refs.bib/main.pdf
(11 pp); .zenodo.json/CITATION.cff/README updated for the new title; SOURCES.md v1.2 note;
secret-scan clean; commit 50c52a5 pushed; GitHub release v1.2 -> Zenodo webhook.

**Published record: https://zenodo.org/records/22236906 — DOI 10.5281/zenodo.22236906,
version 1.2, isNewVersionOf 10.5281/zenodo.22089026, concept 10.5281/zenodo.20779604.**
Follow-up commit adds the v1.2 DOI to CITATION.cff.

Open follow-ups (not blocking): separate Zenodo deposit of the full expert-specialization-data
archive (~6 GB) with its own DOI, then a metadata edit here to link it; optional cites for
arXiv 2604.14434 / 2605.12476; the 10-prompt-per-subject official-checkpoint capture named in
Limitations.

- 13:00 PDF preview added to the Zenodo listing. Zenodo files are immutable post-publish, so
  a new version was minted via the API (Jeffrey's token; browser extension was disconnected):
  record 22236999 = release zip + main.pdf, files.default_preview = main.pdf, version "1.2",
  description gains a files note. Concept DOI now resolves to 22236999; preview endpoint 200.
  22236906 (zip-only, minutes older) remains as a superseded intermediate version. CITATION.cff
  updated. Note: the token was pasted into the session transcript; Jeffrey may want to
  regenerate it at zenodo.org/account/settings/applications after the data deposit.
