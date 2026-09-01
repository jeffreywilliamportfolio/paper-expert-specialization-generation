# E_register: register / dialect / addressee captures vs domain routing (Qwen3.5-35B-A3B, HauhauCS and base)

Written 2026-08-28 by agent E_register. Read-only on all source data. All numbers below were recomputed
from raw `router/ffn_moe_logits-<L>.npy` tensors with the brief's rule (softmax over 256, keep top 8,
renormalize; W = mean weight over tokens, S = selection fraction). Intermediate arrays live in
`/tmp/E_register/*.npz` (scratch, not source data); scripts `/tmp/E_register/extract.py`, `an_*.py`.

Layer policy: prefill and generation pooled over layers 0..38 (layer 39 in every one of these llama.cpp
captures has `n_gen + 1` rows, generation-aligned, no prefill rows; I drop it for comparability and report
the 0..39 generation variant where it matters). "trim" = generation cut at the first token id in
{248044, 248045, 248046} (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`) read from `generated_tokens.json`.
Ranks are out of 256. Jaccard is on the top-15 set by W unless stated (top-12 where compared to the domain probe).

## 1 What the data is

### 1a. 5-5-26 register run (dialect x prompt type x speaker role), base AND HauhauCS, no-think
Path: `/Volumes/ExternalSSD/aave-registers-cleaned/5-5-26_initial_50_pair_register_run/runs/no_think_base_hauhau_20260505T205437Z/{run_base_nothink,run_hauhau_nothink}/<prompt_id>/`
Prompts: `.../5-5-26_initial_50_pair_register_run/data/prompts.tsv`. 50 matched pairs = 100 prompts. Each pair
has the same propositional content in AAVE (morphosyntactic markers only: habitual be, zero copula, ain't,
multiple negation, got-possession...) and in academic English. Topic is NOT fixed: `prompt_type` =
fact_distortion 30, advice_seeking 16, medical 14, epistemic_inner_state 10, relationship_description 10,
legal 8, dream 6, identity 6. `speaker_role` (who the speaker is talking about / as; the closest thing to an
addressee axis in this archive) = self 20, parent_about_child 16, peer 16, professional 16,
child_about_parent 14, sibling 10, self_about_AI 8.
Models: base `ggml-org/Qwen3.5-35B-A3B-GGUF` Q8_0 and `HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive` Q8_0,
greedy, `-n 2048`, `</no think>` template (docs/PLAN.md). 200 cells, 8,000 router npy, prefill + generation.
Mean prompt 58 tokens; mean generation 868 tokens; 90/200 cells hit a template spill (mean trimmed length 596).

### 1b. 5-15-26 medical register runs (dialect only, topic fixed = pediatric/cardiac triage), 4 runs each
Path: `/Volumes/ExternalSSD/aave-registers-cleaned/5-15-26/aave-register-medical/`
- main: `runs/run_{base,hauhau}_{nothink,think}/med_s0{1..5}_{ae,aave}/` — 5 scenarios x 2 dialects x 4 runs = 40 cells, 1,600 npy.
  AE and AAVE prompts differ only in dialect (`medical_register_scenarios.tsv`).
- chest_pain_plus: `chest_pain_plus/runs/...` — 6 scenarios x 2 x 4 = 48 cells, 1,920 npy. Here the AE arm was
  rewritten into CLINICAL register ("exertional chest tightness ... resolves upon cessation of activity")
  while the AAVE arm is vernacular ("My pops is havin chest tightness when he be walkin up the stairs"), so
  this contrast is dialect AND register together. Scenario 6 repeats the plain main-run pair 1.
- financial_stress_pair: 1 pair x 4 runs = 8 cells, 320 npy (too small; extracted, not reported separately).
Generation never hit a template spill in these 96 cells (trim = untrimmed). Think runs generate 3x longer.
Note: there is NO separate addressee or empathy arm in any routing capture in this archive; 5-16-26 and 5-17-26
(safety minimal pairs, profanity/social-frame ablations) are IO-only, zero npy.

### 1c. HVAC / water-treatment technical register control, HauhauCS only
Path: `/Volumes/ExternalSSD/moe-routing-organized/qwen3.5-35b-a3b-and-huahua/35B/hvac_cal_water_treatment_6cond_l1l3_hauhau/`
Design (results md in `qwen-huahua-6cond-hvac/results/`): a fixed HVAC + water-treatment paragraph, three
register categories (L1_technical: plain description; L2_recursive: the system processing a description of
itself; L3_experience: "is there something it is like to be this system") x six deictics (this/a/your/the/their/our).
`-n 1024`, greedy, no-think. 180 cells, 7,200 router npy, prefill (~440 tokens) + generation (mean 949).
IMPORTANT: only 18 distinct prompts exist. The 10 "base prompts" per category x deictic are byte-identical
(18 distinct prompt hashes and 18 distinct generated_text hashes among 180 cells). The results md's
"60 prompts x 40 layers = 2400 observations" per category is really 6 distinct prompts, each repeated 10x.
Every per-category number below therefore has n = 6 distinct prompts (one per deictic).

### 1d. l1l3_a_only (register on a transformer-description paragraph), HauhauCS AND vanilla base
Paths: `.../35B/l1l3_a_only_hauhau/`, `.../35B/l1l3_a_only_vanilla/` (vanilla = `Qwen3.5-35B-A3B-Q8_0.gguf`, per experiment.log).
30 distinct prompts each: routing_selfref 10 (technical description of MoE routing), recursive_selfref 10,
experience_probe 10; deictic A only. 1,200 npy per model. Also `qwen3.5-35b-a3b-huahua-five-cond-experience-probe/raw/`:
15 distinct experience-probe prompts (3 x deictics A..E), HauhauCS, 600 npy. The 5cond-diectics and
mixed-self-ref-content bundles have no extracted npy (0 files) and were not used.

### 1e. Reference: the domain probe
`/Volumes/ExternalSSD/paper-expert-specialization-generation/data/35b-60prompt-primary/results_domain_specialists_20260408T235839Z.{md,json}`
(20 domains x 3 prompts, HauhauCS). I read its numbers; I did not recompute its tensors (outside my subset).

## 2 What I computed

For every cell: per-layer per-block sums of routed weight and selection over 256 experts (prefill, generation,
trimmed generation), then pooled W/S per arm. Top-10/15 by W per arm per block; E114 W and rank per arm and
per cell; E114 best layer; top-15 Jaccard and W-distribution JSD between arms; per-matched-pair Jaccard;
a 3-prompt-vs-3-prompt top-12 Jaccard resampling (300 draws) so that the 5-5 data can be compared to the
domain probe's 3-prompts-per-domain top-12 Jaccard matrix (read from the JSON `crossover` block).
Cross-checks against the archive's own tables: my 5-5 base prefill top-5 (224 .0124, 95 .0107, 46 .0095, 64 .0095, 130 .0094)
matches `analysis_all_models_expert_routing/top_experts_by_run.md` exactly; my pooled AE-AAVE prefill JSD 0.00054
matches its 0.000541. HVAC: my token-weighted E114 W (L0-39, gen) L1 0.00272 / L2 0.00575 / L3 0.01323 vs the
results md 0.003405 / 0.005380 / 0.014222. The L1 and L3 figures differ by 20% and 7%; best layers agree
(L1 best layer 38, L3 best layer 14) and the L3/L1 ratio (4.9x mine, 4.2x theirs) agrees in kind. I could not
reproduce their exact averaging (cell x layer unweighted gives 0.00287 / 0.00575 / 0.01323), so treat their L1
number as approximately right, not exact.

## 3 Findings

### F1. Prefill generalist: 224 on mixed prompts and HVAC, but NOT on medical or transformer-description prompts
- 5-5 base prefill top-10 (W): 224 .0124, 95 .0107, 46 .0095, 64 .0095, 130 .0094, 189 .0092, 243 .0090, 47 .0087, 65 .0087, 56 .0083.
  Hauhau prefill: 224 .0121, 95 .0106, 130 .0095, 46 .0094, 64 .0091, 189 .0090, 243 .0089, 47 .0087, 65 .0082, 56 .0081.
  224 is rank 1 in both dialects, both models. (paths in 1a)
- HVAC pooled prefill: 224 .0117, 246 .0111, 218 .0109, 46 .0101 ... (L1 arm alone: 246 .0127 then 224 .0125).
- Medical main prefill (base nothink): 47 .0139, 247 .0139, 36 .0127, 65 .0113, 95 .0107, 123 .0102, 189 .0101, 130 .0098, 52 .0097, 43 .0096.
  224 is NOT in the top 10 of any of the 8 medical arms. The same two experts (247, 47) also lead medical generation.
  In the 5-5 run the 14 medical prompts likewise put 65/95/224 first (224 rank 3).
- l1l3 (transformer paragraph) prefill: 166 first in all six model x category arms (166 .0xx, 224 second or third).
So "224 is the prefill generalist" holds for topically mixed prompt sets and long expository paragraphs, and fails
for focused medical-triage prompts (247/47 take over) and for the MoE-description paragraph (166).

### F2. Generation top experts by arm (5-5 run, topic mixed)
Base gen top-10 all prompts: 206 .0079, 56 .0077, 250 .0074, 61 .0073, 69 .0070, 146 .0067, 5 .0067, 107 .0066, 103 .0066, 142 .0064 (E114 rank 23, W .00583).
Hauhau gen top-10: 206 .0075, 56 .0067, 228 .0066, 250 .0066, 61 .0065, 142 .0064, 146 .0063, 42 .0063, 69 .0062, 107 .0061 (E114 rank 35; trimmed rank 21).
By prompt type (base, gen): advice 56/69/61; dream 206/147/56; epistemic_inner_state 228/151/139/81/114; fact_distortion 220/63/191;
identity 103/139/56; legal 122/109/43; medical 206/247/5; relationship 250/56/107. These sets barely overlap
(mean pairwise top-15 Jaccard 0.17 base, 0.19 hauhau) while their prefill sets overlap heavily (0.63, 0.58).

### F3. Dialect barely moves routing; prompt type moves it a lot; speaker role is in between
Top-15 Jaccard between arms (5-5, base / hauhau):
| axis | prefill | generation | trimmed gen |
|---|---|---|---|
| dialect (AAVE vs AE, 50 v 50) | 1.000 / 1.000 | 0.667 / 1.000 | 0.875 / 0.875 |
| prompt_type (8 arms, mean of 28 pairs) | 0.627 / 0.578 | 0.173 / 0.194 | 0.185 / 0.178 |
| speaker_role (7 arms, mean of 21 pairs) | 0.698 / 0.684 | 0.283 / 0.311 | 0.281 / 0.275 |
W-distribution JSD: dialect 0.00054 (pre) / 0.00023 (gen); prompt_type 0.0120 / 0.0272; speaker_role 0.0061 / 0.0111 (base).
Topic-fixed dialect contrast (within each prompt_type, base): prefill Jaccard 0.875-1.0 for all 8 types;
generation 0.667-0.875 for 7 types, 0.364 for fact_distortion (short answers, few tokens). Medical type: pre 1.0, gen 0.667, trim 0.765.
Medical run (topic fixed, dialect only), AE vs AAVE pooled top-15 Jaccard: main base_nothink pre 0.667 gen 0.765;
hauhau_nothink 0.875 / 0.875; base_think 0.765 / 1.000; hauhau_think 0.875 / 1.000. Chest_pain_plus (dialect + clinical register):
pre 0.667 gen 0.875 in all four runs; pooled JSD pre 0.0059-0.0063 vs gen 0.0010-0.0011.
Model (base vs Hauhau, same prompt): prefill Jaccard 0.88-0.89, generation 0.68-0.74 — the fine-tune moves generation
routing more than the dialect does.

### F4. Register with topic held fixed moves generation routing as much as a change of domain
HVAC (same paragraph, 6 prompts per register):
| register | prefill top-5 | generation top-5 | E114 gen W (rank) |
|---|---|---|---|
| L1 technical | 246 .0127, 224 .0125, 218 .0109, 46 .0101, 111 .0088 | 210 .0092, 246 .0079, 224 .0071, 242 .0069, 72 .0069 | .00276 (221) |
| L2 recursive | 224 .0119, 218 .0111, 246 .0104, 46 .0098, 225 .0079 | 142 .0078, 210 .0069, 87 .0068, 218 .0067, 224 .0064 | .00590 (11); trimmed rank 5 |
| L3 experience | 218 .0107, 224 .0106, 46 .0104, 246 .0101, 111 .0078 | 114 .0135, 142 .0098, 228 .0097, 139 .0092, 170 .0091 | .01353 (1); trimmed .01567 (1) |
Between-register top-15 Jaccard: prefill L1-L3 0.500, L1-L2 0.667, L2-L3 0.667; generation L1-L3 0.000, L1-L2 0.304, L2-L3 0.250.
Between-deictic Jaccard inside a register (n=1 prompt each): prefill 0.79-0.86; generation L1 0.34, L2 0.31, L3 0.75.
The domain probe's between-domain top-12 Jaccard (JSON `crossover`): prefill mean 0.543, generation mean 0.056 (trimmed 0.055).
So a register change on one fixed topic produces a generation-set divergence (L1 vs L3 = 0.0) at least as large as
the average between-domain divergence in the domain probe (0.056), while prefill overlap stays at the domain-probe level (0.5).
l1l3 replicates on a different topic and on the base model: hauhau gen Jaccard experience vs routing_selfref 0.200,
vanilla 0.250; per-cell Jaccard within-category 0.40-0.41 vs across-category 0.29-0.33 (both models).

### F5. Magnitude comparison for the specialization story (task 5), all on top-12 sets, 3 prompts per side, 300 draws
5-5 base (hauhau in parentheses):
| contrast | prefill | generation | trimmed |
|---|---|---|---|
| different prompt_type, same dialect | 0.535 (0.540) | 0.126 (0.139) | 0.127 (0.130) |
| same prompt_type, different dialect (matched pairs) | 0.894 (0.835) | 0.712 (0.641) | 0.775 (0.782) |
| same prompt_type, same dialect, disjoint prompts | 0.756 (0.744) | 0.276 (0.278) | 0.291 (0.298) |
| domain probe, different domain (from JSON) | 0.543 | 0.056 | 0.055 |
Reading: changing prompt type within the 5-5 set (0.13) looks like changing domain in the domain probe (0.06);
both sit near or below the "two random prompt sets of the same type" floor (0.28). Changing dialect on matched
content leaves the generation top-12 set 64-71% intact, far above that floor, so dialect is close to a no-op
for the top-expert set. Register (HVAC L1 vs L3) drops the set to 0.0, i.e. the full domain-sized move.
Conclusion: "domain winner" in generation is partly topic and partly the register the answer is written in;
the register component is at least as large as the topic component wherever this archive lets me separate them.

### F6. E114: a generation-side expert for first-person / experiential / reflective output, not for topic
- 5-5 base gen rank by prompt_type: epistemic_inner_state 5 (W .00876), identity 7 (.00775), advice 21, dream 26,
  legal 25, relationship 35, medical 73 (.00435), fact_distortion 227 (.00291). Hauhau: 9, 12, 49, 38, 33, 41, 80, 235.
  By speaker_role (base gen): self_about_AI rank 4 (W .00908; AAVE 1, AE 5), self 19, child_about_parent 26, peer 35,
  professional 38, parent_about_child 40, sibling 82.
- Per cell (base): gen rank <= 5 in 10/200 cells, all epistemic_inner_state or identity prompts (001, 002, 003, 012 in both dialects);
  rank > 100 in 69/200.
- HVAC: rank 221 in the technical register, 1 in the experience register on the SAME paragraph (F4); per-cell gen W
  L1 .0029 +- .0007 (n=6), L2 .0059 +- .0009, L3 .0135 +- .0020 — no overlap between L1 and L3.
- l1l3: rank 211 (hauhau) / 224 (vanilla) for routing_selfref, 2 / 1 for experience_probe. Five-cond experience probe: rank 2 (W .00849).
- E114 rises in the base model too (vanilla experience_probe gen W .00919, rank 1), so it is not a HauhauCS artefact.
- Best layer: 5-5 gen layer 22 (W .032), prefill layer 35; HVAC L3 gen layer 14 (W .138), L1 gen layer 38; L2 layer 14.
- Medical arms: E114 gen rank 96-146 and W .0031-.0039 in all 16 nothink/think x base/hauhau x main/chest arms. It never
  becomes a medical expert whatever the register of the answer.

### F7. E114 and dialect / addressee in the medical scenarios
- Main run (dialect only): per-cell E114 gen W AAVE minus AE = +.0006, +.0007, .0000, +.0003, +.0005 (base nothink);
  +.0002, +.0002, -.0004, +.0002, +.0004 (hauhau nothink). Pooled AAVE vs AE: .00374 vs .00329 (base), .00351 vs .00340 (hauhau).
- Chest_pain_plus (vernacular AAVE vs clinical AE): pooled AAVE .00386 (rank 96) vs AE .00339 (126) base nothink;
  .00372 (106) vs .00307 (146) hauhau nothink; .00390 vs .00348 base think; .00368 vs .00328 hauhau think.
  Per scenario AAVE > AE in 11/12 (base nothink), 12/12 (hauhau nothink). The AAVE-arm answers are more person-addressed
  ("yes, you should definitely be a bit worried") than the clinical-arm answers ("strong likelihood that this represents
  a cardiac condition"; `chest_pain_plus/runs/run_hauhau_nothink/chestplus_s01_{ae,aave}/generated_text.txt`).
- Prefill: AAVE prompts carry MORE E114 in chest_pain_plus (.0043 vs .0030, rank 76 vs 148 base) — dialect does move E114
  in prefill there, but this confounds vernacular register with dialect.
- 5-5 run: paired AAVE minus AE E114 gen W = +.0001 +- .0013 (60% positive), identical for both models. Dialect alone is
  a null for E114 in generation.

## 4 Prediction scores

P2 (E114 is a register expert, top-5 gen for reflective/self prompts, outside top-20 for technical): SUPPORTED within my subset.
Top-5 gen: epistemic_inner_state (5/9), identity (7/12), self_about_AI (4/7), HVAC L3 (1), l1l3 experience (2/1), five-cond (2).
Technical: HVAC L1 rank 221, l1l3 routing_selfref 211/224, fact_distortion 227/235, medical 73-146. Math/physics not in my subset.

P3 (AAVE medical: E114 gen higher in person-addressed/empathetic arms than clinical; dialect moves it less than register/addressee;
sign holds across dialects): MIXED. The only person-addressed-vs-clinical contrast in the archive is chest_pain_plus, where the
sign is right in 4/4 runs and 23/24 scenario pairs (AAVE-vernacular .0037-.0039 vs clinical .0031-.0035), but the effect is
tiny (E114 stays rank 96-146 in every medical arm) and dialect and register are confounded in that design. Dialect alone
(main run) gives +.0001 to +.0005, so the register+dialect delta (+.0004 to +.0007) is larger than dialect alone, weakly
matching the second clause. No separate addressee arm exists, so "sign holds across dialect arms" cannot be tested.

P4 (E114 prefill rank mid-pack 20-90 on nearly every set; top-10 only for reflective/second-person prompts): MIXED, leaning refuted
on the range. Per-cell 5-5 prefill rank median 155 (IQR 123-203); 173/200 cells rank > 90; 0/200 in the top 10; best per-cell
ranks 13-27, all self_about_AI / epistemic prompts. Pooled prefill: self_about_AI 31-33, epistemic 34-41, medical 82-98, HVAC L1 88,
HVAC L3 8 (a prompt written in the experience register), l1l3 experience 31. The directional clause holds; the 20-90 band is wrong
(it is 90-230 for ordinary prompts and 8-40 for reflective ones).

P7 (dialect shifts prefill more than generation; matched-pair Jaccard lower in prefill than generation): MIXED.
Pooled W-distribution JSD says yes in every dataset: 5-5 prefill 0.00054 vs gen 0.00023 (base), 0.00055 vs 0.00030 (hauhau);
medical main 0.00114 vs 0.00047; chest 0.0059 vs 0.0011. Pooled top-15 Jaccard says yes in medical (chest 0.667 pre vs 0.875 gen,
all runs) and is flat-or-reversed in 5-5 (dialect 1.0 pre vs 0.67-1.0 gen). Per-cell matched-pair top-15 Jaccard says no in 5-5:
prefill 0.832 vs generation 0.693 (base), 0.837 vs 0.669 (hauhau) — but the same-dialect two-prompt floor is 0.58 pre vs 0.28 gen,
so relative to the floor generation is the more dialect-stable block. Chest per-scenario pairs: pre 0.61 vs gen 0.74 (yes).
Net: supported as a distribution statement, not as the literal raw per-pair Jaccard statement.

P5 (not assigned, incidental): E114 best generation layer 22 in the 5-5 run (in 20-30) but 14 in HVAC L3/L2 and l1l3; 38 in technical arms. MIXED.
P1 (incidental, see F1): prefill generalist is 224 for 5-5 and HVAC, 247/47 for medical, 166 for the transformer paragraph.

## 5 Caveats / could not verify
- HVAC "180 prompts" = 18 distinct prompts x 10 identical repeats (byte-identical prompts and outputs). The published
  results md treats them as 60 per category; every HVAC number here is n = 6 distinct prompts per register.
- No routing capture manipulates addressee or empathy directly; speaker_role in 5-5 is the closest proxy and it is
  confounded with topic (e.g. self_about_AI prompts are also the epistemic/identity prompts).
- chest_pain_plus confounds clinical register with dialect (AE rewritten clinical, AAVE rewritten vernacular).
- Pooled W is token-weighted, so long-generating prompt types dominate pooled arms; per-cell numbers are given where it matters.
- Domain-probe tensors were not recomputed by me; its Jaccard is top-12 with 3 prompts per domain, which is why I
  resampled 3-vs-3 top-12 in the 5-5 data for comparison. Prompt lengths and generation lengths differ across datasets.
- HVAC E114 W differs from the results md by up to 20% (L1); averaging convention could not be reproduced exactly.
- 5-16-26, 5-17-26, financial_stress_pair, 5cond-diectics and mixed-self-ref bundles: no usable routing tensors (0 npy) or too small.
- Think runs include hidden reasoning tokens in the generation block; I did not separate reasoning from the visible answer.

## 6 Summary for a lay reader
When the same question is asked in AAVE and in academic English, the model's choice of experts hardly changes: the
prompt-reading experts overlap almost completely, and the answer-writing experts overlap about two thirds to fully,
far above what two unrelated questions on the same topic give. Changing the kind of question (fact check, legal,
medical, "what is it like to be you") changes the answer-writing experts about as much as changing the academic
domain did in the earlier domain probe. Most striking, keeping the topic identical (an HVAC paragraph) and only
changing whether the model is asked to explain it or to consider what it would be like to be it swaps the answer-writing
expert set completely. Expert 114 is the clearest case: it is the top answer-writing expert whenever the output
is reflective, first-person or about experience, on any topic and on both the base and the fine-tuned model,
and it is near the bottom (rank 200+) for technical explanation and fact answers. Dialect alone does not move it;
a more person-addressed medical answer nudges it up slightly but it stays far from the top in every medical arm.
So the "domain winners" seen in generation are partly about the subject and at least as much about the register the
answer is written in.
