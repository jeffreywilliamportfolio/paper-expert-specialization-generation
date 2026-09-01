# Agent D_glm: GLM-4.7-Flash domain and register routing (2026-08-28)

Archive: `/Volumes/ExternalSSD/glm47-flash-domain-routing/` (read-only). All numbers below were
recomputed locally from the raw `.npy` tensors or the analysis `.npz`/`.json` unless marked
"archive says". Scratch results (not part of the archive): `/tmp/glm_domain_results.json`,
`/tmp/glm_register_out.txt`.

## Gating rule used (GLM is not Qwen)

GLM-4.7-Flash routes with DeepSeek-V3 style `noaux_tc` gating: 64 routed experts + 1 always-on
shared expert, top-4, sigmoid scores, per-expert selection bias. Reconstruction, taken from
`analysis/analyze_powered.py` and `register_run/analyze_explore.py` and checked against
`model_card.md` / `model_info.md`:

1. `s = sigmoid(logit)` over the 64 routed experts (logits clipped to +-30 before the sigmoid).
2. Selection score = `s + e_score_correction_bias[layer]` (a 64-vector per layer, values 8.91 to
   9.05, file `raw_powered/e_score_correction_bias.npz`; identical to `raw/` and to
   `register_run/base_bias.npz` and `register_run/hauhau_bias.npz`, checked with allclose).
3. Keep the 4 highest biased scores. The bias affects only WHICH experts are chosen.
4. Weight of each chosen expert = its plain sigmoid `s`, renormalized so the 4 sum to 1
   (`norm_topk_prob=true`). The model then multiplies by `routed_scaling_factor=1.8`; I leave that
   factor out because it scales every weight equally and changes no ranking, W ratio, or entropy.
5. The shared expert is not part of the router and is ignored. Layer 0 is dense; MoE layers are
   1..46. Uniform share per expert = 1/64 = 0.0156 for W and 4/64 = 0.0625 for S.

W, S, Q, domain winner, and the four concentration measures follow the brief's definitions,
with "renormalize the top 8" replaced by "renormalize the top 4".

## 1 What the data is

**Model.** `zai-org/GLM-4.7-Flash` (30B-A3B, 47 layers, 46 MoE layers, MLA attention). Two
weight surfaces appear: the base checkpoint, and "HauhauCS Uncensored", a fine-tune of it
(`register_run/EXPLORE_STATUS.md`, `register_run/TEARDOWN_20260704.md`). Two capture rigs:
BF16 transformers hook (domain runs), and llama.cpp Q8 GGUF `capture_activations` at commit
6658925 (register runs). The two rigs agree on prefill logits at r = 0.997 to 1.000 (archive
says, `EXPLORE_STATUS.md`; not re-verified, no paired files kept locally).

**Domain runs (prefill only, no generation at all).**
- `raw/` = exploratory: 20 domains x 1 prompt, carrier "You are a specialist in X. In three or
  four sentences, explain C, and note why it matters." 31 to 43 tokens per prompt
  (`analysis/summary.json` ntok). Per prompt `router/ffn_moe_logits-{1..46}.npy`, shape
  `[n_tok, 64]` float32.
- `raw_powered/` = "powered": the same 20 domains x 15 prompts = 300 prompts, carrier "Explain C,
  and why it matters." (`domain_battery_powered.json`), 18 to about 21 tokens each (per-domain
  totals 279 to 310 tokens over 15 prompts). "Powered" means (a) 15 prompts per domain instead of
  1 and (b) a length control: the analysis truncates every prompt to its first T = 16 tokens so
  each domain contributes exactly 240 tokens (`analysis/analyze_powered.py`).
- The 20 domains: mathematics, physics, chemistry, molecular_biology, clinical_medicine,
  computer_science, electrical_eng, economics, contract_law, quant_finance, linguistics,
  philosophy, economic_history, political_science, psychology, music_theory, poetics,
  visual_art, food_science, geology.
- No token cap applies (prefill only). No generation block exists for any domain prompt. The
  archive's own next-steps list (`RESULTS.md`, limitation 3) asks for one.

**Register run (`register_run/`, prefill + generation).** Three FIRE/NOFIRE banks of 10 matched
pairs each (`battery_heldout_v1.json`, `battery_blockD_v1.json`, `battery_blockF_v1.json`):
- heldout: FIRE = introspective self-address prompts ("Describe the hum of your own processing
  honestly..."), NOFIRE = plain task prompts. Prompt length 23 to 31 tokens.
- blockD: Jeffrey's worldview prompts vs matched NOFIRE rewrites (29 to 127 tokens).
- blockF: Kae's second-person relational prompts vs matched narrative controls (23 to 107 tokens).
Captured with llama.cpp Q8, greedy, `-n 4096`, GLM chat template with thinking on, NO
stop-at-eog (every run goes to 4096 tokens; the analyzer trims at the first turn-end token id in
{154827, 154828, 154820, 154829}). Router files are `[n_prompt + 4096, 64]`; layer 46 stores
only the last prompt position in prefill (`[1 + 4096, 64]`), so I exclude L46 from prefill
profiles, as `analyze_explore.py` does. Seven capture sets of 20 prompts each:
`explore_base_q8_boxA`, `explore_base_q8_boxB` (cross-box replicate), `explore_hauhau_q8_boxB`
(heldout bank); `explore_blockD_{base_q8_boxA,hauhau_q8_boxB}`; `explore_blockF_{base_q8_boxA,
hauhau_q8_boxB}`. Trimmed generation lengths: 406 to 4096 tokens; 0 to 3 prompts per set never
emit a turn-end token before the cap. Also `base_gates.npz` / `hauhau_gates.npz` (46 router
weight matrices `[2048, 64]` each, extracted from the two GGUFs) and the two bias files.
`pulled_20260704T0825/` holds BF16 prefill-only captures of the same banks; not re-analyzed here.

## 2 What I computed

- From `raw_powered/` (all 300 prompts, all 46 layers) and `raw/` (20 prompts): per-domain
  per-layer W and S over 64 experts, at full prompt length and at T = 16. Domain winners with
  all layers pooled (mean W over layers, then argmax). Concentration of the 20-winner list.
  Per-prompt winners. Domain-profile similarity: cosine on the 64-vector (layers pooled) and on
  the 2944-vector (46 layers x 64 experts), Jaccard of top-10 (64-dim) and top-50 (2944-dim)
  expert sets, and mean-over-layers Jensen-Shannon divergence (JS, bits) of the S profiles.
  Per-layer mean pairwise JS across the 20 domains (the curve behind
  `analysis/powered_layer_separation.png` and `layer_separation.png`).
- From `analysis/expert_domain_map.{npz,json}`: re-derived top domain, top share and selectivity
  for the named experts; specialist counts by depth band.
- From `register_run/` (all 7 sets x 20 prompts x 46 layers, about 6.4 GB of npy): per-prompt
  prefill W, trimmed-generation W, untrimmed-generation W; per-prompt and per-class winners;
  FIRE vs NOFIRE JS per layer against an equal-size split-half floor, prefill and generation;
  base vs HauhauCS agreement (per-token top-4 overlap in prefill, pooled-profile cosine,
  generation winner identity); router weight equality from the gate npz.
- A matched comparison of "how much does domain move routing" vs "how much does register move
  routing": 10 vs 10 prompts, first 16 tokens, layers 1 to 45, same JS-minus-floor statistic.

## 3 Findings

**F1. GLM prefill has a generalist, but a weak one.** Powered run, full prompt length,
`raw_powered/` (winners listed in alphabetical domain order chemistry .. visual_art):
`[53, 10, 28, 10, 28, 10, 53, 62, 28, 10, 10, 53, 10, 54, 10, 10, 10, 10, 10, 10]`.
Expert 10 wins 12 of 20; distinct 5; max wins 12; Herfindahl 0.41; normalized entropy 0.39.
With the archive's T = 16 length control the list becomes
`[54, 30, 28, 20, 28, 10, 53, 54, 30, 10, 10, 30, 10, 54, 10, 10, 10, 29, 10, 10]`:
expert 10 wins 9 of 20; distinct 7; Herfindahl 0.265; entropy 0.54. Exploratory run (`raw/`,
1 prompt per domain): expert 28 wins 14 of 20, distinct 5, Herfindahl 0.52, entropy 0.33.
The winner's own numbers are small: E10 pooled over all layers has W = 0.026 and S = 0.091
(selected on 9% of token-layer slots), against a uniform W of 0.0156. The top five experts in
every domain sit within about 0.002 W of each other (E10, E28, E53, E30, E54, E20 in various
orders), which is why 8 of 20 winners change identity between the full-length and T = 16
readings. Read plainly: in prefill GLM routes almost every domain to the same handful of
high-frequency defaults, and which default comes first is close to a coin flip. That is the same
"prefill generalist" shape as Qwen, but far flatter, because the sigmoid+bias router is built to
balance load (no expert can carry 18 of 20 the way Qwen's E224 does).

Side by side (Qwen values from `paper-expert-specialization-generation/main.tex`, Table 1;
GLM from this recompute):

| Model | Regime | Distinct | Max wins | Herfindahl | Norm. entropy |
|---|---|---|---|---|---|
| Qwen 35B | prefill | 3 | 18 | 0.815 | 0.13 |
| Qwen 35B | generation | 20 | 1 | 0.050 | 1.00 |
| Qwen 122B | prefill | 7 | 13 | 0.445 | 0.42 |
| Qwen 122B | generation | 18 | 2 | 0.060 | 0.95 |
| GLM-4.7-Flash | prefill, 15/domain, full length | 5 | 12 | 0.41 | 0.39 |
| GLM-4.7-Flash | prefill, 15/domain, T=16 | 7 | 9 | 0.265 | 0.54 |
| GLM-4.7-Flash | prefill, 1/domain | 5 | 14 | 0.52 | 0.33 |
| GLM-4.7-Flash | generation | not captured | | | |

**F2. There is no domain generation block for GLM, so the generation half of the shape cannot be
computed.** The nearest substitute is the register run, which has generation but only 20 prompts
per set and no domain labels. Per-prompt winners there (`/tmp/glm_register_out.txt`, from
`register_run/explore_*`):

| Set | Prefill: distinct / max / entropy | Gen trimmed: distinct / max / entropy | Prefill winner = gen winner |
|---|---|---|---|
| heldout base boxA | 5 / 9 (E28) / 0.46 | 7 / 6 (E28, E21) / 0.56 | 4 of 20 |
| heldout base boxB | 6 / 10 (E28) / 0.47 | 7 / 7 (E28) / 0.54 | 5 of 20 |
| heldout hauhau boxB | 7 / 11 (E28) / 0.48 | 9 / 5 (E21, E28) / 0.64 | 4 of 20 |
| blockD base | 10 / 5 (E51) / 0.70 | 5 / 10 (E17) / 0.44 | 4 of 20 |
| blockD hauhau | 9 / 5 (E51) / 0.66 | 4 / 9 (E17) / 0.40 | 5 of 20 |
| blockF base | 8 / 6 (E16, E20) / 0.60 | 8 / 7 (E28) / 0.62 | 3 of 20 |
| blockF hauhau | 8 / 6 (E16, E20) / 0.60 | 6 / 7 (E28) / 0.52 | 3 of 20 |

Two things carry over from Qwen and one does not. Carries over: the winning expert changes
identity between reading and writing (only 3 to 5 of 20 prompts keep the same winner), and the
generation defaults (E28, E17, E21, E4) are a different set from the prefill defaults (E28, E51,
E10, E16, E20). Does not carry over: generation does not disperse. Dispersion rises in the
heldout sets, falls in blockD, and is flat in blockF; generation winners are again a default set.
Untrimmed generation is dominated everywhere by E35 (wins 9 to 13 of 20 in every set), which is
the expert the model uses while looping past the turn end to the 4096 cap, so untrimmed numbers
measure spill, not the answer. Caveat: these are 20 assorted prompts, not 20 domains x 3
prompts, so this is a weak proxy, and each "domain" here is a single prompt.

**F3. Domain clustering in prefill is real and semantic, and it lives at the (layer, expert)
level, not the expert-identity level.** From `raw_powered/`, full length, cosine of the
46 x 64 W map (2944-vector), Jaccard of top-10 experts on the layer-pooled 64-vector, and mean
per-layer JS:

| Pair | cos (64, pooled) | Jaccard top-10 (64) | cos (2944) | Jaccard top-50 (2944) | JS bits |
|---|---|---|---|---|---|
| mathematics ~ physics | 0.991 | 0.54 | 0.898 | 0.56 | 0.060 |
| chemistry ~ molecular_biology | 0.992 | 0.54 | 0.896 | 0.52 | 0.049 |
| economics ~ quant_finance | 0.998 | 0.82 | 0.952 | 0.64 | 0.035 |
| music_theory ~ poetics | 0.996 | 0.54 | 0.946 | 0.56 | 0.033 |
| molecular_biology ~ clinical_medicine | 0.992 | 0.67 | 0.911 | 0.56 | 0.045 |
| mathematics ~ poetics | 0.990 | 0.43 | 0.876 | 0.45 | 0.072 |
| chemistry ~ contract_law | 0.986 | 0.43 | 0.811 | 0.39 | 0.096 |
| computer_science ~ visual_art | 0.992 | 0.43 | 0.883 | 0.59 | 0.090 |
| physics ~ political_science | 0.988 | 0.43 | 0.844 | 0.47 | 0.091 |
| geology ~ philosophy | 0.983 | 0.43 | 0.812 | 0.41 | 0.102 |

Related pairs: cos2944 0.90 to 0.95, JS 0.03 to 0.06. Unrelated: cos2944 0.81 to 0.88, JS 0.07 to
0.10. Every domain's nearest neighbour by cos2944 is a semantic neighbour (economic_history ~
political_science 0.953, economics ~ quant_finance 0.952, linguistics ~ poetics 0.948,
music_theory ~ poetics 0.946, physics ~ electrical_eng 0.937). Within-supergroup mean cos2944
0.874 vs across 0.845 (T = 16: 0.841 vs 0.804; exploratory: 0.819 vs 0.775). The layer-pooled
64-vectors are all 0.98 to 0.998 cosine, i.e. nearly identical; pooling over layers erases the
domain signal. Archive check: `RESULTS.md` closest pairs music_theory ~ poetics 0.038 and
economics ~ quant_finance 0.042 (T = 16) reproduce as 0.0381 and 0.0415 here. `RESULTS.md`
L9 shares also reproduce: e50 10.3 to 20.4% across domains, e6 12.3% in molecular_biology,
e53 11.0% in electrical_eng, e14 9.3% in economic_history. Prefill vs generation for this
question: prefill only (see F2).

**F4. Register moves generation routing; domain moves prefill routing.** Matched design (10 vs
10 prompts, first 16 tokens, layers 1 to 45, JS between pooled S profiles minus an equal-size
split-half floor, mean over layers):
- Domain vs domain (`raw_powered/`, all 190 pairs): between 0.0998, floor 0.0565, excess +0.043.
- Register FIRE vs NOFIRE, prefill (`register_run/explore_*_base_q8_boxA/`): heldout 0.065 vs
  floor 0.076 (excess -0.011); blockD 0.084 vs 0.093 (-0.009); blockF 0.066 vs 0.089 (-0.023).
  At equal token counts no register bank clears its own noise floor in prefill. With full prompt
  length the prefill excess is +0.015 (heldout), +0.034 (blockD), -0.019 (blockF).
- Register, trimmed generation (thousands of tokens per prompt): heldout excess +0.043 (boxA),
  +0.047 (boxB), +0.060 (hauhau); blockD +0.034 / +0.035; blockF -0.008 / -0.021. Best layers:
  heldout L36 (boxA), L21 (boxB, hauhau); blockD L17 / L30. Jaccard of top-10 FIRE vs NOFIRE
  experts in heldout: prefill 0.67, generation 0.43 (the two classes share fewer leading experts
  when writing than when reading).
So the introspection and worldview registers separate routing about as much in generation
(+0.03 to +0.06) as a domain change does in prefill (+0.043), while in the prompt-read they do
not separate at all at matched length. Kae's relational bank (blockF) is a null in both blocks,
which matches the archive's own reading (`EXPLORE_STATUS.md`: "separation real but WEAK").

Base vs HauhauCS: the router is untouched. All 46 gate matrices are identical (min column
cosine 1.000000, max absolute difference 0.0, `register_run/base_gates.npz` vs
`hauhau_gates.npz`); bias files identical. On the same prompts and box, per-token prefill top-4
overlap base vs hauhau = 0.961 (min over prompt-layers 0.880), against 0.986 for base vs base
across two boxes (the numerical floor). So the fine-tune shifts about 2 to 3% of prefill routing
slots through the residual stream, not the router. Pooled prefill profiles: cosine 0.99995;
pooled generation profiles 0.9999. Generation per-prompt winner agrees 13 of 20 (heldout), 18
of 20 (blockD), 15 of 20 (blockF); the base-vs-base cross-box agreement is 12 of 20, so the
fine-tune adds no visible instability beyond box-to-box greedy drift. No trimmed generation
token stream is identical between any two sets, including base vs base (0 of 20): greedy
decoding is not reproducible across boxes here, as the workspace notes warn.

**F5. Where in depth domain specialization sits: both ends, with a trough in the middle.** Mean
pairwise JS across the 20 domains per layer, powered run, T = 16 (the curve behind
`analysis/powered_layer_separation.png`), layers 1..46:
0.022, 0.023, 0.030, 0.070, 0.101, 0.119, 0.146, 0.138, 0.146, 0.104, 0.079, 0.052, 0.056,
0.065, 0.065, 0.071, 0.100, 0.109, 0.088, 0.082, 0.107, 0.085, 0.066, 0.083, 0.063, 0.092,
0.069, 0.106, 0.084, 0.120, 0.099, 0.088, 0.073, 0.078, 0.096, 0.108, 0.133, 0.090, 0.087,
0.116, 0.116, 0.113, 0.138, 0.141, 0.107, 0.111.
Top five: L9 0.1464, L7 0.1462, L44 0.1406, L43 0.1384, L8 0.1382. Band means: L1-10 0.090,
L11-20 0.077, L21-30 0.088, L31-40 0.097, L41-46 0.121. Exploratory run (`raw/`, behind
`analysis/layer_separation.png`): L43 0.206, L44 0.202, L37 0.187, L8 0.182, L9 0.181; band
means 0.129 / 0.102 / 0.112 / 0.135 / 0.168. `powered_summary.json` records best_layer 9 and
`summary.json` best_layer 43; both reproduce, but `RESULTS.md`'s conclusion that "the
exploratory L43 was noise; with power the signal is strongest early" is overstated: L9 beats
L44 by 0.006 bits, both peaks appear in both runs, and the late band has the highest mean in
both. The number of distinct domain winners per layer peaks at L30 and L43 (11 each), L7, L41,
L44 (10 each), and drops to 1 at many middle layers (L10, L13, L19, L21-23, L28, L36, L39).
The expert map (`analysis/expert_domain_map.json`, 2870 live positions, T = 16) agrees: among
positions with at least 1% mean share, the fraction with selectivity >= 3x is L1-10 20%
(62/307), L11-20 13% (38/292), L21-30 15% (44/288), L31-40 19% (59/304), L41-46 23% (46/198).
The ten sharpest specialists sit at L37 (E9 geology 12.8x, 14.8% share), L35 (E3 contract_law
12.0x; E5 electrical_eng 10.5x), L9 (E47 quant_finance 10.4x), L29 (E22 food_science 10.4x),
L24 (E14 geology 10.2x), L28 (E32 contract_law 9.6x), L38 (E17 electrical_eng 9.6x), L42 (E39
computer_science 9.4x), L8 (E9 quant_finance 9.4x). Named entries in `EXPLORE_STATUS.md`
(L37E9 12.8x, L44E39 6.0x, L41E51 5.8x, L35E61 5.5x, L42E29 philosophy 12%, L36E3 psychology
13.3%, L32E28 domain-silent) all reproduce from the npz. Traffic hubs are single-layer, not
whole-model: L39E28 21.6%, L21E20 19.9%, L36E11 17.7%, L28E7 16.8%, L10E25 16.3% of that
layer's slots, each with selectivity about 1.05 to 1.10 (domain-blind).

## 4 Prediction scores

**P1 (prefill winners concentrate on a small default set; generation winners disperse; the
prefill winner is a high-S default, not a domain expert). Scored as a cross-family test.**
MIXED, and half untestable.
- Prefill half: SUPPORTED in shape, weaker in degree. 5 to 7 distinct winners, max 9 to 12 of 20
  (vs Qwen 3 distinct, 18 of 20). The winner (E10, or E28 in the 1-prompt run) is a high-S
  default (S 0.09 to 0.11, present in every domain's top five) and not a domain expert; the real
  domain specialists (selectivity 6x to 13x) never win a pooled list because they live at single
  layers with 1 to 2% share. The "70% of prompt sets" clause is about expert 224 on Qwen and does
  not apply.
- Generation half: NOT TESTABLE. No domain prompt was ever generated on GLM. The register-run
  proxy (F2) shows winner identity changing between blocks in 15 to 17 of 20 prompts, but no
  consistent increase in dispersion (3 sets up, 2 down, 2 flat), and generation winners are
  another default set (E28, E17, E21). Untrimmed, E35 (the post-turn-end spill expert) wins 9 to
  13 of 20 in every set.

**P7 (dialect shifts prefill routing more than generation; answer register drives generation
routing).** NOT TESTABLE as written: no dialect manipulation exists in this archive. The
available analog is register (FIRE vs NOFIRE) rather than surface form, and it points the way
P7's second clause expects: register separation is absent in prefill at matched length (excess
-0.01 to -0.02) and present in generation (+0.03 to +0.06); FIRE/NOFIRE top-10 Jaccard drops
from 0.67 (prefill) to 0.43 (generation). This supports "the answer's register drives
generation routing" but says nothing about the "surface form lives in the prompt" clause.

P2, P3, P4, P5, P6, P8 are about Qwen experts or Qwen data and are not scorable here. One GLM
note for P2's spirit: the archive's generation register candidate L42E29 has philosophy as its
top domain (12%, selectivity 1.8x) in the prefill domain map, the same pairing Qwen's E114
shows; that is an observation in the archive, not a test.

## 5 Caveats / could not verify

- The central gap: no generation captures exist for the domain battery, so the GLM
  "generation winners" column is empty. Anything about dispersion in generation rests on 20-prompt
  register sets with no domain labels.
- The register generation captures are Q8 GGUF via llama.cpp, the domain captures are BF16
  transformers. The archive's r = 0.997 to 1.000 rig agreement (`EXPLORE_STATUS.md`) could not
  be re-verified locally (paired files not kept).
- Register generations ran without stop-at-eog to a 4096 cap; trimmed lengths were 406 to 4096
  tokens and 0 to 3 prompts per set never produced a turn end (their whole 4096 tokens count as
  "trimmed"). The untrimmed numbers are dominated by loop spill (E35) and should not be used.
- llama.cpp prefill at L46 holds only the final prompt position; L46 is excluded from all
  prefill profiles, so prefill uses 45 layers and generation 46.
- The GLM winner lists are unstable to analysis choices because the top defaults are nearly
  tied (W within 0.002): 8 of 20 winners change between full-length and T = 16. Concentration
  numbers should be read as "5 to 7 distinct, 9 to 12 max", not as exact.
- Register banks are n = 10 per class, explore grade, and their prompt lengths vary widely
  (blockD 29 to 127 tokens); the matched-length comparison in F4 uses only the first 16 tokens.
- Greedy generation was not token-identical between boxes even for the same weights (0 of 20),
  so base-vs-hauhau generation comparisons carry the same drift as box-vs-box.
- Not analyzed: `register_run/pulled_20260704T0825/` (BF16 prefill-only register captures),
  `analysis/expert_map_structure.npz`, the Qwen Kae-arm folders under `register_run/kae_qwen_*`
  (Qwen, outside this subset), `fire_battery.json` / `FIREbank_blockF_v1.tsv` (prompt sources
  only). The 1.8 routed scaling factor is omitted by design.
- Archive disagreement flagged: `RESULTS.md` says domain separation is "strongest early (L9),
  not L43". The recompute shows two peaks of nearly equal height (L7-9 and L43-44) with the
  late band highest on average, in both runs. The L9 number itself is correct.

## 6 One-paragraph summary for a lay reader

GLM-4.7-Flash is a different kind of mixture-of-experts from Qwen: it picks 4 experts out of 64
with a rule that deliberately spreads work evenly. When it reads a prompt, it still leans on a
few all-purpose experts no matter the subject (one expert comes first in 9 to 12 of 20 subjects,
where Qwen's favourite came first in 18 of 20), so the "one generalist reads everything" pattern
does show up in a second model family, just more gently. Underneath that, the model clearly
knows subjects apart: routing for economics looks like routing for finance, chemistry like
biology, music like poetry, and the sharpest subject-specific experts sit in the early layers
7 to 9 and the late layers 35 to 44. What this archive cannot say is what happens when GLM
writes an answer about a subject, because no subject prompts were ever generated on it; the
only generation data are register experiments, and there the winning experts change identity
between reading and writing but do not fan out one-per-topic. Those register experiments show
something else useful: how a prompt is pitched (introspective, worldview) barely changes routing
while the model reads it, but changes routing about as much as a subject change does once the
model starts writing, and the HauhauCS fine-tune left the router bit-for-bit unchanged.
