# A_primary35b: Qwen3.5-35B-A3B (HauhauCS Q8_0) 60-prompt domain-specialist probe

Agent: A_primary35b. Written 2026-08-28. Read-only on data; local CPU only; no sub-agents.
All computations below were re-done from the summary NPZ (and the JSON candidate tables), not copied
from the Markdown reports.

Base directory used for provenance (identical copies exist in the other four locations listed in the task;
MD5 of the primary NPZ is `2f2e58f8ddb2a14d037e04be650a71c7` in every copy checked):

    BASE = /Volumes/ExternalSSD/moe-routing-organized/qwen3.5-35b-a3b-and-huahua/35B/qwen-huahua-expert-identification
    NPZ  = BASE/results/results_domain_specialists_20260408T235839Z.npz
    JSON = BASE/results/results_domain_specialists_20260408T235839Z.json
    NPZ2 = BASE/results/results_domain_specialists_20260415T214918Z.npz   (re-run, see section 7 / finding 10)
    Same NPZ also at /Volumes/ExternalSSD/paper-expert-specialization-generation/data/35b-60prompt-primary/raw/

## 1. What the data is

- Model: HauhauCS `Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf` (sha256 f3235db7…64cb17, from
  `BASE/results/20260415T214918Z_..._run_metadata.json`). 40 MoE layers, 256 experts, top-8. Layer types from
  the HF config (`/Volumes/ExternalSSD/aave-registers/references/hf/Qwen-Qwen3.5-35B-A3B/config.json`):
  `full_attention_interval = 4`, so layers 3, 7, 11, …, 39 (10 layers) are softmax attention and the other 30 are
  gated-DeltaNet ("linear_attention").
- Capture: custom llama.cpp `capture_activations` build 8493 (1772701f), 2x RTX 5090, `--routing-only`, greedy
  (`--temp 0 --top-k 1 --seed 42`), `-n 2056`, ctx 16384, KV cache q8_0, no-think chat template.
- Prompts: `BASE/prompts/domain_specialist_probe_60.json`. 20 domains x 3 prompts (subtypes mechanism / history /
  synthesis). Example (philosophy): "How does epistemology work as a field concerned with justification, belief,
  and knowledge?", "Why did Immanuel Kant matter to metaphysics, ethics, and epistemology?", "Compare philosophy
  of mind with metaphysics in the kinds of questions each asks." These are all third-person expository asks; none
  is written in a first-person or second-person reflective register.
- Sizes (from JSON `cells`): 60 cells; prompt tokens total 2362 (mean 39/prompt); generated tokens total 110455;
  43 of 60 cells hit the 2056-token cap; trimmed generation total 100506 (14 cells had a detectable chat-template
  spill and were cut). Note: `RESULTS.md` says 2404 prompt tokens and 110412 generated; the JSON cell table says
  2362 and 110455. The difference is small (about 40 tokens) and probably a counting convention difference between
  the capture log and the analyzer; it does not affect any result here but is flagged.
- Raw per-cell router tensors (`router/ffn_moe_logits-<layer>.npy`) are NOT on the drive in any of the five
  locations. `non_npy_remote_artifacts/captures/20260408T235839Z_.../<cell>/` holds only `generated_text.txt`,
  `generated_tokens.json`, `prompt_tokens.json`, `metadata.txt` (0 npy files found). Everything below is from the
  summary NPZ: per-domain-per-layer W/S/Q arrays of shape (20, 40, 256) per block, plus (20, 256) pooled arrays.
- Aggregation used by the original analyzer (`BASE/scripts/analyze_domain_specialists.py`): W/S/Q are computed
  per cell per layer, then averaged with equal weight over the 3 cells and the 40 layers (`nanmean`). So "domain
  W" is a prompt-mean, not a token-pooled mean; long and short generations count the same. Prefill = rows
  `[:n_tokens_prompt]`, generation = rows `[n_tokens_prompt:]` of the same capture. The "trimmed" track cuts at
  the 6-token BPE sequence `[27, 91, 316, 6018, 91, 29]` (literal `<|im_end|>` text), which matched in 14/60 cells.

## 2. What I computed

From NPZ `<block>_domain_W` (20x256) and `<block>_domain_layer_W` (20x40x256), blocks prefill /
generation_all / generation_trimmed:
1. Domain winner by W per block; concentration (distinct, max wins, Herfindahl, normalized entropy); also
   winners by S.
2. Winner margin: (W rank1 - W rank2) / W rank1 per domain; "clear" = margin > 10 %; near-tie = < 2 %.
3. Leave-one-prompt-out screen. Per-cell W vectors are not stored, so exact LOO is impossible. The JSON candidate
   tables store `consistency = 1/(1+std/mean)` over the 3 prompts for the top candidates, which gives the
   prompt-level std of W for the rank-1 and rank-2 experts. With 3 values, no single value can sit more than
   std*sqrt(2) from the mean, so dropping one prompt shifts the mean by at most 0.707*std. I mark a domain
   "LOO-safe" when the rank-1/rank-2 gap exceeds 0.707*(std1+std2) (worst case cannot flip), else "possible flip".
   This is a bound, not a count of actual flips.
4. Per-layer winner dispersion: at each layer, number of distinct argmax-W experts across the 20 domains; also
   per-layer mean pairwise Jaccard of top-8 sets across domains; both split by softmax-attention vs DeltaNet layers
   and by depth band.
5. E114: W-rank in each domain per block, top-5/top-10 membership, overall rank, layer profile (mean over
   domains of layer W), best layer overall and per domain.
6. Pairwise Jaccard of top-8-by-W sets between domains, per block; related-group means vs unrelated-pair means.
7. Comparison of the 20260415T214918Z NPZ/JSON against the primary run.

## 3. Findings

1. Headline claims confirmed (NPZ). Prefill winner-by-W: expert 224 wins 18/20 domains; 130 wins chemistry,
   103 wins linguistics; 3 distinct winners, Herfindahl 0.815, normalized entropy 0.13. Generation (all and
   trimmed): 20 distinct winners, max wins 1, Herfindahl 0.05, normalized entropy 1.0. Expert 224 wins only
   political_science in generation (W 0.00894 vs runner-up 96 at 0.00843; margin 5.6 %), and is rank 2 in law,
   rank 3 in history, rank 8 in software_engineering, rank 11 in economics; median generation rank of 224 across
   domains is 47.5. Winners by S tell the same story: prefill 2 distinct (224 x18, 130 x2), generation 18 distinct.
   Overall top-3 by W: prefill 224 / 243 / 56 (0.012708 / 0.010609 / 0.009227); generation_all 210 / 146 / 114
   (0.006228 / 0.006170 / 0.005761). These match `RESULTS.md` exactly.

2. Prefill winners are fairly robust; generation winners are mostly fragile. Prefill: 11/20 domains have a
   clear winner (margin > 10 %; median margin 10.5 %), 1 near-tie (linguistics: 103 vs 224, 1.7 %). LOO screen:
   15 LOO-safe, 3 possible flips (chemistry 130 vs 224, gap 0.00036; economics 224 vs 189, gap 0.00057;
   linguistics 103 vs 224, gap 0.00021), 2 unknown (winner std not in candidate table). So the two non-224 prefill
   winners are exactly the two that could flip back to 224. Generation_all: 7/20 clear (median margin 7.7 %),
   1 near-tie (cybersecurity 188 vs 212, 1.9 %). LOO screen: only 4 LOO-safe (philosophy: 114, margin 49 %;
   psychology: 146, margin 37 %; mathematics: 100, 22.7 %; linguistics: 103, 13.1 %), 16 possible flips. Example
   of fragility: history winner 158 has gap 0.00023 to expert 110 but prompt-level std 0.0059 (25x the gap);
   neuroscience 54 vs 24: gap 0.00023, std 0.0038. So "20 distinct generation winners" is a real dispersion
   signal but the identity of most individual winners is not reliable at n=3.

3. The strong generation specialists are a short list. Only philosophy (E114, W 0.01876, 2x the runner-up) and
   psychology (E146, W 0.01803 all / 0.02160 trimmed) have winners that stand far above rank 2. Mathematics (E100)
   and linguistics (E103, which also wins linguistics in prefill) are moderately clear. Everything else is a
   near-tie among several experts at W around 0.009-0.012.

4. Per-layer winner dispersion is flat across depth, and the attention/DeltaNet split does not matter for it.
   Prefill: mean 1.6 distinct winners per layer (range 1-6; layer 39 = 6, layer 15 = 5, 30 layers have exactly 1);
   softmax-attention layers 2.4 vs DeltaNet 1.33; bands (0-9, 10-19, 20-29, 30-39) = 1.3 / 1.9 / 1.3 / 1.9.
   Generation_all: mean 14.25 (range 8-18); softmax 14.1 vs DeltaNet 14.3; bands 13.5 / 14.1 / 14.5 / 14.9. The
   least dispersed generation layers are 1, 3 (8 each) and 2, 16 (9 each); the most are 8, 20, 35 (18 each).
   Per-layer mean pairwise top-8 Jaccard across domains: prefill 0.69 (softmax 0.65, DeltaNet 0.70; band means
   0.76 / 0.65 / 0.68 / 0.67; layer 39 is the most domain-specific prefill layer at 0.33), generation 0.14
   (softmax 0.16, DeltaNet 0.14; bands 0.147 / 0.134 / 0.134 / 0.155; most specific layer 20 at 0.058, least
   specific layer 39 at 0.29). So the domain signal in generation lives on essentially every layer, slightly
   strongest in the middle third (layers 9-31) and weakest at the very first and very last layers; the 3:1
   DeltaNet/attention pattern leaves no visible imprint on routing dispersion.

5. E114 rank tables (W). Prefill: rank 5 in philosophy (W 0.00982); everywhere else rank 59-218 (archaeology 59,
   comparative_religion 61, linguistics 73, physics 94, statistics 98, biology 115, psychology 126, then 147-218
   for the rest; software_engineering 218 is the lowest). Overall prefill rank 124 (W 0.003482). Generation_all:
   philosophy 1 (0.01876), comparative_religion 2 (0.01040, just 4.5 % behind winner 170), linguistics 3,
   political_science 6, archaeology 8, physics 9, law 12, psychology 14, biology 28, statistics 41, chemistry 64,
   mathematics 99, software_engineering 128, computer_science 136, neuroscience 142, economics 176,
   cybersecurity 178, environmental_science 194, history 238, medicine 242. Trimmed moves political_science to 5,
   psychology to 9, statistics to 24. Overall generation rank 3 (from 124 in prefill). Top-5 in generation:
   philosophy, comparative_religion, linguistics (+ political_science trimmed). Top-8 membership across domains:
   1 domain in prefill, 5 in generation.

6. E114 layer footprint. Mean-over-domains layer W (x1000), generation_all: peaks at layer 26 (35.9), layer 14
   (33.5), layer 20 (26.3), layer 8 (17.5), layer 11 (10.7); nearly zero at layers 7 (0.03) and 32 (0.05). Prefill
   peaks at layer 20 (21.2), 14 (14.5), 35 (14.2), 26 (11.0), 8 (10.3). So the best layer is 20 in prefill and 26 in
   generation; the footprint is a few spikes (8, 14, 20, 26) rather than a band. All four spike layers are
   DeltaNet layers; E114 mean W on DeltaNet layers 0.0064 vs softmax-attention layers 0.0038 in generation. Per
   domain in generation, the best E114 layer is 26 for biology, computer_science, mathematics, physics,
   political_science, psychology, statistics; 20 for philosophy (layer W 0.172), archaeology, linguistics; 14 for
   comparative_religion (0.116) and neuroscience. Layer-26 and layer-20 E114 are clearly the same phenomenon
   across the E114-heavy domains.

7. Generation top-8 sets are domain-specific; prefill sets are shared. Pairwise top-8 Jaccard between domains:
   prefill mean 0.51 (median 0.45; 0 of 190 pairs disjoint; only 30 distinct experts appear in all 20 top-8 sets;
   224 is in all 20). Generation_all mean 0.034 (median 0.0; 119 of 190 pairs fully disjoint; 98 distinct experts
   across the 20 sets; 224 in 4 sets, 114 in 5). Highest generation pairs (all 0.231 = 3 shared of 13):
   philosophy-psychology, linguistics-philosophy, law-political_science, computer_science-software_engineering,
   chemistry-physics. Related groups vs unrelated: generation related-pair mean 0.083 vs unrelated 0.029 (about
   2.8x), with groups: software_engineering/computer_science/cybersecurity 0.099, philosophy/comparative_religion/
   psychology 0.125, economics/political_science/law 0.099, mathematics/statistics/physics 0.070,
   biology/medicine/neuroscience 0.048, history/archaeology 0.0. Prefill related 0.56 vs unrelated 0.51 (no real
   difference; computer_science-economics has Jaccard 1.0 in prefill). So generation sets carry a weak but
   consistent "neighbouring fields share a few experts" structure, but biology/medicine/neuroscience and
   history/archaeology, which a human would call closest, do not share experts.

8. W and S winners agree in generation for 13/20 domains; the exceptions are cybersecurity (W 188, S 212),
   history (W 158, S 224), law (W 48, S 122; 48 is only S-rank 8), medicine (W 152, S 82), neuroscience (W 54,
   S 146), physics (W 139, S 210). Q (weight when selected) of generation winners sits in a narrow 0.106-0.122
   band, so W differences are driven by selection rate S, not by how hard the router commits when it picks.

9. Trimmed vs untrimmed generation: same 20 winners; the only rank changes of note are psychology W of 146
   rising 0.0180 -> 0.0216 and E114 psychology rank 14 -> 9, statistics 41 -> 24. The 14 trimmed cells removed
   about 9 % of generation tokens. The results are not sensitive to the spill.

10. Run 20260415T214918Z is a full re-capture (new output dir `raw/20260415T214918Z_...`, same binary build 8493,
    same model sha, same seed/greedy settings, `-n 2056`), not just a re-analysis. Its NPZ is numerically identical
    to the primary run: max |dW| = 3.5e-18 on every domain-W array, identical winners in all three blocks,
    identical per-cell generation lengths (all 60), identical trim counts (14) and integrity numbers. On this build
    greedy decoding reproduced token-for-token. What it adds is a provenance record (`run_metadata.json` with
    sha256 of binary, source, model, prompt files, and full command) that the primary run lacks. It adds no new
    routing information.

## 4. Prediction scores (this subset only)

- P1: SUPPORTED on this prompt set (one of the sets the prediction covers). Prefill concentrates on 224 (18/20,
  Herfindahl 0.815); generation disperses (20 distinct, Herfindahl 0.05). Where 224 does not win prefill, the
  winners (130 chemistry, 103 linguistics) beat 224 by 3.0 % and 1.7 % and are flagged possible LOO flips; 224 is
  rank 2 in both. Whether 130/103 are "another high-S default" rather than domain experts: 130 is prefill top-7
  overall (S 0.061) so it qualifies as a default; 103 also wins linguistics in generation, so it looks like a
  genuine linguistics expert, a mild exception. The >=70 % threshold is met (90 %). Other prompt sets are outside
  this subset.
- P2: MIXED. Supported parts: E114 wins philosophy, is top-5 in comparative_religion (2) and linguistics (3),
  and is outside the top 20 in mathematics (99), chemistry (64), statistics (41), software_engineering (128).
  Refuted parts: physics is rank 9 (top 10), not outside top 20; psychology is rank 14 (all) / 9 (trimmed), not
  top-5. Also archaeology (8), political_science (6), law (12) place higher than the "register" story predicts
  for expository prompts. Note the prompts are all third-person expository; E114's rise cannot be attributed to
  first-person or reflective output from this subset alone (the generated text was not register-coded here).
- P4: MIXED. Prefill top-10 only in philosophy (rank 5), and that prompt set is not reflective/second-person; it
  is about mind/metaphysics as topics. The "mid-pack 20-90" range is wrong on this set: E114 prefill rank is
  59-98 in only 5 domains and 115-218 in 14 domains (overall rank 124). E114 is a low-rank prefill expert here,
  lower than predicted.
- P5: SUPPORTED. Best layer 26 in generation (layer W 0.036), 20 in prefill (0.021); the other spikes are 14
  and 8. The peak is in the 20-30 window as predicted, but the footprint is spiky (8, 14, 20, 26), not a
  contiguous mid-to-late band.
- P6: NOT TESTABLE here (3 prompts/domain only). Indirect support: the LOO bound flags 16/20 generation winners
  as flippable and 119/190 domain pairs already have disjoint top-8 sets, so the 20-distinct count is a
  mixture of real dispersion and n=3 noise; consolidation with more prompts is plausible. The prefill/generation
  contrast (Herfindahl 0.815 vs 0.05; Jaccard 0.51 vs 0.03) is far too large to be a sample-size artifact.
- P3, P7, P8: outside this subset.

## 5. Caveats / could not verify

- No raw per-cell tensors on the drive; W/S/Q reconstruction (softmax-256, top-8, renormalize) could not be
  re-run. I trust the analyzer code (read in full) and its identity residual (2.8e-17 prefill, 5.6e-17
  generation), but the per-cell numbers themselves are unverifiable. Exact leave-one-out is therefore replaced by
  a worst-case bound from stored std values.
- Domain W is a mean over cells and layers with equal weight, not a token-pooled mean. A token-pooled version
  would down-weight the 17 shorter generations; with 43/60 cells at the 2056 cap the difference is probably small
  but was not checked.
- 43/60 generations hit the -n 2056 cap and 14 cells contain chat-template spill. "generation_all" therefore
  includes some post-turn text; "trimmed" only catches spills that rendered `<|im_end|>` as 6 BPE tokens.
  Winners are unchanged between the two tracks, so this does not matter for the headline numbers.
- Layer indexing: the NPZ layer axis is the capture's `ffn_moe_logits-<layer>` index 0-39, which I assume maps
  1:1 onto the HF config `layer_types` list (attention at 3, 7, ..., 39). Not independently verified.
- The 2404 vs 2362 prompt-token discrepancy between RESULTS.md and the JSON cell table is unresolved.
- Prompt-level std values come from the JSON candidate tables (top-12 by composite score); for prefill
  comparative_religion and philosophy the winner (224) was not in the table, so no LOO bound.
- Generated text was not read or register-coded; P2's mechanism (first-person/reflective output) is untested.

## 6. One-paragraph summary for a lay reader

On this 60-prompt run (20 fields, 3 questions each) the 35B model routes almost every prompt token through the
same handful of experts: expert 224 is the top expert while reading the prompt in 18 of 20 fields, and the
top-8 expert lists for any two fields overlap by about half. Once the model starts writing its answer the
picture changes completely: every field gets a different top expert, most pairs of fields share no top-8 experts
at all, and this holds on nearly every one of the 40 layers, with no difference between the two kinds of
attention layer. But with only three prompts per field, most of those 20 "field winners" are near-ties that a
single different prompt could overturn; only four (philosophy, psychology, mathematics, linguistics) have a clear
winner. Expert 114 is the philosophy winner by a wide margin, is also near the top for comparative religion,
linguistics, political science, archaeology and physics, and sits far down the list for math, code and medicine;
it is a low-rank expert during prompt reading (rank 124 overall) and rises to rank 3 overall during answering,
firing mainly at layers 26, 14 and 20. Fields that people consider neighbours (law and political science,
computer science and software engineering, philosophy and psychology) share a few generation experts, roughly
three times the overlap of unrelated pairs, but biology, medicine and neuroscience share nothing. The 2026-04-15
re-run reproduced the first run to the last token and adds only provenance.
