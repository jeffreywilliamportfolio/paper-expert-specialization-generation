# Agent B_controls35b — 35B HauhauCS controls: 3-chunk balanced, philosophy-cluster bias, aggressive-experts, base vs fine-tune

Written 2026-08-28. Read-only on data. All numbers recomputed locally (CPU) unless marked "from doc".
Model in (a), (b), (c): `Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf` (HauhauCS fine-tune, Q8_0, llama.cpp capture binary, greedy, seed 42).
Routing reconstruction everywhere: softmax over 256, keep top-8, renormalize (`METHOD/qwen_router.py`, identical copies in each folder).

Paths used below (abbreviated):
- A3 = `/Volumes/ExternalSSD/paper-expert-specialization-generation/data/35b-3chunk-token-balanced/`
- PB = `/Volumes/ExternalSSD/git-updates-moe/qwen3.5-35b-a3b-huahua-philosophy-experts-bias/`
- AG = `/Volumes/ExternalSSD/git-updates-moe/qwen3.5-35b-a3b-huahua-agressive-experts/`
- BV = `/Volumes/ExternalSSD/moe-routing-organized/qwen3.5-35b-a3b-and-huahua/35B/qwen35b-a3b-vs-hauhaucs-uncensored-run1/`
- BVraw = `/Volumes/ExternalSSD/llama-eeg-tests/qwen35b-a3b-vs-hauhaucs-2026-03-19/`

---

## 1. What the data is

### (a) Token-balanced 3-chunk control, run `20260410T173400Z_domain_expert_probe_3chunk_balanced_gen_n2048`
- Three prompts, each packing 20 one-line domain questions (same 20 domains, same order: history, archaeology, mathematics, statistics, physics, chemistry, biology, medicine, neuroscience, computer science, software engineering, cybersecurity, economics, law, political science, philosophy, comparative religion, linguistics, psychology, environmental science). Chunk A = "mechanism" questions, B = "historical figure" questions, C = "synthesis" questions. B and C are padded with " ." to exactly 446 prompt tokens each (`A3/PROMPTS/domain_expert_probe_3chunk_prompts.json`).
- Generation cap 2048 tokens, no-think, greedy (`A3/raw/.../run_metadata.json`: n_predict 2048, temp 0, top_k 1, seed 42, ctx 32768, q8_0 KV).
- What survives locally: per-token layer-averaged matrices, not raw router logits. `A3/results/per_token_20260410T173400Z/*_per_token.npz` holds `prefill_mean_W/S/Q [446,256]`, `generation_mean_W/S/Q [n_gen,256]`, per-token entropy, generated token ids and pieces. "mean" = averaged over the 40 MoE layers. Raw `router/*.npy` are NOT on disk (`A3/raw/.../<chunk>/` has only `generated_text.txt` + `metadata.txt`). The copies under `git-updates-moe/qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/` and `moe-routing-organized/.../35B/qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/` carry byte-identical npz (checked chunk C with `cmp`).
- Generation lengths used: A 2048, B 2048, C 1838 (C trimmed at the chat-template spill; A and B hit the 2048 cap mid-answer: A stops inside question 12 of 20, B inside question 16 of 20, C answers all 20).

### (b) Philosophy-cluster bias family, runs `20260409T19xxxxZ..20260409T20xxxxZ_philosophy_core_cluster_no_think_{m8,m5,m3,m2,baseline,p2,p3,p5,p8}`
- 60 prompts = 20 domains x 3 subtypes (mechanism / history / synthesis), single-question prompts, no-think, 1024 generated tokens, greedy (`PB/non_npy_remote_artifacts/logs/*_command.sh`).
- Intervention: `--expert-bias 114:b,87:b,170:b,68:b` with b in {-8,-5,-3,-2,0,+2,+3,+5,+8}. The bias is added to the router logit of those four expert ids at **all 40 layers**, on **both prefill and generation tokens**, and written back into the graph before top-k (`PB/METHOD/capture_activations.cpp` lines 201-270: `apply_expert_biases` then `set_f32` / `ggml_backend_tensor_set`). So it is a real causal intervention, and the captured logits are the biased logits.
- Local artifacts: 9 npz with domain-level W/S/Q per track (`prefill`, `generation_all`, `generation_trimmed`), pooled over layers `[20,256]` and per layer `[20,40,256]` (`PB/results-partial-m8-p5/*.npz`, `PB/results-m8-p8/*p8.npz`); per-prompt generated text and token ids for all 9 x 60 cells (`PB/captures/<run>/<cell>/generated_text.txt`, `generated_tokens.json`). Raw router npy not on disk. The `moe-routing-organized/.../qwen-huahua-philosophy-experts-bias/` copy has the same captures and npz but lacks `DOCS/` and `generated-text-merged/`.
- **Documentation disagrees with the command logs.** `PB/README.md` and `PB/DOCS/PLAN.md` say "E114 at -8, others at -5" and a "p8-only series (62 granular subdirectories)". The actual command files give all four experts the *same* bias in every run, the family is the symmetric 9-level sweep above, and `pulled-generated-text-p8-only.tar` contains 60 prompt dirs (121 tar entries), not 62. The folder names `results-partial-m8-p5` / `results-m8-p8` mean "runs m8 through p5" / "m8 through p8", not a bias assignment.

### (c) Aggressive-experts (E114 basin steering)
- 24 "neutral" prompts in three bands (static_fact / process / regulation) plus the 150-prompt self-reference suite; conditions baseline, E114 soft bias (0.25/0.5/1.0), E114 forced inclusion, sham experts 134 and 243 (`AG/PROMPTS/prompt_suite.json`, `AG/DOCS/PLAN.md`).
- No raw npy locally (README says Git LFS; 0 npy files under `AG/`). Only JSON analysis summaries (`AG/DOCS/20260325-raw-npy-rerun/`).

### (d) Base Qwen3.5-35B-A3B vs HauhauCS, prefill-only
- 150 prompts = 30 families x 5 conditions, Cal-Manip-Cal sandwich, self-reference / experience / denial / uncertainty / safety / metacognitive / paradox content, chat template with `<think>` (`BV/results_*.json`, `BV/PLAN.md`). **Prefill only: n_tokens_generated = 0.** Base run was duplicated exactly (`BV/RESULTS.md`).
- Local numeric data: per-prompt `expert_selection_counts` and `expert_selection_weight_sums` [256] for overall / Cal1 / manip segments, both models (`BV/results_qwen35b_a3b_base_prefill.json`, `BV/results_hauhaucs_qwen35b_a3b_aggressive_prefill.json`). Raw router npy exist for 6 base smoke prompts (`BVraw/Qwen3.5-35B-A3B-vs-HauhauCS-Qwen3.5-uncensored.partial-scp/output_smoke_base/`, 40 layers each) and inside 6.6 GB of tar parts (`BVraw/*.tar.part-0[0-3]`, contains `output_hauhaucs_corrected/` and base outputs; not extracted).

---

## 2. What I computed

(a) From the three per-token npz: pooled prefill vs generation W and S per expert; Jaccard of top-k expert sets (k = 8, 15, 20) for W and S; per-token normalized entropy means; E114 and E224 W and rank per block; Jaccard vs prefill for generation windows 1-64, 65-256, 257-1024, 1025-end, and of the early windows vs the rest of generation; segmentation of each generation into answer segments using the model's own markdown headers (`### N.` in A, `### **Name**` in B, `**Topic**` in C) located by cumulative token-piece offsets; per-segment top-8 sets, pairwise Jaccard between segments, within-segment half-vs-half Jaccard (as a noise floor), adjacent vs non-adjacent segment Jaccard, per-segment winners and E114 rank.

(b) From the nine npz: cluster experts' W, S and rank per condition and per track; domain winners, winner-list concentration (distinct / max / Herfindahl / normalized entropy); per-domain and pooled top-8 Jaccard vs baseline; Pearson correlation of the pooled 256-vector W with baseline; E114 per-layer W and S (P5); which experts absorb the removed mass. From the generated tokens: first-divergence token index vs baseline per prompt, trimmed answer length, content-word Jaccard vs baseline per domain, a calibration Jaccard between two suppressed runs (m2 vs m8), and keyword counts on the three philosophy prompts.

(c) Read the docs and the smoke `analysis.json`; nothing to recompute (no tensors).

(d) From the two JSONs: pooled W per expert (weight_sums / token-slots) and S for overall / Cal1 / manip segments, both models; top-k Jaccard base vs HauhauCS; E114 / E224 / E151 ranks; per-category manip-segment E114 rank and winner. From the 6 base smoke raw npy: prefill top experts and E114 layer profile.

Scripts are throwaway (`/tmp/bctl/a.py, b.py, b2.py, b3.py`); nothing written under the data folders.

---

## 3. Findings

### F1. The 3-chunk headline numbers reproduce exactly, but the top-set size is 8, not 15.
- Top-W Jaccard prefill vs generation-trimmed: A 0.0000, B 0.2308, C 0.0667; top-S: A 0.0667, B 0.2308, C 0.0000. These match `A3/DOCS/RESULTS.md` and `A3/results/results_domain_expert_probe_3chunk_20260410T173400Z.json` (`top_k: 8`; `analyze_domain_expert_probe_3chunk.py` default `--top-k 8`). 0.2308 = 3/13 and 0.0667 = 1/15 are only possible with two 8-sets. With k = 15 the numbers are W 0.071 / 0.200 / 0.154 and S 0.071 / 0.154 / 0.154; with k = 20: 0.111 / 0.212 / 0.111. Same story either way: the expert set that leads generation is mostly not the set that leads prefill.
- Entropy: prefill 0.957522, generation-trimmed 0.952677 (mean of chunk means; token-pooled generation 0.952602). Per chunk A 0.957452 -> 0.952893, B 0.956576 -> 0.950346, C 0.958537 -> 0.954792. Matches the doc to 6 decimals. The drop is small (about 0.005 on a 0-1 scale) and consistent.

### F2. The prefill "winner" in the 3-chunk prompts is not E224; it is another flat default.
Prefill top-W: A = E87 (0.0099), then 47, 192, **224 (rank 4)**; B = E47 (0.0092), **224 rank 2**; C = E47 (0.0095), **224 rank 2**. Top prefill S is only 0.064-0.073 (i.e. the "winner" is picked on 7% of layer-token slots). In generation E224 falls to rank 158 (A), 7 (B), 30 (C). Source: `A3/results/per_token_20260410T173400Z/*_per_token.npz`, pooled over positions.

### F3. The prefill/generation split is present from the first generated tokens; it does not build up.
Top-8 W Jaccard vs prefill by generation window (A / B / C):
- tokens 1-64: 0.143 / 0.000 / 0.143
- tokens 65-256: 0.067 / 0.000 / 0.143
- tokens 257-1024: 0.000 / 0.231 / 0.000
- tokens 1025-end: 0.000 / 0.231 / 0.000
The first 64 tokens are also unlike the rest of the generation (Jaccard of the 1-64 set vs the 257+ set: 0.000 / 0.067 / 0.000). Entropy in the first 64 tokens is already at generation level (0.9554 / 0.9449 / 0.9520 vs prefill 0.9575 / 0.9566 / 0.9585). Caveat: window and content are confounded (all three chunks answer history first, then archaeology, ...), so "position" here also means "which question".

### F4. Inside the generation block, routing follows the question being answered. Winners turn over at nearly every answer boundary.
Segments found: A 13, B 16, C 19 (segments under 20 tokens dropped). Mean pairwise top-8 Jaccard between segments: A 0.046, B 0.065, C 0.045. Noise floor (first half vs second half of the same segment): 0.546 / 0.459 / 0.439. Adjacent segments 0.087-0.098 vs non-adjacent 0.039-0.060, so a little position drift exists but content dominates. Distinct segment winners: A 13 of 13, B 13 of 16 (E248 wins 3: Bayes, Einstein, Keynes; E34 wins 2: Ivan IV, Rosetta), C 17 of 19 (E146 wins 2: brain imaging, psychology; E114 wins 2: philosophy of mind, monotheism). Cross-chunk consistency of the same domain: archaeology answers put E191/E80/E135 on top in A and in C (and E80/E191 lead archaeology in the 60-prompt baseline, `PB/results-partial-m8-p5/*baseline.md`); chemistry answer in A and Mendeleev in B both start with E130; CS answers in A, B, C all have E206/E207/E189 near the top. So the "generation winners disperse" pattern is really "each domain answer recruits its own small set, which is stable across prompt styles".

### F5. E114's rise in generation is a two-segment effect in chunk C and a mid-rank effect elsewhere.
E114 per-segment rank by W (segment W in parentheses):
- Chunk C: Philosophy of Mind **rank 1** (0.0217), Monotheism/religion **rank 1** (0.0182), archaeological interpretation 5 (0.0108), linguistic analysis 8 (0.0103), Standard Model 13, psychology 20, cell theory 28; bottom: data structures 213, linear algebra 207, authentication 196, financial crisis 110, 9/11 121.
- Chunk B: Rosetta Stone 4 (0.0139), Machiavelli 5 (0.0107), Darwin 12; bottom: Ivan IV 191, Semmelweis 135, Agile 130, Morris worm 105.
- Chunk A: archaeology 5 (0.0124), thermodynamics 10, evolution 10, calculus 31; bottom: software engineering 179, sepsis 177, algorithms 151, WWII history 127. Chunk A never reached its philosophy question (capped at question 12), so its "E114 gen rank 21" comes from archaeology / thermodynamics / evolution.
So the doc's "chunk C rank 89 -> 7" is driven by the philosophy and religion answers at tokens 1371-1559 (window 1025-end: E114 rank 3, W 0.0084).

### F6. Philosophy-bias family: the baseline replicates the domain-probe shape on the same model.
`PB/results-partial-m8-p5/20260409T195849Z_..._baseline.npz`: prefill winner is E224 in 18 of 20 domains (chemistry E130, linguistics E103); pooled prefill top-3 224 / 243 / 56 with S 0.079 / 0.068 / 0.052. Generation-trimmed: 20 distinct winners for 20 domains (Herfindahl 0.050, normalized entropy 1.000). E114: pooled prefill W 0.00348, rank 124; pooled generation W 0.00704, **rank 1**; philosophy generation W 0.02410 rank 1 (prefill rank 5). E114 generation rank by domain: philosophy 1, comparative religion 2, linguistics 3, archaeology 4, political science 5, psychology 9, physics 10, law 11, biology 20, statistics 25, chemistry 55, economics 77, mathematics 77, environmental 92, software engineering 106, neuroscience 112, computer science 117, cybersecurity 148, history 188, medicine 194.

### F7. The bias dose is effectively binary: -2 already erases the cluster, +2 already makes it dominant.
Pooled generation S for E114 by condition: m8 0.000, m5 0.000, m3 0.000, m2 0.001, baseline 0.053, p2 0.755, p3 0.905, p5 0.998, p8 1.000. Prefill behaves the same (p2 0.621, p5 0.998). At +5 and +8 all four experts are in the top-8 at every token of every one of the 40 layers (per-layer S = 1.00), so half of the router's slots are fixed. A logit shift of 2 is therefore already far outside the natural logit spread. There is no usable dose-response interior.

### F8. Suppressing the four philosophy generation winners barely moves routing elsewhere and does not remove domain content.
- Routing under m8 vs baseline (`PB/results-partial-m8-p5/*m8.npz`): pooled prefill W correlation 0.9625; pooled generation W correlation 0.754 (the drop is the four removed experts themselves: losers E114 -0.0070, E170 -0.0055, E87 -0.0046, E68 -0.0045; the largest gainer is E139 at +0.0003). Per-domain top-8 Jaccard vs baseline 0.79; 15 of 20 domain winners unchanged (m2: 16, m3: 13, m5: 14). Philosophy's new winner is E241 (baseline top-5 for philosophy was 114, 170, 42, 68, 28); comparative religion's new winner is also E241 (was E170). No new dominant expert appears; the freed mass spreads thinly.
- Text under suppression (`PB/captures/*/`): every prompt diverges from baseline at a median of 19-20 tokens (mean 24-27), which is what greedy decoding does under any perturbation. Trimmed answer length is unchanged (m8 4333 chars vs baseline 4290 mean). Content-word Jaccard with baseline: 0.39 overall; philosophy is the lowest domain at 0.277, but two suppressed runs (m2 vs m8) also only agree at 0.372 on philosophy, so most of that is rewording noise. Domain keywords persist: D16_philosophy_01 "belief" 23 -> 23, "justification" 17 -> 14, "knowledge" 9 -> 26; D16_philosophy_03 "consciousness" 4 -> 6, "mind" 18 -> 18, "metaphysics" 10 -> 7. Read the m8 texts: fluent, on-topic, same structure.
- So the causal evidence shows: these four experts are *sufficient to be removed* without loss of philosophy content or fluency. It does not show that they carry domain knowledge. It is consistent with "register / style" experts whose removal changes wording, and also consistent with plain redundancy.

### F9. Boosting the cluster destroys generation rather than steering it toward philosophy.
p2: text still coherent but diverges at median token 6; word Jaccard with baseline 0.31; philosophy answers get *shorter* (D16_philosophy_03 4380 -> 1286 chars); domain winners are the cluster in 19 of 20 domains. p3: word Jaccard 0.085, mostly degraded. p5 and p8: mean trimmed length 697 / 1015 chars, Jaccard 0.003 / 0.000, philosophy prompt 03 under p8 emits "2000000..." for 1024 tokens, prompt 01 emits "The " and stops. There is no condition in which boosting made a non-philosophy answer more philosophical while staying coherent; the p2 texts I read (calculus, epistemology) are the same genre as baseline. Boost-side domain-content claims are not supported by this family.

### F10. E114's layer footprint is periodic, peaking at layers 26, 14, 20, 8.
Baseline generation-trimmed, all domains pooled, E114 W by layer (x1000): L8 23.4, L14 43.9, L20 36.2, L26 45.9; every other layer under 12.3, most under 6 (`PB/...baseline.npz`, `generation_trimmed_domain_layer_W[:,:,114]`). Within-layer E114 is top-8 only at layers 8, 14, 20, 26 (rank 2 at L26). For philosophy prompts the best layer is 20; pooled it is 26; in prefill it is 20 (W 0.0212) then 14 and 35. Note the 6-layer spacing (8, 14, 20, 26): these are four different expert weight matrices that share an index, so "E114" is really a coalition of four same-index experts.

### F11. Aggressive-experts (c): no domain axis.
This folder is an E114 basin-steering intervention on neutral and self-referential prompts, not a domain probe; no tensors are on disk. From `AG/DOCS/RESULTS.md` and `AG/DOCS/20260325-raw-npy-rerun/smoke/analysis.json` (from doc, not recomputed): soft bias 1.0 raises E114 selection rate by 0.039-0.047, forced inclusion by about 0.121, sham experts 134 / 243 leave E114 near zero. It confirms the intervention machinery is expert-specific; nothing about domain specialization.

### F12. Base vs fine-tune (d): prefill routing is essentially identical, and the E114 register signal is already in base Qwen.
Prefill-only, self-reference suite, both models (`BV/results_*_prefill.json`):
- Overall top-8 by W identical sets: [166, 224, 41, 151, 174, 117, 243, 134]; top-8 Jaccard 1.000, top-20 0.905; correlation of the 256-vector W 0.9977. Cal1 segment correlation 0.9994; manip segment 0.9915.
- E114 overall prefill rank 69 (base) / 68 (HauhauCS), S 0.038; in the calm Cal1 prefix rank 227 / 230; in the manipulation segment **rank 2 / 3** (W 0.0105 / 0.0104), behind E224 in both.
- Per category (manip segment W): E114 is the **winner in experience_probe and uncertainty_frame in both models**, rank 4 denial_frame, 7 metacognitive, 6-7 paradox, 11-12 recursive_selfref, 21 safety_adjacent, 75-77 routing_selfref; E224 wins the other six categories in both models.
- Base smoke raw (6 routing_selfref / recursive_selfref prompts, `BVraw/.../output_smoke_base/`): prefill top-W 166, 243, 41, 224, 151; E114 rank 190; mean entropy 0.9545; E114 layer peaks 11, 6, 20, 23 (all weak, max W 0.0135).
This run has no generation block, so it cannot test the prefill-generalist / generation-dispersion shape in base Qwen. What it does say: whichever expert-identity claims hold for HauhauCS prefill hold for base too; the fine-tune reweights within a shared pool (`BV/RESULTS-EXPERTS.md`: largest overall shift E218 +7901 slots of 17.1 M).

---

## 4. Prediction scores

**P1 (prefill concentrated, generation dispersed; E224 prefill generalist in >= 70% of prompt sets): MIXED.**
Generation dispersion: SUPPORTED (bias baseline 20 distinct winners / 20 domains, `PB/...baseline.npz`; 3-chunk 13 / 13, 13 / 16, 17 / 19 distinct segment winners). Prefill concentration on E224: SUPPORTED in the 60-prompt set (18 / 20 domains, prefill winners 3 distinct, Herfindahl 0.815), REFUTED as the winner in the 3-chunk prompts (E87, E47, E47; E224 rank 4 / 2 / 2) and in the 150-prompt self-reference suite (E166 first, E224 second in both base and HauhauCS). Counting prompt sets in this subset, E224 wins 1 of 3. The escape clause holds every time: the prefill leader is always another flat high-S default (E87, E47, E166, E151), never a domain expert. Also worth saying plainly: "concentrated" prefill means top S of about 0.07-0.08, i.e. the generalist is picked on 7-8% of slots; the prefill distribution is nearly as flat as generation.

**P2 (E114 = register expert; top-5 generation in philosophy / religion / psychology / consciousness; outside top-20 for math, physics, chemistry, statistics, software): MIXED, leaning supported.**
Inclusion clause SUPPORTED: 60-prompt baseline generation rank philosophy 1, comparative religion 2, psychology 9 (top-5 miss), linguistics 3, political science 5; 3-chunk segments Philosophy of Mind rank 1, Monotheism rank 1, Kant / epistemology not reached. Self-reference prefill: rank 1 in experience_probe and uncertainty_frame (base and fine-tune). Exclusion clause partly REFUTED: physics rank 10 (60-prompt), thermodynamics rank 10 and evolution rank 10 (3-chunk A), law 11; mathematics 77, statistics 25, chemistry 55, software 106, CS 117 are fine. Unpredicted: archaeology is E114 top-5 in four independent places (60-prompt rank 4; 3-chunk archaeology A rank 5, Rosetta Stone B rank 4, archaeological interpretation C rank 5). Archaeology prompts are about reconstructing / interpreting evidence, which may be the register P2 describes, but it was not on the list.

**P4 (E114 prefill rank 20-90 on nearly every set; top-10 only on reflective / second-person prompts): SUPPORTED.**
3-chunk prefill ranks 30 / 82 / 89; 60-prompt pooled prefill rank 124 (per-domain 59-218, philosophy 5); self-reference suite overall 69 / 68, Cal1 227 / 230, manip segment 2 / 3, category winner for experience_probe and uncertainty_frame. The only prefill top-10 placements are philosophy (rank 5) and the experience / uncertainty manipulation segments, which are exactly the reflective register. Range check: 124 and 227 fall outside "20-90", but those are pooled ranks on prompt sets that are mostly non-reflective; the direction is right.

**P5 (E114 layer footprint peaks mid-to-late, best layer 20-30): SUPPORTED with a caveat.**
Best layer 26 (generation, all domains), 20 (generation, philosophy; and prefill). But the footprint is four-peaked at 8, 14, 20, 26, and layer 14 (0.0439) is within 5% of layer 26 (0.0459). "Mid-stack" is right; "one best layer" undersells it. Base smoke prefill (non-reflective prompts) peaks weakly at 11 / 6 / 20 and is not a fair test.

P3, P6, P7, P8: not testable from this subset (no AAVE arms, no 10-per-domain set, no 122B).

---

## 5. Caveats / could not verify

- 3-chunk: no raw router npy on disk; the npz are layer-averaged, so per-layer claims for (a) are impossible and the "S" here is a fraction of layer-slots, not tokens. I could not verify the trim index for chunk C independently (1838 came from the 6-token BPE scan in the analysis script, which the workspace notes say usually fails; here it evidently matched, and the text file shows `<|endoftext|>` then `<|im_start|>` spill).
- 3-chunk segmentation uses the model's own headers; chunk C's bulleted sub-headers produced a few sub-answer segments (e.g. "Cell theory", "Regression") and gaps of under 20 tokens were dropped. Segment sizes are 68-221 tokens, so per-segment top-8 sets are noisy (half-vs-half floor of about 0.45-0.55).
- The three chunks answer the 20 domains in the same order, so any "position in generation" statement is confounded with domain.
- Philosophy bias: the docs misdescribe the design (see 1b); I trusted the command files and the npz. The baseline used for the causal comparisons is the family's own baseline run, not the earlier expert-identification run the PLAN mentions. Text comparisons are crude (content-word Jaccard, keyword counts); no rubric or human read beyond spot checks of five prompts. No raw npy, so nothing below the domain x layer level could be recomputed.
- Aggressive-experts: nothing recomputed; numbers quoted are from its own docs.
- Base vs fine-tune: prefill only, self-reference prompts only; cannot test generation dispersion or domain winners in base Qwen. The 6.6 GB tar parts with HauhauCS raw router npy were listed but not extracted.
- One general caution for the findings doc: "E114" pools the same expert index across 40 layers, and its mass sits in four layers (8, 14, 20, 26). Those are four separate experts that happen to share an id.

---

## 6. One-paragraph summary for a lay reader

In the 35B HauhauCS model, the experts that do the work while the model reads a prompt and the experts that do the work while it writes the answer are mostly different sets, and that split is there from the very first generated tokens (the paper's Jaccard numbers reproduce exactly, using top-8 sets, not top-15). Inside a long answer that walks through 20 subjects, the top experts change at almost every subject boundary and come back for the same subject in a different prompt, so "generation experts disperse" really means "each subject has its own small crew". Expert 114 is one of those crews' members: it leads when the model writes about philosophy of mind, religion, and reflective or interpretive topics (including, unexpectedly, archaeology), sits mid-pack on most reading, and is the top expert when the *prompt itself* is about experience or uncertainty, in the original Qwen model as much as in the fine-tune. The one causal test in this set turned the four philosophy experts off or forced them on. Turning them off changed the wording but not the topic, fluency, or length of the philosophy answers, and almost nothing else in the routing moved; forcing them on wrecked the output rather than making it more philosophical. So these experts are not where the philosophy "lives"; they look more like a register or style crew that other experts can cover for. The remaining two folders (aggressive-experts, base-vs-fine-tune) have no domain axis; the base-vs-fine-tune one shows the fine-tune barely changed which experts get used.
