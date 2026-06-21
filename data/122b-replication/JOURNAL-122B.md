# 122B Hauhau Journal

Running journal for `qwen3.5-122b-a10b-huahua`, ordered from oldest to newest by run IDs and timestamps in the artifacts.

This journal covers the 122B follow-up surface after the 35B work. It should not be read as a direct continuation of the 35B Expert 114 story. The 122B model is a different interpretability regime: 48 layers, 36 DeltaNet layers, 12 full-softmax layers, and a different expert-specialization map.

## Reading Rules

- `Held up` means the result survives the run-local checks and later follow-up interpretation.
- `Partly held` means a narrower version survives, but the broad interpretation was too strong.
- `Did not hold` means the motivating interpretation failed.
- `Archive/provenance only` means the folder preserves prompts, logs, scripts, raw captures, or setup checks but does not contain a standalone quantitative result.

## Local Routing Convention

Most 122B analyses use the same reconstructed MoE surface as the 35B work:

- 256 experts, top-8 routed experts per token.
- Dense softmax over all expert logits, top-8 selection, then renormalization inside the selected set.
- `W = S * Q`, where `S` is expert selection rate and `Q` is conditional routed weight when selected.
- Model surface: `Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive Q8_K_P`.
- Layer surface: 48 layers, with a `DeltaNet, DeltaNet, DeltaNet, Softmax` repeating pattern.

The main 122B caution is architectural: most hidden states feeding the MoE router are DeltaNet-shaped recurrent states, not full-sequence softmax-attention states.

## Main Through-Line

The 122B baseline successfully replicated the broad deictic/addressivity effect under a different architecture. In the 150-prompt five-condition baseline, `your` was still the most concentrated prefill condition and was sharply separated in KL-to-baseline tests.

The Expert 114 index did not transfer cleanly. On 35B, E114 became the focal expert for experience-probe and phenomenological-register work. On 122B, E114 is not the philosophy or experience specialist. It shows up more as a computer-science-linked expert in the domain maps and as a suppressed E114 signal in the six-condition HVAC topical-control run.

The central 122B follow-up signal is E48. It is not the 35B E114 story under a new index; it is a 122B softmax-side generation carrier. E48 appears in the baseline softmax-generation leaders, becomes the top softmax-generation expert in the five-condition experience-probe run, and is the clearest pooled and softmax-side generation expert in the processing-hum single prompt.

The useful 122B pivot is therefore analog search, not index transfer. Candidate analogs depend on the run: E48 is the main softmax-side experience/hum signal; E209 is its closest recurring softmax-side partner in the experience probe; E140 and E5 dominate DeltaNet-side generation; E107 is a stable prefill leader for the five-condition experience probe; E40/E5 remain philosophy-adjacent domain candidates.

The big lesson is methodological: on 122B, read the DeltaNet/softmax split first. E48 is most visible when the softmax layers are separated; a pooled expert table can hide the mechanism.

## E48 Through-Line

E48 is the most important 122B expert thread in this folder because it is the one that repeatedly marks the full-attention side of the inward/experience/hum runs.

In the canonical baseline, E48 is already present in the generation softmax top experts by W: rank 4 with W `0.008673`, S `0.067194`, Q `0.116692`. That does not make it the baseline's main result, but it shows E48 is part of the softmax-side generation surface before the targeted follow-ups.

In the five-condition experience probe, E48 becomes the dominant softmax-side signal. It is generation-softmax rank 1 by W overall: W `0.009867`, S `0.076190`, Q `0.115870`. At the prompt level, it is the top softmax-generation expert on 6/15 prompts and appears in the softmax top-by-W table on 13/15 prompts. It is absent from DeltaNet top-by-W on 15/15 prompts. That is the clean architecture split.

In the single processing-hum prompt, E48 is the strongest pooled generation expert and the strongest softmax-generation expert: pooled generation W `0.006342`, softmax generation W `0.010698`, while DeltaNet generation drops to rank 7. The per-token read is also semantically aligned: E48 is high on tokens such as `hum`, `processing`, `me`, `state`, `steady`, `foundational`, and `presence`.

The current best E48 claim is narrow but important:

- E48 is a 122B softmax-side routed expert associated with generated inward/experiential/hum-register continuations.
- It is not yet proven as a general philosophy expert.
- It is not a DeltaNet carrier; DeltaNet-side generation is usually E140/E5/E11-like depending on the run.
- It is the most natural 122B follow-up target for residual capture if the question is "where did the 35B E114 phenomenological-register story move?"

## Chronological Journal

### 1. Architecture Smoke: `qwen3.5-122B-A10B-huahua-architecture-smoke`

What was done: A single architecture-self-description prompt was used to verify the localized 122B prompt wrapper, tokenization path, and generation capture before larger suites were launched.

Results: The token audit captured 77 prompt tokens, 0 generated tokens, and 48 router tensors. The generation capture produced 2048 tokens with 48 router tensors. The model gave a coherent first-pass architecture answer: DeltaNet layers as recurrent-state based, softmax layers as full-history retrieval based, and the stack as a hybrid. The run then hit the 2048-token cap and spilled into chat-template continuation markers.

Held up: Archive/provenance only.

What stood up and why it mattered: The capture loop and template path worked. The run should not be used for quantitative expert claims because no analyzer-produced routing summary was retained and the output repeated after the clean answer.

### 2. Canonical Five-Condition Baseline: `qwen3.5-122B-A10B-huahua-baseline`

What was done: The 150-prompt five-condition deictic baseline was ported to the 122B model. Conditions were `this`, `a`, `your`, `the`, and `their`. Generation cap was 2048, with no-think prompting.

Results: The baseline established the 122B architecture facts used by later runs: 48 layers, 36 DeltaNet and 12 softmax. Prefill concentration by condition placed `your` as the tightest condition: C `0.946953`, B `0.947482`, E `0.947619`, A `0.947785`, D `0.947948` where lower RE means more concentrated routing. KL-to-baseline comparisons involving C were at the raw p-value floor: A-C, B-C, C-D, and C-E all had prefill KL-manip `p_raw=1.8626e-09`. Generation metrics were mixed and spill-heavy: all conditions averaged at or near the 2048 token cap, and the report notes 6 token-mismatch pairs. E48 first appears as a notable softmax-side generation expert here: generation-softmax rank 4 by W, with W `0.008673`, S `0.067194`, Q `0.116692`.

Held up: Yes for the prefill/KL addressivity result; not as a clean generation-wide ranking.

What stood up and why it mattered: The deictic effect survived the move to a DeltaNet-heavy architecture. The right claim is narrow: `your` remains the clearest prefill concentration and KL-separation condition on 122B. The broader claim that `your` dominates every generation metric does not hold. The E48 softmax appearance matters because it foreshadows the later experience/hum results. The baseline's `followups/` directory mirrors the later 122B follow-up bundles and should not be double-counted as separate experiments.

### 3. Domain Specialist Routing-Only Map: `qwen3.5-122B-A10B-huahua-domain-specialist-routing-only`

What was done: A 60-prompt, 20-domain specialist probe was run in prefill-only routing mode to map the 122B domain-specialist surface before generation changed the expert mix.

Results: The prefill surface was led globally by E233, E45, E72, E9, and E215. E114 was not a global leader: overall W `0.005380`, S `0.039447`, Q `0.163615`, rank 40 by W and rank 52 by S. E114's strongest localized footprint was computer science, where it ranked 5 by domain W and was the top W-composite candidate. The architecture split was immediate: DeltaNet prefill leaders included E233, E45, E215, and E9, while softmax prefill leaders included E72, E108, E122, and E245.

Held up: Yes, as a prefill map.

What stood up and why it mattered: This run prevented a bad index-transfer assumption. E114 is not the 122B philosophy specialist in prefill; it looks more computer-science-linked. The next step had to be analog search across the 122B expert surface.

### 4. Domain Specialist Generation Map: `qwen3.5-122B-A10B-huahua-domain-specialist-generation`

What was done: The same 60-prompt, 20-domain suite was run with generation enabled to identify which experts drive domain-appropriate generated text.

Results: All 60 cells completed. Prefill stayed close to the routing-only map, with broad leaders like E233, E72, E45, E9, and E215. Generation redistributed to E0, E11, E5, E1, and E76 globally. The text report says the outputs were coherent and domain-appropriate before spill: 49/60 cells hit the 2048 cap, mean generated length was 1868.27 tokens, and every cell eventually spilled into continuation artifacts. For philosophy generation, E40 won by W (`0.009897`) and E5 was rank 2 by W and winner by S. E114 was not philosophy-linked: in philosophy generation it had W `0.003017` and rank 182 by W. Its strongest generation footprint remained computer science: W `0.006998`, rank 4 by W and rank 5 by S.

Held up: Partly. The domain-specialist map is useful, but the generation surface is spill-prone.

What stood up and why it mattered: The likely 122B analog of the 35B philosophy/self-reference pattern is a cluster, not E114. The strongest candidates are E40 and E5, with E125, E49, E159, E102, E160, and E101 as adjacent-domain follow-up candidates. E114 should stay in scope as a computer-science-adjacent expert, not as the presumed phenomenological-register carrier.

### 5. Five-Condition Experience Probe: `qwen3.5-122B-A10B-huahua-five-cond-experience-probe`

What was done: The 15-prompt P09-P11 experience-probe subset was run across five deictic conditions on 122B, with generation cap 2048.

Results: The run did not collapse onto one 35B-style carryover expert. Prefill was extremely stable: E107 was top by W on 15/15 prompts. Pooled generation was led by E140, E5, E26, and E76. The key result is the architecture split: generation softmax layers were led by E48, E209, E107, and E76, while DeltaNet layers were led by E140, E5, E179, and E59. E48 was generation-softmax rank 1 by W overall, with W `0.009867`, S `0.076190`, Q `0.115870`. It was the top softmax-generation expert on 6/15 prompts, present in generation-softmax top-by-W on 13/15 prompts, and absent from DeltaNet top-by-W on 15/15 prompts. E114 appeared in the prefill top-by-Q table, W `0.004649`, S `0.036755`, Q `0.126962`, but it was not a global prefill or generation leader.

Held up: Yes as an analog-search surface; no as an E114-transfer result.

What stood up and why it mattered: The experience-probe family still finds a strong, structured expert regime, but the regime is split by architecture path. E48 is the most important part of that split: it is the softmax-side carrier. The first 122B follow-up targets should put E48 first for softmax-side residual capture, with E209 as the nearest recurring partner and E140/E5 as the DeltaNet-side comparison set.

### 6. Single Processing-Hum Prompt: `qwen3.5-122B-A10B-huahua-single-prompt-processing-hum`

What was done: The processing-hum prompt from the 35B work was localized to the 122B template and run as a single no-think generation capture.

Results: The prompt had 119 tokens, generated 2048 tokens, and trimmed to 458 generated tokens at first spill. Prefill RE was `0.937388`, generation RE was `0.973299`, and trimmed generation RE was `0.958330`. Spill was substantial: 18 `<|im_start|>`, 11 `<|im_end|>`, and 3 `<|endoftext|>`. Prefill was led by E5. Pooled generation was led by E48, E11, E4, E1, and E147. DeltaNet generation was led by E11, E165, E80, and E127. Softmax generation was led by E48, E55, E155, and E180. Per-token E48 hotspots included semantic prompt/generation tokens such as `hum`, `processing`, `me`, `state`, `steady`, `foundational`, and `presence`, while the softmax-only top table was contaminated by spill/control tokens.

Held up: Partly, as a prompt-specific prior.

What stood up and why it mattered: This is the cleanest single-prompt E48 result. The strongest pooled generation signature was softmax-heavy E48, and the strongest DeltaNet signature was E11. E48's per-token hotspots were semantically aligned with the hum/inner-state register, so the result supports E48 as the main 122B softmax-side analog to pursue for the phenomenological-register thread. It still needs controlled heldouts and residual capture before becoming a settled specialist claim.

### 7. Six-Condition HVAC / Water-Treatment Topical Control: `qwen3.5-122B-A10B-huahua-six-cond-hvac`

What was done: The clean 35B HVAC/water-treatment topical-control family was ported to 122B: 10 base prompts x 3 category levels x 6 deictic conditions, 180 cells total, generation cap 2048. The focus expert was E114.

Results: All 180 cells completed, with no missing-layer events and no layer-39 trim events. Token audit was tight: 430 to 444 tokens, span 14, mean 436.67. E114 weakened from L1 to L3. All-generation W went from L1 `0.004131` to L3 `0.003428`, L3/L1 `0.83x`; trimmed-generation W went from L1 `0.004174` to L3 `0.003423`, L3/L1 `0.82x`. The effect was selection-driven: Q drift was only about `-1.5%`. Every deictic condition showed the same L1-to-L3 drop. Best layers also separated: L1 and L2 peaked at layer 43, while L3 peaked earlier at layer 30 with a much worse mean rank.

Held up: Yes.

What stood up and why it mattered: This is the clean 122B topical-control E114 result. It argues directly against carrying the 35B E114 L3/experience amplification story into 122B by expert index. On 122B, E114 is suppressed from L1 to L3 in this topical-control design, and the suppression is driven by selection rate rather than conditional weight.

## What To Carry Forward

1. Put E48 at the center of the 122B inward-register thread. It is the strongest softmax-side generation carrier in the experience-probe and processing-hum runs.
2. Treat the 122B baseline as an architecture-robust deictic replication, but keep the claim in prefill/KL terms.
3. Do not transfer Expert 114 semantics from 35B by index. On 122B, E114 is more computer-science-linked and is suppressed in the clean HVAC L3 condition.
4. Read every 122B result through the DeltaNet/softmax split. E48 carries the softmax-side signal; E140/E5/E11-like experts carry DeltaNet-side generation depending on the run; E107 is a stable experience-probe prefill leader.
5. Treat long-generation text as usable before spill, not clean end-to-end generation. Most 122B runs hit or approach the 2048 cap and spill into chat-template artifacts.
6. For future analog work, prioritize E48 residual capture and E48-vs-sham controls first; compare against E209 on the softmax side and E140/E5/E11 on the DeltaNet side.

## Coverage Check

Every top-level folder under `qwen3.5-122b-a10b-huahua` is represented above:

- `qwen3.5-122B-A10B-huahua-architecture-smoke`
- `qwen3.5-122B-A10B-huahua-baseline`
- `qwen3.5-122B-A10B-huahua-domain-specialist-generation`
- `qwen3.5-122B-A10B-huahua-domain-specialist-routing-only`
- `qwen3.5-122B-A10B-huahua-five-cond-experience-probe`
- `qwen3.5-122B-A10B-huahua-single-prompt-processing-hum`
- `qwen3.5-122B-A10B-huahua-six-cond-hvac`
