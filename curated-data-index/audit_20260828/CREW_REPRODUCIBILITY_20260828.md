# Same subject, same crew: cross-design reproducibility of generation expert sets (2026-08-28)

Data: 3-chunk token-balanced run 20260410T173400Z (per-token layer-averaged W, chunks A mechanism /
B biography / C synthesis, segmented at the model's own markdown headers: A 12, B 15, C 19 answers)
and the 60-prompt domain probe 20260408T235839Z (generation_trimmed per-domain W). All Qwen3.5-35B-A3B
HauhauCS Q8_0. Expert set = top-8 by mean W over the segment (or domain). Similarity = Jaccard.
Sources: expert-specialization-data/qwen35b/{token_balanced_3chunk,domain_probe_60prompt}.

Floor (same subject, same answer, first half vs second half): 0.49 (n=46 segments).
Between subjects inside one answer: A 0.053, B 0.072, C 0.046.

Nearest-neighbour identification: for subject d in design X, the design-Y subject whose top-8 is most
similar; correct if it is d. Strict argmax, ties count as miss.

| Design pair | Correct | Rate | Matched-subject Jaccard |
|---|---|---|---|
| All ordered pairs of A, B, C, P(probe) | 117/168 | 0.70 | 0.270 (mismatched 0.032) |
| A <-> P | 23/24 | 0.96 | 0.387 |
| C <-> P | 36/38 | 0.95 | 0.379 |
| A <-> C | 14/24 | 0.58 | 0.201 |
| all pairs excluding B | 73/86 | 0.85 | 0.332 |
| pairs involving B (biography) | 44/82 | 0.54 | 0.206 |
| label-shuffle null, 200 draws | mean 0.063 | 95th pct 0.095, max 0.119 | |

Per subject (hits / comparisons; top-3 per design):
archaeology 12/12 (A 191,80,135; C 191,80,116; P 191,80,210); biology 12/12; medicine 12/12;
neuroscience 12/12 (A 54,24,158; B 54,24,220; P 54,24,26); cybersecurity 12/12; chemistry 11/12
(130 leads in A, B, P); software_engineering 11/12; physics 10/12; statistics 9/12 (202,106);
mathematics 8/12 (100 in A, B, P); history 6/12; computer_science 6/12 (206,207,189 in A, C, P; B differs);
economics 5/6; political_science 4/6; law, philosophy (114), comparative_religion (114/170), linguistics
(103,42), psychology (146,228), environmental_science (8,167,39): 2/2 each (C and P only).

Reading: a subject's generation crew measured in one prompt design identifies that subject in an
independent design 85 to 96% of the time (chance 6%). The shortfall is concentrated in chunk B, whose
prompts asked for biographies rather than explanations: the subject expert persists (mathematics 100,
neuroscience 54/24, chemistry 130) but the rest of the crew follows the register. This is Finding 2 and
Finding 4 of the audit in one measurement.

Caveats: segment boundaries are the model's own headers (C skipped law; A capped at 12, B at 15);
per-token W is layer-averaged; probe W is a mean over 3 cells x 40 layers; six subjects have only the
C-P comparison.

## Null controls 4-7 (run 2026-08-28 22:20; script $CLAUDE_JOB_DIR/tmp/nulls.py)

Caveat for #7: HF Qwen3.5 tokenizer gives 392 tokens for the packed prompt vs 446 in the llama.cpp
capture (literal chat-template text and " ." padding tokenized differently); prefill question spans
were rescaled proportionally (edge error about 1-2 tokens on ~20-token questions). Generation
numbers are exact.

#4 Chance Jaccard, two random top-8-of-256: 0.016 (analytic 0.0159, simulated 0.0171).
   Generation matched 0.270 (17x chance); mismatched 0.032 (2x chance).
#5 Position-matched null (A vs C, C reordered): same-position different-subject 0.020 vs
   same-subject different-position 0.194; subject wins 9/10, 1 tie at 0 (history).
#6 Adjacent-pool null (candidates = prev/same/next subject in target order): 139/168 = 0.83
   (chance ~0.33); excl. B 78/86 = 0.91; adjacent Jaccard 0.047.
#7 Prefill control (identification on prefill rows of the question):
   pair        generation        prefill
   A<->P       23/24 = 0.96      7/40 = 0.17   (matched 0.119 / mism 0.054)
   C<->P       36/38 = 0.95      6/40 = 0.15   (matched 0.138 / mism 0.065)
   all pairs   117/168 = 0.70    73/240 = 0.30 (matched 0.202 / mism 0.081)
   A<->C       14/24 = 0.58      27/40 = 0.68  (matched 0.388 / mism 0.098)
   label-shuffle null: generation 0.063, prefill 0.050.
   Per-question prefill: between-question Jaccard A 0.133 / B 0.208 / C 0.136; E224 in top-8 of
   3/12, 6/15, 9/19 questions.
Reading: prefill crews follow the question's tokens (chunk-to-chunk 0.68 where the wording is
shared; chunk-to-probe 0.15-0.17 despite identical questions, because the probe pools three
prompts and the chat template). Generation crews follow the subject of the answer and transfer
across designs at 0.95-0.96.

## Base-model check (official Qwen3.5-35B-A3B, ggml-org Q8_0), 50-pair register run 20260505T205437Z

Scope note: the 20-domain probe was run on HauhauCS only. The nearest base data with generation and
a topic-like axis is the 50-pair run (8 prompt types x 2 dialects x {base, HauhauCS}). Crew = top-8 by
gentrim W per (model, dialect, type), from expert-specialization-data/qwen35b/controls/compact/aave_5-5_register_run.

Nearest-neighbour identification over 8 types (chance 1/8):
  base AAVE -> base AE          gen 8/8 (matched 0.733 / mism 0.092)   prefill 4/8 (0.694 / 0.414)
  base AE -> base AAVE          gen 8/8                                 prefill 7/8
  hauhau AAVE -> hauhau AE      gen 8/8 (0.698 / 0.108)                 prefill 4/8 (0.748 / 0.487)
  base -> hauhau, AAVE          gen 7/8 (0.630 / 0.091)                 prefill 8/8 (0.861 / 0.517)
  base -> hauhau, AE            gen 7/8 (0.658 / 0.089)                 prefill 8/8 (0.789 / 0.419)
  base AAVE -> hauhau AE        gen 8/8 (0.653 / 0.082)                 prefill 6/8 (0.730 / 0.438)
  base within-type split-half floor (gen) 0.518; chance 0.016; base prefill top-5 = 224, 95, 46, 64, 130.
Crews shared across dialect and model: legal 122/109/85; medical 206/247; identity 103/139; dream 206/147;
advice 56/69/250; fact_distortion 220/27; relationship 250/56.
Reading: prefill crews all resemble the default set (mismatched 0.41-0.52), so prefill "identification"
is weak discrimination; generation crews discriminate at 8:1 and are the same experts in base and
fine-tune. Missing: the 20-domain probe on base (one box-hour).

## Base-model rerun of the paper's shape numbers (official Qwen3.5-35B-A3B, ggml-org Q8_0), 2026-08-28 22:30

50-pair run (100 base prompts, 8 types; compact/aave_5-5_register_run, run_base_nothink):
  prefill per-type winners [224 x6, 95, 65] -> 3 distinct, max 6, Herf 0.594, ent 0.35
  generation per-type winners [56,206,228,220,103,122,206,250] -> 7 distinct, max 2, Herf 0.156, ent 0.92
  pooled prefill top-8 {46,47,64,95,130,189,224,243} vs generation {56,61,69,142,146,181,206,250}: Jaccard 0.000
  per-prompt prefill-vs-gen Jaccard mean 0.108; between-type Jaccard prefill 0.421 vs generation 0.093
Medical base (compact/aave_5-15_medical, 24 cells each): prefill top-5 47,247,36,65,95; gen top-5 247,47,189,82,52
  (nothink) / 47,189,247,43,36 (think); pooled prefill-vs-gen Jaccard 0.231 / 0.333; per-cell 0.345 / 0.367.
l1l3 base RAW (controls/l1l3_register_base_raw, 30 prompts, layers 0-38, trimmed at first EOG):
  per-token entropy prefill 0.9557 vs generation 0.9583; per-prompt prefill-vs-gen Jaccard 0.453
  routing_selfref  prefill 166,41,224,151,174 | gen 166,151,117,41,174 | Jaccard 0.778 | E114 gen rank 223
  recursive        prefill 166,224,41,151,117 | gen 166,151,117,243,41 | Jaccard 0.600 | E114 gen rank 63
  experience       prefill 166,224,41,151,117 | gen 114,166,228,151,42 | Jaccard 0.333 | E114 gen rank 1
  between-register Jaccard prefill 0.852 vs generation 0.463
Reading: the prefill-generalist / generation-dispersed shape replicates on base for short, mixed prompts
(50-pair). It does NOT hold when a long coherent technical prompt is followed by a technical answer
(l1l3: shared crew 166/151/41, Jaccard 0.6-0.78) or when a narrow-topic prompt already routes to its
topic crew (medical 247/47). The "regime effect" is a content/register effect that short-prompt prefill
exposes; the paper should scope it that way.
