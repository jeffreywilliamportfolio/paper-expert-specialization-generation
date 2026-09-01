# Numerical audit of the v1.2 drafts (2026-08-30)

Audited: `DRAFT_v12_jeffrey_20260830.md` (JD; title, abstract, intro, methods) and
`DRAFT_v12_sections_20260830.md` (CD; resolutions, Results, Discussion, Limitations, Conclusion, references).
Every number, count, ratio, expert id, layer, model name, quant, hash and citation claim in both drafts was
checked against the sources in the stated priority order. Where no document carried a number it was
recomputed from `/Volumes/ExternalSSD/expert-specialization-data/` (marked RECOMP). The nulls script
(`audit_20260828/crew_reproducibility_nulls.py`) was rerun 2026-08-30 from the repo root; its output is
cited as NULLS. Nothing in either draft file was edited.

Source abbreviations and line numbers: J = `JOURNAL_v12.md`; CREW = `ledger/CREW_REPRODUCIBILITY_20260828.md`;
F2 = `ledger/FINDINGS_expert_specialization_v2_20260829.html` (section / finding id); A, B, C, D, E =
`ledger/agents/{A_primary35b,B_controls35b,C_122b,D_glm,E_register}.md`; PROV = `expert-specialization-data/PROVENANCE.md`;
RM = `expert-specialization-data/README.md`; TEX = `main.tex` (v1.1); NULLS = 2026-08-30 run of the nulls script;
SMOKE = `audit_20260828/base_safety_smoke_crews_OUT.txt`; SC / MY / HB / YE / DO = the per-paper reports in
`ledger/related/`; RW = `ledger/related/RELATED_WORK_REVIEW_20260829.md`; PROMPTS = `qwen35b/token_balanced_3chunk/PROMPTS/domain_expert_probe_3chunk_prompts.json`;
GENTXT = `qwen35b/token_balanced_3chunk/raw/.../generated_text.txt`; ZEN = `job-search/zenodo-redeposit/README.md`;
MEM = `~/.claude/projects/.../memory/entropy-paper2-status.md`; RECOMP = recomputed in this audit.

Counts: 11 MISMATCH, 5 UNSUPPORTED, 170 VERIFIED (rows below). Wilson intervals were recomputed independently
(two-sided 95%, z = 1.96) and all nine in CD reproduce.

## 1. Table, sorted by verdict

### MISMATCH

| # | Draft, line | Sentence (abridged) | Value in draft | Source and line | Verdict / correct value |
|---|---|---|---|---|---|
| M1 | JD 11 (abstract), JD 23, JD 25 (intro), JD 43, JD 47 (methods); CD 135 | "packing multiple subjects into one generation, **reordering them across prompts**" / "reorder the subjects across prompts" / "The subject order in C differs from A" / "A is compared with C, whose subjects occur in a different order" | design choice: C reordered | PROMPTS (all three prompts list the 20 subjects in the identical order history, archaeology, mathematics, ... environmental science); B 237 ("same 20 domains, same order"), B 351 (caveat: "The three chunks answer the 20 domains in the same order"); GENTXT chunk C headers ("History & Politics": 9/11, 2008 crisis, nationalism; then "Archaeology & Science"...); NULLS script `maps['C']` hard-coded to the model's answer order (history, economics, political_science, archaeology, ...) | MISMATCH in attribution. The prompts were never reordered. Chunk C's different subject order is the model's own regrouping of its answer under thematic headers; the position-matched null (0.020 vs 0.194, 9/10) is valid but rests on that emergent reordering, not on a design manipulation. Reword: "in one packed design the model answered the subjects in a different order, which places the same subject at different token positions". |
| M2 | JD 29 (intro) | "expert-set similarity still selects the correct subject 83% of the time, **against 33% by chance**" | 33% | NULLS ("168 queries, 21 edge queries with 2 candidates, exact chance = 0.354"); CD 41-44 | MISMATCH: exact chance is 0.354 (35%), because 21 first/last segments have only two candidates. CD already corrects this; JD intro still says 33%. |
| M3 | JD 51 (provenance) | "the **official-checkpoint** domain probe used llama.cpp build 8493 (1772701f)" | build 8493 = official | PROV 16 (build 8493 (1772701f) is the HauhauCS `domain_probe_60prompt`); A 23; CREW 72 ("the 20-domain probe was run on HauhauCS only") | MISMATCH: build 8493 is the HauhauCS 60-prompt probe (both the 04-08 capture and the bit-identical 04-15 re-capture). The official checkpoint never ran the 60-prompt probe; its runs are the 50-pair register run (PROV 23, "see docs/PLAN.md") and the safety smoke (J 96-100), neither with a recorded build in the ledger. CD 53-55 already flags this. |
| M4 | JD 51 (provenance) | "the GLM **prefill** captures use commit 6658925" | 6658925 = GLM prefill | PROV 52-53 (domain prefill = BF16 transformers hook, no llama.cpp), PROV 56 (commit 6658925 = `compact/register_run`, llama.cpp Q8 GGUF, generation), D 34-35 | MISMATCH: 6658925 is the GLM register generation rig. The domain prefill captures have no llama.cpp commit (Transformers hook on BF16), which JD 39 itself says. CD 56-58 already flags this. |
| M5 | JD 51 (provenance) | "**Per-token tensors** survive for the packed prompts, controls, and GLM captures" | per-token tensors for packed prompts | B 239 ("per-token layer-averaged matrices, not raw router logits ... Raw `router/*.npy` are NOT on disk"); J 8-10; F2 §2 row 2 ("Per-token layer-averaged W/S/Q ... No raw tensors") | MISMATCH (overstated for the packed prompts): what survives is per-token, layer-averaged W/S/Q; no per-layer router tensors. Controls (HVAC, l1l3, AAVE, safety smoke) and GLM do have raw per-layer tensors (RM 25-30, PROV 52-53, F2 §2). Say "per-token layer-averaged profiles for the packed prompts; raw per-layer router tensors for the controls and GLM". |
| M6 | CD 112-113 (Results 3.3) | "overlap between **adjacent** subjects in the same answer is 0.053, 0.072 and 0.046 in prompts A, B and C" | 0.053 / 0.072 / 0.046 labelled adjacent | CREW 10 ("Between subjects inside one answer: A 0.053, B 0.072, C 0.046" = mean over all subject pairs in the answer); NULLS #6 (adjacent-only, pooled: 0.047); B 290 (adjacent 0.087-0.098 vs non-adjacent 0.039-0.060 on its 13/16/19 segmentation) | MISMATCH (label): these three numbers are the all-pairs between-subject means inside one answer, not adjacent pairs. The draft's own Drift paragraph (CD 141) gives adjacent overlap as 0.047, so the two sentences contradict each other. Fix: "overlap between different subjects in the same answer averages 0.053, 0.072 and 0.046". |
| M7 | CD 131-132 (Controls, Wording) | "A to C, 27 of 40 (0.68), where the two packed prompts **share their wording verbatim**" | verbatim shared wording | PROMPTS (A: "Explain calculus as a mathematical system for describing change..."; C: "Explain why linear algebra became foundational..."; every subject has a different question in A and C); CREW 65 ("where the wording is shared"); F2 F2a ("share their wording") | MISMATCH (overstated): A and C share the carrier (one "Explain ..." line per subject, identical subject order, identical chat template and " ." padding), not the question wording. The ledger phrasing "share their wording" is itself loose; "verbatim" is wrong. Say "share the same one-line carrier and subject order". |
| M8 | CD 159-160 (Controls, Checkpoint); CD 235 (Limitations) | "overlap 0.70 between checkpoints **against 0.43 between prompts of the same type**" / "moves routing behaviour by less than a prompt change does" | 0.43 = same-type floor | J 387-390 (table: "different prompt, same checkpoint (floor)" 0.581 prefill / 0.428 generation; no script cited); RECOMP from `controls/compact/aave_5-5_register_run` (200 cells, top-8 by layer-pooled gentrim W): same prompt official vs HauhauCS = 0.702 (min 0.23) generation, 0.853 (min 0.45) prefill, both reproduce exactly; all different-prompt pairs 0.076 gen / 0.410 prefill; same-type different-prompt pairs 0.174 gen / 0.551 prefill; consecutive prompt ids in sorted order (001_aave-001_ae, 001_ae-002_aave, ...) 0.428 gen / 0.581 prefill | MISMATCH (mislabelled floor): 0.428 and 0.581 reproduce only as the mean over consecutive prompt ids, which alternates matched AAVE/AE pairs of the same content (0.75) with unrelated adjacent prompts (about 0.1). It is not a same-type floor. The correct same-type different-prompt floor is 0.17 (generation) / 0.55 (prefill); the all-pairs floor is 0.08 / 0.41. The sentence's direction survives and strengthens: 0.70 between checkpoints on the same prompt vs 0.17 between different prompts of the same type. |
| M9 | CD 178-180 (Discussion) | "our prefill numbers agree with each of them, **on checkpoints they did not test** and under a second gating mechanism" (also J 340-341 "none of which the five 2026 papers used") | none of the five checkpoints tested by the five papers | HB 20 (Herbst Table 2 includes GLM-4.7-Flash: 65 / 4 / 1, 47 layers), HB 25 ("GLM-4.7-Flash appears only in the probing comparison"); RW 15 | MISMATCH: Herbst et al. did test GLM-4.7-Flash (official) in their probing comparison. The claim holds for Qwen3.5-35B-A3B (both checkpoints), Qwen3.5-122B-A10B and the GLM HauhauCS fine-tune. Say "on four checkpoints they did not test and, for GLM-4.7-Flash, on a routing statistic they did not compute". |
| M10 | CD 270-271 (references) | Shorthill (2026). Read Routing Entropy at a Fixed Position: A Cross-Model Study of the Prefix Effect in Mixture-of-Experts **Language Models**. Zenodo 10.5281/zenodo.22151499 | title suffix "Language Models" | MEM 10-11 ("...Prefix Effect in Mixture-of-Experts Routing"); `cc-lens/posconf/hf_bundle/README.md` 17 (same); MEM 23 (record 22151499, v2.0, published 2026-08-28) | MISMATCH (title): the published title ends "...in Mixture-of-Experts Routing". DOI 10.5281/zenodo.22151499 is correct. |
| M11 | JD 43 (methods) | "the Hugging Face tokenizer counts **392** tokens for the packed prompt where llama.cpp counted 446" | 392 for "the packed prompt" | NULLS ("chunk A: tokenizer count 392 vs capture 446; chunk B: 391; chunk C: 391"); CREW 45 (392) | MISMATCH (minor): 392 is chunk A only; B and C tokenize to 391. Say "391-392". CD 242 has the same "392". |

### UNSUPPORTED

| # | Draft, line | Sentence (abridged) | Value in draft | Source searched | Verdict |
|---|---|---|---|---|---|
| U1 | CD 263-264 (references) | Herbst, Wermter, Lee (2026) ... arXiv:2604.02178 **(ICML 2026)** | venue ICML 2026 | HB 10-11 ("the task brief says ICML 2026; the HTML carries an 'Impact Statement' (ICML format) but I did not see the venue printed in the text itself"); RW 15 lists "ICML 2026" from the brief | UNSUPPORTED in the ledger: the venue was never confirmed from the paper or an acceptance record. Verify at the ICML 2026 accepted-papers list before citing the venue. |
| U2 | CD 274-275 (references) | "Keep from v1.1: ... **Belrose et al. (tuned lens**, if the geometry paragraph cites it)" | Belrose was in v1.1 | TEX 450-451 (`\bibliography{refs}`), `refs.bib` and `main.tex`: 0 occurrences of "belrose" | UNSUPPORTED: v1.1 never cited Belrose et al.; it cannot be "kept". If the geometry paragraph needs it, it is a new reference to verify. |
| U3 | JD 51 (provenance) | "no 122B tensors survive" | none | PROV 37 (domain runs: "No router tensors survive"); PROV 38-41 (per-token npz survive for the 122B experience probe, single-prompt hum and baseline follow-ups); C 36-42 | UNSUPPORTED as a blanket statement: true for the 122B domain probe (the only 122B run the paper uses); per-token layer-averaged npz survive for three non-domain 122B runs. Say "no 122B domain-probe tensors survive". |
| U4 | CD 80-84 (Results 3.1) | GLM control: pooled prefill top-4 {10, 20, 28, 51} on both checkpoints (overlap 1.0); {16, 20, 28, 51} on the two secondary batteries; expert 10 rank 2 on the register battery | sets and rank | J 372-379 only (Claude's 2026-08-29 13:05 check "from compact/register_run, 140 cells"); no script archived; D_glm reports per-set winners (D 460-466: E28 / E51 / E16, E20 prefill leaders) and register-battery prefill defaults "E28, E51, E10, E16, E20" (D 143) but not these pooled top-4 sets | UNSUPPORTED by any archived script or agent report; the journal entry is the sole source. Consistent with D 143 (same five experts). Recommend archiving the 140-cell recompute in `audit_20260828/` before publication. |
| U5 | CD 161; JD none | "the movement concentrates in layers 0 to 2" | L0-2 | J 394 ("Per-layer generation agreement lowest at L0-2 (0.46-0.60), highest mid-stack (0.76-0.82)"); RECOMP (top-8 gentrim per-layer Jaccard, same prompt, official vs HauhauCS, mean over 100 prompts: L0 0.56, L1 0.60, L2 0.53; minimum at L2; maximum 0.76 at L19) | Now VERIFIED by RECOMP (listed here because no archived script existed; the journal's range 0.46-0.60 differs slightly from the recompute's 0.53-0.60, probably a trimmed/untrimmed or NaN-handling difference; the ranking L0-2 lowest holds). Archive the script. |

### VERIFIED

| # | Draft, line | Sentence (abridged) | Value | Source and line | Note |
|---|---|---|---|---|---|
| V1 | JD 9, 23; CD 252-254 | five checkpoints across Qwen3.5 and GLM-4.7-Flash families; two families, two gating mechanisms | 5 / 2 / 2 | J 337-340 | 35B official, 35B HauhauCS, 122B HauhauCS, GLM official, GLM HauhauCS |
| V2 | JD 9 | HauhauCS fine-tune of Qwen3.5-35B-A3B-Instruct; official = instruction-tuned | | PROV 8-10; J 289-291, 302-303 | |
| V3 | JD 11, 27, 33; CD 256 | generation identifies the subject at 95-96% | 95-96% | CREW 18-19 (23/24 = 0.96; 36/38 = 0.95); NULLS | |
| V4 | JD 11, 27; CD 129-130 | prefill 15-17% | 15-17% | CREW 58-59; NULLS (7/40 = 0.17; 6/40 = 0.15) | |
| V5 | JD 11, 27; CD 124-125, 256 | 6% by chance (shuffled labels) | 6% | CREW 23 (0.063, first pass); NULLS (seed 11, 200 draws: 0.062); J 330-332 | |
| V6 | JD 11, 29; CD 153-155 | official checkpoint: generation pairs by subject 11 of 12, prefill 4 of 12 | 11/12, 4/12 | SMOKE 26, 41; J 104-107; F2 F11 | |
| V7 | JD 19 | Standing Committee measured at the last prompt token of MMLU questions | | SC 53, 63, 71 | |
| V8 | JD 19 | Herbst et al.: fine-grained task specialists, e.g. closing brackets or function definitions | | HB 111-112 ("closing brackets in LaTeX", verbatim abstract); HB 32 (`is_function_def` is one of their probed concepts) | "function definitions" is a probe concept, not a headline example; acceptable |
| V9 | JD 19 | Ye et al.: individual experts polysemantic, paths across layers monosemantic | | YE 31 (verbatim) | |
| V10 | JD 19 | Do et al.: domain experts from question-token routing; upweighting improves domain accuracy across ten models | 10 models | DO 16 ("Ten models, from Table 7"), DO 84, 96 | |
| V11 | JD 21; CD 180-181 | Wang, Hayou, Nalisnick: converge in prefill, diverge in generation; prompt-level routing does not predict rollout routing; complete trajectories needed | | MY 77; J 10-14 (verbatim quotes); RW 17-20 | |
| V12 | JD 27; CD 70; TEX 100 | prefill: one expert (224) wins 18 of 20 subjects | 18/20 | A 68; TEX 178 | |
| V13 | JD 29; CD 137 | same subject at different positions beats different subjects at same position in 9 of 10 | 9/10 | NULLS #5; CREW 52-53 | one tie at zero (history) |
| V14 | JD 29; CD 140 | adjacent-candidate identification 83% | 139/168 = 0.83 | NULLS #6; CREW 54 | |
| V15 | JD 29; CD 157-158 | legal and medical experts survive dialect and checkpoint change | | CREW 84 (legal 122/109/85; medical 206/247) | |
| V16 | JD 31; CD 165-167 | long coherent technical prompt: prefill and generation share the set | | CREW 99-107 (Jaccard 0.6-0.78) | |
| V17 | JD 37 | Qwen3.5-35B-A3B: 256 routed experts, top-8, 40 MoE layers | 256 / 8 / 40 | PROV 12; A 19 | |
| V18 | JD 37 | official checkpoint Q8_0, ggml-org | | PROV 23; CREW 70; E 664 | |
| V19 | JD 37 | HauhauCS fine-tune Q8_0 | | PROV 10 | |
| V20 | JD 37 | Qwen3.5-122B-A10B HauhauCS Q8_K_P; 256 experts, top-8, 48 MoE layers | | PROV 29-31; C 4-5 | |
| V21 | JD 37, 41; CD 74-75 | GLM-4.7-Flash: 4 of 64 experts + 1 shared, sigmoid gating, per-layer correction bias | 4 / 64 / 1 | PROV 45-46; D 10-24; RM 56-57 | |
| V22 | JD 39 | greedy: temperature 0, top-k 1, fixed seed | seed 42 | PROV 4-5; A 23-24; B 238 | |
| V23 | JD 39 | prompt and generation blocks never combined; generation cut at first end-of-turn token | | RM 62-64 | |
| V24 | JD 39 | generation captured for both 35B checkpoints; 122B generation exploratory (trimming failed) | | PROV 16, 23, 37; C 168-177 | |
| V25 | JD 39 | GLM captures prefill only, BF16 weights, Transformers hook | | PROV 52-53, 58; D 38-53 | |
| V26 | JD 41 | softmax over 256, top-8 kept, rescaled to sum to 1; W, S, Q; W = S.Q | | RM 55, 58-60; TEX 214-224 | |
| V27 | JD 41 | layers pooled by unweighted mean over per-layer W vectors | | RM 78-79; A 39-41 | |
| V28 | JD 41 | expert-index set = top 8 by layer-pooled W; Jaccard | | CREW 6 | |
| V29 | JD 41; CD 123 | exact expected Jaccard of two random 8-of-256 sets 0.017; ratio approximation k/(2E-k) = 0.016 | 0.017 / 0.016 | RECOMP (hypergeometric exact 0.0169; 8/504 = 0.0159); J 317-318; NULLS ("analytic 0.0159; simulated 0.0171") | the nulls script's "analytic" is the ratio approximation; CREW 50 quotes 0.016 |
| V30 | JD 41 | GLM: top 4 by sigmoid + bias, weights = plain sigmoid rescaled; 64 experts, sets of 4 | | RM 56-57; D 15-22 | |
| V31 | JD 43 | independent probe: 20 subjects, 3 questions each, one per prompt, up to 2,056 generated tokens | 20 x 3; 2056 | PROV 16; A 24-25; TEX 194-198, 211 | |
| V32 | JD 43 | three packed prompts A (explanations), B (biographies), C (synthesis) | | PROV 17 (mechanism / historical figure / synthesis); B 237 | |
| V33 | JD 43; CD 99; TEX 251-252 | each padded to 446 prompt tokens, up to 2,048 generated | 446 / 2048 | PROV 17 (padding 0/73/44); B 237-238 | |
| V34 | JD 43; CD 20, 39 | 12 segments in A, 15 in B, 19 in C; law skipped in C; A and B stopped early | 12 / 15 / 19 | CREW 4, 39; NULLS `maps` | B_controls used 13/16/19 with a different minimum-length rule (B 290); the paper's 12/15/19 is the CREW/NULLS segmentation |
| V35 | JD 43 | only C reached an end-of-turn token before the cap, at token 1,812 | 1812 | J 70 ("C trims at 1812 of 1838, A and B hit the cap") | |
| V36 | JD 43; CD 242-243 | prefill question spans rescaled; off by one or two positions; generation spans exact | | CREW 45-48 | |
| V37 | JD 43; CD 228 | official-checkpoint design: 12 prompts; finance two pairs (4), medical / physical / professional / legal one pair each (8); 256 generated tokens | 12 / 4+8 / 256 | J 96-99; SMOKE 1-12 | two professional cells end at 212 / 230 (J 99; SMOKE 9-10) |
| V38 | JD 43; CD 165 | 30-prompt set, one long technical paragraph, three registers, official checkpoint | 30 / 3 | PROV 21; CREW 99; E 693-695 | |
| V39 | JD 45; CD 38 | A<->P 12 + 12 = 24; C<->P 19 + 19 = 38 | 24 / 38 | NULLS per-pair table; CREW 18-19 | |
| V40 | JD 45; CD 20, 36 | 168 bidirectional generation queries | 168 | NULLS ("total generation queries 168"); CREW 17 | |
| V41 | JD 45; CD 36, 131 | prefill: 40 per pair, six pairs, 240 | 40 / 240 | NULLS #7; CREW 60 | |
| V42 | JD 47; CD 59-60 | shuffle null 200 draws, NumPy default_rng(11), generation | 200 / seed 11 | nulls script L99-108; J 330-331 | |
| V43 | JD 47; CD 59-61 | shuffle null 100 draws, seed 7, prefill | 100 / seed 7 | nulls script L84-93; J 332 | |
| V44 | JD 47; CD 141 | adjacent-segment null: previous / same / next subject | | NULLS #6; nulls script L75-76 | |
| V45 | JD 47 | official-checkpoint replication: within vs between Jaccard, then nearest neighbour among the other 11 | 11 | SMOKE 14-41; J 104-107 | |
| V46 | JD 47; CD 101 | set-size sensitivity top-8 / top-15 / top-30, same conclusion | | J 72-79; F2 §4 | |
| V47 | JD 49; CD 119-155 | two-sided 95% Wilson intervals | | RECOMP (all nine reproduce, see V95-V103) | J 326 (block draft) said "no CIs"; superseded by CD |
| V48 | JD 51; J 9 | curated analysis repository frozen at commit f06cd3e | f06cd3e | J 9 | |
| V49 | JD 51 | 35B fine-tune SHA-256 f3235db7... | | PROV 10; A 18 | full hash in the 04-15 run_metadata.json |
| V50 | JD 51 | 122B capture binary SHA-256 c3e205b3... | | PROV 31-32; C 19-20 | |
| V51 | JD 51 | independent probe retains per-subject, per-layer summaries | | A 37-38 (20 x 40 x 256) | |
| V52 | JD 51; CD 98, 229 | 122B trimmer no-op, about 45% spill | 45% | C 173-176 (mean 1868.3 generated, 1027.4 clean); F2 §4 | |
| V53 | JD 51; CD 72 | 122B generation-run prefill has last-layer artifact; routing-only capture clean; 13 of 20 to one expert (233) | 13/20; E233 | C 84-129, 132; F2 §4 | |
| V54 | JD 51; CD 230 | GLM has no generation capture for the domain prompts | | PROV 58; D 126 | |
| V55 | CD 8-11 | packed per-token file stores W already averaged over layers; probe per-layer 20 x 40 x 256 | | B 239; A 38 | |
| V56 | CD 14-15; CD 239 | about 14 distinct subject winners per layer at every one of 40 layers (prefill 1.6); stronger middle third; weakest first and last | 14 / 1.6 | A 94-101 (14.25, range 8-18; 1.6) | |
| V57 | CD 24-35 | per-pair table, 12 rows: shared 12/12/12/12/14/15/12/14/19/12/15/19; gen hits 7/7/11/6/5/9/7/6/18/12/11/18; prefill hits 7/14/4/6/6/3/13/10/4/3/1/2 | | NULLS per-pair table (identical) | sums: 168, 117, 73 check |
| V58 | CD 38 | excl. B 73/86; involving B 44/82 | | CREW 21-22; NULLS; row sums of V57 | |
| V59 | CD 41-44 | edge queries not excluded; 21 of 168 have two candidates; exact chance 0.354; excl. B 78/86 = 0.91 | 21 / 0.354 / 0.91 | NULLS | |
| V60 | CD 46-48 | probe W per answer, then unweighted mean over 3 answers and 40 layers (`mean_over_cells_and_layers`) | | `qwen35b/domain_probe_60prompt/scripts/analyze_domain_specialists.py` L214, 397-399; A 39-41 | |
| V61 | CD 53-55 | build 8493 = HauhauCS 60-prompt probe, both captures, 04-15 re-capture bit-identical | | PROV 16; A 148-152 (max dW 3.5e-18) | |
| V62 | CD 56-58 | commit 6658925 = GLM register generation runs (llama.cpp Q8) | | PROV 56; D 34-35 | |
| V63 | CD 60-61 | generation null mean 0.062, 95th 0.095, max 0.119; prefill null mean 0.050, 95th 0.067 | | NULLS; J 330-332 | |
| V64 | CD 70-71 | 35B HauhauCS: Herfindahl 0.815; pairwise top-8 overlap between subjects 0.51 | 0.815 / 0.51 | A 68, 127; TEX 178; RECOMP Herfindahl([18,1,1]) = 0.815 | |
| V65 | CD 72 | 122B routing-only: expert 233 wins 13 of 20 | | C 132; J 368 | |
| V66 | CD 73; CD 105 | official 35B, 50-pair run: expert 224 wins 6 of 8 prompt types | 6/8 | CREW 93; J 368 | |
| V67 | CD 73-74 | experts 95 and 224 together win 8 of 12 prompts (safety smoke) | 8/12 | SMOKE 27 (95 x5, 224 x3); J 368 | |
| V68 | CD 74-76 | GLM: expert 10 wins 12 of 20 (15 prompts each); five distinct winners; Herfindahl 0.41 | 12/20; 5; 0.41 | D 95-98; F2 F10 | full-length reading; T=16 gives 9/20, 7 distinct (D 99-101) |
| V69 | CD 78-80; CD 236 | GLM official and HauhauCS router matrices bit-identical | | PROV 47-48; D 199-201 | |
| V70 | CD 88-89 | Standing Committee on Qwen3-30B-A3B: cross-domain Jaccard 0.87, minimum 0.53 | 0.87 / 0.53 | SC 66-67, 77 | |
| V71 | CD 89-91 | 0.51 vs 0.87: 20 subjects vs nine pooled domains; all prompt tokens vs last; routed weight vs full softmax | | SC 159-162; RW 49-52 | |
| V72 | CD 95-96; TEX 179 | generation: 20 distinct winners, Herfindahl 0.05; overlap 0.51 to 0.03; 119 of 190 pairs share no expert | 20; 0.05; 0.03; 119/190 | A 68-69, 127-128; RECOMP Herfindahl = 0.05 | |
| V73 | CD 97 | expert 224 wins one subject (political science) by a 5.6% margin | 5.6% | A 71 (0.00894 vs 0.00843) | |
| V74 | CD 97-98; TEX 181 | 122B generation: 18 distinct winners | 18 | C 146 | |
| V75 | CD 99-101; TEX 312-314 | top-8 prefill/generation overlap 0.00, 0.23, 0.07 by W | | J 74-78; B 275 (0.0000 / 0.2308 / 0.0667) | |
| V76 | CD 101 | top-15: 0.07, 0.20, 0.11; top-30: 0.18, 0.28, 0.22 | | J 72-76 (0.071/0.200/0.111; 0.176/0.277/0.224) | |
| V77 | CD 101-102; TEX 317 | per-token entropy 0.958 vs 0.953 | | B 276 (0.957522 / 0.952677); F2 F1 | |
| V78 | CD 104-105 | official 50-pair: 100 prompts, 8 types; prefill 3 distinct (224 in 6 of 8), Herfindahl 0.59 | 3; 0.59 | CREW 92-93 (0.594); RECOMP Herfindahl([6,1,1]) = 0.594 | |
| V79 | CD 105-106 | generation 7 distinct, Herfindahl 0.16 | 7; 0.16 | CREW 94 (0.156); RECOMP = 0.156 | |
| V80 | CD 106 | pooled prefill and generation top-8 sets share no expert | Jaccard 0.000 | CREW 95 | |
| V81 | CD 106-107 | between-type overlap 0.42 prefill, 0.09 generation | | CREW 96 (0.421 / 0.093) | |
| V82 | CD 112 | two halves of one segment overlap 0.49 (n = 46) | 0.49 / 46 | CREW 9 | 46 = 12 + 15 + 19 |
| V83 | CD 114-115 | archaeology recruits 191, 80 and 135 in A, C and P | | CREW 26 (top-3: A 191,80,135; C 191,80,116; P 191,80,210); RECOMP top-8: A [191,80,135,...], C [191,80,116,135,...], P [191,80,210,135,...] | 135 is rank 3 / 4 / 4; true at top-8, not at top-3 |
| V84 | CD 115 | neuroscience 54 and 24 in A, B and P | | CREW 27 | |
| V85 | CD 115-116 | computer science 206, 207, 189 in A, C and P | | CREW 29; B 290 | |
| V86 | CD 116 | chemistry 130 leads in A, B and P | | CREW 27-28 | |
| V87 | CD 116; CD 146 | mathematics 100 in A, B and P | | CREW 29, 35 | |
| V88 | CD 122-123 | matched-subject Jaccard 0.39 (A to P), 0.38 (C to P); mismatched 0.03 | | CREW 18-19 (0.387 / 0.379); NULLS (mism 0.029 / 0.032) | |
| V89 | CD 130 | prefill A to P 7 of 40 (0.17); C to P 6 of 40 (0.15) | | NULLS #7; CREW 58-59 | |
| V90 | CD 131 | all six pairs 73 of 240 (0.30) | | NULLS; CREW 60 | |
| V91 | CD 131-132 | A to C prefill 27 of 40 (0.68) | | NULLS; CREW 61 | bidirectional (14/20 + 13/20) |
| V92 | CD 136 | same position different subject 0.020; same subject different position 0.194 | | NULLS #5; CREW 52-53 | |
| V93 | CD 141 | adjacent subjects overlap 0.047 | | NULLS #6; CREW 55 | contradicts CD 113 (see M6) |
| V94 | CD 145 | pairs involving B 44 of 82 (0.54) vs 0.85 | | CREW 21-22 | |
| V95 | CD 119-120 | 23/24: 0.96, CI 0.80 to 0.99 | | RECOMP Wilson (0.798, 0.993) | |
| V96 | CD 120 | 36/38: 0.95, CI 0.83 to 0.99 | | RECOMP (0.827, 0.985) | |
| V97 | CD 121 | 117/168: 0.70, CI 0.62 to 0.76 | | RECOMP (0.623, 0.761) | |
| V98 | CD 122 | 73/86: 0.85, CI 0.76 to 0.91 | | RECOMP (0.758, 0.909) | |
| V99 | CD 130 | 7/40: 0.17, CI 0.09 to 0.32 | | RECOMP (0.087, 0.319) | |
| V100 | CD 130 | 6/40: 0.15, CI 0.07 to 0.29 | | RECOMP (0.071, 0.291) | |
| V101 | CD 140 | 139/168: 0.83, CI 0.76 to 0.88 | | RECOMP (0.763, 0.877) | |
| V102 | CD 145 | 44/82: 0.54, CI 0.43 to 0.64 | | RECOMP (0.429, 0.640) | |
| V103 | CD 154 | 11/12: 0.92, CI 0.65 to 0.99; chance 1 in 11 | | RECOMP (0.646, 0.985); SMOKE 41 | |
| V104 | CD 146-147 | subject expert persists (mathematics 100, neuroscience 54/24, chemistry 130) while rest of set follows register | | CREW 35-36 | |
| V105 | CD 147-149 | HVAC: technical vs first-person experiential voice; prefill top-15 overlap 0.50, generation 0.00; n = 6 per register | 0.50 / 0.00 / 6 | E 762 (L1-L3 prefill 0.500, generation 0.000); E 690, 844-845 (n = 6) | |
| V106 | CD 149-151 | dialect: AAVE vs academic English generation overlap 0.64 to 0.71, floor 0.28 for unrelated prompts of same type | | E 776-777 (0.712 base / 0.641 HauhauCS; floor 0.276 / 0.278); F2 F4 | top-12 sets, 3-vs-3 resampled, 300 draws; the draft does not state the set size |
| V107 | CD 155-156 | safety smoke: generation within 0.46 vs between 0.035; prefill 4 of 12, 0.52 vs 0.47 | | SMOKE 25-26, 40-41 (0.515 / 0.473; 0.457 / 0.035); J 104-107 | |
| V108 | CD 156-157 | official -> HauhauCS identifies the same prompt type in 7 of 8 types in both dialects | 7/8, 7/8 | CREW 80-81 | |
| V109 | CD 157-158 | legal (122, 109, 85), medical (206, 247), identity (103, 139) lead on both checkpoints and both dialects | | CREW 84; F2 F11 | |
| V110 | CD 160; CD 235 | generation overlap 0.70 between checkpoints on identical prompts | 0.70 | J 390 (0.702, min 0.23); RECOMP 0.702 (min 0.23) | the comparison floor is M8 |
| V111 | CD 166-167 | l1l3: overlap 0.60 to 0.78; leaders 166, 151, 41; entropy 0.956 vs 0.958 | | CREW 100-104 (0.600 / 0.778; 0.9557 / 0.9583) | |
| V112 | CD 167-168 | medical triage routes to 247, 47 in prefill; overlap with generation 0.23 to 0.37 | | CREW 97-98 (0.231 / 0.333 pooled; 0.345 / 0.367 per cell); E 725-726 | |
| V113 | CD 184-185 | Wang et al.: "is not predictable from the prompt alone" | quote | MY 52 (verbatim) | |
| V114 | CD 193-194 | biasing four philosophy leaders off: 15 of 20 winners in place; per-subject overlap 0.79; answers on topic, same length | 15/20; 0.79 | B 306-307; F2 F7 | |
| V115 | CD 198-199 | register change moves generation set as much as subject; dialect barely moves it | | E 765-767, 778-783; F2 F4 | |
| V116 | CD 205-206 | Wang et al.: routing is a linear readout of hidden-state geometry | | MY 56, 100 | |
| V117 | CD 208-209 | Wang et al. attribute prefill collapse to reasoning preambles | | MY 77, 82, 95 | |
| V118 | CD 209-210 | flat default appears on a non-reasoning fine-tune and on the official checkpoint with no chain of thought | | A 24 (no-think); CREW 92 (run_base_nothink) | |
| V119 | CD 210-211 | their rollout curves pool the prompt with the rollout; ours exclude it | | MY 46; RW 120-122 | |
| V120 | CD 211-212 | divergence present in the first 64 generated tokens | 64 | B 283-287; F2 F2 | |
| V121 | CD 218-220 | only four winners survive dropping one prompt: philosophy 114 (49%), psychology 146 (37%), mathematics 100 (23%), linguistics 103 (13%) | | A 82-83 (49 / 37 / 22.7 / 13.1) | |
| V122 | CD 220-221 | worst-case bound allows 16 of 20 to change | 16/20 | A 83; F2 F3 | |
| V123 | CD 221 | Herfindahl 0.815 against 0.05 | | A 68, 181 | |
| V124 | CD 222-223 | per-prompt tensors for the probe did not survive; leave-one-out impossible | | A 35-36; J 82-85 | |
| V125 | CD 227-228 | official checkpoint designs: 100 prompts across eight types; 12 prompts across five subjects | | CREW 92; J 97-98 | |
| V126 | CD 236 | Qwen router matrices never compared directly | | J 398-399 | |
| V127 | CD 238-239 | expert 114 mass sits on distinct per-layer modules | | B 314 (layers 8, 14, 20, 26); A 116-124 | |
| V128 | CD 261-262 | Wang, Y., Xu, Y., Shen, N., Su, J., Huang, J., Zhu, Z. (2026), arXiv:2601.03425 | | SC 10-13 (6 Jan 2026) | |
| V129 | CD 263-264 | Herbst, J., Wermter, S., Lee, J. H. (2026), arXiv:2604.02178 | | HB 8-10 (v2 15 May 2026) | venue: see U1 |
| V130 | CD 265-266 | Ye, C., Yuan, B., Sharkey, L. (2026), arXiv:2604.17837, ICLR 2026 Re-Align workshop | | YE 7 ("To appear at the ICLR 2026 Workshop on Representational Alignment (Re-Align)") | |
| V131 | CD 267 | Do, G., Le, H., Tran, T. (2026), arXiv:2604.05267 | | DO 9-11 (7 Apr 2026) | |
| V132 | CD 268-269 | Wang, X., Hayou, S., Nalisnick, E. (2026), arXiv:2604.09780 | | MY 7 (10 Apr 2026) | |
| V133 | CD 270-271 | Zenodo 10.5281/zenodo.22151499 (position confound) | | MEM 23; J 223 | title: see M10 |
| V134 | CD 272-273 | Shorthill (2026). In-Context Availability Reorganizes Expert Routing. Zenodo 10.5281/zenodo.22088887 | | ZEN 3, 10 (record 22088887 = v1.1 of concept 21892999); `job-search/PAPER_SAMPLES_20260818.md` 10 (full title "...: A Measurement Confound for Multi-Turn Evaluations of MoE Language Models") | subtitle omitted in the draft |
| V135 | CD 274-275 | keep from v1.1: Jiang et al. 2024 (Mixtral), Qwen3.5 and GLM-4.7-Flash model cards, llama.cpp | | TEX 80, 130 (jiang2024mixtral) | Belrose: see U2 |
| V136 | CD 278-279 | Table 1 rows in J 2026-08-29 13:05 entry | | J 365-370 | |
| V137 | CD 282 | Figure 2 from the 3-chunk per-token npz, not yet drawn | | J 424 | |
| V138 | JD 45 | ties count as misses; strict argmax | | CREW 13 | |
| V139 | JD 41 | expert index e at one layer and e at another are different modules | | B 314, 355; J 315-317 | |
| V140 | JD 43 | B changes task and register, same subject labels | | CREW 34-36; B 237 | |
| V141 | JD 25, 33 | register: subject survives when register controlled; larger register changes account for some failures | | CREW 33-36; E F4 | |
| V142 | CD 3 | sources listed for every number | | this audit | see M6-M8 for the three sentences whose source labels do not match |

## 2. Contradictions inside a draft and between the two drafts

Between JD and CD:
1. **Adjacent-candidate chance.** JD 29 says 83% "against 33% by chance"; CD 41-44 and CD 140 give the exact chance 0.354 (21 of 168 queries at a segment edge). JD's intro number is stale.
2. **Build 8493.** JD 51 assigns it to the official-checkpoint domain probe; CD 53-55 (and PROV 16) assign it to the HauhauCS 60-prompt probe. CD is right.
3. **Commit 6658925.** JD 51 assigns it to the GLM prefill captures; CD 56-58 (and PROV 56) assign it to the GLM register generation runs. CD is right; JD 39 already states the prefill captures used a Transformers hook, so JD 51 contradicts JD 39 as well.
4. **Layer-resolved identification.** JD 41 leaves a placeholder for a layer-resolved identification result to be described "as a robustness check"; CD 8-17 states it is not computable for the packed designs and offers the per-layer winner count on the probe instead. JD's framing ("insert the layer-resolved identification result") must not survive as written.
5. **Confidence intervals.** JD 49 says accuracies are reported with Wilson intervals, which CD supplies; the journal's block draft (J 326) said "no CIs". Consistent between the two drafts; the journal note is superseded.
6. **Reordering.** Both drafts describe C's subject order as a design manipulation (JD 11, 23, 25, 43, 47; CD 135); the prompts are identical in order and the model regrouped its answer (M1). Both drafts need the same fix.

Inside CD:
7. CD 113 gives "adjacent" overlap as 0.053 / 0.072 / 0.046; CD 141 gives adjacent overlap as 0.047. The first triple is the all-pairs between-subject mean (M6).
8. CD 123 states chance overlap 0.017 (exact); CREW 50 and the nulls script label 0.016 "analytic". JD 41 reconciles them (exact vs ratio approximation); CD should use the same wording or cite JD's Methods sentence.
9. CD 160 "0.43 between prompts of the same type" is inconsistent with CD 106-107 "between-type overlap ... 0.09 in generation" and with the recomputed same-type floor of 0.17 (M8); a reader will notice that a same-type floor of 0.43 cannot sit above a between-type value of 0.09 by that much on the same run without explanation.

Inside JD:
10. JD 39 (GLM prefill = Transformers hook, BF16) vs JD 51 (GLM prefill = llama.cpp commit 6658925); see item 3.
11. JD 43 "392 tokens" for "the packed prompt" vs the three chunks tokenizing to 392 / 391 / 391 (M11).

Ledger-level notes a reviewer could raise (not draft errors, but worth a sentence):
- CREW 23 reports the generation shuffle null as 0.063 (unseeded first pass); the archived seeded run gives 0.062. CD uses 0.062; fine, but the CREW file still says 0.063.
- B_controls (B 290) segmented the packed answers as 13 / 16 / 19 with a 20-token minimum; CREW and the nulls script use 12 / 15 / 19. The paper uses the latter; the two agree on every headline number but the within-segment floor differs (0.49 vs 0.44-0.55).
- J 340-341 ("none of which the five 2026 papers used") carries the same error as M9 (Herbst et al. include GLM-4.7-Flash).
- The GLM register-battery top-4 sets (U4) and the official-vs-HauhauCS shift table (J 387-394; M8, U5) exist only as journal entries; neither script is archived under `audit_20260828/`.
