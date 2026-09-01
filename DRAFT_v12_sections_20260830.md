# v1.2 draft: resolutions for the four {AUTHOR} holes, Methods corrections, and the remaining sections
Claude, 2026-08-30 11:20. For Jeffrey to rewrite. Every number traces to JOURNAL_v12.md,
ledger/CREW_REPRODUCIBILITY_20260828.md, ledger/FINDINGS_expert_specialization_v2_20260829.pdf, or the
per-pair run of audit_20260828/crew_reproducibility_nulls.py (2026-08-30).

## A. The four {AUTHOR} holes

**1. Layer-resolved identification (Expert-set construction).** Not computable from surviving tensors: the
packed-prompt per-token file stores W already averaged over layers, and no per-layer per-token router
tensors survive for that run (raw/ holds text and metadata only). What exists per layer is the
independent probe (20 x 40 x 256). Replace the placeholder with:
> Per-layer identification is not available for the packed designs, whose surviving per-token tensors are
> layer-averaged. On the independent probe, where per-layer profiles survive, the generation-time
> dispersion is present at every layer: about 14 distinct subject winners per layer across all 40 layers
> (prefill 1.6), slightly stronger in the middle third and weakest at the first and last layers. The
> layer-pooled profile therefore summarizes a signal that is flat across depth rather than concentrated
> in a few layers. (Appendix: per-layer winner counts.)

**2. Per-pair table (Cross-design identification).** Shared-subject counts and bidirectional queries; all
twelve ordered pairs sum to 168. Segments per design: A 12, B 15, C 19, P 20.

| pair (X->Y) | shared subjects | generation hits | prefill hits (of 20) |
|---|---|---|---|
| A->B | 12 | 7 | 7 |
| A->C | 12 | 7 | 14 |
| A->P | 12 | 11 | 4 |
| B->A | 12 | 6 | 6 |
| B->C | 14 | 5 | 6 |
| B->P | 15 | 9 | 3 |
| C->A | 12 | 7 | 13 |
| C->B | 14 | 6 | 10 |
| C->P | 19 | 18 | 4 |
| P->A | 12 | 12 | 3 |
| P->B | 15 | 11 | 1 |
| P->C | 19 | 18 | 2 |
| total | 168 | 117 | 73 (of 240) |

A<->P = 23/24; C<->P = 36/38; all pairs excluding B = 73/86; pairs involving B = 44/82. Exclusions: law
absent from C (skipped by the model); A capped at 12 and B at 15 segments by the generation budget.

**3. Adjacent-segment null, edge handling.** Edge queries were NOT excluded: a first or last segment has
two candidates (self plus one neighbour), the rest three. 21 of 168 queries are edge queries. Exact
chance = mean over queries of 1/|pool| = 0.354. Replace "chance 1/3" with "chance 0.354 (21 of 168
queries sit at a segment edge and have two candidates)". Observed 139/168 = 0.83; excluding B, 78/86 = 0.91.

**4. Independent-probe pooling (Statistical analysis).** The three answers were not concatenated. W was
computed per answer (mean over that answer's tokens, per layer), then the domain profile is the
unweighted mean over the three answers and the 40 layers (`mean_over_cells_and_layers`). Each answer
therefore carries equal weight regardless of its length; a 2,056-token answer and a 900-token answer
count the same.

## B. Two provenance corrections in the Methods draft
- "the official-checkpoint domain probe used llama.cpp build 8493 (1772701f)": build 8493 is the
  **HauhauCS** 60-prompt domain probe (both captures, 04-08 and the bit-identical 04-15 re-capture). The
  official checkpoint never ran the 60-prompt probe.
- "the GLM prefill captures use commit 6658925": commit 6658925 is the GLM **register generation** runs
  (llama.cpp Q8). The GLM domain prefill captures came from BF16 weights with a Transformers hook, as the
  first Methods paragraph says. Drop the commit from the prefill sentence or move it to the register runs.
- Shuffled-label null sentence: "200 times (default_rng(11)) for generation" and "100 times (seed 7)
  for prefill" are correct as written; observed generation null mean 0.062, 95th 0.095, max 0.119;
  prefill null mean 0.050, 95th 0.067.

## C. Results (draft)

\section{Results}

\subsection{Prefill expert sets reproduce across five checkpoints}

Pooled over the prompt, one expert leads prefill routing for most subjects on every checkpoint we
captured (Table 1). On the HauhauCS Qwen3.5-35B-A3B, expert 224 wins 18 of 20 subjects; the Herfindahl
index of the winner distribution is 0.815 and the pairwise top-8 overlap between subjects averages 0.51.
On Qwen3.5-122B-A10B, expert 233 wins 13 of 20 (routing-only capture). On the official instruction-tuned
35B checkpoint, expert 224 wins 6 of the 8 prompt types in the 50-pair run, and experts 95 and 224
together win 8 of the 12 prompts in the domain/register design. On GLM-4.7-Flash, with sigmoid gating,
64 experts and top-4 selection, expert 10 wins 12 of 20 subjects (15 prompts each), with five distinct
winners and Herfindahl 0.41.

The GLM fine-tune serves as a control on what this default is made of. Its register battery varies the
voice of the prompt with no subject axis, on both the official checkpoint and the HauhauCS fine-tune,
whose router matrices are bit-identical. The pooled prefill top-4 set on the main register battery is
{10, 20, 28, 51} on both checkpoints (overlap 1.0), and {16, 20, 28, 51} on the two secondary
batteries, again identical across checkpoints. Expert 10, the winner on the subject battery, sits at
rank 2 on the register battery. The same four or five experts carry prefill whether the prompts vary in
subject or in voice, and whichever one wins on a given battery is a near-tie inside that set. The
prefill default is a property of reading a prompt: subject-independent, fine-tune-independent, and
present under both gating mechanisms.

This is the pattern the Standing Committee analysis reports on Qwen3-30B-A3B (cross-domain Jaccard
0.87, minimum 0.53) [citation]. Our 0.51 is lower because it is computed over 20 subjects rather than
nine pooled domains, over all prompt tokens rather than the last, and with routed weight rather than
the full softmax; their minimum sits inside our range.

\subsection{Generation disperses the same experts by subject}

On the same 60 prompts, generation-time routing has 20 distinct subject winners on the 35B (Herfindahl
0.05), and pairwise top-8 overlap between subjects falls from 0.51 to 0.03; 119 of 190 subject pairs
share no expert. Expert 224 wins one subject (political science, by a 5.6% margin). The 122B
generation block, exploratory because its trimmer failed, shows 18 distinct winners. The token-balanced
control holds: with 446 prompt tokens and 2,048 generated tokens per packed prompt, the top-8 overlap
between the prefill and generation profiles is 0.00, 0.23 and 0.07 across the three prompts by routed
weight (0.07, 0.20 and 0.11 at top-15; 0.18, 0.28 and 0.22 at top-30), and per-token routing entropy is
unchanged (0.958 versus 0.953). The dispersion is a regime effect, not a token-count effect.

The official instruction-tuned checkpoint reproduces the shape. On its 100 prompts across eight types,
prefill has three distinct winners (224 in six of eight; Herfindahl 0.59) and generation seven
(Herfindahl 0.16); the pooled prefill and generation top-8 sets share no expert. Between-type overlap is
0.42 in prefill and 0.09 in generation.

\subsection{The same subject recruits the same expert set}

Inside a packed answer that walks through 20 subjects, the top-8 set changes at almost every subject
boundary. Overlap between the two halves of one subject's segment averages 0.49 (n = 46 segments; the
within-segment floor); overlap between adjacent subjects in the same answer is 0.053, 0.072 and 0.046 in
prompts A, B and C. The sets also return: archaeology recruits experts 191, 80 and 135 in A, in C and in
the independent probe; neuroscience 54 and 24 in A, B and P; computer science 206, 207 and 189 in A, C
and P; chemistry 130 leads in A, B and P; mathematics 100 in A, B and P.

Table 2 gives the identification test. A subject's generation-time set from the packed explanations (A)
identifies that subject among the 20 independently asked subjects in 23 of 24 queries (0.96; Wilson
95% CI 0.80 to 0.99); from the packed synthesis (C), 36 of 38 (0.95; 0.83 to 0.99). Over all 168
bidirectional queries across the four designs, 117 are correct (0.70; 0.62 to 0.76); excluding the
biography prompt, 73 of 86 (0.85; 0.76 to 0.91). The matched-subject Jaccard averages 0.39 (A to P) and
0.38 (C to P) against 0.03 for mismatched subjects; two random top-8 sets overlap at 0.017. Shuffling
the subject labels 200 times gives a mean identification of 0.062 (95th percentile 0.095, maximum
0.119).

\subsection{Controls}

\textbf{Wording.} The identical procedure on the prefill rows of the same questions identifies the
subject in 7 of 40 queries (A to P, 0.17; CI 0.09 to 0.32) and 6 of 40 (C to P, 0.15; 0.07 to 0.29);
over all six design pairs, 73 of 240 (0.30). The one prefill pair that identifies well is A to C, 27 of
40 (0.68), where the two packed prompts share their wording verbatim. Prefill sets follow the question's
tokens; generation sets follow the subject of the answer.

\textbf{Position.} Comparing A with C, whose subjects occur in a different order: different subjects at
the same position overlap at 0.020; the same subject at different positions overlaps at 0.194. The
subject wins 9 of 10 comparisons, with one tie at zero (history).

\textbf{Drift.} Restricting candidates to the previous, same and next subject in the target's order,
identification is 139 of 168 (0.83; CI 0.76 to 0.88) against an exact chance of 0.354; excluding B, 78
of 86 (0.91). Adjacent subjects overlap at 0.047. The sets change at boundaries rather than sliding
along the answer.

\textbf{Register.} The biography prompt (B) holds the same subject labels but changes the requested
task and voice. Pairs involving B identify at 44 of 82 (0.54; CI 0.43 to 0.64) against 0.85 for the
other pairs. The subject expert persists (mathematics 100, neuroscience 54 and 24, chemistry 130) while
the rest of the set follows the register. In the HVAC control, one fixed paragraph asked in a technical
voice and in a first-person experiential voice gives a prefill top-15 overlap of 0.50 and a generation
overlap of 0.00 (n = 6 per register). Dialect is close to a no-op: AAVE and academic-English versions of
matched prompts overlap at 0.64 to 0.71 in generation, above the 0.28 floor for unrelated prompts of the
same type.

\textbf{Checkpoint.} On the official instruction-tuned 35B, in the domain/register design, generation
sets pair prompts by subject in 11 of 12 nearest-neighbour queries (0.92; CI 0.65 to 0.99; chance 1 in
11) with within-subject overlap 0.46 against between-subject 0.035; prefill pairs 4 of 12 (0.52 against
0.47). Across the 50-pair run, a subject's generation set on the official checkpoint identifies the same
prompt type on the HauhauCS checkpoint in 7 of 8 types in both dialects, and the same legal (122, 109,
85), medical (206, 247) and identity (103, 139) experts lead on both checkpoints and both dialects. On
identical prompts the fine-tune moves the generation top-8 set less than a change of prompt does
(overlap 0.70 between checkpoints against 0.43 between prompts of the same type), and the movement
concentrates in layers 0 to 2.

\subsection{Scope}

On a long, coherent technical paragraph followed by a technical answer (30 prompts, official
checkpoint), prefill and generation share the set: overlap 0.60 to 0.78, the same leaders (166, 151,
41) on both sides, and per-token entropy 0.956 against 0.958. A narrow-topic prompt (medical triage)
already routes to its topic set in prefill (247, 47; overlap with generation 0.23 to 0.37). The
prefill/generation separation is strongest when a short question in one voice initiates a long answer
organized around its subject.

## D. Discussion (draft)

\section{Discussion}

The two routing regimes reconcile the prior findings without rejecting any of them. The Standing
Committee, the fine-grained task experts of Herbst et al., the polysemantic single experts of Ye et
al., and the question-token-scored experts of Do et al. are all measurements on tokens supplied as
input; our prefill numbers agree with each of them, on checkpoints they did not test and under a second
gating mechanism. Wang, Hayou and Nalisnick showed that prompt-level routing does not predict rollout
routing and left the organization of the divergence open; the identification test answers it: for short
mixed prompts, the rollout is organized by the local subject of the answer. Their two regimes are pairs
of prompts that converge or diverge over a rollout; ours are the prompt and the answer of a single
prompt. Their observation that which pair-regime applies "is not predictable from the prompt alone" is
what a subject-organized rollout looks like from the pair side: pairs that share a subject stay similar,
pairs that split subjects diverge.

The unit matters. Herbst et al. probe single experts and find operations; Ye et al. follow single tokens
across layers and find paths; we pool routed weight over a subject's span and find sets that recur.
These are three units, and the disagreement between "no domain specialists" and "domain specialists"
is largely a disagreement about which one is being counted. A subject's set may well be a coalition of
fine-grained operation experts that co-occur when the model writes about chemistry; the causal test
supports that reading. Biasing the four philosophy leaders off leaves 15 of 20 subject winners in place,
per-subject overlap with baseline at 0.79, and the philosophy answers on topic and of the same length.
The leaders are removable, so the paper's claim is about routing organization, not about experts as the
seat of subject knowledge.

Register is the second organizer. Holding subject fixed and changing the voice of the answer moves the
generation set as much as changing the subject does, while dialect barely moves it. The 60-prompt probe
holds register roughly constant, which is why its winners read as subject; the biography prompt is where
register was allowed to move, and its identification rate is the measure of how much of the set register
owns. A probe that crossed the two axes on the full subject list would separate them on one design
rather than across three.

The mechanism proposed by Wang, Hayou and Nalisnick, that routing is a linear readout of hidden-state
geometry, is consistent with everything here and offers the simplest account of it: generated tokens
about chemistry have chemistry-shaped hidden states, and the router reads them; prompt tokens in one
voice have voice-shaped hidden states, and the router reads those. Their attribution of prefill
collapse to reasoning preambles does not carry over: the flat default appears on a non-reasoning
fine-tune and on the official checkpoint with no chain of thought. Their rollout curves pool the prompt
with the rollout; ours exclude it, which is why their divergence appears gradual while ours is present in
the first 64 generated tokens.

## E. Limitations (draft)

\section{Limitations}

Three prompts per subject make the winner tables the weakest layer of the evidence. Only four
generation winners on the 35B lead by a margin that survives dropping one prompt (philosophy, expert 114,
49% ahead; psychology, 146, 37%; mathematics, 100, 23%; linguistics, 103, 13%), and a worst-case bound
allows 16 of 20 to change. The dispersion (Herfindahl 0.815 against 0.05) is far outside this noise; the
identity of most named experts is not. Per-prompt tensors for the probe did not survive, so a
leave-one-out is not possible; a 10-prompt-per-subject capture on the official checkpoint would close
this.

The primary packed-generation evidence is one model, the HauhauCS Qwen3.5-35B-A3B. The official
instruction-tuned checkpoint replicates the shape and the subject pairing on smaller designs (100
prompts across eight types; 12 prompts across five subjects). The 122B contributes a clean prefill
replication and an exploratory generation result whose block is about 45% post-answer spill. GLM
contributes the prefill half only; its generation captures have no subject axis. The cross-family
claim is therefore for prefill, and the generation claim is for one architecture on two checkpoints.

Our "official" checkpoint is the instruction-tuned release; the pretrained base was not captured. The
checkpoint comparison shows the organization persists across the uncensoring fine-tune, not that it
precedes instruction tuning. The fine-tune moves routing behaviour by less than a prompt change does and
leaves the GLM routers bit-identical; the Qwen router matrices were not compared directly.

Expert sets are layer-pooled by expert index, and expert 114 at one layer is a different module from
expert 114 at another. Per-layer profiles on the probe show the generation dispersion at every layer,
but per-layer identification on the packed designs is not available. Sets are defined by routed weight,
which measures selection rather than what the selected expert writes; a residual-write-norm variant is
a cheap follow-up. Prefill spans for the packed prompts were rescaled across a tokenizer mismatch (392
versus 446 tokens) and may be off by one or two positions; generation spans are exact.

Finally, the regime effect is scoped: on a long coherent technical prompt the prefill and generation
sets coincide. The claim is for short mixed prompts that initiate longer subject-organized answers.

## F. Conclusion (draft)

\section{Conclusion}

Measured on the tokens a model reads, expert routing converges on a small default set whatever the
subject, and we reproduce that result on five checkpoints across two families and two gating
mechanisms. Measured on the tokens the same model writes, the same routers organize by the local
subject of the answer: sets stay put within a subject, turn over at its boundary, and return for the
same subject under a different prompt design at 95 to 96% identification against 6% by chance, on both
the official and the fine-tuned checkpoint. Prior work measured the reading. Specialization is visible
in the writing.

## G. References to add (verified 2026-08-29; see ledger/related/RELATED_WORK_REVIEW_20260829.md)
- Wang, Y., Xu, Y., Shen, N., Su, J., Huang, J., Zhu, Z. (2026). The Illusion of Specialization:
  Unveiling the Domain-Invariant "Standing Committee" in Mixture-of-Experts Models. arXiv:2601.03425.
- Herbst, J., Wermter, S., Lee, J. H. (2026). The Expert Strikes Back: Interpreting Mixture-of-Experts
  Language Models at Expert Level. arXiv:2604.02178 (ICML 2026).
- Ye, C., Yuan, B., Sharkey, L. (2026). Polysemantic Experts, Monosemantic Paths: Routing as Control in
  MoEs. arXiv:2604.17837 (ICLR 2026 Re-Align workshop).
- Do, G., Le, H., Tran, T. (2026). Do Domain-specific Experts exist in MoE-based LLMs? arXiv:2604.05267.
- Wang, X., Hayou, S., Nalisnick, E. (2026). The Myth of Expert Specialization in MoEs: Why Routing
  Reflects Geometry, Not Necessarily Domain Expertise. arXiv:2604.09780.
- Shorthill, J. (2026). Read Routing Entropy at a Fixed Position: A Cross-Model Study of the Prefix
  Effect in Mixture-of-Experts Language Models. Zenodo, 10.5281/zenodo.22151499. (position confound)
- Shorthill, J. (2026). In-Context Availability Reorganizes Expert Routing. Zenodo,
  10.5281/zenodo.22088887. (copy regime, cited if the aftereffect/verbatim point is used)
- Keep from v1.1: Jiang et al. 2024 (Mixtral), Belrose et al. (tuned lens, if the geometry paragraph
  cites it), the Qwen3.5 and GLM-4.7-Flash model cards, llama.cpp.

## H. Tables the text refers to
Table 1: prefill replication, one row per checkpoint (in JOURNAL 2026-08-29 13:05 entry; add the
GLM-control row with the {10,20,28,51} set). Table 2: the per-pair identification table above (or its
collapsed form: A<->P, C<->P, excl. B, all, B-involving, shuffle null), with Wilson CIs. Figure 1 (kept
from v1.1): prefill vs generation winner distributions. Figure 2: within-answer overlap along the packed
answer with subject boundaries marked (from the 3-chunk per-token npz; not yet drawn).
