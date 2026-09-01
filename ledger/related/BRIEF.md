# Brief: read one related paper against our result

You are reading ONE arXiv paper for a related-work review. Do NOT spawn sub-agents. Fetch the arXiv
HTML version (https://arxiv.org/html/<id>) with WebFetch or curl; if HTML is missing use the PDF via
`curl -sL https://arxiv.org/pdf/<id> -o /tmp/p.pdf && pdftotext /tmp/p.pdf -` (pdftotext is at
/opt/homebrew/bin). Quote verbatim wherever you make a claim about what the paper says; mark anything
paraphrased. Do not invent section numbers or numbers you did not see.

## Our result (what you are comparing against)

Paper: "Generation-Time Routing Reveals Expert Specialization That Prefill Measurements Miss"
(Shorthill, Zenodo 22089026). Model: Qwen3.5-35B-A3B (256 experts, top-8, 40 MoE layers; a HauhauCS
fine-tune, replicated on the base checkpoint), plus Qwen3.5-122B-A10B and GLM-4.7-Flash (prefill only).
Method: llama.cpp router-logit capture, prefill block (prompt tokens) and generation block (produced
tokens) kept separate, never pooled. Expert crew = top-8 by mean routed weight over a token set.
Findings:
1. Prefill: one flat default expert (E224 on the 60-prompt probe) wins 18 of 20 subjects; pairwise
   top-8 overlap between subjects 0.51. Generation: 20 distinct winners, overlap 0.03.
2. Same subject, same crew: inside one packed answer that walks through 20 subjects, the crew turns
   over at every subject boundary (0.05) and stays within a subject (0.45-0.55). The crew measured in
   one prompt design identifies the subject in an independent design 95-96% of the time (chance 6%);
   the same test on the prefill rows of the identical questions gives 15-17%. Nulls: chance overlap
   0.016; position-matched 0.02 vs 0.19; drift 0.83 vs 0.33; label shuffle 0.06.
3. Register moves generation crews as much as subject does (technical vs first-person on the same
   paragraph: generation overlap 0.00); dialect (AAVE vs academic English) is a near no-op.
4. Scoping: the effect is for short mixed prompts; a long coherent technical prompt followed by a
   technical answer shares the crew across prefill and generation (0.6-0.78).
5. Causal: biasing the four philosophy leaders off leaves 15/20 winners and the philosophy answers
   intact; they are removable, not load-bearing.
6. Fine-tunes never touched the routers (GLM bit-identical; Qwen prefill sets identical).

## What to report (write to the output path given in your prompt, markdown, <= 1500 words)

1. Citation line: authors, title, arXiv id, date, venue if any.
2. Models, sizes, expert counts, top-k, and the exact data/prompts used (verbatim where possible).
3. Their definition of "specialization" and of "expert usage" (verbatim). Pooled or per-token?
   Which layers?
4. PREFILL vs GENERATION: do they separate the two at all? Do they measure routing on generated
   tokens, or only on teacher-forced / prefilled text? Quote the sentence that settles it. If they
   measure only prefill, say so plainly; this is the single most important item.
5. Do they test whether routing is organized by SUBJECT / domain during generation? If they test
   domain at all, what were the domains and what did they find (numbers)?
6. Their headline claim in one verbatim sentence, and their main evidence for it in <= 5 lines.
7. Any causal test (ablation, steering, biasing experts) and its result.
8. Terms they use that we should reuse or avoid ("generalist", "polysemantic", "standing committee",
   "router collapse", etc.), verbatim with context.
9. Agreement / disagreement with our findings 1-6, item by item, one line each: AGREES / DISAGREES /
   SILENT, with the quote that supports the verdict.
10. What they did that we did not, and what we did that they did not.
11. Anything in the paper that could be turned against our claim by a reviewer, and the honest reply.
12. A one-paragraph related-work sentence set (3-4 sentences) we could adapt, plain prose, no
    em-dashes, no hype.
