# Related-work read: "The Illusion of Specialization" (arXiv 2601.03425)

Source: arXiv HTML v1 (https://arxiv.org/html/2601.03425v1), fetched 2026-08-29. All quotes verbatim
from that HTML unless marked (paraphrase). Section numbers are the paper's own. Note: `curl` of
`arxiv.org/pdf/2601.03425` returned a different paper (2604.05267, Do et al.) on first try; the HTML
and abs page are the correct paper and were used throughout.

## 1. Citation

Yan Wang (The Fin AI), Yitao Xu (Georgia Tech), Nanhan Shen (Georgia Tech), Jinyan Su (Cornell),
Jimin Huang (The Fin AI), Zining Zhu (Stevens, corresponding). "The Illusion of Specialization:
Unveiling the Domain-Invariant 'Standing Committee' in Mixture-of-Experts Models." arXiv:2601.03425v1
[cs.LG], 6 Jan 2026. No venue stated (has an ACL-style Limitations / Ethics / Licenses block).

## 2. Models, sizes, experts, top-k, data

Table 3 (verbatim values): DeepSeek-V2-Lite, 64 experts, top-6, 2 shared, 16B; Qwen3-30B-A3B,
128 experts, top-8, 0 shared, 30B; OLMoE-1B-7B, 64 experts, top-8, 0 shared, 7B.
"All models are used in inference-only mode, and we extract routing weights from every MoE layer for
analysis."

Data: "We evaluate CommitteeAudit on the Massive Multitask Language Understanding (MMLU) benchmark
Hendrycks et al. (2021), which contains 57 multiple-choice subjects spanning science, humanities,
social science, and professional domains." The 57 subjects are collapsed to nine domains (Table 2):
STEM-Math, STEM-Physics, STEM-Chemistry, STEM-BioMed, CS-Eng, SocSci, Humanities, Lang-Ling, Biz-Law.
"All routing analyses in this paper, including task-specificity and Standing Committee extraction,
are conducted at the domain level." Appendix A: "We evaluate each model on the full MMLU benchmark
using two NVIDIA A100 (80 GB) GPUs ... The total computational cost of the routing analyses is about
40 GPU-hours, including forward passes and the collection of routing statistics." The exact prompt
template (few-shot or zero-shot, whether choices are included) is not given anywhere in the paper.
(Note: their Hendrycks 2021 reference is actually the MATH dataset paper, not the MMLU paper; a
citation slip on their side.)

## 3. Definitions of "specialization" and "expert usage"; pooled vs per-token; layers

Expert usage is the Expert Contribution Index: "For expert i at layer l, the ECI is the expected
routing weight: c^(l)_{i,tau} = E_{x in D_tau}[G^(l)(x)_i]" where "G^(l)(x) = softmax(z^(l)(x))",
i.e. the full softmax over all E experts, not the top-k gates: "We use the full routing distribution
(rather than discrete Top-k activations) because it preserves the complete preference structure over
experts." ECI is pooled over all samples in a domain, then experts are ranked per domain, with
"a penalty rank k+1 to experts that do not appear in the Top-k". Committee candidates are experts in
the domain top-k for a fraction "gamma > 0.8" of domains; the Standing Committee is the Pareto set
over (mean rank, rank variance). Specialization is defined operationally: "We hypothesize that
specialization is expressed through a structured distribution over a subset of experts, referred to
as a committee." Per-domain "task-specificity" is a silhouette score on cosine distance between
per-sample routing vectors (Eq. 5-7). Layers: all MoE layers (Appendix B Tables 6-8 give a committee
per layer for all 16 / 26 / 48 layers); Table 5 shows shallow / middle / deep snapshots.

## 4. PREFILL vs GENERATION (the settling item)

They do not separate the two, and they never measure routing on generated tokens. The settling
sentence (Sec. 3.2, Stage I): "For every sample x in D_tau and MoE layer l, we run the model and
record the full routing vector G^(l)(x) taken at the last token unless otherwise specified". That is
one routing vector per MMLU prompt, at the final prompt token, from a single forward pass. Supporting
statements: "All models are used in inference-only mode" (3.3.2); Appendix A budgets "forward passes
and the collection of routing statistics"; Limitations: "our study is observational and
inference-only." The word "generation" does not occur in the paper (grep, 0 hits); "decoding" does
not occur either. The one place they leave "last token" is the Figure 7 case study, which is also
prompt-token routing ("Which, What, Suppose, and question marks"; "the, a, and in"), i.e. tokens of
the MMLU question, not produced tokens.

So: everything in this paper is prefill. More narrowly, the committee statistics are the routing of
the LAST PROMPT TOKEN of a multiple-choice question, pooled by domain. Their "Standing Committee" is
therefore the prefill-side object. This is the direct analogue of our prefill default expert (E224
winning 18 of 20 subjects, pairwise overlap 0.51). Their Qwen3-30B-A3B (128 experts, 48 layers)
numbers for that object: cross-domain Jaccard of domain top-8 sets "Overall 0.8670" with "Min:
0.5300", Gini of ECI "0.9465", committee size 1-5 experts covering "up to 70.5%" (DeepSeek deep) and
67.0% (Qwen L33) of routing mass. The reading is consistent with ours: at the prompt side, a small
fixed set of experts dominates regardless of domain. Their overlap is higher than our 0.51 because
they aggregate 57 subjects into 9 domains, use full-softmax ECI rather than routed weight, and take
a single last token per prompt, which for MMLU is a highly stereotyped position (end of an "Answer:"
style prompt), so their measurement is even more prefill-shaped than ours.

## 5. Subject / domain organization during generation

Not tested. Domain is tested only on prefill: nine MMLU domains, cross-domain Jaccard (Table 4:
OLMoE 0.8735, DeepSeek-V2-Lite 0.8670, Qwen3-30B-A3B 0.8670 overall; min 0.7963 / 0.7103 / 0.5300)
and "Figure 4 ... similarity remains consistently high (often >= 0.85)". Their conclusion for
prefill: "the routing network converges to a shared backbone of experts that is largely invariant to
both input domain and layer position." Domain effects on generated tokens are absent by construction.

## 6. Headline claim and evidence

"Across three representative models and the MMLU benchmark, we uncover a domain invariant Standing
Committee. This is a compact coalition of routed experts that consistently captures the majority of
routing mass across domains, layers, and routing budgets, even when architectures already include
shared experts."
Evidence:
- Table 4 Jaccard 0.87 / 0.87 / 0.87 overall across nine domains; Gini > 0.88 in every model.
- Table 5 / Appendix B: per-layer committees of 1-5 experts, ECI coverage 15% to 70.5%.
- Figure 5-6 (OLMoE only): committee persists under top-k sweep; retention vs k=8 core "falls to 0.39
  at k=6, 0.17 at k=4, and below 0.3 once k is expanded to 12 or 16".
- Figure 7 qualitative: function words and question words map to committee experts; "domain-specific
  terminology rarely stabilizes".

## 7. Causal tests

None. "Second, our study is observational and inference-only. We do not directly intervene in
routing or measure causal effects of modifying committee members. Future work should incorporate
targeted ablations and routing perturbations."

## 8. Terms

- "Standing Committee": "a compact coalition of routed experts that consistently captures the
  majority of routing mass across domains, layers, and routing budgets". Reuse, with attribution,
  as the name for the prefill-side default set; our E224 is a one-member standing committee.
- "generalist core": "a Standing Committee ... acting as a generalist core hidden within the routed
  experts." Usable.
- "core-periphery organization": "committee members anchor logical and syntactic structures, while
  peripheral experts manage domain-specific knowledge." Usable; our generation crews are their
  periphery, measured where it does the work.
- "peripheral experts", "Peripheral specialization": "domain-specific terminology rarely stabilizes:
  chemical symbols, biomedical identifiers, and financial jargon are distributed across many experts
  depending on context." Note this is the opposite of our generation finding.
- "polysemous" (attributed to Lo et al.): "Experts often behave polysemously rather than strictly
  specializing". Avoid asserting for our data.
- "Representation Collapse" (Chi et al.) they explicitly distinguish: "Unlike representation
  collapse, where experts die out due to optimization failure, these shared experts are highly
  active and functionally competent but simply refuse to specialize." They do not use "router
  collapse".
- "Super Experts" (Su et al.): individual-level criticality; they contrast it with committees.
- "Illusion of Specialization" (title): avoid adopting; our claim is that the illusion is a
  measurement artifact of prefill.

## 9. Agreement with our findings 1-6

1. Prefill default expert, generation 20 distinct winners: AGREES on the prefill half, SILENT on
   generation. "the gating networks repeatedly allocate computation to a compact, stable subset of
   experts that serve as default processing routes, regardless of layer position or domain."
2. Crew turnover at subject boundaries inside one answer, 95-96% identification: SILENT (no
   generated tokens, no within-sequence analysis).
3. Register moves crews as much as subject: SILENT; closest is the Figure 7 note that function words
   and question words go to the committee, which is about prompt syntax not output register.
4. Long coherent technical prompt shares crew across prefill and generation: SILENT, but their
   design (single last-token vector per MMLU prompt) is the fully prefill-shaped limit.
5. Biasing leaders off leaves winners intact: SILENT; "We do not directly intervene in routing".
6. Fine-tunes never touched routers: SILENT (base/instruct checkpoints not compared; which
   checkpoint of Qwen3-30B-A3B was used is not stated).

Disagreement of emphasis rather than data: they read the prefill committee as evidence that
"specialization in Mixture of Experts models is far less pervasive than commonly believed" and that
"domain-specific terminology rarely stabilizes". Our generation data say the opposite for produced
tokens. Their evidence cannot decide this because it never looks at produced tokens.

## 10. What they did that we did not, and vice versa

They did: three architectures including one with shared experts; Gini / Lorenz concentration over
the full softmax; a Pareto (mean rank, rank variance) committee extraction with a formal threshold
(gamma > 0.8); a top-k sensitivity sweep (OLMoE); per-layer committee tables for every layer; a
silhouette-based task-specificity gate; token-level function-word case study.
We did: separate prefill from generation; per-subject rather than nine-domain granularity; overlap
nulls (chance, position-matched, drift, label shuffle); cross-design subject identification; register
and dialect manipulations; causal expert biasing; base vs fine-tune router comparison; a 256-expert
Qwen3.5 model plus 122B and GLM.

## 11. Reviewer ammunition and honest reply

- "Wang et al. show cross-domain Jaccard 0.87 on Qwen3-30B-A3B; your 0.51 prefill number and your
  generation number 0.03 look like a different regime." Reply: their 0.87 is nine pooled domains,
  full-softmax ECI, last prompt token of a stereotyped MMLU format; our 0.51 is 20 subjects, routed
  weight, all prompt tokens. Both are prefill and both find one dominant set. Their Qwen minimum
  (0.53) is in our range. Nothing they report bears on produced tokens.
- "Their Figure 7 says domain terminology does not stabilize on any expert, so subject crews are
  noise." Reply: that panel is prompt tokens in an MMLU question, marked only when a token hits a
  committee expert in at least three domains; it is a test of whether content words join the
  committee, not of whether content tokens have stable crews of their own. Our subject crews are
  measured on generated tokens with nulls.
- "Standing committees are per-layer and you pool over 40 layers." Reply: state the pooling
  explicitly and, if space allows, show one per-layer panel; their own Appendix B shows committees
  change membership by layer, which our per-layer capture already contains.
- "You cite a paper titled Illusion of Specialization as support." Reply: cite it as the strongest
  recent statement of the prefill-only result, and as an example of the measurement that misses
  generation.

## 12. Related-work sentences

Wang et al. (2026) audit routing at the group level on OLMoE, DeepSeek-V2-Lite and Qwen3-30B-A3B
over nine MMLU domains and find a small "Standing Committee" of experts that holds most of the
routing mass in every domain and at every layer, with cross-domain Jaccard near 0.87. Their routing
vector is taken at the last prompt token of each question in a single forward pass, so the result
describes prefill only. We reproduce this prefill picture on Qwen3.5-35B-A3B, where one expert wins
18 of 20 subjects, and then show that the same model routes generated tokens through subject-specific
crews that the prefill measurement never sees. The standing committee is real, but it is a property
of reading the prompt, not of writing the answer.
