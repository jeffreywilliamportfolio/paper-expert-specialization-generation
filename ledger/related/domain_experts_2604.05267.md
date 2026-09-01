# Related-work read: arXiv 2604.05267, "Do Domain-specific Experts exist in MoE-based LLMs?"

Source: arXiv PDF v1 (HTML version not used; PDF fetched and converted with pdftotext). Code checked at
https://github.com/giangdip2410/Domain-specific-Experts (files `analysis_specialize/token_expert_analysis.py`,
`expert_importance.py`). All quotes are verbatim from the PDF text or the repo unless marked [paraphrase].
Section numbers are those printed in the PDF.

## 1. Citation

Giang Do, Hung Le, Truyen Tran (Applied Artificial Intelligence Initiative, Deakin University). "Do
Domain-specific Experts exist in MoE-based LLMs?" arXiv:2604.05267v1 [cs.CL], 7 Apr 2026. No venue stated
(ACL-style template; "Limitations" and "Ethics Statement" sections suggest an ACL-family submission).

## 2. Models, data, prompts

Ten models, from Table 7 (Params B / Active B / Experts N / Top-K / Layers / HF id):

| Model | Params | Active | Experts | Top-K | Layers | HF id |
|---|---|---|---|---|---|---|
| PhiMoE-Tiny | 3.8 | 1.1 | 16 | 2 | 32 | microsoft/Phi-tiny-MoE-instruct |
| OLMoE | 7.0 | 1.0 | 64 | 8 | 16 | allenai/OLMoE-1B-7B-0924 |
| Qwen1.5-MoE | 14.3 | 2.7 | 60 | 4 | 24 | Qwen/Qwen1.5-MoE-A2.7B |
| DeepSeek-MoE | 16.0 | 2.8 | 64 | 6 | 28 | deepseek-ai/deepseek-moe-16b-base |
| GPT-OSS-20B | 20.0 | 3.6 | 32 | 4 | 24 | openai/gpt-oss-20b |
| ERNIE-4.5 | 21.0 | 3.0 | 64 | 6 | 28 | baidu/ERNIE-4.5-21B-A3B-Thinking |
| Qwen3MoE-Instruct | 30.0 | 3.0 | 128 | 8 | 48 | Qwen/Qwen3-30B-A3B-Instruct-2507 |
| Qwen3MoE-Think | 30.0 | 3.0 | 128 | 8 | 48 | Qwen/Qwen3-30B-A3B-Thinking-2507 |
| Qwen3-Next | 80.0 | 3.0 | 512 | 10 | 48 | Qwen/Qwen3-Next-80B-A3B-Thinking |
| GPT-OSS-120B | 120.0 | 5.1 | 128 | 4 | 36 | openai/gpt-oss-120b |

(Table 7 as printed says Qwen3-Next has Top-K 10 and 512 experts.) No Qwen3.5 model; Qwen3-30B-A3B is the
nearest relative to our 35B-A3B.

Data for the existence test (Sec. 4.2): "For each model, we sample 10% of mathematics questions from the
MMLU-Pro dataset." For DSMoE (Sec. 4.3): "we focus on four target domains (Math, Biology, Physics, and
Chemistry)" and "For each target domain, we use only question texts (without labels) from MMLU-Pro to
identify domain-specific tokens and domain-specific experts." Evaluation benchmarks: MMLU-Pro, GPQA Diamond,
AIME 24/25. Prompt format from the repo (`token_expert_analysis.py`, `analyze_domain_data`):
`prompt = question + "\n" + "\n".join([f"{chr(65+i)}. {opt}" ...]) + "\nAnswer:"`.

## 3. Definitions of specialization and expert usage

Definition 3.4 (Domain-specific Expert): "These experts are characterized by a dual property: they are
frequently activated within a specific domain, and when activated, they exhibit a strong preference for
processing domain-specific tokens rather than common tokens." Score, Eq. 8: "g(ej) = P(ej|D) · [P(s in S|ej)
- P(s in C|ej)] where P(ej|D) is the activation frequency of ej on domain D, and the conditional
probabilities represent the expert's token preference." Domain-specific tokens are defined by gradient
saliency, Eq. 5: "ri = ||ei (Hadamard) grad_ei L||_2", thresholded at a quantile p ("values in the range
(0.15, 0.5) are effective").

Expert usage is binary top-k membership counts, pooled over all tokens and prompts of a domain, per
(layer, expert) pair. Repo: "e_score = (f_important - f_unimportant) * p_e" with "f_important: frequency of
expert activation on important tokens" and "p_e: overall activation frequency of the expert"; activations
come from `torch.topk(router_logits, top_k)` in a forward hook on every MoE layer. Pooled, not per-token;
no routed-weight magnitude, only membership. All MoE layers are scored jointly (Figures 3 to 6 are
layer x expert heatmaps); K experts are chosen from the global ranking, so K can land in any layers.

## 4. PREFILL vs GENERATION (the key item)

They do not separate the two, and expert identification is prefill only. Three sentences settle it:

- Sec. 4.2: "For Step (1), to ensure a fair evaluation, we identify domain-specific tokens using only the
  question tokens, excluding answer tokens."
- Sec. 4.3: "For each target domain, we use only question texts (without labels) from MMLU-Pro to
  identify domain-specific tokens and domain-specific experts."
- Sec. 4.5: "For DSMoE, the one-time identification cost scales linearly with the number of samples in a
  domain, yielding a time complexity of O(L) forward passes, where L denotes the number of domain-specific
  samples. In contrast, the RICE baseline incurs a substantially higher cost of O(L x M) forward passes,
  where M is the number of generated thinking tokens per sample."

The last quote is explicit: one forward pass per question, and the contrast with RICE is precisely that
RICE looks at generated tokens and they do not. The repo confirms: `_ = model(input_ids=input_ids,
use_cache=False)` under `torch.no_grad()` on the tokenized question string; there is no `generate` call in
the analysis path. The steering, by contrast, acts on every token at inference including generated ones
(Eq. 11 rescales the router weight for every "given input token"; evaluation is on generated answers, e.g.
Table 1 shows full chain-of-thought output). So the design is: experts identified from prefilled
question text, then applied during generation. Whether the identified experts are the ones generation
would itself have chosen is never measured.

## 5. Subject / domain organization during generation

Not tested. Domain organization is tested on prefill only (question tokens), for four MMLU-Pro domains
(Math, Biology, Physics, Chemistry). The finding is not a routing statistic but a downstream accuracy
gain: steering K=1 expert at alpha=3.0 on Math improves accuracy on all ten models, "with gains ranging
from 3% to 45%" (Figure 1). In Table 2, DSMoE gives "average absolute improvements of +1.5, +14.5, +3.6,
and +3.7 percentage points for Qwen3-30B-Instruct, GPT-OSS-120B, Qwen3-30B-Thinking, and GPT-OSS-20B."
No overlap, no crew turnover, no per-subject winner counts. Qualitative only: "certain layers exhibit
clusters of highly specialized experts ... while others contain more domain-agnostic experts" (App. A.1).

## 6. Headline claim and evidence

Verbatim (Conclusion): "Following a comprehensive analysis of models up to 120B parameters, we confirmed
that distinct experts align with specific domains."

Evidence, in their order:
1. Figure 1: upweighting the top-1 scored expert (alpha=3.0) raises MMLU-Pro Math accuracy on all ten models.
2. Table 2: DSMoE beats Original, RICE, and LoRA SFT on four MMLU-Pro domains for four models.
3. Table 3: transfer to GPQA Diamond, "average gains ranging from +4.8 to +27.1 percentage points."
4. Table 4: AIME 24/25 gains "+12.3 to +27.3 percentage points" (Qwen3-30B-Instruct, GPT-OSS-20B).
5. Tables 5 and 6: K=20 and alpha=5.0 best for GPT-OSS-20B Biology; "K to approximately 1% of the total number of experts."

Note that the existence claim rests entirely on accuracy improving under steering; there is no
independent routing-level test of domain alignment and no null model.

## 7. Causal test

Yes, the whole method is a causal upweighting: Eq. 11 "w~j = alpha · wj if ej in E*, wj otherwise", then
"the weights are typically re-normalized (e.g., via a Softmax function)". Only positive steering
(alpha > 1); no ablation or suppression of the identified experts. Result: accuracy gains above. Table 6
shows alpha=50 hurts (68.0 vs 76.4 baseline) and Table 5 shows K=5 and K=10 slightly hurt (73.6, 73.3 vs
76.4) before K=20 helps (78.7). [paraphrase: the K curve is non-monotone, which they do not discuss.]

## 8. Terms

- "domain-specific experts" and "Common Token" / "Domain-specific Token" (Defs 3.1, 3.2). Their
  "common" is a token-level notion (low gradient saliency), not an expert-level one. Avoid conflating
  with our "default expert".
- "domain-agnostic experts": "others contain more domain-agnostic experts" (App. A.1). Closest to our
  default/generalist expert; usable.
- "thinking experts": from RICE, "RICE targets thinking experts that exhibit substantial variability
  across samples or domains." Avoid.
- "expert specialization" used throughout as a training objective ("extensive work on improving expert
  specialization"). "Domain steering" for their intervention. They do not use "generalist",
  "polysemantic", "standing committee", or "router collapse" (they cite Chi et al. 2022 on
  "representation collapse" only in related work).

## 9. Agreement with our findings

1. Prefill default expert vs generation winners: SILENT on generation; on prefill they implicitly
   acknowledge frequent-but-uninformative experts: "This penalizes experts that are active frequently but
   only process common, non-informative tokens." Their frequency term P(ej|D) is where our E224 would
   dominate, and their correction (subtracting common-token preference) is a workaround for exactly the
   flat-default problem we report.
2. Crew turnover at subject boundaries / crew identifies subject: SILENT. No overlap or
   identification statistics.
3. Register moves crews: SILENT. All prompts are one register (MCQ questions).
4. Long technical prompt shares crew across prefill and generation: SILENT, but their success is
   consistent with it: experts picked from question text still help during generation on the same
   domain. [paraphrase]
5. Leaders removable, not load-bearing: PARTLY DISAGREES in spirit. They show upweighting one expert
   changes accuracy by "3% to 45%"; they never test removal. Our removal-null is on generation-identified
   leaders; theirs is a gain from prefill-identified experts, so the two are not the same experts.
6. Fine-tunes do not touch routers: SILENT (they compare to LoRA SFT on accuracy only; LoRA targets
   include gate_proj, up_proj, down_proj, which are expert MLPs, not the router).

## 10. What they did that we did not, and the reverse

They: ten architectures from 16 to 512 experts; gradient-saliency token weighting; a positive-steering
method with downstream accuracy on MMLU-Pro, GPQA, AIME; comparison to SFT and RICE; K and alpha
ablations; vLLM implementation.

We: separate prefill and generation blocks; routed-weight crews rather than binary membership; overlap
and cross-design identification statistics with nulls; register and dialect manipulations; a packed
multi-subject answer; suppression (bias-off) of leaders; router-identity check across fine-tunes;
three model families including a 256-expert Qwen3.5.

## 11. Reviewer risks and honest replies

- "Do et al. find domain experts from prefill and steering them works, so prefill is not missing
  anything." Reply: their identification is on question text with a saliency correction that explicitly
  discounts frequent experts on common tokens; the gains show that some prefill-scored expert helps, not
  that prefill routing is organized by subject. They report no routing statistic on generated tokens, and
  their own cost analysis states identification is O(L) forward passes over questions. Our claim is about
  which tokens carry the subject signal, and they did not measure that.
- "Their K=1 steering changes accuracy up to 45%, so single experts are load-bearing, contradicting
  finding 5." Reply: upweighting and removal are different tests, the experts differ (prefill-scored vs
  generation leaders), and their Table 5 shows small K can also hurt. We should say our removal result is
  for generation-identified leaders on generation behavior and does not speak to gains from upweighting.
- "Ten models vs your one main model." Reply: acknowledge; our 122B and GLM legs are prefill only.

## 12. Draft related-work sentences

Do, Le, and Tran (2026) ask whether domain-specific experts exist in ten MoE models from 3.8B to 120B
parameters and answer yes by scoring experts on their activation frequency and their preference for
gradient-salient tokens, then upweighting the top scored experts at inference. Their expert statistics
come from a single forward pass over MMLU-Pro question text, with answer tokens excluded, and no routing
statistic is reported for generated tokens. The accuracy gains they obtain on MMLU-Pro, GPQA Diamond, and
AIME show that prefill-scored experts can help generation, but they leave open whether generation routing
itself is organized by subject. Their correction for experts that are "active frequently but only process
common, non-informative tokens" is a prefill-side symptom of the flat default expert we measure directly.
