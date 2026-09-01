# Related-work read: "The Expert Strikes Back" (arXiv 2604.02178)

Source read: arXiv HTML v2 (`arXiv:2604.02178v2 [cs.CL] 15 May 2026`). All quotes verbatim from that
version; paraphrase is marked [para]. Section numbers are the paper's own.

## 1. Citation

Jeremy Herbst, Stefan Wermter, Jae Hee Lee (University of Hamburg). "The Expert Strikes Back:
Interpreting Mixture-of-Experts Language Models at Expert Level." arXiv:2604.02178, v2 dated 15 May
2026. Venue: the task brief says ICML 2026; the HTML carries an "Impact Statement" (ICML format) but I
did not see the venue printed in the text itself. Code: https://github.com/jerryy33/MoE_analysis.

## 2. Models, sizes, experts, top-k, data

Appendix B, Table 2 (N = total experts, N_A = active, N_SE = shared, layers):
- OLMoE-1B-7B: 64 / 8 / 0, 16 layers
- ERNIE-4.5-21B-A3B: 66 / 6 / 2, 28 layers
- Qwen3-30B-A3B: 128 / 8 / 0, 48 layers
- gpt-oss-20b: 32 / 4 / 0, 24 layers
- GLM-4.7-Flash: 65 / 4 / 1, 47 layers
- Mixtral-8x7B-v0.1: 8 / 2 / 0, 32 layers (8-bit quantized)
- Dense controls: OLMo-1B, OLMo-7B, Llama-3.2-3B, Qwen3-4B-Base, Ministral-3-3B, pythia-12b.

No Qwen3.5 model; Qwen3-30B-A3B (128 experts, top-8) is their closest relative to our 35B-A3B (256
experts, top-8). GLM-4.7-Flash appears only in the probing comparison.

Data, probing (Sec. 4.1, App. C): "We evaluate 12 different models ... across 58 concepts spanning four
categories: Part-of-Speech, LaTeX, code, and natural language text". "For each concept, we initially
collect 5,000 token samples, balanced between positive and negative classes." Datasets (Table 3): POS
tagged Wikipedia (16 concepts), pile-uncopyrighted ArXiv subset for LaTeX (12), GitHub subset for code
(20), all subsets for text (10). Concepts are regex-defined token properties, e.g. `is_inline_math`,
`is_function_def`, `leading_capital`.

Data, auto-interp (App. D): "We extract activations over the pile-uncopyrighted dataset ... For each
document, we extract one random sequence containing 32 tokens until 2 x 10^6 tokens have been
processed." Explainer/scorer: "For both the explainer and scorer model we use Gemini 3 Flash Preview."

Data, specialization score (Sec. 6.2): "we analyze expert behavior over 10^6 tokens across multiple
resolutions (k in {10, 50, 100, 1000, 5000})". The corpus for those 10^6 tokens is not named in the
text I read; presumably the same pile-uncopyrighted sample [para].

## 3. Definitions of "specialization" and "expert usage"

"We measure expert specialization: the degree to which an expert's behavior isolates specific
functional or semantic domains." (Sec. 6.2)

Domains are not human categories: "we define domains natively by performing unsupervised k-means
clustering on the model's output embedding matrix (the unembedding)."

Two scores, both JSD of an expert's cluster distribution against the layer base rate (App. I):
"Routing Specialization (Input): We track the actual tokens the router assigns to the expert."
"Functional Specialization (Output): We apply Logit Lens to the tokens promoted by the expert's output
vector." "A Specialization Score of 0 indicates the expert processes tokens in the exact same
proportions as the layer average (no specialization)." A multinomial "Random Expert Baseline" is
subtracted.

Expert "usage"/activity for auto-interp is NOT the router weight: "We cannot simply rely on the router
weight g_i(x), because being selected by the router only means the expert was given the opportunity to
process the token". They use the residual-write norm: "g_i(x) ||E_i(x)||_2", and a sequence score
"score(s, E_i) = max over x in s of g_i(x) ||E_i(x)||_2".

Per-token or pooled: token-level throughout. Probes are per-token on the expert's intermediate
activation h, restricted to routed tokens: "For MoE models, we filter this dataset to include only the
subset of tokens that were routed to the target expert." Specialization scores are token-count
distributions over clusters per expert, i.e. pooled over 10^6 tokens but without any per-prompt or
per-block split [para].

Layers: probing uses every layer with best-layer selection ("we train probes on every layer (and every
expert for MoEs)"). Auto-interp: "all experts in 8 layers of OLMoE-1B-7B, 3 layers of ERNIE-4.5-21B-A3B,
and 3 layers of Qwen3-30B-A3B". Causal test: "Layers 4, 9 and 14 of the OLMoE-1B-7B model".
Specialization curves (Fig. 7): all layers of OLMoE-1B-7B.

## 4. Prefill vs generation (the key item)

They do not separate the two, and they never measure routing on model-generated tokens. Every
measurement is a forward pass over existing text (pile, Wikipedia, or LLM-written test sentences).
The words "generation", "decoding", "sampling", "prefill" and "teacher forcing" do not occur in a
routing-measurement sense anywhere in the paper; "generate" appears only for Gemini producing labels
and test cases.

Settling sentences:
- Probing: "For each concept, we initially collect 5,000 token samples ... For MoE models, we filter
  this dataset to include only the subset of tokens that were routed to the target expert."
- Auto-interp: "We extract activations over the pile-uncopyrighted dataset ... one random sequence
  containing 32 tokens".
- Causal test: "We run a forward pass and measure the expert's ranking among all experts from the same
  layer in terms of its DLA contribution to the target word." The target word is already in the prompt
  (footnote 2: in "We need to address the elephant in the room", "the is the trigger word and room is
  the target word"), so the attribution is to a teacher-forced next token, not a produced one.

Plainly: this is a prefill-only paper. Every routing decision they analyze is the router's response to
corpus text supplied as input. Nothing in it speaks to how the router behaves on tokens the model
itself is emitting.

## 5. Do they test subject/domain organization during generation?

No (see 4). They test domain organization on corpus text, and their answer is that domain is not the
organizing axis: "If experts were broad domain specialists, the score for broad categories (k=10) would
be high. Instead, we see that the highest granularity (k=5000) pulls dramatically ahead of the broad
semantic lines." Domains are unsupervised unembedding clusters, not named subjects; the only numeric
anchor given is "a JSD of 0.4 in a high-dimensional vocabulary space indicates that an expert is
strongly biased away from the layer's common base rate toward a narrow set of tokens." Fig. 7 (OLMoE)
is the evidence; I saw no table of per-domain numbers. They do concede some domain experts exist:
"Semantic: We find experts, mostly in mid-to-late layers, that represent closely what one would call a
domain expert. For example, OLMoE-L4-E3 operates mostly in legal and patent related documents".

## 6. Headline claim and evidence

Verbatim (Abstract): "This analysis allows us to resolve the debate on specialization: experts are
neither broad domain specialists (e.g., biology) nor simple token-level processors. Instead, they
function as fine-grained task experts, specializing in linguistic operations or semantic tasks (e.g.,
closing brackets in LaTeX)."

The requested "fine-grained task experts" sentence also appears in Sec. 2: "Our work bridges this gap
by demonstrating that experts function as fine-grained task experts." And Sec. 6: "we demonstrate that
experts function as fine-grained task experts, performing precise computational operations that are
often domain-restricted but functionally specific."

Evidence (<= 5 lines):
- k-sparse probing: "MoE experts often achieve near-optimal F1 scores at k=1"; gap vs dense widens with
  sparsity (Mixtral N_A/N=0.25 lower than Qwen3-30B-A3B N_A/N ~ 0.06).
- OLMoE-1B-7B beats dense OLMo-7B "Despite the dense model having 7x more active parameters per token".
- Auto-interp F1: "Most experts achieve F1 scores above 0.8"; Qwen3-30B-A3B "frequently exceeding 0.9".
- Trigger-Target DLA: expert "was either the Top-1 or among the Top-8 contributors to the target word";
  "In 80% of the cases, the specific expert was not even routed to control prompts".
- k-sweep of JSD specialization: k=5000 curve "pulls dramatically ahead" of k=10 in late layers.

## 7. Causal test

Sec. 5.3 "Trigger-Target": 3 layers x 10 random OLMoE experts x 20 Gemini-written test cases; metric
is the expert's DLA rank on the target token. Result as quoted in item 6. This is attribution on a
forward pass, not ablation, steering or router biasing. The Impact Statement itself flags this:
"We therefore recommend treating expert labels as tentative summaries, validating them with
counterfactual tests (including ablations and out-of-distribution checks)". No expert is removed or
biased anywhere in the paper.

## 8. Terms to reuse or avoid

- "fine-grained task experts" / "task specialists" (their thesis term; reuse when contrasting).
- "modular monosemanticity": "two distinct mechanisms work in tandem to create what we term modular
  monosemanticity."
- "routing sparsity": "We define the routing sparsity as the ratio N_A/N".
- "Routing Specialization (Input)" and "Functional Specialization (Output)" (useful split; our crews are
  routing-side only).
- "domain-restricted": "An expert may be domain-restricted (e.g., LaTeX), but its role is better
  described as a concrete computational operation".
- "Hyper-specialized experts" (e.g. "Qwen3-L44-E12 responds to Iranian administrative geography").
- "the router acts as a filter, ensuring an expert is only activated for a restricted, semantically or
  syntactically related subset of tokens."
- They do NOT use "generalist", "standing committee", "router collapse", "default expert" or
  "polysemantic expert"; "polysemanticity" is used only at the neuron level. No vocabulary for a
  flat, always-on expert like our E224 exists in the paper.

## 9. Agreement with our findings 1-6

1. Prefill default expert / generation 20 winners: SILENT. They never see generation and never report
   a layer-wide dominant expert; the closest is the design admission that base rates are skewed: "a
   random sample of text will naturally be dominated by common function words."
2. Crew turnover at subject boundaries, 95-96% identification: SILENT on generation; on corpus text
   DISAGREES in spirit: "They do not represent broad semantic domains; rather, they take in relatively
   general signals and apply a highly precise functional or syntactic transformation".
3. Register moves crews as much as subject: SILENT (no register variable), though their taxonomy is
   compatible: "Operational: ... experts that primarily enforce local validity constraints."
4. Long coherent technical prompt shares crew across prefill and generation: SILENT; but their whole
   corpus is long coherent text, which is exactly our regime where prefill and generation agree [para].
5. Philosophy leaders removable: SILENT on ablation; their only causal evidence is DLA rank, and they
   ask for ablations as future validation (quote in item 7).
6. Fine-tunes never touched the routers: SILENT (base/pretrained checkpoints only; GLM-4.7-Flash is
   probed but no fine-tune comparison).

## 10. What they did that we did not, and vice versa

They did: per-neuron k-sparse probes inside experts across 12 models with dense controls; automatic
natural-language labels for hundreds of experts with LLM scoring; DLA rank attribution; an unsupervised
unembedding-cluster JSD score with a multinomial null; a k-sweep showing fine-grained beats coarse.

We did: separate prefill from generation blocks; measure crews on tokens the model produced; a
subject-boundary turnover test with position-matched, drift and label-shuffle nulls; a cross-design
crew-to-subject identification test; a register manipulation; a router-bias ablation; a fine-tune vs
base router identity check; 256-expert Qwen3.5 and 122B-A10B models.

## 11. What a reviewer could turn against us, and the honest reply

Objection: Herbst et al. show with a proper null (multinomial baseline, k-sweep) that expert
specialization is at the fine-grained operation level, not the domain level, so our "subject crews" may
be an artifact of coarse labels or of the top-8-by-mean-weight aggregation.
Reply: their score is measured on corpus text presented as input, which is our prefill condition, and
on prefill our numbers agree with theirs (overlap 0.51, one default winner, 15-17% identification). The
subject organization we report appears only on generated tokens, a condition they never measure. Our
scoping result (finding 4) further says that long coherent input text, which is all they use, shares
the crew across blocks. So their result and ours are not in conflict; they cover different token
sources. Also, "subject" in our test is operationalized as crew identity across independent designs
(95-96% vs 6% chance), not as a human category imposed on tokens.

Objection: router weight is a poor activity measure ("being selected by the router only means the
expert was given the opportunity to process the token"). Our crews use mean routed weight only.
Reply: fair; we should say our crew is a routing-side (their "Routing Specialization") object and not
claim it measures what the expert writes. A residual-write-norm crew is a cheap follow-up.

Objection: they show OLMoE-L4-E3 style domain experts exist in mid-to-late layers, so "domain" is not
absent, just rarer than task experts; our per-subject winners could be those.
Reply: agreed and consistent; the interesting part of our result is that domain crews are invisible
on prefill and visible on generation, which their design cannot see.

## 12. Draft related-work sentences

Herbst, Wermter and Lee (2026) compare MoE experts with dense feed-forward layers using k-sparse probes
on 58 token-level concepts across twelve models and find that expert neurons are markedly less
polysemantic, with the gap widening as routing gets sparser. They then label hundreds of experts
automatically and score expert specialization as the Jensen-Shannon divergence of an expert's token
distribution from its layer base rate over unembedding clusters, concluding that experts are
fine-grained task experts rather than broad domain specialists. All of their routing and attribution
measurements are forward passes over corpus or hand-written text, so they characterize what we call
the prefill block. Our results on prefill agree with theirs; the subject-organized crews we report
appear only on generated tokens, which their design does not sample.
