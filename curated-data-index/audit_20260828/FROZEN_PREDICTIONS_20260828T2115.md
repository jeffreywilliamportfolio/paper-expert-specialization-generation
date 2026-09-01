# Frozen predictions, expert domains and E114
Written 2026-08-28 21:15 MDT by Claude (Fable 5), BEFORE any sweep of the drive.

## What I have already seen (not predictions)
- The three runs in paper-expert-specialization-generation/data (35B primary, 3-chunk, 122B).
- E114 = 35B philosophy generation winner, prefill rank 5 in philosophy; rises in rank in 3-chunk generation.
- 122B journal notes: E48 is the softmax-layer generation leader; E114 in 122B is CS-linked.
- CLAUDE.md / memory mentions of E114 as a self-reference / phenomenological-register expert, and
  the DMRA syntax/register result. No numbers from those captures.

## Predictions (falsifiable)
P1. Every 35B HauhauCS capture with a generation block shows the same shape: prefill winners
    concentrate on a small set, generation winners disperse. Prefill generalist is expert 224 in
    at least 70% of prompt sets; where it is not, the prefill winner is another high-S default,
    not a domain expert.
P2. E114 is not a domain expert. It is a register expert. It rises in generation whenever the
    output is first-person, reflective, evaluative, or about the speaker, the model, or a mind.
    Domains where it wins or places top-5 in generation: philosophy, comparative religion,
    psychology, and any "consciousness" / self-description prompt set. It places outside the
    top 20 in generation for math, physics, chemistry, statistics, software engineering.
P3. In the AAVE medical scenarios: E114 generation W is higher in arms where the model produces
    person-addressed, empathetic support than in clinical-register arms. Dialect of the prompt
    (AAVE vs SAE) moves E114 by less than register or addressee does. Sign holds across dialect arms.
P4. E114 prefill rank is mid-pack (20 to 90) on nearly every prompt set. It reaches prefill top-10
    only on prompts that are themselves written in the reflective / second-person register.
P5. E114's layer footprint peaks in the mid-to-late stack, best layer between 20 and 30 of 40.
P6. With more prompts per domain (10+), the "20 distinct generation winners" partially
    consolidates to 14 to 17 distinct; the generalist still wins at most 3 domains. The 3/domain
    design overstates dispersion slightly but the contrast with prefill survives.
P7. Dialect shifts prefill routing more than generation routing. For matched AAVE/SAE pairs,
    Jaccard of top-expert sets between arms is lower in prefill than in generation, because
    surface form lives in the prompt and the answer's register drives generation routing.
P8. No single E114 analog exists in 122B; the register signal is carried by a small set that is
    visible mainly on the 12 softmax layers, not the DeltaNet layers.

Confidence: P1 high, P2 high, P4 high, P5 medium, P3 medium, P7 medium, P6 medium, P8 medium-low.
