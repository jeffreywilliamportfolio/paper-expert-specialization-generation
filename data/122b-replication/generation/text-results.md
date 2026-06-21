# Text Results — 122B Domain Specialist Generation

This report summarizes the generated text behavior for the 60-prompt domain-specialist run and connects it to the routing analysis without treating Expert 114 as the primary target. For this model, the question is which experts or expert cluster look behaviorally analogous to the 35B philosophy/self-reference pattern.

Primary artifacts:

- [results.md](results.md)
- [results-generated.txt](results-generated.txt)
- [results_20260412T161833Z_qwen122_domain_specialist_probe_60_gen_n2048.json](RESULTS/results_20260412T161833Z_qwen122_domain_specialist_probe_60_gen_n2048.json)

## Working Takeaway

The text generation run shows that the 122B model can produce coherent, domain-appropriate expository answers across all 20 disciplines, but the likely analog of the 35B philosophy-sensitive routing pattern is not a single carryover expert. The strongest candidates are a small philosophy-adjacent cluster led by `E40` and `E5`, with supporting candidates in comparative religion, linguistics, political science, and psychology.

## Text Behavior

- The model generated long, content-rich answers for all `60` prompts.
- `49/60` cells hit the `2048` token cap.
- Mean generated length was `1868.27` tokens.
- Minimum generated length was `809` tokens.
- `11/60` cells stopped before `2048`.
- Every cell eventually spilled into chat-template continuation artifacts.
- Total spill counts across the bundle were `<|im_start|> = 258` and `<|endoftext|> = 86`.

So the outputs are usable for content inspection, but they are not cleanly self-terminating. The dominant failure mode is not incoherence inside the domain answer. It is over-generation into a follow-up turn after the answer is already substantively complete.

## What The Text Actually Looks Like

The generation quality is strong at the domain-answer level. Examples from [results-generated.txt](results-generated.txt):

- `D01_history_01` opens with a structured explanation of WWII historiography using archives, logistics, intelligence, and command documents, then only later spills into a new user turn.
- `D16_philosophy_01` gives a clean epistemology answer organized around belief, justification, and knowledge.
- `D16_philosophy_02` gives a conventional Kant summary with the expected epistemology / metaphysics split.
- `D18_linguistics_03` cleanly distinguishes phonology from syntax in terms of units, hierarchy, and structure.
- `D19_psychology_03` cleanly separates cognition from emotion in the expected explanatory style.
- `D10_computer_science_03` gives a standard data-structures-versus-algorithms comparison.

The practical read is:

- The model is not failing to stay on domain.
- It is mostly producing textbook-style expository answers.
- The spill happens after the core answer, not instead of it.

## Why This Matters For The Expert Search

This run is not a self-reference probe. It is a domain-specialist map with generation enabled. So the text evidence here should not be used to claim a direct self-report mechanism. It should be used to identify which experts appear to underwrite philosophy-adjacent or introspection-adjacent domain behavior in 122B.

That means the most relevant experts are not just the global top generation experts by routed weight. Those broad leaders:

- `E140`
- `E76`
- `E107`
- `E8`
- `E4`
- `E5`

are useful as dominant generation drivers, but they are not all plausible 122B analogs of the 35B `E114` pattern. Some are clearly broad global generation contributors rather than narrow content specialists.

## Candidate 122B Analogs To The 35B Philosophy / Self-Reference Pattern

The strongest philosophy-adjacent candidates from the generation run are:

- `E40`
  - philosophy winner by `W`
  - `W = 0.009897`
  - `S = 0.057217`
  - `Q = 0.124330`
  - top-vs-second philosophy ratio `2.22x`
- `E5`
  - philosophy rank `2` by `W`
  - philosophy winner by `S`
  - `W = 0.009398`
  - `S = 0.061432`
  - `Q = 0.127272`
- `E125`
  - philosophy candidate rank `2` by composite `W`
- `E49`
  - philosophy candidate rank `4`
  - also appears in comparative religion
- `E101`
  - philosophy candidate rank `5`
  - also appears strongly in psychology

Supporting adjacent-domain candidates:

- Comparative religion:
  - `E159` wins by `W`
  - `E234` is the top composite candidate by `W`
- Linguistics:
  - `E102` wins by both `W` and `S`
  - `E85` and `E63` are the strongest selective candidates
- Political science:
  - `E160` wins by both `W` and `S`
  - `E170` is the next strongest composite candidate
- Psychology:
  - `E223` wins by `W`
  - `E101` wins by `S`
  - `E147` is also a strong selective candidate

This points to a cluster, not a singleton:

- philosophy core: `E40`, `E5`, `E125`
- adjacent interpretive / symbolic domains: `E49`, `E159`, `E102`, `E160`, `E101`

## Where Expert 114 Actually Lands

`E114` is relevant, but not central, in this model’s domain-generation map.

- In philosophy generation:
  - `E114 W = 0.003017`
  - rank `182` by `W`
- In comparative religion generation:
  - rank `228` by `W`
- In linguistics generation:
  - rank `192` by `W`
- In political science generation:
  - rank `189` by `W`
- In psychology generation:
  - rank `222` by `W`

The one place `E114` still matters here is computer science:

- computer science generation:
  - `E114 W = 0.006998`
  - `S = 0.049487`
  - `Q = 0.114485`
  - rank `4` by `W`
  - rank `5` by `S`
  - top composite candidate by `W` within that domain

So the clean interpretation is:

- `E114` is not the 122B philosophy specialist.
- `E114` still has a real footprint, but it looks more computer-science-linked in this run.
- The likely 122B analog of the old philosophy / self-reference-sensitive role is distributed across a different expert cluster.

## Text-Side Implication

Because the generated outputs are domain-correct and stable before spill, this run supports using the 122B domain-specialist map as a valid search surface for candidate analogs. The text does not show a collapse into generic filler. It shows sustained domain exposition, which means the routing specialization is attached to meaningful content production rather than to degenerate decoding behavior.

That makes the next step straightforward:

- probe the philosophy-adjacent cluster rather than chasing `E114` by index
- prioritize `E40` and `E5`
- treat `E125`, `E49`, `E159`, `E102`, `E160`, and `E101` as second-line follow-up candidates
- keep `E114` in scope as a computer-science-adjacent specialist, not as the presumed main target
