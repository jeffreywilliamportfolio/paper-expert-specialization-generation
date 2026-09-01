# Shared brief for the expert-specialization analysis agents (2026-08-28)

Goal: figure out what the archived routing captures actually say about EXPERT SPECIALIZATION BY DOMAIN in
MoE models, with prefill vs generation kept separate. Output feeds a plain-language findings document
for Jeffrey (regular math OK, no heavy statistics). Report numbers with provenance (file, run id).

Definitions (use these exactly):
- Router weights: take the router logits over 256 experts for one token at one layer, softmax, keep top 8,
  renormalize so the 8 sum to 1. (Qwen3.5 MoE, top-8 of 256. GLM-4.7-Flash differs: check its own
  qwen_router-equivalent / analysis scripts and its e_score_correction_bias before reconstructing.)
- For expert e on a token set: W = mean weight over all tokens (0 where unselected); S = fraction of tokens
  selecting e; Q = mean weight when selected. W = S*Q per token set.
- Prefill block = prompt token positions; generation block = produced tokens. Never pool them.
- Domain winner = highest-W expert over a domain's prompts, all layers pooled, per block.
- Concentration of a 20-winner list: distinct count, max wins, Herfindahl sum(p_i^2), normalized entropy.
- Generation "trimmed" = cut at the first chat-template spill (<|im_start|> / <|im_end|> / <|endoftext|>
  rendered as literal text). Report trimmed and untrimmed when both are cheap.

Existing helper: qwen_router.py (reconstruct_probs, normalized_entropy with ENTROPY_MAX=log2(8)) lives in
several run folders, e.g. paper-expert-specialization-generation/data/35b-3chunk-token-balanced/METHOD/.
Router tensor files are named ffn_moe_logits-<layer>.npy (shape [n_tokens, 256] or similar; verify).
Per-cell metadata.txt gives n_tokens_prompt so you can split prefill/generation.

Frozen predictions to score where your subset allows:
paper-expert-specialization-generation/ledger/FROZEN_PREDICTIONS_20260828T2115.md
For each prediction you can test, say SUPPORTED / REFUTED / MIXED with the numbers.

Rules:
- Read-only on all source data. Write ONLY inside paper-expert-specialization-generation/ledger/agents/.
- Do NOT spawn sub-agents. Do not touch any GPU box or network service. Local CPU only.
- Do not trust README/RESULTS summaries blindly: recompute at least the headline numbers from the
  tensors or JSON where they exist, and flag any disagreement.
- Report what you could NOT verify and why.
- Plain language in the findings file. Every number carries its file path.
- Output file: ledger/agents/<your-name>.md with sections: 1 What the data is (runs, models, prompts,
  sizes); 2 What I computed; 3 Findings (numbered, each with numbers + paths); 4 Prediction scores;
  5 Caveats / could not verify; 6 One-paragraph summary for a lay reader.
