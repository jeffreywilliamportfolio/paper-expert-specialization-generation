#!/usr/bin/env python3
"""Analyze all-expert domain specialization for the 60-prompt 122B HauhauCS run.

This script is scoped to the expert-specialization questions:
- identify experts associated with each domain
- compare domain winners against other specialists rather than only Expert 114
- separate prefill from generation
- expose expert selection (S), routed weight (W), and conditional weight (Q)
- summarize cross-domain crossover instead of assuming one clean expert/domain map
- make normalization explicit: softmax over all 256 experts, top-8, renormalize

Expected prompt metadata is read from a local JSON file with fields:
  id, domain, subtype, prompt

Subtype is constrained to:
  mechanism, history, synthesis
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


N_EXPERTS = 256
N_LAYERS = 48
TOP_K = 8
TRACKS = ("prefill", "generation_all", "generation_trimmed")
SUBTYPES = ("mechanism", "history", "synthesis")
IM_END_TOKEN_SEQUENCE = [27, 91, 316, 6018, 91, 29]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze all-expert domain specialization.")
    parser.add_argument("--capture-dir", required=True, help="Directory containing per-prompt cell subdirectories.")
    parser.add_argument("--prompt-json", required=True, help="Prompt metadata JSON for the 60-prompt set.")
    parser.add_argument("--report", required=True, help="Path to write Markdown summary.")
    parser.add_argument("--results-json", required=True, help="Path to write JSON summary.")
    parser.add_argument("--matrices-npz", default=None, help="Optional path to write dense NumPy matrices.")
    parser.add_argument("--model-label", required=True, help="Human-readable model label.")
    parser.add_argument("--top-k", type=int, default=12, help="Number of top experts to retain in summaries.")
    parser.add_argument(
        "--highlight-expert",
        type=int,
        default=114,
        help="Expert id to summarize explicitly alongside all-expert results.",
    )
    return parser.parse_args()


def load_reconstruct_probs() -> Any:
    qwen_router = Path(__file__).resolve().parent / "qwen_router.py"
    spec = importlib.util.spec_from_file_location("qwen_router", qwen_router)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load qwen_router from {qwen_router}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reconstruct_probs


reconstruct_probs = load_reconstruct_probs()


def parse_metadata(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def load_generated_token_ids(cell_dir: Path) -> list[int]:
    raw = (cell_dir / "generated_tokens.json").read_bytes()
    return [int(m.group(1)) for m in re.finditer(rb'"token_id"\s*:\s*(\d+)', raw)]


def find_im_end_index(token_ids: list[int]) -> int | None:
    seq_len = len(IM_END_TOKEN_SEQUENCE)
    for i in range(len(token_ids) - seq_len + 1):
        if token_ids[i : i + seq_len] == IM_END_TOKEN_SEQUENCE:
            return i
    return None


def compute_metric_vectors(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if probs.size == 0:
        nan = np.full((N_EXPERTS,), np.nan, dtype=np.float64)
        return nan.copy(), nan.copy(), nan.copy()
    weights = probs.astype(np.float64)
    selected = weights > 0
    counts = selected.sum(axis=0).astype(np.float64)
    W = weights.mean(axis=0)
    S = selected.mean(axis=0)
    totals = weights.sum(axis=0)
    Q = np.full((N_EXPERTS,), np.nan, dtype=np.float64)
    np.divide(totals, counts, out=Q, where=counts > 0)
    return W, S, Q


def empty_track_dict() -> dict[str, Any]:
    return {
        "W": np.full((N_LAYERS, N_EXPERTS), np.nan, dtype=np.float64),
        "S": np.full((N_LAYERS, N_EXPERTS), np.nan, dtype=np.float64),
        "Q": np.full((N_LAYERS, N_EXPERTS), np.nan, dtype=np.float64),
        "n_tokens_by_layer": np.zeros((N_LAYERS,), dtype=np.int32),
    }


def load_prompt_spec(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    rows = json.loads(path.read_text())
    by_id = {row["id"]: row for row in rows}
    domains = sorted({row["domain"] for row in rows})
    return rows, by_id, domains


def process_cell(cell_dir: Path, prompt_lookup: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    cell_id = cell_dir.name
    spec = prompt_lookup.get(cell_id)
    if spec is None:
        return None

    md = parse_metadata(cell_dir / "metadata.txt")
    n_prompt = int(md["n_tokens_prompt"])
    n_gen = int(md["n_tokens_generated"])
    token_ids = load_generated_token_ids(cell_dir)
    if len(token_ids) != n_gen:
        n_gen = min(n_gen, len(token_ids))
        token_ids = token_ids[:n_gen]
    trim_idx = find_im_end_index(token_ids)
    n_gen_trim = trim_idx if trim_idx is not None else n_gen

    tracks = {track: empty_track_dict() for track in TRACKS}
    missing_prefill_layers: list[int] = []
    missing_generation_layers: list[int] = []
    l39_trimmed = False

    for layer in range(N_LAYERS):
        logits_path = cell_dir / "router" / f"ffn_moe_logits-{layer}.npy"
        if not logits_path.exists():
            missing_prefill_layers.append(layer)
            missing_generation_layers.append(layer)
            continue

        arr = np.load(logits_path)
        if arr.ndim != 2 or arr.shape[1] != N_EXPERTS:
            missing_prefill_layers.append(layer)
            missing_generation_layers.append(layer)
            continue

        if arr.shape[0] >= n_prompt:
            prefill_logits = arr[:n_prompt]
            W, S, Q = compute_metric_vectors(reconstruct_probs(prefill_logits))
            tracks["prefill"]["W"][layer] = W
            tracks["prefill"]["S"][layer] = S
            tracks["prefill"]["Q"][layer] = Q
            tracks["prefill"]["n_tokens_by_layer"][layer] = n_prompt
        else:
            missing_prefill_layers.append(layer)

        expected_rows = n_prompt + n_gen
        if arr.shape[0] == expected_rows:
            gen_logits = arr[n_prompt : n_prompt + n_gen]
        elif arr.shape[0] == n_gen + 1:
            gen_logits = arr[1:]
            if layer == 39:
                l39_trimmed = True
        else:
            missing_generation_layers.append(layer)
            continue

        if gen_logits.shape[0] != n_gen:
            missing_generation_layers.append(layer)
            continue

        W, S, Q = compute_metric_vectors(reconstruct_probs(gen_logits))
        tracks["generation_all"]["W"][layer] = W
        tracks["generation_all"]["S"][layer] = S
        tracks["generation_all"]["Q"][layer] = Q
        tracks["generation_all"]["n_tokens_by_layer"][layer] = n_gen

        if n_gen_trim > 0:
            Wt, St, Qt = compute_metric_vectors(reconstruct_probs(gen_logits[:n_gen_trim]))
            tracks["generation_trimmed"]["W"][layer] = Wt
            tracks["generation_trimmed"]["S"][layer] = St
            tracks["generation_trimmed"]["Q"][layer] = Qt
        tracks["generation_trimmed"]["n_tokens_by_layer"][layer] = n_gen_trim

    return {
        "cell_id": cell_id,
        "domain": spec["domain"],
        "subtype": spec["subtype"],
        "prompt": spec["prompt"],
        "n_tokens_prompt": n_prompt,
        "n_tokens_generated": n_gen,
        "n_tokens_generation_trimmed": n_gen_trim,
        "trim_index": trim_idx,
        "l39_trimmed": l39_trimmed,
        "missing_prefill_layers": missing_prefill_layers,
        "missing_generation_layers": missing_generation_layers,
        "tracks": tracks,
    }


def mean_over_cells_and_layers(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.full((N_EXPERTS,), np.nan, dtype=np.float64)
    stacked = np.stack(arrays, axis=0)  # [cells, layers, experts]
    flat = stacked.reshape(-1, N_EXPERTS)
    return np.nanmean(flat, axis=0)


def mean_over_cells(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.full((N_LAYERS, N_EXPERTS), np.nan, dtype=np.float64)
    stacked = np.stack(arrays, axis=0)  # [cells, layers, experts]
    return np.nanmean(stacked, axis=0)


def top_experts(primary: np.ndarray, W: np.ndarray, S: np.ndarray, Q: np.ndarray, limit: int) -> list[dict[str, Any]]:
    score = np.where(np.isnan(primary), -np.inf, primary)
    order = np.argsort(-score)[:limit]
    rows = []
    for rank, expert in enumerate(order, start=1):
        if not np.isfinite(score[expert]):
            continue
        rows.append(
            {
                "rank": rank,
                "expert": int(expert),
                "W": float(W[expert]),
                "S": float(S[expert]),
                "Q": float(Q[expert]) if np.isfinite(Q[expert]) else None,
            }
        )
    return rows


def top_domain_pairs_from_jaccard(domains: list[str], matrix: np.ndarray, limit: int = 20) -> list[dict[str, Any]]:
    rows = []
    for i, left in enumerate(domains):
        for j in range(i + 1, len(domains)):
            rows.append({"left": left, "right": domains[j], "jaccard": float(matrix[i, j])})
    rows.sort(key=lambda row: row["jaccard"], reverse=True)
    return rows[:limit]


def normalized_entropy(values: np.ndarray) -> float | None:
    positive = values[np.isfinite(values) & (values > 0)]
    total = float(positive.sum())
    if total <= 0:
        return None
    p = positive / total
    ent = -float(np.sum(p * np.log(p)))
    max_ent = math.log(len(values)) if len(values) > 1 else 1.0
    return ent / max_ent if max_ent > 0 else 0.0


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def prompt_level_expert_means(cells: list[dict[str, Any]], track: str, metric: str) -> np.ndarray:
    """Return [n_cells, 256] prompt-level expert means over layers for one metric."""
    rows = []
    for cell in cells:
        rows.append(np.nanmean(cell["tracks"][track][metric], axis=0))
    return np.stack(rows, axis=0) if rows else np.empty((0, N_EXPERTS), dtype=np.float64)


def candidate_scores_for_domain(
    *,
    domain: str,
    metric_label: str,
    domain_prompt_metric: np.ndarray,
    domain_metric: np.ndarray,
    domain_W: np.ndarray,
    domain_S: np.ndarray,
    domain_Q: np.ndarray,
    domains: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    domain_idx = domains.index(domain)
    rows: list[dict[str, Any]] = []
    for expert in range(N_EXPERTS):
        prompt_vals = domain_prompt_metric[:, expert]
        domain_metric_mean = safe_mean(prompt_vals)
        if domain_metric_mean <= 0:
            continue

        finite_prompt_vals = prompt_vals[np.isfinite(prompt_vals)]
        prompt_std = float(np.std(finite_prompt_vals)) if finite_prompt_vals.size else 0.0
        consistency = 1.0 / (1.0 + (prompt_std / (domain_metric_mean + 1e-12)))

        expert_domain_means = np.where(np.isfinite(domain_metric[:, expert]), domain_metric[:, expert], 0.0)
        total_domain_mass = float(np.sum(expert_domain_means))
        selectivity = domain_metric_mean / total_domain_mass if total_domain_mass > 0 else 0.0

        other = np.delete(expert_domain_means, domain_idx)
        other_mean = float(np.mean(other)) if other.size else 0.0
        other_max = float(np.max(other)) if other.size else 0.0
        separation = (
            domain_metric_mean / (domain_metric_mean + other_mean)
            if (domain_metric_mean + other_mean) > 0
            else 0.0
        )
        separation_ratio = None if other_mean <= 0 else domain_metric_mean / other_mean
        max_gap = domain_metric_mean - other_max

        composite = domain_metric_mean * consistency * selectivity * separation
        rows.append(
            {
                "expert": expert,
                "metric": metric_label,
                "domain_metric": domain_metric_mean,
                "consistency": consistency,
                "selectivity": selectivity,
                "other_19_mean_metric": other_mean,
                "other_19_max_metric": other_max,
                "separation": separation,
                "separation_ratio": separation_ratio,
                "max_gap": max_gap,
                "W": float(domain_W[domain_idx, expert]),
                "S": float(domain_S[domain_idx, expert]),
                "Q": float(domain_Q[domain_idx, expert]) if np.isfinite(domain_Q[domain_idx, expert]) else None,
                "rank_by_W": rank_by_desc(domain_W[domain_idx], expert),
                "rank_by_S": rank_by_desc(domain_S[domain_idx], expert),
                "score": composite,
            }
        )

    rows.sort(
        key=lambda row: (
            -row["score"],
            -row["domain_metric"],
            -row["consistency"],
            -row["selectivity"],
        )
    )
    for rank, row in enumerate(rows[:top_k], start=1):
        row["rank"] = rank
    return rows[:top_k]


def rank_by_desc(values: np.ndarray, expert: int) -> int | None:
    if not np.isfinite(values).any():
        return None
    order = np.argsort(-np.where(np.isnan(values), -np.inf, values))
    return int(np.where(order == expert)[0][0]) + 1


def build_track_summary(
    cells: list[dict[str, Any]],
    track: str,
    domains: list[str],
    top_k: int,
    highlight_expert: int,
) -> dict[str, Any]:
    domain_index = {domain: idx for idx, domain in enumerate(domains)}
    subtype_index = {subtype: idx for idx, subtype in enumerate(SUBTYPES)}

    overall_W = mean_over_cells_and_layers([cell["tracks"][track]["W"] for cell in cells])
    overall_S = mean_over_cells_and_layers([cell["tracks"][track]["S"] for cell in cells])
    overall_Q = mean_over_cells_and_layers([cell["tracks"][track]["Q"] for cell in cells])

    domain_W = np.full((len(domains), N_EXPERTS), np.nan, dtype=np.float64)
    domain_S = np.full((len(domains), N_EXPERTS), np.nan, dtype=np.float64)
    domain_Q = np.full((len(domains), N_EXPERTS), np.nan, dtype=np.float64)
    domain_layer_W = np.full((len(domains), N_LAYERS, N_EXPERTS), np.nan, dtype=np.float64)
    domain_layer_S = np.full((len(domains), N_LAYERS, N_EXPERTS), np.nan, dtype=np.float64)
    domain_layer_Q = np.full((len(domains), N_LAYERS, N_EXPERTS), np.nan, dtype=np.float64)
    subtype_W = np.full((len(SUBTYPES), N_EXPERTS), np.nan, dtype=np.float64)
    subtype_S = np.full((len(SUBTYPES), N_EXPERTS), np.nan, dtype=np.float64)
    subtype_Q = np.full((len(SUBTYPES), N_EXPERTS), np.nan, dtype=np.float64)

    summary_domains: dict[str, Any] = {}
    summary_subtypes: dict[str, Any] = {}
    prompt_level_W_by_domain: dict[str, np.ndarray] = {}
    prompt_level_S_by_domain: dict[str, np.ndarray] = {}

    for domain in domains:
        subset = [cell for cell in cells if cell["domain"] == domain]
        dW = mean_over_cells_and_layers([cell["tracks"][track]["W"] for cell in subset])
        dS = mean_over_cells_and_layers([cell["tracks"][track]["S"] for cell in subset])
        dQ = mean_over_cells_and_layers([cell["tracks"][track]["Q"] for cell in subset])
        dLW = mean_over_cells([cell["tracks"][track]["W"] for cell in subset])
        dLS = mean_over_cells([cell["tracks"][track]["S"] for cell in subset])
        dLQ = mean_over_cells([cell["tracks"][track]["Q"] for cell in subset])
        prompt_level_W_by_domain[domain] = prompt_level_expert_means(subset, track, "W")
        prompt_level_S_by_domain[domain] = prompt_level_expert_means(subset, track, "S")
        idx = domain_index[domain]
        domain_W[idx] = dW
        domain_S[idx] = dS
        domain_Q[idx] = dQ
        domain_layer_W[idx] = dLW
        domain_layer_S[idx] = dLS
        domain_layer_Q[idx] = dLQ
        winner = int(np.nanargmax(dW)) if np.isfinite(dW).any() else None
        summary_domains[domain] = {
            "n_prompts": len(subset),
            "top_experts_by_W": top_experts(dW, dW, dS, dQ, top_k),
            "top_experts_by_S": top_experts(dS, dW, dS, dQ, top_k),
            "winner_by_W": None
            if winner is None
            else {
                "expert": winner,
                "W": float(dW[winner]),
                "S": float(dS[winner]),
                "Q": float(dQ[winner]) if np.isfinite(dQ[winner]) else None,
            },
            "winner_by_S": None
            if not np.isfinite(dS).any()
            else {
                "expert": int(np.nanargmax(dS)),
                "W": float(dW[int(np.nanargmax(dS))]),
                "S": float(dS[int(np.nanargmax(dS))]),
                "Q": float(dQ[int(np.nanargmax(dS))]) if np.isfinite(dQ[int(np.nanargmax(dS))]) else None,
            },
            "highlight_expert": {
                "expert": highlight_expert,
                "W": float(dW[highlight_expert]),
                "S": float(dS[highlight_expert]),
                "Q": float(dQ[highlight_expert]) if np.isfinite(dQ[highlight_expert]) else None,
                "rank_by_W": rank_by_desc(dW, highlight_expert),
                "rank_by_S": rank_by_desc(dS, highlight_expert),
            },
        }

    for domain in domains:
        candidate_rows_W = candidate_scores_for_domain(
            domain=domain,
            metric_label="W",
            domain_prompt_metric=prompt_level_W_by_domain[domain],
            domain_metric=domain_W,
            domain_W=domain_W,
            domain_S=domain_S,
            domain_Q=domain_Q,
            domains=domains,
            top_k=top_k,
        )
        candidate_rows_S = candidate_scores_for_domain(
            domain=domain,
            metric_label="S",
            domain_prompt_metric=prompt_level_S_by_domain[domain],
            domain_metric=domain_S,
            domain_W=domain_W,
            domain_S=domain_S,
            domain_Q=domain_Q,
            domains=domains,
            top_k=top_k,
        )
        summary_domains[domain]["candidate_experts_by_W"] = candidate_rows_W
        summary_domains[domain]["candidate_experts_by_S"] = candidate_rows_S

    for subtype in SUBTYPES:
        subset = [cell for cell in cells if cell["subtype"] == subtype]
        sW = mean_over_cells_and_layers([cell["tracks"][track]["W"] for cell in subset])
        sS = mean_over_cells_and_layers([cell["tracks"][track]["S"] for cell in subset])
        sQ = mean_over_cells_and_layers([cell["tracks"][track]["Q"] for cell in subset])
        idx = subtype_index[subtype]
        subtype_W[idx] = sW
        subtype_S[idx] = sS
        subtype_Q[idx] = sQ
        summary_subtypes[subtype] = {
            "n_prompts": len(subset),
            "top_experts_by_W": top_experts(sW, sW, sS, sQ, top_k),
            "top_experts_by_S": top_experts(sS, sW, sS, sQ, top_k),
        }

    top_sets: dict[str, set[int]] = {}
    for domain in domains:
        top_sets[domain] = {row["expert"] for row in summary_domains[domain]["top_experts_by_W"]}

    jaccard = np.zeros((len(domains), len(domains)), dtype=np.float64)
    for i, left in enumerate(domains):
        for j, right in enumerate(domains):
            union = top_sets[left] | top_sets[right]
            inter = top_sets[left] & top_sets[right]
            jaccard[i, j] = len(inter) / len(union) if union else 0.0

    domain_winners = [summary_domains[domain]["winner_by_W"]["expert"] for domain in domains if summary_domains[domain]["winner_by_W"]]
    winner_counts: dict[int, int] = {}
    for expert in domain_winners:
        winner_counts[expert] = winner_counts.get(expert, 0) + 1

    expert_profiles = []
    for expert in range(N_EXPERTS):
        scores = domain_W[:, expert]
        if not np.isfinite(scores).any():
            continue
        score_vec = np.where(np.isnan(scores), 0.0, scores)
        order = np.argsort(-score_vec)
        top_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else top_idx
        top_score = float(score_vec[top_idx])
        second_score = float(score_vec[second_idx])
        expert_profiles.append(
            {
                "expert": expert,
                "top_domain": domains[top_idx],
                "top_W": top_score,
                "second_domain": domains[second_idx],
                "second_W": second_score,
                "top_vs_second_ratio": None if second_score <= 0 else top_score / second_score,
                "top_share": None if score_vec.sum() <= 0 else top_score / float(score_vec.sum()),
                "domain_entropy": normalized_entropy(score_vec),
                "n_domains_in_top_k": sum(expert in top_sets[domain] for domain in domains),
                "winner_count": winner_counts.get(expert, 0),
            }
        )

    expert_profiles_by_global = sorted(expert_profiles, key=lambda row: row["top_W"], reverse=True)
    expert_profiles_by_selectivity = sorted(
        expert_profiles,
        key=lambda row: (
            -1 if row["top_vs_second_ratio"] is None else -row["top_vs_second_ratio"],
            -row["top_W"],
        ),
    )

    return {
        "overall": {
            "top_experts_by_W": top_experts(overall_W, overall_W, overall_S, overall_Q, top_k),
            "top_experts_by_S": top_experts(overall_S, overall_W, overall_S, overall_Q, top_k),
            "highlight_expert": {
                "expert": highlight_expert,
                "W": float(overall_W[highlight_expert]),
                "S": float(overall_S[highlight_expert]),
                "Q": float(overall_Q[highlight_expert]) if np.isfinite(overall_Q[highlight_expert]) else None,
            },
        },
        "domains": summary_domains,
        "subtypes": summary_subtypes,
        "crossover": {
            "top_k": top_k,
            "domain_top_k_jaccard": {
                "domains": domains,
                "matrix": jaccard.tolist(),
                "top_pairs": top_domain_pairs_from_jaccard(domains, jaccard),
            },
            "winning_expert_counts": [
                {"expert": expert, "domains_won": count}
                for expert, count in sorted(winner_counts.items(), key=lambda item: (-item[1], item[0]))
            ],
        },
        "expert_specialization": {
            "top_by_global_strength": expert_profiles_by_global[: max(top_k, 24)],
            "top_by_selectivity_ratio": expert_profiles_by_selectivity[: max(top_k, 24)],
        },
        "matrices": {
            "domain_W": domain_W,
            "domain_S": domain_S,
            "domain_Q": domain_Q,
            "domain_layer_W": domain_layer_W,
            "domain_layer_S": domain_layer_S,
            "domain_layer_Q": domain_layer_Q,
            "subtype_W": subtype_W,
            "subtype_S": subtype_S,
            "subtype_Q": subtype_Q,
        },
    }


def identity_residual(cells: list[dict[str, Any]], track: str) -> float:
    resid = 0.0
    for cell in cells:
        W = cell["tracks"][track]["W"]
        S = cell["tracks"][track]["S"]
        Q = cell["tracks"][track]["Q"]
        mask = np.isfinite(W) & np.isfinite(S) & np.isfinite(Q) & (S > 0)
        if mask.any():
            resid = max(resid, float(np.max(np.abs(W[mask] - (S[mask] * Q[mask])))))
    return resid


def render_track_md(lines: list[str], track_name: str, summary: dict[str, Any], domains: list[str]) -> None:
    lines.append(f"## {track_name}\n\n")
    lines.append("### Overall top experts by routed weight\n\n")
    lines.append("| Rank | Expert | W | S | Q |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for row in summary["overall"]["top_experts_by_W"]:
        q = "nan" if row["Q"] is None else f"{row['Q']:.6f}"
        lines.append(f"| {row['rank']} | {row['expert']} | {row['W']:.6f} | {row['S']:.6f} | {q} |\n")
    lines.append("\n")

    lines.append("### Domain winners\n\n")
    lines.append("| Domain | Winner expert | W | S | Q | E114 rank |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for domain in domains:
        winner = summary["domains"][domain]["winner_by_W"]
        hi = summary["domains"][domain]["highlight_expert"]
        q = "nan" if winner is None or winner["Q"] is None else f"{winner['Q']:.6f}"
        if winner is None:
            lines.append(f"| {domain} | nan | nan | nan | nan | {hi['rank_by_W']} |\n")
        else:
            lines.append(
                f"| {domain} | {winner['expert']} | {winner['W']:.6f} | {winner['S']:.6f} | {q} | {hi['rank_by_W']} |\n"
            )
    lines.append("\n")

    lines.append("### Candidate specialists by composite score (W)\n\n")
    lines.append(
        "Composite score uses four W-based terms: in-domain mean W, prompt-level consistency across the domain's three prompts, "
        "domain selectivity share across all 20 domains, and separation from the mean of the other 19 domains.\n\n"
    )
    for domain in domains:
        lines.append(f"#### {domain}\n\n")
        lines.append("| Rank | Expert | Domain W | Consistency | Selectivity | Separation | Other-19 mean W | Max gap |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary["domains"][domain]["candidate_experts_by_W"][:5]:
            lines.append(
                f"| {row['rank']} | {row['expert']} | {row['domain_metric']:.6f} | {row['consistency']:.3f} | "
                f"{row['selectivity']:.3f} | {row['separation']:.3f} | {row['other_19_mean_metric']:.6f} | {row['max_gap']:.6f} |\n"
            )
        lines.append("\n")

    lines.append("### Candidate specialists by composite score (S)\n\n")
    lines.append(
        "This mirrors the same framework with selection rate S instead of routed weight W: in-domain mean S, prompt-level consistency in S, "
        "selectivity across all 20 domains, and separation from the mean S of the other 19 domains.\n\n"
    )
    for domain in domains:
        lines.append(f"#### {domain}\n\n")
        lines.append("| Rank | Expert | Domain S | Consistency | Selectivity | Separation | Other-19 mean S | Max gap |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary["domains"][domain]["candidate_experts_by_S"][:5]:
            lines.append(
                f"| {row['rank']} | {row['expert']} | {row['domain_metric']:.6f} | {row['consistency']:.3f} | "
                f"{row['selectivity']:.3f} | {row['separation']:.3f} | {row['other_19_mean_metric']:.6f} | {row['max_gap']:.6f} |\n"
            )
        lines.append("\n")

    lines.append("### Comparison Experts Versus E114 (W candidates)\n\n")
    lines.append(
        "Each domain is shown with Expert 114 plus the top three composite-score candidates, so the domain-leading specialists "
        "can be compared directly against E114 on routed weight, selection rate, conditional weight, and W-rank.\n\n"
    )
    for domain in domains:
        lines.append(f"#### {domain}\n\n")
        lines.append("| Slot | Expert | Composite rank | W rank | W | S | Q | Composite score |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        hi = summary["domains"][domain]["highlight_expert"]
        hi_q = "nan" if hi["Q"] is None else f"{hi['Q']:.6f}"
        lines.append(
            f"| E114 | {hi['expert']} | - | {hi['rank_by_W']} | {hi['W']:.6f} | {hi['S']:.6f} | {hi_q} | - |\n"
        )
        for row in summary["domains"][domain]["candidate_experts_by_W"][:3]:
            q = "nan" if row["Q"] is None else f"{row['Q']:.6f}"
            lines.append(
                f"| Candidate {row['rank']} | {row['expert']} | {row['rank']} | {row['rank_by_W']} | "
                f"{row['W']:.6f} | {row['S']:.6f} | {q} | {row['score']:.8f} |\n"
            )
        lines.append("\n")

    lines.append("### Comparison Experts Versus E114 (S candidates)\n\n")
    lines.append(
        "This parallel view uses the top three S-based composite candidates and compares them directly against E114 on S-rank as well as W/S/Q.\n\n"
    )
    for domain in domains:
        lines.append(f"#### {domain}\n\n")
        lines.append("| Slot | Expert | Composite rank | S rank | W | S | Q | Composite score |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        hi = summary["domains"][domain]["highlight_expert"]
        hi_q = "nan" if hi["Q"] is None else f"{hi['Q']:.6f}"
        lines.append(
            f"| E114 | {hi['expert']} | - | {hi['rank_by_S']} | {hi['W']:.6f} | {hi['S']:.6f} | {hi_q} | - |\n"
        )
        for row in summary["domains"][domain]["candidate_experts_by_S"][:3]:
            q = "nan" if row["Q"] is None else f"{row['Q']:.6f}"
            lines.append(
                f"| Candidate {row['rank']} | {row['expert']} | {row['rank']} | {row['rank_by_S']} | "
                f"{row['W']:.6f} | {row['S']:.6f} | {q} | {row['score']:.8f} |\n"
            )
        lines.append("\n")

    lines.append("### Highest domain-overlap pairs by top-expert Jaccard\n\n")
    lines.append("| Left domain | Right domain | Jaccard overlap |\n")
    lines.append("|---|---|---:|\n")
    for row in summary["crossover"]["domain_top_k_jaccard"]["top_pairs"][:12]:
        lines.append(f"| {row['left']} | {row['right']} | {row['jaccard']:.3f} |\n")
    lines.append("\n")

    lines.append("### Most domain-selective experts\n\n")
    lines.append("| Expert | Top domain | Top W | Second domain | Second W | Top/second | Domains in top-k | Domains won |\n")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|\n")
    for row in summary["expert_specialization"]["top_by_selectivity_ratio"][:16]:
        ratio = "inf" if row["top_vs_second_ratio"] is None else f"{row['top_vs_second_ratio']:.2f}"
        lines.append(
            f"| {row['expert']} | {row['top_domain']} | {row['top_W']:.6f} | "
            f"{row['second_domain']} | {row['second_W']:.6f} | {ratio} | "
            f"{row['n_domains_in_top_k']} | {row['winner_count']} |\n"
        )
    lines.append("\n")


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: to_builtin(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(value) for value in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main() -> None:
    args = parse_args()
    capture_dir = Path(args.capture_dir).resolve()
    prompt_json = Path(args.prompt_json).resolve()
    report_path = Path(args.report).resolve()
    results_json_path = Path(args.results_json).resolve()
    matrices_npz_path = Path(args.matrices_npz).resolve() if args.matrices_npz else report_path.with_suffix(".npz")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_json_path.parent.mkdir(parents=True, exist_ok=True)
    matrices_npz_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_rows, prompt_lookup, domains = load_prompt_spec(prompt_json)
    cells: list[dict[str, Any]] = []
    for entry in sorted(capture_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "router").is_dir():
            continue
        if not (entry / "metadata.txt").exists():
            continue
        cell = process_cell(entry, prompt_lookup)
        if cell is not None:
            cells.append(cell)

    if not cells:
        raise SystemExit(f"No matching capture cells found under {capture_dir}")

    track_summaries = {
        track: build_track_summary(cells, track, domains, args.top_k, args.highlight_expert)
        for track in TRACKS
    }

    lines: list[str] = []
    lines.append(f"# Expert Identification Results — {args.model_label}\n\n")
    lines.append(f"**Capture dir**: `{capture_dir}`\n\n")
    lines.append(f"**Prompt metadata**: `{prompt_json}`\n\n")
    lines.append(f"**Cells processed**: {len(cells)}/{len(prompt_rows)}\n\n")
    lines.append("## Normalization\n\n")
    lines.append(
        "Routing reconstruction follows the HauhauCS/Qwen path used elsewhere in this repo: "
        "softmax over all 256 experts, top-8 selection, then renormalization over the selected experts. "
        "Selection-rate `S`, mean routed weight `W`, and conditional weight `Q` are all computed from that reconstructed top-8 routing.\n\n"
    )
    lines.append("## Prompt inventory\n\n")
    lines.append("| Domain | Mechanism | History | Synthesis |\n")
    lines.append("|---|---|---|---|\n")
    for domain in domains:
        rows = [row for row in prompt_rows if row["domain"] == domain]
        by_subtype = {row["subtype"]: row["id"] for row in rows}
        lines.append(
            f"| {domain} | {by_subtype.get('mechanism', '')} | {by_subtype.get('history', '')} | {by_subtype.get('synthesis', '')} |\n"
        )
    lines.append("\n")
    lines.append("## Capture integrity\n\n")
    lines.append(f"- Layer-39 trim events: `{sum(1 for cell in cells if cell['l39_trimmed'])}`\n")
    lines.append(f"- Missing prefill-layer events: `{sum(len(cell['missing_prefill_layers']) for cell in cells)}`\n")
    lines.append(f"- Missing generation-layer events: `{sum(len(cell['missing_generation_layers']) for cell in cells)}`\n")
    lines.append(
        f"- Identity residual (prefill): `{identity_residual(cells, 'prefill'):.2e}`\n"
    )
    lines.append(
        f"- Identity residual (generation all): `{identity_residual(cells, 'generation_all'):.2e}`\n"
    )
    lines.append(
        f"- Identity residual (generation trimmed): `{identity_residual(cells, 'generation_trimmed'):.2e}`\n\n"
    )

    render_track_md(lines, "Prefill", track_summaries["prefill"], domains)
    render_track_md(lines, "Generation All", track_summaries["generation_all"], domains)
    render_track_md(lines, "Generation Trimmed", track_summaries["generation_trimmed"], domains)
    report_path.write_text("".join(lines))

    json_track_summaries = {
        track: {key: to_builtin(value) for key, value in summary.items() if key != "matrices"}
        for track, summary in track_summaries.items()
    }

    results_json = {
        "capture_dir": str(capture_dir),
        "prompt_json": str(prompt_json),
        "model_label": args.model_label,
        "n_cells": len(cells),
        "n_prompts_expected": len(prompt_rows),
        "domains": domains,
        "subtypes": list(SUBTYPES),
        "normalization": {
            "family": "Qwen/HauhauCS",
            "description": "softmax over 256 experts, top-8, renormalize over selected experts",
            "top_k": TOP_K,
        },
        "integrity": {
            "layer39_trim_events": sum(1 for cell in cells if cell["l39_trimmed"]),
            "missing_prefill_layer_events": sum(len(cell["missing_prefill_layers"]) for cell in cells),
            "missing_generation_layer_events": sum(len(cell["missing_generation_layers"]) for cell in cells),
            "identity_residual": {track: identity_residual(cells, track) for track in TRACKS},
        },
        "cells": [
            {
                "cell_id": cell["cell_id"],
                "domain": cell["domain"],
                "subtype": cell["subtype"],
                "n_tokens_prompt": cell["n_tokens_prompt"],
                "n_tokens_generated": cell["n_tokens_generated"],
                "n_tokens_generation_trimmed": cell["n_tokens_generation_trimmed"],
                "trim_index": cell["trim_index"],
                "l39_trimmed": cell["l39_trimmed"],
            }
            for cell in cells
        ],
        "tracks": json_track_summaries,
    }
    results_json_path.write_text(json.dumps(results_json, indent=2))

    npz_payload: dict[str, Any] = {}
    for track, summary in track_summaries.items():
        matrices = summary["matrices"]
        for key, value in matrices.items():
            npz_payload[f"{track}_{key}"] = value
    npz_payload["domains"] = np.array(domains, dtype=object)
    npz_payload["subtypes"] = np.array(SUBTYPES, dtype=object)
    np.savez_compressed(matrices_npz_path, **npz_payload)


if __name__ == "__main__":
    main()
