#!/usr/bin/env python3
"""Analyze the 122B routing-only domain specialist probe (prefill only)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

N_EXPERTS = 256
N_LAYERS = 48
TOP_K = 8
SUBTYPES = ("mechanism", "history", "synthesis")
SOFTMAX_LAYERS = [layer for layer in range(N_LAYERS) if (layer + 1) % 4 == 0]
DELTANET_LAYERS = [layer for layer in range(N_LAYERS) if layer not in SOFTMAX_LAYERS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze 122B routing-only domain specialist probe.")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--prompt-json", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--highlight-expert", type=int, default=114)
    return parser.parse_args()


def load_router_helpers() -> Any:
    script_dir = Path(__file__).resolve().parent
    qwen_router = script_dir / "qwen_router.py"
    spec = importlib.util.spec_from_file_location("qwen_router", qwen_router)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {qwen_router}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reconstruct_probs


reconstruct_probs = load_router_helpers()


def parse_metadata(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


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
    tracks = empty_track_dict()
    missing_prefill_layers: list[int] = []

    for layer in range(N_LAYERS):
        logits_path = cell_dir / "router" / f"ffn_moe_logits-{layer}.npy"
        if not logits_path.exists():
            missing_prefill_layers.append(layer)
            continue

        arr = np.load(logits_path)
        if arr.ndim != 2 or arr.shape[1] != N_EXPERTS:
            missing_prefill_layers.append(layer)
            continue

        if arr.shape[0] < n_prompt:
            missing_prefill_layers.append(layer)
            continue

        prefill_logits = arr[:n_prompt]
        W, S, Q = compute_metric_vectors(reconstruct_probs(prefill_logits))
        tracks["W"][layer] = W
        tracks["S"][layer] = S
        tracks["Q"][layer] = Q
        tracks["n_tokens_by_layer"][layer] = n_prompt

    return {
        "cell_id": cell_id,
        "domain": spec["domain"],
        "subtype": spec["subtype"],
        "prompt": spec["prompt"],
        "n_tokens_prompt": n_prompt,
        "missing_prefill_layers": missing_prefill_layers,
        "tracks": tracks,
    }


def mean_over_cells_and_layers(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.full((N_EXPERTS,), np.nan, dtype=np.float64)
    stacked = np.stack(arrays, axis=0)
    flat = stacked.reshape(-1, N_EXPERTS)
    return np.nanmean(flat, axis=0)


def mean_over_cells(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.full((N_LAYERS, N_EXPERTS), np.nan, dtype=np.float64)
    stacked = np.stack(arrays, axis=0)
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


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def rank_by_desc(values: np.ndarray, expert: int) -> int | None:
    if not np.isfinite(values).any():
        return None
    order = np.argsort(-np.where(np.isnan(values), -np.inf, values))
    return int(np.where(order == expert)[0][0]) + 1


def prompt_level_expert_means(cells: list[dict[str, Any]], metric: str) -> np.ndarray:
    rows = []
    for cell in cells:
        rows.append(np.nanmean(cell["tracks"][metric], axis=0))
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
        separation = (
            domain_metric_mean / (domain_metric_mean + other_mean)
            if (domain_metric_mean + other_mean) > 0
            else 0.0
        )
        max_gap = domain_metric_mean - (float(np.max(other)) if other.size else 0.0)

        composite = domain_metric_mean * consistency * selectivity * separation
        rows.append(
            {
                "expert": expert,
                "metric": metric_label,
                "domain_metric": domain_metric_mean,
                "consistency": consistency,
                "selectivity": selectivity,
                "other_19_mean_metric": other_mean,
                "separation": separation,
                "max_gap": max_gap,
                "W": float(domain_W[domain_idx, expert]),
                "S": float(domain_S[domain_idx, expert]),
                "Q": float(domain_Q[domain_idx, expert]) if np.isfinite(domain_Q[domain_idx, expert]) else None,
                "rank_by_W": rank_by_desc(domain_W[domain_idx], expert),
                "rank_by_S": rank_by_desc(domain_S[domain_idx], expert),
                "score": composite,
            }
        )

    rows.sort(key=lambda row: (-row["score"], -row["domain_metric"], -row["consistency"], -row["selectivity"]))
    for rank, row in enumerate(rows[:top_k], start=1):
        row["rank"] = rank
    return rows[:top_k]


def normalized_entropy(values: np.ndarray) -> float | None:
    positive = values[np.isfinite(values) & (values > 0)]
    total = float(positive.sum())
    if total <= 0:
        return None
    p = positive / total
    ent = -float(np.sum(p * np.log(p)))
    max_ent = math.log(len(values)) if len(values) > 1 else 1.0
    return ent / max_ent if max_ent > 0 else 0.0


def identity_residual(cells: list[dict[str, Any]]) -> float:
    resid = 0.0
    for cell in cells:
        W = cell["tracks"]["W"]
        S = cell["tracks"]["S"]
        Q = cell["tracks"]["Q"]
        mask = np.isfinite(W) & np.isfinite(S) & np.isfinite(Q) & (S > 0)
        if mask.any():
            resid = max(resid, float(np.max(np.abs(W[mask] - (S[mask] * Q[mask])))))
    return resid


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
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_json_path.parent.mkdir(parents=True, exist_ok=True)

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

    overall_W = mean_over_cells_and_layers([cell["tracks"]["W"] for cell in cells])
    overall_S = mean_over_cells_and_layers([cell["tracks"]["S"] for cell in cells])
    overall_Q = mean_over_cells_and_layers([cell["tracks"]["Q"] for cell in cells])

    delta_W = mean_over_cells_and_layers([cell["tracks"]["W"][DELTANET_LAYERS] for cell in cells])
    delta_S = mean_over_cells_and_layers([cell["tracks"]["S"][DELTANET_LAYERS] for cell in cells])
    delta_Q = mean_over_cells_and_layers([cell["tracks"]["Q"][DELTANET_LAYERS] for cell in cells])
    soft_W = mean_over_cells_and_layers([cell["tracks"]["W"][SOFTMAX_LAYERS] for cell in cells])
    soft_S = mean_over_cells_and_layers([cell["tracks"]["S"][SOFTMAX_LAYERS] for cell in cells])
    soft_Q = mean_over_cells_and_layers([cell["tracks"]["Q"][SOFTMAX_LAYERS] for cell in cells])

    domain_W = np.full((len(domains), N_EXPERTS), np.nan, dtype=np.float64)
    domain_S = np.full((len(domains), N_EXPERTS), np.nan, dtype=np.float64)
    domain_Q = np.full((len(domains), N_EXPERTS), np.nan, dtype=np.float64)
    summary_domains: dict[str, Any] = {}
    prompt_level_W_by_domain: dict[str, np.ndarray] = {}
    prompt_level_S_by_domain: dict[str, np.ndarray] = {}

    for idx, domain in enumerate(domains):
        subset = [cell for cell in cells if cell["domain"] == domain]
        dW = mean_over_cells_and_layers([cell["tracks"]["W"] for cell in subset])
        dS = mean_over_cells_and_layers([cell["tracks"]["S"] for cell in subset])
        dQ = mean_over_cells_and_layers([cell["tracks"]["Q"] for cell in subset])
        domain_W[idx] = dW
        domain_S[idx] = dS
        domain_Q[idx] = dQ
        prompt_level_W_by_domain[domain] = prompt_level_expert_means(subset, "W")
        prompt_level_S_by_domain[domain] = prompt_level_expert_means(subset, "S")
        winner = int(np.nanargmax(dW)) if np.isfinite(dW).any() else None
        summary_domains[domain] = {
            "n_prompts": len(subset),
            "winner_by_W": None if winner is None else {
                "expert": winner,
                "W": float(dW[winner]),
                "S": float(dS[winner]),
                "Q": float(dQ[winner]) if np.isfinite(dQ[winner]) else None,
            },
            "top_experts_by_W": top_experts(dW, dW, dS, dQ, args.top_k),
            "top_experts_by_S": top_experts(dS, dW, dS, dQ, args.top_k),
            "highlight_expert": {
                "expert": args.highlight_expert,
                "W": float(dW[args.highlight_expert]),
                "S": float(dS[args.highlight_expert]),
                "Q": float(dQ[args.highlight_expert]) if np.isfinite(dQ[args.highlight_expert]) else None,
                "rank_by_W": rank_by_desc(dW, args.highlight_expert),
                "rank_by_S": rank_by_desc(dS, args.highlight_expert),
            },
        }

    for domain in domains:
        summary_domains[domain]["candidate_experts_by_W"] = candidate_scores_for_domain(
            domain=domain,
            metric_label="W",
            domain_prompt_metric=prompt_level_W_by_domain[domain],
            domain_metric=domain_W,
            domain_W=domain_W,
            domain_S=domain_S,
            domain_Q=domain_Q,
            domains=domains,
            top_k=args.top_k,
        )
        summary_domains[domain]["candidate_experts_by_S"] = candidate_scores_for_domain(
            domain=domain,
            metric_label="S",
            domain_prompt_metric=prompt_level_S_by_domain[domain],
            domain_metric=domain_S,
            domain_W=domain_W,
            domain_S=domain_S,
            domain_Q=domain_Q,
            domains=domains,
            top_k=args.top_k,
        )

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
                "winner_count": winner_counts.get(expert, 0),
            }
        )
    expert_profiles_by_selectivity = sorted(
        expert_profiles,
        key=lambda row: (-1 if row["top_vs_second_ratio"] is None else -row["top_vs_second_ratio"], -row["top_W"]),
    )

    lines: list[str] = []
    lines.append("# Qwen 122B Domain Specialist Probe — Routing Only\n\n")
    lines.append(f"- Model: `{args.model_label}`\n")
    lines.append("- Mode: `prefill-only / routing-only`\n")
    lines.append("- Routing reconstruction: `softmax_then_topk8_renorm`\n")
    lines.append("- Experts: `256` total, top-`8` selected\n")
    lines.append("- Layers: `48` total, `36` DeltaNet + `12` Softmax\n")
    lines.append(f"- Prompts: `{len(cells)}`\n\n")
    lines.append("## Run Integrity\n\n")
    lines.append(f"- Capture dir: `{capture_dir}`\n")
    lines.append(f"- Prompt metadata: `{prompt_json}`\n")
    lines.append(f"- Identity residual: `{identity_residual(cells):.2e}`\n")
    lines.append(f"- Missing prefill-layer events: `{sum(len(cell['missing_prefill_layers']) for cell in cells)}`\n")
    lines.append(f"- Mean prompt tokens: `{np.mean([cell['n_tokens_prompt'] for cell in cells]):.2f}`\n\n")

    lines.append("## Overall Prefill Routing\n\n")
    lines.append("### Top experts by W\n\n")
    lines.append("| Rank | Expert | W | S | Q |\n")
    lines.append("| ---: | ---: | ---: | ---: | ---: |\n")
    for row in top_experts(overall_W, overall_W, overall_S, overall_Q, args.top_k):
        q = "nan" if row["Q"] is None else f"{row['Q']:.6f}"
        lines.append(f"| {row['rank']} | E{row['expert']} | {row['W']:.6f} | {row['S']:.6f} | {q} |\n")
    lines.append("\n")

    lines.append("### Top experts by S\n\n")
    lines.append("| Rank | Expert | W | S | Q |\n")
    lines.append("| ---: | ---: | ---: | ---: | ---: |\n")
    for row in top_experts(overall_S, overall_W, overall_S, overall_Q, args.top_k):
        q = "nan" if row["Q"] is None else f"{row['Q']:.6f}"
        lines.append(f"| {row['rank']} | E{row['expert']} | {row['W']:.6f} | {row['S']:.6f} | {q} |\n")
    lines.append("\n")

    hiW = overall_W[args.highlight_expert]
    hiS = overall_S[args.highlight_expert]
    hiQ = overall_Q[args.highlight_expert]
    lines.append("### E114 overall position\n\n")
    lines.append(f"- E114 overall `W={hiW:.6f}`, `S={hiS:.6f}`, `Q={hiQ:.6f}`\n")
    lines.append(f"- E114 rank by W: `{rank_by_desc(overall_W, args.highlight_expert)}`\n")
    lines.append(f"- E114 rank by S: `{rank_by_desc(overall_S, args.highlight_expert)}`\n\n")

    lines.append("## DeltaNet vs Softmax Prefill Split\n\n")
    lines.append("### DeltaNet top experts by W\n\n")
    lines.append("| Rank | Expert | W | S | Q |\n")
    lines.append("| ---: | ---: | ---: | ---: | ---: |\n")
    for row in top_experts(delta_W, delta_W, delta_S, delta_Q, args.top_k):
        q = "nan" if row["Q"] is None else f"{row['Q']:.6f}"
        lines.append(f"| {row['rank']} | E{row['expert']} | {row['W']:.6f} | {row['S']:.6f} | {q} |\n")
    lines.append("\n")

    lines.append("### Softmax top experts by W\n\n")
    lines.append("| Rank | Expert | W | S | Q |\n")
    lines.append("| ---: | ---: | ---: | ---: | ---: |\n")
    for row in top_experts(soft_W, soft_W, soft_S, soft_Q, args.top_k):
        q = "nan" if row["Q"] is None else f"{row['Q']:.6f}"
        lines.append(f"| {row['rank']} | E{row['expert']} | {row['W']:.6f} | {row['S']:.6f} | {q} |\n")
    lines.append("\n")

    lines.append("## Domain Winners\n\n")
    lines.append("| Domain | Winner | W | S | Q | E114 rank |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |\n")
    for domain in domains:
        winner = summary_domains[domain]["winner_by_W"]
        hi = summary_domains[domain]["highlight_expert"]
        if winner is None:
            lines.append(f"| {domain} | nan | nan | nan | nan | {hi['rank_by_W']} |\n")
        else:
            q = "nan" if winner["Q"] is None else f"{winner['Q']:.6f}"
            lines.append(f"| {domain} | E{winner['expert']} | {winner['W']:.6f} | {winner['S']:.6f} | {q} | {hi['rank_by_W']} |\n")
    lines.append("\n")

    lines.append("## Candidate Specialists By Composite Score (W)\n\n")
    lines.append("Composite score uses in-domain mean W, prompt-level consistency, domain selectivity across all 20 domains, and separation from the other 19 domains.\n\n")
    for domain in domains:
        lines.append(f"### {domain}\n\n")
        lines.append("| Rank | Expert | Domain W | Consistency | Selectivity | Separation | Other-19 mean W | Max gap |\n")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summary_domains[domain]["candidate_experts_by_W"][:5]:
            lines.append(
                f"| {row['rank']} | E{row['expert']} | {row['domain_metric']:.6f} | {row['consistency']:.3f} | "
                f"{row['selectivity']:.3f} | {row['separation']:.3f} | {row['other_19_mean_metric']:.6f} | {row['max_gap']:.6f} |\n"
            )
        lines.append("\n")

    lines.append("## Most Domain-Selective Experts\n\n")
    lines.append("| Expert | Top domain | Top W | Second domain | Second W | Top/second | Domains won |\n")
    lines.append("| ---: | --- | ---: | --- | ---: | ---: | ---: |\n")
    for row in expert_profiles_by_selectivity[:16]:
        ratio = "inf" if row["top_vs_second_ratio"] is None else f"{row['top_vs_second_ratio']:.2f}"
        lines.append(
            f"| E{row['expert']} | {row['top_domain']} | {row['top_W']:.6f} | "
            f"{row['second_domain']} | {row['second_W']:.6f} | {ratio} | {row['winner_count']} |\n"
        )
    lines.append("\n")

    report_path.write_text("".join(lines))

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
            "missing_prefill_layer_events": sum(len(cell["missing_prefill_layers"]) for cell in cells),
            "identity_residual": identity_residual(cells),
            "mean_prompt_tokens": float(np.mean([cell["n_tokens_prompt"] for cell in cells])),
        },
        "overall": {
            "top_experts_by_W": top_experts(overall_W, overall_W, overall_S, overall_Q, args.top_k),
            "top_experts_by_S": top_experts(overall_S, overall_W, overall_S, overall_Q, args.top_k),
            "highlight_expert": {
                "expert": args.highlight_expert,
                "W": float(hiW),
                "S": float(hiS),
                "Q": float(hiQ) if np.isfinite(hiQ) else None,
                "rank_by_W": rank_by_desc(overall_W, args.highlight_expert),
                "rank_by_S": rank_by_desc(overall_S, args.highlight_expert),
            },
        },
        "prefill_architecture_split": {
            "deltanet_top_by_W": top_experts(delta_W, delta_W, delta_S, delta_Q, args.top_k),
            "softmax_top_by_W": top_experts(soft_W, soft_W, soft_S, soft_Q, args.top_k),
        },
        "domains_summary": summary_domains,
        "expert_specialization": {
            "top_by_selectivity_ratio": expert_profiles_by_selectivity[: max(args.top_k, 24)],
        },
    }
    results_json_path.write_text(json.dumps(to_builtin(results_json), indent=2))


if __name__ == "__main__":
    main()
