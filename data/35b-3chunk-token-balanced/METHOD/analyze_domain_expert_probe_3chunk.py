#!/usr/bin/env python3
"""Analyze the balanced 3-chunk domain probe capture for prefill vs generation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


N_EXPERTS = 256
N_LAYERS = 40
IM_END_TOKEN_SEQUENCE = [27, 91, 316, 6018, 91, 29]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze 3-chunk domain probe routing metrics.")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--prompt-json", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--highlight-expert", type=int, default=114)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def load_router_helpers() -> tuple[Any, Any]:
    script_dir = Path(__file__).resolve().parent
    qwen_router = script_dir / "qwen_router.py"
    spec = importlib.util.spec_from_file_location("qwen_router", qwen_router)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {qwen_router}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reconstruct_probs, module.normalized_entropy


reconstruct_probs, normalized_entropy = load_router_helpers()


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


def rank_of(expert: int, values: np.ndarray) -> int | None:
    score = np.where(np.isnan(values), -np.inf, values)
    order = np.argsort(-score)
    for idx, candidate in enumerate(order, start=1):
        if int(candidate) == expert:
            if not np.isfinite(score[candidate]):
                return None
            return idx
    return None


def top_experts(values: np.ndarray, W: np.ndarray, S: np.ndarray, Q: np.ndarray, limit: int) -> list[dict[str, Any]]:
    score = np.where(np.isnan(values), -np.inf, values)
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


def jaccard_of_topk(left: np.ndarray, right: np.ndarray, k: int) -> float:
    left_score = np.where(np.isnan(left), -np.inf, left)
    right_score = np.where(np.isnan(right), -np.inf, right)
    left_set = {int(x) for x in np.argsort(-left_score)[:k] if np.isfinite(left_score[x])}
    right_set = {int(x) for x in np.argsort(-right_score)[:k] if np.isfinite(right_score[x])}
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def summarize_track(layers: list[dict[str, np.ndarray]], highlight_expert: int, top_k: int) -> dict[str, Any]:
    if not layers:
        return {
            "entropy_mean": None,
            "entropy_by_layer": [],
            "pooled_W": [],
            "pooled_S": [],
            "pooled_Q": [],
            "top_by_W": [],
            "top_by_S": [],
            "highlight": None,
        }

    entropy_layers = np.array([row["entropy_mean"] for row in layers], dtype=np.float64)
    W_layers = np.stack([row["W"] for row in layers], axis=0)
    S_layers = np.stack([row["S"] for row in layers], axis=0)
    Q_layers = np.stack([row["Q"] for row in layers], axis=0)

    pooled_W = np.nanmean(W_layers, axis=0)
    pooled_S = np.nanmean(S_layers, axis=0)
    pooled_Q = np.nanmean(Q_layers, axis=0)

    best_layer_by_W = int(np.nanargmax(W_layers[:, highlight_expert]))
    best_layer_by_S = int(np.nanargmax(S_layers[:, highlight_expert]))

    return {
        "entropy_mean": float(np.nanmean(entropy_layers)),
        "entropy_by_layer": [float(x) for x in entropy_layers],
        "pooled_W": pooled_W,
        "pooled_S": pooled_S,
        "pooled_Q": pooled_Q,
        "top_by_W": top_experts(pooled_W, pooled_W, pooled_S, pooled_Q, top_k),
        "top_by_S": top_experts(pooled_S, pooled_W, pooled_S, pooled_Q, top_k),
        "highlight": {
            "expert": highlight_expert,
            "W": float(pooled_W[highlight_expert]),
            "S": float(pooled_S[highlight_expert]),
            "Q": float(pooled_Q[highlight_expert]) if np.isfinite(pooled_Q[highlight_expert]) else None,
            "rank_by_W": rank_of(highlight_expert, pooled_W),
            "rank_by_S": rank_of(highlight_expert, pooled_S),
            "best_layer_by_W": best_layer_by_W,
            "best_layer_W": float(W_layers[best_layer_by_W, highlight_expert]),
            "best_layer_by_S": best_layer_by_S,
            "best_layer_S": float(S_layers[best_layer_by_S, highlight_expert]),
        },
    }


def analyze_cell(cell_dir: Path, prompt_row: dict[str, Any], highlight_expert: int, top_k: int) -> dict[str, Any]:
    md = parse_metadata(cell_dir / "metadata.txt")
    n_prompt = int(md["n_tokens_prompt"])
    n_gen = int(md["n_tokens_generated"])
    token_ids = load_generated_token_ids(cell_dir)
    if len(token_ids) != n_gen:
        n_gen = min(n_gen, len(token_ids))
        token_ids = token_ids[:n_gen]
    trim_idx = find_im_end_index(token_ids)
    n_gen_trim = trim_idx if trim_idx is not None else n_gen

    prefill_layers: list[dict[str, np.ndarray]] = []
    generation_all_layers: list[dict[str, np.ndarray]] = []
    generation_trimmed_layers: list[dict[str, np.ndarray]] = []

    for layer in range(N_LAYERS):
        logits_path = cell_dir / "router" / f"ffn_moe_logits-{layer}.npy"
        if not logits_path.exists():
            continue
        arr = np.load(logits_path)
        if arr.ndim != 2 or arr.shape[1] != N_EXPERTS:
            continue

        if arr.shape[0] >= n_prompt:
            prefill_logits = arr[:n_prompt]
            prefill_probs = reconstruct_probs(prefill_logits)
            W, S, Q = compute_metric_vectors(prefill_probs)
            prefill_layers.append(
                {
                    "layer": layer,
                    "entropy_mean": float(np.mean(normalized_entropy(prefill_probs))),
                    "W": W,
                    "S": S,
                    "Q": Q,
                }
            )

        expected_rows = n_prompt + n_gen
        if arr.shape[0] == expected_rows:
            gen_logits = arr[n_prompt : n_prompt + n_gen]
        elif arr.shape[0] == n_gen + 1:
            gen_logits = arr[1:]
        else:
            continue

        if gen_logits.shape[0] != n_gen:
            continue

        gen_probs = reconstruct_probs(gen_logits)
        Wg, Sg, Qg = compute_metric_vectors(gen_probs)
        generation_all_layers.append(
            {
                "layer": layer,
                "entropy_mean": float(np.mean(normalized_entropy(gen_probs))),
                "W": Wg,
                "S": Sg,
                "Q": Qg,
            }
        )

        if n_gen_trim > 0:
            trim_probs = reconstruct_probs(gen_logits[:n_gen_trim])
            Wt, St, Qt = compute_metric_vectors(trim_probs)
            generation_trimmed_layers.append(
                {
                    "layer": layer,
                    "entropy_mean": float(np.mean(normalized_entropy(trim_probs))),
                    "W": Wt,
                    "S": St,
                    "Q": Qt,
                }
            )

    prefill = summarize_track(prefill_layers, highlight_expert, top_k)
    generation_all = summarize_track(generation_all_layers, highlight_expert, top_k)
    generation_trimmed = summarize_track(generation_trimmed_layers, highlight_expert, top_k)

    return {
        "id": cell_dir.name,
        "chunk": prompt_row["chunk"],
        "slot": prompt_row["slot"],
        "n_tokens_prompt": n_prompt,
        "n_tokens_generated": n_gen,
        "n_tokens_generation_trimmed": n_gen_trim,
        "padding_count": prompt_row.get("padding_count"),
        "source_domains": prompt_row["source_domains"],
        "prefill": prefill,
        "generation_all": generation_all,
        "generation_trimmed": generation_trimmed,
        "comparisons": {
            "entropy_delta_generation_trimmed_minus_prefill": (
                generation_trimmed["entropy_mean"] - prefill["entropy_mean"]
                if generation_trimmed["entropy_mean"] is not None and prefill["entropy_mean"] is not None
                else None
            ),
            "topW_jaccard_prefill_vs_generation_trimmed": jaccard_of_topk(
                prefill["pooled_W"], generation_trimmed["pooled_W"], top_k
            ),
            "topS_jaccard_prefill_vs_generation_trimmed": jaccard_of_topk(
                prefill["pooled_S"], generation_trimmed["pooled_S"], top_k
            ),
        },
    }


def json_ready(summary: dict[str, Any]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(summary, default=lambda x: None))
    for chunk in cleaned["chunks"]:
        for track_name in ("prefill", "generation_all", "generation_trimmed"):
            for key in ("pooled_W", "pooled_S", "pooled_Q"):
                chunk[track_name].pop(key, None)
    return cleaned


def main() -> None:
    args = parse_args()
    capture_dir = Path(args.capture_dir)
    prompt_rows = json.loads(Path(args.prompt_json).read_text())
    prompt_lookup = {row["id"]: row for row in prompt_rows}

    chunk_summaries = []
    for row in prompt_rows:
        cell_dir = capture_dir / row["id"]
        chunk_summaries.append(analyze_cell(cell_dir, row, args.highlight_expert, args.top_k))

    overall = {
        "prefill_entropy_mean": float(np.mean([row["prefill"]["entropy_mean"] for row in chunk_summaries])),
        "generation_trimmed_entropy_mean": float(
            np.mean([row["generation_trimmed"]["entropy_mean"] for row in chunk_summaries])
        ),
    }

    summary = {
        "model_label": args.model_label,
        "capture_dir": str(capture_dir),
        "highlight_expert": args.highlight_expert,
        "top_k": args.top_k,
        "overall": overall,
        "chunks": chunk_summaries,
    }

    report_lines = [
        f"# Domain Expert Probe 3-Chunk Analysis",
        "",
        f"- Model: `{args.model_label}`",
        f"- Capture dir: `{capture_dir}`",
        f"- Highlight expert: `{args.highlight_expert}`",
        "",
        f"- Mean prefill entropy: `{overall['prefill_entropy_mean']:.6f}`",
        f"- Mean generation-trimmed entropy: `{overall['generation_trimmed_entropy_mean']:.6f}`",
        "",
    ]

    for row in chunk_summaries:
        report_lines.extend(
            [
                f"## {row['id']}",
                "",
                f"- Prompt tokens: `{row['n_tokens_prompt']}`",
                f"- Generated tokens: `{row['n_tokens_generated']}`",
                f"- Trimmed generated tokens: `{row['n_tokens_generation_trimmed']}`",
                f"- Prefill entropy mean: `{row['prefill']['entropy_mean']:.6f}`",
                f"- Generation-trimmed entropy mean: `{row['generation_trimmed']['entropy_mean']:.6f}`",
                f"- Entropy delta (gen_trim - prefill): `{row['comparisons']['entropy_delta_generation_trimmed_minus_prefill']:.6f}`",
                f"- Top-W Jaccard prefill vs generation-trimmed: `{row['comparisons']['topW_jaccard_prefill_vs_generation_trimmed']:.6f}`",
                f"- Top-S Jaccard prefill vs generation-trimmed: `{row['comparisons']['topS_jaccard_prefill_vs_generation_trimmed']:.6f}`",
                f"- E{args.highlight_expert} prefill W/S/Q: `{row['prefill']['highlight']['W']:.6f}` / `{row['prefill']['highlight']['S']:.6f}` / `{row['prefill']['highlight']['Q']:.6f}`",
                f"- E{args.highlight_expert} generation-trimmed W/S/Q: `{row['generation_trimmed']['highlight']['W']:.6f}` / `{row['generation_trimmed']['highlight']['S']:.6f}` / `{row['generation_trimmed']['highlight']['Q']:.6f}`",
                f"- E{args.highlight_expert} prefill rank by W/S: `{row['prefill']['highlight']['rank_by_W']}` / `{row['prefill']['highlight']['rank_by_S']}`",
                f"- E{args.highlight_expert} generation-trimmed rank by W/S: `{row['generation_trimmed']['highlight']['rank_by_W']}` / `{row['generation_trimmed']['highlight']['rank_by_S']}`",
                "",
                "Top generation-trimmed experts by W:",
            ]
        )
        for top_row in row["generation_trimmed"]["top_by_W"]:
            report_lines.append(
                f"- `#{top_row['rank']}` E{top_row['expert']}: W `{top_row['W']:.6f}`, S `{top_row['S']:.6f}`, Q `{top_row['Q']:.6f}`"
            )
        report_lines.append("")
        report_lines.append("Top generation-trimmed experts by S:")
        for top_row in row["generation_trimmed"]["top_by_S"]:
            report_lines.append(
                f"- `#{top_row['rank']}` E{top_row['expert']}: W `{top_row['W']:.6f}`, S `{top_row['S']:.6f}`, Q `{top_row['Q']:.6f}`"
            )
        report_lines.append("")

    Path(args.report).write_text("\n".join(report_lines) + "\n")
    Path(args.results_json).write_text(json.dumps(json_ready(summary), indent=2) + "\n")


if __name__ == "__main__":
    main()
