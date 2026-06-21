#!/usr/bin/env python3
"""Per-token all-expert breakdown for the balanced 3-chunk domain probe run."""

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
    parser = argparse.ArgumentParser(description="Export per-token all-expert breakdowns for 3-chunk Huahua run.")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--out-dir", required=True)
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


def load_generated_tokens(cell_dir: Path) -> list[dict[str, Any]]:
    return json.loads((cell_dir / "generated_tokens.json").read_text())


def find_im_end_index(token_ids: list[int]) -> int | None:
    seq_len = len(IM_END_TOKEN_SEQUENCE)
    for i in range(len(token_ids) - seq_len + 1):
        if token_ids[i : i + seq_len] == IM_END_TOKEN_SEQUENCE:
            return i
    return None


def rank_of(expert: int, values: np.ndarray) -> int | None:
    score = np.where(np.isnan(values), -np.inf, values)
    order = np.argsort(-score)
    for idx, candidate in enumerate(order, start=1):
        if int(candidate) == expert:
            if not np.isfinite(score[candidate]):
                return None
            return idx
    return None


def build_phase_tensor_stack(cell_dir: Path, n_prompt: int, n_gen: int) -> tuple[np.ndarray, np.ndarray]:
    prefill_layers: list[np.ndarray] = []
    generation_layers: list[np.ndarray] = []

    for layer in range(N_LAYERS):
        logits_path = cell_dir / "router" / f"ffn_moe_logits-{layer}.npy"
        if not logits_path.exists():
            continue
        arr = np.load(logits_path)
        if arr.ndim != 2 or arr.shape[1] != N_EXPERTS:
            continue

        if arr.shape[0] >= n_prompt:
            prefill_layers.append(reconstruct_probs(arr[:n_prompt]))

        expected_rows = n_prompt + n_gen
        if arr.shape[0] == expected_rows:
            gen_logits = arr[n_prompt : n_prompt + n_gen]
        elif arr.shape[0] == n_gen + 1:
            gen_logits = arr[1:]
        else:
            continue

        if gen_logits.shape[0] == n_gen:
            generation_layers.append(reconstruct_probs(gen_logits))

    if not prefill_layers:
        raise RuntimeError(f"No usable prefill layers found in {cell_dir}")
    if not generation_layers:
        raise RuntimeError(f"No usable generation layers found in {cell_dir}")

    return np.stack(prefill_layers, axis=0), np.stack(generation_layers, axis=0)


def reduce_phase(layer_probs: np.ndarray) -> dict[str, np.ndarray]:
    selected = layer_probs > 0
    counts = selected.sum(axis=0).astype(np.float64)
    sums = layer_probs.sum(axis=0).astype(np.float64)
    mean_W = np.mean(layer_probs, axis=0)
    mean_S = np.mean(selected, axis=0)
    mean_entropy = np.mean(normalized_entropy(layer_probs), axis=0)
    mean_Q = np.full_like(mean_W, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=mean_Q, where=counts > 0)
    return {
        "mean_W": mean_W,
        "mean_S": mean_S,
        "mean_Q": mean_Q,
        "mean_entropy": mean_entropy,
    }


def top_rows(mean_W: np.ndarray, mean_S: np.ndarray, mean_Q: np.ndarray, limit: int) -> list[dict[str, Any]]:
    order = np.argsort(-mean_W)[:limit]
    rows = []
    for expert in order:
        rows.append(
            {
                "expert": int(expert),
                "W": float(mean_W[expert]),
                "S": float(mean_S[expert]),
                "Q": float(mean_Q[expert]) if np.isfinite(mean_Q[expert]) else None,
            }
        )
    return rows


def write_token_tsv(
    out_path: Path,
    prefill: dict[str, np.ndarray],
    generation: dict[str, np.ndarray],
    generated_tokens: list[dict[str, Any]],
    highlight_expert: int,
    top_k: int,
) -> None:
    header = [
        "phase",
        "token_index_in_phase",
        "global_token_index",
        "token_id",
        "token_piece",
        "entropy_mean",
        f"E{highlight_expert}_W",
        f"E{highlight_expert}_S",
        f"E{highlight_expert}_Q",
        f"E{highlight_expert}_rank_by_W",
        f"E{highlight_expert}_rank_by_S",
    ]
    for idx in range(1, top_k + 1):
        header.extend([f"top{idx}_expert", f"top{idx}_W", f"top{idx}_S", f"top{idx}_Q"])

    lines = ["\t".join(header)]

    def append_rows(phase: str, token_meta: list[tuple[int, str]], offset: int, reduced: dict[str, np.ndarray]) -> None:
        mean_W = reduced["mean_W"]
        mean_S = reduced["mean_S"]
        mean_Q = reduced["mean_Q"]
        mean_entropy = reduced["mean_entropy"]
        for token_idx in range(mean_W.shape[0]):
            token_id, token_piece = token_meta[token_idx]
            row = [
                phase,
                str(token_idx),
                str(offset + token_idx),
                "" if token_id < 0 else str(token_id),
                token_piece.replace("\t", "\\t").replace("\n", "\\n"),
                f"{float(mean_entropy[token_idx]):.8f}",
                f"{float(mean_W[token_idx, highlight_expert]):.8f}",
                f"{float(mean_S[token_idx, highlight_expert]):.8f}",
                "" if not np.isfinite(mean_Q[token_idx, highlight_expert]) else f"{float(mean_Q[token_idx, highlight_expert]):.8f}",
                "" if rank_of(highlight_expert, mean_W[token_idx]) is None else str(rank_of(highlight_expert, mean_W[token_idx])),
                "" if rank_of(highlight_expert, mean_S[token_idx]) is None else str(rank_of(highlight_expert, mean_S[token_idx])),
            ]
            top = top_rows(mean_W[token_idx], mean_S[token_idx], mean_Q[token_idx], top_k)
            for top_row in top:
                row.extend(
                    [
                        str(top_row["expert"]),
                        f"{top_row['W']:.8f}",
                        f"{top_row['S']:.8f}",
                        "" if top_row["Q"] is None else f"{top_row['Q']:.8f}",
                    ]
                )
            if len(top) < top_k:
                row.extend([""] * ((top_k - len(top)) * 4))
            lines.append("\t".join(row))

    n_prompt = prefill["mean_W"].shape[0]
    prefill_meta = [(-1, "") for _ in range(n_prompt)]
    generation_meta = [
        (int(tok["token_id"]), str(tok.get("piece", "")))
        for tok in generated_tokens[: generation["mean_W"].shape[0]]
    ]

    append_rows("prefill", prefill_meta, 0, prefill)
    append_rows("generation", generation_meta, n_prompt, generation)
    out_path.write_text("\n".join(lines) + "\n")


def write_npz(
    out_path: Path,
    prefill: dict[str, np.ndarray],
    generation: dict[str, np.ndarray],
    generated_tokens: list[dict[str, Any]],
    n_prompt: int,
    n_gen_trim: int,
) -> None:
    gen_ids = np.array([int(tok["token_id"]) for tok in generated_tokens[:n_gen_trim]], dtype=np.int32)
    gen_pieces = np.array([str(tok.get("piece", "")) for tok in generated_tokens[:n_gen_trim]], dtype="<U128")
    np.savez_compressed(
        out_path,
        prefill_mean_W=prefill["mean_W"],
        prefill_mean_S=prefill["mean_S"],
        prefill_mean_Q=prefill["mean_Q"],
        prefill_mean_entropy=prefill["mean_entropy"],
        generation_mean_W=generation["mean_W"],
        generation_mean_S=generation["mean_S"],
        generation_mean_Q=generation["mean_Q"],
        generation_mean_entropy=generation["mean_entropy"],
        generation_token_ids=gen_ids,
        generation_token_pieces=gen_pieces,
        n_prompt=np.array([n_prompt], dtype=np.int32),
        n_generation=np.array([n_gen_trim], dtype=np.int32),
    )


def main() -> None:
    args = parse_args()
    capture_dir = Path(args.capture_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        "# Per-Token All-Expert Breakdown",
        "",
        f"- Capture dir: `{capture_dir}`",
        f"- Highlight expert: `E{args.highlight_expert}`",
        f"- Export format: dense `.npz` all-expert matrices plus readable token-level `.tsv`",
        "",
    ]

    for cell_dir in sorted(p for p in capture_dir.iterdir() if p.is_dir()):
        md = parse_metadata(cell_dir / "metadata.txt")
        n_prompt = int(md["n_tokens_prompt"])
        n_gen = int(md["n_tokens_generated"])
        generated_tokens = load_generated_tokens(cell_dir)
        token_ids = [int(tok["token_id"]) for tok in generated_tokens]
        if len(generated_tokens) != n_gen:
            n_gen = min(n_gen, len(generated_tokens))
            generated_tokens = generated_tokens[:n_gen]
            token_ids = token_ids[:n_gen]

        trim_idx = find_im_end_index(token_ids)
        n_gen_trim = trim_idx if trim_idx is not None else n_gen

        prefill_stack, generation_stack = build_phase_tensor_stack(cell_dir, n_prompt, n_gen)
        prefill = reduce_phase(prefill_stack)
        generation = reduce_phase(generation_stack[:, :n_gen_trim, :])

        stem = f"{capture_dir.name}_{cell_dir.name}_per_token"
        npz_path = out_dir / f"{stem}.npz"
        tsv_path = out_dir / f"{stem}.tsv"
        write_npz(npz_path, prefill, generation, generated_tokens, n_prompt, n_gen_trim)
        write_token_tsv(tsv_path, prefill, generation, generated_tokens, args.highlight_expert, args.top_k)

        highlight_prefill_rank = rank_of(args.highlight_expert, np.nanmean(prefill["mean_W"], axis=0))
        highlight_gen_rank = rank_of(args.highlight_expert, np.nanmean(generation["mean_W"], axis=0))
        summary_lines.extend(
            [
                f"## {cell_dir.name}",
                "",
                f"- Prompt tokens: `{n_prompt}`",
                f"- Generation tokens exported: `{n_gen_trim}`",
                f"- Dense matrix: `{npz_path.name}`",
                f"- Readable TSV: `{tsv_path.name}`",
                f"- Mean prefill E{args.highlight_expert} W: `{float(np.nanmean(prefill['mean_W'][:, args.highlight_expert])):.6f}`",
                f"- Mean generation E{args.highlight_expert} W: `{float(np.nanmean(generation['mean_W'][:, args.highlight_expert])):.6f}`",
                f"- E{args.highlight_expert} rank by pooled token-mean W in prefill: `{highlight_prefill_rank}`",
                f"- E{args.highlight_expert} rank by pooled token-mean W in generation: `{highlight_gen_rank}`",
                "",
            ]
        )

    (out_dir / f"{capture_dir.name}_per_token_summary.md").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
