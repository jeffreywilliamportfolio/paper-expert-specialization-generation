#!/usr/bin/env python3
"""Routing reconstruction helpers for Qwen3.5-35B-A3B routed experts."""
from __future__ import annotations

import numpy as np

N_EXPERTS = 256
TOP_K = 8
ENTROPY_MAX = np.log2(TOP_K)
RECONSTRUCTION_NAME = "softmax_then_topk8_renorm"


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_full_probs(logits: np.ndarray) -> np.ndarray:
    return softmax(logits, axis=-1)


def reconstruct_probs(logits: np.ndarray) -> np.ndarray:
    squeeze = logits.ndim == 1
    if squeeze:
        logits = logits[np.newaxis, :]

    logits = np.asarray(logits, dtype=np.float64)
    dense_probs = softmax_full_probs(logits)
    n_tokens, n_experts = dense_probs.shape
    k = min(TOP_K, n_experts)

    topk_indices = np.argpartition(dense_probs, -k, axis=-1)[:, -k:]
    rows = np.arange(n_tokens)[:, None]
    topk_probs = dense_probs[rows, topk_indices]
    topk_probs /= np.sum(topk_probs, axis=-1, keepdims=True)

    probs = np.zeros_like(dense_probs, dtype=np.float64)
    probs[rows, topk_indices] = topk_probs
    if squeeze:
        probs = probs[0]
    return probs


def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    return -np.sum(probs * np.log2(probs + 1e-30), axis=-1) / ENTROPY_MAX


def probability_from_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return np.full_like(counts, 1.0 / max(len(counts), 1), dtype=np.float64)
    return counts / total


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = probability_from_counts(p)
    q = probability_from_counts(q)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2((p + 1e-30) / (m + 1e-30))) +
                 0.5 * np.sum(q * np.log2((q + 1e-30) / (m + 1e-30))))
