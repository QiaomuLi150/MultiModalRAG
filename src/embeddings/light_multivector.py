from __future__ import annotations

import math

import numpy as np


def compress_multivector(
    vectors: np.ndarray,
    mode: str = "none",
    target_tokens: int | None = None,
) -> np.ndarray:
    array = np.asarray(vectors, dtype="float32")
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D multivector array, got shape={array.shape!r}")

    if mode == "none" or target_tokens is None or target_tokens <= 0 or len(array) <= target_tokens:
        return array
    if mode == "mean_pool":
        return _mean_pool_tokens(array, target_tokens)
    if mode == "similarity_merge":
        return _similarity_merge_tokens(array, target_tokens)
    raise ValueError(f"Unsupported multivector compression mode: {mode}")


def _mean_pool_tokens(vectors: np.ndarray, target_tokens: int) -> np.ndarray:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if len(vectors) <= target_tokens:
        return vectors

    group_size = math.ceil(len(vectors) / target_tokens)
    pooled = []
    for start in range(0, len(vectors), group_size):
        batch = vectors[start : start + group_size]
        pooled.append(batch.mean(axis=0))
    return np.stack(pooled, axis=0).astype("float32")


def _similarity_merge_tokens(vectors: np.ndarray, target_tokens: int) -> np.ndarray:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if len(vectors) <= target_tokens:
        return vectors

    groups = [vectors[index : index + 1].copy() for index in range(len(vectors))]
    group_weights = [1 for _ in range(len(vectors))]

    while len(groups) > target_tokens:
        merged_index = _best_adjacent_merge_index(groups)
        left = groups[merged_index]
        right = groups[merged_index + 1]
        left_weight = group_weights[merged_index]
        right_weight = group_weights[merged_index + 1]
        merged = ((left.mean(axis=0) * left_weight) + (right.mean(axis=0) * right_weight)) / (left_weight + right_weight)
        groups[merged_index] = merged.reshape(1, -1).astype("float32")
        group_weights[merged_index] = left_weight + right_weight
        del groups[merged_index + 1]
        del group_weights[merged_index + 1]

    return np.concatenate(groups, axis=0).astype("float32")


def _best_adjacent_merge_index(groups: list[np.ndarray]) -> int:
    best_index = 0
    best_score = -float("inf")
    for index in range(len(groups) - 1):
        left = groups[index].mean(axis=0)
        right = groups[index + 1].mean(axis=0)
        score = _cosine_similarity(left, right)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return float(np.dot(left, right) / (left_norm * right_norm))
