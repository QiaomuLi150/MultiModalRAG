from __future__ import annotations

import numpy as np


def build_proxy_vector(multivector: np.ndarray) -> np.ndarray:
    vectors = np.asarray(multivector, dtype="float32")
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D multivector, got shape={vectors.shape!r}")
    pooled = vectors.mean(axis=0)
    return _normalize(pooled)


def exact_maxsim_score(query_multivector: np.ndarray, doc_multivector: np.ndarray) -> float:
    query = _normalize_rows(np.asarray(query_multivector, dtype="float32"))
    doc = _normalize_rows(np.asarray(doc_multivector, dtype="float32"))
    similarities = query @ doc.T
    return float(similarities.max(axis=1).sum())


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype("float32")
    return (vector / norm).astype("float32")

