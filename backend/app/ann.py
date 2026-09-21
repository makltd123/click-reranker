"""Baseline ANN: чистый numpy, косинусное сходство по нормированным векторам.

Шум в эмбеддингах документов делает выдачу неидеальной относительно
скрытой релевантности — ровно то, что re-ranker должен чинить.
"""

import numpy as np

from .config import NOISE_EXTRA_SIGMA, NOISE_SIGMA, NOISE_SUBSET_FRACTION


def add_noise(embeddings: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n, dim = embeddings.shape
    noise = rng.normal(0.0, NOISE_SIGMA, size=(n, dim)).astype(np.float32)
    subset = rng.random(n) < NOISE_SUBSET_FRACTION
    if subset.any():
        noise[subset] += rng.normal(
            0.0, NOISE_EXTRA_SIGMA, size=(int(subset.sum()), dim)
        ).astype(np.float32)
    noisy = embeddings + noise
    norms = np.linalg.norm(noisy, axis=1, keepdims=True)
    return (noisy / np.clip(norms, 1e-12, None)).astype(np.float32)


def top_k(query_emb: np.ndarray, doc_embs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (индексы документов по убыванию cosine, сами cosine)."""
    sims = doc_embs @ query_emb
    order = np.argsort(-sims)
    return order[:k], sims
