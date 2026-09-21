"""Обёртка над sentence-transformers. Модель грузится лениво,
чтобы приложение стартовало быстро без корпуса."""

from .config import EMBEDDING_MODEL

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def encode(texts: list[str]) -> "np.ndarray":
    import numpy as np

    emb = get_model().encode(
        texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    )
    return np.asarray(emb, dtype=np.float32)
