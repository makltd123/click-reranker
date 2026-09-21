"""Признаки пары (запрос, документ) для re-ranker.

Колонки матрицы (порядок фиксирован):
    0 cosine         — сырое сходство из шумного ANN-индекса
    1 bm25           — лексическое совпадение (BM25Okapi)
    2 original_rank  — позиция документа в полном ANN-порядке (1-based)
    3 doc_length     — число токенов title+text
    4 topic_match    — 1 если topic запроса == topic документа
"""

import numpy as np
from rank_bm25 import BM25Okapi

from .text_utils import tokenize


class FeatureSpace:
    def __init__(self, docs: list[dict], queries: list[dict], doc_emb: np.ndarray, query_emb: np.ndarray):
        self.docs = docs
        self.doc_index = {d["doc_id"]: i for i, d in enumerate(docs)}
        self.queries = queries
        self.query_index = {q["query_id"]: i for i, q in enumerate(queries)}

        self.doc_emb = doc_emb
        self.query_emb = query_emb

        corpus_tokens = [tokenize(d["title"] + " " + d["text"]) for d in docs]
        self.doc_len = np.array([len(t) for t in corpus_tokens], dtype=np.float32)
        self.bm25 = BM25Okapi(corpus_tokens)

        self._rank_cache: dict[str, dict[str, int]] = {}
        self._bm25_cache: dict[str, np.ndarray] = {}
        self._order_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def ann_order(self, query_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Полный порядок всех документов по cosine для запроса (кэшируется)."""
        if query_id not in self._order_cache:
            qi = self.query_index[query_id]
            sims = self.doc_emb @ self.query_emb[qi]
            self._order_cache[query_id] = (np.argsort(-sims), sims)
        return self._order_cache[query_id]

    def ranks_for(self, query_id: str) -> dict[str, int]:
        if query_id not in self._rank_cache:
            order, _ = self.ann_order(query_id)
            self._rank_cache[query_id] = {
                self.docs[i]["doc_id"]: pos + 1 for pos, i in enumerate(order)
            }
        return self._rank_cache[query_id]

    def candidates(self, query_id: str, depth: int) -> list[tuple[str, float]]:
        """Топ-depth кандидатов из ANN в порядке убывания cosine."""
        order, sims = self.ann_order(query_id)
        idxs = order[:depth]
        return [(self.docs[i]["doc_id"], float(sims[i])) for i in idxs]

    def bm25_scores(self, query_id: str) -> np.ndarray:
        if query_id not in self._bm25_cache:
            q = self.queries[self.query_index[query_id]]
            self._bm25_cache[query_id] = np.asarray(
                self.bm25.get_scores(tokenize(q["query_text"])), dtype=np.float32
            )
        return self._bm25_cache[query_id]

    def row_features(self, query_id: str, doc_ids: list[str]) -> np.ndarray:
        ranks = self.ranks_for(query_id)
        q = self.queries[self.query_index[query_id]]
        bm25_vec = self.bm25_scores(query_id)

        idxs = [self.doc_index[d] for d in doc_ids]
        cos = self.doc_emb[idxs] @ self.query_emb[self.query_index[query_id]]
        rows = np.column_stack([
            cos.astype(np.float32),
            bm25_vec[idxs],
            np.array([ranks[d] for d in doc_ids], dtype=np.float32),
            self.doc_len[idxs],
            np.array(
                [1.0 if self.docs[i]["topic"] == q["topic"] else 0.0 for i in idxs],
                dtype=np.float32,
            ),
        ])
        return rows
