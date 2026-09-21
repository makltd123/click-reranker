"""Оценка качества: NDCG@10 на отложенных запросах, baseline ANN vs re-ranked."""

import math

import numpy as np

from . import ranker as ranker_mod
from .config import RETRIEVE_DEPTH, TOP_K


def ndcg_at_k(grades: list[int], k: int = TOP_K) -> float:
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))
    ideal = sorted(grades, reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def rerank_ids(fs, bundle: dict, query_id: str, depth: int = RETRIEVE_DEPTH) -> list[str]:
    cands = fs.candidates(query_id, depth)
    doc_ids = [d for d, _ in cands]
    X = fs.row_features(query_id, doc_ids)
    scores = ranker_mod.predict(bundle, X)
    order = np.argsort(-scores)
    return [doc_ids[i] for i in order]


def evaluate(fs, relevance: dict[tuple[str, str], int], bundle: dict | None) -> dict:
    eval_queries = [q for q in fs.queries if q["is_eval"]]
    per_query = []
    base_scores, rr_scores = [], []

    for q in eval_queries:
        qid = q["query_id"]
        base_ids = [d for d, _ in fs.candidates(qid, TOP_K)]
        rr_ids = rerank_ids(fs, bundle, qid)[:TOP_K] if bundle else base_ids

        def g(ids):
            return [relevance.get((qid, d), 0) for d in ids]

        b = ndcg_at_k(g(base_ids))
        r = ndcg_at_k(g(rr_ids))
        base_scores.append(b)
        rr_scores.append(r)
        per_query.append({
            "query_id": qid,
            "query_text": q["query_text"],
            "ndcg_baseline": round(b, 4),
            "ndcg_reranked": round(r, 4),
            "delta": round(r - b, 4),
        })

    mean_base = sum(base_scores) / len(base_scores) if base_scores else 0.0
    mean_rr = sum(rr_scores) / len(rr_scores) if rr_scores else 0.0
    return {
        "n_eval_queries": len(eval_queries),
        "ndcg_baseline": round(mean_base, 4),
        "ndcg_reranked": round(mean_rr, 4),
        "uplift": round(mean_rr - mean_base, 4),
        "reranker_used": bundle is not None,
        "per_query": sorted(per_query, key=lambda x: x["delta"]),
    }
