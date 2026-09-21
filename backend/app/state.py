"""Состояние приложения: корпус, эмбеддинги, FeatureSpace, обученный re-ranker.

Все артефакты живут в backend/data: SQLite + два .npy + reranker.pkl.
"""

import time
from pathlib import Path

import joblib
import numpy as np

from . import ann, db, data_gen, eval_ndcg, ranker, simulate
from .config import (
    DOC_EMB_PATH,
    QUERY_EMB_PATH,
    RETRIEVE_DEPTH,
    RERANKER_PATH,
    SEED,
    SIMULATION_SEED,
    TOP_K,
)
from .features import FeatureSpace


class PipelineError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class AppState:
    def __init__(self) -> None:
        self.available = False
        self.docs: list[dict] = []
        self.doc_by_id: dict[str, dict] = {}
        self.queries: list[dict] = []
        self.relevance: dict[tuple[str, str], int] = {}
        self.fs: FeatureSpace | None = None
        self.bundle: dict | None = None

    # ---------- загрузка / генерация ----------

    def reload_if_exists(self) -> bool:
        if not (db.DB_PATH.exists() and DOC_EMB_PATH.exists() and QUERY_EMB_PATH.exists()):
            self.available = False
            return False
        try:
            self._load()
            self.available = True
            return True
        except Exception:
            self.available = False
            return False

    def _load(self) -> None:
        db.init_db()
        self.docs = db.list_documents()
        if not self.docs:
            raise RuntimeError("corpus is empty")
        self.doc_by_id = {d["doc_id"]: d for d in self.docs}
        self.queries = db.list_queries()
        self.relevance = {
            (r["query_id"], r["doc_id"]): r["grade"] for r in db.list_relevance()
        }
        doc_emb = np.load(DOC_EMB_PATH)
        query_emb = np.load(QUERY_EMB_PATH)
        self.fs = FeatureSpace(self.docs, self.queries, doc_emb, query_emb)
        self.bundle = None
        if Path(RERANKER_PATH).exists():
            self.bundle = joblib.load(RERANKER_PATH)

    def generate(self) -> dict:
        started = time.perf_counter()

        for path in (db.DB_PATH, DOC_EMB_PATH, QUERY_EMB_PATH, RERANKER_PATH):
            Path(path).unlink(missing_ok=True)

        docs, queries, relevance_rows = data_gen.generate(SEED)

        from . import embedder

        doc_texts = [f'{d["title"]}\n{d["text"]}' for d in docs]
        query_texts = [q["query_text"] for q in queries]
        doc_emb = embedder.encode(doc_texts)
        query_emb = embedder.encode(query_texts)

        rng = np.random.default_rng(SEED)
        doc_emb_noisy = ann.add_noise(doc_emb, rng)

        db.init_db()
        db.replace_corpus(docs, queries, relevance_rows)
        np.save(DOC_EMB_PATH, doc_emb_noisy)
        np.save(QUERY_EMB_PATH, query_emb)

        self._load()
        self.available = True
        return {
            "n_docs": len(docs),
            "n_queries": len(queries),
            "n_eval_queries": sum(1 for q in queries if q["is_eval"]),
            "embedding_model": embedder.EMBEDDING_MODEL,
            "elapsed_s": round(time.perf_counter() - started, 1),
        }

    # ---------- вспомогательное ----------

    def _require_available(self) -> None:
        if not self.available or self.fs is None:
            raise PipelineError("Corpus is not generated yet — call POST /generate first", 409)

    def get_query(self, query_id: str) -> dict:
        for q in self.queries:
            if q["query_id"] == query_id:
                return q
        raise PipelineError(f"Unknown query_id: {query_id}", 404)

    # ---------- операции демо ----------

    def search(self, query_id: str, mode: str) -> dict:
        self._require_available()
        if mode not in ("baseline", "reranked"):
            raise PipelineError("mode must be 'baseline' or 'reranked'")
        q = self.get_query(query_id)

        depth = TOP_K if mode == "baseline" else RETRIEVE_DEPTH
        cands = self.fs.candidates(query_id, depth)
        doc_ids = [d for d, _ in cands]
        X = self.fs.row_features(query_id, doc_ids)

        scores = None
        fallback = False
        if mode == "reranked" and self.bundle is not None:
            scores = ranker.predict(self.bundle, X)
        elif mode == "reranked":
            fallback = True  # модель ещё не обучена — порядок совпадёт с baseline

        if scores is not None:
            order = np.argsort(-scores)[:TOP_K]
        else:
            order = np.arange(len(doc_ids))[:TOP_K]

        items = []
        for pos, i in enumerate(order.tolist(), start=1):
            doc = self.doc_by_id[doc_ids[i]]
            feats = X[i]
            item = {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "snippet": doc["text"][:200],
                "topic": doc["topic"],
                "subtopic": doc["subtopic"],
                "position": pos,
                "features": {
                    "cosine": round(float(feats[0]), 3),
                    "bm25": round(float(feats[1]), 2),
                    "original_rank": int(feats[2]),
                    "doc_length": int(feats[3]),
                    "topic_match": int(feats[4]),
                },
            }
            if scores is not None:
                item["rerank_score"] = round(float(scores[i]), 4)
            items.append(item)

        resp = {
            "query_id": query_id,
            "query_text": q["query_text"],
            "query_topic": q["topic"],
            "mode": mode,
            "candidates_considered": len(doc_ids),
            "reranker_trained": self.bundle is not None,
            "items": items,
        }
        if fallback:
            resp["note"] = "Re-ranker is not trained yet — order equals baseline. Train it first."
        return resp

    def log_click(self, query_id: str, doc_id: str, position_shown: int) -> dict:
        self._require_available()
        self.get_query(query_id)
        if doc_id not in self.doc_by_id:
            raise PipelineError(f"Unknown doc_id: {doc_id}", 404)
        # Проверяем, что документ был показан в топ-10 ANN для этого запроса
        cands = self.fs.candidates(query_id, TOP_K)
        shown_ids = {d for d, _ in cands}
        if doc_id not in shown_ids:
            raise PipelineError(
                f"Document {doc_id} was not in the top-{TOP_K} ANN results for query {query_id}",
                400,
            )
        # Используем ANN-позицию, а не позицию в отображении (после re-ranking они могут отличаться)
        ann_position = next(pos for pos, (d, _) in enumerate(cands, start=1) if d == doc_id)
        click_id = db.add_click(query_id, doc_id, ann_position, 1, "manual")
        return {"click_id": click_id}

    def simulate_clicks(self) -> dict:
        self._require_available()
        db.clear_clicks()
        total_impressions = 0
        total_clicks = 0
        # Повторяем симуляцию 3 раза, чтобы накопить достаточно данных для обучения.
        # Генератор создаётся один раз: результат детерминирован, но проходы разные.
        rng = np.random.default_rng(SIMULATION_SEED)
        for _ in range(3):
            stats = simulate.simulate(self.fs, self.relevance, rng)
            total_impressions += stats["impressions"]
            total_clicks += stats["clicks"]
        total, positives = db.count_clicks()
        return {
            "impressions": total_impressions,
            "clicks": total_clicks,
            "total_clicks": total,
            "positive_clicks": positives,
        }

    def train_reranker(self) -> dict:
        self._require_available()
        clicks = db.list_clicks()
        if not clicks:
            raise PipelineError("No clicks logged yet — run POST /simulate-clicks or click results in the UI")

        positives = sum(c["clicked"] for c in clicks)
        if positives == 0:
            raise PipelineError("Click log contains no positive clicks — nothing to learn from")

        groups_by_query: dict[str, list[dict]] = {}
        for c in clicks:
            groups_by_query.setdefault(c["query_id"], []).append(c)

        X_blocks, y_blocks, group_sizes = [], [], []
        for qid, rows in groups_by_query.items():
            doc_ids = [r["doc_id"] for r in rows]
            try:
                X_blocks.append(self.fs.row_features(qid, doc_ids))
            except KeyError:
                continue  # клики по документам из старого корпуса игнорируем
            y_blocks.append(np.array([r["clicked"] for r in rows], dtype=np.int32))
            group_sizes.append(len(rows))

        X = np.vstack(X_blocks)
        y = np.concatenate(y_blocks)

        bundle = ranker.train(X, y, group_sizes)
        joblib.dump(bundle, RERANKER_PATH)
        self.bundle = bundle

        return {
            "model_type": bundle["model_type"],
            "n_impressions": int(len(clicks)),
            "n_positive_clicks": int(positives),
            "n_query_groups": len(group_sizes),
            "n_features": X.shape[1],
        }

    def evaluate(self) -> dict:
        self._require_available()
        result = eval_ndcg.evaluate(self.fs, self.relevance, self.bundle)
        if self.bundle is None:
            result["note"] = "Re-ranker is not trained — reranked column equals baseline."
        return result

    def status(self) -> dict:
        generated = self.available and self.fs is not None
        total, positives = db.count_clicks() if db.DB_PATH.exists() else (0, 0)
        n_queries = len(self.queries) if generated else 0
        return {
            "generated": generated,
            "n_docs": len(self.docs) if generated else 0,
            "n_queries": n_queries,
            "n_eval_queries": sum(1 for q in self.queries if q["is_eval"]) if generated else 0,
            "n_clicks": total,
            "n_positive_clicks": positives,
            "reranker_trained": self.bundle is not None,
            "model_type": self.bundle["model_type"] if self.bundle else None,
        }


state = AppState()
