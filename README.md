# Click-Trained Re-ranker over Vector Search

**English** | [Русский](README.ru.md)

A prototype: plain vector search (ANN by cosine similarity) plus a lightweight learning-to-rank layer (LightGBM lambdarank) that keeps learning from accumulated clicks and re-ranks the results.

## The idea

Typical vector search is static: it returns the top-K by cosine similarity, and that similarity never learns from user behavior. Here clicks are logged on top of it, and a small model trained on them reorders the ANN candidates. The ANN itself is not replaced.

The model does not look at the embeddings. It looks at features of the (query, document) pair: cosine similarity, BM25, original position in the results, document length and topic match. That makes it visible why a document moved up or down.

There are no real clicks, so both the corpus and the user behavior are synthetic. Noise is deliberately added to the document embeddings so the original ranking is imperfect and the re-ranker has something to fix.

## Structure

```
backend/
  requirements.txt
  app/
    main.py        FastAPI: /generate, /simulate-clicks, /search, /click,
                   /train-reranker, /eval, /queries, /status
    config.py      constants (embedding noise, K, seeds)
    data_gen.py    synthetic corpus: 288 documents (8 topics × 4 subtopics),
                   40 queries, hidden relevance 0-3
    embedder.py    sentence-transformers (all-MiniLM-L6-v2)
    ann.py         embedding noise + numpy ANN (cosine top-K)
    features.py    (query, document) features: cosine, bm25,
                   original_rank, doc_length, topic_match
    ranker.py      LGBMRanker (lambdarank), logistic regression fallback
    simulate.py    click simulation: P(click | relevance) / sqrt(position)
    eval_ndcg.py   NDCG@10, baseline vs reranked, on held-out queries
    state.py       orchestration, artifacts in backend/data/
    db.py          SQLite: documents, queries, relevance, clicks
frontend/          Vite + React: one screen, two columns side by side
```

## Running it

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

The first `POST /generate` call downloads the embedding model (~90 MB) — a one-time step.

### Frontend

```bash
cd frontend
npm install
npm run dev                     # http://localhost:5173
```

## Demo walkthrough

1. **1 · Generate dataset** — corpus, queries, noisy embeddings, ANN index.
2. Pick a query → **Search** — both columns show the features of every document.
3. **2 · Simulate clicks** — clicks on training queries with position bias; the ANN is deliberately imperfect (noise pushes some relevant documents down).
4. **3 · Train re-ranker** — the LGBMRanker learns from the accumulated clicks.
5. **Search** again — the order in the second column has changed, and the features explain why.
6. **Evaluate NDCG** — NDCG@10 on held-out queries: baseline vs reranked.

Manual clicks on the cards go into the same `clicks` table — after them you can retrain the model and see the order change.

## Notes

- Artifacts live in `backend/data/`: `app.db`, `doc_embeddings.npy`, `query_embeddings.npy`, `reranker.pkl`. `POST /generate` rebuilds everything, clicks included.
- The corpus and the noise are seeded (`SEED=42`), and so is the click simulation (`SIMULATION_SEED`). Every press of **Simulate clicks** clears the clicks table — **manual clicks too** — and refills it with three simulation passes, so the result is reproducible.
- Held-out queries (~25%) are flagged `is_eval`: the simulation never clicks on them, and quality is measured on them only.
- Re-ranking takes 50 candidates from the ANN and returns the best 10, so it can pull up documents that the noise pushed out of the top 10.
- The ranker falls back to logistic regression if `lightgbm` is not installed, or if there are clicks for fewer than 5 queries (LightGBM cannot train on 1–2 groups). The model type is shown in the status.
- A manual click is accepted only for documents in the ANN top 10 and is recorded with the document's position in the original ANN results, not in the re-ranked column.

## License

[MIT](LICENSE)
