"""FastAPI-приложение прототипа Click-Trained Re-ranker.

Запуск из каталога backend:
    uvicorn app.main:app --port 8000
или
    python -m app.main
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .state import AppState, PipelineError

app = FastAPI(title="Click-Trained Re-ranker", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

state = AppState()


@app.on_event("startup")
def load_artifacts() -> None:
    state.reload_if_exists()


def _fail(err: PipelineError) -> HTTPException:
    return HTTPException(status_code=err.status_code, detail=str(err))


@app.get("/")
def root():
    return {
        "service": "click-trained re-ranker",
        "endpoints": [
            "POST /generate",
            "POST /simulate-clicks",
            "GET /search?query_id=...&mode=baseline|reranked",
            "POST /click",
            "POST /train-reranker",
            "GET /eval",
            "GET /queries",
            "GET /status",
        ],
    }


@app.post("/generate")
def generate():
    try:
        return state.generate()
    except PipelineError as e:
        raise _fail(e)
    except Exception as e:  # например, отсутствие сети при скачивании модели
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


@app.post("/simulate-clicks")
def simulate_clicks():
    try:
        return state.simulate_clicks()
    except PipelineError as e:
        raise _fail(e)


@app.get("/search")
def search(query_id: str, mode: str = "baseline"):
    try:
        return state.search(query_id, mode)
    except PipelineError as e:
        raise _fail(e)


class ClickBody(BaseModel):
    query_id: str
    doc_id: str
    position_shown: int


@app.post("/click")
def click(body: ClickBody):
    try:
        return state.log_click(body.query_id, body.doc_id, body.position_shown)
    except PipelineError as e:
        raise _fail(e)


@app.post("/train-reranker")
def train_reranker():
    try:
        return state.train_reranker()
    except PipelineError as e:
        raise _fail(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


@app.get("/eval")
def evaluate():
    try:
        return state.evaluate()
    except PipelineError as e:
        raise _fail(e)


@app.get("/queries")
def queries():
    if not state.available:
        return []
    return [
        {
            "query_id": q["query_id"],
            "query_text": q["query_text"],
            "topic": q["topic"],
            "style": q["style"],
            "is_eval": bool(q["is_eval"]),
        }
        for q in state.queries
    ]


@app.get("/status")
def status():
    return state.status()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
