import sqlite3
from contextlib import closing
from typing import Any

from .config import DATA_DIR, DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id   TEXT PRIMARY KEY,
    title    TEXT NOT NULL,
    text     TEXT NOT NULL,
    topic    TEXT NOT NULL,
    subtopic TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS queries (
    query_id   TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    topic      TEXT NOT NULL,
    subtopic   TEXT NOT NULL,
    style      TEXT NOT NULL,
    is_eval    INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS relevance (
    query_id TEXT NOT NULL,
    doc_id   TEXT NOT NULL,
    grade    INTEGER NOT NULL,
    PRIMARY KEY (query_id, doc_id)
);
CREATE TABLE IF NOT EXISTS clicks (
    click_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id       TEXT NOT NULL,
    doc_id         TEXT NOT NULL,
    position_shown INTEGER NOT NULL,
    clicked        INTEGER NOT NULL,
    source         TEXT NOT NULL DEFAULT 'manual',
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with closing(connect()) as c:
        c.executescript(SCHEMA)
        c.commit()


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(r) for r in cur.fetchall()]


def replace_corpus(
    docs: list[dict],
    queries: list[dict],
    relevance: list[tuple[str, str, int]],
) -> None:
    with closing(connect()) as c:
        c.execute("DELETE FROM relevance")
        c.execute("DELETE FROM queries")
        c.execute("DELETE FROM documents")
        c.executemany(
            "INSERT INTO documents (doc_id, title, text, topic, subtopic) VALUES (?, ?, ?, ?, ?)",
            [(d["doc_id"], d["title"], d["text"], d["topic"], d["subtopic"]) for d in docs],
        )
        c.executemany(
            "INSERT INTO queries (query_id, query_text, topic, subtopic, style, is_eval)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                (q["query_id"], q["query_text"], q["topic"], q["subtopic"], q["style"], int(q["is_eval"]))
                for q in queries
            ],
        )
        c.executemany(
            "INSERT INTO relevance (query_id, doc_id, grade) VALUES (?, ?, ?)", relevance
        )
        c.commit()


def clear_clicks() -> None:
    with closing(connect()) as c:
        c.execute("DELETE FROM clicks")
        c.commit()


def add_click(query_id: str, doc_id: str, position_shown: int, clicked: int, source: str) -> int:
    with closing(connect()) as c:
        cur = c.execute(
            "INSERT INTO clicks (query_id, doc_id, position_shown, clicked, source)"
            " VALUES (?, ?, ?, ?, ?)",
            (query_id, doc_id, position_shown, clicked, source),
        )
        c.commit()
        return int(cur.lastrowid)


def insert_clicks(rows: list[tuple[str, str, int, int, str]]) -> int:
    with closing(connect()) as c:
        c.executemany(
            "INSERT INTO clicks (query_id, doc_id, position_shown, clicked, source)"
            " VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        c.commit()
        return len(rows)


def list_clicks() -> list[dict[str, Any]]:
    with closing(connect()) as c:
        return _rows(c.execute("SELECT * FROM clicks ORDER BY click_id"))


def list_queries() -> list[dict[str, Any]]:
    with closing(connect()) as c:
        return _rows(c.execute("SELECT * FROM queries ORDER BY query_id"))


def list_documents() -> list[dict[str, Any]]:
    with closing(connect()) as c:
        return _rows(c.execute("SELECT * FROM documents ORDER BY doc_id"))


def list_relevance() -> list[dict[str, Any]]:
    with closing(connect()) as c:
        return _rows(c.execute("SELECT query_id, doc_id, grade FROM relevance"))


def get_document(doc_id: str) -> dict[str, Any] | None:
    with closing(connect()) as c:
        rows = _rows(c.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)))
        return rows[0] if rows else None


def count_clicks() -> tuple[int, int]:
    with closing(connect()) as c:
        row = c.execute(
            "SELECT COUNT(*) AS total, COALESCE(SUM(clicked), 0) AS positives FROM clicks"
        ).fetchone()
        return int(row["total"]), int(row["positives"])
