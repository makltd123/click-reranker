"""Симуляция пользовательских кликов.

Для каждого обучающего запроса берём top-10 baseline ANN (это то, что
«показано пользователю»); клик сэмплируется из P(клик | истинная
релевантность) x position bias 1/sqrt(позиция). Логируются ВСЕ
показы (impression + clicked flag) — это даёт отрицательные примеры
для обучения re-ranker.
"""

import math

import numpy as np

from . import db
from .config import CLICK_PROB_BY_GRADE, TOP_K


def simulate(fs, relevance: dict[tuple[str, str], int], rng: np.random.Generator) -> dict:
    rows: list[tuple[str, str, int, int, str]] = []
    n_clicked = 0

    for q in fs.queries:
        if q["is_eval"]:
            continue  # отложенные запросы в симуляции не участвуют
        cands = fs.candidates(q["query_id"], TOP_K)
        for position, (doc_id, _cos) in enumerate(cands, start=1):
            grade = relevance.get((q["query_id"], doc_id), 0)
            p_click = CLICK_PROB_BY_GRADE[grade] / math.sqrt(position)
            clicked = 1 if rng.random() < p_click else 0
            n_clicked += clicked
            rows.append((q["query_id"], doc_id, position, clicked, "simulated"))

    inserted = db.insert_clicks(rows)
    return {"impressions": inserted, "clicks": n_clicked}
