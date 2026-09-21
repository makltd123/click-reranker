"""Re-ranker: LightGBM LGBMRanker (lambdarank) поверх накопленных кликов.

Если lightgbm не установлен — автоматический фолбэк на логистическую
регрессию (clicked vs not clicked). Бандл сохраняется через joblib.
При недостатке данных (мало query groups) принудительно используется logreg,
так как LightGBM не обучится на 1–2 группах.
"""

import numpy as np

SEED = 42

# Минимальное число query groups, при котором ещё имеет смысл LightGBM
MIN_GROUPS_FOR_LGBM = 5


def train(X: np.ndarray, y: np.ndarray, group: list[int]) -> dict:
    n_groups = len(group)
    n_pos = int(y.sum())

    if n_groups < MIN_GROUPS_FOR_LGBM:
        return _train_logreg(X, y)

    try:
        from lightgbm import LGBMRanker

        model = LGBMRanker(
            objective="lambdarank",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=5,
            random_state=SEED,
            verbose=-1,
        )
        model.fit(X, y, group=group)
        return {"model_type": "lgbm", "model": model}
    except ImportError:
        return _train_logreg(X, y)


def _train_logreg(X: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(X, y)
    return {"model_type": "logreg", "model": clf}


def predict(bundle: dict, X: np.ndarray) -> np.ndarray:
    if bundle["model_type"] == "lgbm":
        return np.asarray(bundle["model"].predict(X), dtype=np.float32)
    return bundle["model"].predict_proba(X)[:, 1].astype(np.float32)
