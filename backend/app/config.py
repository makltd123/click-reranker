from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "app.db"
DOC_EMB_PATH = DATA_DIR / "doc_embeddings.npy"
QUERY_EMB_PATH = DATA_DIR / "query_embeddings.npy"
RERANKER_PATH = DATA_DIR / "reranker.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 10           # результаты, отдаваемые клиенту
RETRIEVE_DEPTH = 50  # глубина выборки кандидатов из ANN перед ре-ранжингом

# Контролируемый шум в эмбеддингах документов: делает baseline ANN неидеальным
NOISE_SIGMA = 0.35         # глобальный шум
NOISE_EXTRA_SIGMA = 0.55   # дополнительный шум для подмножества документов
NOISE_SUBSET_FRACTION = 0.25

SEED = 42                  # фиксированный сид корпуса/шума (воспроизводимость)
SIMULATION_SEED = 12345    # сид симуляции кликов — детерминированный результат для демонстрации

N_DOCS_PER_SUBTOPIC = 9    # 8 тем x 4 подтемы x 9 доков = 288 документов

CLICK_PROB_BY_GRADE = [0.12, 0.35, 0.62, 0.92]  # P(клик | скрытая релевантность 0..3)
EVAL_QUERY_FRACTION = 0.25                       # доля отложенных (held-out) запросов

FEATURE_NAMES = ["cosine", "bm25", "original_rank", "doc_length", "topic_match"]
