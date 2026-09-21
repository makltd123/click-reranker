"""Синтетический корпус: документы, запросы, скрытая истинная релевантность.

Всё детерминировано относительно seed. Контент собирается из шаблонов
на основе тем (topic -> subtopic -> key_terms), чтобы у корпуса было
реальное лексическое и семантическое разнообразие.
"""

import numpy as np

from .config import EVAL_QUERY_FRACTION, N_DOCS_PER_SUBTOPIC

TOPICS = [
    {
        "key": "ml",
        "name": "Machine Learning",
        "subtopics": [
            {"key": "regularization", "name": "Regularization",
             "terms": ["dropout layers", "weight decay", "early stopping", "overfitting",
                        "validation loss", "penalty term"]},
            {"key": "gradient_boosting", "name": "Gradient Boosting",
             "terms": ["gradient boosting", "learning rate schedule", "boosting rounds",
                        "shrinkage", "tree ensemble", "xgboost"]},
            {"key": "embeddings", "name": "Embeddings",
             "terms": ["embedding space", "cosine similarity", "semantic search",
                        "vector representation", "nearest neighbors"]},
            {"key": "model_evaluation", "name": "Model Evaluation",
             "terms": ["cross validation folds", "roc auc", "confusion matrix",
                        "precision recall trade-off", "test set leakage"]},
        ],
    },
    {
        "key": "db",
        "name": "Databases",
        "subtopics": [
            {"key": "sql_tuning", "name": "SQL Tuning",
             "terms": ["execution plan", "composite index", "slow query log",
                        "query planner statistics", "covering index"]},
            {"key": "normalization", "name": "Schema Normalization",
             "terms": ["normal forms", "foreign key constraints", "denormalization trade-offs",
                        "schema migration", "data redundancy"]},
            {"key": "replication", "name": "Replication",
             "terms": ["read replicas", "replication lag", "failover procedure",
                        "leader election", "write amplification"]},
            {"key": "nosql", "name": "NoSQL Stores",
             "terms": ["partition key choice", "eventual consistency", "document store",
                        "wide column layout", "hot partition"]},
        ],
    },
    {
        "key": "web",
        "name": "Web Development",
        "subtopics": [
            {"key": "perf", "name": "Frontend Performance",
             "terms": ["bundle size budget", "code splitting boundaries", "list virtualization",
                        "memoization strategy", "hydration cost"]},
            {"key": "rest", "name": "REST APIs",
             "terms": ["idempotent methods", "pagination cursor", "rate limiting headers",
                        "http status semantics", "versioning strategy"]},
            {"key": "a11y", "name": "Accessibility",
             "terms": ["aria attributes", "focus management", "screen reader testing",
                        "contrast ratio", "keyboard traps"]},
            {"key": "auth", "name": "Authentication",
             "terms": ["refresh token rotation", "pkce flow", "session cookie flags",
                        "oauth scopes", "token storage"]},
        ],
    },
    {
        "key": "sec",
        "name": "Cybersecurity",
        "subtopics": [
            {"key": "xss", "name": "Cross-Site Scripting",
             "terms": ["output escaping", "content security policy", "input sanitization",
                        "dom injection sinks", "trusted types"]},
            {"key": "netsec", "name": "Network Security",
             "terms": ["tls handshake", "cipher suite choice", "certificate pinning",
                        "egress firewall rules", "packet capture"]},
            {"key": "secrets", "name": "Secrets Management",
             "terms": ["secret rotation schedule", "vault access policies",
                        "hardcoded credentials scan", "short lived tokens", "env var hygiene"]},
            {"key": "phishing", "name": "Phishing Defense",
             "terms": ["lookalike domains", "email spoofing headers", "report phishing button",
                        "credential harvesting kits", "awareness drills"]},
        ],
    },
    {
        "key": "cloud",
        "name": "Cloud Computing",
        "subtopics": [
            {"key": "scaling", "name": "Autoscaling",
             "terms": ["target tracking policy", "scale out threshold", "cooldown period",
                        "instance warmup", "capacity buffer"]},
            {"key": "serverless", "name": "Serverless",
             "terms": ["cold start latency", "function timeout budget", "event trigger fanout",
                        "provisioned concurrency", "payload limits"]},
            {"key": "storage", "name": "Object Storage",
             "terms": ["lifecycle tiering rules", "presigned upload urls", "multipart upload parts",
                        "retrieval fees", "bucket inventory"]},
            {"key": "iam", "name": "IAM Policies",
             "terms": ["least privilege policies", "role assumption chains",
                        "wildcard action audit", "permission boundaries", "access analyzer"]},
        ],
    },
    {
        "key": "data",
        "name": "Data Engineering",
        "subtopics": [
            {"key": "etl", "name": "ETL Pipelines",
             "terms": ["idempotent job design", "backfill windows", "watermark columns",
                        "orchestration retries", "late arriving data"]},
            {"key": "streaming", "name": "Stream Processing",
             "terms": ["consumer lag alarms", "exactly once sinks", "event time watermarks",
                        "windowed aggregates", "replay from offset"]},
            {"key": "quality", "name": "Data Quality",
             "terms": ["schema drift detection", "null rate monitors", "data contracts",
                        "quarantine tables", "freshness sla"]},
            {"key": "warehouse", "name": "Warehousing",
             "terms": ["partition pruning", "columnar compression", "star schema grain",
                        "vacuum schedules", "materialized views"]},
        ],
    },
    {
        "key": "mobile",
        "name": "Mobile Development",
        "subtopics": [
            {"key": "android", "name": "Android Performance",
             "terms": ["startup tracing", "baseline profiles", "frame jank budgets",
                        "memory churn", "anr watchdogs"]},
            {"key": "ios", "name": "iOS and SwiftUI",
             "terms": ["view identity", "observable object lifetime", "navigation stack depth",
                        "state restoration", "instrument traces"]},
            {"key": "offline", "name": "Offline Sync",
             "terms": ["sync conflict resolution", "last write wins risks", "tombstone records",
                        "delta sync cursors", "queued mutations"]},
            {"key": "release", "name": "Release Management",
             "terms": ["staged rollout percentages", "crash free sessions", "release checklist gates",
                        "ota update channels", "rollback plans"]},
        ],
    },
    {
        "key": "ops",
        "name": "DevOps",
        "subtopics": [
            {"key": "cicd", "name": "CI/CD",
             "terms": ["pipeline caching layers", "flaky test quarantine", "artifact promotion gates",
                        "merge queues", "hermetic builds"]},
            {"key": "k8s", "name": "Kubernetes",
             "terms": ["pod disruption budgets", "horizontal pod autoscaling", "node affinity rules",
                        "crash loop backoff", "resource requests tuning"]},
            {"key": "obs", "name": "Observability",
             "terms": ["golden signal dashboards", "structured log fields", "trace span sampling",
                        "metric cardinality budgets", "alert fatigue triage"]},
            {"key": "incident", "name": "Incident Response",
             "terms": ["blameless postmortems", "on call escalation paths", "severity rubric",
                        "runbook coverage", "incident command roles"]},
        ],
    },
]

SENTENCE_TEMPLATES = [
    "In practice, {a} matters most when it interacts with {b} under real workloads.",
    "This article breaks down {a} and its relationship to {b} without hand-waving.",
    "We compare approaches to {a}, from quick fixes to structural changes built around {b}.",
    "A short primer on {a}: what it is, how {b} fits in, and where the trade-offs live.",
    "Field notes on {a}, collected from systems where {b} was the actual bottleneck.",
    "If you have ever wondered how {a} affects {b}, this walkthrough keeps the math light.",
    "Common misconceptions about {a} are contrasted with what practitioners observe around {b}.",
    "The guide covers {a} step by step, then connects it to {b} with worked examples.",
    "Readers will learn to reason about {a} and to diagnose problems that involve {b}.",
    "{Sub} is one of those corners of {topic} where small decisions compound, and {a} is a good example.",
    "An opinionated take on {a}, arguing that most teams overinvest in tooling and underinvest in {b}.",
    "From first principles: {a}, {b}, and the gap between tutorials and production reality.",
    "How to instrument, measure, and iterate on {a} without destabilizing {b}.",
    "A case study in treating {a} and {b} as first-class engineering concerns.",
    "What changes when {a} moves from a weekend experiment to something users rely on?",
    "A checklist-driven tour of {a}, with detours into {b} where it genuinely helps.",
    "Notes on the failure modes of {a} and the cheap safeguards that prevent them.",
    "Why senior engineers keep returning to {a} whenever discussions drift away from {b}.",
]

TITLE_TEMPLATES = [
    "{A}: a practical guide",
    "{A} explained with examples",
    "Understanding {a} in {sub}",
    "{A} vs {b}: trade-offs and defaults",
    "Debugging {a} issues in real projects",
    "Getting {a} right in production",
    "{Sub} deep dive: {a} and beyond",
    "Five lessons about {a}",
    "{A}: pitfalls and how to avoid them",
    "When {a} meets {b}",
    "Refactoring around {a}",
    "The minimal viable guide to {a}",
]

# (text, topic_key, subtopic_key, style) — 5 запросов на тему
QUERY_SPECS = [
    ("dropout weight decay early stopping overfitting", "ml", "regularization", "keyword"),
    ("xgboost learning rate boosting rounds shrinkage", "ml", "gradient_boosting", "keyword"),
    ("cosine similarity nearest neighbors embedding space", "ml", "embeddings", "keyword"),
    ("how do i stop my model from memorizing the training data", "ml", "regularization", "semantic"),
    ("picking a metric that does not lie when classes are imbalanced", "ml", "model_evaluation", "semantic"),

    ("composite index execution plan slow query statistics", "db", "sql_tuning", "keyword"),
    ("read replicas replication lag failover procedure", "db", "replication", "keyword"),
    ("partition key hot partition wide column layout", "db", "nosql", "keyword"),
    ("the reporting dashboard slows production every morning where do we start", "db", "sql_tuning", "semantic"),
    ("untangling a giant table into a schema people can reason about", "db", "normalization", "semantic"),

    ("refresh token rotation pkce session cookie flags", "web", "auth", "keyword"),
    ("aria attributes focus management keyboard traps", "web", "a11y", "keyword"),
    ("bundle size code splitting list virtualization hydration", "web", "perf", "keyword"),
    ("make the product fully usable without ever touching a mouse", "web", "a11y", "semantic"),
    ("the feed stutters once it grows past a few hundred rows", "web", "perf", "semantic"),

    ("output escaping content security policy dom injection", "sec", "xss", "keyword"),
    ("tls handshake cipher suites certificate pinning", "sec", "netsec", "keyword"),
    ("secret rotation vault policies hardcoded credentials", "sec", "secrets", "keyword"),
    ("staff received password reset emails we never sent", "sec", "phishing", "semantic"),
    ("api keys should not live in the repository where do they go", "sec", "secrets", "semantic"),

    ("target tracking scale out threshold cooldown period", "cloud", "scaling", "keyword"),
    ("cold start latency provisioned concurrency timeout budget", "cloud", "serverless", "keyword"),
    ("lifecycle tiering presigned urls multipart upload", "cloud", "storage", "keyword"),
    ("every service should hold only the permissions it actually uses", "cloud", "iam", "semantic"),
    ("nobody opens those archived files but storage bills keep growing", "cloud", "storage", "semantic"),

    ("idempotent jobs backfill windows watermark columns", "data", "etl", "keyword"),
    ("consumer lag watermarks exactly once sinks replay offsets", "data", "streaming", "keyword"),
    ("schema drift null rates data contracts quarantine", "data", "quality", "keyword"),
    ("nightly overwrite wiped a day of fresh orders how do we prevent repeats", "data", "etl", "semantic"),
    ("reports sped up once unused columns stopped being read", "data", "warehouse", "semantic"),

    ("startup tracing baseline profiles frame jank memory churn", "mobile", "android", "keyword"),
    ("view identity observable object navigation stack restoration", "mobile", "ios", "keyword"),
    ("sync conflicts last write wins tombstones queued mutations", "mobile", "offline", "keyword"),
    ("users say scrolling feels choppy on older devices", "mobile", "android", "semantic"),
    ("release to one percent first and watch crash dashboards", "mobile", "release", "semantic"),

    ("pipeline caching flaky test quarantine merge queue", "ops", "cicd", "keyword"),
    ("pod disruption budgets horizontal autoscaling resource requests", "ops", "k8s", "keyword"),
    ("golden signals structured logging span sampling cardinality", "ops", "obs", "keyword"),
    ("review the outage without hunting for someone to blame", "ops", "incident", "semantic"),
    ("want dashboards that warn before the pager fires", "ops", "obs", "semantic"),
]


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _fill(template: str, a: str, b: str, sub_name: str, topic_name: str) -> str:
    return (
        template.replace("{a}", a)
        .replace("{b}", b)
        .replace("{A}", _cap(a))
        .replace("{B}", _cap(b))
        .replace("{sub}", sub_name.lower())
        .replace("{Sub}", sub_name)
        .replace("{TOPIC}", topic_name)
        .replace("{topic}", topic_name.lower())
    )


def generate_docs(rng: np.random.Generator) -> list[dict]:
    docs: list[dict] = []
    seen_titles: set[str] = set()
    n_templates = len(SENTENCE_TEMPLATES)

    for topic in TOPICS:
        for sub in topic["subtopics"]:
            terms = sub["terms"]
            for _ in range(N_DOCS_PER_SUBTOPIC):
                for _attempt in range(50):
                    a, b = rng.choice(terms, size=2, replace=False).tolist()
                    title_tpl = TITLE_TEMPLATES[rng.integers(len(TITLE_TEMPLATES))]
                    title = _fill(title_tpl, a, b, sub["name"], topic["name"])
                    if title not in seen_titles:
                        break
                else:
                    title = f"{title} ({len(docs) + 1})"
                seen_titles.add(title)

                order = rng.permutation(n_templates)
                para_sizes = sorted(rng.choice([3, 4, 5], size=3, replace=False).tolist(), reverse=True)
                sentences = []
                start = 0
                for size in para_sizes:
                    chunk = [SENTENCE_TEMPLATES[i] for i in order[start:start + size]]
                    sentences.append(" ".join(_fill(s, a, b, sub["name"], topic["name"]) for s in chunk))
                    start += size
                text = "\n\n".join(sentences[: rng.integers(2, 4)])

                docs.append({
                    "doc_id": f"d{len(docs):04d}",
                    "title": title,
                    "text": text,
                    "topic": topic["key"],
                    "subtopic": sub["key"],
                })
    return docs


def generate_queries(rng: np.random.Generator) -> list[dict]:
    queries = []
    for i, (text, topic_key, sub_key, style) in enumerate(QUERY_SPECS):
        queries.append({
            "query_id": f"q{i:03d}",
            "query_text": text,
            "topic": topic_key,
            "subtopic": sub_key,
            "style": style,
            "is_eval": False,
        })

    flags = [bool(rng.random() < EVAL_QUERY_FRACTION) for _ in queries]
    while sum(flags) < 6:
        idx = int(rng.integers(len(flags)))
        flags[idx] = True
    for q, is_eval in zip(queries, flags):
        q["is_eval"] = is_eval
    return queries


def generate_relevance(
    rng: np.random.Generator,
    docs: list[dict],
    queries: list[dict],
) -> list[tuple[str, str, int]]:
    """Скрытая истинная релевантность 0-3: та же подтема -> 3, та же тема -> 1,
    другая тема -> 0; лёгкий шум сдвигает часть оценок на ±1."""
    rows: list[tuple[str, str, int]] = []
    for q in queries:
        for d in docs:
            if d["subtopic"] == q["subtopic"]:
                grade = 3
            elif d["topic"] == q["topic"]:
                grade = 1
            else:
                grade = 0
            r = rng.random()
            if r < 0.06:
                grade = min(3, grade + 1)
            elif r < 0.14:
                grade = max(0, grade - 1)
            rows.append((q["query_id"], d["doc_id"], grade))
    return rows


def generate(seed: int):
    """Возвращает (docs, queries, relevance_rows), всё воспроизводимо от seed."""
    rng = np.random.default_rng(seed)
    docs = generate_docs(rng)
    queries = generate_queries(rng)
    relevance = generate_relevance(rng, docs, queries)
    return docs, queries, relevance
