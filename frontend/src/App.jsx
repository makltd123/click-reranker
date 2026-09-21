import { useCallback, useEffect, useState } from 'react'
import {
  evaluate,
  generate,
  getQueries,
  getStatus,
  postClick,
  search,
  simulateClicks,
  trainRanker,
  apiError,
} from './api'

function Chip({ label, value, good }) {
  return (
    <span className={`chip${good === true ? ' chip-good' : good === false ? ' chip-bad' : ''}`}>
      <span className="chip-label">{label}</span> {value}
    </span>
  )
}

function ResultCard({ item, clicked, onClick }) {
  const f = item.features
  return (
    <div
      className={`card${clicked ? ' card-clicked' : ''}`}
      onClick={onClick}
      title={clicked ? 'Click logged' : 'Click to log a real click'}
    >
      <div className="card-head">
        <span className="pos">#{item.position}</span>
        <span className="title">{item.title}</span>
        {clicked && <span className="clicked-tag">clicked</span>}
      </div>
      <div className="snippet">{item.snippet}…</div>
      <div className="chips">
        <Chip label="cos" value={f.cosine.toFixed(3)} />
        <Chip label="bm25" value={f.bm25.toFixed(2)} />
        <Chip label="ann" value={'#' + f.original_rank} />
        <Chip label="len" value={f.doc_length} />
        <Chip label="topic" value={f.topic_match ? 'match' : 'no'} good={!!f.topic_match} />
        {item.rerank_score !== undefined && (
          <Chip label="score" value={item.rerank_score.toFixed(3)} />
        )}
      </div>
    </div>
  )
}

function Column({ heading, subheading, items, clickedSet, onCardClick }) {
  return (
    <section className="column">
      <h2>
        {heading}
        <small>{subheading}</small>
      </h2>
      {!items.length && (
        <div className="empty">No results yet — pick a query and press Search.</div>
      )}
      {items.map((it) => (
        <ResultCard
          key={it.doc_id}
          item={it}
          clicked={clickedSet.has(it.doc_id)}
          onClick={() => onCardClick(it)}
        />
      ))}
    </section>
  )
}

function EvalPanel({ result }) {
  if (!result) return null
  const up = result.uplift >= 0
  return (
    <section className="eval-panel">
      <h2>NDCG@10 on held-out queries ({result.n_eval_queries})</h2>
      <div className="eval-numbers">
        <div>
          <div className="eval-label">Baseline ANN</div>
          <div className="eval-value">{result.ndcg_baseline.toFixed(4)}</div>
        </div>
        <div className={`delta ${up ? 'delta-up' : 'delta-down'}`}>
          {up ? '▲' : '▼'} {Math.abs(result.uplift).toFixed(4)}
        </div>
        <div>
          <div className="eval-label">Re-ranked</div>
          <div className="eval-value">{result.ndcg_reranked.toFixed(4)}</div>
        </div>
      </div>
      {!result.reranker_used && (
        <p className="note">Re-ranker is not trained — reranked equals baseline.</p>
      )}
      <table className="per-query">
        <thead>
          <tr>
            <th>query</th>
            <th>baseline</th>
            <th>reranked</th>
            <th>Δ</th>
          </tr>
        </thead>
        <tbody>
          {result.per_query.map((q) => (
            <tr key={q.query_id}>
              <td className="qtext">{q.query_text}</td>
              <td>{q.ndcg_baseline.toFixed(3)}</td>
              <td>{q.ndcg_reranked.toFixed(3)}</td>
              <td className={q.delta >= 0 ? 'delta-up' : 'delta-down'}>
                {q.delta >= 0 ? '+' : ''}
                {q.delta.toFixed(3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}

export default function App() {
  const [status, setStatus] = useState(null)
  const [queries, setQueries] = useState([])
  const [selectedId, setSelectedId] = useState('')
  const [baseline, setBaseline] = useState([])
  const [reranked, setReranked] = useState([])
  const [searchInfo, setSearchInfo] = useState(null)
  const [clickedSet, setClickedSet] = useState(new Set())
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')
  const [evalResult, setEvalResult] = useState(null)

  const refreshStatus = useCallback(async () => {
    try {
      const { data } = await getStatus()
      setStatus(data)
      return data
    } catch (e) {
      setError('Backend unreachable on http://127.0.0.1:8000 — start uvicorn first.')
      return null
    }
  }, [])

  useEffect(() => {
    ;(async () => {
      const st = await refreshStatus()
      if (st?.generated) {
        const q = await getQueries()
        setQueries(q.data)
      }
    })()
  }, [refreshStatus])

  const runSearch = useCallback(async (qid) => {
    if (!qid) return
    setBusy('search')
    setError('')
    setInfo('')
    try {
      const [b, r] = await Promise.all([search(qid, 'baseline'), search(qid, 'reranked')])
      setBaseline(b.data.items)
      setReranked(r.data.items)
      setSearchInfo(r.data)
      setClickedSet(new Set())
    } catch (e) {
      setError(apiError(e))
    } finally {
      setBusy('')
    }
  }, [])

  const withBusy = useCallback(
    async (label, action, { reloadQueries = false } = {}) => {
      setBusy(label)
      setError('')
      setInfo('')
      setEvalResult(null)
      try {
        const { data } = await action()
        await refreshStatus()
        if (reloadQueries) {
          const q = await getQueries()
          setQueries(q.data)
        }
        if (selectedId && !reloadQueries) await runSearch(selectedId)
        return data
      } catch (e) {
        setError(apiError(e))
        return null
      } finally {
        setBusy('')
      }
    },
    [refreshStatus, runSearch, selectedId],
  )

  const handleGenerate = async () => {
    setSelectedId('')
    const data = await withBusy('generate', generate, { reloadQueries: true })
    if (data) {
      setBaseline([])
      setReranked([])
      setSearchInfo(null)
      setInfo(`Corpus generated: ${data.n_docs} docs, ${data.n_queries} queries in ${data.elapsed_s}s`)
    }
  }

  const handleSimulate = async () => {
    const data = await withBusy('simulate', simulateClicks)
    if (data)
      setInfo(
        `Simulated ${data.impressions} impressions, ${data.clicks} positive clicks (total now ${data.total_clicks})`,
      )
  }

  const handleTrain = async () => {
    const data = await withBusy('train', trainRanker)
    if (data)
      setInfo(
        `Trained ${data.model_type} re-ranker on ${data.n_impressions} impressions from ${data.n_query_groups} queries`,
      )
  }

  const handleEval = async () => {
    setBusy('eval')
    setError('')
    setInfo('')
    try {
      const { data } = await evaluate()
      setEvalResult(data)
    } catch (e) {
      setError(apiError(e))
    } finally {
      setBusy('')
    }
  }

  const onCardClick = async (item) => {
    if (!selectedId || clickedSet.has(item.doc_id)) return
    setClickedSet((prev) => new Set(prev).add(item.doc_id))
    try {
      await postClick({
        query_id: selectedId,
        doc_id: item.doc_id,
        position_shown: item.features.original_rank,
      })
      refreshStatus()
    } catch (e) {
      setError(apiError(e))
    }
  }

  const trainQueries = queries.filter((q) => !q.is_eval)
  const evalQueries = queries.filter((q) => q.is_eval)

  return (
    <div className="app">
      <header>
        <h1>Click-Trained Re-ranker over Vector Search</h1>
        {status && (
          <div className="status-chips">
            <span className={`chip ${status.generated ? '' : 'chip-bad'}`}>
              {status.generated ? `${status.n_docs} docs` : 'not generated'}
            </span>
            <span className="chip">{status.n_queries} queries</span>
            <span className="chip">
              {status.n_clicks} clicks ({status.n_positive_clicks} +)
            </span>
            <span className={`chip ${status.reranker_trained ? 'chip-good' : ''}`}>
              model: {status.reranker_trained ? status.model_type : 'untrained'}
            </span>
          </div>
        )}
      </header>

      {error && (
        <div className="banner banner-error" onClick={() => setError('')}>
          {error}
        </div>
      )}
      {info && (
        <div className="banner banner-info" onClick={() => setInfo('')}>
          {info}
        </div>
      )}

      <div className="toolbar">
        <select value={selectedId} onChange={(e) => setSelectedId(e.target.value)}>
          <option value="">— select a query —</option>
          {!!trainQueries.length && (
            <optgroup label="Train queries (clicks simulated)">
              {trainQueries.map((q) => (
                <option key={q.query_id} value={q.query_id}>
                  [{q.style}] {q.query_text}
                </option>
              ))}
            </optgroup>
          )}
          {!!evalQueries.length && (
            <optgroup label="Held-out eval queries">
              {evalQueries.map((q) => (
                <option key={q.query_id} value={q.query_id}>
                  [{q.style}] {q.query_text}
                </option>
              ))}
            </optgroup>
          )}
        </select>
        <button disabled={!selectedId || !!busy} onClick={() => runSearch(selectedId)}>
          Search
        </button>
        <span className="spacer" />
        <button disabled={!!busy} onClick={handleGenerate}>
          {busy === 'generate' ? 'Generating…' : '1 · Generate dataset'}
        </button>
        <button disabled={!!busy || !status?.generated} onClick={handleSimulate}>
          {busy === 'simulate' ? 'Simulating…' : '2 · Simulate clicks'}
        </button>
        <button disabled={!!busy || !status?.n_positive_clicks} onClick={handleTrain}>
          {busy === 'train' ? 'Training…' : '3 · Train re-ranker'}
        </button>
        <button disabled={!!busy || !status?.generated} onClick={handleEval}>
          {busy === 'eval' ? 'Evaluating…' : 'Evaluate NDCG'}
        </button>
      </div>

      {searchInfo?.note && <div className="note-banner">{searchInfo.note}</div>}
      {!status?.generated && !busy && (
        <div className="note-banner">
          Start with «1 · Generate dataset» — it builds the corpus, embeddings and the noisy ANN index
          (first run downloads the embedding model).
        </div>
      )}

      <main className="columns">
        <Column
          heading="Baseline (raw ANN)"
          subheading="cosine similarity top-10 over noisy embeddings"
          items={baseline}
          clickedSet={clickedSet}
          onCardClick={onCardClick}
        />
        <Column
          heading="Re-ranked (LTR)"
          subheading={`top-${searchInfo?.candidates_considered ?? 50} candidates re-ordered by the click-trained model`}
          items={reranked}
          clickedSet={clickedSet}
          onCardClick={onCardClick}
        />
      </main>

      <EvalPanel result={evalResult} />

      <footer>
        Click any result card in either column to log a real click — it lands in the same{' '}
        <code>clicks</code> table the re-ranker learns from. Then press «3 · Train re-ranker» and
        search again to see the order change.
      </footer>
    </div>
  )
}
