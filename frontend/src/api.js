import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000',
})

export const getStatus = () => api.get('/status')
export const getQueries = () => api.get('/queries')
export const generate = () => api.post('/generate', null, { timeout: 600000 })
export const search = (queryId, mode) =>
  api.get('/search', { params: { query_id: queryId, mode } })
export const postClick = (body) => api.post('/click', body)
export const simulateClicks = () => api.post('/simulate-clicks')
export const trainRanker = () => api.post('/train-reranker')
export const evaluate = () => api.get('/eval')

export function apiError(err) {
  return err?.response?.data?.detail || String(err)
}
