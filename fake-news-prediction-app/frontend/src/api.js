const API_BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

async function readJson(response) {
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const message = typeof payload.detail === 'string' ? payload.detail : 'The request could not be completed.'
    throw new Error(message)
  }
  return payload
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE}/api/health`)
  return readJson(response)
}

export async function getModelMetrics() {
  const response = await fetch(`${API_BASE}/api/metrics`)
  return readJson(response)
}

export async function predictNews(payload) {
  const response = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return readJson(response)
}
