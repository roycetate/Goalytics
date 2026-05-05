const API_BASE =
  import.meta.env.VITE_API_BASE ??
  (window.location.port === '5173'
    ? 'http://127.0.0.1:8765'
    : 'http://127.0.0.1:8000')

function asUrl(path) {
  return `${API_BASE}${path}`
}

async function fetchJson(path) {
  const response = await fetch(asUrl(path))
  if (!response.ok) {
    throw new Error(`Request failed (${response.status})`)
  }
  return response.json()
}

export function fetchStats() {
  return fetchJson('/stats')
}

export function fetchVideos() {
  return fetchJson('/videos')
}

export function fetchHeatmaps() {
  return fetchJson('/heatmaps')
}

export function videoUrl(relativePath) {
  return asUrl(relativePath)
}

export function heatmapUrl(relativePath, version) {
  const suffix = version ? `?v=${encodeURIComponent(version)}` : ''
  return asUrl(`${relativePath}${suffix}`)
}

export function analysisFrameUrl(version) {
  const suffix = version ? `?v=${encodeURIComponent(version)}` : ''
  return asUrl(`/analysis-frame${suffix}`)
}

export function analysisStreamUrl() {
  return asUrl('/analysis-stream')
}
