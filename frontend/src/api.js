const API_BASE =
  import.meta.env.VITE_API_BASE ??
  (window.location.port === '5173'
    ? 'http://127.0.0.1:8765'
    : 'http://127.0.0.1:8000')

function asUrl(path) {
  return `${API_BASE}${path}`
}

async function parseError(response) {
  try {
    const data = await response.json()
    if (typeof data?.detail === 'string') return data.detail
    if (Array.isArray(data?.detail)) {
      return data.detail.map((item) => item.msg || JSON.stringify(item)).join(', ')
    }
  } catch {
    // Fall back to status text when the error body is not JSON.
  }
  return `Request failed (${response.status})`
}

async function fetchJson(path, options) {
  const response = await fetch(asUrl(path), options)
  if (!response.ok) {
    throw new Error(await parseError(response))
  }
  if (response.status === 204) return null
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

export function fetchMatchConfig() {
  return fetchJson('/match-config')
}

export function saveMatchConfig(config) {
  return fetchJson('/match-config', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
}

export function fetchAnalysis() {
  return fetchJson('/analysis')
}

export function startAnalysis(payload) {
  return fetchJson('/analysis/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function stopAnalysis() {
  return fetchJson('/analysis/stop', { method: 'POST' })
}

export function uploadVideo(file, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest()
    request.open('POST', asUrl('/uploads'))

    request.upload.onprogress = (event) => {
      if (!onProgress || !event.lengthComputable) return
      onProgress(Math.round((event.loaded / event.total) * 100))
    }

    request.onload = () => {
      try {
        const data = JSON.parse(request.responseText || '{}')
        if (request.status >= 200 && request.status < 300) {
          resolve(data)
          return
        }
        reject(new Error(data.detail || `Upload failed (${request.status})`))
      } catch (error) {
        reject(error)
      }
    }

    request.onerror = () => reject(new Error('Upload failed'))
    const body = new FormData()
    body.append('file', file)
    request.send(body)
  })
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
