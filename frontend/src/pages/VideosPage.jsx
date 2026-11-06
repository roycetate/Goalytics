import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  fetchAnalysis,
  fetchMatchConfig,
  fetchVideos,
  startAnalysis,
  stopAnalysis,
  uploadVideo,
  videoUrl,
} from '../api'

function bytesToMB(bytes) {
  if (!bytes && bytes !== 0) return '-'
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function statusLabel(status) {
  if (status === 'running') return 'Analysis running'
  if (status === 'stopping') return 'Stopping analysis'
  if (status === 'completed') return 'Analysis completed'
  if (status === 'stopped') return 'Analysis stopped'
  if (status === 'failed') return 'Analysis failed'
  return 'Ready to analyze'
}

export default function VideosPage() {
  const [videos, setVideos] = useState({ uploads: [], library: [], processed: [] })
  const [selected, setSelected] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [dragOver, setDragOver] = useState(false)
  const [analysis, setAnalysis] = useState({ status: 'idle' })
  const [busy, setBusy] = useState(false)

  const sources = useMemo(
    () => [...(videos.uploads ?? []), ...(videos.library ?? [])],
    [videos],
  )
  const processed = videos.processed ?? []
  const selectedVideo = [...sources, ...processed].find((video) => video.url === selected)

  const loadVideos = async () => {
    const data = await fetchVideos()
    const next = {
      uploads: data.uploads ?? [],
      library: data.library ?? [],
      processed: data.processed ?? [],
    }
    setVideos(next)
    const combined = [...next.uploads, ...next.library, ...next.processed]
    setSelected((curr) => curr || combined[0]?.url || '')
    return next
  }

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        await loadVideos()
        if (alive) setError('')
      } catch (err) {
        if (alive) setError(err instanceof Error ? err.message : 'Failed to load videos')
      } finally {
        if (alive) setLoading(false)
      }
    }
    load()
    return () => {
      alive = false
    }
  }, [])

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const data = await fetchAnalysis()
        if (alive) setAnalysis(data)
      } catch {
        // Keep the upload UI usable if the job endpoint is temporarily down.
      }
    }
    load()
    const timer = setInterval(load, 1500)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [])

  const handleFiles = async (fileList) => {
    const file = fileList?.[0]
    if (!file) return
    setUploading(true)
    setProgress(0)
    setError('')
    try {
      const uploaded = await uploadVideo(file, setProgress)
      const next = await loadVideos()
      const match = [...next.uploads, ...next.library].find((video) => video.name === uploaded.name)
      if (match) setSelected(match.url)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const handleAnalyze = async () => {
    if (!selectedVideo) return
    setBusy(true)
    setError('')
    try {
      const match = await fetchMatchConfig().catch(() => ({}))
      const data = await startAnalysis({
        video_name: selectedVideo.name,
        ...match,
      })
      setAnalysis(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not start analysis')
    } finally {
      setBusy(false)
    }
  }

  const handleStop = async () => {
    setBusy(true)
    setError('')
    try {
      const data = await stopAnalysis()
      setAnalysis(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not stop analysis')
    } finally {
      setBusy(false)
    }
  }

  const running = analysis.status === 'running' || analysis.status === 'stopping'

  if (loading) {
    return <section className="panel">Loading videos...</section>
  }

  return (
    <section className="videos-page">
      <div className="videos-stack">
        <section className="panel">
          <div className="panel-header">
            <div>
              <h2 className="panel-title">Upload a match</h2>
              <p className="panel-subtitle">Drop a clip here, then run live analysis from the browser.</p>
            </div>
          </div>

          <label
            className={`upload-dropzone ${dragOver ? 'is-over' : ''} ${uploading ? 'is-busy' : ''}`}
            onDragOver={(event) => {
              event.preventDefault()
              setDragOver(true)
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(event) => {
              event.preventDefault()
              setDragOver(false)
              handleFiles(event.dataTransfer.files)
            }}
          >
            <input
              type="file"
              accept="video/mp4,video/quicktime,video/x-matroska,video/webm,video/x-msvideo,.mp4,.mov,.mkv,.webm,.avi"
              hidden
              onChange={(event) => {
                handleFiles(event.target.files)
                event.target.value = ''
              }}
            />
            <strong>{uploading ? `Uploading ${progress}%` : 'Drop a video or click to browse'}</strong>
            <span>MP4, MOV, MKV, WEBM, or AVI up to 500 MB</span>
            {uploading ? (
              <span className="upload-progress" style={{ '--progress': `${progress}%` }} />
            ) : null}
          </label>

          <div className={`analysis-status is-${analysis.status || 'idle'}`}>
            <div>
              <strong>{statusLabel(analysis.status)}</strong>
              <p>
                {analysis.video_name
                  ? `${analysis.video_name}${running ? ' is being processed.' : '.'}`
                  : 'Choose an uploaded or library video, then start analysis.'}
              </p>
              {analysis.error ? <p className="analysis-error">{analysis.error}</p> : null}
            </div>
            <div className="analysis-actions">
              {running ? (
                <button type="button" className="ghost-button" onClick={handleStop} disabled={busy}>
                  Stop
                </button>
              ) : (
                <button
                  type="button"
                  className="primary-button"
                  onClick={handleAnalyze}
                  disabled={busy || !selectedVideo || selectedVideo.kind === 'processed'}
                >
                  Analyze selected
                </button>
              )}
              <Link className="text-link" to="/dashboard">Open dashboard</Link>
            </div>
          </div>

          {error ? <p className="form-error">{error}</p> : null}
        </section>

        <section className="panel">
          <h3 className="panel-title">Match videos</h3>
          {sources.length ? (
            <ul className="video-list">
              {sources.map((video) => (
                <li key={`${video.kind}-${video.name}`}>
                  <button
                    type="button"
                    className={selected === video.url ? 'active' : ''}
                    onClick={() => setSelected(video.url)}
                  >
                    <span>{video.name}</span>
                    <small>{video.kind === 'upload' ? 'Uploaded' : 'Library'} · {bytesToMB(video.size_bytes)}</small>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="empty-copy">No source videos yet. Upload a match clip to get started.</p>
          )}
        </section>

        <section className="panel">
          <h3 className="panel-title">Processed output</h3>
          {processed.length ? (
            <ul className="video-list">
              {processed.map((video) => (
                <li key={`${video.kind}-${video.name}`}>
                  <button
                    type="button"
                    className={selected === video.url ? 'active' : ''}
                    onClick={() => setSelected(video.url)}
                  >
                    <span>{video.name}</span>
                    <small>{bytesToMB(video.size_bytes)}</small>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="empty-copy">Processed video will appear here after analysis finishes.</p>
          )}
        </section>
      </div>

      <section className="panel">
        <h3 className="panel-title">Preview</h3>
        {selected ? (
          <video key={selected} controls className="player" src={videoUrl(selected)}>
            Your browser does not support video playback.
          </video>
        ) : (
          <p className="empty-copy">Upload a video to preview it here.</p>
        )}
      </section>
    </section>
  )
}
