import { useEffect, useState } from 'react'
import { fetchVideos, videoUrl } from '../api'

function bytesToMB(bytes) {
  if (!bytes && bytes !== 0) return '-'
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export default function VideosPage() {
  const [videos, setVideos] = useState([])
  const [selected, setSelected] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const data = await fetchVideos()
        if (!alive) return
        const list = data.videos ?? []
        setVideos(list)
        setSelected((curr) => curr || list[0]?.url || '')
        setError('')
      } catch (err) {
        if (!alive) return
        setError(err instanceof Error ? err.message : 'Failed to load videos')
      } finally {
        if (alive) setLoading(false)
      }
    }
    load()
    return () => {
      alive = false
    }
  }, [])

  if (loading) {
    return <section className="panel">Loading processed videos...</section>
  }

  if (error && videos.length === 0) {
    return <section className="panel error">Unable to load videos: {error}</section>
  }

  if (!videos.length) {
    return (
      <section className="panel">
        No processed videos found in `agent/` yet. Run the detector first.
      </section>
    )
  }

  return (
    <section className="videos-page">
      <aside className="panel">
        <h3>Available Videos</h3>
        <ul className="video-list">
          {videos.map((video) => (
            <li key={video.name}>
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
      </aside>

      <section className="panel">
        <h3>Preview</h3>
        <video
          key={selected}
          controls
          className="player"
          src={videoUrl(selected)}
        >
          Your browser does not support video playback.
        </video>
      </section>
    </section>
  )
}
