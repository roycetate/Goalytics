import { useEffect, useState } from 'react'
import { Users, Zap, Cpu, Clock } from 'lucide-react'
import { analysisStreamUrl, fetchHeatmaps, fetchStats, heatmapUrl } from '../api'
import { teamAbbrev, useMatchConfig } from '../matchConfig'

const POLL_MS = 200
const UNAVAILABLE = 'Unavailable'

function asNumber(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function asPercent(value, fallback) {
  const number = Number(value)
  return Number.isFinite(number) ? number : fallback
}

function formatPercent(value) {
  return `${Number(value).toFixed(2)}%`
}

function formatMetric(value, suffix = '') {
  const number = asNumber(value)
  if (number === null) return UNAVAILABLE
  return `${Number.isInteger(number) ? number : number.toFixed(2)}${suffix}`
}

function getStat(stats, keys) {
  for (const key of keys) {
    if (asNumber(stats[key]) !== null) return stats[key]
  }
  return null
}

function getAggregateStat(stats, totalKeys, teamKeys) {
  const total = getStat(stats, totalKeys)
  if (asNumber(total) !== null) return total

  const values = teamKeys.map((keys) => asNumber(getStat(stats, keys)))
  if (values.every((value) => value === null)) return null
  return values.reduce((sum, value) => sum + (value ?? 0), 0)
}

function getTopEntry(record) {
  const entries = Object.entries(record ?? {})
    .map(([id, value]) => [id, asNumber(value)])
    .filter((entry) => entry[1] !== null && entry[1] > 0)

  if (!entries.length) return null
  return entries.sort((a, b) => b[1] - a[1])[0]
}

function getTrackingQuality(stats) {
  if (!stats.timestamp) return 'Waiting'

  const players = asNumber(stats.players_detected) ?? 0
  const fps = asNumber(stats.fps) ?? 0

  if (players >= 20 && fps >= 12) return 'High'
  if (players >= 16 && fps >= 8) return 'Good'
  if (players > 0) return 'Low'
  return 'Waiting'
}

function MetricCard({ icon: Icon, value, label }) {
  return (
    <div className="metric-card">
      <div className="metric-header">
        <Icon size={20} />
      </div>
      <h2 className="metric-value">{value}</h2>
      <p className="metric-label">{label}</p>
    </div>
  )
}

export default function DashboardPage() {
  const [stats, setStats] = useState(null)
  const [heatmaps, setHeatmaps] = useState([])
  const [streamAvailable, setStreamAvailable] = useState(true)
  
  const [match, setMatch] = useMatchConfig()
  const [activeHeatmapTeam, setActiveHeatmapTeam] = useState('team1')
  const team1Name = match.team1_name
  const team2Name = match.team2_name
  const team1Color = match.team1_color
  const team2Color = match.team2_color
  const team1Short = teamAbbrev(team1Name, 'T1')
  const team2Short = teamAbbrev(team2Name, 'T2')

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const data = await fetchStats()
        if (alive) setStats(data)
      } catch {
        // silently fail
      }
    }
    load()
    const timer = setInterval(load, POLL_MS)
    return () => { alive = false; clearInterval(timer) }
  }, [])

  useEffect(() => {
    let alive = true
    const loadHeatmaps = async () => {
      try {
        const data = await fetchHeatmaps()
        if (alive) setHeatmaps(data.heatmaps ?? [])
      } catch {
        // silently fail
      }
    }
    loadHeatmaps()
    const timer = setInterval(loadHeatmaps, 5000)
    return () => { alive = false; clearInterval(timer) }
  }, [])

  const displayStats = stats || {}
  const analysis = displayStats.analysis || {}
  const hasLive = Boolean(displayStats.timestamp)
  const team1Possession = asPercent(displayStats.team1_possession_pct, hasLive ? 58 : 0)
  const team2Possession = asPercent(displayStats.team2_possession_pct, hasLive ? 42 : 0)
  const possessionTotal = team1Possession + team2Possession
  const team1Share = possessionTotal > 0 ? (team1Possession / possessionTotal) * 100 : 50
  const possessionLeader =
    team1Possession === team2Possession
      ? 'Even'
      : team1Possession > team2Possession
        ? team1Name
        : team2Name
  const topTouch = getTopEntry(displayStats.player_touch_counts)
  const teamTouches =
    displayStats.team1_touches == null && displayStats.team2_touches == null
      ? UNAVAILABLE
      : `${formatMetric(displayStats.team1_touches)}-${formatMetric(displayStats.team2_touches)}`
  const totalGoals = getAggregateStat(
    displayStats,
    ['total_goals', 'goals'],
    [['team1_goals', 'liverpool_goals'], ['team2_goals', 'real_madrid_goals']]
  )
  const totalShots = getAggregateStat(
    displayStats,
    ['total_shots', 'shots'],
    [['team1_shots', 'liverpool_shots'], ['team2_shots', 'real_madrid_shots']]
  )
  const totalShotsOnTarget = getAggregateStat(
    displayStats,
    ['total_shots_on_target', 'shots_on_target'],
    [['team1_shots_on_target', 'liverpool_shots_on_target'], ['team2_shots_on_target', 'real_madrid_shots_on_target']]
  )
  const team1Goals = getStat(displayStats, ['team1_goals', 'liverpool_goals'])
  const team2Goals = getStat(displayStats, ['team2_goals', 'real_madrid_goals'])
  const team1Shots = getStat(displayStats, ['team1_shots', 'liverpool_shots'])
  const team2Shots = getStat(displayStats, ['team2_shots', 'real_madrid_shots'])
  const team1ShotsOnTarget = getStat(displayStats, ['team1_shots_on_target', 'liverpool_shots_on_target'])
  const team2ShotsOnTarget = getStat(displayStats, ['team2_shots_on_target', 'real_madrid_shots_on_target'])
  const teamComparison = [
    {
      label: 'Possession',
      team1: formatPercent(team1Possession),
      team2: formatPercent(team2Possession),
    },
    {
      label: 'Touches',
      team1: formatMetric(displayStats.team1_touches),
      team2: formatMetric(displayStats.team2_touches),
    },
    {
      label: 'Players',
      team1: formatMetric(displayStats.team1_players),
      team2: formatMetric(displayStats.team2_players),
    },
    {
      label: 'Shots',
      team1: formatMetric(team1Shots),
      team2: formatMetric(team2Shots),
    },
    {
      label: 'On Target',
      team1: formatMetric(team1ShotsOnTarget),
      team2: formatMetric(team2ShotsOnTarget),
    },
  ]

  return (
    <div className="dashboard">
      
      <section className="team-settings-panel">
        <div className="team-settings-title">
          <span>Teams</span>
          <small>Match identity</small>
        </div>
        <label className="team-editor">
          <input 
            type="color" 
            value={team1Color} 
            onChange={e => setMatch((curr) => ({ ...curr, team1_color: e.target.value }))}
            className="team-color-input"
            title="Team 1 Color"
          />
          <input 
            type="text" 
            value={team1Name} 
            onChange={e => setMatch((curr) => ({ ...curr, team1_name: e.target.value }))} 
            className="team-name-input"
            aria-label="Team 1 name"
          />
        </label>
        <label className="team-editor">
          <input 
            type="color" 
            value={team2Color} 
            onChange={e => setMatch((curr) => ({ ...curr, team2_color: e.target.value }))}
            className="team-color-input"
            title="Team 2 Color"
          />
          <input 
            type="text" 
            value={team2Name} 
            onChange={e => setMatch((curr) => ({ ...curr, team2_name: e.target.value }))} 
            className="team-name-input"
            aria-label="Team 2 name"
          />
        </label>
      </section>

      <section className="match-header">
        <div className="match-team match-team-left">
          <span className="match-team-mark" style={{ background: team1Color }} />
          <div>
            <p>{team1Name}</p>
            <small>{formatPercent(team1Possession)} possession</small>
          </div>
        </div>
        <div className="match-scoreboard">
          <div className="match-status">
            <span className="status-dot" /> {analysis.status === 'running' ? 'Live analysis' : analysis.status === 'completed' ? 'Analysis complete' : 'Waiting'}
          </div>
          <div className="match-score">
            <span>{formatMetric(team1Goals)}</span>
            <b>:</b>
            <span>{formatMetric(team2Goals)}</span>
          </div>
          <div className="match-meta">
            <span>Leader: {possessionLeader}</span>
            <span>{formatMetric(displayStats.fps)} FPS</span>
          </div>
        </div>
        <div className="match-team match-team-right">
          <div>
            <p>{team2Name}</p>
            <small>{formatPercent(team2Possession)} possession</small>
          </div>
          <span className="match-team-mark" style={{ background: team2Color }} />
        </div>
      </section>

      <section className="team-comparison-strip">
        <div className="comparison-team-name" style={{ '--team-color': team1Color }}>{team1Name}</div>
        <div className="comparison-stats">
          {teamComparison.map((item) => (
            <div className="comparison-item" key={item.label}>
              <span className="comparison-value">{item.team1}</span>
              <span className="comparison-label">{item.label}</span>
              <span className="comparison-value">{item.team2}</span>
            </div>
          ))}
        </div>
        <div className="comparison-team-name comparison-team-name-right" style={{ '--team-color': team2Color }}>{team2Name}</div>
      </section>

      {/* Live Match Hero Section */}
      <section className="panel" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h2 className="panel-title" style={{ fontSize: '1.4rem' }}>Live Match Analysis</h2>
            <p className="panel-subtitle">Real-time player tracking and performance metrics</p>
          </div>
          <div className="live-pill">
            <div className="dot"></div> LIVE
          </div>
        </div>

        <div style={{ display: 'flex', borderTop: '1px solid var(--border-color)', minHeight: '400px' }}>
          
          {/* Video Feed */}
          <div style={{ flex: 2, borderRight: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000', position: 'relative', overflow: 'hidden' }}>
            {streamAvailable ? (
              <img
                src={analysisStreamUrl()}
                alt="Live Match"
                onError={() => setStreamAvailable(false)}
                onLoad={() => setStreamAvailable(true)}
                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
              />
            ) : (
              <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '1rem' }}>
                Upload a match on the Videos page and start analysis to see the live feed.
              </div>
            )}
          </div>

          {/* 2D Tactical Radar */}
          <div style={{ flex: 1, padding: '1.5rem', display: 'flex', flexDirection: 'column', background: 'var(--bg-main)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h4 style={{ margin: 0, color: 'var(--text-muted)', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '1px' }}>2D Tactical Radar</h4>
            </div>
            
            <div style={{ display: 'flex', flex: 1, alignItems: 'center', justifyContent: 'center' }}>
              <div style={{ position: 'relative', width: '100%', aspectRatio: '1.54', background: '#1A3324', border: '1px solid rgba(255,255,255,0.3)', borderRadius: '4px', overflow: 'hidden' }}>
                
                {/* Field Markings */}
                <div style={{ position: 'absolute', top: 0, bottom: 0, left: '50%', width: '1px', background: 'rgba(255,255,255,0.3)' }}></div>
                <div style={{ position: 'absolute', top: '50%', left: '50%', width: '20%', aspectRatio: '1', borderRadius: '50%', border: '1px solid rgba(255,255,255,0.3)', transform: 'translate(-50%, -50%)' }}></div>
                <div style={{ position: 'absolute', top: '20%', bottom: '20%', left: 0, width: '15%', border: '1px solid rgba(255,255,255,0.3)', borderLeft: 'none' }}></div>
                <div style={{ position: 'absolute', top: '20%', bottom: '20%', right: 0, width: '15%', border: '1px solid rgba(255,255,255,0.3)', borderRight: 'none' }}></div>
                <div style={{ position: 'absolute', top: '35%', bottom: '35%', left: 0, width: '6%', border: '1px solid rgba(255,255,255,0.3)', borderLeft: 'none' }}></div>
                <div style={{ position: 'absolute', top: '35%', bottom: '35%', right: 0, width: '6%', border: '1px solid rgba(255,255,255,0.3)', borderRight: 'none' }}></div>
                
                {/* Dots */}
                {displayStats.radar_points?.map((pt, i) => {
                  let color = '#FFF'
                  let size = 10
                  let zIndex = 10
                  let hasJersey = pt.type === 'player' && pt.jersey != null
                  
                  if (pt.type === 'player') {
                    color = pt.team === 0 ? team1Color : team2Color
                    size = hasJersey ? 16 : 12
                  } else if (pt.type === 'ball') {
                    color = '#FDE047'
                    size = 8
                    zIndex = 20
                  }
                  return (
                    <div key={i} style={{
                      position: 'absolute',
                      left: `${pt.x * 100}%`,
                      top: `${pt.y * 100}%`,
                      width: `${size}px`,
                      height: `${size}px`,
                      background: color,
                      borderRadius: '50%',
                      transform: 'translate(-50%, -50%)',
                      boxShadow: '0 2px 4px rgba(0,0,0,0.4)',
                      border: pt.type === 'ball' ? 'none' : '1px solid rgba(255,255,255,0.8)',
                      transition: 'all 0.2s linear',
                      zIndex: zIndex,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#FFF',
                      fontSize: '9px',
                      fontWeight: 'bold',
                      lineHeight: 1
                    }}>
                      {hasJersey ? pt.jersey : ''}
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4 Metric Cards */}
      <div className="metrics-row">
        <MetricCard icon={Users} value={formatMetric(displayStats.players_detected)} label="Players Detected" />
        <MetricCard icon={Zap} value={teamTouches} label="Team Touches" />
        <MetricCard icon={Cpu} value={formatMetric(displayStats.fps)} label="Processing FPS" />
      </div>

      {/* Charts Grid */}
      <div className="charts-grid">
        <section className="panel">
          <div className="panel-header">
            <h3 className="panel-title">Possession Analysis</h3>
            <Clock size={16} color="var(--text-muted)" />
          </div>
          <div className="possession-chart-wrap">
            <div
              className="possession-donut"
              style={{
                '--team1-color': team1Color,
                '--team2-color': team2Color,
                '--team1-share': `${team1Share}%`,
              }}
              role="img"
              aria-label={`${team1Name} ${formatPercent(team1Possession)}, ${team2Name} ${formatPercent(team2Possession)}`}
            >
              <div className="possession-donut-center">
                <span>{formatPercent(team1Possession)}</span>
                <small>{team1Name}</small>
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', fontSize: '0.85rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}><div className="score-dot" style={{ background: team1Color }}></div> {team1Name} {formatPercent(team1Possession)}</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}><div className="score-dot" style={{ background: team2Color }}></div> {team2Name} {formatPercent(team2Possession)}</div>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header" style={{ marginBottom: '1rem' }}>
            <h3 className="panel-title">Team Heatmap</h3>
            <button
              type="button"
              className={`team-toggle ${activeHeatmapTeam === 'team2' ? 'team-toggle-right' : ''}`}
              role="switch"
              aria-checked={activeHeatmapTeam === 'team2'}
              aria-label={`Showing ${activeHeatmapTeam === 'team1' ? team1Name : team2Name} heatmap. Switch heatmap team.`}
              onClick={() => setActiveHeatmapTeam(activeHeatmapTeam === 'team1' ? 'team2' : 'team1')}
            >
              <span className="team-toggle-track">
                <span className="team-toggle-label">{team1Short}</span>
                <span className="team-toggle-label">{team2Short}</span>
                <span className="team-toggle-thumb">
                  {activeHeatmapTeam === 'team1' ? team1Short : team2Short}
                </span>
              </span>
            </button>
          </div>
          
          <div style={{ height: '240px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            {heatmaps.length ? (
              heatmaps.filter(hm => hm.team === activeHeatmapTeam).map(hm => (
                <div key={hm.team} className="heatmap-img-container" style={{ width: '100%', height: '100%', background: 'transparent', border: 'none' }}>
                  <img src={heatmapUrl(hm.url, hm.updated_at)} alt="Heatmap" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                </div>
              ))
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                Heatmap pending...
              </div>
            )}
          </div>
        </section>
      </div>

      {/* Tracking Details Table */}
      <section className="panel">
        <h3 className="panel-title" style={{ fontSize: '1.2rem', marginBottom: '2rem' }}>Live Insights</h3>
        <div className="tracking-grid">
          <div>
            <div className="tracking-row">
              <span className="tracking-label">Total Passes</span>
              <span className="tracking-val">{formatMetric(displayStats.total_passes)}</span>
            </div>
            <div className="tracking-row">
              <span className="tracking-label">Possession Leader</span>
              <span className="tracking-val val-green">{possessionLeader}</span>
            </div>
            <div className="tracking-row">
              <span className="tracking-label">Team Touches</span>
              <span className="tracking-val">{teamTouches}</span>
            </div>
          </div>
          <div>
            <div className="tracking-row">
              <span className="tracking-label">Top Touch Player</span>
              <span className="tracking-val">
                {topTouch ? `#${topTouch[0]} (${topTouch[1]} touches)` : UNAVAILABLE}
              </span>
            </div>
            <div className="tracking-row">
              <span className="tracking-label">Tracked Players</span>
              <span className="tracking-val">{formatMetric(displayStats.tracked_player_ids?.length)}</span>
            </div>
          </div>
          <div>
            <div className="tracking-row">
              <span className="tracking-label">Goals</span>
              <span className="tracking-val">{formatMetric(totalGoals)}</span>
            </div>
            <div className="tracking-row">
              <span className="tracking-label">Shots</span>
              <span className="tracking-val">{formatMetric(totalShots)}</span>
            </div>
          </div>
          <div>
            <div className="tracking-row">
              <span className="tracking-label">Shots On Target</span>
              <span className="tracking-val">{formatMetric(totalShotsOnTarget)}</span>
            </div>
            <div className="tracking-row">
              <span className="tracking-label">Tracking Quality</span>
              <span className="tracking-val">{getTrackingQuality(displayStats)}</span>
            </div>
          </div>
        </div>
      </section>

    </div>
  )
}
