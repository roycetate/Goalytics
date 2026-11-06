import { useEffect, useState } from 'react'
import { Clock } from 'lucide-react'
import { fetchHeatmaps, fetchStats, heatmapUrl } from '../api'
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

function formatMetric(value) {
  const number = asNumber(value)
  if (number === null) return UNAVAILABLE
  return Number.isInteger(number) ? String(number) : number.toFixed(2)
}

function getStat(stats, keys) {
  for (const key of keys) {
    if (asNumber(stats[key]) !== null) return stats[key]
  }
  return null
}

function getTopTeamTouch(record, playerIds) {
  if (!playerIds.size) return null

  const entries = Object.entries(record ?? {})
    .map(([id, value]) => [id, asNumber(value)])
    .filter(([id, value]) => playerIds.has(String(id)) && value !== null && value > 0)

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

function InsightRow({ label, value, help, valueClassName = 'tracking-val' }) {
  return (
    <div className="tracking-row insight-row" tabIndex={0} aria-label={`${label}: ${help}`}>
      <span className="tracking-label">{label}</span>
      <span className={valueClassName}>{value}</span>
      <span className="insight-help" role="tooltip">{help}</span>
    </div>
  )
}

export default function InsightsPage() {
  const [stats, setStats] = useState(null)
  const [heatmaps, setHeatmaps] = useState([])
  const [selectedTeam, setSelectedTeam] = useState('team1')
  const [match] = useMatchConfig()

  const teams = {
    team1: {
      id: 'team1',
      index: 0,
      name: match.team1_name,
      color: match.team1_color,
      short: teamAbbrev(match.team1_name, 'T1'),
      possessionKey: 'team1_possession_pct',
      playersKey: 'team1_players',
      touchesKey: 'team1_touches',
      goalsKeys: ['team1_goals', 'liverpool_goals'],
      shotsKeys: ['team1_shots', 'liverpool_shots'],
      shotsOnTargetKeys: ['team1_shots_on_target', 'liverpool_shots_on_target'],
    },
    team2: {
      id: 'team2',
      index: 1,
      name: match.team2_name,
      color: match.team2_color,
      short: teamAbbrev(match.team2_name, 'T2'),
      possessionKey: 'team2_possession_pct',
      playersKey: 'team2_players',
      touchesKey: 'team2_touches',
      goalsKeys: ['team2_goals', 'real_madrid_goals'],
      shotsKeys: ['team2_shots', 'real_madrid_shots'],
      shotsOnTargetKeys: ['team2_shots_on_target', 'real_madrid_shots_on_target'],
    },
  }

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const data = await fetchStats()
        if (alive) setStats(data)
      } catch {
        // Keep showing unavailable values if backend is offline.
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
        // Keep the page usable while generated heatmaps are unavailable.
      }
    }
    loadHeatmaps()
    const timer = setInterval(loadHeatmaps, 5000)
    return () => { alive = false; clearInterval(timer) }
  }, [])

  const displayStats = stats || {}
  const selected = teams[selectedTeam]
  const opponent = selectedTeam === 'team1' ? teams.team2 : teams.team1
  const team1Name = teams.team1.name
  const team2Name = teams.team2.name
  const team1Color = teams.team1.color
  const team2Color = teams.team2.color
  const hasLive = Boolean(displayStats.timestamp)
  const team1Possession = asPercent(displayStats.team1_possession_pct, hasLive ? 58 : 0)
  const team2Possession = asPercent(displayStats.team2_possession_pct, hasLive ? 42 : 0)
  const possessionTotal = team1Possession + team2Possession
  const team1Share = possessionTotal > 0 ? (team1Possession / possessionTotal) * 100 : 50
  const selectedPossession = asPercent(displayStats[selected.possessionKey], hasLive ? (selectedTeam === 'team1' ? 58 : 42) : 0)
  const opponentPossession = asPercent(displayStats[opponent.possessionKey], hasLive ? (selectedTeam === 'team1' ? 42 : 58) : 0)
  const possessionDiff = selectedPossession - opponentPossession
  const selectedPlayerIds = new Set(
    (displayStats.radar_points ?? [])
      .filter((point) => point.type === 'player' && point.team === selected.index && point.id != null)
      .map((point) => String(point.id))
  )
  const selectedTopTouch = getTopTeamTouch(displayStats.player_touch_counts, selectedPlayerIds)
  const selectedHeatmap = heatmaps.find((heatmap) => heatmap.team === selectedTeam)

  return (
    <div className="dashboard">
      <section className="panel">
        <div className="panel-header">
          <div>
            <h2 className="panel-title" style={{ fontSize: '1.35rem' }}>{selected.name} Insights</h2>
            <p className="panel-subtitle">Team-specific possession, activity, heatmap, and tracking quality</p>
          </div>
          <button
            type="button"
            className={`team-toggle ${selectedTeam === 'team2' ? 'team-toggle-right' : ''}`}
            role="switch"
            aria-checked={selectedTeam === 'team2'}
            aria-label={`Showing ${selected.name}. Switch team insights.`}
            onClick={() => setSelectedTeam(selectedTeam === 'team1' ? 'team2' : 'team1')}
          >
            <span className="team-toggle-track">
              <span className="team-toggle-label">{teams.team1.short}</span>
              <span className="team-toggle-label">{teams.team2.short}</span>
              <span className="team-toggle-thumb">
                {selected.short}
              </span>
            </span>
          </button>
        </div>
      </section>

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
                <span>{formatPercent(selectedPossession)}</span>
                <small>{selected.name}</small>
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
            <h3 className="panel-title">{selected.name} Heatmap</h3>
            <div className="team-chip" style={{ '--team-color': selected.color }}>{selected.name}</div>
          </div>
          
          <div style={{ height: '240px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            {selectedHeatmap ? (
              <div className="heatmap-img-container" style={{ width: '100%', height: '100%', background: 'transparent', border: 'none' }}>
                <img src={heatmapUrl(selectedHeatmap.url, selectedHeatmap.updated_at)} alt={`${selectedHeatmap.label} heatmap`} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              </div>
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                {selected.name} heatmap pending...
              </div>
            )}
          </div>
        </section>
      </div>

      <section className="panel">
        <h3 className="panel-title" style={{ fontSize: '1.2rem', marginBottom: '2rem' }}>{selected.name} Live Insights</h3>
        <div className="tracking-grid">
          <div>
            <InsightRow
              label="Possession"
              value={formatPercent(selectedPossession)}
              help={`Share of tracked possession time assigned to ${selected.name}.`}
            />
            <InsightRow
              label="Possession Edge"
              value={possessionDiff === 0 ? 'Even' : `${possessionDiff > 0 ? '+' : ''}${formatPercent(possessionDiff)}`}
              valueClassName={possessionDiff >= 0 ? 'tracking-val val-green' : 'tracking-val'}
              help={`Difference between ${selected.name}'s possession and ${opponent.name}'s possession.`}
            />
          </div>
          <div>
            <InsightRow
              label="Team Touches"
              value={formatMetric(displayStats[selected.touchesKey])}
              help={`Number of possession touches credited to ${selected.name}.`}
            />
            <InsightRow
              label="Top Touch Player"
              value={selectedTopTouch ? `#${selectedTopTouch[0]} (${selectedTopTouch[1]} touches)` : UNAVAILABLE}
              help={`Player track ID on ${selected.name} with the most credited touches.`}
            />
          </div>
          <div>
            <InsightRow
              label="Goals"
              value={formatMetric(getStat(displayStats, selected.goalsKeys))}
              help={`Goals detected or recorded for ${selected.name}.`}
            />
            <InsightRow
              label="Shots"
              value={formatMetric(getStat(displayStats, selected.shotsKeys))}
              help={`Shot attempts detected or recorded for ${selected.name}.`}
            />
          </div>
          <div>
            <InsightRow
              label="Shots On Target"
              value={formatMetric(getStat(displayStats, selected.shotsOnTargetKeys))}
              help={`Shots by ${selected.name} that were detected or recorded as on target.`}
            />
            <InsightRow
              label="Players Detected"
              value={formatMetric(displayStats[selected.playersKey])}
              help={`Current number of detected players assigned to ${selected.name}.`}
            />
            <InsightRow
              label="Tracking Quality"
              value={getTrackingQuality(displayStats)}
              help="Overall confidence based on detected player count and processing FPS."
            />
          </div>
        </div>
      </section>
    </div>
  )
}
