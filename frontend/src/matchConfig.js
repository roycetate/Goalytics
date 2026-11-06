import { useEffect, useRef, useState } from 'react'
import { fetchMatchConfig, saveMatchConfig } from './api'

export const DEFAULT_MATCH_CONFIG = {
  team1_name: 'Liverpool',
  team2_name: 'Real Madrid',
  team1_color: '#EF4444',
  team2_color: '#FFFFFF',
}

export function teamAbbrev(name, fallback) {
  const compact = String(name || '').replace(/[^A-Za-z0-9]/g, '')
  if (compact.length >= 3) return compact.slice(0, 3).toUpperCase()
  return fallback
}

export function useMatchConfig() {
  const [config, setConfig] = useState(DEFAULT_MATCH_CONFIG)
  const skipSave = useRef(true)

  useEffect(() => {
    let alive = true
    fetchMatchConfig()
      .then((data) => {
        if (alive) setConfig({ ...DEFAULT_MATCH_CONFIG, ...data })
      })
      .catch(() => {})
    return () => {
      alive = false
    }
  }, [])

  useEffect(() => {
    if (skipSave.current) {
      skipSave.current = false
      return
    }
    const timer = setTimeout(() => {
      saveMatchConfig(config).catch(() => {})
    }, 400)
    return () => clearTimeout(timer)
  }, [config])

  return [config, setConfig]
}
