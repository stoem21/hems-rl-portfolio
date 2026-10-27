import { useState, useEffect } from 'react'
import { apiFetch } from '../config'

// Check on entry or a stale return to the tab, never on an idle timer.
// Share the result and pending request across simulation remounts/StrictMode.
const CACHE_MS = 5 * 60 * 1000
let cachedOnline = false
let checkedAt = null
let pending = null
const subscribers = new Set()

function checkBackendHealth(force = false) {
  if (document.hidden) return
  if (pending) return pending
  if (!force && checkedAt !== null && Date.now() - checkedAt < CACHE_MS) return

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 10000)
  pending = Promise.resolve().then(async () => {
    try {
      const response = await apiFetch('/health', { signal: controller.signal })
      cachedOnline = response.ok
    } catch {
      cachedOnline = false
    } finally {
      clearTimeout(timeout)
      checkedAt = Date.now()
      pending = null
      subscribers.forEach(notify => notify(cachedOnline))
    }
  })
  return pending
}

export function retryBackendHealth() {
  return checkBackendHealth(true)
}

export function useBackendHealth() {
  const [online, setOnline] = useState(cachedOnline)
  useEffect(() => {
    subscribers.add(setOnline)
    setOnline(cachedOnline)
    const check = () => checkBackendHealth()
    check()
    document.addEventListener('visibilitychange', check)
    return () => {
      subscribers.delete(setOnline)
      document.removeEventListener('visibilitychange', check)
    }
  }, [])
  return online
}
