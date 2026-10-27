import { useState, useEffect } from 'react'
import { apiFetch } from '../config'
export function useBackendHealth() {
  const [online, setOnline] = useState(false)
  useEffect(() => {
    let stopped = false
    let checking = false
    let controller
    const check = async () => {
      if (stopped || checking || document.hidden) return
      checking = true
      controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 10000)
      try {
        const response = await apiFetch('/health', { signal: controller.signal })
        if (!stopped) setOnline(response.ok)
      } catch { if (!stopped) setOnline(false) }
      finally { clearTimeout(timeout); checking = false }
    }
    check()
    const timer = setInterval(check, 300000)
    document.addEventListener('visibilitychange', check)
    return () => {
      stopped = true; controller?.abort(); clearInterval(timer)
      document.removeEventListener('visibilitychange', check)
    }
  }, [])
  return online
}
