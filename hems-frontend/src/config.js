export const BACKEND_URL = (import.meta.env.VITE_BACKEND_URL || '').replace(/\/+$/, '')
export const API_HEADERS = BACKEND_URL.includes('.ngrok')
  ? { 'ngrok-skip-browser-warning': '1' } : {}
export function apiFetch(path, options = {}) {
  return fetch(`${BACKEND_URL}${path}`, {
    ...options, headers: { ...API_HEADERS, ...options.headers },
  })
}
