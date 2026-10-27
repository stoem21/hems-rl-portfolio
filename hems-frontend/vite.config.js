import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({
  base: process.env.VITE_SITE_BASE || '/',
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    proxy: {
      ...Object.fromEntries(['/simulate', '/health', '/explain', '/household-range', '/models', '/ask']
        .map(path => [path, 'http://127.0.0.1:8000'])),
      '/dashboard-data': { target: 'http://127.0.0.1:8787', rewrite: path => `/hems${path}` },
    },
  },
  build: { outDir: '../hems-backend/dist', emptyOutDir: true },
})
