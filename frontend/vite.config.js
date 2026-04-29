import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/upload': 'http://localhost:8000',
      '/analyze': 'http://localhost:8000',
      '/report': 'http://localhost:8000',
      '/visualizations': 'http://localhost:8000',
    }
  }
})
