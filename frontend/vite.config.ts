import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: ['skc', '.local'],
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://backend:8800',
        changeOrigin: true,
      },
    },
  },
})
