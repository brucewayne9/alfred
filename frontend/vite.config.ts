import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'
import path from 'path'

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['icon-192.png', 'icon-512.png', 'apple-touch-icon.png'],
      manifest: {
        name: 'Alfred',
        short_name: 'Alfred',
        description: 'Personal AI Assistant',
        start_url: '/',
        display: 'standalone',
        background_color: '#0a0a0a',
        theme_color: '#0a0a0a',
        orientation: 'any',
        icons: [
          { src: '/icon-192.png', sizes: '192x192', type: 'image/png', purpose: 'any maskable' },
          { src: '/icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any maskable' },
        ],
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff2}'],
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/aialfred\.groundrushcloud\.com\/api\/.*/i,
            handler: 'NetworkOnly',
          },
          {
            urlPattern: /\.(png|jpg|jpeg|svg|gif|webp)$/,
            handler: 'CacheFirst',
            options: { cacheName: 'images', expiration: { maxEntries: 50, maxAgeSeconds: 30 * 24 * 60 * 60 } },
          },
        ],
      },
    }),
  ],
  resolve: {
    alias: { '@': path.resolve(__dirname, 'src') },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          markdown: ['react-markdown', 'remark-gfm', 'rehype-highlight'],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/auth': { target: 'http://localhost:8400', changeOrigin: true },
      '/chat': { target: 'http://localhost:8400', changeOrigin: true },
      '/conversations': { target: 'http://localhost:8400', changeOrigin: true },
      '/projects': { target: 'http://localhost:8400', changeOrigin: true },
      '/references': { target: 'http://localhost:8400', changeOrigin: true },
      '/voice': { target: 'http://localhost:8400', changeOrigin: true },
      '/memory': { target: 'http://localhost:8400', changeOrigin: true },
      '/agents': { target: 'http://localhost:8400', changeOrigin: true },
      '/knowledge': { target: 'http://localhost:8400', changeOrigin: true },
      '/integrations': { target: 'http://localhost:8400', changeOrigin: true },
      '/notifications': { target: 'http://localhost:8400', changeOrigin: true },
      '/push': { target: 'http://localhost:8400', changeOrigin: true },
      '/settings': { target: 'http://localhost:8400', changeOrigin: true },
      '/static': { target: 'http://localhost:8400', changeOrigin: true },
      '/uploads': { target: 'http://localhost:8400', changeOrigin: true },
      '/ws': { target: 'http://localhost:8400', changeOrigin: true, ws: true },
    },
  },
})
