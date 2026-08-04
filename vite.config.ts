import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The AI proxy (server/main.ts) listens only on 127.0.0.1:8787; the browser reaches it same-origin.
const aiProxy = { '/api': { target: 'http://127.0.0.1:8787', changeOrigin: true } };

export default defineConfig({
  plugins: [react()],
  // Relative asset URLs: the build works from any static host and sub-path.
  base: './',
  server: { proxy: aiProxy },
  preview: { proxy: aiProxy },
  // The MuJoCo loader locates its .wasm via import.meta.url; keep it out of pre-bundling.
  optimizeDeps: { exclude: ['@mujoco/mujoco'] },
  worker: { format: 'es' },
  build: { target: 'es2022', chunkSizeWarningLimit: 1500 },
});
