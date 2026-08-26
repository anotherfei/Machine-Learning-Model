import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';

const apiPort = process.env.SPINDLE_API_PORT || '8000';
const apiTarget = `http://127.0.0.1:${apiPort}`;

export default defineConfig({
  plugins:[react()],
  server:{
    host:'127.0.0.1',
    port:5173,
    strictPort:true,
    headers:{
      'Cache-Control':'no-store, no-cache, must-revalidate, max-age=0',
      'Pragma':'no-cache',
      'Expires':'0',
    },
    proxy:{
      '/api': apiTarget,
      '/ws': {target: apiTarget, ws:true}
    }
  }
});
