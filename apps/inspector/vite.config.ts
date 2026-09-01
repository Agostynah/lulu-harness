import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxies /api to lulu-server (default port 8420, see server.py's `run()`)
// so the dev server and the API can share an origin -- no CORS dance
// needed beyond what server.py already allows for localhost:5173.
//
// port 5183, not 5173: the Tauri shell (src-tauri/tauri.conf.json)
// launches this dev server itself and needs a fixed, known port to point
// its devUrl at -- strictPort:true fails loudly instead of silently
// drifting to whatever port happens to be free (5173/5174 are commonly
// taken by other local projects on this machine).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5183,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8420",
        changeOrigin: true,
      },
    },
  },
});
