import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxies /api to lulu-server (default port 8420, see server.py's `run()`)
// so the dev server and the API can share an origin -- no CORS dance
// needed beyond what server.py already allows for localhost:5173.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8420",
        changeOrigin: true,
      },
    },
  },
});
