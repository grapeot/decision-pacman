import { defineConfig, loadEnv } from "vite";

// In development the page calls /decide/*, which Vite forwards to the decision
// endpoint. Same-origin requests avoid CORS setup for any port or host.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_DECISION_BASE_URL || "http://localhost:11434";
  return {
    // Relative asset paths, so the build also works when bundled inside the iOS app.
    base: "./",
    server: {
      port: 5173,
      proxy: {
        "/decide": {
          target,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/decide/, ""),
        },
      },
    },
    test: {
      include: ["tests/**/*.test.ts"],
    },
  };
});
