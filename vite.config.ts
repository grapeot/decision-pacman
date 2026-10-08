import { defineConfig, loadEnv } from "vite";

// In development the page calls /decide/*, which Vite forwards to the decision
// endpoint. Same-origin requests avoid CORS setup for any port or host.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_DECISION_BASE_URL || "http://localhost:11434";
  // A hosted endpoint's key is added here, in the dev server, so it never reaches the page.
  const apiKey = process.env.TYPESAFE_API_KEY;
  // Same for OpenAI's Decisions API: the page calls /openai/v1/decisions and the key is added here.
  const openaiKey = process.env.OPENAI_API_KEY;
  return {
    // Relative asset paths, so the build also works when bundled inside the iOS app.
    base: "./",
    // Tells the page that /decide leads to a hosted endpoint, so requests leave out Ollama-only fields.
    define: { "import.meta.env.VITE_DECISION_HOSTED": JSON.stringify(apiKey ? "1" : "") },
    server: {
      port: 5173,
      proxy: {
        "/decide": {
          target,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/decide/, ""),
          ...(apiKey ? { headers: { Authorization: `Bearer ${apiKey}` } } : {}),
        },
        "/openai": {
          target: "https://api.openai.com",
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/openai/, ""),
          ...(openaiKey ? { headers: { Authorization: `Bearer ${openaiKey}` } } : {}),
        },
      },
    },
    test: {
      include: ["tests/**/*.test.ts"],
    },
  };
});
