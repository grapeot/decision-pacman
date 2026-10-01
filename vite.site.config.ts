import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url));

// The public site (GitHub Pages): its own entry in site/, reusing the game modules in src/.
export default defineConfig({
  root: here("./site"),
  base: "/decision-pacman/",
  publicDir: here("./site/public"),
  build: {
    outDir: here("./dist-site"),
    emptyOutDir: true,
  },
  server: { port: 5174 },
  preview: { port: 4174 },
});
