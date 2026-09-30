// Records the browser game with a model playing, then converts it to MP4.
//
//   npm run record -- --model tev1:4b --seconds 90 --seed 100 --out docs/media/demo_tev1_4b.mp4
//
// Needs a running decision server and ffmpeg on PATH (for the MP4 step).
import { execFileSync } from "node:child_process";
import { mkdirSync, renameSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { parseArgs } from "node:util";
import { chromium } from "playwright";
import { createServer } from "vite";

const { values: args } = parseArgs({
  options: {
    model: { type: "string", default: "tev1:4b" },
    encoder: { type: "string", default: "features" },
    seed: { type: "string", default: "100" },
    speed: { type: "string", default: "1" },
    seconds: { type: "string", default: "90" },
    out: { type: "string", default: "docs/media/demo.mp4" },
  },
});

const WIDTH = 1040;
const HEIGHT = 660;

async function main() {
  const server = await createServer({ server: { port: 5199, strictPort: true }, logLevel: "error" });
  await server.listen();
  const videoDir = join("tmp", "video");
  rmSync(videoDir, { recursive: true, force: true });
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: WIDTH, height: HEIGHT },
    recordVideo: { dir: videoDir, size: { width: WIDTH, height: HEIGHT } },
  });
  const page = await context.newPage();
  const qs = new URLSearchParams({ model: args.model!, encoder: args.encoder!, seed: args.seed!, speed: args.speed! });
  // Warm a local model so the first decision does not include load time.
  if (!process.env.TYPESAFE_API_KEY) await fetch(`${process.env.VITE_DECISION_BASE_URL ?? "http://localhost:11434"}/api/generate`, {
    method: "POST",
    body: JSON.stringify({ model: args.model, keep_alive: -1 }),
  }).catch(() => undefined);
  await page.goto(`http://localhost:5199/?${qs}`);

  const deadline = Date.now() + Number(args.seconds) * 1000;
  let final = { phase: "", score: 0, left: 0 };
  while (Date.now() < deadline) {
    await page.waitForTimeout(1000);
    final = await page.evaluate(() => {
      const g = (window as unknown as { __pacman: { game: { phase: string; score: number; foodLeft: number } } }).__pacman.game;
      return { phase: g.phase, score: g.score, left: g.foodLeft };
    });
    if (final.phase === "gameover") {
      await page.waitForTimeout(2500);
      break;
    }
  }
  const video = page.video();
  await context.close();
  await browser.close();
  await server.close();
  const webm = await video!.path();
  mkdirSync(dirname(args.out!), { recursive: true });
  if (args.out!.endsWith(".webm")) {
    renameSync(webm, args.out!);
  } else {
    execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", webm, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", "-preset", "slow", "-movflags", "+faststart", args.out!]);
  }
  console.log(JSON.stringify({ out: args.out, ...final }));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
