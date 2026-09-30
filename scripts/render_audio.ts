// Renders the music and sound-effect tour to a WAV in a headless browser,
// so levels can be measured and the result heard without playing the game.
//
//   npm run render-audio -- --out tmp/audio_demo.wav
import { writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { chromium } from "playwright";
import { createServer } from "vite";

const { values: args } = parseArgs({ options: { out: { type: "string", default: "tmp/audio_demo.wav" } } });

async function main() {
  const server = await createServer({ server: { port: 5198, strictPort: true }, logLevel: "error" });
  await server.listen();
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    await page.goto("http://localhost:5198/");
    const b64 = await page.evaluate(async (path) => {
      // Imported by Vite inside the page, not by Node.
      const m = (await import(path)) as typeof import("../src/audio/render.ts");
      const bytes: Uint8Array = m.toWav(await m.renderDemo());
      let s = "";
      for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
      return btoa(s);
    }, "/src/audio/render.ts");
    writeFileSync(args.out!, Buffer.from(b64, "base64"));
    console.log(`wrote ${args.out}`);
  } finally {
    await browser.close();
    await server.close();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
