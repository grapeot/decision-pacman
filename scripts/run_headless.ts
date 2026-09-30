// Plays seeded games without rendering and writes one JSONL record per decision.
//
//   npm run headless -- --policy tev1:4b --games 5 --seed 100 --clock realtime
//   npm run headless -- --policy greedy --games 20
//
// realtime: the model's measured latency turns into game ticks that pass before
//           the answer applies, as in the browser.
// lockstep: the game waits for every answer (an upper bound on decision quality).
import { mkdirSync, writeFileSync, appendFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { createGame, step, TPS } from "../src/engine/game.ts";
import { findDecisionPoint, isStillAhead } from "../src/engine/decision.ts";
import { LatencyEstimate, lookaheadTiles } from "../src/agent/loop.ts";
import { ENCODERS, optionToInput } from "../src/encoders/index.ts";
import { greedyPolicy, randomPolicy, systemOnePolicy, type Policy } from "../src/agent/policies.ts";
import type { GameState, StepInput } from "../src/engine/types.ts";

const { values: args } = parseArgs({
  options: {
    policy: { type: "string", default: "tev1:4b" },
    "base-url": { type: "string", default: process.env.VITE_DECISION_BASE_URL ?? "http://localhost:11434" },
    encoder: { type: "string", default: "features" },
    clock: { type: "string", default: "realtime" },
    games: { type: "string", default: "3" },
    seed: { type: "string", default: "100" },
    speed: { type: "string", default: "1" },
    "max-seconds": { type: "string", default: "180" },
    out: { type: "string", default: "runs" },
  },
});

const clock = args.clock as "realtime" | "lockstep";
if (clock !== "realtime" && clock !== "lockstep") throw new Error("--clock must be realtime or lockstep");

function makePolicy(name: string, seed: number): Policy {
  if (name === "random") return randomPolicy(seed);
  if (name === "greedy") return greedyPolicy();
  return systemOnePolicy({ baseUrl: args["base-url"]!, model: name });
}

interface GameSummary {
  seed: number;
  score: number;
  pellets: number;
  ghostsEaten: number;
  deaths: number;
  level: number;
  secondsPlayed: number;
  decisions: number;
  stale: number;
  latencyP50: number;
  latencyP90: number;
  endedBy: "gameover" | "time";
}

function percentile(xs: number[], p: number): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
}

async function playGame(seed: number, policy: Policy, logPath: string): Promise<GameSummary> {
  const encName = policy.requiresEncoder ?? args.encoder!;
  const encoder = ENCODERS[encName];
  if (!encoder) throw new Error(`unknown encoder ${encName}`);
  const s: GameState = createGame(seed, { speed: Number(args.speed) });
  const maxTicks = Number(args["max-seconds"]) * TPS;
  const latencies: number[] = [];
  let decisions = 0;
  let stale = 0;
  const estimate = new LatencyEstimate(clock === "realtime" ? 250 : 0);

  const advance = (n: number, input?: StepInput | null) => {
    for (let i = 0; i < n && s.phase !== "gameover"; i++) step(s, i === 0 && input ? input : {});
  };

  while (s.phase !== "gameover" && s.stats.ticksPlaying < maxTicks) {
    if (s.phase !== "playing") {
      advance(1);
      continue;
    }
    const lookahead = clock === "realtime" ? lookaheadTiles(s, estimate.value) : 0;
    const dp = findDecisionPoint(s, lookahead);
    const enc = encoder.encode(s, dp);
    if (enc.keys.length === 1) {
      advance(1, optionToInput(enc.keys[0], dp));
      continue;
    }
    const askedTick = s.tick;
    const d = await policy.decide(enc);
    decisions++;
    latencies.push(d.latencyMs);
    estimate.update(d.latencyMs);
    const input = optionToInput(d.choice, dp);
    let isStale = false;
    if (clock === "realtime") {
      advance(Math.max(0, Math.round((d.latencyMs / 1000) * TPS)));
      isStale = s.phase !== "playing" || !isStillAhead(s, dp, lookahead);
      if (isStale) stale++;
      advance(1, isStale ? null : input);
    } else {
      advance(1, input);
      // Ask again after passing the junction, or every 8 ticks to react to ghosts.
      for (let i = 0; i < 7 && s.phase === "playing" && findDecisionPoint(s).key === dp.key; i++) advance(1);
    }
    appendFileSync(
      logPath,
      JSON.stringify({
        seed,
        askedTick,
        answeredTick: s.tick,
        junction: dp.key,
        keys: enc.keys,
        choice: d.choice,
        probabilities: d.probabilities,
        confidence: d.confidence,
        latencyMs: Math.round(d.latencyMs),
        inputTokens: d.inputTokens,
        stale: isStale,
        state: enc.state,
      }) + "\n",
    );
  }

  return {
    seed,
    score: s.score,
    pellets: s.stats.pelletsEaten,
    ghostsEaten: s.stats.ghostsEaten,
    deaths: s.stats.deaths,
    level: s.level,
    secondsPlayed: Math.round(s.stats.ticksPlaying / TPS),
    decisions,
    stale,
    latencyP50: Math.round(percentile(latencies, 0.5)),
    latencyP90: Math.round(percentile(latencies, 0.9)),
    endedBy: s.phase === "gameover" ? "gameover" : "time",
  };
}

async function main() {
  const games = Number(args.games);
  const seed0 = Number(args.seed);
  const tag = `${new Date().toISOString().replace(/[:.]/g, "-")}_${args.policy!.replace(/[^\w.-]/g, "_")}_${args.encoder}_${clock}`;
  const dir = join(args.out!, tag);
  mkdirSync(dir, { recursive: true });
  const logPath = join(dir, "decisions.jsonl");
  writeFileSync(logPath, "");

  const summaries: GameSummary[] = [];
  for (let g = 0; g < games; g++) {
    const seed = seed0 + g;
    const policy = makePolicy(args.policy!, seed);
    const sum = await playGame(seed, policy, logPath);
    summaries.push(sum);
    console.log(
      `seed ${seed}: score ${sum.score}, pellets ${sum.pellets}/244, ghosts ${sum.ghostsEaten}, deaths ${sum.deaths}, ` +
        `level ${sum.level}, ${sum.secondsPlayed}s, ${sum.decisions} decisions (${sum.stale} stale), p50 ${sum.latencyP50} ms`,
    );
  }
  const mean = (k: keyof GameSummary) => summaries.reduce((a, s) => a + (s[k] as number), 0) / summaries.length;
  const overall = {
    policy: args.policy,
    encoder: args.encoder,
    clock,
    speed: Number(args.speed),
    games,
    seed0,
    meanScore: Math.round(mean("score")),
    meanPellets: Math.round(mean("pellets") * 10) / 10,
    meanSeconds: Math.round(mean("secondsPlayed")),
    meanLatencyP50: Math.round(mean("latencyP50")),
    staleRate: Math.round((summaries.reduce((a, s) => a + s.stale, 0) / Math.max(1, summaries.reduce((a, s) => a + s.decisions, 0))) * 1000) / 1000,
  };
  writeFileSync(join(dir, "summary.json"), JSON.stringify({ overall, games: summaries }, null, 2));
  console.log(JSON.stringify(overall));
  console.log(`wrote ${dir}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
