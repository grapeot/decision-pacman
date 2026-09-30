// Generates labeled training states: a player drives the game, and the teacher
// labels every decision state in the same encoding the student will see.
//
//   npm run gen-data -- --player teacher --seconds 30 --seed 1000
//   npm run gen-data -- --player greedy --labeler teacher-think --games 5 --seed 1100
//
// The game waits for answers (lockstep clock). Train on seeds 1000 and up,
// validate on 900-999, and keep 100-199 for evaluation only.
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { loadDotEnv, makePolicy, policyEnvFromProcess } from "../src/agent/factory.ts";
import type { TeacherDecision } from "../src/agent/teacher.ts";
import { ENCODERS } from "../src/encoders/index.ts";
import { playGame, percentile, type GameSummary } from "../src/sim/runner.ts";

loadDotEnv();

const { values: args } = parseArgs({
  options: {
    player: { type: "string", default: "teacher" },
    labeler: { type: "string", default: "teacher" },
    encoder: { type: "string", default: "features" },
    games: { type: "string", default: "1" },
    seed: { type: "string", default: "1000" },
    seconds: { type: "string", default: "180" },
    speed: { type: "string", default: "1" },
    out: { type: "string", default: "data/teacher" },
  },
});

const EVAL_SEEDS = { min: 100, max: 199 };
export const SCHEMA = "decision-pacman/teacher-v1";

async function main() {
  const env = policyEnvFromProcess();
  const seed0 = Number(args.seed);
  const games = Number(args.games);
  if (seed0 <= EVAL_SEEDS.max && seed0 + games - 1 >= EVAL_SEEDS.min) {
    throw new Error(`seeds ${EVAL_SEEDS.min}-${EVAL_SEEDS.max} are reserved for evaluation`);
  }
  const encoder = ENCODERS[args.encoder!];
  if (!encoder) throw new Error(`unknown encoder ${args.encoder}`);

  // Include the encoder and pid so runs started in the same millisecond never share a directory.
  const tag = `${new Date().toISOString().replace(/[:.]/g, "-")}_${args.player}_${args.labeler}_${args.encoder}_s${seed0}_p${process.pid}`.replace(/[^\w.-]/g, "_");
  const dir = join(args.out!, tag);
  mkdirSync(dir, { recursive: true });
  const dataPath = join(dir, "states.jsonl");
  writeFileSync(dataPath, "");

  let labeled = 0;
  let unlabeled = 0;
  const labelLatency: number[] = [];
  const labelCounts: Record<string, number> = {};
  const errors: string[] = [];
  const summaries: GameSummary[] = [];
  const started = Date.now();

  for (let g = 0; g < games; g++) {
    const seed = seed0 + g;
    const player = makePolicy(args.player!, seed, env);
    const labeler = args.labeler === args.player ? player : makePolicy(args.labeler!, seed, env);
    const sum = await playGame({
      seed,
      policy: player,
      labeler,
      encoder,
      clock: "lockstep",
      speed: Number(args.speed),
      maxSeconds: Number(args.seconds),
      onDecision: (ev) => {
        const label = ev.label as TeacherDecision | null | undefined;
        if (!label) {
          unlabeled++;
          if (ev.error) errors.push(ev.error);
          return;
        }
        labeled++;
        labelLatency.push(label.latencyMs);
        labelCounts[label.choice] = (labelCounts[label.choice] ?? 0) + 1;
        appendFileSync(
          dataPath,
          JSON.stringify({
            schema: SCHEMA,
            seed,
            tick: ev.askedTick,
            player: player.name,
            labeler: labeler.name,
            encoder: ev.enc.encoder,
            junction: ev.dp.key,
            options: ev.enc.keys,
            state: ev.enc.state,
            instructions: ev.enc.instructions,
            criteria: ev.enc.criteria,
            label: label.choice,
            reason: label.reason ?? null,
            label_latency_ms: Math.round(label.latencyMs),
            player_choice: ev.decision?.choice ?? null,
          }) + "\n",
        );
      },
    });
    summaries.push(sum);
    console.log(
      `seed ${seed}: ${sum.secondsPlayed}s played, score ${sum.score}, pellets ${sum.pellets}, deaths ${sum.deaths}, ` +
        `${sum.decisions} decisions, ${sum.failures} failed`,
    );
  }

  const report = {
    schema: SCHEMA,
    player: args.player,
    labeler: args.labeler,
    encoder: args.encoder,
    seeds: [seed0, seed0 + games - 1],
    wallSeconds: Math.round((Date.now() - started) / 1000),
    gameSeconds: summaries.reduce((a, s) => a + s.secondsPlayed, 0),
    labeled,
    unlabeled,
    labelsPerWallSecond: Math.round((labeled / Math.max(1, (Date.now() - started) / 1000)) * 100) / 100,
    labelLatencyP50: Math.round(percentile(labelLatency, 0.5)),
    labelLatencyP90: Math.round(percentile(labelLatency, 0.9)),
    labelCounts,
    errors: errors.slice(0, 10),
    games: summaries,
  };
  writeFileSync(join(dir, "report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ ...report, games: undefined, errors: report.errors.length }));
  console.log(`wrote ${dataPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
