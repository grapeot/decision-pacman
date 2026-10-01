// Exports real game decisions from a headless `llm:<model>` run (features encoder) as on-device chat
// prompts, with Ollama's recorded answers, for checking the iOS app's chat engine on a Mac
// (ios/scripts/mac_check.sh chat). The prompt is rebuilt from the logged state exactly as the page builds it.
//   npx tsx scripts/export_chat_prompts.ts --run runs/<dir>/decisions.jsonl --n 40 --out ios/mac_check/chat_prompts.json
import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { moveGrammar, renderChatDecision } from "../src/agent/llm.ts";
import { findDecisionPoint } from "../src/engine/decision.ts";
import { createGame, step } from "../src/engine/game.ts";
import { ENCODERS } from "../src/encoders/index.ts";
import type { EncodedDecision, OptionKey } from "../src/encoders/types.ts";

interface Logged {
  seed: number;
  keys: OptionKey[];
  choice: string;
  probabilities?: Record<string, number>;
  inputTokens?: number;
  latencyMs: number;
  state: unknown;
}

const { values } = parseArgs({
  options: { run: { type: "string" }, n: { type: "string", default: "40" }, out: { type: "string" } },
});
if (!values.run || !values.out) throw new Error("usage: --run <decisions.jsonl> --out <file.json> [--n 40]");

// The features encoder's instructions and criteria do not depend on the state.
const sample = createGame(1);
while (sample.phase !== "playing") step(sample);
const template = ENCODERS.features.encode(sample, findDecisionPoint(sample));

const rows = readFileSync(values.run, "utf8").trim().split("\n").map((l) => JSON.parse(l) as Logged);
const n = Math.min(Number(values.n), rows.length);
const picked = Array.from({ length: n }, (_, i) => rows[Math.floor((i * rows.length) / n)]);

const out = picked.map((r) => {
  const criteria = Object.fromEntries(r.keys.map((k) => [k, k === "back" ? "turn around now" : null]));
  const enc: EncodedDecision = { ...template, state: r.state, keys: r.keys, criteria };
  return {
    prompt: renderChatDecision(enc),
    grammar: moveGrammar(r.keys),
    keys: r.keys,
    reference: {
      choice: r.choice,
      probabilities: r.probabilities ? r.keys.map((k) => r.probabilities![k] ?? 0) : null,
      inputTokens: r.inputTokens ?? null,
    },
  };
});
writeFileSync(values.out, JSON.stringify(out, null, 1) + "\n");
console.log(`wrote ${out.length} prompts from ${rows.length} decisions to ${values.out}`);
