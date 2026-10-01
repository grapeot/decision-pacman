// "Play with the model's eyes": the player sees only the features encoding and answers at junctions.
import { findDecisionPoint, pacTilesPerSecond, type DecisionPoint } from "../src/engine/decision.ts";
import { TPS } from "../src/engine/game.ts";
import { opposite, type Dir, type GameState } from "../src/engine/types.ts";
import { featuresEncoder } from "../src/encoders/features.ts";
import type { EncodedDecision, OptionKey } from "../src/encoders/types.ts";

/** A real choice: more than one way out of the junction ahead (turning back does not count). */
export function isChoice(dp: DecisionPoint): boolean {
  return dp.exits.length > 1;
}

/**
 * Whether the game should hold this tick: Pac-Man is about to reach a junction with a real
 * choice and the player has not answered for it. Same rule as the "Wait" switch (shouldHold).
 */
export function holdForPlayer(s: GameState, answered: string | null): boolean {
  if (s.phase !== "playing") return false;
  const dp = findDecisionPoint(s);
  if (!isChoice(dp) || dp.key === answered) return false;
  return dp.distance <= pacTilesPerSecond(s) / TPS + 0.05;
}

/** The encoding the model would get for the junction ahead. */
export function encodeAhead(s: GameState): EncodedDecision {
  return featuresEncoder.encode(s, findDecisionPoint(s));
}

/** A key press mapped to an option: the opposite of the heading means "back". */
export function keyToOption(d: Dir, s: GameState, keys: readonly OptionKey[]): OptionKey | null {
  const opt: OptionKey = s.pac.dir && d === opposite(s.pac.dir) ? "back" : d;
  return keys.includes(opt) ? opt : null;
}

/**
 * The state as compact JSON: top-level fields first, then one line per option, the way a person
 * can scan it. The content is exactly what the model receives.
 */
export function compactState(state: unknown): string {
  const { options, ...rest } = state as { options: Record<string, unknown> } & Record<string, unknown>;
  const head = Object.entries(rest).map(([k, v]) => `  ${JSON.stringify(k)}: ${JSON.stringify(v)}`);
  const opts = Object.entries(options ?? {}).map(([k, v]) => `    ${JSON.stringify(k)}: ${JSON.stringify(v).replace(/,"/g, ', "').replace(/":/g, '": ')}`);
  return `{\n${head.join(",\n")},\n  "options": {\n${opts.join(",\n")}\n  }\n}`;
}
