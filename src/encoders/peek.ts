// The features encoder plus one simulated future per option, for models that
// cannot run the engine themselves (for example a hosted decision API). The
// rollout is the oracle's: play the option, then greedy moves for a few
// seconds. The encoder reports what happened, never which option is better.
import type { DecisionPoint } from "../engine/decision.ts";
import { TPS } from "../engine/game.ts";
import type { GameState } from "../engine/types.ts";
import { peek, type PeekOutcome } from "../agent/oracle.ts";
import { featuresEncoder } from "./features.ts";
import type { EncodedDecision, Encoder } from "./types.ts";

function lookaheadInstructions(seconds: number): string {
  return (
    ` Lookahead (lookahead_${seconds}s): for each option, the game engine played it and then simple greedy moves ` +
    `for the next ${seconds} s. Normal ghost moves are exact; frightened ghosts turn at random. ` +
    "It shows one possible future, not the only one. dies_after_s = seconds until Pac-Man loses a life " +
    "(null if he survives), pellets and power_pellets = pellets eaten, ghosts_eaten = frightened ghosts eaten, " +
    "points = points gained."
  );
}

/** `features` with a per-option rollout outcome of `seconds` game seconds under `lookahead_<seconds>s`. */
export function featuresPeekEncoder(seconds: number): Encoder {
  const horizonTicks = Math.round(seconds * TPS);
  const extra = lookaheadInstructions(seconds);
  const name = `features-peek${seconds}s`;
  return {
    name,
    encode(s: GameState, dp: DecisionPoint): EncodedDecision {
      const base = featuresEncoder.encode(s, dp);
      const outcomes: Record<string, PeekOutcome> = {};
      // A single option needs no question, so skip the rollouts.
      if (base.keys.length > 1) for (const key of base.keys) outcomes[key] = peek(s, base, key, horizonTicks);
      const state = { ...(base.state as Record<string, unknown>), [`lookahead_${seconds}s`]: outcomes };
      return { ...base, encoder: name, state, instructions: base.instructions + extra };
    },
  };
}

export const featuresPeek5sEncoder = featuresPeekEncoder(5);
