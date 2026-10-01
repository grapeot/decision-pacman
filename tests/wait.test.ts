import { describe, expect, it } from "vitest";
import { shouldHold } from "../src/agent/loop.ts";
import { findDecisionPoint } from "../src/engine/decision.ts";
import { createGame, step } from "../src/engine/game.ts";

function playing(seed = 1) {
  const s = createGame(seed);
  while (s.phase !== "playing") step(s);
  return s;
}

describe("shouldHold (wait for the model)", () => {
  it("never holds without a question in flight", () => {
    expect(shouldHold(playing(), null)).toBe(false);
  });

  it("holds only once Pac-Man is about to reach the junction being asked about", () => {
    const s = playing();
    const pending = findDecisionPoint(s);
    let held = false;
    for (let i = 0; i < 300 && !held; i++) {
      held = shouldHold(s, pending);
      if (!held) step(s);
    }
    expect(held).toBe(true);
    const dp = findDecisionPoint(s);
    expect(dp.key).toBe(pending.key);
    expect(dp.distance).toBeLessThan(0.5);
  });

  it("does not hold for a junction Pac-Man is not heading to", () => {
    const s = playing();
    const other = { ...findDecisionPoint(s), key: "elsewhere" };
    expect(shouldHold(s, other)).toBe(false);
  });
});
