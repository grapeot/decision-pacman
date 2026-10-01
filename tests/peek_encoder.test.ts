import { describe, expect, it } from "vitest";
import { createGame, step } from "../src/engine/game.ts";
import { findDecisionPoint } from "../src/engine/decision.ts";
import { ENCODERS } from "../src/encoders/index.ts";
import { peek, type PeekOutcome } from "../src/agent/oracle.ts";
import type { GameState } from "../src/engine/types.ts";

function playing(seed = 1): GameState {
  const s = createGame(seed);
  while (s.phase !== "playing") step(s);
  return s;
}

describe("features-peek5s encoder", () => {
  const enc5 = ENCODERS["features-peek5s"];

  it("is the features encoding plus one 5 s rollout outcome per option", () => {
    const s = playing();
    const dp = findDecisionPoint(s);
    const base = ENCODERS.features.encode(s, dp);
    const enc = enc5.encode(s, dp);
    expect(enc.keys).toEqual(base.keys);
    const state = enc.state as Record<string, unknown>;
    expect(state.options).toEqual((base.state as Record<string, unknown>).options);
    const outcomes = state.lookahead_5s as Record<string, PeekOutcome>;
    expect(Object.keys(outcomes).sort()).toEqual([...enc.keys].sort());
    for (const key of enc.keys) expect(outcomes[key]).toEqual(peek(s, base, key, 150));
    expect(enc.instructions.startsWith(base.instructions)).toBe(true);
    expect(enc.instructions).toMatch(/one possible future/);
  });

  it("does not change the live game", () => {
    const s = playing();
    const before = JSON.stringify({ ...s, food: [...s.food] });
    enc5.encode(s, findDecisionPoint(s));
    expect(JSON.stringify({ ...s, food: [...s.food] })).toBe(before);
  });

  it("reports facts and never verdicts", () => {
    const s = playing();
    for (let i = 0; i < 600; i++) step(s);
    const text = JSON.stringify(enc5.encode(s, findDecisionPoint(s)).state);
    expect(text).not.toMatch(/safe|danger|recommend|best|score|value/i);
  });

  it("shows a death in the future of walking into a ghost", () => {
    const s = playing();
    for (const g of s.ghosts) Object.assign(g, { state: "house", x: 13.5, y: 14 });
    s.dotsEatenThisLife = -1000;
    s.ticksSinceDot = -100000;
    Object.assign(s.pac, { x: 9, y: 5, dir: "left", lastDir: "left" });
    // Blinky sits on the next junction (6,5), heading toward Pac-Man.
    Object.assign(s.ghosts[0], { x: 6, y: 5, dir: "right", state: "active", frightened: false });
    const outcomes = (enc5.encode(s, findDecisionPoint(s)).state as { lookahead_5s: Record<string, PeekOutcome> }).lookahead_5s;
    const forward = Object.entries(outcomes).filter(([k]) => k !== "back");
    expect(forward.every(([, o]) => o.dies_after_s !== null)).toBe(true);
    expect(outcomes.back.dies_after_s).toBeNull();
  });
});
