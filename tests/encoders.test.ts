import { describe, expect, it } from "vitest";
import { createGame, step } from "../src/engine/game.ts";
import { findDecisionPoint } from "../src/engine/decision.ts";
import { ENCODERS, optionToInput } from "../src/encoders/index.ts";
import { greedyPolicy } from "../src/agent/policies.ts";
import type { GameState } from "../src/engine/types.ts";

function playing(): GameState {
  const s = createGame(1);
  while (s.phase !== "playing") step(s);
  return s;
}

describe("features encoder", () => {
  it("offers exactly the junction exits plus back", () => {
    const s = playing();
    const dp = findDecisionPoint(s);
    const enc = ENCODERS.features.encode(s, dp);
    expect(enc.keys.sort()).toEqual(["back", "left", "up"]);
    expect(Object.keys(enc.criteria).sort()).toEqual(["back", "left", "up"]);
    const opts = (enc.state as { options: Record<string, unknown> }).options;
    expect(Object.keys(opts).sort()).toEqual(["back", "left", "up"]);
  });

  it("reports facts and never verdicts", () => {
    const s = playing();
    const text = JSON.stringify(ENCODERS.features.encode(s, findDecisionPoint(s)).state);
    expect(text).not.toMatch(/safe|danger|recommend|best|score/i);
  });

  it("stays within the token budget (character proxy)", () => {
    const s = playing();
    for (let i = 0; i < 600; i++) step(s);
    const text = JSON.stringify(ENCODERS.features.encode(s, findDecisionPoint(s)).state);
    expect(text.length).toBeLessThan(700);
  });

  it("sees a ghost that is close through one exit", () => {
    const s = playing();
    const dp = findDecisionPoint(s); // junction (12,23), exits left and up
    Object.assign(s.ghosts[0], { x: 12, y: 21, state: "active", frightened: false, dir: "down" });
    const enc = ENCODERS.features.encode(s, dp);
    const opts = (enc.state as { options: Record<string, { ghost: number | null; ghost_coming?: boolean }> }).options;
    expect(opts.up.ghost).not.toBeNull();
    expect(opts.up.ghost!).toBeLessThan(5);
    expect(opts.up.ghost_coming).toBe(true);
  });

  it("maps back to an immediate reverse and exits to turns planned at the junction", () => {
    const s = playing();
    const dp = findDecisionPoint(s);
    expect(optionToInput("back", dp)).toEqual({ reverse: true });
    expect(optionToInput("up", dp)).toEqual({ turn: { x: 12, y: 23, dir: "up" } });
  });

  it("feeds the greedy baseline", async () => {
    const s = playing();
    const enc = ENCODERS.features.encode(s, findDecisionPoint(s));
    const d = await greedyPolicy().decide(enc);
    expect(enc.keys).toContain(d.choice);
  });
});

describe("ascii window encoder", () => {
  it("renders an 11x11 view with Pac-Man at the center", () => {
    const s = playing();
    const enc = ENCODERS["ascii-window"].encode(s, findDecisionPoint(s));
    const rows = (enc.state as string).split("\n").slice(1);
    expect(rows).toHaveLength(11);
    expect(rows[5][5]).toBe("P");
  });
});
