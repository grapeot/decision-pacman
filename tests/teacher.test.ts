import { describe, expect, it } from "vitest";
import { parseTeacherReply, TeacherParseError, teacherPrompt } from "../src/agent/teacher.ts";
import { createGame, step } from "../src/engine/game.ts";
import { findDecisionPoint } from "../src/engine/decision.ts";
import { ENCODERS } from "../src/encoders/index.ts";

describe("teacher reply parsing", () => {
  const keys = ["up", "left", "back"];

  it("reads the move and reason", () => {
    expect(parseTeacherReply('{"move": "left", "reason": "more pellets"}', keys)).toEqual({ move: "left", reason: "more pellets" });
  });

  it("uses the last JSON answer after any thinking text", () => {
    const text = 'I could go {"move": "up"} maybe... Final: {"move": "back", "reason": "ghost ahead"}';
    expect(parseTeacherReply(text, keys).move).toBe("back");
  });

  it("rejects moves that are not options", () => {
    expect(() => parseTeacherReply('{"move": "down", "reason": "x"}', keys)).toThrow(TeacherParseError);
  });

  it("rejects replies without an answer", () => {
    expect(() => parseTeacherReply("left seems fine", keys)).toThrow(TeacherParseError);
  });
});

describe("teacher prompt", () => {
  it("contains the rules, the student's state, and every option", () => {
    const s = createGame(1);
    while (s.phase !== "playing") step(s);
    const enc = ENCODERS.features.encode(s, findDecisionPoint(s));
    const prompt = teacherPrompt(enc);
    expect(prompt).toContain(enc.instructions);
    expect(prompt).toContain(JSON.stringify(enc.state));
    for (const k of enc.keys) expect(prompt).toContain(`- ${k}`);
  });
});

describe("teacherPrompt with lookahead", () => {
  it("adds one simulated outcome per option", () => {
    const s = createGame(1);
    while (s.phase !== "playing") step(s);
    const enc = ENCODERS.features.encode(s, findDecisionPoint(s));
    const outcomes = Object.fromEntries(enc.keys.map((k) => [k, { dies_after_s: null, pellets: 3, power_pellets: 0, ghosts_eaten: 0, points: 30 }]));
    const prompt = teacherPrompt(enc, { seconds: 5, outcomes });
    expect(prompt).toContain("Lookahead:");
    expect(prompt).toContain(JSON.stringify(outcomes));
    expect(teacherPrompt(enc)).not.toContain("Lookahead:");
  });
});
