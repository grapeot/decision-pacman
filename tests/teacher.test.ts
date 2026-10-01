import { describe, expect, it } from "vitest";
import { parseTeacherReply, TeacherParseError, teacherPolicy, teacherPrompt, type TeacherDecision } from "../src/agent/teacher.ts";
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

describe("teacher answer cache", () => {
  it("asks the server once per distinct prompt", async () => {
    let calls = 0;
    const fetchFn = (async () => {
      calls++;
      return new Response(JSON.stringify({ choices: [{ message: { content: '{"move": "back", "reason": "r"}' } }], usage: { completion_tokens: 9 } }));
    }) as unknown as typeof fetch;
    const cache = new Map();
    const policy = teacherPolicy({ baseUrl: "http://teacher.invalid/v1", model: "m", cache, fetchFn });
    const s = createGame(1);
    while (s.phase !== "playing") step(s);
    const enc = ENCODERS.features.encode(s, findDecisionPoint(s));
    const first = (await policy.decide(enc, undefined, s)) as TeacherDecision;
    const second = (await policy.decide(enc, undefined, s)) as TeacherDecision;
    expect(calls).toBe(1);
    expect(first.cached).toBeUndefined();
    expect(second.cached).toBe(true);
    expect(second.choice).toBe(first.choice);
    expect(cache.size).toBe(1);
  });
});
