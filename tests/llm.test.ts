import { describe, expect, it } from "vitest";
import { buildChatRequest, LlmAnswerError, moveProbabilities, moveSchema, parseMove } from "../src/agent/llm.ts";
import { createGame, step } from "../src/engine/game.ts";
import { findDecisionPoint } from "../src/engine/decision.ts";
import { ENCODERS } from "../src/encoders/index.ts";

const keys = ["up", "left", "back"];

describe("plain LLM request", () => {
  it("constrains the move to the legal options, with thinking off and temperature 0", () => {
    const s = createGame(1);
    while (s.phase !== "playing") step(s);
    const enc = ENCODERS.features.encode(s, findDecisionPoint(s));
    const req = buildChatRequest({ baseUrl: "http://localhost:11434", model: "qwen3.5:4b" }, enc) as Record<string, any>;
    expect(req.format).toEqual(moveSchema(enc.keys));
    expect(req.format.properties.move.enum).toEqual(enc.keys);
    expect(req).toMatchObject({ think: false, stream: false, keep_alive: -1, options: { temperature: 0 } });
    expect(req.messages[0].content).toContain(JSON.stringify(enc.state));
    expect(req.messages[0].content).not.toContain('"reason"');
  });
});

describe("parseMove", () => {
  it("reads a legal move", () => {
    expect(parseMove('{"move": "left"}', keys)).toBe("left");
  });

  it("rejects illegal or unparseable answers", () => {
    expect(() => parseMove('{"move": "down"}', keys)).toThrow(LlmAnswerError);
    expect(() => parseMove("left", keys)).toThrow(LlmAnswerError);
  });
});

describe("moveProbabilities", () => {
  const lp = (token: string, top: [string, number][]) => ({
    token,
    logprob: 0,
    top_logprobs: top.map(([t, p]) => ({ token: t, logprob: Math.log(p) })),
  });
  const content = '{"move": "left"}';
  const logprobs = [
    lp('{"', [['{"', 1]]),
    lp("move", [["move", 1]]),
    lp('":', [['":', 1]]),
    lp(' "', [[' "', 1]]),
    lp("left", [["left", 0.6], ["up", 0.2], ["right", 0.1], ["back", 0.05], ["backward", 0.05]]),
    lp('"}', [['"}', 1]]),
  ];

  it("reads the options' mass at the move token and renormalizes over legal options", () => {
    const p = moveProbabilities(content, logprobs, keys)!;
    expect(p.left).toBeCloseTo(0.6 / 0.85);
    expect(p.up).toBeCloseTo(0.2 / 0.85);
    expect(p.back).toBeCloseTo(0.05 / 0.85);
  });

  it("returns undefined without logprobs", () => {
    expect(moveProbabilities(content, undefined, keys)).toBeUndefined();
  });
});
