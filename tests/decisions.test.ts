import { afterEach, describe, expect, it } from "vitest";
import { buildDecisionsRequest, DecisionsAnswerError, decisionsPolicy, parseDecisionsAnswer } from "../src/agent/decisions.ts";
import { makePolicy } from "../src/agent/factory.ts";
import { findDecisionPoint } from "../src/engine/decision.ts";
import { createGame, step } from "../src/engine/game.ts";
import { ENCODERS } from "../src/encoders/index.ts";

function junction() {
  const s = createGame(1);
  while (s.phase !== "playing") step(s);
  return ENCODERS.features.encode(s, findDecisionPoint(s));
}

const reply = (choice: string, probs: [string, number][], extra: object = {}) => ({
  model: "gpt-6-luna",
  answers: [{ type: "choice", name: "move", choice, probabilities: probs.map(([value, probability]) => ({ value, probability })), confidence: 0.5, ...extra }],
  usage: { input_tokens: 360, output_tokens: 0 },
});

describe("OpenAI Decisions request", () => {
  it("sends the encoded state as input and the legal options as choices of one question", () => {
    const enc = junction();
    const req = buildDecisionsRequest("gpt-6-luna", enc) as Record<string, any>;
    expect(req.model).toBe("gpt-6-luna");
    expect(req.input).toBe(JSON.stringify(enc.state));
    expect(req.questions).toHaveLength(1);
    expect(req.questions[0]).toMatchObject({ type: "choice", name: "move", instructions: enc.instructions });
    expect(req.questions[0].choices.map((c: { value: string }) => c.value)).toEqual(enc.keys);
  });

  it("passes option descriptions and leaves them out when the name describes itself", () => {
    const enc = { state: { a: 1 }, instructions: "Which way?", criteria: { up: null, back: "turn around now" }, keys: ["up", "back"] };
    const req = buildDecisionsRequest("gpt-6-luna", enc as never) as Record<string, any>;
    expect(req.questions[0].choices).toEqual([{ value: "up" }, { value: "back", description: "turn around now" }]);
  });
});

describe("parseDecisionsAnswer", () => {
  const keys = ["up", "left", "back"];

  it("reads the choice, the per-option probabilities, and the input tokens", () => {
    const a = parseDecisionsAnswer(reply("left", [["up", 0.2], ["left", 0.7], ["back", 0.1]]), keys);
    expect(a).toEqual({ choice: "left", probabilities: { up: 0.2, left: 0.7, back: 0.1 }, confidence: 0.5, inputTokens: 360 });
  });

  it("rejects a choice outside the legal options and a refusal", () => {
    expect(() => parseDecisionsAnswer(reply("down", [["down", 1]]), keys)).toThrow(DecisionsAnswerError);
    expect(() => parseDecisionsAnswer({ answers: [{ type: "refusal", name: "move" }] }, keys)).toThrow(/refusal/);
    expect(() => parseDecisionsAnswer({ answers: [] }, keys)).toThrow(DecisionsAnswerError);
  });
});

describe("decisionsPolicy", () => {
  it("posts to /v1/decisions with the bearer key and returns the decision", async () => {
    const enc = junction();
    const calls: { url: string; init: RequestInit }[] = [];
    const fetchFn = (async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      return new Response(JSON.stringify(reply(enc.keys[0], enc.keys.map((k, i) => [k, i === 0 ? 0.9 : 0.1 / (enc.keys.length - 1)]))));
    }) as unknown as typeof fetch;
    const policy = decisionsPolicy({ baseUrl: "https://api.example.invalid/", model: "gpt-6-luna", apiKey: "k", fetchFn });
    const d = await policy.decide(enc);
    expect(policy.name).toBe("openai:gpt-6-luna");
    expect(calls[0].url).toBe("https://api.example.invalid/v1/decisions");
    expect((calls[0].init.headers as Record<string, string>).Authorization).toBe("Bearer k");
    expect(d.choice).toBe(enc.keys[0]);
    expect(d.probabilities![enc.keys[0]]).toBe(0.9);
    expect(d.inputTokens).toBe(360);
  });

  it("throws on an HTTP error", async () => {
    const fetchFn = (async () => new Response('{"error":{"message":"bad"}}', { status: 400 })) as unknown as typeof fetch;
    await expect(decisionsPolicy({ baseUrl: "x", model: "gpt-6-luna", fetchFn }).decide(junction())).rejects.toThrow(/HTTP 400/);
  });
});

describe("makePolicy openai:<model>", () => {
  const saved = process.env.OPENAI_API_KEY;
  afterEach(() => {
    if (saved === undefined) delete process.env.OPENAI_API_KEY;
    else process.env.OPENAI_API_KEY = saved;
  });

  it("needs OPENAI_API_KEY", () => {
    delete process.env.OPENAI_API_KEY;
    expect(() => makePolicy("openai:gpt-6-luna", 100, { decisionBaseUrl: "http://localhost:11434" })).toThrow(/OPENAI_API_KEY/);
  });

  it("makes a Decisions API player for the named model", () => {
    process.env.OPENAI_API_KEY = "test";
    expect(makePolicy("openai:gpt-6-luna", 100, { decisionBaseUrl: "http://localhost:11434" }).name).toBe("openai:gpt-6-luna");
  });
});
