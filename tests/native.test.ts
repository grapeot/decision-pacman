import { describe, expect, it } from "vitest";
import { DecisionApiError } from "../src/agent/client.ts";
import { buildChatRequest, moveGrammar, phi4MiniPrompt, renderChatDecision } from "../src/agent/llm.ts";
import { AgentLoop } from "../src/agent/loop.ts";
import { nativeChatPolicy, nativeJevFetch, nativePolicy, type NativeBridge } from "../src/agent/native.ts";
import { isPlayerId, PLAYERS } from "../src/agent/players.ts";
import { systemOnePolicy, type Policy } from "../src/agent/policies.ts";
import { renderDecision } from "../src/agent/prompt.ts";
import { findDecisionPoint } from "../src/engine/decision.ts";
import { createGame, step } from "../src/engine/game.ts";
import { ENCODERS } from "../src/encoders/index.ts";

function playingDecision() {
  const s = createGame(1);
  while (s.phase !== "playing") step(s);
  return { game: s, enc: ENCODERS.features.encode(s, findDecisionPoint(s)) };
}

const fakeBridge = (reply: object, sent: unknown[] = []): NativeBridge => ({
  async postMessage(message: unknown) {
    sent.push(message);
    return reply;
  },
});

describe("on-device chat prompt", () => {
  it("wraps exactly the text the llm: policy sends to Ollama in phi4-mini's template", () => {
    const { enc } = playingDecision();
    const req = buildChatRequest({ baseUrl: "http://localhost:11434", model: "phi4-mini" }, enc) as { messages: { content: string }[] };
    expect(renderChatDecision(enc)).toBe(`<|user|>${req.messages[0].content}<|end|><|assistant|>`);
    expect(phi4MiniPrompt("hi")).toBe("<|user|>hi<|end|><|assistant|>");
  });

  it("builds a grammar whose only values are the options, as JSON strings", () => {
    const g = moveGrammar(["up", "left", "back"]);
    expect(g).toContain('root ::= "{" space move-kv "}" space');
    expect(g).toContain('move-kv ::= "\\"move\\"" space ":" space move');
    expect(g).toContain('move ::= ("\\"up\\"" | "\\"left\\"" | "\\"back\\"") space');
    expect(g).not.toContain("down");
  });
});

describe("native policies", () => {
  it("scores the fine-tuned model from letter probabilities and names its player", async () => {
    const { enc } = playingDecision();
    const sent: unknown[] = [];
    const p = enc.keys.map((_, i) => (i === 1 ? 0.7 : 0.3 / (enc.keys.length - 1)));
    const d = await nativePolicy(fakeBridge({ probabilities: p, tokens: 380, ms: 42 }, sent), "ft").decide(enc);
    expect(sent[0]).toEqual({ player: "finetuned", prompt: renderDecision(enc), options: enc.keys.length });
    expect(d.choice).toBe(enc.keys[1]);
    expect(d.latencyMs).toBe(42);
  });

  it("reads the chat model's JSON answer and its probabilities", async () => {
    const { enc } = playingDecision();
    const sent: unknown[] = [];
    const move = enc.keys[0];
    const probabilities = enc.keys.map((k) => (k === move ? 0.9 : 0.1 / (enc.keys.length - 1)));
    const bridge = fakeBridge({ text: `{"move": "${move}"}`, probabilities, tokens: 290, ms: 900 }, sent);
    const d = await nativeChatPolicy(bridge, "phi4").decide(enc);
    expect(sent[0]).toEqual({ player: "phi4-mini", prompt: renderChatDecision(enc), grammar: moveGrammar(enc.keys), keys: enc.keys });
    expect(d.choice).toBe(move);
    expect(d.confidence).toBeCloseTo(0.9);
    expect(d.inputTokens).toBe(290);
  });

  it("rejects an illegal chat answer and reports native errors", async () => {
    const { enc } = playingDecision();
    await expect(nativeChatPolicy(fakeBridge({ text: '{"move": "sideways"}' }), "phi4").decide(enc)).rejects.toThrow(/illegal move/);
    await expect(nativeChatPolicy(fakeBridge({ error: "no model" }), "phi4").decide(enc)).rejects.toBeInstanceOf(DecisionApiError);
  });
});

describe("Jev through the native bridge", () => {
  const answer = { answers: { move: { choice: "up", probabilities: { up: 0.8, left: 0.2 }, confidence: 0.4 } }, usage: { input_tokens: 600 } };

  it("sends the hosted request body (no keep_alive) and parses the reply", async () => {
    const { enc } = playingDecision();
    const sent: { body: string }[] = [];
    const bridge = { async postMessage(m: unknown) { sent.push(m as { body: string }); return { status: 200, body: JSON.stringify(answer) }; } };
    const policy = systemOnePolicy({ baseUrl: "https://example.invalid", model: "jev-latest", hosted: true, fetch: nativeJevFetch(bridge) });
    const d = await policy.decide(enc);
    expect(d.choice).toBe("up");
    expect(d.inputTokens).toBe(600);
    const body = JSON.parse(sent[0].body);
    expect(body.model).toBe("jev-latest");
    expect(body).not.toHaveProperty("keep_alive");
  });

  it("surfaces a missing key or network failure as an API error, and HTTP errors by status", async () => {
    const { enc } = playingDecision();
    const missing = nativeJevFetch({ async postMessage() { return { error: "no key" }; } });
    await expect(systemOnePolicy({ baseUrl: "x", model: "jev-latest", hosted: true, fetch: missing }).decide(enc)).rejects.toMatchObject({ kind: "network", message: "no key" });
    const unauthorized = nativeJevFetch({ async postMessage() { return { status: 401, body: '{"error":"bad key"}' }; } });
    await expect(systemOnePolicy({ baseUrl: "x", model: "jev-latest", hosted: true, fetch: unauthorized }).decide(enc)).rejects.toMatchObject({ kind: "server" });
  });
});

describe("players", () => {
  it("knows the three app players and nothing else", () => {
    expect(Object.keys(PLAYERS)).toEqual(["finetuned", "phi4-mini", "jev"]);
    expect(isPlayerId("jev")).toBe(true);
    expect(isPlayerId("toString")).toBe(false);
  });
});

describe("switching players mid-request", () => {
  it("drops the answer of a policy that was replaced while it was thinking", async () => {
    const { game } = playingDecision();
    let release!: () => void;
    const slow: Policy = {
      name: "slow",
      decide: () => new Promise((resolve) => (release = () => resolve({ choice: "stale-answer", latencyMs: 1 }))),
    };
    const fast: Policy = { name: "fast", decide: async (enc) => ({ choice: enc.keys[0], latencyMs: 1 }) };
    const seen: string[] = [];
    const loop = new AgentLoop(
      { getState: () => game, apply: () => {}, onDecision: (r) => seen.push(`${r.model}:${r.choice}`), onError: (e) => { throw e; } },
      slow,
      ENCODERS.features,
    );
    loop.start();
    await new Promise((r) => setTimeout(r, 20));
    loop.policy = fast;
    release();
    await new Promise((r) => setTimeout(r, 120));
    loop.stop();
    expect(seen.length).toBeGreaterThan(0);
    expect(seen.every((m) => m.startsWith("fast:"))).toBe(true);
    expect(seen.some((m) => m.endsWith("stale-answer"))).toBe(false);
  });
});
