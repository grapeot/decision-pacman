import { describe, expect, it } from "vitest";
import { buildRequest } from "../src/agent/client.ts";

const enc = { state: "{}", instructions: "Which way?", criteria: { up: "go up", left: "go left" }, keys: ["up", "left"] };

describe("buildRequest", () => {
  it("keeps the model loaded on a local Ollama endpoint", () => {
    expect(buildRequest({ baseUrl: "http://localhost:11434", model: "tev1:4b" }, enc as never)).toMatchObject({ keep_alive: -1 });
  });

  it("leaves out keep_alive for hosted endpoints, which reject it", () => {
    expect(buildRequest({ baseUrl: "https://api.typesafe.ai", model: "jev-latest", apiKey: "k" }, enc as never)).not.toHaveProperty("keep_alive");
    expect(buildRequest({ baseUrl: "/decide", model: "jev-latest", hosted: true }, enc as never)).not.toHaveProperty("keep_alive");
  });
});
