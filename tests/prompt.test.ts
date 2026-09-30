import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { goJson, renderPrompt } from "../src/agent/prompt.ts";

interface Row {
  state: unknown;
  instructions: string;
  criteria: Record<string, string | null>;
  prompt: string;
}

const rows: Row[] = JSON.parse(readFileSync(new URL("./fixtures/prompt_parity.json", import.meta.url), "utf8"));

describe("prompt rendering", () => {
  it("matches the Hugging Face chat template output byte for byte", () => {
    expect(rows.length).toBeGreaterThan(10);
    for (const r of rows) expect(renderPrompt(r.state, r.instructions, r.criteria)).toBe(r.prompt);
  });

  it("escapes like Go's json.Marshal", () => {
    expect(goJson({ a: "<b>&" })).toBe('{"a":"\\u003cb\\u003e\\u0026"}');
  });
});
