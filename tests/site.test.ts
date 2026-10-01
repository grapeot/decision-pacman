import { describe, expect, it } from "vitest";
import { fill, flattenKeys, lookup, pickLang } from "../site/i18n.ts";
import { rankPlayer } from "../site/ranking.ts";
import { compactState, encodeAhead, holdForPlayer } from "../site/eyes.ts";
import { findDecisionPoint } from "../src/engine/decision.ts";
import { createGame, step } from "../src/engine/game.ts";
import en from "../site/copy/en.json";
import zh from "../site/copy/zh.json";

describe("site language choice", () => {
  it("prefers ?lang, then the saved choice, then the browser", () => {
    expect(pickLang("?lang=zh", "en", ["en-US"])).toBe("zh");
    expect(pickLang("?lang=en", "zh", ["zh-CN"])).toBe("en");
    expect(pickLang("?lang=fr", "zh", ["en-US"])).toBe("zh");
    expect(pickLang("", null, ["zh-TW", "en"])).toBe("zh");
    expect(pickLang("", null, ["en-GB", "zh-CN"])).toBe("en");
    expect(pickLang("", "bogus", [])).toBe("en");
  });

  it("looks up dotted keys and fills placeholders", () => {
    const copy = { a: { b: "{n} pellets" } };
    expect(lookup(copy, "a.b")).toBe("{n} pellets");
    expect(lookup(copy, "a.c")).toBe("a.c");
    expect(lookup(copy, "a")).toBe("a");
    expect(fill("{n} pellets, {m}", { n: 86 })).toBe("86 pellets, {m}");
  });

  it("both languages carry the same keys", () => {
    expect(flattenKeys(zh)).toEqual(flattenKeys(en));
  });
});

describe("ranking", () => {
  const others = [
    { id: "jev", pellets: 86 },
    { id: "ours", pellets: 176 },
    { id: "phi4_mini", pellets: 143 },
  ];
  it("places the player among the models, best first", () => {
    const r = rankPlayer(100, others);
    expect(r.order.map((e) => e.id)).toEqual(["ours", "phi4_mini", "you", "jev"]);
    expect(r.rank).toBe(3);
  });
  it("gives ties to the player and handles both ends", () => {
    expect(rankPlayer(176, others).rank).toBe(1);
    expect(rankPlayer(500, others).rank).toBe(1);
    expect(rankPlayer(0, others).rank).toBe(4);
  });
});

describe("model's-eyes mode", () => {
  function playing() {
    const s = createGame(100);
    while (s.phase !== "playing") step(s);
    return s;
  }

  it("holds at the first real junction until it is answered", () => {
    const s = playing();
    let held = false;
    for (let i = 0; i < 600 && !held; i++) {
      held = holdForPlayer(s, null);
      if (!held) step(s);
    }
    expect(held).toBe(true);
    const dp = findDecisionPoint(s);
    expect(dp.exits.length).toBeGreaterThan(1);
    expect(holdForPlayer(s, dp.key)).toBe(false);
  });

  it("shows the features encoding with one line per option", () => {
    const s = playing();
    const enc = encodeAhead(s);
    const text = compactState(enc.state);
    expect(JSON.parse(text)).toEqual(enc.state);
    for (const k of enc.keys) expect(text).toContain(`"${k}": {`);
  });
});
