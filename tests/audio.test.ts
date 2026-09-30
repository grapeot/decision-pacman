import { describe, expect, it } from "vitest";
import { midiToHz, noteToMidi, parseSong, parseTrack } from "../src/audio/notation.ts";
import { songFor } from "../src/audio/player.ts";
import { SONGS } from "../src/audio/songs.ts";
import { pulseCoefficients } from "../src/audio/synth.ts";

describe("notation", () => {
  it("maps note names to MIDI numbers", () => {
    expect(noteToMidi("A4")).toBe(69);
    expect(noteToMidi("C4")).toBe(60);
    expect(noteToMidi("G#5")).toBe(80);
    expect(noteToMidi("Bb3")).toBe(58);
    expect(midiToHz(69)).toBeCloseTo(440);
    expect(midiToHz(81)).toBeCloseTo(880);
    expect(() => noteToMidi("H4")).toThrow();
  });

  it("places notes on the sixteenth grid and skips rests and bar lines", () => {
    const t = parseTrack({ voice: "pulse25", gain: 1, notes: "E5:3 -:1 | A5:4" });
    expect(t.steps).toBe(8);
    expect(t.events).toEqual([
      { step: 0, steps: 3, midi: 76 },
      { step: 4, steps: 4, midi: 81 },
    ]);
  });

  it("rejects bad drums and lengths", () => {
    expect(() => parseTrack({ voice: "drums", gain: 1, notes: "x:2" })).toThrow();
    expect(() => parseTrack({ voice: "pulse25", gain: 1, notes: "A4:0" })).toThrow();
  });
});

describe("songs", () => {
  it("all parse, and looping songs have equal-length tracks in whole bars", () => {
    for (const song of Object.values(SONGS)) {
      const { steps } = parseSong(song);
      if (song.loop) expect(steps % 16).toBe(0);
    }
    expect(parseSong(SONGS.main).steps).toBe(128);
  });

  it("one-shot songs fit inside their game phase (2 s)", () => {
    for (const name of ["ready", "clear"] as const) {
      const { steps, stepSeconds } = parseSong(SONGS[name]);
      expect(steps * stepSeconds).toBeLessThanOrEqual(2.001);
    }
  });
});

describe("songFor", () => {
  it("follows the game phase", () => {
    expect(songFor({ phase: "ready", frightTicks: 0 })).toBe("ready");
    expect(songFor({ phase: "playing", frightTicks: 0 })).toBe("main");
    expect(songFor({ phase: "playing", frightTicks: 10 })).toBe("fright");
    expect(songFor({ phase: "dying", frightTicks: 0 })).toBeNull();
    expect(songFor({ phase: "levelclear", frightTicks: 0 })).toBe("clear");
    expect(songFor({ phase: "gameover", frightTicks: 0 })).toBeNull();
  });
});

describe("pulseCoefficients", () => {
  it("a 50% pulse is a square wave: no even harmonics", () => {
    const { real } = pulseCoefficients(0.5, 8);
    expect(real[0]).toBe(0);
    expect(Math.abs(real[2])).toBeLessThan(1e-9);
    expect(Math.abs(real[4])).toBeLessThan(1e-9);
    expect(real[1]).toBeCloseTo(2 / Math.PI);
  });
});
