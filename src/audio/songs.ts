// Original chiptune music for the game. Nothing here is taken from the arcade
// soundtrack. Steps are sixteenth notes; `|` marks bar lines.
import type { Song } from "./notation.ts";

const rep = (s: string, n: number) => Array(n).fill(s).join(" ");

// Main loop: A minor, 8 bars, Am F C G | Am F G E. The last bar sits on E
// major so the G# leads back into A when the loop restarts.
export const MAIN: Song = {
  name: "main",
  bpm: 150,
  loop: true,
  tracks: [
    {
      voice: "pulse25",
      gain: 0.1,
      notes: [
        "E5:3 E5:1 A5:2 -:2 G5:2 E5:2 C5:2 E5:2",
        "F5:3 F5:1 A5:2 -:2 C6:2 A5:2 F5:2 A5:2",
        "G5:3 G5:1 E5:2 -:2 C5:2 E5:2 G5:2 C6:2",
        "D6:4 B5:2 G5:2 D5:2 G5:2 B5:2 -:2",
        "E6:3 D6:1 C6:2 B5:2 A5:4 E5:2 A5:2",
        "C6:3 B5:1 A5:2 G5:2 F5:4 A5:2 C6:2",
        "B5:3 A5:1 G5:2 A5:2 B5:2 D6:2 G6:2 D6:2",
        "E6:4 D6:2 B5:2 G#5:4 E5:2 G#5:2",
      ].join(" | "),
    },
    {
      voice: "pulse12",
      gain: 0.035,
      notes: [
        rep("A4:1 C5:1 E5:1 C5:1", 4),
        rep("F4:1 A4:1 C5:1 A4:1", 4),
        rep("C5:1 E5:1 G5:1 E5:1", 4),
        rep("G4:1 B4:1 D5:1 B4:1", 4),
        rep("A4:1 C5:1 E5:1 C5:1", 4),
        rep("F4:1 A4:1 C5:1 A4:1", 4),
        rep("G4:1 B4:1 D5:1 B4:1", 4),
        rep("E4:1 G#4:1 B4:1 G#4:1", 4),
      ].join(" | "),
    },
    {
      voice: "triangle",
      gain: 0.2,
      notes: [
        rep("A2:2 A3:2", 4),
        rep("F2:2 F3:2", 4),
        rep("C3:2 C4:2", 4),
        rep("G2:2 G3:2", 4),
        rep("A2:2 A3:2", 4),
        rep("F2:2 F3:2", 4),
        rep("G2:2 G3:2", 4),
        rep("E2:2 E3:2", 4),
      ].join(" | "),
    },
    { voice: "drums", gain: 0.5, notes: rep("k:2 h:2 s:2 h:2 k:2 k:2 s:2 h:2", 8) },
  ],
};

// While ghosts are frightened: faster, a creeping half-step wobble over an E pedal.
export const FRIGHT: Song = {
  name: "fright",
  bpm: 170,
  loop: true,
  tracks: [
    { voice: "pulse50", gain: 0.06, notes: `${rep("E5:1 F5:1", 8)} | ${rep("F#5:1 G5:1", 8)}` },
    { voice: "triangle", gain: 0.2, notes: rep("E2:2 -:2 E3:2 -:2", 4) },
    { voice: "drums", gain: 0.35, notes: rep("h:2 h:2 h:2 h:2", 4) },
  ],
};

// Before each life: a rising A minor figure resolving up, 24 steps (2 s at 180 bpm).
export const READY: Song = {
  name: "ready",
  bpm: 180,
  loop: false,
  tracks: [
    { voice: "pulse25", gain: 0.1, notes: "A4:2 C5:2 E5:2 A5:2 E5:2 A5:2 C6:4 | A5:8" },
    { voice: "pulse12", gain: 0.04, notes: "E4:2 A4:2 C5:2 E5:2 C5:2 E5:2 A5:4 | E5:8" },
    { voice: "triangle", gain: 0.2, notes: "A2:4 E3:4 A2:4 E3:4 | A2:8" },
    { voice: "drums", gain: 0.45, notes: "k:4 k:4 k:4 s:4 | k:2 s:2 s:2 s:2" },
  ],
};

// Level clear: a short major fanfare.
export const CLEAR: Song = {
  name: "clear",
  bpm: 180,
  loop: false,
  tracks: [
    { voice: "pulse25", gain: 0.1, notes: "C5:2 E5:2 G5:2 C6:4 G5:2 C6:8" },
    { voice: "pulse12", gain: 0.04, notes: "E4:2 G4:2 C5:2 E5:4 C5:2 E5:8" },
    { voice: "triangle", gain: 0.2, notes: "C3:6 G2:4 C3:10" },
  ],
};

export const SONGS = { main: MAIN, fright: FRIGHT, ready: READY, clear: CLEAR };
export type SongName = keyof typeof SONGS;
