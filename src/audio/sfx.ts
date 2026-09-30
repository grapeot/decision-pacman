// Sound effects, each a short sequence of sweeps and notes on the chip voices.
import { midiToHz, noteToMidi } from "./notation.ts";
import type { Synth } from "./synth.ts";

export type SfxName = "pellet" | "power" | "ghost" | "death" | "gameover";

let wakaUp = true;

/** Schedules `name` at time `t` into `dest`; returns when it ends. */
export function playSfx(synth: Synth, dest: AudioNode, name: SfxName, t: number): number {
  switch (name) {
    case "pellet": {
      // Alternate a rising and a falling blip, so a run of pellets chatters.
      wakaUp = !wakaUp;
      const [a, b] = wakaUp ? [330, 560] : [560, 330];
      return synth.sweep(dest, "triangle", 0.22, t, [[0, a], [0.07, b]]);
    }
    case "power": {
      // Three quick upward sweeps.
      let end = t;
      for (let i = 0; i < 3; i++) end = synth.sweep(dest, "pulse50", 0.07, t + i * 0.07, [[0, 300 + i * 150], [0.065, 900 + i * 300]]);
      return end;
    }
    case "ghost": {
      // A fast rising arpeggio, then a high blip.
      const notes = ["C5", "E5", "G5", "C6", "E6", "G6"];
      notes.forEach((n, i) => synth.sweep(dest, "pulse25", 0.08, t + i * 0.035, [[0, midiToHz(noteToMidi(n))], [0.035, midiToHz(noteToMidi(n))]]));
      return synth.sweep(dest, "pulse12", 0.07, t + 0.21, [[0, 1568], [0.12, 2093]]);
    }
    case "death": {
      // A descending wobble that fits in the 1.5 s dying animation.
      let at = t;
      for (let i = 0; i < 8; i++) {
        const hi = 880 * 2 ** (-i / 6);
        at = synth.sweep(dest, "pulse25", 0.08, at, [[0, hi], [0.06, hi * 1.12], [0.14, hi * 0.8]]);
      }
      synth.drum(dest, "k", 0.5, at);
      synth.drum(dest, "k", 0.5, at + 0.12);
      return at + 0.25;
    }
    case "gameover": {
      const notes: [string, number][] = [["E4", 0.18], ["D#4", 0.18], ["D4", 0.18], ["C#4", 0.6]];
      let at = t;
      for (const [n, len] of notes) {
        const hz = midiToHz(noteToMidi(n));
        synth.sweep(dest, "pulse25", 0.08, at, [[0, hz], [len * 0.9, hz]]);
        synth.sweep(dest, "triangle", 0.18, at, [[0, hz / 4], [len * 0.9, hz / 4]]);
        at += len;
      }
      return at;
    }
  }
}
