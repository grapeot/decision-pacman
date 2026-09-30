// Compact text notation for chiptune tracks, parsed into timed events.
//
// A track is a space-separated list of tokens `<note>:<steps>`, where steps
// are sixteenth notes. `<note>` is a pitch such as `E5`, `C#6`, `Bb3`, a rest
// `-`, or a drum hit (`k` kick, `s` snare, `h` hi-hat). `|` separates bars
// for readability and is ignored.

export type Voice = "pulse12" | "pulse25" | "pulse50" | "triangle" | "drums";

export interface NoteEvent {
  step: number;
  steps: number;
  /** MIDI note number for pitched voices. */
  midi?: number;
  /** Drum hit for the drums voice. */
  drum?: "k" | "s" | "h";
}

export interface Track {
  voice: Voice;
  gain: number;
  notes: string;
}

export interface Song {
  name: string;
  bpm: number;
  loop: boolean;
  tracks: Track[];
}

export interface ParsedTrack {
  voice: Voice;
  gain: number;
  events: NoteEvent[];
  steps: number;
}

const PITCH: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

export function noteToMidi(name: string): number {
  const m = /^([A-G])([#b]?)(-?\d)$/.exec(name);
  if (!m) throw new Error(`bad note "${name}"`);
  const accidental = m[2] === "#" ? 1 : m[2] === "b" ? -1 : 0;
  return (Number(m[3]) + 1) * 12 + PITCH[m[1]] + accidental;
}

export function midiToHz(midi: number): number {
  return 440 * 2 ** ((midi - 69) / 12);
}

export function parseTrack(track: Track): ParsedTrack {
  const events: NoteEvent[] = [];
  let step = 0;
  for (const token of track.notes.split(/\s+/).filter((t) => t && t !== "|")) {
    const [name, len] = token.split(":");
    const steps = Number(len);
    if (!Number.isInteger(steps) || steps <= 0) throw new Error(`bad length in "${token}"`);
    if (name !== "-") {
      if (track.voice === "drums") {
        if (name !== "k" && name !== "s" && name !== "h") throw new Error(`bad drum "${token}"`);
        events.push({ step, steps, drum: name });
      } else {
        events.push({ step, steps, midi: noteToMidi(name) });
      }
    }
    step += steps;
  }
  return { voice: track.voice, gain: track.gain, events, steps: step };
}

export function parseSong(song: Song): { tracks: ParsedTrack[]; steps: number; stepSeconds: number } {
  const tracks = song.tracks.map(parseTrack);
  const steps = Math.max(...tracks.map((t) => t.steps));
  for (const t of tracks) {
    if (song.loop && t.steps !== steps) throw new Error(`${song.name}: track lengths differ (${t.steps} vs ${steps})`);
  }
  return { tracks, steps, stepSeconds: 60 / song.bpm / 4 };
}
