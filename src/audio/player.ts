// Game audio: picks the music for the current game phase and turns game
// events into sound effects. Songs are scheduled a little ahead of the audio
// clock, so timing does not depend on frame rate.
import type { GameEvent, GameState } from "../engine/types.ts";
import { parseSong, type ParsedTrack } from "./notation.ts";
import { playSfx, type SfxName } from "./sfx.ts";
import { SONGS, type SongName } from "./songs.ts";
import { Synth } from "./synth.ts";

const LOOKAHEAD_S = 0.15;
const PUMP_MS = 25;

interface Playing {
  name: SongName;
  tracks: ParsedTrack[];
  steps: number;
  stepSeconds: number;
  loop: boolean;
  bus: GainNode;
  /** Next step to schedule, counted from the start, across loops. */
  next: number;
  start: number;
}

/** Which song belongs to this moment of the game, or null for silence. */
export function songFor(game: Pick<GameState, "phase" | "frightTicks">): SongName | null {
  switch (game.phase) {
    case "ready":
      return "ready";
    case "playing":
      return game.frightTicks > 0 ? "fright" : "main";
    case "levelclear":
      return "clear";
    default:
      return null;
  }
}

const EVENT_SFX: Partial<Record<GameEvent["type"], SfxName>> = {
  pellet: "pellet",
  power: "power",
  "ghost-eaten": "ghost",
  death: "death",
  "game-over": "gameover",
};

/** Schedules every note of `name` from `t` for `loops` passes. Used for offline renders. */
export function scheduleSong(synth: Synth, dest: AudioNode, name: SongName, t: number, loops = 1): number {
  const { tracks, steps, stepSeconds } = parseSong(SONGS[name]);
  for (let pass = 0; pass < loops; pass++) {
    const base = t + pass * steps * stepSeconds;
    for (const tr of tracks) for (const ev of tr.events) synth.note(dest, tr.voice, tr.gain, ev, base + ev.step * stepSeconds, stepSeconds);
  }
  return t + loops * steps * stepSeconds;
}

export class GameAudio {
  private ctx: AudioContext | null = null;
  private synth: Synth | null = null;
  private master: GainNode | null = null;
  private music: GainNode | null = null;
  private sfx: GainNode | null = null;
  private playing: Playing | null = null;
  /** The song wanted for the last phase seen, so one-shot songs play once per phase. */
  private wanted: SongName | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;
  private lastPelletSfx = 0;
  muted: boolean;

  constructor(muted = false) {
    this.muted = muted;
  }

  get state(): string {
    return this.ctx ? this.ctx.state : "none";
  }

  /** Creates or resumes the audio context. Browsers allow this only after a user gesture. */
  unlock(): void {
    if (!this.ctx) {
      const Ctor = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!Ctor) return;
      const ctx = new Ctor();
      this.ctx = ctx;
      this.synth = new Synth(ctx);
      this.master = ctx.createGain();
      this.master.gain.value = this.muted ? 0 : 1;
      this.master.connect(ctx.destination);
      this.music = ctx.createGain();
      this.music.gain.value = 0.8;
      this.music.connect(this.master);
      this.sfx = ctx.createGain();
      this.sfx.connect(this.master);
      this.timer = setInterval(() => this.pump(), PUMP_MS);
      this.switchTo(this.wanted);
    }
    if (this.ctx.state === "suspended") void this.ctx.resume().catch(() => {});
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
    if (this.master && this.ctx) this.master.gain.setTargetAtTime(muted ? 0 : 1, this.ctx.currentTime, 0.02);
  }

  /** Stops the clock while the game is paused or hidden, and restarts it after. */
  setSuspended(suspended: boolean): void {
    if (!this.ctx) return;
    if (suspended && this.ctx.state === "running") void this.ctx.suspend();
    else if (!suspended && this.ctx.state === "suspended") void this.ctx.resume().catch(() => {});
  }

  /** Call once per simulation tick with the state after the tick. */
  onTick(game: GameState): void {
    const want = songFor(game);
    if (want !== this.wanted) {
      this.wanted = want;
      this.switchTo(want);
    }
    if (!this.ctx || !this.synth || !this.sfx || this.ctx.state !== "running") return;
    const now = this.ctx.currentTime;
    for (const ev of game.events) {
      const name = EVENT_SFX[ev.type];
      if (!name) continue;
      if (name === "pellet") {
        // Pellets can arrive a tick apart; keep the chatter from smearing.
        if (now - this.lastPelletSfx < 0.06) continue;
        this.lastPelletSfx = now;
      }
      playSfx(this.synth, this.sfx, name, now + 0.005);
    }
  }

  private switchTo(name: SongName | null): void {
    if (!this.ctx || !this.music) return;
    const now = this.ctx.currentTime;
    if (this.playing) {
      const old = this.playing.bus;
      old.gain.setTargetAtTime(0, now, 0.03);
      setTimeout(() => old.disconnect(), 400);
      this.playing = null;
    }
    if (!name) return;
    const song = SONGS[name];
    const parsed = parseSong(song);
    const bus = this.ctx.createGain();
    bus.connect(this.music);
    this.playing = { name, ...parsed, loop: song.loop, bus, next: 0, start: now + 0.03 };
    this.pump();
  }

  private pump(): void {
    const p = this.playing;
    if (!p || !this.ctx || !this.synth || this.ctx.state !== "running") return;
    const horizon = this.ctx.currentTime + LOOKAHEAD_S;
    // If the clock ran ahead (a stalled tab), skip the missed notes instead of bunching them.
    const behind = Math.floor((this.ctx.currentTime - p.start) / p.stepSeconds);
    if (behind > p.next + 4) p.next = behind;
    while (p.start + p.next * p.stepSeconds < horizon) {
      if (!p.loop && p.next >= p.steps) return;
      const at = p.start + p.next * p.stepSeconds;
      const inSong = p.next % p.steps;
      for (const tr of p.tracks) {
        for (const ev of tr.events) if (ev.step === inSong) this.synth.note(p.bus, tr.voice, tr.gain, ev, at, p.stepSeconds);
      }
      p.next++;
    }
  }

  dispose(): void {
    if (this.timer) clearInterval(this.timer);
    void this.ctx?.close();
  }
}
