// Offline renders of the music and effects, for previews and level checks.
import { scheduleSong } from "./player.ts";
import { playSfx } from "./sfx.ts";
import { Synth } from "./synth.ts";

export const SAMPLE_RATE = 44100;

/** A scripted tour: ready jingle, main loop with pellets, a power pellet, a death, level clear. */
export async function renderDemo(): Promise<AudioBuffer> {
  const ctx = new OfflineAudioContext(1, SAMPLE_RATE * 40, SAMPLE_RATE);
  const synth = new Synth(ctx);
  const music = ctx.createGain();
  music.gain.value = 0.8;
  music.connect(ctx.destination);
  const fx = ctx.destination;

  let t = scheduleSong(synth, music, "ready", 0.1);
  const mainStart = t;
  t = scheduleSong(synth, music, "main", t, 2);
  // Pellets in runs, as when Pac-Man clears a corridor.
  for (let run = 0; run < 6; run++) {
    const s = mainStart + 1 + run * 4;
    for (let i = 0; i < 12; i++) playSfx(synth, fx, "pellet", s + i * 0.13);
  }
  playSfx(synth, fx, "power", t);
  const frightStart = t;
  t = scheduleSong(synth, music, "fright", t, 2);
  playSfx(synth, fx, "ghost", frightStart + 1.5);
  playSfx(synth, fx, "ghost", frightStart + 3.6);
  t = playSfx(synth, fx, "death", t + 0.2);
  t = playSfx(synth, fx, "gameover", t + 0.4);
  scheduleSong(synth, music, "clear", t + 0.6);
  return ctx.startRendering();
}

/** 16-bit mono WAV bytes. */
export function toWav(buf: AudioBuffer): Uint8Array {
  const data = buf.getChannelData(0);
  const out = new DataView(new ArrayBuffer(44 + data.length * 2));
  const str = (o: number, s: string) => [...s].forEach((c, i) => out.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF");
  out.setUint32(4, 36 + data.length * 2, true);
  str(8, "WAVEfmt ");
  out.setUint32(16, 16, true);
  out.setUint16(20, 1, true);
  out.setUint16(22, 1, true);
  out.setUint32(24, buf.sampleRate, true);
  out.setUint32(28, buf.sampleRate * 2, true);
  out.setUint16(32, 2, true);
  out.setUint16(34, 16, true);
  str(36, "data");
  out.setUint32(40, data.length * 2, true);
  for (let i = 0; i < data.length; i++) out.setInt16(44 + i * 2, Math.max(-1, Math.min(1, data[i])) * 0x7fff, true);
  return new Uint8Array(out.buffer);
}
