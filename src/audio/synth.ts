// Chip-style voices on Web Audio: pulse waves at three duty cycles, a
// triangle bass, and noise drums. Works on a live AudioContext and on an
// OfflineAudioContext, so the same code renders previews and plays in game.
import { midiToHz, type NoteEvent, type Voice } from "./notation.ts";

/** Fourier series of a pulse wave with duty cycle `d`: b_n = 0, a_n = (2 / nπ) sin(nπd). */
export function pulseCoefficients(duty: number, harmonics = 64): { real: Float32Array; imag: Float32Array } {
  const real = new Float32Array(harmonics + 1);
  const imag = new Float32Array(harmonics + 1);
  for (let n = 1; n <= harmonics; n++) real[n] = (2 / (n * Math.PI)) * Math.sin(n * Math.PI * duty);
  return { real, imag };
}

export class Synth {
  readonly ctx: BaseAudioContext;
  private readonly pulses: Record<"pulse12" | "pulse25" | "pulse50", PeriodicWave>;
  private readonly noise: AudioBuffer;

  constructor(ctx: BaseAudioContext) {
    this.ctx = ctx;
    const wave = (duty: number) => {
      const { real, imag } = pulseCoefficients(duty);
      return ctx.createPeriodicWave(real, imag);
    };
    this.pulses = { pulse12: wave(0.125), pulse25: wave(0.25), pulse50: wave(0.5) };
    this.noise = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
    const data = this.noise.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
  }

  /** An oscillator for a pitched voice, not yet started. */
  osc(voice: Exclude<Voice, "drums">, hz: number): OscillatorNode {
    const o = this.ctx.createOscillator();
    if (voice === "triangle") o.type = "triangle";
    else o.setPeriodicWave(this.pulses[voice]);
    o.frequency.value = hz;
    return o;
  }

  /** A gain node shaped as attack, short decay to a sustain level, and release at `end`. */
  envelope(dest: AudioNode, t: number, end: number, peak: number, sustain = 0.7): GainNode {
    const g = this.ctx.createGain();
    const release = Math.min(0.03, (end - t) / 3);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(peak, t + 0.004);
    g.gain.linearRampToValueAtTime(peak * sustain, t + Math.min(0.06, (end - t) / 2));
    g.gain.setValueAtTime(peak * sustain, end - release);
    g.gain.linearRampToValueAtTime(0, end);
    g.connect(dest);
    return g;
  }

  /** Plays one parsed note at time `t`; `stepSeconds` is the length of a sixteenth. */
  note(dest: AudioNode, voice: Voice, gain: number, ev: NoteEvent, t: number, stepSeconds: number): void {
    if (voice === "drums") {
      this.drum(dest, ev.drum!, gain, t);
      return;
    }
    // Leave a small gap so repeated notes re-articulate.
    const end = t + ev.steps * stepSeconds * (voice === "triangle" ? 0.95 : 0.85);
    const o = this.osc(voice, midiToHz(ev.midi!));
    o.connect(this.envelope(dest, t, end, gain, voice === "triangle" ? 1 : 0.7));
    o.start(t);
    o.stop(end + 0.01);
  }

  drum(dest: AudioNode, kind: "k" | "s" | "h", gain: number, t: number): void {
    const ctx = this.ctx;
    if (kind === "k") {
      // A fast downward pitch sweep reads as a kick.
      const o = ctx.createOscillator();
      o.type = "triangle";
      o.frequency.setValueAtTime(160, t);
      o.frequency.exponentialRampToValueAtTime(45, t + 0.09);
      const g = ctx.createGain();
      g.gain.setValueAtTime(gain, t);
      g.gain.exponentialRampToValueAtTime(0.001, t + 0.14);
      o.connect(g).connect(dest);
      o.start(t);
      o.stop(t + 0.15);
      return;
    }
    const src = ctx.createBufferSource();
    src.buffer = this.noise;
    const filter = ctx.createBiquadFilter();
    filter.type = "highpass";
    filter.frequency.value = kind === "s" ? 1200 : 7000;
    const g = ctx.createGain();
    const length = kind === "s" ? 0.13 : 0.035;
    const level = kind === "s" ? gain * 0.55 : gain * 0.22;
    g.gain.setValueAtTime(level, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + length);
    src.connect(filter).connect(g).connect(dest);
    // Start at a random offset so repeated hits do not sound identical.
    src.start(t, Math.random() * 0.5, length + 0.01);
  }

  /** A single tone whose pitch moves through `points` ([seconds from t, hz]). */
  sweep(dest: AudioNode, voice: Exclude<Voice, "drums">, gain: number, t: number, points: [number, number][]): number {
    const o = this.osc(voice, points[0][1]);
    for (const [dt, hz] of points.slice(1)) o.frequency.linearRampToValueAtTime(hz, t + dt);
    const end = t + points[points.length - 1][0];
    o.connect(this.envelope(dest, t, end, gain, 0.9));
    o.start(t);
    o.stop(end + 0.01);
    return end;
  }
}
