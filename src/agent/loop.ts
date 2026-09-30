import { findDecisionPoint, isStillAhead, pacTilesPerSecond } from "../engine/decision.ts";
import type { GameState, StepInput } from "../engine/types.ts";
import { optionToInput, type Encoder } from "../encoders/types.ts";
import type { Policy } from "./policies.ts";

export interface DecisionRecord {
  time: number;
  model: string;
  encoder: string;
  junction: string;
  keys: string[];
  choice: string;
  probabilities?: Record<string, number>;
  confidence?: number;
  latencyMs: number;
  inputTokens?: number;
  askedTick: number;
  answeredTick: number;
  /** True when the answer arrived after Pac-Man had left that junction. */
  stale: boolean;
  state: unknown;
}

export interface AgentHooks {
  getState(): GameState;
  /** Queue input for the next game tick. */
  apply(input: StepInput): void;
  onDecision(rec: DecisionRecord): void;
  onError(err: unknown): void;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * How far ahead (in tiles) to plan: the distance Pac-Man covers while an
 * answer is in flight, with some margin.
 */
export function lookaheadTiles(s: GameState, expectedLatencyMs: number): number {
  return pacTilesPerSecond(s) * (expectedLatencyMs / 1000) * 1.25 + 0.3;
}

/** Exponential moving average of decision latency, used to size the lookahead. */
export class LatencyEstimate {
  constructor(public value = 250) {}
  update(ms: number): void {
    if (ms > 5) this.value = 0.8 * this.value + 0.2 * ms;
  }
}

/**
 * Browser agent loop: one request in flight, the newest answer wins, and an
 * answer about a junction Pac-Man has already passed is dropped. The game
 * never waits for it.
 */
export class AgentLoop {
  private running = false;
  private controller: AbortController | null = null;
  private latency = new LatencyEstimate();

  constructor(
    private hooks: AgentHooks,
    public policy: Policy,
    public encoder: Encoder,
  ) {}

  get isRunning(): boolean {
    return this.running;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    void this.run();
  }

  stop(): void {
    this.running = false;
    this.controller?.abort();
  }

  private async run(): Promise<void> {
    let backoff = 500;
    while (this.running) {
      const game = this.hooks.getState();
      // Only ask while the game is moving: a frozen state repeats the same
      // request, which the server answers from cache and skews latency stats.
      if (game.phase !== "playing") {
        await sleep(50);
        continue;
      }
      const lookahead = lookaheadTiles(game, this.latency.value);
      const dp = findDecisionPoint(game, lookahead);
      const enc = this.encoder.encode(game, dp);
      if (enc.keys.length === 1) {
        const only = optionToInput(enc.keys[0], dp);
        if (only) this.hooks.apply(only);
        await sleep(33);
        continue;
      }
      const askedTick = game.tick;
      this.controller = new AbortController();
      try {
        const d = await this.policy.decide(enc, this.controller.signal, game);
        backoff = 500;
        this.latency.update(d.latencyMs);
        const now = this.hooks.getState();
        const stale = now !== game || now.phase !== "playing" || !isStillAhead(now, dp, lookahead);
        const input = optionToInput(d.choice, dp);
        if (!stale && input) this.hooks.apply(input);
        this.hooks.onDecision({
          time: Date.now(),
          model: this.policy.name,
          encoder: this.encoder.name,
          junction: dp.key,
          keys: enc.keys,
          choice: d.choice,
          probabilities: d.probabilities,
          confidence: d.confidence,
          latencyMs: d.latencyMs,
          inputTokens: d.inputTokens,
          askedTick,
          answeredTick: now.tick,
          stale,
          state: enc.state,
        });
        // Let the queued input reach the game before planning the next
        // decision, so the lookahead sees the turn that was just planned.
        const applied = now.tick;
        for (let i = 0; i < 10 && this.hooks.getState().tick <= applied; i++) await sleep(5);
        if (d.latencyMs < 5) await sleep(33);
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        this.hooks.onError(err);
        await sleep(backoff);
        backoff = Math.min(backoff * 2, 4000);
      }
    }
  }
}
