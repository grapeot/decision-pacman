// "Play it yourself": the real engine on the models' seed, a 30-second game-time limit, and an
// optional mode that hides the maze and shows only what the model receives.
import { GameAudio } from "../src/audio/player.ts";
import { findDecisionPoint } from "../src/engine/decision.ts";
import { createGame, step, TPS } from "../src/engine/game.ts";
import { opposite, type Dir, type GameState, type StepInput } from "../src/engine/types.ts";
import { optionToInput, type EncodedDecision, type OptionKey } from "../src/encoders/types.ts";
import { HEIGHT, render, WIDTH } from "../src/render/canvas.ts";
import { compactState, encodeAhead, holdForPlayer, isChoice, keyToOption } from "./eyes.ts";
import { rankPlayer, type Entry } from "./ranking.ts";

export interface PlayOptions {
  seed: number;
  seconds: number;
  opponents: readonly Entry[];
  t: (key: string, vars?: Record<string, string | number>) => string;
}

const KEYS: Record<string, Dir> = {
  ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
  w: "up", s: "down", a: "left", d: "right", W: "up", S: "down", A: "left", D: "right",
};
const ARROW: Record<OptionKey, string> = { up: "↑", down: "↓", left: "←", right: "→", back: "↺" };
const STEP_MS = 1000 / TPS;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

function escapeHtml(s: string): string {
  return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]!);
}

/** Light syntax colouring for the compact JSON; option names stand out. */
function highlight(json: string, keys: readonly string[]): string {
  return escapeHtml(json)
    .replace(/"([a-z_]+)":/g, (_m, k: string) => `<span class="${keys.includes(k) ? "opt" : "k"}">"${k}"</span>:`)
    .replace(/: (-?\d+(?:\.\d+)?)/g, ': <span class="n">$1</span>')
    .replace(/: null/g, ': <span class="z">null</span>')
    .replace(/: (true|false)/g, ': <span class="b">$1</span>');
}

export class PlaySection {
  private game: GameState;
  private pending: StepInput = {};
  private running = false;
  private finished = false;
  private eyes = false;
  /** Junction the player has answered for in eyes mode. */
  private answered: string | null = null;
  private chosen: OptionKey | null = null;
  private waiting = false;
  private acc = 0;
  private last = performance.now();
  private lastEyesDraw = 0;
  private lastEyesSig = "";
  private enc: EncodedDecision | null = null;
  private touchStart: { x: number; y: number } | null = null;
  private ctx: CanvasRenderingContext2D;
  private audio = new GameAudio(true);
  private readonly limit: number;

  constructor(private opts: PlayOptions) {
    this.limit = opts.seconds * TPS;
    this.game = createGame(opts.seed, { speed: 1 });
    const canvas = $<HTMLCanvasElement>("game");
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = WIDTH * dpr;
    canvas.height = HEIGHT * dpr;
    this.ctx = canvas.getContext("2d")!;

    $("start").addEventListener("click", () => this.start());
    $("again").addEventListener("click", () => this.start());
    const toggle = $<HTMLInputElement>("eyes-toggle");
    toggle.addEventListener("change", () => this.setEyes(toggle.checked));
    $("sound").addEventListener("click", () => {
      this.audio.unlock();
      this.audio.setMuted(!this.audio.muted);
      if (!this.running) this.audio.setSuspended(true);
      this.relabel();
    });
    window.addEventListener("keydown", (e) => this.onKey(e));
    const stage = $("stage");
    stage.addEventListener("touchstart", (e) => {
      const t = e.changedTouches[0];
      this.touchStart = { x: t.clientX, y: t.clientY };
    }, { passive: true });
    stage.addEventListener("touchend", (e) => this.onSwipe(e), { passive: true });
    $("eyes-instructions").textContent = encodeAhead(this.game).instructions;

    this.drawHud();
    this.relabel();
    requestAnimationFrame((now) => this.frame(now));
  }

  /** Re-renders the texts that depend on the language. */
  relabel(): void {
    $("sound").textContent = this.opts.t(this.audio.muted ? "play.sound_off" : "play.sound_on");
    if (this.finished) this.showResult();
    this.lastEyesSig = "";
  }

  private setEyes(on: boolean): void {
    this.eyes = on;
    $("eyes").hidden = !on || this.finished;
    $("game").style.visibility = on && !this.finished ? "hidden" : "visible";
    this.lastEyesSig = "";
  }

  start(): void {
    this.game = createGame(this.opts.seed, { speed: 1 });
    this.pending = {};
    this.answered = null;
    this.chosen = null;
    this.waiting = false;
    this.acc = 0;
    this.running = true;
    this.finished = false;
    this.audio.unlock();
    this.audio.setSuspended(false);
    $("start-overlay").hidden = true;
    $("result").hidden = true;
    $("stage").classList.add("running");
    $<HTMLInputElement>("eyes-toggle").disabled = true;
    this.setEyes(this.eyes);
    $("stage").scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  private finish(): void {
    this.running = false;
    this.finished = true;
    this.audio.setSuspended(true);
    $("stage").classList.remove("running");
    $<HTMLInputElement>("eyes-toggle").disabled = false;
    // Reveal the maze where the player ended up.
    this.setEyes(this.eyes);
    this.showResult();
  }

  private showResult(): void {
    const you = this.game.stats.pelletsEaten;
    const { order, rank } = rankPlayer(you, this.opts.opponents);
    $("result-title").textContent = this.opts.t("play.result_title", { pellets: you, rank, total: order.length });
    const max = Math.max(1, ...order.map((e) => e.pellets));
    const list = $("ranking");
    list.replaceChildren(
      ...order.map((e, i) => {
        const li = document.createElement("li");
        li.className = e.you ? "you" : e.id;
        const name = document.createElement("span");
        name.textContent = e.you ? this.opts.t("play.result_you") : this.opts.t(`halo.chart_labels.${e.id}`);
        const bar = document.createElement("div");
        bar.className = "bar";
        bar.style.width = `${Math.max(2, (100 * e.pellets) / max)}%`;
        bar.style.animationDelay = `${i * 80}ms`;
        const track = document.createElement("div");
        track.appendChild(bar);
        const val = document.createElement("span");
        val.className = "val";
        val.textContent = String(e.pellets);
        li.append(name, track, val);
        return li;
      }),
    );
    $("result").hidden = false;
  }

  // ---- input ----------------------------------------------------------------
  private steer(d: Dir): void {
    if (this.eyes) {
      const enc = this.enc ?? encodeAhead(this.game);
      const opt = keyToOption(d, this.game, enc.keys);
      if (opt) this.choose(opt);
      return;
    }
    if (this.game.pac.dir && d === opposite(this.game.pac.dir)) this.pending.reverse = true;
    else this.pending.intent = d;
  }

  private choose(opt: OptionKey): void {
    if (!this.running) return;
    const dp = findDecisionPoint(this.game);
    const input = optionToInput(opt, dp);
    if (!input) return;
    if (input.reverse) this.pending.reverse = true;
    if (input.turn) this.pending.turn = input.turn;
    this.answered = opt === "back" ? null : dp.key;
    this.chosen = opt === "back" ? null : opt;
    this.lastEyesSig = "";
  }

  private onKey(e: KeyboardEvent): void {
    if (!this.running) return;
    const target = e.target as HTMLElement;
    if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
    const d = KEYS[e.key];
    if (!d) return;
    e.preventDefault();
    this.steer(d);
  }

  private onSwipe(e: TouchEvent): void {
    if (!this.touchStart || !this.running || this.eyes) return;
    const t = e.changedTouches[0];
    const dx = t.clientX - this.touchStart.x;
    const dy = t.clientY - this.touchStart.y;
    this.touchStart = null;
    if (Math.max(Math.abs(dx), Math.abs(dy)) < 20) return;
    this.steer(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? "right" : "left") : dy > 0 ? "down" : "up");
  }

  // ---- simulation -----------------------------------------------------------
  /** One fixed tick, unless eyes mode is holding at a junction. Returns false when nothing ran. */
  private tick(): boolean {
    if (this.eyes && holdForPlayer(this.game, this.answered)) {
      this.waiting = true;
      return false;
    }
    this.waiting = false;
    const before = findDecisionPoint(this.game).key;
    step(this.game, this.pending);
    this.pending = {};
    this.audio.onTick(this.game);
    if (findDecisionPoint(this.game).key !== before) this.chosen = null;
    if (this.game.tick >= this.limit || this.game.phase === "gameover") this.finish();
    return true;
  }

  private frame(now: number): void {
    const dt = Math.min(250, now - this.last);
    this.last = now;
    if (this.running) {
      this.acc += dt;
      let steps = 0;
      while (this.running && this.acc >= STEP_MS && steps < 5) {
        if (!this.tick()) {
          this.acc = 0;
          break;
        }
        this.acc -= STEP_MS;
        steps++;
      }
      if (steps === 5) this.acc = 0;
    }
    if (!this.eyes || this.finished || !this.running) render(this.ctx, this.game);
    if (this.eyes && this.running) this.drawEyes(now);
    this.drawHud();
    requestAnimationFrame((n) => this.frame(n));
  }

  /** Test hook: runs `seconds` of game time at once (eyes mode still stops at an unanswered junction). */
  fastForward(seconds: number): void {
    for (let i = 0; i < seconds * TPS && this.running; i++) if (!this.tick()) break;
  }

  get state() {
    return { running: this.running, finished: this.finished, eyes: this.eyes, waiting: this.waiting, tick: this.game.tick, pellets: this.game.stats.pelletsEaten, keys: this.enc?.keys ?? [] };
  }

  // ---- drawing ----------------------------------------------------------------
  private drawHud(): void {
    const left = Math.max(0, (this.limit - this.game.tick) / TPS);
    $("hud-time").textContent = left.toFixed(1);
    $("hud-time-bar").style.transform = `scaleX(${left / this.opts.seconds})`;
    $("hud-time").parentElement!.classList.toggle("low", this.running && left < 5);
    $("hud-pellets").textContent = String(this.game.stats.pelletsEaten);
    $("hud-lives").textContent = String(this.game.lives);
  }

  private drawEyes(now: number): void {
    const dp = findDecisionPoint(this.game);
    const sig = `${dp.key}|${this.waiting}|${this.chosen}|${this.game.phase}`;
    // The numbers change every tick; redraw a few times a second, or at once when the junction changes.
    if (sig === this.lastEyesSig && now - this.lastEyesDraw < 200) return;
    this.lastEyesSig = sig;
    this.lastEyesDraw = now;
    const enc = encodeAhead(this.game);
    this.enc = enc;
    $("eyes-state").innerHTML = highlight(compactState(enc.state), enc.keys);
    const box = $("eyes");
    const waiting = this.waiting && isChoice(dp);
    box.classList.toggle("waiting", waiting);
    const status = $("eyes-status");
    status.classList.toggle("waiting", waiting);
    status.textContent = this.game.phase === "playing" ? this.opts.t(waiting ? "play.eyes_waiting" : "play.eyes_next") : "";
    const holder = $("eyes-options");
    const current = [...holder.children].map((b) => (b as HTMLElement).dataset.key).join(",");
    if (current !== enc.keys.join(",")) {
      holder.replaceChildren(
        ...enc.keys.map((k) => {
          const b = document.createElement("button");
          b.type = "button";
          b.dataset.key = k;
          b.textContent = `${ARROW[k]} ${k}`;
          b.addEventListener("click", () => this.choose(k));
          return b;
        }),
      );
    }
    for (const b of holder.children) (b as HTMLElement).classList.toggle("chosen", (b as HTMLElement).dataset.key === this.chosen);
  }
}
