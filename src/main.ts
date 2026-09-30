import { AgentLoop, type DecisionRecord } from "./agent/loop.ts";
import { DecisionApiError } from "./agent/client.ts";
import { systemOnePolicy } from "./agent/policies.ts";
import { nativeBridge, nativePolicy } from "./agent/native.ts";
import { findDecisionPoint } from "./engine/decision.ts";
import { createGame, step, TPS } from "./engine/game.ts";
import { opposite, type Dir, type GameState, type StepInput } from "./engine/types.ts";
import { DEFAULT_ENCODER, ENCODERS } from "./encoders/index.ts";
import { HEIGHT, render, WIDTH, type Overlay } from "./render/canvas.ts";

const params = new URLSearchParams(location.search);
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ---- configuration -----------------------------------------------------------
// Inside the iOS app, decisions go to the on-device model instead of an HTTP endpoint.
const bridge = nativeBridge();
const ON_DEVICE_MODEL = "pacman-0.8b (on device)";
const defaultEndpoint = import.meta.env.DEV ? "/decide" : (import.meta.env.VITE_DECISION_BASE_URL ?? "http://localhost:11434");
const config = {
  mode: (params.get("mode") ?? "ai") as "ai" | "human",
  model: bridge ? ON_DEVICE_MODEL : (params.get("model") ?? import.meta.env.VITE_DEFAULT_MODEL ?? "tev1:4b"),
  encoder: params.get("encoder") ?? DEFAULT_ENCODER,
  speed: Number(params.get("speed") ?? "1"),
  endpoint: params.get("endpoint") ?? defaultEndpoint,
  seed: Number(params.get("seed") ?? Math.floor(Math.random() * 1e6)),
};

// ---- canvas ------------------------------------------------------------------
const canvas = $<HTMLCanvasElement>("game");
const dpr = Math.min(2, window.devicePixelRatio || 1);
canvas.width = WIDTH * dpr;
canvas.height = HEIGHT * dpr;
// Scale down to fit narrow screens while keeping the aspect ratio.
canvas.style.width = "100%";
canvas.style.maxWidth = `${WIDTH}px`;
canvas.style.height = "auto";
const ctx = canvas.getContext("2d")!;

// ---- game state ----------------------------------------------------------------
let game: GameState = createGame(config.seed, { speed: config.speed });
let pending: StepInput = {};
let paused = false;
const records: DecisionRecord[] = [];
let lastRecord: DecisionRecord | null = null;
const overlay: Overlay = {};

function queue(input: StepInput): void {
  if (input.intent !== undefined) pending.intent = input.intent;
  if (input.reverse) pending.reverse = true;
  if (input.turn) pending.turn = input.turn;
}

function restart(): void {
  game = createGame(Math.floor(Math.random() * 1e6), { speed: config.speed });
  pending = {};
  lastRecord = null;
}

// ---- agent -------------------------------------------------------------------
function makePolicy() {
  if (bridge) return nativePolicy(bridge, ON_DEVICE_MODEL);
  return systemOnePolicy({ baseUrl: config.endpoint, model: config.model });
}

const agent = new AgentLoop(
  {
    getState: () => game,
    apply: queue,
    onDecision: (rec) => {
      records.push(rec);
      if (records.length > 5000) records.shift();
      lastRecord = rec;
      hideError();
    },
    onError: (err) => showError(err),
  },
  makePolicy(),
  ENCODERS[config.encoder] ?? ENCODERS[DEFAULT_ENCODER],
);

function syncAgent(): void {
  agent.policy = makePolicy();
  agent.encoder = ENCODERS[config.encoder] ?? ENCODERS[DEFAULT_ENCODER];
  if (config.mode === "ai" && !paused) agent.start();
  else agent.stop();
}

// ---- fixed-step loop -------------------------------------------------------------
const STEP_MS = 1000 / TPS;
let acc = 0;
let last = performance.now();
const tickTimes: number[] = [];

// Frame timing, reported in the status heartbeat to diagnose stalls.
const frameStats = { frames: 0, ticks: 0, maxGapMs: 0, droppedMs: 0 };

function frame(now: number): void {
  const gap = now - last;
  frameStats.frames++;
  frameStats.maxGapMs = Math.max(frameStats.maxGapMs, gap);
  if (gap > 250) frameStats.droppedMs += gap - 250;
  const dt = Math.min(250, gap);
  last = now;
  if (!paused) {
    acc += dt;
    let steps = 0;
    while (acc >= STEP_MS && steps < 5) {
      step(game, pending);
      pending = {};
      acc -= STEP_MS;
      steps++;
      tickTimes.push(now);
      frameStats.ticks++;
    }
    if (steps === 5) {
      frameStats.droppedMs += acc;
      acc = 0;
    }
  }
  while (tickTimes.length && now - tickTimes[0] > 1000) tickTimes.shift();

  if (config.mode === "ai" && game.phase === "playing") {
    const dp = findDecisionPoint(game);
    overlay.junction = { x: dp.x, y: dp.y };
    overlay.arrow =
      lastRecord && !lastRecord.stale && lastRecord.junction === dp.key && lastRecord.choice !== "back"
        ? (lastRecord.choice as Dir)
        : null;
  } else {
    overlay.junction = undefined;
    overlay.arrow = null;
  }
  render(ctx, game, overlay);
  requestAnimationFrame(frame);
}

// ---- HUD ---------------------------------------------------------------------
function pct(xs: number[], p: number): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
}

function updateHud(): void {
  $("score").textContent = String(game.score);
  $("lives").textContent = String(game.lives);
  $("level").textContent = String(game.level);
  $("pellets").textContent = String(game.foodLeft);
  $("tps").textContent = String(tickTimes.length);
  $("who").textContent = config.mode === "ai" ? `${config.model} · ${config.encoder}` : "keyboard";

  const recent = records.slice(-100);
  const lat = recent.map((r) => r.latencyMs);
  const nowMs = Date.now();
  const lastFive = records.filter((r) => nowMs - r.time < 5000).length;
  $("lat-last").textContent = lastRecord ? `${Math.round(lastRecord.latencyMs)} ms` : "-";
  $("lat-p50").textContent = lat.length ? `${Math.round(pct(lat, 0.5))} ms` : "-";
  $("lat-p90").textContent = lat.length ? `${Math.round(pct(lat, 0.9))} ms` : "-";
  $("dps").textContent = config.mode === "ai" ? (lastFive / 5).toFixed(1) : "-";
  $("stale").textContent = recent.length ? `${Math.round((100 * recent.filter((r) => r.stale).length) / recent.length)}%` : "-";

  const bars = $("bars");
  bars.innerHTML = "";
  if (lastRecord?.probabilities) {
    $("confidence").textContent = `confidence ${lastRecord.confidence?.toFixed(2)}`;
    for (const key of lastRecord.keys) {
      const p = lastRecord.probabilities[key] ?? 0;
      const row = document.createElement("div");
      row.className = `bar${key === lastRecord.choice ? " chosen" : ""}`;
      row.innerHTML = `<span class="label">${key}</span><div class="track"><div class="fill" style="width:${(p * 100).toFixed(1)}%"></div></div><span>${(p * 100).toFixed(0)}%</span>`;
      bars.appendChild(row);
    }
    $("state-view").textContent = JSON.stringify(lastRecord.state, null, 1);
  } else {
    $("confidence").textContent = "";
    $("state-view").textContent = config.mode === "ai" ? "waiting for the first decision..." : "-";
  }
}

// ---- errors ------------------------------------------------------------------
function showError(err: unknown): void {
  const box = $("error");
  box.classList.add("show");
  if (err instanceof DecisionApiError) {
    $("error-message").textContent = err.message;
    $("error-hint").textContent = err.hint;
  } else {
    $("error-message").textContent = String(err);
    $("error-hint").textContent = "";
  }
}

function hideError(): void {
  $("error").classList.remove("show");
}

// ---- controls ----------------------------------------------------------------
const KEYS: Record<string, Dir> = {
  ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
  w: "up", s: "down", a: "left", d: "right",
};

window.addEventListener("keydown", (e) => {
  if ((e.target as HTMLElement).tagName === "INPUT") return;
  if (e.key === " ") {
    togglePause();
    e.preventDefault();
    return;
  }
  if (e.key === "r" || e.key === "R") {
    restart();
    return;
  }
  const d = KEYS[e.key];
  if (!d || config.mode !== "human") return;
  e.preventDefault();
  if (game.pac.dir && d === opposite(game.pac.dir)) queue({ reverse: true });
  else queue({ intent: d });
});

// Swipes steer Pac-Man in keyboard mode on touch screens.
let touchStart: { x: number; y: number } | null = null;
canvas.addEventListener("touchstart", (e) => {
  const t = e.changedTouches[0];
  touchStart = { x: t.clientX, y: t.clientY };
}, { passive: true });
canvas.addEventListener("touchend", (e) => {
  if (!touchStart || config.mode !== "human") return;
  const t = e.changedTouches[0];
  const dx = t.clientX - touchStart.x;
  const dy = t.clientY - touchStart.y;
  touchStart = null;
  if (Math.max(Math.abs(dx), Math.abs(dy)) < 20) return;
  const d: Dir = Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? "right" : "left") : dy > 0 ? "down" : "up";
  if (game.pac.dir && d === opposite(game.pac.dir)) queue({ reverse: true });
  else queue({ intent: d });
}, { passive: true });

function togglePause(): void {
  paused = !paused;
  $("pause").textContent = paused ? "Resume" : "Pause";
  syncAgent();
}

function setupControls(): void {
  const mode = $<HTMLSelectElement>("mode");
  const model = $<HTMLInputElement>("model");
  const encoder = $<HTMLSelectElement>("encoder");
  const speed = $<HTMLSelectElement>("speed");
  const endpoint = $<HTMLInputElement>("endpoint");

  for (const name of Object.keys(ENCODERS)) encoder.add(new Option(name, name));
  if (bridge) {
    // The on-device model is fixed; hide the HTTP endpoint and model fields.
    model.disabled = true;
    for (const el of [endpoint, endpoint.previousElementSibling]) (el as HTMLElement).style.display = "none";
  }
  mode.value = config.mode;
  model.value = config.model;
  encoder.value = config.encoder;
  speed.value = String(config.speed);
  endpoint.value = config.endpoint;

  mode.onchange = () => {
    config.mode = mode.value as "ai" | "human";
    syncAgent();
  };
  model.onchange = () => {
    config.model = model.value.trim();
    records.length = 0;
    lastRecord = null;
    syncAgent();
  };
  encoder.onchange = () => {
    config.encoder = encoder.value;
    syncAgent();
  };
  speed.onchange = () => {
    config.speed = Number(speed.value);
    game.options.speed = config.speed;
  };
  endpoint.onchange = () => {
    config.endpoint = endpoint.value.trim();
    loadModels();
    syncAgent();
  };
  $("pause").onclick = togglePause;
  $("restart").onclick = restart;
  $("export").onclick = () => {
    const blob = new Blob(records.map((r) => JSON.stringify(r) + "\n"), { type: "application/x-ndjson" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `decisions-${Date.now()}.jsonl`;
    a.click();
  };
}

async function loadModels(): Promise<void> {
  if (bridge) return;
  try {
    const res = await fetch(`${config.endpoint.replace(/\/$/, "")}/api/tags`);
    const body = (await res.json()) as { models?: { name: string }[] };
    const list = $("model-list");
    list.innerHTML = "";
    for (const m of body.models ?? []) list.appendChild(new Option(m.name, m.name));
  } catch {
    // The model list is a convenience; decisions report their own errors.
  }
}

// Inside the iOS app, report game status so the run can be checked from the Mac.
const statusBridge = (window as unknown as { webkit?: { messageHandlers?: { status?: { postMessage(m: unknown): void } } } })
  .webkit?.messageHandlers?.status;
if (statusBridge) {
  let seq = 0;
  const median = (xs: number[]) => (xs.length ? [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] : null);
  setInterval(() => {
    const recent = records.slice(-50);
    statusBridge.postMessage({
      seq: ++seq, tick: game.tick, phase: game.phase, score: game.score, lives: game.lives, level: game.level,
      pelletsLeft: game.foodLeft, decisions: records.length, stale: records.filter((r) => r.stale).length,
      latencyP50: median(recent.map((r) => r.latencyMs)),
      tokensP50: median(recent.flatMap((r) => (r.inputTokens ? [r.inputTokens] : []))),
      ticksPerSecond: tickTimes.length, mode: config.mode, model: config.model, paused,
      window: { seconds: 2, ...frameStats, maxGapMs: Math.round(frameStats.maxGapMs), droppedMs: Math.round(frameStats.droppedMs) },
    });
    Object.assign(frameStats, { frames: 0, ticks: 0, maxGapMs: 0, droppedMs: 0 });
  }, 2000);
}

setupControls();
void loadModels();
syncAgent();
setInterval(updateHud, 100);
requestAnimationFrame(frame);

// Exposed for scripted recording, debugging, and native control (the iOS app calls these).
(window as unknown as { __pacman: object }).__pacman = {
  get game() {
    return game;
  },
  records,
  setMode(mode: "ai" | "human") {
    config.mode = mode;
    $<HTMLSelectElement>("mode").value = mode;
    syncAgent();
  },
  setPaused(value: boolean) {
    if (paused !== value) togglePause();
  },
  setSpeed(value: number) {
    config.speed = value;
    game.options.speed = value;
    $<HTMLSelectElement>("speed").value = String(value);
  },
  restart,
};
