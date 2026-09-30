import { COLS, Food, initialFood, index, isOpen, isTunnelSlowZone, wrapX } from "./maze.ts";
import { nextRandom } from "./rng.ts";
import {
  DIRS,
  opposite,
  VEC,
  type Dir,
  type GameOptions,
  type GameState,
  type Ghost,
  type GhostName,
  type StepInput,
} from "./types.ts";

export const TPS = 30;
const EPS = 1e-6;

// Arcade level-1 base speed: 75.76 px/s on 8 px tiles.
const BASE_TILES_PER_SEC = 75.75757625 / 8;

interface LevelSpec {
  pac: number;
  pacFright: number;
  ghost: number;
  ghostFright: number;
  ghostTunnel: number;
  frightSeconds: number;
}

function levelSpec(level: number): LevelSpec {
  if (level === 1) return { pac: 0.8, pacFright: 0.9, ghost: 0.75, ghostFright: 0.5, ghostTunnel: 0.4, frightSeconds: 6 };
  if (level <= 4) return { pac: 0.9, pacFright: 0.95, ghost: 0.85, ghostFright: 0.55, ghostTunnel: 0.45, frightSeconds: 5 - (level - 2) };
  return { pac: 1.0, pacFright: 1.0, ghost: 0.95, ghostFright: 0.6, ghostTunnel: 0.5, frightSeconds: Math.max(1, 7 - level) };
}

const EYES_SPEED = 1.6;
const HOUSE_SPEED = 0.4;

// Scatter/chase schedule in seconds; the last chase phase never ends.
const MODE_SCHEDULE = [7, 20, 7, 20, 5, 20, 5, Infinity];

const READY_TICKS = 2 * TPS;
const DYING_TICKS = Math.round(1.5 * TPS);
const LEVEL_CLEAR_TICKS = 2 * TPS;
const NO_DOT_RELEASE_TICKS = 4 * TPS;

export const PAC_START = { x: 13.5, y: 23 };
export const HOUSE_EXIT = { x: 13.5, y: 11 };
const HOUSE_CENTER_Y = 14;

const GHOST_SETUP: { name: GhostName; x: number; y: number; state: Ghost["state"]; dir: Dir }[] = [
  { name: "blinky", x: 13.5, y: 11, state: "active", dir: "left" },
  { name: "pinky", x: 13.5, y: 14, state: "house", dir: "down" },
  { name: "inky", x: 11.5, y: 14, state: "house", dir: "up" },
  { name: "clyde", x: 15.5, y: 14, state: "house", dir: "up" },
];

export const SCATTER_TARGET: Record<GhostName, { x: number; y: number }> = {
  blinky: { x: 25, y: -3 },
  pinky: { x: 2, y: -3 },
  inky: { x: 27, y: 31 },
  clyde: { x: 0, y: 31 },
};

const RELEASE_ORDER: GhostName[] = ["pinky", "inky", "clyde"];
const DOT_LIMITS_FIRST_LIFE: Record<string, number> = { pinky: 0, inky: 30, clyde: 60 };
const DOT_LIMITS_AFTER_DEATH: Record<string, number> = { pinky: 7, inky: 17, clyde: 32 };

export const DEFAULT_OPTIONS: GameOptions = { speed: 1, autoCorner: true, lives: 3 };

export function createGame(seed = 1, options: Partial<GameOptions> = {}): GameState {
  const food = initialFood();
  let foodLeft = 0;
  for (const f of food) if (f !== Food.None) foodLeft++;
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const state: GameState = {
    seed,
    rng: seed | 0,
    tick: 0,
    phase: "ready",
    phaseTicks: READY_TICKS,
    level: 1,
    score: 0,
    lives: opts.lives,
    food,
    foodLeft,
    dotsEatenThisLife: 0,
    ticksSinceDot: 0,
    pac: { x: PAC_START.x, y: PAC_START.y, dir: "left", lastDir: "left", intent: null, turns: [] },
    ghosts: [],
    mode: "scatter",
    modeIndex: 0,
    modeTicks: MODE_SCHEDULE[0] * TPS,
    frightTicks: 0,
    ghostCombo: 0,
    events: [],
    options: opts,
    stats: { pelletsEaten: 0, ghostsEaten: 0, deaths: 0, ticksPlaying: 0 },
  };
  resetActors(state);
  return state;
}

function resetActors(s: GameState): void {
  s.pac = { x: PAC_START.x, y: PAC_START.y, dir: "left", lastDir: "left", intent: null, turns: [] };
  s.ghosts = GHOST_SETUP.map((g) => ({ name: g.name, x: g.x, y: g.y, dir: g.dir, state: g.state, frightened: false }));
  s.mode = "scatter";
  s.modeIndex = 0;
  s.modeTicks = MODE_SCHEDULE[0] * TPS;
  s.frightTicks = 0;
  s.ghostCombo = 0;
  s.ticksSinceDot = 0;
}

export function pacTilesPerSecond(s: GameState): number {
  const spec = levelSpec(s.level);
  return BASE_TILES_PER_SEC * s.options.speed * (s.frightTicks > 0 ? spec.pacFright : spec.pac);
}

export function canPacMove(x: number, y: number, d: Dir): boolean {
  const v = VEC[d];
  return isOpen(x + v.x, y + v.y);
}

/** Advances one fixed tick (1/TPS s). Mutates and returns the state. */
export function step(s: GameState, input: StepInput = {}): GameState {
  s.events = [];
  s.tick++;
  applyInput(s, input);

  switch (s.phase) {
    case "ready":
      if (--s.phaseTicks <= 0) s.phase = "playing";
      return s;
    case "dying":
      if (--s.phaseTicks <= 0) {
        if (s.lives <= 0) {
          s.phase = "gameover";
          s.events.push({ type: "game-over" });
        } else {
          resetActors(s);
          s.phase = "ready";
          s.phaseTicks = READY_TICKS;
        }
      }
      return s;
    case "levelclear":
      if (--s.phaseTicks <= 0) {
        s.level++;
        s.food = initialFood();
        s.foodLeft = 0;
        for (const f of s.food) if (f !== Food.None) s.foodLeft++;
        s.dotsEatenThisLife = 0;
        resetActors(s);
        s.phase = "ready";
        s.phaseTicks = READY_TICKS;
      }
      return s;
    case "gameover":
      return s;
    case "playing":
      break;
  }

  s.stats.ticksPlaying++;
  const spec = levelSpec(s.level);
  const perTick = (BASE_TILES_PER_SEC * s.options.speed) / TPS;

  updateTimers(s);
  movePac(s, (s.frightTicks > 0 ? spec.pacFright : spec.pac) * perTick);
  eatFood(s, spec);
  if (checkCollisions(s)) return s;
  for (const g of s.ghosts) moveGhost(s, g, spec, perTick);
  if (checkCollisions(s)) return s;

  if (s.foodLeft === 0) {
    s.phase = "levelclear";
    s.phaseTicks = LEVEL_CLEAR_TICKS;
    s.events.push({ type: "level-clear", level: s.level });
  }
  return s;
}

const MAX_PLANNED_TURNS = 3;

function applyInput(s: GameState, input: StepInput): void {
  const p = s.pac;
  if (input.intent !== undefined) p.intent = input.intent;
  if (input.reverse && p.dir) {
    p.dir = opposite(p.dir);
    p.lastDir = p.dir;
    p.intent = null;
    p.turns = [];
  }
  if (input.turn) {
    const t = input.turn;
    p.turns = p.turns.filter((o) => o.x !== t.x || o.y !== t.y);
    p.turns.push({ x: t.x, y: t.y, dir: t.dir });
    if (p.turns.length > MAX_PLANNED_TURNS) p.turns.shift();
  }
}

function updateTimers(s: GameState): void {
  if (s.frightTicks > 0) {
    if (--s.frightTicks === 0) {
      for (const g of s.ghosts) g.frightened = false;
    }
  } else if (Number.isFinite(s.modeTicks) && --s.modeTicks <= 0) {
    s.modeIndex = Math.min(s.modeIndex + 1, MODE_SCHEDULE.length - 1);
    s.mode = s.modeIndex % 2 === 0 ? "scatter" : "chase";
    s.modeTicks = MODE_SCHEDULE[s.modeIndex] * TPS;
    reverseActiveGhosts(s);
  }

  s.ticksSinceDot++;
  const limits = s.stats.deaths > 0 ? DOT_LIMITS_AFTER_DEATH : DOT_LIMITS_FIRST_LIFE;
  const waiting = RELEASE_ORDER.map((n) => s.ghosts.find((g) => g.name === n)!).filter((g) => g.state === "house");
  if (waiting.length > 0) {
    const next = waiting[0];
    if (s.dotsEatenThisLife >= limits[next.name] || s.ticksSinceDot >= NO_DOT_RELEASE_TICKS) {
      next.state = "leaving";
      s.ticksSinceDot = 0;
    }
  }
}

function reverseActiveGhosts(s: GameState): void {
  for (const g of s.ghosts) if (g.state === "active") g.dir = opposite(g.dir);
}

type Chooser = (tx: number, ty: number, heading: Dir) => Dir | null;

interface Mover {
  x: number;
  y: number;
  dir: Dir | null;
}

/** Moves along the grid by `dist` tiles, asking `choose` for a heading at every tile center. Returns the last heading. */
function advance(m: Mover, dist: number, lastDir: Dir, choose: Chooser): Dir {
  let remaining = dist;
  let last: Dir = m.dir ?? lastDir;
  for (let guard = 0; remaining > EPS && guard < 16; guard++) {
    const cx = Math.round(m.x);
    const cy = Math.round(m.y);
    const atCenter = Math.abs(m.x - cx) < EPS && Math.abs(m.y - cy) < EPS;
    if (atCenter) {
      m.x = cx;
      m.y = cy;
      const nd = choose(cx, cy, last);
      if (!nd) {
        m.dir = null;
        return last;
      }
      m.dir = nd;
      last = nd;
    } else if (m.dir === null) {
      return last;
    }
    const v = VEC[m.dir];
    const coord = v.x !== 0 ? m.x : m.y;
    const forward = v.x + v.y > 0;
    const toCenter = atCenter ? 1 : forward ? Math.ceil(coord) - coord : coord - Math.floor(coord);
    const d = Math.min(remaining, toCenter);
    m.x += v.x * d;
    m.y += v.y * d;
    remaining -= d;
    if (m.x < -0.5) m.x += COLS;
    else if (m.x >= COLS - 0.5) m.x -= COLS;
  }
  return last;
}

function movePac(s: GameState, dist: number): void {
  const p = s.pac;
  const choose: Chooser = (tx, ty, heading) => {
    // A buffered turn never reverses a moving Pac-Man; only `reverse` does that.
    const stopped = p.dir === null;
    const wx = wrapX(tx);
    const planned = p.turns.find((t) => t.x === wx && t.y === ty);
    if (planned) {
      p.turns = p.turns.filter((t) => t !== planned);
      if ((stopped || planned.dir !== opposite(heading)) && canPacMove(tx, ty, planned.dir)) return planned.dir;
    }
    if (p.intent && p.intent !== heading && (stopped || p.intent !== opposite(heading)) && canPacMove(tx, ty, p.intent)) {
      return p.intent;
    }
    if (canPacMove(tx, ty, heading)) return heading;
    if (s.options.autoCorner) {
      const exits = DIRS.filter((d) => d !== opposite(heading) && canPacMove(tx, ty, d));
      if (exits.length === 1) return exits[0];
    }
    return null;
  };
  p.lastDir = advance(p, dist, p.lastDir, choose);
}

function eatFood(s: GameState, spec: LevelSpec): void {
  const i = index(Math.round(s.pac.x), Math.round(s.pac.y));
  const f = s.food[i];
  if (f === Food.None) return;
  s.food[i] = Food.None;
  s.foodLeft--;
  s.dotsEatenThisLife++;
  s.ticksSinceDot = 0;
  s.stats.pelletsEaten++;
  if (f === Food.Pellet) {
    s.score += 10;
    s.events.push({ type: "pellet" });
  } else {
    s.score += 50;
    s.events.push({ type: "power" });
    s.frightTicks = Math.round(spec.frightSeconds * TPS);
    s.ghostCombo = 0;
    for (const g of s.ghosts) {
      if (g.state !== "eaten" && g.state !== "entering") g.frightened = true;
    }
    reverseActiveGhosts(s);
  }
}

export function ghostTarget(s: GameState, g: Ghost): { x: number; y: number } {
  if (g.state === "eaten") return { x: 13, y: HOUSE_EXIT.y };
  if (s.mode === "scatter") return SCATTER_TARGET[g.name];
  const px = Math.round(s.pac.x);
  const py = Math.round(s.pac.y);
  const pv = VEC[s.pac.dir ?? s.pac.lastDir];
  switch (g.name) {
    case "blinky":
      return { x: px, y: py };
    case "pinky":
      return { x: px + 4 * pv.x, y: py + 4 * pv.y };
    case "inky": {
      const b = s.ghosts.find((o) => o.name === "blinky")!;
      const ax = px + 2 * pv.x;
      const ay = py + 2 * pv.y;
      return { x: 2 * ax - Math.round(b.x), y: 2 * ay - Math.round(b.y) };
    }
    case "clyde": {
      const dx = g.x - s.pac.x;
      const dy = g.y - s.pac.y;
      return dx * dx + dy * dy > 64 ? { x: px, y: py } : SCATTER_TARGET.clyde;
    }
  }
}

function moveGhost(s: GameState, g: Ghost, spec: LevelSpec, perTick: number): void {
  switch (g.state) {
    case "house":
      return;
    case "leaving":
      if (moveToward(g, HOUSE_EXIT.x, HOUSE_EXIT.y, HOUSE_SPEED * perTick)) {
        g.state = "active";
        g.dir = "left";
      }
      return;
    case "entering":
      if (moveToward(g, HOUSE_EXIT.x, HOUSE_CENTER_Y, EYES_SPEED * perTick)) {
        g.state = "leaving";
        g.frightened = false;
      }
      return;
    case "eaten":
    case "active": {
      const speed =
        g.state === "eaten"
          ? EYES_SPEED
          : isTunnelSlowZone(g.x, g.y)
            ? spec.ghostTunnel
            : g.frightened
              ? spec.ghostFright
              : spec.ghost;
      const choose: Chooser = (tx, ty, heading) => {
        if (g.state === "eaten" && ty === HOUSE_EXIT.y && (wrapX(tx) === 13 || wrapX(tx) === 14)) return null;
        let options = DIRS.filter((d) => d !== opposite(heading) && canPacMove(tx, ty, d));
        if (options.length === 0) options = DIRS.filter((d) => canPacMove(tx, ty, d));
        if (options.length === 0) return null;
        if (g.frightened && g.state === "active") return options[Math.floor(nextRandom(s) * options.length)];
        const t = ghostTarget(s, g);
        let best = options[0];
        let bestDist = Infinity;
        for (const d of options) {
          const v = VEC[d];
          const dx = tx + v.x - t.x;
          const dy = ty + v.y - t.y;
          const dist = dx * dx + dy * dy;
          if (dist < bestDist - EPS) {
            bestDist = dist;
            best = d;
          }
        }
        return best;
      };
      const mover: Mover = g;
      const last = advance(mover, speed * perTick, g.dir, choose);
      const stopped = mover.dir === null;
      g.dir = last;
      // Only eyes stop on purpose: they have reached the door.
      if (stopped && g.state === "eaten") g.state = "entering";
      return;
    }
  }
}

/** Scripted house movement: horizontal first, then vertical. Returns true on arrival. */
function moveToward(g: Ghost, tx: number, ty: number, dist: number): boolean {
  let remaining = dist;
  if (Math.abs(g.x - tx) > EPS) {
    const d = Math.min(remaining, Math.abs(g.x - tx));
    g.dir = tx > g.x ? "right" : "left";
    g.x += Math.sign(tx - g.x) * d;
    remaining -= d;
  }
  if (remaining > EPS && Math.abs(g.y - ty) > EPS) {
    const d = Math.min(remaining, Math.abs(g.y - ty));
    g.dir = ty > g.y ? "down" : "up";
    g.y += Math.sign(ty - g.y) * d;
  }
  return Math.abs(g.x - tx) < EPS && Math.abs(g.y - ty) < EPS;
}

function wrappedDistance(ax: number, ay: number, bx: number, by: number): number {
  let dx = Math.abs(ax - bx);
  if (dx > COLS / 2) dx = COLS - dx;
  const dy = ay - by;
  return Math.sqrt(dx * dx + dy * dy);
}

function checkCollisions(s: GameState): boolean {
  for (const g of s.ghosts) {
    if (g.state !== "active") continue;
    if (wrappedDistance(g.x, g.y, s.pac.x, s.pac.y) >= 0.6) continue;
    if (g.frightened) {
      const points = 200 * 2 ** s.ghostCombo;
      s.ghostCombo = Math.min(s.ghostCombo + 1, 3);
      s.score += points;
      g.state = "eaten";
      g.frightened = false;
      s.stats.ghostsEaten++;
      s.events.push({ type: "ghost-eaten", ghost: g.name, points });
    } else {
      s.lives--;
      s.stats.deaths++;
      s.dotsEatenThisLife = 0;
      s.phase = "dying";
      s.phaseTicks = DYING_TICKS;
      s.events.push({ type: "death" });
      return true;
    }
  }
  return false;
}
