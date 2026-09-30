export type Dir = "up" | "down" | "left" | "right";

export const DIRS: readonly Dir[] = ["up", "left", "down", "right"]; // arcade tie-break order

export const VEC: Readonly<Record<Dir, { x: number; y: number }>> = {
  up: { x: 0, y: -1 },
  down: { x: 0, y: 1 },
  left: { x: -1, y: 0 },
  right: { x: 1, y: 0 },
};

export function opposite(d: Dir): Dir {
  return d === "up" ? "down" : d === "down" ? "up" : d === "left" ? "right" : "left";
}

export type GhostName = "blinky" | "pinky" | "inky" | "clyde";

/** house: waiting inside; leaving/entering: scripted moves through the door; eaten: eyes returning. */
export type GhostState = "house" | "leaving" | "active" | "eaten" | "entering";

export interface Ghost {
  name: GhostName;
  x: number;
  y: number;
  dir: Dir;
  state: GhostState;
  frightened: boolean;
}

export interface PacMan {
  x: number;
  y: number;
  /** Current heading; null while stopped against a wall. */
  dir: Dir | null;
  /** Last heading, kept while stopped. */
  lastDir: Dir;
  /** Desired direction, applied at the first tile center where it is legal (keyboard play). */
  intent: Dir | null;
  /** Turns planned for specific junction tiles (model play). Consumed when Pac-Man reaches the tile. */
  turns: PlannedTurn[];
}

export interface PlannedTurn {
  x: number;
  y: number;
  dir: Dir;
}

export type Phase = "ready" | "playing" | "dying" | "levelclear" | "gameover";

export type GameEvent =
  | { type: "pellet" }
  | { type: "power" }
  | { type: "ghost-eaten"; ghost: GhostName; points: number }
  | { type: "death" }
  | { type: "level-clear"; level: number }
  | { type: "game-over" };

export interface GameOptions {
  /** Multiplies every speed; 1 is arcade level-1 speed. */
  speed: number;
  /** Turn automatically at corners that have a single way forward. */
  autoCorner: boolean;
  lives: number;
}

export interface GameState {
  seed: number;
  rng: number;
  tick: number;
  phase: Phase;
  phaseTicks: number;
  level: number;
  score: number;
  lives: number;
  food: Uint8Array;
  foodLeft: number;
  dotsEatenThisLife: number;
  ticksSinceDot: number;
  pac: PacMan;
  ghosts: Ghost[];
  mode: "scatter" | "chase";
  modeIndex: number;
  modeTicks: number;
  frightTicks: number;
  ghostCombo: number;
  events: GameEvent[];
  options: GameOptions;
  stats: { pelletsEaten: number; ghostsEaten: number; deaths: number; ticksPlaying: number };
}

export interface StepInput {
  /** A new buffered turn: taken at the first tile center where it is legal. Never reverses Pac-Man. */
  intent?: Dir | null;
  /** Turn around immediately. */
  reverse?: boolean;
  /** Take `dir` when Pac-Man reaches the center of tile (x, y). Replaces any plan for that tile. */
  turn?: PlannedTurn;
}
