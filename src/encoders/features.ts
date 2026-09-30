import { distancesThrough, type DecisionPoint } from "../engine/decision.ts";
import { TPS } from "../engine/game.ts";
import { COLS, Food, ROWS, index, wrapX } from "../engine/maze.ts";
import { VEC, type Dir, type GameState, type Ghost } from "../engine/types.ts";
import { optionKeys, type EncodedDecision, type Encoder, type OptionKey } from "./types.ts";

const GHOST_HORIZON = 15;
const PELLET_WINDOW = 6;
const FOOD_HORIZON = 40;
const POWER_HORIZON = 25;

const INSTRUCTIONS =
  "You control Pac-Man in a maze. Choose which way to go at the next junction. " +
  "Eating pellets scores points. Touching a normal ghost loses a life. " +
  "After a power pellet, ghosts are frightened for a few seconds and can be eaten for bonus points. " +
  "Each option lists facts about the path through that exit: ghost = steps to the nearest normal ghost " +
  "(null if none within 15), ghost_coming = that ghost is moving toward you, edible = steps to the nearest " +
  "frightened ghost, pellets = pellets within 6 steps, food = steps to the nearest pellet, power = steps to " +
  "the nearest power pellet. Steps count from Pac-Man, including the corridor up to the junction. " +
  "'back' means turn around right now instead of going to the junction.";

interface OptionFacts {
  ghost: number | null;
  ghost_coming?: boolean;
  edible?: number | null;
  pellets: number;
  food: number | null;
  power?: number | null;
}

function isDangerous(g: Ghost): boolean {
  return g.state === "active" && !g.frightened;
}

function isEdible(g: Ghost): boolean {
  return g.state === "active" && g.frightened;
}

function ghostTile(g: Ghost): { x: number; y: number } {
  return { x: wrapX(Math.round(g.x)), y: Math.round(g.y) };
}

interface PathTile {
  x: number;
  y: number;
  /** Steps from Pac-Man to this tile center. */
  steps: number;
}

/**
 * Facts for one option. `dist` holds BFS distances through the option's exit,
 * `offset` converts them to steps from Pac-Man, and `prefix` lists the tiles
 * Pac-Man passes before reaching that exit (the corridor to the junction).
 */
function factsFor(s: GameState, dist: Int16Array, offset: number, prefix: PathTile[]): OptionFacts {
  const at = (x: number, y: number) => (y < 0 || y >= ROWS ? -1 : dist[index(x, y)]);

  let ghost: number | null = null;
  let coming = false;
  let edible: number | null = null;
  const consider = (g: Ghost, total: number, isComing: boolean) => {
    if (isDangerous(g) && total <= GHOST_HORIZON && (ghost === null || total < ghost)) {
      ghost = total;
      coming = isComing;
    }
    if (isEdible(g) && total <= GHOST_HORIZON && (edible === null || total < edible)) edible = total;
  };

  for (const g of s.ghosts) {
    const t = ghostTile(g);
    const onPrefix = prefix.findIndex((p) => p.x === t.x && p.y === t.y);
    if (onPrefix >= 0) {
      // On the corridor ahead: coming if its next tile is closer to Pac-Man.
      const v = VEC[g.dir];
      const prev = onPrefix > 0 ? prefix[onPrefix - 1] : null;
      const isComing = prev ? wrapX(t.x + v.x) === prev.x && t.y + v.y === prev.y : true;
      consider(g, Math.max(0, prefix[onPrefix].steps), isComing);
      continue;
    }
    const d = at(t.x, t.y);
    if (d < 0) continue;
    const v = VEC[g.dir];
    const ahead = at(wrapX(t.x + v.x), t.y + v.y);
    consider(g, Math.max(0, d + offset), ahead >= 0 && ahead < d);
  }

  let pellets = 0;
  let food: number | null = null;
  let power: number | null = null;
  const addFood = (f: number, total: number) => {
    if (total <= PELLET_WINDOW) pellets++;
    if (total <= FOOD_HORIZON && (food === null || total < food)) food = total;
    if (f === Food.Power && total <= POWER_HORIZON && (power === null || total < power)) power = total;
  };
  for (const p of prefix) {
    const f = s.food[index(p.x, p.y)];
    if (f !== Food.None) addFood(f, p.steps);
  }
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      const i = index(x, y);
      const f = s.food[i];
      const d = dist[i];
      if (f === Food.None || d <= 0) continue;
      addFood(f, d + offset);
    }
  }

  const round = (v: number | null) => (v === null ? null : Math.round(v * 10) / 10);
  const facts: OptionFacts = { ghost: round(ghost), pellets, food: round(food) };
  if (ghost !== null) facts.ghost_coming = coming;
  if (s.frightTicks > 0) facts.edible = round(edible);
  if (power !== null) facts.power = round(power);
  return facts;
}

export const featuresEncoder: Encoder = {
  name: "features",
  encode(s: GameState, dp: DecisionPoint): EncodedDecision {
    const keys = optionKeys(dp);
    const firstSteps = dp.distance - (dp.path.length - 1);
    const corridor: PathTile[] = dp.path.map((t, i) => ({ x: t.x, y: t.y, steps: firstSteps + i }));
    const options: Partial<Record<OptionKey, OptionFacts>> = {};
    for (const key of keys) {
      if (key === "back") {
        // Turn around now: distances through the tile behind Pac-Man.
        const first = dp.path[0];
        options.back = factsFor(s, distancesThrough(first.x, first.y, dp.back as Dir), -firstSteps, []);
      } else {
        options[key] = factsFor(s, distancesThrough(dp.x, dp.y, key), dp.distance, corridor);
      }
    }

    const state: Record<string, unknown> = {
      lives: s.lives,
      pellets_left: s.foodLeft,
      to_junction: Math.round(dp.distance * 10) / 10,
    };
    if (s.frightTicks > 0) state.frightened_seconds_left = Math.round((s.frightTicks / TPS) * 10) / 10;
    state.options = options;

    const criteria: Record<string, string | null> = {};
    for (const key of keys) criteria[key] = key === "back" ? "turn around now" : null;
    return { encoder: "features", state, instructions: INSTRUCTIONS, criteria, keys, decision: dp };
  },
};
