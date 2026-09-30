import { COLS, ROWS, index, isOpen, wrapX } from "./maze.ts";
import { canPacMove } from "./game.ts";
import { DIRS, opposite, VEC, type Dir, type GameState } from "./types.ts";

const EPS = 1e-6;

/** The next place where Pac-Man has a real choice. */
export interface DecisionPoint {
  /** Stable id of the junction tile; answers for an old key are stale. */
  key: string;
  x: number;
  y: number;
  /** Tiles Pac-Man still has to travel to reach the junction center. */
  distance: number;
  /** Heading on arrival (null when Pac-Man is stopped on the junction). */
  arrival: Dir | null;
  /** Ways out of the junction, excluding going straight back. */
  exits: Dir[];
  /** Direction that turns Pac-Man around right now, if he is moving. */
  back: Dir | null;
  /** Tile centers Pac-Man passes on the way, ending with the junction itself. */
  path: { x: number; y: number }[];
}

function exitsAt(x: number, y: number, arrival: Dir | null): Dir[] {
  return DIRS.filter((d) => (arrival === null || d !== opposite(arrival)) && canPacMove(x, y, d));
}

/**
 * The next junction where Pac-Man needs a decision. With `lookahead` > 0, if
 * Pac-Man is closer than that many tiles to the next junction and a turn is
 * already planned there, the answer is about the junction after it instead:
 * a decision for the near one would arrive too late to matter.
 */
export function findDecisionPoint(s: GameState, lookahead = 0): DecisionPoint {
  const p = s.pac;
  const cx = Math.round(p.x);
  const cy = Math.round(p.y);
  const atCenter = Math.abs(p.x - cx) < EPS && Math.abs(p.y - cy) < EPS;

  if (p.dir === null) {
    const x = wrapX(cx);
    return { key: `${x},${cy}`, x, y: cy, distance: 0, arrival: null, exits: exitsAt(x, cy, null), back: null, path: [{ x, y: cy }] };
  }

  const v = VEC[p.dir];
  let x: number;
  let y: number;
  let distance: number;
  if (atCenter) {
    x = cx + v.x;
    y = cy + v.y;
    distance = 1;
  } else {
    const coord = v.x !== 0 ? p.x : p.y;
    const forward = v.x + v.y > 0;
    const next = forward ? Math.ceil(coord) : Math.floor(coord);
    distance = Math.abs(next - coord);
    x = v.x !== 0 ? next : cx;
    y = v.y !== 0 ? next : cy;
  }

  const path: { x: number; y: number }[] = [];
  let dp = walkToJunction(s, x, y, p.dir, distance, path);
  if (lookahead > 0 && dp.distance < lookahead) {
    const planned = p.turns.find((t) => t.x === dp.x && t.y === dp.y && dp.exits.includes(t.dir));
    if (planned) {
      const w = VEC[planned.dir];
      dp = walkToJunction(s, dp.x + w.x, dp.y + w.y, planned.dir, dp.distance + 1, [...dp.path]);
    }
  }
  return dp;
}

function walkToJunction(s: GameState, x: number, y: number, heading: Dir, distance: number, path: { x: number; y: number }[]): DecisionPoint {
  const back = s.pac.dir ? opposite(s.pac.dir) : null;
  for (let guard = 0; guard < COLS * ROWS; guard++) {
    x = wrapX(x);
    path.push({ x, y });
    const exits = exitsAt(x, y, heading);
    if (exits.length !== 1 || (!s.options.autoCorner && exits[0] !== heading)) {
      return { key: `${x},${y}`, x, y, distance, arrival: heading, exits, back, path };
    }
    heading = exits[0];
    x += VEC[heading].x;
    y += VEC[heading].y;
    distance += 1;
  }
  throw new Error("no decision point found");
}

/** True while the answer for `dp` can still be acted on. */
export function isStillAhead(s: GameState, dp: DecisionPoint, lookahead: number): boolean {
  return findDecisionPoint(s).key === dp.key || findDecisionPoint(s, lookahead).key === dp.key;
}

/** Pac-Man's current speed in tiles per second. */
export { pacTilesPerSecond } from "./game.ts";

/**
 * Breadth-first distances over Pac-Man's walkable tiles, starting by stepping
 * from (x, y) in direction `first`. The start tile is treated as visited, so
 * every distance is measured through that exit. Unreachable tiles are -1.
 */
export function distancesThrough(x: number, y: number, first: Dir): Int16Array {
  const dist = new Int16Array(COLS * ROWS).fill(-1);
  const sx = wrapX(x);
  const nx = wrapX(sx + VEC[first].x);
  const ny = y + VEC[first].y;
  if (!isOpen(nx, ny)) return dist;
  dist[index(sx, y)] = 0;
  dist[index(nx, ny)] = 1;
  const queue: number[] = [nx, ny];
  for (let head = 0; head < queue.length; head += 2) {
    const qx = queue[head];
    const qy = queue[head + 1];
    const d = dist[index(qx, qy)];
    for (const dir of DIRS) {
      const tx = wrapX(qx + VEC[dir].x);
      const ty = qy + VEC[dir].y;
      if (!isOpen(tx, ty)) continue;
      const i = index(tx, ty);
      if (dist[i] !== -1) continue;
      dist[i] = d + 1;
      queue.push(tx, ty);
    }
  }
  return dist;
}

/** Plain BFS distances from a tile in every direction. */
export function distancesFrom(x: number, y: number): Int16Array {
  const dist = new Int16Array(COLS * ROWS).fill(-1);
  const sx = wrapX(x);
  if (!isOpen(sx, y)) return dist;
  dist[index(sx, y)] = 0;
  const queue: number[] = [sx, y];
  for (let head = 0; head < queue.length; head += 2) {
    const qx = queue[head];
    const qy = queue[head + 1];
    const d = dist[index(qx, qy)];
    for (const dir of DIRS) {
      const tx = wrapX(qx + VEC[dir].x);
      const ty = qy + VEC[dir].y;
      if (!isOpen(tx, ty)) continue;
      const i = index(tx, ty);
      if (dist[i] !== -1) continue;
      dist[i] = d + 1;
      queue.push(tx, ty);
    }
  }
  return dist;
}
