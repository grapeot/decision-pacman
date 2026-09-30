import { describe, expect, it } from "vitest";
import { createGame, ghostTarget, step, TPS } from "../src/engine/game.ts";
import { findDecisionPoint } from "../src/engine/decision.ts";
import { COLS, Food, index, isOpen, MAZE_ROWS, ROWS } from "../src/engine/maze.ts";
import type { Dir, GameState } from "../src/engine/types.ts";

function playing(seed = 1): GameState {
  const s = createGame(seed);
  while (s.phase !== "playing") step(s);
  return s;
}

function run(s: GameState, ticks: number, intent?: Dir): GameState {
  for (let i = 0; i < ticks; i++) step(s, i === 0 && intent ? { intent } : {});
  return s;
}

function placePac(s: GameState, x: number, y: number, dir: Dir | null): void {
  s.pac.x = x;
  s.pac.y = y;
  s.pac.dir = dir;
  s.pac.lastDir = dir ?? "left";
  s.pac.intent = null;
}

function parkGhosts(s: GameState): void {
  for (const g of s.ghosts) {
    g.state = "house";
    g.x = 13.5;
    g.y = 14;
  }
  s.dotsEatenThisLife = -1000;
  s.ticksSinceDot = -100000;
}

describe("maze", () => {
  it("has 28x31 tiles and the classic 240 pellets plus 4 power pellets", () => {
    expect(MAZE_ROWS).toHaveLength(ROWS);
    for (const row of MAZE_ROWS) expect(row).toHaveLength(COLS);
    const s = createGame();
    let pellets = 0;
    let power = 0;
    for (const f of s.food) {
      if (f === Food.Pellet) pellets++;
      if (f === Food.Power) power++;
    }
    expect(pellets).toBe(240);
    expect(power).toBe(4);
    expect(s.foodLeft).toBe(244);
  });

  it("wraps through the tunnel row", () => {
    expect(isOpen(-1, 14)).toBe(true);
    expect(isOpen(28, 14)).toBe(true);
    expect(isOpen(-1, 13)).toBe(false);
  });
});

describe("determinism", () => {
  it("replays the same game from the same seed and inputs", () => {
    const a = createGame(7);
    const b = createGame(7);
    const inputs: (Dir | undefined)[] = [];
    for (let i = 0; i < 2000; i++) inputs.push(i % 97 === 0 ? (["up", "down", "left", "right"] as Dir[])[(i / 97) % 4] : undefined);
    for (const i of inputs) {
      step(a, i ? { intent: i } : {});
      step(b, i ? { intent: i } : {});
    }
    expect(JSON.stringify({ ...a, food: [...a.food] })).toBe(JSON.stringify({ ...b, food: [...b.food] }));
  });
});

describe("pac-man movement", () => {
  it("starts after the ready phase and moves left", () => {
    const s = createGame();
    run(s, 2 * TPS - 1);
    expect(s.pac.x).toBe(13.5);
    run(s, 10);
    expect(s.pac.x).toBeLessThan(13.5);
  });

  it("stops at a wall when there is no way forward and no intent", () => {
    const s = playing();
    parkGhosts(s);
    s.options.autoCorner = false;
    placePac(s, 3, 1, "left"); // top-left corridor, wall at x=0
    run(s, 60);
    expect(s.pac.x).toBe(1);
    expect(s.pac.dir).toBeNull();
  });

  it("follows a corner automatically when autoCorner is on", () => {
    const s = playing();
    parkGhosts(s);
    placePac(s, 3, 1, "left"); // corner at (1,1): only way on is down
    run(s, 30);
    expect(s.pac.x).toBe(1);
    expect(s.pac.y).toBeGreaterThan(1);
  });

  it("reverses immediately on a reverse input", () => {
    const s = playing();
    parkGhosts(s);
    placePac(s, 20.4, 5, "left");
    step(s, { reverse: true });
    expect(s.pac.dir).toBe("right");
    expect(s.pac.x).toBeGreaterThan(20.4);
  });

  it("never reverses on a buffered turn, even when it points backward", () => {
    const s = playing();
    parkGhosts(s);
    placePac(s, 20.4, 5, "left");
    step(s, { intent: "right" });
    expect(s.pac.dir).toBe("left");
    run(s, 8);
    expect(s.pac.x).toBeLessThan(20.4);
  });

  it("turns at the first tile center where the intent is legal", () => {
    const s = playing();
    parkGhosts(s);
    // Row 5 is a long corridor; column 6 opens upward and downward.
    placePac(s, 8.5, 5, "left");
    step(s, { intent: "up" });
    run(s, 30);
    expect(s.pac.x).toBe(6);
    expect(s.pac.y).toBeLessThan(5);
  });

  it("wraps from the left tunnel exit to the right side", () => {
    const s = playing();
    parkGhosts(s);
    placePac(s, 1, 14, "left");
    run(s, 12);
    expect(s.pac.x).toBeGreaterThan(20);
    expect(s.pac.y).toBe(14);
  });
});

describe("food and ghosts", () => {
  it("scores 10 per pellet and 50 per power pellet and frightens ghosts", () => {
    const s = playing();
    parkGhosts(s);
    placePac(s, 1, 5, "up"); // pellet at (1,4), power pellet at (1,3)
    s.food[index(1, 5)] = Food.None;
    const before = s.score;
    run(s, 9); // two tiles: (1,4) then (1,3)
    expect(s.score).toBe(before + 10 + 50);
    expect(s.frightTicks).toBeGreaterThan(0);
    expect(s.ghosts.every((g) => g.frightened)).toBe(true);
  });

  it("eats a frightened ghost for 200 points and sends its eyes home", () => {
    const s = playing();
    parkGhosts(s);
    placePac(s, 10, 5, "left");
    s.food[index(10, 5)] = Food.None;
    s.food[index(9, 5)] = Food.None;
    const g = s.ghosts[0];
    Object.assign(g, { x: 9, y: 5, dir: "right", state: "active", frightened: true });
    s.frightTicks = 100;
    const before = s.score;
    step(s);
    expect(g.state).toBe("eaten");
    expect(s.score).toBe(before + 200);
    for (let i = 0; i < 20 * TPS && g.state !== "active"; i++) step(s);
    expect(g.state).toBe("active");
  });

  it("loses a life on contact with a normal ghost and resets", () => {
    const s = playing();
    parkGhosts(s);
    placePac(s, 10, 5, "left");
    const g = s.ghosts[0];
    Object.assign(g, { x: 9, y: 5, dir: "right", state: "active", frightened: false });
    step(s);
    expect(s.phase).toBe("dying");
    expect(s.lives).toBe(2);
    run(s, 45 + 2 * TPS);
    expect(s.phase).toBe("playing");
    expect(s.pac.x).toBe(13.5);
  });

  it("ends the game after the last life", () => {
    const s = playing();
    s.lives = 1;
    parkGhosts(s);
    placePac(s, 10, 5, "left");
    Object.assign(s.ghosts[0], { x: 9, y: 5, dir: "right", state: "active", frightened: false });
    run(s, 3 * TPS);
    expect(s.phase).toBe("gameover");
  });

  it("computes the classic chase targets", () => {
    const s = playing();
    placePac(s, 10, 5, "left");
    s.mode = "chase";
    const [blinky, pinky, inky, clyde] = s.ghosts;
    Object.assign(blinky, { x: 12, y: 8 });
    expect(ghostTarget(s, blinky)).toEqual({ x: 10, y: 5 });
    expect(ghostTarget(s, pinky)).toEqual({ x: 6, y: 5 });
    // Two tiles ahead of Pac-Man is (8,5); double the vector from Blinky (12,8).
    expect(ghostTarget(s, inky)).toEqual({ x: 4, y: 2 });
    Object.assign(clyde, { x: 11, y: 5 });
    expect(ghostTarget(s, clyde)).toEqual({ x: 0, y: 31 });
  });

  it("keeps ghosts on open tiles for a long random game", () => {
    const s = createGame(3);
    const dirs: Dir[] = ["up", "left", "down", "right"];
    for (let i = 0; i < 6000; i++) {
      step(s, i % 45 === 0 ? { intent: dirs[(i / 45) % 4] } : {});
      for (const g of s.ghosts) {
        if (g.state !== "active") continue;
        expect(isOpen(Math.round(g.x), Math.round(g.y))).toBe(true);
      }
      if (s.phase === "gameover") break;
    }
  });
});

describe("planned turns", () => {
  it("takes a planned turn only at its tile", () => {
    const s = playing();
    parkGhosts(s);
    placePac(s, 8.5, 5, "left"); // row 5: column 6 opens up and down
    step(s, { turn: { x: 1, y: 5, dir: "down" } }); // a plan for a later tile
    step(s, { turn: { x: 6, y: 5, dir: "up" } });
    run(s, 30);
    expect(s.pac.x).toBe(6);
    expect(s.pac.y).toBeLessThan(5);
    expect(s.pac.turns.some((t) => t.x === 6)).toBe(false);
  });

  it("looks past a junction whose turn is already planned", () => {
    const s = playing();
    const near = findDecisionPoint(s);
    expect(near.key).toBe("12,23");
    step(s, { turn: { x: 12, y: 23, dir: "up" } });
    const far = findDecisionPoint(s, 3);
    expect(far.key).not.toBe("12,23");
    expect(far.path.some((t) => t.x === 12 && t.y === 23)).toBe(true);
    // Without a plan, lookahead stays on the near junction.
    s.pac.turns = [];
    expect(findDecisionPoint(s, 3).key).toBe("12,23");
  });
});

describe("decision points", () => {
  it("finds the first junction ahead from the start position", () => {
    const s = playing();
    const dp = findDecisionPoint(s);
    expect(dp.key).toBe("12,23");
    expect(dp.exits.sort()).toEqual(["left", "up"]);
    expect(dp.back).toBe("right");
    expect(dp.distance).toBeCloseTo(1.5, 5);
  });

  it("follows corners to reach the next real junction", () => {
    const s = playing();
    parkGhosts(s);
    placePac(s, 3, 1, "left"); // corner at (1,1), then down to the junction at (1,5)
    const dp = findDecisionPoint(s);
    expect(dp.key).toBe("1,5");
    expect(dp.exits.sort()).toEqual(["down", "right"]);
  });

  it("offers every open direction when Pac-Man is stopped", () => {
    const s = playing();
    placePac(s, 6, 5, null);
    const dp = findDecisionPoint(s);
    expect(dp.distance).toBe(0);
    expect(dp.back).toBeNull();
    expect(dp.exits.sort()).toEqual(["down", "left", "right", "up"]);
  });
});
