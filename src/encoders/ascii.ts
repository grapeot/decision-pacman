import type { DecisionPoint } from "../engine/decision.ts";
import { Food, MAZE_ROWS, ROWS, index, wrapX } from "../engine/maze.ts";
import type { GameState } from "../engine/types.ts";
import { optionKeys, type EncodedDecision, type Encoder } from "./types.ts";

const RADIUS = 5;

const INSTRUCTIONS =
  "You control Pac-Man (P) in a maze. Choose which way to go at the next junction (J). " +
  "Legend: # wall, . pellet, o power pellet, G normal ghost, F frightened ghost, space empty. " +
  "Up is toward the top of the view. Touching G loses a life. F can be eaten for points. " +
  "'back' means turn around right now.";

function charAt(s: GameState, x: number, y: number, dp: DecisionPoint): string {
  if (y < 0 || y >= ROWS) return "#";
  const wx = wrapX(x);
  for (const g of s.ghosts) {
    if (g.state !== "active") continue;
    if (wrapX(Math.round(g.x)) === wx && Math.round(g.y) === y) return g.frightened ? "F" : "G";
  }
  if (wrapX(Math.round(s.pac.x)) === wx && Math.round(s.pac.y) === y) return "P";
  if (wx === dp.x && y === dp.y) return "J";
  const wall = MAZE_ROWS[y][wx];
  if (wall === "#" || wall === "-") return "#";
  const f = s.food[index(wx, y)];
  return f === Food.Pellet ? "." : f === Food.Power ? "o" : " ";
}

export const asciiWindowEncoder: Encoder = {
  name: "ascii-window",
  encode(s: GameState, dp: DecisionPoint): EncodedDecision {
    const px = Math.round(s.pac.x);
    const py = Math.round(s.pac.y);
    const rows: string[] = [];
    for (let y = py - RADIUS; y <= py + RADIUS; y++) {
      let row = "";
      for (let x = px - RADIUS; x <= px + RADIUS; x++) row += charAt(s, x, y, dp);
      rows.push(row);
    }
    const keys = optionKeys(dp);
    const state = `lives ${s.lives}, pellets left ${s.foodLeft}${s.frightTicks > 0 ? ", ghosts frightened" : ""}\n${rows.join("\n")}`;
    const criteria: Record<string, string | null> = {};
    for (const key of keys) criteria[key] = key === "back" ? "turn around now" : null;
    return { encoder: "ascii-window", state, instructions: INSTRUCTIONS, criteria, keys, decision: dp };
  },
};

const FULL_INSTRUCTIONS =
  "You control Pac-Man (P) in the maze drawn below. Choose which way to go at the next junction (J). " +
  "Legend: # wall, - ghost-house door, . pellet, o power pellet, G normal ghost, F frightened ghost, " +
  "space empty floor. Rows are numbered from the top; 'up' decreases the row, 'left' decreases the column. " +
  "The row through the middle wraps around at both edges (a tunnel). Eating pellets scores points. " +
  "Touching a normal ghost loses a life. Frightened ghosts can be eaten for bonus points. " +
  "'back' means turn around right now instead of going to the junction.";

const GHOST_LABEL: Record<string, string> = { blinky: "red", pinky: "pink", inky: "cyan", clyde: "orange" };

/** The whole board as a grid with row and column numbers, plus positions in words. */
export const asciiFullEncoder: Encoder = {
  name: "ascii-full",
  encode(s: GameState, dp: DecisionPoint): EncodedDecision {
    const lines: string[] = [];
    const header = Array.from({ length: 28 }, (_, x) => String(x % 10)).join("");
    lines.push(`   ${header}`);
    for (let y = 0; y < ROWS; y++) {
      let row = "";
      for (let x = 0; x < 28; x++) {
        const ch = charAt(s, x, y, dp);
        row += ch === "#" && MAZE_ROWS[y][x] === "-" ? "-" : ch;
      }
      lines.push(`${String(y).padStart(2, " ")} ${row}`);
    }
    const pac = s.pac;
    const heading = pac.dir ?? "stopped";
    const ghosts = s.ghosts
      .filter((g) => g.state === "active")
      .map((g) => `${GHOST_LABEL[g.name]} ${g.frightened ? "F" : "G"} at column ${wrapX(Math.round(g.x))}, row ${Math.round(g.y)}, moving ${g.dir}`);
    const facts = [
      `Pac-Man at column ${wrapX(Math.round(pac.x))}, row ${Math.round(pac.y)}, heading ${heading}.`,
      `Next junction J at column ${dp.x}, row ${dp.y}, ${Math.round(dp.distance * 10) / 10} steps ahead.`,
      ghosts.length ? `Ghosts outside the house: ${ghosts.join("; ")}.` : "No ghosts outside the house.",
      s.frightTicks > 0 ? `Ghosts are frightened for ${Math.round((s.frightTicks / 30) * 10) / 10} more seconds.` : "",
      `Lives ${s.lives}, pellets left ${s.foodLeft}.`,
    ].filter(Boolean);
    const keys = optionKeys(dp);
    const criteria: Record<string, string | null> = {};
    for (const key of keys) criteria[key] = key === "back" ? "turn around now" : null;
    return {
      encoder: "ascii-full",
      state: `${facts.join("\n")}\nMap:\n${lines.join("\n")}`,
      instructions: FULL_INSTRUCTIONS,
      criteria,
      keys,
      decision: dp,
    };
  },
};
