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
