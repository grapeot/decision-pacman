import { TPS } from "../engine/game.ts";
import { COLS, Food, MAZE_ROWS, ROWS, index } from "../engine/maze.ts";
import type { Dir, GameState, Ghost } from "../engine/types.ts";

export const TILE = 20;
export const WIDTH = COLS * TILE;
export const HEIGHT = ROWS * TILE;

const GHOST_COLORS: Record<Ghost["name"], string> = {
  blinky: "#ff3b3b",
  pinky: "#ffb8ff",
  inky: "#00e5ff",
  clyde: "#ffb852",
};

const ANGLE: Record<Dir, number> = { right: 0, down: Math.PI / 2, left: Math.PI, up: -Math.PI / 2 };

function isWall(x: number, y: number): boolean {
  if (y < 0 || y >= ROWS || x < 0 || x >= COLS) return false;
  const ch = MAZE_ROWS[y][x];
  return ch === "#";
}

let mazeLayer: HTMLCanvasElement | null = null;

function drawMazeLayer(scale: number): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = WIDTH * scale;
  c.height = HEIGHT * scale;
  const ctx = c.getContext("2d")!;
  ctx.scale(scale, scale);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, WIDTH, HEIGHT);
  ctx.strokeStyle = "#2b44ff";
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  const inset = TILE * 0.3;
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      if (!isWall(x, y)) continue;
      const x0 = x * TILE;
      const y0 = y * TILE;
      ctx.beginPath();
      // Draw an outline on each side that faces open floor.
      if (!isWall(x, y - 1) && y > 0) {
        ctx.moveTo(x0 + (isWall(x - 1, y) ? 0 : inset), y0 + inset);
        ctx.lineTo(x0 + TILE - (isWall(x + 1, y) ? 0 : inset), y0 + inset);
      }
      if (!isWall(x, y + 1) && y < ROWS - 1) {
        ctx.moveTo(x0 + (isWall(x - 1, y) ? 0 : inset), y0 + TILE - inset);
        ctx.lineTo(x0 + TILE - (isWall(x + 1, y) ? 0 : inset), y0 + TILE - inset);
      }
      if (!isWall(x - 1, y) && x > 0) {
        ctx.moveTo(x0 + inset, y0 + (isWall(x, y - 1) ? 0 : inset));
        ctx.lineTo(x0 + inset, y0 + TILE - (isWall(x, y + 1) ? 0 : inset));
      }
      if (!isWall(x + 1, y) && x < COLS - 1) {
        ctx.moveTo(x0 + TILE - inset, y0 + (isWall(x, y - 1) ? 0 : inset));
        ctx.lineTo(x0 + TILE - inset, y0 + TILE - (isWall(x, y + 1) ? 0 : inset));
      }
      ctx.stroke();
    }
  }
  // Ghost-house door.
  ctx.fillStyle = "#ffb8ff";
  ctx.fillRect(13 * TILE, 12 * TILE + TILE * 0.4, 2 * TILE, TILE * 0.2);
  return c;
}

function drawPac(ctx: CanvasRenderingContext2D, s: GameState): void {
  const p = s.pac;
  const cx = (p.x + 0.5) * TILE;
  const cy = (p.y + 0.5) * TILE;
  const r = TILE * 0.8;
  let mouth = p.dir ? 0.05 + 0.22 * Math.abs(Math.sin(s.tick * 0.45)) : 0.18;
  if (s.phase === "dying") {
    const t = 1 - s.phaseTicks / (1.5 * TPS);
    mouth = Math.min(1, 0.2 + t);
  }
  const a = ANGLE[p.dir ?? p.lastDir];
  ctx.fillStyle = "#ffe600";
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.arc(cx, cy, r, a + mouth * Math.PI, a - mouth * Math.PI + 2 * Math.PI);
  ctx.closePath();
  ctx.fill();
}

function drawGhost(ctx: CanvasRenderingContext2D, s: GameState, g: Ghost): void {
  const cx = (g.x + 0.5) * TILE;
  const cy = (g.y + 0.5) * TILE;
  const r = TILE * 0.8;
  if (g.state !== "eaten" && g.state !== "entering") {
    let body = GHOST_COLORS[g.name];
    if (g.frightened) {
      const flashing = s.frightTicks < 2 * TPS && Math.floor(s.tick / 6) % 2 === 0;
      body = flashing ? "#f4f4f4" : "#2233ff";
    }
    ctx.fillStyle = body;
    ctx.beginPath();
    ctx.arc(cx, cy - r * 0.15, r, Math.PI, 0);
    const bottom = cy + r * 0.85;
    ctx.lineTo(cx + r, bottom);
    const waves = 4;
    const phase = Math.floor(s.tick / 5) % 2;
    for (let i = 0; i < waves; i++) {
      const x1 = cx + r - ((i + 0.5) * 2 * r) / waves;
      const x2 = cx + r - ((i + 1) * 2 * r) / waves;
      ctx.lineTo(x1, bottom - (i % 2 === phase ? r * 0.25 : 0));
      ctx.lineTo(x2, bottom);
    }
    ctx.closePath();
    ctx.fill();
    if (g.frightened) {
      ctx.fillStyle = "#ffd0a0";
      ctx.fillRect(cx - r * 0.4, cy - r * 0.3, r * 0.2, r * 0.2);
      ctx.fillRect(cx + r * 0.2, cy - r * 0.3, r * 0.2, r * 0.2);
      return;
    }
  }
  const v = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] }[g.dir];
  for (const side of [-1, 1]) {
    const ex = cx + side * r * 0.38;
    const ey = cy - r * 0.2;
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.ellipse(ex, ey, r * 0.26, r * 0.32, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#1d3bff";
    ctx.beginPath();
    ctx.arc(ex + v[0] * r * 0.12, ey + v[1] * r * 0.14, r * 0.13, 0, Math.PI * 2);
    ctx.fill();
  }
}

export interface Overlay {
  /** Junction the model is being asked about. */
  junction?: { x: number; y: number };
  /** Option the latest answer chose, drawn as an arrow at the junction. */
  arrow?: Dir | null;
}

export function render(ctx: CanvasRenderingContext2D, s: GameState, overlay: Overlay = {}): void {
  const scale = ctx.canvas.width / WIDTH;
  if (!mazeLayer || mazeLayer.width !== ctx.canvas.width) mazeLayer = drawMazeLayer(scale);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(mazeLayer, 0, 0);
  ctx.setTransform(scale, 0, 0, scale, 0, 0);

  const blink = Math.floor(s.tick / 8) % 2 === 0;
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      const f = s.food[index(x, y)];
      if (f === Food.None) continue;
      const cx = (x + 0.5) * TILE;
      const cy = (y + 0.5) * TILE;
      ctx.fillStyle = "#ffcfa8";
      ctx.beginPath();
      if (f === Food.Pellet) ctx.arc(cx, cy, TILE * 0.12, 0, Math.PI * 2);
      else if (blink || s.phase !== "playing") ctx.arc(cx, cy, TILE * 0.38, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (overlay.junction && s.phase === "playing") {
    const jx = (overlay.junction.x + 0.5) * TILE;
    const jy = (overlay.junction.y + 0.5) * TILE;
    ctx.strokeStyle = "rgba(120, 255, 170, 0.8)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(jx - TILE * 0.7, jy - TILE * 0.7, TILE * 1.4, TILE * 1.4);
    if (overlay.arrow) {
      const a = ANGLE[overlay.arrow];
      ctx.save();
      ctx.translate(jx, jy);
      ctx.rotate(a);
      ctx.fillStyle = "rgba(120, 255, 170, 0.9)";
      ctx.beginPath();
      ctx.moveTo(TILE * 1.3, 0);
      ctx.lineTo(TILE * 0.8, -TILE * 0.35);
      ctx.lineTo(TILE * 0.8, TILE * 0.35);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
  }

  if (s.phase !== "gameover") {
    for (const g of s.ghosts) if (s.phase !== "dying" || s.phaseTicks > 1.2 * TPS) drawGhost(ctx, s, g);
    drawPac(ctx, s);
  }

  const banner = s.phase === "ready" ? "READY!" : s.phase === "gameover" ? "GAME OVER" : s.phase === "levelclear" ? "CLEAR!" : "";
  if (banner) {
    ctx.fillStyle = s.phase === "gameover" ? "#ff3b3b" : "#ffe600";
    ctx.font = `bold ${TILE * 0.9}px ui-monospace, Menlo, monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(banner, WIDTH / 2, 17.5 * TILE);
  }
}
