// Classic 28x31 tile layout. '#' wall, '-' ghost-house door, '.' pellet,
// 'o' power pellet, ' ' empty floor. Row 14 is the wrap-around tunnel.
export const MAZE_ROWS: readonly string[] = [
  "############################",
  "#............##............#",
  "#.####.#####.##.#####.####.#",
  "#o####.#####.##.#####.####o#",
  "#.####.#####.##.#####.####.#",
  "#..........................#",
  "#.####.##.########.##.####.#",
  "#.####.##.########.##.####.#",
  "#......##....##....##......#",
  "######.##### ## #####.######",
  "     #.##### ## #####.#     ",
  "     #.##          ##.#     ",
  "     #.## ###--### ##.#     ",
  "######.## #      # ##.######",
  "      .   #      #   .      ",
  "######.## #      # ##.######",
  "     #.## ######## ##.#     ",
  "     #.##          ##.#     ",
  "     #.## ######## ##.#     ",
  "######.## ######## ##.######",
  "#............##............#",
  "#.####.#####.##.#####.####.#",
  "#.####.#####.##.#####.####.#",
  "#o..##.......  .......##..o#",
  "###.##.##.########.##.##.###",
  "###.##.##.########.##.##.###",
  "#......##....##....##......#",
  "#.##########.##.##########.#",
  "#.##########.##.##########.#",
  "#..........................#",
  "############################",
];

export const COLS = 28;
export const ROWS = 31;
export const TUNNEL_ROW = 14;

export const enum Tile {
  Floor = 0,
  Wall = 1,
  Door = 2,
}

export const enum Food {
  None = 0,
  Pellet = 1,
  Power = 2,
}

export const WALLS: Uint8Array = (() => {
  const grid = new Uint8Array(COLS * ROWS);
  MAZE_ROWS.forEach((row, y) => {
    for (let x = 0; x < COLS; x++) {
      const ch = row[x];
      grid[y * COLS + x] = ch === "#" ? Tile.Wall : ch === "-" ? Tile.Door : Tile.Floor;
    }
  });
  return grid;
})();

export function initialFood(): Uint8Array {
  const food = new Uint8Array(COLS * ROWS);
  MAZE_ROWS.forEach((row, y) => {
    for (let x = 0; x < COLS; x++) {
      const ch = row[x];
      food[y * COLS + x] = ch === "." ? Food.Pellet : ch === "o" ? Food.Power : Food.None;
    }
  });
  return food;
}

export function wrapX(x: number): number {
  return ((x % COLS) + COLS) % COLS;
}

export function index(x: number, y: number): number {
  return y * COLS + wrapX(x);
}

export function inBounds(y: number): boolean {
  return y >= 0 && y < ROWS;
}

/** Walkable for Pac-Man and for ghosts outside the house (the door blocks both). */
export function isOpen(x: number, y: number): boolean {
  return inBounds(y) && WALLS[index(x, y)] === Tile.Floor;
}

export function isTunnelSlowZone(x: number, y: number): boolean {
  const wx = wrapX(Math.round(x));
  return Math.round(y) === TUNNEL_ROW && (wx < 6 || wx > 21);
}
